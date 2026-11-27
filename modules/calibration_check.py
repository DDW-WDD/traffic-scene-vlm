"""投影质量评估：自洽性、几何一致性、锚点敏感度与尺度歧义。

设计原则（对应教程 §11）：
1. 四点单应性只有 4 组对应点，恰好确定 8 个自由度，**不存在留一交叉验证**：
   去掉任意一点后方程数少于未知数，解不唯一。因此本模块不伪造 LOO 误差，
   而是用三类"不依赖标定四边形本身"的检查来量化投影可信度：
   (a) 求解自洽性 —— 只证明求解正确，不证明标定准确；
   (b) 几何一致性 —— 车道线/路缘在道路坐标中是否保持平行、竖直，
       偏差直接反映"标定矩形与道路走向是否对齐"；
   (c) 锚点敏感度 —— 锚点像素级抖动在道路坐标中放大成多少 local_unit，
       给出可被论文引用的误差量级。
2. 所有误差单位一律写 local_unit（局部单位误差），不换算成米。
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np

from modules.road_plane_projection import compute_homography


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _project(H, pts):
    arr = np.array([[p[0], p[1]] for p in pts], dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(arr, np.array(H, dtype=np.float64))
    return out.reshape(-1, 2)


def _fit_x_of_y(points):
    """道路坐标中拟合 x = a*y + b，返回斜率 a、截距 b、最大残差。"""
    y = np.array([p[1] for p in points], dtype=np.float64)
    x = np.array([p[0] for p in points], dtype=np.float64)
    A = np.vstack([y, np.ones_like(y)]).T
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    residual = np.abs(A @ coef - x)
    return a, b, float(residual.max())


def run(calibration_path, road_geometry_path, spatial_scene_path,
        output_path=None, anchor_shift_px=5.0):
    calibration = _load(calibration_path)
    geometry = _load(road_geometry_path)
    scene = _load(spatial_scene_path)

    image_quad = calibration["image_points"]
    ground_quad = calibration["ground_points"]
    H, reproj = compute_homography(image_quad, ground_quad)

    report = {
        "stage": "stage4_calibration_check",
        "unit": "local_unit",
        "unit_note": "1 local_unit = ground_points 中 1 个坐标单位，未经米制标定",
        "control_points": {
            "image_points": image_quad,
            "ground_points": ground_quad,
            "image_quad_px_area": round(abs(cv2.contourArea(
                np.array(image_quad, dtype=np.float32))), 1),
            "ground_quad_shape": "8 x 3 的矩形（人为设定，非实测）",
        },
    }

    # ---------- (a) 求解自洽性 ----------
    report["self_consistency"] = {
        "method": "四点回代（把 image_points 用解出的 H 投回 ground_points）",
        "max_error_local_unit": round(float(np.max(np.abs(
            _project(H, image_quad) - np.array(ground_quad, dtype=np.float64).reshape(-1, 2)))), 6),
        "reprojection_error_mean": reproj.get("reprojection_error_mean") if isinstance(reproj, dict) else None,
        "interpretation": "该误差为 0 只说明 DLT 求解闭合，不能作为标定精度；"
                          "标定精度取决于 4 个控制点本身的选取误差。",
    }

    # ---------- (b1) 车道线在道路坐标中的一致性 ----------
    lane_reports = []
    for lane in geometry.get("lane_lines", []):
        measured = lane.get("measured_points", [])
        if len(measured) >= 2:
            ground_pts = _project(H, measured)
            slope, intercept, resid = _fit_x_of_y(ground_pts.tolist())
            lane_reports.append({
                "id": lane["id"],
                "name": lane.get("name"),
                "measured_point_count": len(measured),
                "ground_slope_dx_per_dy": round(slope, 4),
                "ground_intercept_x": round(intercept, 3),
                "max_residual_local_unit": round(resid, 4),
                "deviation_from_vertical_deg": round(math.degrees(math.atan(abs(slope))), 2),
                "interpretation": (
                    "斜率接近 0 表示该车道线在道路坐标中保持竖直，"
                    "说明标定矩形与道路走向基本对齐；"
                    "残差接近 0 是单应性保持共线的必然结果，不代表测量精度。"
                ),
            })
    report["lane_line_alignment"] = lane_reports

    if len(lane_reports) >= 2:
        a1, a2 = lane_reports[0]["ground_slope_dx_per_dy"], lane_reports[1]["ground_slope_dx_per_dy"]
        angle = math.degrees(abs(math.atan(a1) - math.atan(a2)))
        gap_top = (lane_reports[1]["ground_intercept_x"] + a2 * 3.0) - \
                  (lane_reports[0]["ground_intercept_x"] + a1 * 3.0)
        gap_bottom = lane_reports[1]["ground_intercept_x"] - lane_reports[0]["ground_intercept_x"]
        report["lane_pair_parallelism"] = {
            "angle_between_lanes_deg": round(angle, 2),
            "gap_at_y_0_local_unit": round(gap_bottom, 3),
            "gap_at_y_3_local_unit": round(gap_top, 3),
            "gap_change_local_unit": round(gap_top - gap_bottom, 3),
            "interpretation": "两车道线在道路坐标中平行度良好（夹角<1°）时才可认为"
                              "标定矩形的横向轴与车道方向垂直。",
        }

    # ---------- (b2) 路缘/边界的平行性 ----------
    boundary_reports = []
    for b in geometry.get("road_boundaries", []):
        pts = [[p["x"], p["y"]] if isinstance(p, dict) else p for p in b.get("points", [])]
        if len(pts) >= 2:
            gp = _project(H, pts)
            dx = float(gp[-1][0] - gp[0][0])
            dy = float(gp[-1][1] - gp[0][1])
            boundary_reports.append({
                "id": b["id"],
                "name": b.get("name"),
                "dx_local_unit": round(dx, 3),
                "dy_local_unit": round(dy, 3),
                "angle_from_y_axis_deg": round(math.degrees(math.atan2(abs(dx), abs(dy))), 2),
                "interpretation": ("与车道线方向夹角明显（>5°）时，可能是真实的停车带/路口收窄，"
                                   "也可能说明该边界的人工标注存在偏差，二者目前无法区分。"),
            })
    report["boundary_alignment"] = boundary_reports

    # ---------- (c) 锚点像素敏感度 ----------
    sensitivity = []
    for obj in scene.get("objects", []):
        gp = obj.get("ground_position", {})
        if gp.get("status") not in ("projected", "projected_extrapolated"):
            continue
        anchor = obj.get("ground_anchor_image") or gp.get("image_anchor")
        if not anchor:
            continue
        base = _project(H, [anchor])[0]
        deltas = []
        for dx, dy in ((anchor_shift_px, 0), (-anchor_shift_px, 0),
                       (0, anchor_shift_px), (0, -anchor_shift_px)):
            probe = _project(H, [[anchor[0] + dx, anchor[1] + dy]])[0]
            deltas.append(float(math.hypot(probe[0] - base[0], probe[1] - base[1])))
        sensitivity.append({
            "id": obj["id"],
            "anchor_pixel": anchor,
            "shift_px": anchor_shift_px,
            "delta_local_unit": [round(d, 4) for d in deltas],
            "max_delta_local_unit": round(max(deltas), 4),
        })
    if sensitivity:
        all_max = [s["max_delta_local_unit"] for s in sensitivity]
        report["anchor_sensitivity"] = {
            "method": f"锚点分别沿 ±X/±Y 平移 {anchor_shift_px:g} px 后的坐标位移",
            "per_object": sensitivity,
            "max_local_unit": round(max(all_max), 4),
            "mean_local_unit": round(sum(all_max) / len(all_max), 4),
            "interpretation": (f"锚点标注抖动 {anchor_shift_px:g} px，在近处（Y 小）换算成约 "
                               f"{min(all_max):.3f} local_unit，在远处（Y 大）放大到约 "
                               f"{max(all_max):.3f} local_unit；距离越远，同样的像素误差代价越大。"),
        }

    # ---------- (d) 尺度歧义 ----------
    width_local = float(np.max(np.array(ground_quad)[:, 0]) - np.min(np.array(ground_quad)[:, 0]))
    height_local = float(np.max(np.array(ground_quad)[:, 1]) - np.min(np.array(ground_quad)[:, 1]))
    report["scale_ambiguity"] = {
        "assumed_ground_rectangle_local_unit": [width_local, height_local],
        "metric_factor": "unknown",
        "statement": (f"标定矩形被设为 {width_local:g}×{height_local:g} local_unit。"
                      "若真实道路横向宽度为 W 米，则 1 local_unit = W/"
                      f"{width_local:g} 米；W 未实测，因此任何 local_unit 数值都不能写成米。"
                      "论文中应表述为'局部单位'，并说明该单位由人为设定的标定矩形定义。"),
    }

    # ---------- 结论 ----------
    report["verdict"] = {
        "accuracy_level": "人工近似标定（manually_selected_unverified_metric）",
        "can_claim": [
            "同一帧内交通参与者的相对位置关系（左右、前后、靠近/远离）",
            "标定区内对象之间的相对距离排序与量级（local_unit）",
            "车道线、路缘等道路结构在道路坐标中的相对排布",
        ],
        "cannot_claim": [
            "米制距离、车速、碰撞能量等物理量",
            "绝对精度等级（需全站仪/激光测距等外部控制点）",
            "标定区外（外推）对象的可靠距离",
        ],
        "suggested_improvement": (
            "在场景中实测 2~3 个独立检验点（如车道线端点、井盖中心）的米制坐标，"
            "用 E=(1/N)Σ√[(Xᵢ−X̂ᵢ)²+(Yᵢ−Ŷᵢ)²] 评估局部单位误差，并把结果写进论文的误差分析小节。"
        ),
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    d = root / "outputs" / "ACCIDENT_001"
    r = run(d / "calibration.json", d / "road_geometry.json",
            d / "scene_model_spatial.json", d / "calibration_check.json")
    print(json.dumps({k: r[k] for k in ("self_consistency", "lane_pair_parallelism",
                                        "anchor_sensitivity")}, ensure_ascii=False, indent=2)[:2000])
