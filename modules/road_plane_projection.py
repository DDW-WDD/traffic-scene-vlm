"""第四阶段：道路平面空间坐标建模（单应性变换 Homography）。

流程：
1. 读取第三阶段融合场景 JSON
2. 读取道路控制点标定文件
3. 建立 Homography 单应性矩阵（cv2.findHomography）
4. 取车辆/行人的近似地面接触点（bbox 底边中点）
5. 将图像坐标投影到道路局部平面
6. 计算参与者之间的道路平面距离
7. 保存空间场景 JSON / 距离 JSON / 模型信息 JSON

坐标状态（status）：
- projected                   地面锚点落在标定四边形内
- projected_extrapolated      落在四边形外，但仍在允许的外推范围内（可靠性下降）
- outside_calibration_region  外推距离过大，不给道路坐标
- excluded_non_ground_object  悬空设施（红绿灯等）不参与地面投影
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np

# 允许外推的范围（相对标定矩形尺寸），仅用于状态判定，不改变坐标计算
DEFAULT_MARGIN_RATIO = 0.35


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ground_anchor(bbox):
    """检测框 -> 近似地面接触点：底边中点。"""
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / 2.0, float(y2)]


def compute_homography(image_points, ground_points):
    """4 点求解单应性矩阵，并返回自洽性误差。"""
    src = np.array(image_points, dtype=np.float64).reshape(-1, 1, 2)
    dst = np.array(ground_points, dtype=np.float64).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, method=0)   # method=0: 直接 4 点求解
    if H is None or not np.all(np.isfinite(H)):
        raise ValueError("单应性矩阵求解失败：控制点可能共线或退化")

    # 自洽性检查：把控制点正投影回道路平面，误差应为 0（同一组点既求解又检验）
    reproj = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    target = dst.reshape(-1, 2)
    errors = np.linalg.norm(reproj - target, axis=1)
    return H, {
        "reprojection_error_max": round(float(errors.max()), 6),
        "reprojection_error_mean": round(float(errors.mean()), 6),
        "note": "控制点同时用于求解与自洽性检查，误差接近 0 只说明求解正确，不代表真实定位精度。",
    }


def transform_points(H, points):
    """批量把图像点变换到道路平面。"""
    if not points:
        return []
    src = np.array(points, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in out]


def inside_quad(point, quad):
    """点是否落在标定四边形内（含边界）。"""
    poly = np.array(quad, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(poly, (float(point[0]), float(point[1]),), False) >= 0


def classify_status(ground_xy, ground_quad, margin_ratio=DEFAULT_MARGIN_RATIO):
    """根据道路平面坐标落在标定矩形什么范围内判定可靠性。"""
    xs = [p[0] for p in ground_quad]
    ys = [p[1] for p in ground_quad]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    mx, my = w * margin_ratio, h * margin_ratio
    x, y = ground_xy
    if (min(xs) - mx) <= x <= (max(xs) + mx) and (min(ys) - my) <= y <= (max(ys) + my):
        # 在四边形内还是仅在扩展矩形内
        inside = cv2.pointPolygonTest(
            np.array(ground_quad, dtype=np.float32), (x, y), False) >= 0
        return "projected" if inside else "projected_extrapolated"
    return "outside_calibration_region"


def project_scene_to_ground(scene_path, calibration_path, output_path,
                            distances_path=None, info_path=None):
    """执行第四阶段核心投影，并写出结果文件。"""
    scene = read_json(scene_path)
    calibration = read_json(calibration_path)

    image_quad = [[float(x), float(y)] for x, y in calibration["image_points"]]
    ground_quad = [[float(x), float(y)] for x, y in calibration["ground_points"]]
    unit = calibration.get("unit", "local_unit")
    margin_ratio = float(calibration.get("extrapolation_margin_ratio",
                                         DEFAULT_MARGIN_RATIO))

    H, reproj_info = compute_homography(image_quad, ground_quad)

    spatial_scene = json.loads(json.dumps(scene, ensure_ascii=False))  # 深拷贝，不覆盖原文件
    counters = {
        "projected": 0,
        "projected_extrapolated": 0,
        "outside_calibration_region": 0,
        "excluded_non_ground_object": 0,
        "no_bbox_available": 0,
    }

    for obj in spatial_scene["objects"]:
        # 悬空设施不参与地面投影
        if obj.get("object_group") == "facility":
            obj["ground_position"] = {
                "x": None, "y": None, "unit": unit,
                "status": "excluded_non_ground_object",
                "method": "none",
                "reason": "悬空道路交通设施，无地面接触点",
            }
            counters["excluded_non_ground_object"] += 1
            continue

        if not obj.get("bbox"):
            obj["ground_position"] = {
                "x": None, "y": None, "unit": unit,
                "status": "no_bbox_available",
                "method": "none",
                "reason": "仅 VLM 语义对象，缺少检测框，无法确定地面锚点",
            }
            counters["no_bbox_available"] += 1
            continue

        anchor = ground_anchor(obj["bbox"])
        obj["ground_anchor_image"] = [round(anchor[0], 2), round(anchor[1], 2)]
        gx, gy = transform_points(H, [anchor])[0]

        inside = inside_quad(anchor, image_quad)
        if inside:
            status = "projected"
        else:
            status = classify_status((gx, gy), ground_quad, margin_ratio)

        obj["ground_position"] = {
            "x": round(gx, 3) if status != "outside_calibration_region" else None,
            "y": round(gy, 3) if status != "outside_calibration_region" else None,
            "unit": unit,
            "status": status,
            "method": "homography",
            "anchor_rule": "bbox_bottom_center",
        }
        if status == "outside_calibration_region":
            # 外推过远，坐标不可用：保留原始投影值仅用于追溯，不作为道路坐标使用
            obj["ground_position"]["rejected_projection"] = {
                "x": round(gx, 3), "y": round(gy, 3),
                "reason": "外推距离超过标定矩形的允许容差，坐标不可信",
            }
        counters[status] += 1

    # ---- 道路平面距离 ----
    # 只有双方都落在标定区内（status=projected）时，距离才作为"可靠距离"；
    # 含外推点的距离仍计算并保留，但标记 reliable=False，不进入可靠距离统计。
    eligible = [o for o in spatial_scene["objects"]
                if o["ground_position"]["status"] in
                ("projected", "projected_extrapolated")]
    pairs = []
    for i in range(len(eligible)):
        for j in range(i + 1, len(eligible)):
            a, b = eligible[i], eligible[j]
            ax, ay = a["ground_position"]["x"], a["ground_position"]["y"]
            bx, by = b["ground_position"]["x"], b["ground_position"]["y"]
            dist = math.hypot(ax - bx, ay - by)
            reliable = (a["ground_position"]["status"] == "projected" and
                        b["ground_position"]["status"] == "projected")
            pairs.append({
                "a": a["id"],
                "b": b["id"],
                "distance": round(dist, 3),
                "unit": unit,
                "a_status": a["ground_position"]["status"],
                "b_status": b["ground_position"]["status"],
                "reliable": reliable,
                "reliability_note": ("双方均在标定区内，可作为候选测量"
                                     if reliable else
                                     "至少一方为外推结果，仅保留数值供追溯，不用于测量结论"),
                "source": "homography_distance",
                "status": "computed",
            })
    pairs.sort(key=lambda p: p["distance"])
    reliable_pairs = [p for p in pairs if p["reliable"]]

    # ---- 语义关系上补充道路平面距离（仅可靠时给值） ----
    relation_updates = []
    pos_by_id = {o["id"]: o["ground_position"] for o in spatial_scene["objects"]}
    for rel in spatial_scene.get("semantic_relations", []):
        s, t = rel.get("subject"), rel.get("object")
        ps, pt = pos_by_id.get(s, {}), pos_by_id.get(t, {})
        if (ps.get("status") == "projected" and pt.get("status") == "projected"):
            d = math.hypot(ps["x"] - pt["x"], ps["y"] - pt["y"])
            rel["ground_distance"] = {"value": round(d, 3), "unit": unit,
                                      "reliable": True,
                                      "source": "homography_distance"}
            relation_updates.append({"relation": rel.get("predicate"),
                                     "subject": s, "object": t,
                                     "distance": round(d, 3),
                                     "reliable": True})
        elif (ps.get("status") in ("projected", "projected_extrapolated") and
              pt.get("status") in ("projected", "projected_extrapolated")):
            d = math.hypot(ps["x"] - pt["x"], ps["y"] - pt["y"])
            rel["ground_distance"] = {
                "value": None, "unit": unit, "reliable": False,
                "approx_value_for_trace": round(d, 3),
                "status": "unreliable_extrapolated",
                "reason": "至少一个对象为标定区外推结果，距离不用于测量结论",
            }
            relation_updates.append({"relation": rel.get("predicate"),
                                     "subject": s, "object": t,
                                     "distance": None,
                                     "approx_value_for_trace": round(d, 3),
                                     "reliable": False})
        else:
            rel["ground_distance"] = {"value": None, "unit": unit,
                                      "reliable": False,
                                      "status": "unavailable",
                                      "reason": "至少一个对象未成功投影"}

    eligible_count = counters["projected"] + counters["projected_extrapolated"]
    total_ground_candidates = eligible_count
    for k in ("outside_calibration_region", "no_bbox_available"):
        total_ground_candidates += counters[k]
    success_rate = (eligible_count / total_ground_candidates * 100
                    if total_ground_candidates else 0.0)

    # ---- 道路平面信息 ----
    road_plane = {
        "coordinate_system": calibration.get("coordinate_system"),
        "unit": unit,
        "origin_description": calibration.get("origin_description"),
        "axis_description": calibration.get("axis_description"),
        "image_points": image_quad,
        "ground_points": ground_quad,
        "homography_matrix": [[round(float(v), 6) for v in row] for row in H.tolist()],
        "ground_extent": {
            "x_min": min(p[0] for p in ground_quad),
            "x_max": max(p[0] for p in ground_quad),
            "y_min": min(p[1] for p in ground_quad),
            "y_max": max(p[1] for p in ground_quad),
        },
        "assumptions": calibration.get("assumptions", {}),
        "extrapolation_margin_ratio": margin_ratio,
        "calibration_status": calibration.get("calibration_status"),
    }

    spatial_scene["road_plane"] = road_plane
    spatial_scene["projection_summary"] = {
        "eligible_ground_objects": total_ground_candidates,
        "projected": counters["projected"],
        "projected_extrapolated": counters["projected_extrapolated"],
        "outside_calibration_region": counters["outside_calibration_region"],
        "excluded_non_ground_object": counters["excluded_non_ground_object"],
        "no_bbox_available": counters["no_bbox_available"],
        "success_rate_percent": round(success_rate, 1),
        "unit": unit,
        "distance_policy": "仅双方 status=projected 的距离为可靠距离",
        "pair_count": len(pairs),
        "reliable_pair_count": len(reliable_pairs),
        "self_consistency_error": reproj_info,
    }

    save_json(spatial_scene, output_path)

    if distances_path:
        save_json({
            "unit": unit,
            "source": "homography_distance",
            "note": ("距离为道路局部坐标下的欧氏距离，单位 local_unit，不可表述为米制精度；"
                     "只有双方都位于标定区内（reliable=true）的距离才可用于测量结论。"),
            "pair_count": len(pairs),
            "reliable_pair_count": len(reliable_pairs),
            "unreliable_pair_count": len(pairs) - len(reliable_pairs),
            "pairs": pairs,
            "reliable_pairs": reliable_pairs,
            "nearest_pair": pairs[0] if pairs else None,
            "nearest_reliable_pair": reliable_pairs[0] if reliable_pairs else None,
            "semantic_relation_distances": relation_updates,
        }, distances_path)

    if info_path:
        save_json({
            "stage": "stage4_road_plane_projection",
            "method": "planar homography (4 control points, DLT)",
            "unit": unit,
            "calibration": {
                "image_points": image_quad,
                "ground_points": ground_quad,
                "calibration_status": calibration.get("calibration_status"),
                "assumptions": calibration.get("assumptions", {}),
            },
            "homography_matrix": road_plane["homography_matrix"],
            "anchor_rule": {
                "vehicle": "bbox 底边中点近似轮胎接地点",
                "pedestrian": "bbox 底边中点近似脚部位置",
                "facility": "悬空设施排除，不参与地面投影",
            },
            "counts": counters,
            "success_rate_percent": round(success_rate, 1),
            "distance_policy": {
                "reliable": "双方 status=projected（均在标定区内）",
                "unreliable": "含 projected_extrapolated 的距离只保留数值，不进入可靠统计",
                "reliable_pair_count": len(reliable_pairs),
                "pair_count": len(pairs),
            },
            "self_consistency": reproj_info,
            "limitations": calibration.get("limitations", []),
        }, info_path)

    return spatial_scene, {
        "counters": counters,
        "success_rate_percent": round(success_rate, 1),
        "pair_count": len(pairs),
        "reliable_pair_count": len(reliable_pairs),
        "reprojection": reproj_info,
    }
