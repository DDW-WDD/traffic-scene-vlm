"""第四阶段 4.2：道路几何结构的标注、投影与路面俯视底图。

职责：
1. 读取 road_geometry.json（原图像素坐标的人工标注）
2. 把道路多边形、路缘线、车道线逐点投影到道路局部平面（与对象投影共用同一个 H）
3. 生成"道路表面掩膜"：道路多边形内部、扣掉所有对象检测框（避免把车辆、行人当作路面）
4. 用同一个 H 对路面掩膜做透视变换，得到道路平面俯视底图（不对车辆/建筑做平面解释）
"""

from pathlib import Path

import cv2
import numpy as np

from modules.road_plane_projection import read_json, save_json, transform_points


# ---------------------------------------------------------------- 车道线拟合
def fit_x_of_y(points):
    """以 y 为自变量拟合 x = a*y + b（车道线在图像中近似直线）。"""
    pts = np.array([[float(p[0]), float(p[1])] for p in points], dtype=np.float64)
    a, b = np.polyfit(pts[:, 1], pts[:, 0], 1)
    return float(a), float(b)


def fit_residual(points, a, b):
    """拟合残差（像素），用于说明标注线与扫描点的贴合程度。"""
    pts = np.array([[float(p[0]), float(p[1])] for p in points], dtype=np.float64)
    pred = a * pts[:, 1] + b
    return float(np.max(np.abs(pts[:, 0] - pred)))


def lane_line_segment(line):
    """把车道线还原成一条"从上到下"的线段（上端=远端外推行，下端=画面底边）。"""
    a, b = fit_x_of_y(line["measured_points"])
    y_top = float(line.get("extrapolated_to_row", min(p[1] for p in line["measured_points"])))
    y_bot = 768.0
    return [[round(a * y_top + b, 2), y_top], [round(a * y_bot + b, 2), y_bot]], (a, b)


# ---------------------------------------------------------------- 几何投影
def project_road_geometry(geom, H, ground_quad, unit):
    """道路几何逐点投影，并标出每个点相对标定矩形的位置状态。"""
    quad = np.array(ground_quad, dtype=np.float32)

    def status_of(pt):
        inside = cv2.pointPolygonTest(quad, (float(pt[0]), float(pt[1])), False) >= 0
        return "inside_calibration_region" if inside else "outside_calibration_region"

    def conv(points):
        local = transform_points(H, points)
        return [
            {
                "x": round(x, 3),
                "y": round(y, 3),
                "status": status_of((x, y)),
            }
            for x, y in local
        ]

    out = {
        "unit": unit,
        "coordinate_system": "road_local_plane",
        "source": "road_geometry.json 经同一单应性矩阵变换",
        "road_polygon": conv(geom["road_polygon"]),
        "road_boundaries": [
            {**{k: v for k, v in b.items() if k not in ("points",)},
             "points_image": b["points"],
             "points_local": conv(b["points"])}
            for b in geom.get("road_boundaries", [])
        ],
        "lane_lines": [],
        "stop_lines": [{"id": s.get("id"), "points_local": conv(s["points"])}
                       for s in geom.get("stop_lines", [])],
        "crosswalks": [{"id": c.get("id"), "points_local": conv(c["points"])}
                       for c in geom.get("crosswalks", [])],
        "absent_features": geom.get("absent_features", {}),
        "quality": geom.get("quality", {}),
    }

    for line in geom.get("lane_lines", []):
        seg, (a, b) = lane_line_segment(line)
        residual = fit_residual(line["measured_points"], a, b)
        row_vis_top = line.get("visible_row_range", [None, None])[0]
        y_vis_local = None
        if row_vis_top is not None:
            x_at = a * row_vis_top + b
            y_vis_local = round(float(transform_points(
                H, [[x_at, float(row_vis_top)]])[0][1]), 3)
        out["lane_lines"].append({
            "id": line["id"],
            "name": line.get("name", ""),
            "type": line.get("type", "solid"),
            "points_image": seg,
            "points_local": conv(seg),
            "visible_split_local": {
                "y_at_visible_top_row": y_vis_local,
                "visible_row_range": line.get("visible_row_range"),
                "extrapolated_row_range": [line.get("extrapolated_to_row"), row_vis_top],
            },
            "fit": {"x_of_y_slope": round(a, 4), "x_of_y_intercept": round(b, 2),
                    "fit_max_residual_px": round(residual, 2),
                    "measured_rows": [p[1] for p in line["measured_points"]]},
            "split": {
                "visible_part_rows": line.get("visible_row_range"),
                "extrapolated_part_rows": [line.get("extrapolated_to_row"),
                                           line.get("visible_row_range", [None, None])[0]],
                "extrapolation_reason": line.get("extrapolation_reason", ""),
            },
        })
    return out


# ---------------------------------------------------------------- 路面掩膜
def build_road_mask(shape, geom, objects, image_bgr=None, dilate_px=8):
    """道路表面掩膜：道路多边形内、颜色上像沥青的像素、扣掉全部对象检测框。

    - 多边形限定搜索范围（避免把天空、绿化带算成路面）
    - 颜色阈值进一步剔除多边形内实际不是沥青的部分（如远处建筑、树）
    - 扣掉检测框，保证不会把车辆、行人当路面做平面解释
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    poly = np.array(geom["road_polygon"], dtype=np.int32)
    cv2.fillPoly(mask, [poly], 255)

    if image_bgr is not None:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        asphalt = ((hsv[..., 1] < 70) & (hsv[..., 2] > 40) &
                   (hsv[..., 2] < 190)).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, asphalt)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    for obj in objects:
        bbox = obj.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1, y1 = max(0, x1 - dilate_px), max(0, y1 - dilate_px)
        x2, y2 = min(w - 1, x2 + dilate_px), min(h - 1, y2 + dilate_px)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)

    # 轻微闭运算，抹掉掩膜边缘的锯齿
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    return mask


def canvas_matrix(H, xlim, ylim, px_per_unit):
    """图像像素 → 俯视画布像素 的复合变换：先 H 到道路坐标，再线性映射到画布。"""
    xmin, xmax = xlim
    ymin, ymax = ylim
    s = float(px_per_unit)
    A = np.array([
        [s, 0, -s * xmin],
        [0, -s, s * ymax],
        [0, 0, 1],
    ], dtype=np.float64)
    return A @ H, (int(round(s * (xmax - xmin))), int(round(s * (ymax - ymin))))


def warp_road_surface(image_bgr, mask, H, xlim, ylim, px_per_unit,
                      texture_mix=0.5, lift=18):
    """把路面掩膜透视变换成道路平面俯视底图，返回 (RGBA, extent)。"""
    W, (cw, ch) = canvas_matrix(H, xlim, ylim, px_per_unit)

    warped = cv2.warpPerspective(image_bgr, W, (cw, ch),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))
    a = cv2.warpPerspective(mask, W, (cw, ch), flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # 降饱和 + 提亮，让后续符号和标注压在底图上也清晰
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    mixed = cv2.addWeighted(warped, 1 - texture_mix, gray[..., None].repeat(3, 2),
                            texture_mix, lift)

    rgba = np.dstack([cv2.cvtColor(mixed, cv2.COLOR_BGR2RGB), a])
    extent = (xlim[0], xlim[1], ylim[0], ylim[1])
    return rgba, extent


# ---------------------------------------------------------------- 落盘
def run(case_dir, image_path, spatial_scene_path, H, xlim, ylim, px_per_unit=160):
    """投影道路几何、生成路面底图，并写 road_geometry_local.json / road_surface_topdown.png。"""
    case_dir = Path(case_dir)
    geom = read_json(case_dir / "road_geometry.json")
    scene = read_json(spatial_scene_path)
    ground_quad = scene["road_plane"]["ground_points"]
    unit = scene["road_plane"].get("unit", "local_unit")

    local = project_road_geometry(geom, H, ground_quad, unit)

    img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    mask = build_road_mask(img.shape, geom, scene["objects"], image_bgr=img)
    rgba, extent = warp_road_surface(img, mask, H, xlim, ylim, px_per_unit)

    save_json(local, case_dir / "road_geometry_local.json")
    cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))[1].tofile(
        str(case_dir / "road_surface_topdown.png"))

    return local, rgba, extent, mask
