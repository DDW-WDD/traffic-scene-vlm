"""第四阶段 4.3：道路俯视场景重建（符号化）。

在道路局部坐标系里绘制：
- 道路区域（多边形填充）+ 透视校正后的路面纹理（仅可见段）
- 右侧路缘线、两条车道白线（可见段实线 / 外推段虚线）
- 交通参与者：车辆俯视符号、行人标记（不同图形、不同颜色）
- 悬空设施：不进入路面，单独列在右侧面板
- 道路平面距离：只在双方均位于标定区内时才画线与标注
- 语义关系（VLM 观察）与几何候选关系分层呈现
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

INK = "#20242c"
MUTED = "#6b7280"
LANE_WHITE = "#f4f6f8"
ASPHALT = "#59606a"
COLOR = {
    "car": "#63b3ed",
    "truck": "#f6ad55",
    "bus": "#f6ad55",
    "motorcycle": "#f6ad55",
    "bicycle": "#f6ad55",
    "person": "#68d391",
    "facility": "#b794f4",
}


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- 符号
def draw_vehicle(ax, x, y, color, alpha=1.0, scale=1.0, zorder=5):
    """车辆俯视符号：圆角车身 + 对称车顶 + 四个轮位。朝向未建模，故图形上下左右对称。"""
    w, h = 0.26 * scale, 0.58 * scale
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={0.07 * scale}",
        linewidth=0.8, edgecolor="#2b3038", facecolor=color,
        alpha=alpha, zorder=zorder))
    ax.add_patch(FancyBboxPatch(
        (x - 0.085 * scale, y - 0.13 * scale), 0.17 * scale, 0.26 * scale,
        boxstyle=f"round,pad=0,rounding_size={0.03 * scale}",
        linewidth=0.5, edgecolor="#2b3038", facecolor="#9aa5b1",
        alpha=alpha * 0.85, zorder=zorder + 1))
    for dy in (-0.19, 0.19):
        for dx in (-0.12, 0.12):
            ax.add_patch(Rectangle((x + dx * scale - 0.02, y + dy * scale - 0.045),
                                   0.04, 0.09, facecolor="#343a40",
                                   alpha=alpha, zorder=zorder, linewidth=0))


def draw_pedestrian(ax, x, y, color, alpha=1.0, scale=1.0, zorder=6):
    """行人标记：圆点 + 短腿线，表示脚部近似位置，不表示人体轮廓。"""
    ax.add_patch(Circle((x, y), 0.075 * scale, facecolor=color,
                        edgecolor="#2b3038", linewidth=0.8, alpha=alpha,
                        zorder=zorder))
    ax.plot([x - 0.03 * scale, x - 0.05 * scale], [y - 0.04, y - 0.11 * scale],
            color="#2b3038", lw=0.8, alpha=alpha, zorder=zorder)
    ax.plot([x + 0.03 * scale, x + 0.05 * scale], [y - 0.04, y - 0.11 * scale],
            color="#2b3038", lw=0.8, alpha=alpha, zorder=zorder)


def _normal_offset(p, q, length=0.14):
    """沿线段法向的偏移量，用于把距离标注挪开线段本身。"""
    dx, dy = q[0] - p[0], q[1] - p[1]
    norm = float(np.hypot(dx, dy)) or 1.0
    nx, ny = -dy / norm, dx / norm
    mid = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
    if ny < 0:
        nx, ny = -nx, -ny
    return mid[0] + nx * length, mid[1] + ny * length


# ---------------------------------------------------------------- 主图
def build_scene_figure(spatial_scene_path, road_geometry_local_path,
                       surface_png_path, output_path,
                       xlim=(-3.9, 10.0), ylim=(-0.45, 3.3),
                       title="交通事故道路平面二维场景重建"):
    scene = _load(spatial_scene_path)
    geom = _load(road_geometry_local_path)
    unit = scene["road_plane"].get("unit", "local_unit")

    fig = plt.figure(figsize=(15.4, 5.2), dpi=150)
    ax = fig.add_axes([0.048, 0.185, 0.685, 0.625])

    # ---- 1. 道路区域底色 ----
    poly = [(p["x"], p["y"]) for p in geom["road_polygon"]]
    ax.add_patch(Polygon(poly, closed=True, facecolor=ASPHALT,
                         edgecolor="#2b3038", linewidth=1.2, zorder=1))

    # ---- 2. 透视校正后的路面纹理（仅可见段，透明度随纵向衰减） ----
    if Path(surface_png_path).exists():
        tex = plt.imread(str(surface_png_path))
        tex = tex.copy()
        rows = np.linspace(ylim[1], ylim[0], tex.shape[0])[:, None]
        fade = np.clip(1.0 - (rows - 1.1) / 1.6, 0.0, 1.0)   # Y>1.1 后逐渐淡出
        tex[..., 3] = tex[..., 3] * 0.55 * fade
        ax.imshow(tex, extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
                  origin="upper", zorder=2, interpolation="bilinear")

    # 被车辆遮挡、没有路面纹理的区域：用斜线阴影标注为"几何推断"
    occl_y = 1.15
    ax.add_patch(Polygon(
        [(p[0], p[1]) if p[1] > occl_y else (p[0], occl_y) for p in poly],
        closed=True, facecolor="none", edgecolor="#aab4bf", hatch="//",
        linewidth=0, alpha=0.16, zorder=3))
    ax.annotate("斜线区＝路面被车辆遮挡、无纹理，仅按道路几何推断",
                (xlim[1] - 0.2, occl_y + 0.06), fontsize=6.8, color="#f1f3f5",
                ha="right", va="bottom", zorder=13)

    # ---- 3. 车道白线：可见段实线、外推段虚线 ----
    for lane in geom["lane_lines"]:
        pts = lane["points_local"]
        x_top, y_top = pts[0]["x"], pts[0]["y"]
        x_bot, y_bot = pts[1]["x"], pts[1]["y"]
        y_vis = lane.get("visible_split_local", {}).get("y_at_visible_top_row")
        y_split = float(y_vis) if y_vis is not None else y_top + 0.1
        y_split = min(max(y_split, y_bot + 0.05), y_top - 0.05)
        ax.plot([x_top, x_bot], [y_split, y_bot], color=LANE_WHITE, lw=3.0,
                solid_capstyle="butt", zorder=4)
        ax.plot([x_top, x_bot], [y_top, y_split], color=LANE_WHITE, lw=2.6,
                ls=(0, (4, 3)), alpha=0.85, zorder=4)
        ax.annotate(f"{lane['id']}", (x_bot + (0.12 if x_bot < 4 else -0.12), 1.02), fontsize=7.2,
                    color=INK, ha="left" if x_bot < 4 else "right", va="center", zorder=13,
                    bbox=dict(boxstyle="round,pad=0.14", fc="white",
                              ec="#ced4da", lw=0.5, alpha=0.85))

    # ---- 4. 路缘线 ----
    for b in geom["road_boundaries"]:
        pts = b["points_local"]
        ax.plot([p["x"] for p in pts], [p["y"] for p in pts],
                color="#ffd43b", lw=1.5, ls=(0, (6, 3)), zorder=3)
        ax.annotate(f"{b['id']} 路缘（近似）", (pts[0]["x"] + 0.16, 2.25), fontsize=7.0,
                    color="#343a40", ha="left", va="center", zorder=13,
                    bbox=dict(boxstyle="round,pad=0.14", fc="#fff9db",
                              ec="#ffd43b", lw=0.6, alpha=0.9))

    # ---- 5. 交通参与者符号 ----
    inside, extrapolated, outside, facilities = [], [], [], []
    for obj in scene["objects"]:
        st = obj.get("ground_position", {}).get("status")
        {"projected": inside, "projected_extrapolated": extrapolated,
         "outside_calibration_region": outside,
         "excluded_non_ground_object": facilities}.get(st, []).append(obj)

    def symbol(obj, alpha):
        gp = obj["ground_position"]
        color = COLOR.get(obj.get("category"), "#adb5bd")
        if obj.get("category") == "person":
            draw_pedestrian(ax, gp["x"], gp["y"], color, alpha=alpha)
        else:
            draw_vehicle(ax, gp["x"], gp["y"], color, alpha=alpha)

    for obj in inside:
        symbol(obj, 1.0)
    for obj in extrapolated:
        symbol(obj, 0.42)

    pos = {o["id"]: o["ground_position"] for o in scene["objects"]}

    # ---- 6. 对象编号标注 ----
    for group, alpha in ((inside, 1.0), (extrapolated, 0.72)):
        for obj in group:
            gp = obj["ground_position"]
            tag = "" if gp["status"] == "projected" else "（外推）"
            person = obj.get("category") == "person"
            dy = 0.20 if person else 0.36
            above = gp["y"] < 0.55 or gp["y"] > 2.35   # 贴近上下边界时标注翻到符号上方
            ax.annotate(f"{obj['id']} {obj.get('category_zh', '')}{tag}",
                        (gp["x"] + (0 if above else (0.22 if person else 0)),
                         gp["y"] + dy * 0.55 if above else gp["y"] - dy),
                        fontsize=8.0, color=INK,
                        ha="center" if above else ("left" if person else "center"),
                        va="bottom" if above else "top", alpha=alpha, zorder=12,
                        bbox=dict(boxstyle="round,pad=0.16", fc="white",
                                  ec="#ced4da", lw=0.5, alpha=0.8))

    # ---- 7. 距离：只画双方都在标定区内的对 ----
    pairs = []
    for i, a in enumerate(inside):
        for b in inside[i + 1:]:
            pa, pb = pos[a["id"]], pos[b["id"]]
            d = float(np.hypot(pa["x"] - pb["x"], pa["y"] - pb["y"]))
            pairs.append((a["id"], b["id"], d, pa, pb))
    pairs.sort(key=lambda t: t[2])

    for oid_a, oid_b, d, pa, pb in pairs:
        nearest = d == pairs[0][2]
        ax.plot([pa["x"], pb["x"]], [pa["y"], pb["y"]],
                color="#e8590c" if nearest else "#c92a2a",
                lw=2.4 if nearest else 1.6, zorder=8, alpha=0.95,
                solid_capstyle="round")
        lx, ly = _normal_offset((pa["x"], pa["y"]), (pb["x"], pb["y"]),
                                0.13 if nearest else 0.11)
        ax.annotate(f"{oid_a}—{oid_b} {d:.2f}", (lx, ly),
                    fontsize=7.2, ha="center", va="center",
                    color="#7f1d1d" if nearest else "#8a2b2b", zorder=14,
                    bbox=dict(boxstyle="round,pad=0.16", fc="#fff5f5",
                              ec="#e8590c" if nearest else "#c92a2a", lw=0.8))

    # ---- 8. 语义关系（VLM）：虚线，且不给可靠距离 ----
    for rel in scene.get("semantic_relations", []):
        a, b = rel.get("subject"), rel.get("object")
        pa, pb = pos.get(a, {}), pos.get(b, {})
        if pa.get("x") is None or pb.get("x") is None:
            continue
        ax.plot([pa["x"], pb["x"]], [pa["y"], pb["y"]], color="#862e9c",
                lw=1.4, ls=(0, (5, 3)), zorder=7, alpha=0.9)
        lx, ly = _normal_offset((pa["x"], pa["y"]), (pb["x"], pb["y"]), -0.55)
        ax.annotate(f"{a} —{rel.get('predicate', '')}→ {b}\n语义关系（未验证碰撞）",
                    (lx, ly), fontsize=7.2, color="#5f3dc4", ha="center", va="top",
                    zorder=15,
                    bbox=dict(boxstyle="round,pad=0.18", fc="#f8f0fc",
                              ec="#862e9c", lw=0.7, alpha=0.95))

    # ---- 9. 标定区域边界 ----
    quad = scene["road_plane"]["ground_points"]
    ax.plot([p[0] for p in quad] + [quad[0][0]],
            [p[1] for p in quad] + [quad[0][1]],
            color="#ffe8cc", lw=1.0, ls=(0, (2, 2)), zorder=3, alpha=0.85)
    ax.annotate("标定区域边界", (quad[3][0] + 0.1, quad[3][1] - 0.02), fontsize=7.0,
                color="#ffe8cc", va="top", zorder=13)

    # ---- 10. 坐标轴与比例尺 ----
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"X  道路横向（{unit}）", fontsize=9, color=MUTED)
    ax.set_ylabel(f"Y  道路纵向（向远端，{unit}）", fontsize=9, color=MUTED)
    ax.tick_params(labelsize=7.4, colors=MUTED, length=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#ced4da")

    ax.plot([xlim[0] + 0.15, xlim[0] + 1.15], [ylim[0] + 0.13] * 2,
            color="white", lw=2.0, zorder=13)
    ax.annotate("1 local_unit（未标定为米）", (xlim[0] + 0.15, ylim[0] + 0.19),
                fontsize=7.0, color="white", zorder=13)

    # ---- 11. 图例（图幅下缘） ----
    handles_layer = [
        Line2D([0], [0], color=LANE_WHITE, lw=3.0, label="车道白线（实线=可见段）"),
        Line2D([0], [0], color=LANE_WHITE, lw=2.6, ls=(0, (4, 3)), label="车道白线外推段"),
        Line2D([0], [0], color="#ffd43b", lw=1.5, ls=(0, (6, 3)), label="路缘线（近似标注）"),
        Line2D([0], [0], color="#c92a2a", lw=1.6, label="道路平面距离（区内可靠）"),
        Line2D([0], [0], color="#862e9c", lw=1.4, ls=(0, (5, 3)),
               label="语义关系（VLM 观察，非碰撞事实）"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR["car"],
               markeredgecolor="#2b3038", markersize=7, label="车辆符号（示意，非实测尺寸）"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR["person"],
               markeredgecolor="#2b3038", markersize=6, label="行人标记（脚部锚点）"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR["car"],
               markeredgecolor="#adb5bd", markersize=7, alpha=0.42,
               label="半透明=外推（不参与可靠距离）"),
    ]
    fig.legend(handles=handles_layer, loc="lower left", bbox_to_anchor=(0.048, 0.038),
               ncol=4, fontsize=7.4, frameon=True, framealpha=0.95,
               borderpad=0.6, labelspacing=0.5, columnspacing=1.4,
               title="图例", title_fontsize=8.2)

    # ---- 12. 右侧面板 ----
    lines_out = ["未给道路坐标（超出标定范围）"]
    for obj in outside:
        lines_out.append(f"  {obj['id']} {obj.get('category_zh', '')}")
    lines_fac = ["悬空设施（不参与地面投影）"]
    for obj in facilities:
        lines_fac.append(f"  {obj['id']} {obj.get('category_zh', '')}")

    fig.text(0.742, 0.945, "\n".join(lines_out), fontsize=7.8, color="#495057",
             va="top", linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.45", fc="#f8f9fa", ec="#dee2e6"))
    fig.text(0.742, 0.575, "\n".join(lines_fac), fontsize=7.8, color="#495057",
             va="top", linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.45", fc="#f5f3ff", ec="#d0bfff"))

    stat = scene.get("projection_summary", {})
    panel = [
        "投影状态与距离",
        f"  区内 projected：{stat.get('projected', 0)}",
        f"  外推 extrapolated：{stat.get('projected_extrapolated', 0)}",
        f"  超出标定范围：{stat.get('outside_calibration_region', 0)}",
        f"  悬空排除：{stat.get('excluded_non_ground_object', 0)}",
        "",
        f"可靠距离对（双方均在区内）{len(pairs)} 对：",
    ]
    for oid_a, oid_b, d, _, _ in pairs:
        panel.append(f"  {oid_a}—{oid_b} = {d:.2f} {unit}")
    fig.text(0.742, 0.435, "\n".join(panel), fontsize=7.6, color="#343a40", va="top",
             linespacing=1.45,
             bbox=dict(boxstyle="round,pad=0.45", fc="#ffffff", ec="#ced4da"))

    fig.text(
        0.048, 0.010,
        f"说明：坐标为道路局部坐标系（{unit}），未做米制标定，图上不出现米；"
        "车辆/行人符号仅为示意，不代表实测车身尺寸与朝向；道路多边形与路缘线为人工近似标注，"
        "车道线由逐行扫描拟合；只有双方都在标定区内时才给出道路平面距离。",
        fontsize=7.2, color="#868e96")

    fig.text(0.048, 0.985, title, fontsize=15, color=INK, va="top")
    fig.text(0.048, 0.925,
             f"案例 ACCIDENT_001 ｜ 四点单应性标定 ｜ 道路几何：2 条车道线 + 1 条路缘线 ｜ 单位 {unit}",
             fontsize=8.6, color=MUTED)

    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(output_path)
