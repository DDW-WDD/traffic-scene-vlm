"""第四阶段：道路平面二维场景重建可视化。

绘制内容：
- 标定矩形（假设的道路局部范围，灰色虚线框）
- 交通参与者地面锚点（车辆=方块，行人=圆点）
- 关键对象之间的道路平面距离（语义关系对）
- 未投影/外推对象清单
- 坐标轴、单位与假设声明
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon


_COLOR = {
    "car": "#5aa9e6",
    "truck": "#f4a261",
    "bus": "#f4a261",
    "motorcycle": "#f4a261",
    "bicycle": "#f4a261",
    "person": "#69b578",
    "default": "#adb5bd",
}


def _label(obj):
    desc = obj.get("vlm_description") or obj.get("category_zh") or ""
    if len(desc) > 12:
        desc = desc[:12] + "…"
    return desc


def visualize_spatial_scene(scene_path, calibration_path, output_path):
    with open(scene_path, "r", encoding="utf-8") as f:
        scene = json.load(f)
    with open(calibration_path, "r", encoding="utf-8") as f:
        calibration = json.load(f)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    unit = calibration.get("unit", "local_unit")
    ground_quad = calibration["ground_points"]
    qx = [p[0] for p in ground_quad]
    qy = [p[1] for p in ground_quad]

    projected, extrapolated, outside = [], [], []
    for obj in scene["objects"]:
        gp = obj.get("ground_position", {})
        status = gp.get("status")
        if status in ("projected", "projected_extrapolated"):
            (projected if status == "projected" else extrapolated).append(obj)
        elif status == "outside_calibration_region":
            outside.append(obj)

    # 画布范围：含标定矩形与所有已投影点
    px = [o["ground_position"]["x"] for o in projected + extrapolated]
    py = [o["ground_position"]["y"] for o in projected + extrapolated]
    xmin = min(qx + px) - 1.0
    xmax = max(qx + px) + 1.2
    ymin = min(qy + py) - 1.4
    ymax = max(qy + py) + 1.7

    fig, ax = plt.subplots(figsize=(13.5, 7.4))

    # 标定区域
    ax.add_patch(MplPolygon(ground_quad, closed=True, facecolor="#f1f3f5",
                            edgecolor="#868e96", linestyle="--", linewidth=1.6,
                            zorder=0))
    ax.text((min(qx) + max(qx)) / 2, min(qy) + 0.05 * (max(qy) - min(qy)),
            "标定区域（假设局部矩形 8 × 3，非实测道路边界）",
            ha="center", va="bottom", fontsize=9, color="#868e96", zorder=1)

    # 对象
    for group, alpha in ((projected, 1.0), (extrapolated, 0.55)):
        for obj in group:
            gp = obj["ground_position"]
            x, y = gp["x"], gp["y"]
            color = _COLOR.get(obj.get("category"), _COLOR["default"])
            marker = "o" if obj.get("category") == "person" else "s"
            ax.plot(x, y, marker=marker, markersize=12, color=color,
                    markeredgecolor="#343a40", markeredgewidth=1.2,
                    linestyle="None", alpha=alpha, zorder=3)
            if marker == "s":
                ax.plot([x - 0.3, x + 0.3], [y, y], color="#343a40",
                        lw=1.4, alpha=alpha, zorder=2)
            status_tag = "" if gp["status"] == "projected" else "（外推）"
            ax.annotate(f"{obj['id']} {obj.get('category_zh', '')}{status_tag}\n"
                        f"({x:.2f}, {y:.2f})",
                        (x, y), textcoords="offset points", xytext=(0, -22),
                        ha="center", va="top", fontsize=8.5, color="#212529",
                        zorder=4)

    # 语义关系对的道路平面距离
    pos = {o["id"]: o["ground_position"] for o in scene["objects"]
           if o.get("ground_position", {}).get("status")
           in ("projected", "projected_extrapolated")}
    for rel in scene.get("semantic_relations", []):
        gd = rel.get("ground_distance", {})
        a, b = rel["subject"], rel["object"]
        if gd.get("value") is None or a not in pos or b not in pos:
            continue
        ax.plot([pos[a]["x"], pos[b]["x"]], [pos[a]["y"], pos[b]["y"]],
                color="#e03131", lw=2.4, zorder=2)
        mid = ((pos[a]["x"] + pos[b]["x"]) / 2, (pos[a]["y"] + pos[b]["y"]) / 2)
        ax.annotate(f"{a} —{rel.get('predicate', '')}→ {b}\n"
                    f"道路平面距离 {gd['value']:.2f} {unit}",
                    mid, textcoords="offset points", xytext=(0, 16),
                    ha="center", fontsize=9.5, color="#e03131",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="#e03131", alpha=0.9), zorder=5)

    # 未投影 / 外推清单
    lines = []
    if extrapolated:
        lines.append("外推投影（可靠性下降）：")
        for obj in extrapolated:
            g = obj["ground_position"]
            lines.append(f"  {obj['id']} {obj.get('category_zh', '')}  "
                         f"({g['x']:.2f}, {g['y']:.2f})")
    if outside:
        lines.append("")
        lines.append("未给坐标（超出标定范围）：")
        for obj in outside:
            lines.append(f"  {obj['id']} {obj.get('category_zh', '')}")
    if lines:
        ax.text(xmax - 0.15, ymax - 0.12, "\n".join(lines),
                ha="right", va="top", fontsize=8.8, color="#495057",
                bbox=dict(boxstyle="round,pad=0.4", fc="#f8f9fa", ec="#ced4da"),
                zorder=6)

    legend = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=_COLOR["car"],
               markeredgecolor="#343a40", markersize=10, label="车辆（实心=标定区内）"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_COLOR["person"],
               markeredgecolor="#343a40", markersize=10, label="行人（实心=标定区内）"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=_COLOR["car"],
               markeredgecolor="#adb5bd", markersize=10, alpha=0.55,
               label="半透明=外推结果"),
        Line2D([0], [0], color="#e03131", lw=2.4, label="语义关系对的道路平面距离"),
        Line2D([0], [0], color="#868e96", lw=1.6, ls="--", label="标定区域边界"),
    ]
    ax.legend(handles=legend, loc="upper left", ncol=2, fontsize=9,
              framealpha=0.95)

    ax.set_xlabel(f"X  道路横向（{unit}）")
    ax.set_ylabel(f"Y  道路纵向（向远端，{unit}）")
    ax.set_title("交通事故道路平面二维场景重建（四点单应性变换投影）",
                 fontsize=15, pad=14)
    ax.grid(True, ls=":", color="#dee2e6", lw=0.7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    fig.text(
        0.012, 0.012,
        "说明：坐标为道路局部坐标系（" + unit + "），非米制实测，仅用于相对位置与距离分析；"
        "标定区域为假设矩形，不代表真实道路边界；锚点取检测框底边中点，仅近似表示地面接触位置。",
        fontsize=8.5, color="#868e96")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)
