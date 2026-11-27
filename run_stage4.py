"""第四阶段运行脚本（升级版）：道路结构重建 + 道路平面投影 + 场景/图谱多视图。

运行：
    python run_stage4.py

输入：
    outputs/ACCIDENT_001/scene_model.json       第三阶段融合场景
    outputs/ACCIDENT_001/calibration.json       道路控制点标定
    outputs/ACCIDENT_001/road_geometry.json     道路几何人工标注（像素坐标）
    outputs/ACCIDENT_001/original.jpg           原始监控画面

输出（全部写入 outputs/ACCIDENT_001/）：
    road_geometry_local.json      道路几何投影到道路坐标的结果（逐点状态）
    road_surface_topdown.png      路面掩膜透视变换后的俯视底图
    scene_model_spatial.json      带道路平面坐标的空间场景
    ground_distances.json         道路平面两两距离（含 reliable 标记）
    calibration_check.json        投影质量评估（自洽性/几何一致性/锚点敏感度/尺度歧义）
    spatial_model_info.json       标定/矩阵/成功率/局限说明
    scene_topdown.png             道路平面二维场景重建图（静态版）
    scene_topdown.html            交互版场景（点击对象联动表格）
    knowledge_graph_spatial.json/.png/.html  含道路坐标的知识图谱
"""

import json
from pathlib import Path

from modules.road_plane_projection import project_scene_to_ground, compute_homography
from modules.road_geometry import run as run_road_geometry
from modules.calibration_check import run as run_calibration_check
from modules.scene_topdown import build_scene_figure
from modules.scene_interactive import build_scene_html
from modules.knowledge_graph import (build_fused_knowledge_graph,
                                     visualize_fused_graph)
from modules.interactive_graph import build_interactive_3d_figure, write_graph_html

ROOT = Path(__file__).resolve().parent
SCENE_DIR = ROOT / "outputs" / "ACCIDENT_001"

SCENE_PATH = SCENE_DIR / "scene_model.json"
CALIBRATION_PATH = SCENE_DIR / "calibration.json"
IMAGE_PATH = SCENE_DIR / "original.jpg"

SPATIAL_SCENE_PATH = SCENE_DIR / "scene_model_spatial.json"
GEOM_LOCAL_PATH = SCENE_DIR / "road_geometry_local.json"
SURFACE_PATH = SCENE_DIR / "road_surface_topdown.png"
DISTANCES_PATH = SCENE_DIR / "ground_distances.json"
INFO_PATH = SCENE_DIR / "spatial_model_info.json"
CHECK_PATH = SCENE_DIR / "calibration_check.json"
TOPDOWN_PATH = SCENE_DIR / "scene_topdown.png"
TOPDOWN_HTML_PATH = SCENE_DIR / "scene_topdown.html"
GRAPH_JSON_PATH = SCENE_DIR / "knowledge_graph_spatial.json"
GRAPH_PNG_PATH = SCENE_DIR / "knowledge_graph_spatial.png"
GRAPH_HTML_PATH = SCENE_DIR / "interactive_graph_spatial.html"

XLIM, YLIM = (-3.9, 10.0), (-0.45, 3.3)


def graph_to_json(graph):
    nodes = [{"id": n, **d} for n, d in graph.nodes(data=True)]
    edges = [{"source": u, "target": v, **d} for u, v, d in graph.edges(data=True)]
    return {"nodes": nodes, "edges": edges}


def main():
    for p in (SCENE_PATH, CALIBRATION_PATH):
        if not p.exists():
            raise FileNotFoundError(f"缺少输入文件：{p}")

    print("=" * 68)
    print("第四阶段：道路平面场景重建（几何 + 投影 + 场景 + 图谱）")
    print("=" * 68)

    # ---------- 第一步：道路平面投影 ----------
    spatial_scene, summary = project_scene_to_ground(
        SCENE_PATH, CALIBRATION_PATH, SPATIAL_SCENE_PATH,
        distances_path=DISTANCES_PATH, info_path=INFO_PATH,
    )
    c = summary["counters"]
    print(f"场景对象总数：{len(spatial_scene['objects'])}")
    print(f"  标定区内投影（projected）        ：{c['projected']}")
    print(f"  外推投影（projected_extrapolated）：{c['projected_extrapolated']}")
    print(f"  超出标定范围（outside）          ：{c['outside_calibration_region']}")
    print(f"  悬空设施排除                     ：{c['excluded_non_ground_object']}")
    print(f"投影成功率：{summary['success_rate_percent']}%")
    print(f"距离对：{summary['pair_count']} 对，其中可靠 {summary.get('reliable_pair_count', 0)} 对")

    # ---------- 第二步：道路几何投影 + 路面俯视底图 ----------
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    H, _ = compute_homography(calibration["image_points"], calibration["ground_points"])
    geom = json.loads((SCENE_DIR / "road_geometry.json").read_text(encoding="utf-8"))
    geometry_local, rgba, extent, mask = run_road_geometry(
        SCENE_DIR, IMAGE_PATH, SPATIAL_SCENE_PATH, H, XLIM, YLIM, px_per_unit=160)
    coverage = float((mask > 0).mean() * 100)
    # 把道路几何写回空间场景，供知识图谱读取
    spatial_scene["road_geometry"] = geometry_local
    spatial_scene["road_geometry_source"] = str((SCENE_DIR / "road_geometry.json").name)
    with open(SPATIAL_SCENE_PATH, "w", encoding="utf-8") as f:
        json.dump(spatial_scene, f, ensure_ascii=False, indent=2)
    print(f"道路几何：多边形 {len(geom.get('road_polygon', []))} 点，"
          f"车道线 {len(geom.get('lane_lines', []))} 条，"
          f"边界 {len(geom.get('road_boundaries', []))} 条")
    print(f"路面纹理覆盖：{coverage:.1f}% 像素（其余被车辆遮挡，只能按几何推断）")

    # ---------- 第三步：投影质量评估 ----------
    check = run_calibration_check(CALIBRATION_PATH, SCENE_DIR / "road_geometry.json",
                                  SPATIAL_SCENE_PATH, CHECK_PATH)
    par = check.get("lane_pair_parallelism", {})
    sens = check.get("anchor_sensitivity", {})
    print(f"车道线平行度偏差：{par.get('angle_between_lanes_deg', 'n/a')}°；"
          f"锚点 5px 抖动 → {sens.get('mean_local_unit', 'n/a')} local_unit（均值）")

    # ---------- 第四步：二维场景重建（静态图 + 交互页） ----------
    build_scene_figure(SPATIAL_SCENE_PATH, GEOM_LOCAL_PATH, SURFACE_PATH, TOPDOWN_PATH,
                       xlim=XLIM, ylim=YLIM)
    print(f"道路平面场景图：{TOPDOWN_PATH.name}")

    build_scene_html(SPATIAL_SCENE_PATH, GEOM_LOCAL_PATH, TOPDOWN_HTML_PATH,
                     surface_png_name=SURFACE_PATH.name,
                     xlim=XLIM, ylim=YLIM,
                     include_plotlyjs="directory")
    print(f"交互场景页：{TOPDOWN_HTML_PATH.name}")

    # ---------- 第五步：知识图谱（含道路坐标属性） ----------
    graph = build_fused_knowledge_graph(spatial_scene)
    with open(GRAPH_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(graph_to_json(graph), f, ensure_ascii=False, indent=2)
    visualize_fused_graph(graph).save(GRAPH_PNG_PATH)
    fig3d = build_interactive_3d_figure(graph)
    write_graph_html(fig3d, GRAPH_HTML_PATH)
    n_nodes, n_edges = graph.number_of_nodes(), graph.number_of_edges()
    print(f"知识图谱：{n_nodes} 节点 / {n_edges} 边（含道路坐标属性与 CoordinateSystem 节点）")

    print("=" * 68)
    print("第四阶段完成。生成文件：")
    for p in (GEOM_LOCAL_PATH, SURFACE_PATH, SPATIAL_SCENE_PATH, DISTANCES_PATH,
              CHECK_PATH, INFO_PATH, TOPDOWN_PATH, TOPDOWN_HTML_PATH,
              GRAPH_JSON_PATH, GRAPH_PNG_PATH, GRAPH_HTML_PATH):
        if p.exists():
            print(f"  - {p.name}  ({p.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
