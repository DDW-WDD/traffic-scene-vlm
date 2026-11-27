"""第四阶段运行脚本：道路平面投影 + 二维场景重建 + 图谱更新。

运行：
    python run_stage4.py

输入：
    outputs/ACCIDENT_001/scene_model.json       第三阶段融合场景
    outputs/ACCIDENT_001/calibration.json       道路控制点标定

输出（全部写入 outputs/ACCIDENT_001/）：
    scene_model_spatial.json   带道路平面坐标的空间场景（不覆盖第三阶段文件）
    ground_distances.json      道路平面两两距离
    spatial_model_info.json    标定/矩阵/成功率/局限说明
    scene_topdown.png          道路平面二维场景重建图
    knowledge_graph_spatial.json / .png  含道路坐标的知识图谱
"""

import json
from pathlib import Path

from modules.road_plane_projection import project_scene_to_ground
from modules.spatial_visualization import visualize_spatial_scene
from modules.knowledge_graph import (build_fused_knowledge_graph,
                                     visualize_fused_graph)
from modules.interactive_graph import build_interactive_3d_figure

ROOT = Path(__file__).resolve().parent
SCENE_DIR = ROOT / "outputs" / "ACCIDENT_001"

SCENE_PATH = SCENE_DIR / "scene_model.json"
CALIBRATION_PATH = SCENE_DIR / "calibration.json"

SPATIAL_SCENE_PATH = SCENE_DIR / "scene_model_spatial.json"
DISTANCES_PATH = SCENE_DIR / "ground_distances.json"
INFO_PATH = SCENE_DIR / "spatial_model_info.json"
TOPDOWN_PATH = SCENE_DIR / "scene_topdown.png"
GRAPH_JSON_PATH = SCENE_DIR / "knowledge_graph_spatial.json"
GRAPH_PNG_PATH = SCENE_DIR / "knowledge_graph_spatial.png"
GRAPH_HTML_PATH = SCENE_DIR / "interactive_graph_spatial.html"


def graph_to_json(graph):
    nodes = [{"id": n, **d} for n, d in graph.nodes(data=True)]
    edges = [{"source": u, "target": v, **d} for u, v, d in graph.edges(data=True)]
    return {"nodes": nodes, "edges": edges}


def main():
    if not SCENE_PATH.exists():
        raise FileNotFoundError(f"缺少第三阶段场景文件：{SCENE_PATH}")
    if not CALIBRATION_PATH.exists():
        raise FileNotFoundError(f"缺少标定文件：{CALIBRATION_PATH}")

    print("=" * 64)
    print("第四阶段：道路平面空间建模（单应性变换）")
    print("=" * 64)

    # 第一步：坐标投影
    spatial_scene, summary = project_scene_to_ground(
        SCENE_PATH, CALIBRATION_PATH, SPATIAL_SCENE_PATH,
        distances_path=DISTANCES_PATH, info_path=INFO_PATH,
    )
    c = summary["counters"]
    print(f"场景对象总数：{len(spatial_scene['objects'])}")
    print(f"标定区内投影（projected）        ：{c['projected']}")
    print(f"外推投影（projected_extrapolated）：{c['projected_extrapolated']}")
    print(f"超出标定范围（outside）          ：{c['outside_calibration_region']}")
    print(f"悬空设施排除                     ：{c['excluded_non_ground_object']}")
    print(f"无检测框                           ：{c['no_bbox_available']}")
    print(f"投影成功率                        ：{summary['success_rate_percent']}%")
    print(f"道路平面距离对数                  ：{summary['pair_count']}")
    print(f"自洽性误差（仅验证求解正确）      ："
          f"{summary['reprojection']['reprojection_error_max']}")

    # 关键对象坐标
    print("-" * 64)
    print("交通参与者道路平面坐标：")
    for obj in spatial_scene["objects"]:
        gp = obj.get("ground_position", {})
        st = gp.get("status")
        if st in ("projected", "projected_extrapolated"):
            print(f"  {obj['id']:>4} {obj.get('category_zh', ''):<6} "
                  f"X={gp['x']:>7.3f} Y={gp['y']:>7.3f} [{st}]")
    print("-" * 64)
    print("语义关系的道路平面距离：")
    for rel in spatial_scene.get("semantic_relations", []):
        gd = rel.get("ground_distance", {})
        if gd.get("value") is not None:
            print(f"  {rel['subject']} --{rel['predicate']}--> {rel['object']} : "
                  f"{gd['value']} {gd['unit']}")

    # 第二步：二维场景重建
    visualize_spatial_scene(SPATIAL_SCENE_PATH, CALIBRATION_PATH, TOPDOWN_PATH)
    print(f"道路平面场景图已生成：{TOPDOWN_PATH.name}")

    # 第三步：知识图谱（含道路平面坐标）
    graph = build_fused_knowledge_graph(spatial_scene)
    with open(GRAPH_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(graph_to_json(graph), f, ensure_ascii=False, indent=2)
    visualize_fused_graph(graph).save(GRAPH_PNG_PATH)
    fig3d = build_interactive_3d_figure(graph)
    fig3d.write_html(GRAPH_HTML_PATH, include_plotlyjs=True, full_html=True)
    print(f"知识图谱：{graph.number_of_nodes()} 节点 / "
          f"{graph.number_of_edges()} 边（含道路平面坐标引用）")

    print("=" * 64)
    print("第四阶段完成。生成文件：")
    for p in (SPATIAL_SCENE_PATH, DISTANCES_PATH, INFO_PATH, TOPDOWN_PATH,
              GRAPH_JSON_PATH, GRAPH_PNG_PATH, GRAPH_HTML_PATH):
        print(f"  - {p.name}  ({p.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
