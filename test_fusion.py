"""第三阶段融合流程测试：把 VLM 与 YOLO 数据融合成统一场景模型。

运行：
    python test_fusion.py

读取：
    data/vlm_result.json       第一阶段 VLM 结果
    data/yolo_detections.json  第二阶段 YOLO 结果（D 编号）
    data/object_mapping.json  人工匹配表
生成：
    outputs/ACCIDENT_001/ 下的完整数据包
"""

import json
import shutil
from pathlib import Path

import networkx as nx

from modules.scene_fusion import (
    build_fused_scene,
    load_json,
    save_json,
    validate_scene,
)
from modules.knowledge_graph import build_fused_knowledge_graph, visualize_fused_graph

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs" / "ACCIDENT_001"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def graph_to_json(graph: nx.MultiDiGraph):
    """把 MultiDiGraph 导出为可序列化的节点/边结构。"""
    nodes = []
    for n, data in graph.nodes(data=True):
        nodes.append({"id": n, **data})
    edges = []
    for u, v, data in graph.edges(data=True):
        edges.append({"source": u, "target": v, **data})
    return {"nodes": nodes, "edges": edges}


def main():
    vlm_data = load_json(DATA_DIR / "vlm_result.json")
    yolo_data = load_json(DATA_DIR / "yolo_detections.json")
    mapping = load_json(DATA_DIR / "object_mapping.json")

    scene = build_fused_scene(
        vlm_data=vlm_data,
        yolo_data=yolo_data,
        object_mapping=mapping,
        scene_id="ACCIDENT_001",
    )

    # 质量校验
    validate_scene(scene)
    print("质量校验通过：对象 ID 唯一、bbox 在图内、关系端点存在、来源标记正确。")

    # 知识图谱
    graph = build_fused_knowledge_graph(scene)
    graph_image = visualize_fused_graph(graph)

    # 写出完整数据包
    save_json(scene, OUT_DIR / "scene_model.json")
    save_json(scene["geometry_relations"], OUT_DIR / "spatial_relations.json")
    save_json(graph_to_json(graph), OUT_DIR / "knowledge_graph.json")
    graph_image.save(OUT_DIR / "knowledge_graph.png")

    # 复制输入数据进数据包
    for name in ("vlm_result.json", "yolo_detections.json", "object_mapping.json"):
        shutil.copy(DATA_DIR / name, OUT_DIR / name)

    # 摘要
    matched = [o for o in scene["objects"] if o["semantic_match_status"] == "manually_confirmed"]
    print(f"\n统一对象数: {len(scene['objects'])}")
    print(f"  人工确认匹配: {len(matched)}")
    for o in matched:
        print(f"    {o['id']} = VLM[{o['vlm_id']}] + YOLO[{o['detection_id']}] "
              f"({o['category_zh']}) - {o['vlm_description'][:24]}")
    print(f"语义关系(L3): {len(scene['semantic_relations'])} 条")
    print(f"几何关系(L2): {len(scene['geometry_relations'])} 条")
    print(f"未对齐关系: {len(scene['unresolved_relations'])} 条")
    print(f"事故事件: {scene['accident_event']['type']} "
          f"(置信度 {scene['accident_event']['confidence']})")
    print(f"知识图谱: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
    print(f"\n数据包已保存到: {OUT_DIR}")


if __name__ == "__main__":
    main()
