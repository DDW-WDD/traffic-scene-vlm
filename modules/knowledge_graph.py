"""第三阶段：融合知识图谱（NetworkX MultiDiGraph）。

- MultiDiGraph：同一对节点之间可保留多条具有独立属性的平行边
  （例如 O1→O2 同时有 near_candidate、bbox_overlap_candidate、vlm_contact）。
- 关系分三层：
    L1 视觉事实（YOLO 检测到的对象节点）
    L2 几何关系（near_candidate / bbox_overlap_candidate，status=candidate）
    L3 语义关系（VLM 观察到的 contact 等，status=observed）
- 事故事件单独作为节点，对象通过 accident_involves 边关联到它。
"""

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from PIL import Image


def _node_type(obj):
    group = obj.get("object_group")
    if group == "facility":
        return "Facility"
    if group == "vlm_only":
        return "VLMROnly"
    if obj.get("category") == "person":
        return "Pedestrian"
    return "Vehicle"


def _node_color(node_type):
    return {
        "Scene": "#f4a261",
        "Vehicle": "#8ecae6",
        "Pedestrian": "#90be6d",
        "Facility": "#d0bfff",
        "VLMROnly": "#e0e0e0",
        "AccidentEvent": "#ff8787",
        "CoordinateSystem": "#74c0fc",
        "RoadArea": "#dee2e6",
        "LaneLine": "#ced4da",
        "RoadBoundary": "#ffe066",
    }.get(node_type, "#cccccc")


def build_fused_knowledge_graph(scene):
    """从融合场景构建 MultiDiGraph 知识图谱。"""
    G = nx.MultiDiGraph()
    scene_id = scene["scene_id"]

    # 场景节点
    G.add_node(scene_id, node_type="Scene", label=scene_id)

    # 对象节点
    for obj in scene["objects"]:
        ntype = _node_type(obj)
        gp = obj.get("ground_position") or {}
        G.add_node(
            obj["id"],
            node_type=ntype,
            label=(obj.get("vlm_description") or "")[:14] or obj["category_zh"],
            category=obj["category"],
            category_zh=obj["category_zh"],
            confidence=obj.get("confidence"),
            match_status=obj["semantic_match_status"],
            # 第四阶段：道路平面坐标（未投影时保留状态，不伪造坐标）
            ground_position=gp,
            ground_status=gp.get("status"),
            # 第四阶段：把坐标平铺成可查询属性，便于按属性筛选/检索
            ground_x=gp.get("x"),
            ground_y=gp.get("y"),
            coordinate_unit=gp.get("unit") if gp.get("x") is not None else None,
            projection_status=gp.get("status"),
            coordinate_source=("homography_4pt" if gp.get("x") is not None else None),
            ground_anchor_image=obj.get("ground_anchor_image"),
            position_text=(
                f"X={gp['x']:.2f}, Y={gp['y']:.2f} {gp.get('unit', '')}"
                if gp.get("x") is not None else
                f"未给出道路坐标（{gp.get('status')}）"
            ),
        )
        G.add_edge(obj["id"], scene_id, relation="belongs_to")

    # 第四阶段：道路局部坐标系节点 + 位置引用关系
    road_plane = scene.get("road_plane") or {}
    has_ground = any(
        (o.get("ground_position") or {}).get("status") in
        ("projected", "projected_extrapolated")
        for o in scene["objects"]
    )
    if has_ground:
        G.add_node(
            road_plane.get("coordinate_system", "ROAD_PLANE_001"),
            node_type="CoordinateSystem",
            label="道路局部坐标系",
            unit=road_plane.get("unit"),
            homography_matrix=road_plane.get("homography_matrix"),
            calibration_status=road_plane.get("calibration_status"),
        )
        for obj in scene["objects"]:
            gp = obj.get("ground_position") or {}
            if gp.get("status") not in ("projected", "projected_extrapolated"):
                continue
            G.add_edge(
                obj["id"],
                road_plane.get("coordinate_system", "ROAD_PLANE_001"),
                relation="position_in_coordinate_system",
                source="homography",
                status=gp.get("status"),
                ground_x=gp.get("x"),
                ground_y=gp.get("y"),
                unit=gp.get("unit"),
            )

    # 第四阶段：道路结构要素入图（人工标注 + 单应性投影，属场景几何层）
    road_geom = scene.get("road_geometry")
    if road_geom:
        G.add_node(
            "ROAD_AREA",
            node_type="RoadArea",
            label="路面区域（人工标注）",
            unit=road_geom.get("unit"),
            quality=road_geom.get("quality", {}).get("type"),
            point_count=len(road_geom.get("road_polygon", [])),
            source="manual_annotation",
        )
        G.add_edge("ROAD_AREA", scene_id, relation="belongs_to")
        G.add_edge("ROAD_AREA", road_plane.get("coordinate_system", "ROAD_PLANE_001"),
                   relation="position_in_coordinate_system",
                   source="homography", status="projected")

        for lane in road_geom.get("lane_lines", []):
            G.add_node(
                lane["id"],
                node_type="LaneLine",
                label=lane.get("name", lane["id"]),
                line_type=lane.get("type"),
                fit=lane.get("fit"),
                point_count=len(lane.get("points_local", [])),
                source="manual_annotation",
            )
            G.add_edge(lane["id"], "ROAD_AREA", relation="belongs_to")
            G.add_edge(lane["id"], road_plane.get("coordinate_system", "ROAD_PLANE_001"),
                       relation="position_in_coordinate_system",
                       source="homography", status="projected")

        for b in road_geom.get("road_boundaries", []):
            G.add_node(
                b["id"],
                node_type="RoadBoundary",
                label=b.get("name", b["id"]),
                visibility=b.get("visibility"),
                quality=b.get("quality"),
                point_count=len(b.get("points_local", [])),
                source="manual_annotation",
            )
            G.add_edge(b["id"], "ROAD_AREA", relation="belongs_to")
            G.add_edge(b["id"], road_plane.get("coordinate_system", "ROAD_PLANE_001"),
                       relation="position_in_coordinate_system",
                       source="homography", status="projected")

    # 事故事件节点
    accident = scene.get("accident_event", {})
    accident_label = accident.get("type", "未知事件")
    G.add_node(
        "ACCIDENT_EVENT",
        node_type="AccidentEvent",
        label=f"事故事件：{accident_label}",
        confidence=accident.get("confidence"),
        source="vlm",
    )
    # 所有参与者关联到事故事件
    for obj in scene["objects"]:
        if obj["object_group"] == "participant":
            G.add_edge(obj["id"], "ACCIDENT_EVENT", relation="accident_involves")

    # 语义关系（VLM，L3）
    for rel in scene.get("semantic_relations", []):
        G.add_edge(
            rel["subject"],
            rel["object"],
            relation=rel["predicate"],
            source="vlm",
            status="observed",
            evidence=rel.get("evidence", ""),
        )

    # 几何关系（L2，候选）
    for rel in scene.get("geometry_relations", []):
        G.add_edge(
            rel["subject"],
            rel["object"],
            relation=rel["predicate"],
            source="geometry",
            status="candidate",
            evidence=rel.get("evidence", ""),
            iou=rel.get("metrics", {}).get("iou"),
        )

    return G


def visualize_fused_graph(graph: nx.MultiDiGraph) -> Image.Image:
    """绘制融合知识图谱并返回 PIL 图片。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(15.5, 10.5))
    pos = nx.spring_layout(graph, seed=42, k=2.9)

    # 节点
    node_colors = [_node_color(graph.nodes[n].get("node_type")) for n in graph.nodes]
    nx.draw_networkx_nodes(
        graph, pos, node_size=3600, node_color=node_colors,
        edgecolors="#333333", linewidths=1.2, ax=ax,
    )
    labels = {}
    id_types = ("Vehicle", "Pedestrian", "Facility")
    for n in graph.nodes:
        text = graph.nodes[n].get("label", n)
        if graph.nodes[n].get("node_type") in id_types and not n.startswith("ROAD"):
            # 静态图里必须能看到对象编号，否则无法与表格/场景图对照
            text = f"{n} {text}"
        if len(text) > 16:
            text = text[:15] + "…"
        if len(text) > 7 and any("\u4e00" <= ch <= "\u9fff" for ch in text):
            # 中文长描述折成两行，避免横向撑破圆形节点
            text = text[:7] + "\n" + text[7:]
        labels[n] = text
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=7.6, ax=ax)

    # 边按来源分组绘制
    vlm_edges, geo_edges, belongs_edges, acc_edges, pos_edges = [], [], [], [], []
    edge_labels = {}
    for u, v, data in graph.edges(data=True):
        rel = data["relation"]
        if rel == "belongs_to":
            belongs_edges.append((u, v))
        elif rel == "accident_involves":
            acc_edges.append((u, v))
        elif rel == "position_in_coordinate_system":
            # 该关系数量多且语义单一，图中只画线不标字，避免标签互相压盖
            pos_edges.append((u, v))
        elif data.get("source") == "vlm":
            vlm_edges.append((u, v))
            edge_labels[(u, v)] = rel
        else:
            geo_edges.append((u, v))
            edge_labels[(u, v)] = rel

    nx.draw_networkx_edges(
        graph, pos, edgelist=pos_edges, style="dotted",
        edge_color="#4dabf7", width=1.0, arrows=False, ax=ax,
    )
    nx.draw_networkx_edges(
        graph, pos, edgelist=belongs_edges, style="dashed",
        edge_color="#bbbbbb", width=0.9, arrows=False, ax=ax,
    )
    nx.draw_networkx_edges(
        graph, pos, edgelist=acc_edges, style="dotted",
        edge_color="#ff8787", width=1.0, arrows=True, arrowsize=11, ax=ax,
    )
    nx.draw_networkx_edges(
        graph, pos, edgelist=geo_edges, style="dotted",
        edge_color="#f4a261", width=1.1, arrows=True, arrowsize=11, ax=ax,
    )
    nx.draw_networkx_edges(
        graph, pos, edgelist=vlm_edges, style="solid",
        edge_color="#e03131", width=2.6, arrows=True, arrowsize=16, ax=ax,
    )
    nx.draw_networkx_edge_labels(
        graph, pos, edge_labels=edge_labels, font_size=8,
        connectionstyle="arc3,rad=0.12", ax=ax,
    )

    legend = [
        Line2D([0], [0], color="#e03131", lw=2.6, label="L3 语义关系（VLM 观察）"),
        Line2D([0], [0], color="#f4a261", lw=1.1, ls=":", label="L2 几何关系（候选）"),
        Line2D([0], [0], color="#ff8787", lw=1.0, ls=":", label="事故事件关联"),
        Line2D([0], [0], color="#43a6f0", lw=1.0, ls=":", label="道路坐标引用"),
        Line2D([0], [0], color="#bbbbbb", lw=0.9, ls="--", label="属于场景"),
    ]
    ax.legend(handles=legend, loc="lower left", fontsize=9, framealpha=0.9)
    ax.set_title(f"交通事故融合知识图谱（VLM × YOLO）｜{graph.number_of_nodes()} 节点 / "
                 f"{graph.number_of_edges()} 边", fontsize=14)
    ax.axis("off")

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
