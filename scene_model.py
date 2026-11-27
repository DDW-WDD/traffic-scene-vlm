"""第二阶段核心模块：交通事故场景建模。

处理链路：
    PIL 图片
        -> detect_traffic_objects   YOLO 检测 + 像素坐标提取
        -> build_scene_data         统一交通场景数据结构
        -> calculate_spatial_relations  计算对象间空间关系
        -> build_knowledge_graph    用 NetworkX 构建交通事故知识图谱
        -> visualize_*              检测框 / 空间关系 / 知识图谱可视化
        -> process_traffic_scene    总控函数，一次跑完整条链路

重要说明：
1. 当前 P1、P2 只是“单张图片内”的对象编号，不是跨视频帧的跟踪 ID。
2. bbox 重叠、目标接近都只是图像平面上的几何事实，不能据此认定真实碰撞。
3. 道路区域 R1 只有在可靠识别后才创建；当前版本不自动生成道路节点。
"""

import io
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # 无界面后端，只把图写入内存，不弹窗
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image, ImageDraw
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# 常量配置
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "yolo26n.pt"

# 第二阶段关注的六类交通参与者（COCO 英文类别名）
TARGET_CLASSES = {
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "person",
}

CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 640

# 归一化中心距离低于该值，标记为“接近候选”。
# 这是初期实验的启发式阈值，不是交通工程通用标准，后续需按数据集调整。
NEAR_DISTANCE_THRESHOLD = 0.15

# 中文类别名映射，便于可视化与论文展示
CATEGORY_ZH = {
    "car": "小汽车",
    "truck": "卡车",
    "bus": "公交车",
    "motorcycle": "摩托车",
    "bicycle": "自行车",
    "person": "行人",
}

# 模型全局单例，避免每次请求都重新加载权重
_model: Optional[YOLO] = None


def get_model() -> YOLO:
    """懒加载并缓存 YOLO 模型。"""
    global _model
    if _model is None:
        _model = YOLO(str(MODEL_PATH))
    return _model


# ---------------------------------------------------------------------------
# 1. YOLO 检测 + 像素坐标提取
# ---------------------------------------------------------------------------
def detect_traffic_objects(
    image: Image.Image,
) -> Tuple[List[Dict], Image.Image]:
    """识别图片中的交通参与者并提取像素坐标。

    参数：
        image: PIL 图片。
    返回：
        (participants, annotated_image)
        participants 为统一格式的参与者字典列表；
        annotated_image 为画好检测框的 PIL 图片。
    """
    model = get_model()
    results = model.predict(
        source=image,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=IMAGE_SIZE,
        verbose=False,
    )
    result = results[0]
    image_width, image_height = image.size

    participants: List[Dict] = []
    ordinal = 0

    for box in result.boxes:
        cls_id = int(box.cls[0])
        category = result.names[cls_id]
        if category not in TARGET_CLASSES:
            continue  # 只保留六类交通参与者，过滤交通信号灯等其他目标

        ordinal += 1
        confidence = round(float(box.conf[0]), 3)
        x_min, y_min, x_max, y_max = [
            float(value) for value in box.xyxy[0]
        ]

        # 中心点、宽高
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min

        # 归一化坐标：不同分辨率图片可使用相同的相对位置描述
        normalized_x = round(center_x / image_width, 4)
        normalized_y = round(center_y / image_height, 4)

        participants.append(
            {
                "id": f"P{ordinal}",
                "category": category,
                "category_zh": CATEGORY_ZH[category],
                "confidence": confidence,
                "bbox": [
                    round(x_min, 1),
                    round(y_min, 1),
                    round(x_max, 1),
                    round(y_max, 1),
                ],
                "center": [round(center_x, 1), round(center_y, 1)],
                "width": round(width, 1),
                "height": round(height, 1),
                "image_position_normalized": [normalized_x, normalized_y],
            }
        )

    # result.plot() 返回 BGR numpy 数组，转成 PIL 图片
    annotated_bgr = result.plot()
    annotated_image = Image.fromarray(annotated_bgr[:, :, ::-1])

    return participants, annotated_image


# ---------------------------------------------------------------------------
# 2. 统一交通场景数据结构
# ---------------------------------------------------------------------------
def build_scene_data(
    image: Image.Image,
    participants: List[Dict],
    vlm_analysis: Optional[Dict] = None,
    scene_id: str = "ACCIDENT_001",
) -> Dict:
    """构建统一的交通场景数据对象（整个毕业设计的数据中心）。"""
    image_width, image_height = image.size
    return {
        "scene_id": scene_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "image_info": {
            "width": image_width,
            "height": image_height,
            "source": "single_image",
        },
        "participants": participants,
        "spatial_relations": [],
        "road_environment": {
            # 占位值，不代表已识别出道路类型
            "road_type": "待分析",
            "visible_facilities": [],
        },
        "accident_analysis": vlm_analysis if vlm_analysis else {},
    }


# ---------------------------------------------------------------------------
# 3. 空间关系计算
# ---------------------------------------------------------------------------
def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """计算两个边界框的交并比 IoU（取值 0~1）。"""
    ax_min, ay_min, ax_max, ay_max = box_a
    bx_min, by_min, bx_max, by_max = box_b

    inter_x_min = max(ax_min, bx_min)
    inter_y_min = max(ay_min, by_min)
    inter_x_max = min(ax_max, bx_max)
    inter_y_max = min(ay_max, by_max)

    inter_width = max(0.0, inter_x_max - inter_x_min)
    inter_height = max(0.0, inter_y_max - inter_y_min)
    intersection = inter_width * inter_height

    area_a = (ax_max - ax_min) * (ay_max - ay_min)
    area_b = (bx_max - bx_min) * (by_max - by_min)
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def calculate_spatial_relations(participants: List[Dict]) -> List[Dict]:
    """根据检测框，用数学方法计算每对参与者之间的初步空间关系。

    包含：水平/垂直方位关系、IoU 重叠、归一化中心距离与接近标记。
    """
    relations: List[Dict] = []

    for i in range(len(participants)):
        for j in range(i + 1, len(participants)):
            a = participants[i]
            b = participants[j]

            ax, ay = a["center"]
            bx, by = b["center"]

            # 水平方位：比较中心点横坐标（图像坐标，不代表真实道路方向）
            horizontal = "left_of" if ax < bx else "right_of"
            # 垂直方位：比较中心点纵坐标，y 轴向下为正
            vertical = "above" if ay < by else "below"

            # 归一化中心欧氏距离
            an = a["image_position_normalized"]
            bn = b["image_position_normalized"]
            distance = math.sqrt(
                (an[0] - bn[0]) ** 2 + (an[1] - bn[1]) ** 2
            )

            # 边界框 IoU
            iou = compute_iou(a["bbox"], b["bbox"])

            near_candidate = distance < NEAR_DISTANCE_THRESHOLD
            overlap_candidate = iou > 0.0

            relations.append(
                {
                    "source": a["id"],
                    "target": b["id"],
                    "horizontal_relation": horizontal,
                    "vertical_relation": vertical,
                    "near_candidate": near_candidate,
                    "bbox_overlap_candidate": overlap_candidate,
                    "metrics": {
                        "center_distance_normalized": round(distance, 4),
                        "iou": round(iou, 4),
                    },
                    "evidence": (
                        f"几何计算：{a['id']} 中心({ax},{ay})，"
                        f"{b['id']} 中心({bx},{by})；"
                        f"归一化距离 {distance:.3f}，IoU {iou:.3f}"
                    ),
                }
            )

    return relations


# ---------------------------------------------------------------------------
# 4. NetworkX 交通事故知识图谱
# ---------------------------------------------------------------------------
def build_knowledge_graph(scene: Dict) -> nx.DiGraph:
    """根据统一场景数据构建有向知识图谱。

    节点类型：Scene / Vehicle / Pedestrian
    关系类型：belongs_to / left_of / right_of / above / below /
              near_candidate / bbox_overlap_candidate
    """
    graph = nx.DiGraph()
    scene_id = scene["scene_id"]

    # 场景节点
    graph.add_node(
        scene_id,
        node_type="Scene",
        label=scene_id,
    )

    # 参与者实体节点
    for participant in scene["participants"]:
        pid = participant["id"]
        is_person = participant["category"] == "person"
        node_type = "Pedestrian" if is_person else "Vehicle"
        graph.add_node(
            pid,
            node_type=node_type,
            label=f"{pid}·{participant['category_zh']}",
            category=participant["category"],
            confidence=participant["confidence"],
            center=participant["center"],
            bbox=participant["bbox"],
        )
        # 参与者属于该场景
        graph.add_edge(pid, scene_id, relation="belongs_to")

    # 空间关系边
    for rel in scene["spatial_relations"]:
        source = rel["source"]
        target = rel["target"]

        graph.add_edge(
            source,
            target,
            relation=rel["horizontal_relation"],
            evidence=rel["evidence"],
            **rel["metrics"],
        )
        graph.add_edge(
            source,
            target,
            relation=rel["vertical_relation"],
            evidence=rel["evidence"],
            **rel["metrics"],
        )
        if rel["near_candidate"]:
            graph.add_edge(
                source,
                target,
                relation="near_candidate",
                evidence=rel["evidence"],
                **rel["metrics"],
            )
        if rel["bbox_overlap_candidate"]:
            graph.add_edge(
                source,
                target,
                relation="bbox_overlap_candidate",
                evidence=rel["evidence"],
                **rel["metrics"],
            )

    return graph


# ---------------------------------------------------------------------------
# 5. 可视化
# ---------------------------------------------------------------------------
def _node_color(graph: nx.DiGraph, node) -> str:
    node_type = graph.nodes[node].get("node_type")
    return {
        "Scene": "#f4a261",
        "Vehicle": "#8ecae6",
        "Pedestrian": "#90be6d",
    }.get(node_type, "#cccccc")


def visualize_knowledge_graph(graph: nx.DiGraph) -> Image.Image:
    """把知识图谱绘制为 PIL 图片。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(12.5, 9))
    pos = nx.spring_layout(graph, seed=42, k=2.2)

    node_colors = [_node_color(graph, node) for node in graph.nodes]
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=2600,
        node_color=node_colors,
        edgecolors="#333333",
        linewidths=1.2,
        ax=ax,
    )

    labels = {
        node: graph.nodes[node].get("label", node) for node in graph.nodes
    }
    nx.draw_networkx_labels(
        graph, pos, labels=labels, font_size=10, ax=ax
    )

    # 把边按“属于场景”和“参与者之间的语义关系”分开聚合。
    # 方位关系（left/right/above/below）每对节点都存在，全画会变成完全图、
    # 没有重点，因此可视化只突出 near_candidate / bbox_overlap_candidate。
    belongs_edges = []
    pair_relations = {}
    for u, v, data in graph.edges(data=True):
        if data["relation"] == "belongs_to":
            belongs_edges.append((u, v))
        else:
            pair_relations.setdefault((u, v), set()).add(data["relation"])

    # belongs_to：灰色虚线，不标文字
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=belongs_edges,
        style="dashed",
        edge_color="#999999",
        width=1.0,
        arrows=False,
        ax=ax,
    )

    semantic_edges = []
    semantic_colors = []
    edge_label_text = {}
    for (u, v), rels in pair_relations.items():
        has_near = "near_candidate" in rels
        has_overlap = "bbox_overlap_candidate" in rels
        if not (has_near or has_overlap):
            continue  # 仅方位关系的对，不在主图中绘制

        semantic_edges.append((u, v))
        if has_near and has_overlap:
            color = "#d62828"
        elif has_near:
            color = "#9d4edd"
        else:
            color = "#f77f00"
        semantic_colors.append(color)

        text_parts = []
        if has_near:
            text_parts.append("接近")
        if has_overlap:
            text_parts.append("框重叠")
        horiz = next(
            (r for r in rels if r in ("left_of", "right_of")), None
        )
        if horiz:
            text_parts.append(horiz)
        edge_label_text[(u, v)] = "\n".join(text_parts)

    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=semantic_edges,
        edge_color=semantic_colors,
        connectionstyle="arc3,rad=0.1",
        arrows=True,
        arrowsize=13,
        width=2.0,
        ax=ax,
    )
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_label_text,
        font_size=8,
        connectionstyle="arc3,rad=0.1",
        ax=ax,
    )

    legend_items = [
        Line2D([0], [0], color="#999999", lw=1, ls="--",
               label="属于场景 belongs_to"),
        Line2D([0], [0], color="#9d4edd", lw=2,
               label="接近候选 near_candidate"),
        Line2D([0], [0], color="#f77f00", lw=2,
               label="检测框重叠 bbox_overlap_candidate"),
        Line2D([0], [0], color="#d62828", lw=2,
               label="同时接近且重叠"),
    ]
    ax.legend(
        handles=legend_items, loc="lower left", fontsize=9, framealpha=0.9
    )

    ax.set_title("交通事故知识图谱（NetworkX）", fontsize=14)
    ax.axis("off")

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


def visualize_spatial_relations(
    image: Image.Image, participants: List[Dict]
) -> Image.Image:
    """在图片坐标系中绘制参与者位置、ID 与接近关系连线。"""
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")

    for participant in participants:
        x_min, y_min, x_max, y_max = participant["bbox"]
        center_x, center_y = participant["center"]

        # 半透明边界框
        draw.rectangle(
            [x_min, y_min, x_max, y_max],
            outline=(255, 215, 0, 255),
            width=3,
        )
        # 中心点
        radius = 6
        draw.ellipse(
            [
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ],
            fill=(255, 0, 0, 255),
        )
        # ID 标签
        draw.text(
            (x_min + 4, y_min + 4),
            participant["id"],
            fill=(255, 255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0, 255),
        )

    return canvas


def draw_near_links(
    image: Image.Image,
    participants: List[Dict],
    relations: List[Dict],
) -> Image.Image:
    """在空间关系图上为“接近候选”对象绘制中心连线。"""
    canvas = visualize_spatial_relations(image, participants)
    draw = ImageDraw.Draw(canvas, "RGBA")

    center_map = {p["id"]: tuple(p["center"]) for p in participants}
    for rel in relations:
        if not rel["near_candidate"]:
            continue
        source = center_map.get(rel["source"])
        target = center_map.get(rel["target"])
        if source and target:
            draw.line(
                [source, target], fill=(255, 0, 255, 200), width=2
            )
    return canvas


# ---------------------------------------------------------------------------
# 6. 总控函数
# ---------------------------------------------------------------------------
def process_traffic_scene(
    image: Image.Image,
    vlm_analysis: Optional[Dict] = None,
) -> Dict:
    """一次跑通第二阶段完整处理链。

    返回字典包含：
        scene            统一场景数据（含参与者与空间关系）
        annotated_image  YOLO 检测框标注图
        spatial_image    图片坐标系空间关系图
        graph            NetworkX 知识图谱对象
        graph_image      知识图谱可视化图
    """
    # 第一步：YOLO 检测 + 坐标提取
    participants, annotated_image = detect_traffic_objects(image)

    # 第二步：统一场景数据
    scene = build_scene_data(image, participants, vlm_analysis)

    # 第三步：空间关系
    relations = calculate_spatial_relations(participants)
    scene["spatial_relations"] = relations

    # 第四步：知识图谱
    graph = build_knowledge_graph(scene)

    # 第五步：可视化
    graph_image = visualize_knowledge_graph(graph)
    spatial_image = draw_near_links(image, participants, relations)

    return {
        "scene": scene,
        "annotated_image": annotated_image,
        "spatial_image": spatial_image,
        "graph": graph,
        "graph_image": graph_image,
    }
