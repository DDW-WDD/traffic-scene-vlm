"""第三阶段：VLM 语义与 YOLO 几何的融合场景模型。

核心设计：
- 三套 ID：V（VLM 原始 id）、D（YOLO 检测 id）、O（融合后的统一对象 id）。
- 不覆盖原则：YOLO 的位置/置信度不会被 VLM 文字覆盖；VLM 的语义描述也不会被 YOLO 覆盖。
- 无法匹配的 YOLO 目标不丢弃；无法匹配的 VLM 关系放入 unresolved_relations。
- 每条关系都带 source（vlm / geometry）与 status（observed / candidate）。
- 几何重叠不等于碰撞：几何关系只标记为 near_candidate / bbox_overlap_candidate。
"""

import json
import math
from pathlib import Path

# 几何关系阈值（初步调试参数，非交通工程通用标准）
MAX_NEAR_DISTANCE = 0.12
MIN_IOU = 0.05

ALLOWED_GROUPS = {"participant", "facility"}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_yolo_detections(data):
    """兼容不同的 YOLO JSON 结构：list / {"detections": [...]} / {"participants": [...]}。"""
    if isinstance(data, list):
        return data
    if "detections" in data:
        return data["detections"]
    if "participants" in data:
        return data["participants"]
    raise ValueError("没有找到 YOLO 检测对象列表")


# ---------------------------------------------------------------------------
# 几何计算
# ---------------------------------------------------------------------------
def compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# 融合主逻辑
# ---------------------------------------------------------------------------
def build_fused_scene(
    vlm_data,
    yolo_data,
    object_mapping,
    scene_id="ACCIDENT_001",
    image_width=None,
    image_height=None,
):
    """把 VLM 语义数据与 YOLO 检测数据融合成统一场景对象。

    参数：
        vlm_data: 第一阶段 VLM 输出（含 participants、spatial_relations、
                  possible_accident_type、road_environment）。
        yolo_data: 第二阶段 YOLO 输出（含 detections 或 participants）。
        object_mapping: {vlm_id: detection_id} 人工匹配表。
    返回：
        融合后的场景 dict。
    """
    detections = get_yolo_detections(yolo_data)
    if image_width is None:
        image_width = yolo_data.get("image_info", {}).get("width", 0)
    if image_height is None:
        image_height = yolo_data.get("image_info", {}).get("height", 0)

    vlm_participants = {p["id"]: p for p in vlm_data.get("participants", [])}
    det_by_id = {d["id"]: d for d in detections}

    # 反向映射：detection_id -> vlm_id
    vlm_to_det = {k: v for k, v in object_mapping.items() if not k.startswith("_")}
    det_to_vlm = {v: k for k, v in vlm_to_det.items()}

    objects = []
    vlm_id_to_oid = {}
    det_id_to_oid = {}

    # 1) 先按 YOLO 检测顺序生成统一对象 O（有坐标的优先）
    for det in detections:
        det_id = det["id"]
        vlm_id = det_to_vlm.get(det_id)
        oid = f"O{len(objects) + 1}"

        vlm_info = vlm_participants.get(vlm_id) if vlm_id else None
        if vlm_info:
            match_status = "manually_confirmed"
            vlm_description = vlm_info.get("visible_features", "")
            vlm_image_position = vlm_info.get("image_position", "")
            vlm_category = vlm_info.get("category", "")
        else:
            match_status = "unmatched_detection"
            vlm_description = ""
            vlm_image_position = ""
            vlm_category = ""

        objects.append(
            {
                "id": oid,
                "detection_id": det_id,
                "vlm_id": vlm_id,
                "category": det["category"],
                "category_zh": det.get("category_zh", det["category"]),
                "object_group": det.get("object_group", "participant"),
                "bbox": det["bbox"],
                "center": det["center"],
                "confidence": det.get("confidence"),
                "vlm_description": vlm_description,
                "vlm_image_position": vlm_image_position,
                "vlm_category": vlm_category,
                "semantic_match_status": match_status,
            }
        )
        det_id_to_oid[det_id] = oid
        if vlm_id:
            vlm_id_to_oid[vlm_id] = oid

    # 2) 未被匹配到的 VLM 对象也保留为对象（没有几何框）
    for vlm_id, vlm_info in vlm_participants.items():
        if vlm_id in vlm_id_to_oid:
            continue
        oid = f"O{len(objects) + 1}"
        objects.append(
            {
                "id": oid,
                "detection_id": None,
                "vlm_id": vlm_id,
                "category": "unknown",
                "category_zh": vlm_info.get("category", "未知"),
                "object_group": "vlm_only",
                "bbox": None,
                "center": None,
                "confidence": None,
                "vlm_description": vlm_info.get("visible_features", ""),
                "vlm_image_position": vlm_info.get("image_position", ""),
                "vlm_category": vlm_info.get("category", ""),
                "semantic_match_status": "vlm_only",
            }
        )
        vlm_id_to_oid[vlm_id] = oid

    # 3) 语义关系（VLM）：两端都能映射到 O 的进入 semantic_relations，否则 unresolved
    semantic_relations = []
    unresolved_relations = []
    for rel in vlm_data.get("spatial_relations", []):
        s_vlm = rel.get("subject")
        o_vlm = rel.get("object")
        s_oid = vlm_id_to_oid.get(s_vlm)
        o_oid = vlm_id_to_oid.get(o_vlm)
        if s_oid and o_oid:
            semantic_relations.append(
                {
                    "subject": s_oid,
                    "predicate": rel.get("relation", "related_to"),
                    "object": o_oid,
                    "source": "vlm",
                    "confidence": None,
                    "status": "observed",
                    "evidence": rel.get("visual_evidence", ""),
                }
            )
        else:
            unresolved_relations.append(
                {
                    "subject": s_vlm,
                    "predicate": rel.get("relation"),
                    "object": o_vlm,
                    "reason": "关系端点未能与检测对象对齐",
                }
            )

    # 4) 几何关系：只对有 bbox 的对象计算，只记录达标的 near / overlap
    geometry_relations = []
    o_with_box = [o for o in objects if o["bbox"] is not None]
    for i in range(len(o_with_box)):
        for j in range(i + 1, len(o_with_box)):
            a, b = o_with_box[i], o_with_box[j]
            an = None
            bn = None
            # 归一化中心距离
            a_det = det_by_id.get(a["detection_id"], {})
            b_det = det_by_id.get(b["detection_id"], {})
            a_n = a_det.get("image_position_normalized")
            b_n = b_det.get("image_position_normalized")
            if a_n and b_n:
                dist = math.sqrt((a_n[0] - b_n[0]) ** 2 + (a_n[1] - b_n[1]) ** 2)
            else:
                dist = None
            iou = compute_iou(a["bbox"], b["bbox"])

            ax, ay = a["center"]
            bx, by = b["center"]
            horizontal = "left_of" if ax < bx else "right_of"
            vertical = "above" if ay < by else "below"

            metrics = {
                "center_distance_normalized": round(dist, 4) if dist is not None else None,
                "iou": round(iou, 4),
            }

            if dist is not None and dist < MAX_NEAR_DISTANCE:
                geometry_relations.append(
                    {
                        "subject": a["id"],
                        "predicate": "near_candidate",
                        "object": b["id"],
                        "source": "geometry",
                        "confidence": None,
                        "status": "candidate",
                        "horizontal_relation": horizontal,
                        "vertical_relation": vertical,
                        "evidence": f"归一化中心距离 {dist:.3f} < {MAX_NEAR_DISTANCE}",
                        "metrics": metrics,
                    }
                )
            if iou > MIN_IOU:
                geometry_relations.append(
                    {
                        "subject": a["id"],
                        "predicate": "bbox_overlap_candidate",
                        "object": b["id"],
                        "source": "geometry",
                        "confidence": None,
                        "status": "candidate",
                        "horizontal_relation": horizontal,
                        "vertical_relation": vertical,
                        "evidence": f"检测框 IoU {iou:.3f} > {MIN_IOU}",
                        "metrics": metrics,
                    }
                )

    accident = vlm_data.get("possible_accident_type", {})
    accident_event = {
        "type": accident.get("type", "无法判断"),
        "confidence": accident.get("confidence"),
        "evidence": accident.get("evidence", ""),
        "source": "vlm",
        "status": "inferred",
    }

    return {
        "scene_id": scene_id,
        "image_info": {"width": image_width, "height": image_height},
        "objects": objects,
        "semantic_relations": semantic_relations,
        "geometry_relations": geometry_relations,
        "unresolved_relations": unresolved_relations,
        "accident_event": accident_event,
        "road_environment": vlm_data.get("road_environment", {}),
    }


# ---------------------------------------------------------------------------
# 质量校验
# ---------------------------------------------------------------------------
def validate_scene(scene):
    """基本数据一致性检查，返回 True，发现问题抛出 AssertionError。"""
    object_ids = [o["id"] for o in scene["objects"]]
    assert len(object_ids) == len(set(object_ids)), "存在重复对象 ID"

    width = scene["image_info"]["width"]
    height = scene["image_info"]["height"]
    for obj in scene["objects"]:
        if obj["object_group"] not in ALLOWED_GROUPS | {"vlm_only"}:
            raise AssertionError(f"非法对象分组: {obj['object_group']}")
        if obj["bbox"] is not None:
            x1, y1, x2, y2 = obj["bbox"]
            assert 0 <= x1 < x2 <= width, f"{obj['id']} bbox 超出图像宽度"
            assert 0 <= y1 < y2 <= height, f"{obj['id']} bbox 超出图像高度"

    id_set = set(object_ids)
    for rel in scene["semantic_relations"]:
        assert rel["subject"] in id_set, f"语义关系主体 {rel['subject']} 不存在"
        assert rel["object"] in id_set, f"语义关系客体 {rel['object']} 不存在"
        assert rel["source"] == "vlm"
    for rel in scene["geometry_relations"]:
        assert rel["subject"] in id_set, f"几何关系主体 {rel['subject']} 不存在"
        assert rel["object"] in id_set, f"几何关系客体 {rel['object']} 不存在"
        assert rel["source"] == "geometry"
        assert rel["status"] == "candidate"

    return True
