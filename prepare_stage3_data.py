"""第三阶段数据准备：生成 VLM 语义结果与带 D 编号的 YOLO 检测结果。

运行：
    python prepare_stage3_data.py

产出：
    data/vlm_result.json            第一阶段 VLM 语义分析结果
    data/yolo_detections.json       第二阶段 YOLO 检测结果（D1/D2... 编号）
    outputs/ACCIDENT_001/detection_visualization.png  带编号检测图（供人工对照）
"""

import base64
import io
import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs" / "ACCIDENT_001"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_PATH = BASE_DIR / "test.jpg"
MODEL_PATH = BASE_DIR / "yolo26n.pt"

# 参与者（交通参与者）与设施（道路交通设施）分类
PARTICIPANT_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "person"}
FACILITY_CLASSES = {"traffic light", "stop sign", "parking meter"}

CATEGORY_ZH = {
    "car": "小汽车", "truck": "卡车", "bus": "公交车",
    "motorcycle": "摩托车", "bicycle": "自行车", "person": "行人",
    "traffic light": "交通信号灯", "stop sign": "停车标志",
}

VLM_PROMPT = """
你是一名谨慎的交通场景图像分析助手。请分析这张交通场景图片。
只描述能直接观察到的内容，区分"可见事实"与"初步推测"；单张静态图片无法确定事故过程、真实速度、责任。
请只输出一个合法 JSON 对象，不要使用 Markdown 代码块，结构如下：
{
  "scene_description": "场景整体描述",
  "participants": [
    {
      "id": "V1",
      "category": "车辆/行人/非机动车",
      "visible_features": "颜色、车型、姿态等可见特征",
      "image_position": "画面左/中/右、前/后景等相对位置"
    }
  ],
  "spatial_relations": [
    {
      "subject": "V1",
      "relation": "接触/接近/位于旁边等",
      "object": "V2",
      "visual_evidence": "支持该关系的可见证据"
    }
  ],
  "possible_accident_type": {
    "type": "事故类型或无法判断",
    "confidence": "高/中/低",
    "evidence": "可见依据"
  },
  "road_environment": {
    "road_type": "道路类型",
    "visible_facilities": ["可见设施"]
  }
}
""".strip()


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def call_vlm(image: Image.Image) -> dict:
    """调用第一阶段 VLM，返回解析后的 JSON。"""
    load_dotenv(BASE_DIR / ".env", override=True)
    api_key = os.getenv("DASHSCOPE_API_KEY").strip()
    base_url = os.getenv("DASHSCOPE_BASE_URL").strip()
    model_name = os.getenv("VLM_MODEL", "qwen3.7-plus").strip()

    # trust_env=False：不读取系统/Clash 代理，直连阿里云 MaaS，避免连接重置
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.Client(trust_env=False, timeout=120.0),
    )
    data_url = image_to_data_url(image)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": VLM_PROMPT},
                ],
            }
        ],
        temperature=0.2,
    )
    text = response.choices[0].message.content.strip()
    # 容错：去掉可能出现的 markdown 代码块
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def run_yolo_detections(image: Image.Image) -> list:
    """跑 YOLO，返回带 D 编号的检测列表（保留参与者与设施）。"""
    model = YOLO(str(MODEL_PATH))
    results = model.predict(source=image, conf=0.25, imgsz=640, verbose=False)
    result = results[0]
    img_w, img_h = image.size

    detections = []
    for idx, box in enumerate(result.boxes, start=1):
        category = result.names[int(box.cls[0])]
        if category not in PARTICIPANT_CLASSES and category not in FACILITY_CLASSES:
            continue
        x_min, y_min, x_max, y_max = [float(v) for v in box.xyxy[0]]
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        if category in PARTICIPANT_CLASSES:
            object_group = "participant"
        else:
            object_group = "facility"
        detections.append(
            {
                "id": f"D{idx}",
                "category": category,
                "category_zh": CATEGORY_ZH.get(category, category),
                "object_group": object_group,
                "confidence": round(float(box.conf[0]), 3),
                "bbox": [round(x_min, 1), round(y_min, 1),
                         round(x_max, 1), round(y_max, 1)],
                "center": [round(cx, 1), round(cy, 1)],
                "image_position_normalized": [round(cx / img_w, 4),
                                              round(cy / img_h, 4)],
            }
        )
    return detections


def draw_detections(image: Image.Image, detections: list) -> Image.Image:
    """在图上画 D 编号、类别、置信度，参与者与设施用不同颜色。"""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    try:
        font = ImageFont.truetype("msyh.ttc", 18)
    except OSError:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = (0, 200, 0, 255) if det["object_group"] == "participant" else (
            200, 0, 200, 255)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{det['id']} {det['category_zh']} {det['confidence']:.2f}"
        # 标签背景
        bbox = draw.textbbox((x1, max(0, y1 - 26)), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, max(0, y1 - 26)), label, fill=(255, 255, 255, 255), font=font)
    return canvas


def main():
    image = Image.open(IMAGE_PATH).convert("RGB")

    print("=" * 60)
    print("第一步：调用 VLM 分析图片...")
    vlm_data = call_vlm(image)
    with open(DATA_DIR / "vlm_result.json", "w", encoding="utf-8") as f:
        json.dump(vlm_data, f, ensure_ascii=False, indent=2)
    print("VLM 对象数:", len(vlm_data.get("participants", [])))

    print("=" * 60)
    print("第二步：YOLO 检测（D 编号）...")
    detections = run_yolo_detections(image)
    yolo_payload = {
        "image_info": {"width": image.size[0], "height": image.size[1]},
        "detections": detections,
    }
    with open(DATA_DIR / "yolo_detections.json", "w", encoding="utf-8") as f:
        json.dump(yolo_payload, f, ensure_ascii=False, indent=2)
    for det in detections:
        print(f"  {det['id']} [{det['object_group']:<11}] "
              f"{det['category']:<12} conf {det['confidence']:.2f} "
              f"center {det['center']}")

    annotated = draw_detections(image, detections)
    annotated.save(OUT_DIR / "detection_visualization.png")
    print("=" * 60)
    print("带编号检测图已保存: outputs/ACCIDENT_001/detection_visualization.png")


if __name__ == "__main__":
    main()
