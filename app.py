"""Traffic accident image understanding demo using a vision-language model."""


import base64
import io
import os

# 让 Gradio 访问本地地址时绕过网络代理
local_hosts = "localhost,127.0.0.1,::1"

for name in ("NO_PROXY", "no_proxy"):
    existing = os.environ.get(name, "")
    hosts = [
        item.strip()
        for item in (existing + "," + local_hosts).split(",")
        if item.strip()
    ]
    os.environ[name] = ",".join(dict.fromkeys(hosts))

from typing import Optional

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image



load_dotenv(override=True)

API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
).strip()
MODEL_NAME = os.getenv("VLM_MODEL", "qwen3.7-plus").strip()

ANALYSIS_PROMPT = """
你是一名谨慎的交通场景图像分析助手。请分析用户提供的单张交通场景图片。

请遵守以下规则：
1. 只描述图片中能够直接观察到的内容，不要编造被遮挡或看不清的信息。
2. 必须区分“图片中可见的事实”和“基于画面的初步推测”。
3. 单张静态图片通常无法确定事故发生的先后顺序、车辆真实速度、精确距离、碰撞原因和事故责任；无法确定时请明确写“无法仅凭该图片判断”。
4. 位置只使用“画面左侧/中间/右侧、前景/背景”等相对描述，不要把像素位置冒充真实世界坐标。
5. 如果没有明显事故证据，不要强行判断为交通事故；可以把事故类型写为“无法判断”或“未见明确碰撞证据”。
6. 不要判断任何人的法律责任。

请只输出一个合法 JSON 对象，不要使用 Markdown 代码块。使用以下结构和中文内容：
{
  "scene_description": "场景整体描述",
  "road_environment": {
    "road_type": "道路类型；看不清则写无法判断",
    "visible_facilities": ["可见的道路设施、标线、信号灯等"]
  },
  "participants": [
    {
      "id": "P1",
      "category": "车辆/行人/非机动车/其他",
      "visible_features": "可见外观特征",
      "image_position": "画面中的相对位置"
    }
  ],
  "spatial_relations": [
    {
      "subject": "P1",
      "relation": "例如位于前方/旁边/接近/接触；不确定则说明",
      "object": "P2",
      "visual_evidence": "支持该关系的可见依据"
    }
  ],
  "possible_accident_type": {
    "type": "例如追尾/车辆与行人碰撞/侧面碰撞/无法判断",
    "confidence": "高/中/低/无法判断",
    "evidence": "图片中支持这一判断的可见证据"
  },
  "not_observable_from_single_image": [
    "仅凭这张图片无法确定的信息"
  ]
}
""".strip()


def image_to_data_url(image: Image.Image) -> str:
    """Convert uploaded PIL image to a JPEG Base64 data URL."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def analyze_image(image: Optional[Image.Image]) -> str:
    """Send the uploaded image and analysis prompt to the VLM API."""
    if image is None:
        return "请先上传一张交通场景图片。"

    if not API_KEY:
        return (
            "尚未配置 DASHSCOPE_API_KEY。\n\n"
            "请复制项目中的 .env.example 为 .env，并在 .env 中填写你自己的百炼 API Key。"
        )

    try:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        image_data_url = image_to_data_url(image)

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url},
                        },
                        {"type": "text", "text": ANALYSIS_PROMPT},
                    ],
                }
            ],
            temperature=0.2,
        )

        result = response.choices[0].message.content
        if isinstance(result, str) and result.strip():
            return result.strip()
        return "模型已返回响应，但没有得到可显示的文本内容。请检查模型和 API 返回格式。"


    except Exception as exc:

        root = exc

        while True:

            next_error = (

                    getattr(root, "__cause__", None)

                    or getattr(root, "__context__", None)

            )

            if next_error is None or next_error is root:
                break

            root = next_error

        return (

            "调用模型失败。\n\n"

            f"异常类型：{type(exc).__name__}\n"

            f"异常信息：{exc}\n\n"

            f"底层错误类型：{type(root).__name__}\n"

            f"底层错误信息：{root}"

        )


with gr.Blocks(title="交通事故图片 VLM 分析 Demo") as demo:
    gr.Markdown(
        """
        # 交通事故图片智能分析（第一版）

        上传一张交通场景图片，点击“开始分析”，由视觉语言模型生成结构化场景描述。

        **注意：** 当前版本只分析单张静态图片，不推断事故完整过程、车辆真实速度或事故责任。
        """
    )

    image_input = gr.Image(
        type="pil",
        sources=["upload"],
        label="上传交通事故图片",
        height=360,
    )
    analyze_button = gr.Button("开始分析", variant="primary")
    result_output = gr.Textbox(
        label="VLM 分析结果（JSON）",
        lines=24,
        max_lines=40,
        buttons=["copy"],
    )

    analyze_button.click(
        fn=analyze_image,
        inputs=image_input,
        outputs=result_output,
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
