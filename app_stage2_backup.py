"""第二阶段：交通事故场景建模 Gradio 应用。

运行方式：
    python app_stage2.py
浏览器打开：
    http://127.0.0.1:7861

端口使用 7861，避免与第一阶段 app.py 的 7860 冲突。
"""

import json

import gradio as gr

from scene_model import process_traffic_scene


def run_modeling(image):
    """点击按钮后的处理函数：跑通整条场景建模链路。"""
    if image is None:
        return None, None, None, "", "请先上传交通事故图片。"

    try:
        result = process_traffic_scene(image)
        scene = result["scene"]

        scene_json = json.dumps(scene, ensure_ascii=False, indent=2)
        n_participants = len(scene["participants"])
        n_relations = len(scene["spatial_relations"])
        status = (
            f"建模成功：识别 {n_participants} 个交通参与者，"
            f"计算 {n_relations} 对空间关系，知识图谱已构建。"
        )

        return (
            result["annotated_image"],
            result["spatial_image"],
            result["graph_image"],
            scene_json,
            status,
        )
    except Exception as exc:  # 捕获后回显，避免整个界面崩溃
        error = f"建模失败：{type(exc).__name__} —— {exc}"
        return None, None, None, "", error


with gr.Blocks(title="交通事故场景建模系统") as demo:
    gr.Markdown(
        """
        # 交通事故场景建模系统（第二阶段）

        上传交通事故图片，自动完成：

        1. **YOLO 目标检测**：识别汽车、卡车、公交车、摩托车、自行车、行人
        2. **像素坐标提取**：边界框、中心点、宽高、归一化坐标
        3. **统一场景数据结构**：每个参与者生成 ID 与属性对象
        4. **空间关系计算**：方位关系、IoU 重叠、接近候选
        5. **交通事故知识图谱**：NetworkX 实体节点与语义关系

        > 注意：检测框重叠、目标接近仅为图像平面上的几何事实，
        > 不直接等同于真实碰撞或事故责任。
        """
    )

    with gr.Row():
        image_input = gr.Image(
            type="pil",
            sources=["upload"],
            label="上传交通事故图片",
            height=380,
        )

    run_button = gr.Button("开始场景建模", variant="primary")
    status_text = gr.Textbox(label="运行状态", interactive=False)

    with gr.Tabs():
        with gr.Tab("① 目标检测"):
            annotated_output = gr.Image(label="YOLO 检测框标注图")
        with gr.Tab("② 空间关系"):
            spatial_output = gr.Image(label="图片坐标系空间关系图")
        with gr.Tab("③ 知识图谱"):
            graph_output = gr.Image(label="交通事故知识图谱")
        with gr.Tab("④ 场景数据 JSON"):
            json_output = gr.Textbox(
                label="统一交通场景数据",
                lines=28,
                max_lines=50,
                buttons=["copy"],
            )

    run_button.click(
        fn=run_modeling,
        inputs=image_input,
        outputs=[
            annotated_output,
            spatial_output,
            graph_output,
            json_output,
            status_text,
        ],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)
