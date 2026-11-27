"""第三阶段：VLM × YOLO 融合的交通事故语义场景建模 Gradio 应用。

运行方式：
    python app_stage3.py
浏览器打开：
    http://127.0.0.1:7862

端口 7862，避免与第一阶段 7860、第二阶段 7861 冲突。

两个模式：
1. 加载案例 ACCIDENT_001：读本地已生成的 VLM/YOLO/人工匹配数据，秒级展示融合结果。
2. 上传新图片：跑 YOLO 检测 + VLM 语义分析 + 融合（空匹配表，新图需人工对照后再对齐）。
"""

import json
from pathlib import Path

import gradio as gr
from PIL import Image

from modules.scene_fusion import build_fused_scene, load_json, validate_scene
from modules.knowledge_graph import build_fused_knowledge_graph, visualize_fused_graph
from modules.interactive_graph import build_interactive_3d_figure

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs" / "ACCIDENT_001"


def _iframe_html(html_path: Path, height=700) -> str:
    """用 iframe 嵌入本地 Plotly HTML（Gradio 会过滤内嵌 script，iframe 不受影响）。"""
    src = "/gradio_api/file=" + str(html_path).replace("\\", "/")
    return (
        f'<iframe src="{src}" style="width:100%;height:{height}px;'
        'border:1px solid #dee2e6;border-radius:8px;background:#fff;" '
        'loading="lazy"></iframe>'
    )


def load_case():
    """加载 ACCIDENT_001 已有数据，跑融合并返回展示结果。"""
    vlm_data = load_json(DATA_DIR / "vlm_result.json")
    yolo_data = load_json(DATA_DIR / "yolo_detections.json")
    mapping = load_json(DATA_DIR / "object_mapping.json")

    scene = build_fused_scene(
        vlm_data=vlm_data,
        yolo_data=yolo_data,
        object_mapping=mapping,
        scene_id="ACCIDENT_001",
    )
    validate_scene(scene)
    graph = build_fused_knowledge_graph(scene)
    graph_image = visualize_fused_graph(graph)
    fig3d = build_interactive_3d_figure(graph)
    # 写出独立 HTML（include_plotlyjs=True 内嵌引擎，离线可用），再用 iframe 嵌入界面
    graph_html_path = OUT_DIR / "interactive_graph.html"
    fig3d.write_html(graph_html_path, include_plotlyjs=True, full_html=True)

    detection_img = Image.open(OUT_DIR / "detection_visualization.png")
    scene_json = json.dumps(scene, ensure_ascii=False, indent=2)

    matched = [o for o in scene["objects"]
               if o["semantic_match_status"] == "manually_confirmed"]
    status = (
        f"融合成功：统一对象 {len(scene['objects'])} 个，"
        f"人工匹配 {len(matched)} 个；"
        f"L3 语义关系 {len(scene['semantic_relations'])} 条，"
        f"L2 几何候选 {len(scene['geometry_relations'])} 条；"
        f"事故事件：{scene['accident_event']['type']}"
        f"（置信度 {scene['accident_event']['confidence']}）。"
    )
    return (detection_img, graph_image, _iframe_html(graph_html_path),
            str(graph_html_path), scene_json, status)


def analyze_new(image):
    """上传新图片：跑 YOLO + VLM + 融合（空匹配表）。"""
    if image is None:
        return None, None, "", None, "", "请先上传图片。"
    try:
        from prepare_stage3_data import call_vlm, run_yolo_detections, draw_detections

        vlm_data = call_vlm(image)
        detections = run_yolo_detections(image)
        yolo_data = {
            "image_info": {"width": image.size[0], "height": image.size[1]},
            "detections": detections,
        }
        scene = build_fused_scene(
            vlm_data=vlm_data,
            yolo_data=yolo_data,
            object_mapping={},  # 新图需人工对照带编号图后再填匹配表
            scene_id="ACCIDENT_NEW",
        )
        validate_scene(scene)
        graph = build_fused_knowledge_graph(scene)
        graph_image = visualize_fused_graph(graph)
        fig3d = build_interactive_3d_figure(graph)
        graph_html_path = OUT_DIR / "interactive_graph_new.html"
        fig3d.write_html(graph_html_path, include_plotlyjs=True, full_html=True)
        detection_img = draw_detections(image, detections)
        scene_json = json.dumps(scene, ensure_ascii=False, indent=2)
        status = (
            f"完成：检测 {len(detections)} 个目标，统一对象 {len(scene['objects'])} 个；"
            f"新图未加载人工匹配表，VLM 与 YOLO 对象暂未对齐，"
            f"请对照带编号检测图后在 data/object_mapping.json 中补映射。"
        )
        html_note = _iframe_html(graph_html_path)
        return (detection_img, graph_image, html_note,
                str(graph_html_path), scene_json, status)
    except Exception as exc:
        return None, None, "", None, "", f"分析失败：{type(exc).__name__} —— {exc}"


with gr.Blocks(title="VLM × YOLO 融合事故场景建模（第三阶段）") as demo:
    gr.Markdown(
        """
        # 交通事故语义场景建模（第三阶段：VLM × YOLO 融合）

        把第一阶段 VLM 语义结果与第二阶段 YOLO 几何结果对齐融合：
        - **三套 ID**：V=VLM 对象、D=YOLO 检测对象、O=统一场景对象
        - **关系分三层**：L1 视觉事实（对象节点）、L2 几何候选（near/overlap）、L3 语义推断（VLM 观察）
        - **不覆盖原则**：YOLO 坐标与 VLM 描述并存，未匹配对象保留
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 模式一：加载已标注案例")
            load_btn = gr.Button("加载案例 ACCIDENT_001", variant="secondary")
            gr.Markdown("### 模式二：上传新图片")
            new_image = gr.Image(type="pil", sources=["upload"], height=280,
                                 label="上传交通事故图片")
            new_btn = gr.Button("开始融合分析", variant="primary")
            status_text = gr.Textbox(label="运行状态", interactive=False, lines=3)
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("① 带编号检测图"):
                    detection_out = gr.Image(label="YOLO 检测（D 编号）")
                with gr.Tab("② 融合知识图谱（2D）"):
                    graph_out = gr.Image(label="MultiDiGraph 融合图谱")
                with gr.Tab("③ 3D 交互图谱"):
                    plot_out = gr.HTML()
                    graph_file = gr.File(label="下载 3D 交互图谱（HTML，浏览器打开可旋转/缩放）")
                with gr.Tab("④ 融合场景 JSON"):
                    json_out = gr.Textbox(label="scene_model.json", lines=24,
                                          max_lines=60, buttons=["copy"])

    load_btn.click(
        fn=load_case,
        inputs=None,
        outputs=[detection_out, graph_out, plot_out, graph_file, json_out, status_text],
    )
    new_btn.click(
        fn=analyze_new,
        inputs=new_image,
        outputs=[detection_out, graph_out, plot_out, graph_file, json_out, status_text],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7862,
        allowed_paths=[str(OUT_DIR)],  # 允许 iframe 加载 outputs 下的 HTML
    )
