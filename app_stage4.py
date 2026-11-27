"""第四阶段 Gradio 界面：道路平面空间建模与二维场景重建（端口 7863）。

四个区域：
① 原始事故图片 + 检测框 + 标定四边形
② 道路标定控制点（图像坐标 <-> 道路坐标）
③ 交通参与者道路平面位置（含投影状态）
④ 道路平面二维场景重建 + 模型信息 + 结果文件下载

运行：
    python app_stage4.py     ->  http://127.0.0.1:7863
"""

import json
from pathlib import Path
from urllib.parse import quote

import gradio as gr
from PIL import Image, ImageDraw, ImageFont

from modules.road_plane_projection import project_scene_to_ground
from modules.spatial_visualization import visualize_spatial_scene
from modules.knowledge_graph import (build_fused_knowledge_graph,
                                     visualize_fused_graph)
from modules.interactive_graph import build_interactive_3d_figure

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
]

GROUP_COLOR = {
    "vehicle_participant": (90, 169, 230),
    "pedestrian_participant": (105, 181, 120),
    "facility": (200, 150, 210),
    "background_object": (170, 170, 170),
}

STATUS_ZH = {
    "projected": "标定区内投影",
    "projected_extrapolated": "外推投影（可靠性下降）",
    "outside_calibration_region": "超出标定范围（不给坐标）",
    "excluded_non_ground_object": "悬空设施（不参与投影）",
    "no_bbox_available": "无检测框（不给坐标）",
}


def _font(size):
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def list_cases():
    if not OUT_DIR.exists():
        return []
    return sorted(d.name for d in OUT_DIR.iterdir()
                  if (d / "scene_model.json").exists())


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_overlay(case_dir, scene, calibration, out_path):
    """在原始图片上画检测框、对象编号、地面锚点与标定四边形。"""
    img = Image.open(case_dir / "original.jpg").convert("RGB")
    draw = ImageDraw.Draw(img)
    f_id, f_small = _font(20), _font(15)

    for obj in scene["objects"]:
        bbox = obj.get("bbox")
        if not bbox:
            continue
        color = GROUP_COLOR.get(obj.get("object_group"), (170, 170, 170))
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{obj['id']} {obj.get('category_zh', '')}"
        tw = draw.textlength(label, font=f_id)
        draw.rectangle([x1, y1 - 24, x1 + tw + 10, y1], fill=color)
        draw.text((x1 + 5, y1 - 23), label, font=f_id, fill=(20, 20, 20))
        anchor = obj.get("ground_anchor_image")
        if anchor:
            ax, ay = anchor
            draw.ellipse([ax - 6, ay - 6, ax + 6, ay + 6],
                         outline=(255, 60, 60), width=3)

    quad = calibration["image_points"]
    draw.line([tuple(p) for p in quad] + [tuple(quad[0])],
              fill=(255, 200, 0), width=4)
    for i, (x, y) in enumerate(quad):
        draw.ellipse([x - 7, y - 7, x + 7, y + 7], fill=(255, 200, 0))
        draw.text((x + 9, y - 20), f"P{i + 1}", font=f_small, fill=(255, 255, 0))

    draw.text((12, 10),
              f"检测框：蓝=车辆  绿=行人  紫=设施  灰=背景  |  "
              f"⭐红圈=地面锚点(bbox底边中点)  |  黄框=P1-P4 标定四边形",
              font=f_small, fill=(255, 255, 255),
              stroke_width=2, stroke_fill=(0, 0, 0))

    img.save(out_path)
    return out_path


def run_case(case):
    if not case:
        return ("尚未选择案例。", [], "", [], None, None, "", "", {},
                gr.update(value=[]))

    case_dir = OUT_DIR / case
    scene_path = case_dir / "scene_model.json"
    calib_path = case_dir / "calibration.json"
    if not scene_path.exists() or not calib_path.exists():
        return ("该案例缺少 scene_model.json 或 calibration.json。", [], "", [],
                None, None, "", "", {}, gr.update(value=[]))

    spatial_path = case_dir / "scene_model_spatial.json"
    distances_path = case_dir / "ground_distances.json"
    info_path = case_dir / "spatial_model_info.json"
    topdown_path = case_dir / "scene_topdown.png"
    overlay_path = case_dir / "_stage4_overlay.png"

    scene, calibration = _read_json(scene_path), _read_json(calib_path)

    spatial, summary = project_scene_to_ground(
        scene_path, calib_path, spatial_path,
        distances_path=distances_path, info_path=info_path)
    visualize_spatial_scene(spatial_path, calib_path, topdown_path)
    build_overlay(case_dir, spatial, calibration, overlay_path)

    # 知识图谱（含道路平面坐标引用）
    graph = build_fused_knowledge_graph(spatial)
    graph_json = {
        "nodes": [{"id": n, **d} for n, d in graph.nodes(data=True)],
        "edges": [{"source": u, "target": v, **d}
                  for u, v, d in graph.edges(data=True)],
    }
    (case_dir / "knowledge_graph_spatial.json").write_text(
        json.dumps(graph_json, ensure_ascii=False, indent=2), encoding="utf-8")
    visualize_fused_graph(graph).save(case_dir / "knowledge_graph_spatial.png")
    fig3d = build_interactive_3d_figure(graph)
    fig3d.write_html(case_dir / "interactive_graph_spatial.html",
                     include_plotlyjs=True, full_html=True)

    # ① 概览
    c = summary["counters"]
    unit = spatial["road_plane"]["unit"]
    overview = f"""
### {case} 道路平面建模结果

| 指标 | 数值 |
|---|---|
| 场景对象总数 | {len(spatial['objects'])} |
| 标定区内投影（projected） | {c['projected']} |
| 外推投影（projected_extrapolated） | {c['projected_extrapolated']} |
| 超出标定范围（不给坐标） | {c['outside_calibration_region']} |
| 悬空设施排除 | {c['excluded_non_ground_object']} |
| 投影成功率（按有效地面目标计） | **{summary['success_rate_percent']}%** |
| 道路平面距离对数 | {summary['pair_count']} |
| 坐标单位 | `{unit}` |

> 自洽性误差 `{summary['reprojection']['reprojection_error_max']}`：因为四个控制点同时用于求解与检验，
> 该值只能说明矩阵求解正确，**不能**作为真实定位精度。单位是 `local_unit`，不是米。
"""

    # ② 控制点表
    ground_pts = calibration["ground_points"]
    order = calibration.get("point_order", ["P1", "P2", "P3", "P4"])
    ctrl_rows = []
    for i, (ip, gp) in enumerate(zip(calibration["image_points"], ground_pts)):
        ctrl_rows.append([f"P{i + 1}", order[i] if i < len(order) else "",
                          f"({ip[0]:.0f}, {ip[1]:.0f})",
                          f"({gp[0]:.1f}, {gp[1]:.1f})"])
    ctrl_note = (
        f"**坐标定义**：原点=近端左侧控制点；X=道路横向（向右为正）；Y=道路纵向（向远端为正）；"
        f"单位=`{unit}`。\n\n"
        f"**假设**：标定矩形 = {calibration['assumptions']['road_width_local_unit']:.1f} × "
        f"{calibration['assumptions']['visible_depth_local_unit']:.1f}（"
        f"非实测道路尺寸）。\n\n"
        f"**标定状态**：`{calibration.get('calibration_status')}`。"
    )

    # ③ 对象道路坐标表
    rows = []
    for obj in spatial["objects"]:
        gp = obj.get("ground_position", {})
        anchor = obj.get("ground_anchor_image")
        x = gp.get("x")
        rows.append([
            obj["id"],
            obj.get("category_zh") or obj.get("category"),
            obj.get("object_group", ""),
            obj.get("semantic_match_status", ""),
            f"({anchor[0]:.1f}, {anchor[1]:.1f})" if anchor else "—",
            f"{x:.3f}" if x is not None else "—",
            f"{gp['y']:.3f}" if gp.get("y") is not None else "—",
            STATUS_ZH.get(gp.get("status"), gp.get("status", "")),
        ])

    # ④ 距离信息
    dist = _read_json(distances_path) if distances_path.exists() else {}
    pair_lines = []
    for rel in spatial.get("semantic_relations", []):
        gd = rel.get("ground_distance", {})
        if gd.get("value") is not None:
            pair_lines.append(
                f"- `{rel['subject']} —{rel.get('predicate', '')}→ {rel['object']}`："
                f"**{gd['value']:.2f} {unit}**（语义关系对的道路平面距离）")
    nearest = dist.get("nearest_pair")
    if nearest:
        pair_lines.append(
            f"- 全场最近一对：`{nearest['a']} — {nearest['b']}`："
            f"**{nearest['distance']:.2f} {unit}**")
    dist_md = "\n".join(pair_lines) if pair_lines else "- 暂无可计算的道路平面距离。"

    info_md = (
        f"#### 单应性矩阵 H（3×3）\n```\n"
        + "\n".join("  ".join(f"{v: .6f}" for v in row)
                    for row in spatial["road_plane"]["homography_matrix"])
        + "\n```\n\n"
        f"#### 模型说明\n"
        f"- 变换方法：平面单应性（4 控制点 DLT），`cv2.findHomography`\n"
        f"- 地面锚点规则：车辆/行人取 bbox 底边中点（近似轮胎/脚部接触点）\n"
        f"- 悬空设施（红绿灯）不参与地面投影\n"
        f"- 投影状态：`projected` / `projected_extrapolated` / `outside_calibration_region`\n\n"
        f"#### 局限\n"
        + "\n".join(f"- {s}" for s in calibration.get("limitations", []))
    )

    files = [str(p) for p in (spatial_path, distances_path, info_path,
                              topdown_path, case_dir / "knowledge_graph_spatial.json",
                              case_dir / "knowledge_graph_spatial.png",
                              case_dir / "interactive_graph_spatial.html") if p.exists()]

    # ⑤ 3D 交互图谱（含 L4 道路平面坐标引用边）
    html_path = case_dir / "interactive_graph_spatial.html"
    if html_path.exists():
        src = "/gradio_api/file=" + quote(str(html_path).replace("\\", "/"))
        graph_html = (f'<iframe src="{src}" width="100%" height="720" '
                      f'style="border:1px solid #dee2e6;border-radius:6px;"></iframe>')
    else:
        graph_html = "<p>尚未生成 3D 交互图谱。</p>"

    return (overview, ctrl_rows, ctrl_note, rows, str(topdown_path),
            str(overlay_path), dist_md, info_md, spatial, files, graph_html)


def load_case_ui(case):
    res = run_case(case)
    return (res[0], gr.update(value=res[1]), res[2], gr.update(value=res[3]),
            res[4], res[5], res[6], res[7], res[8], gr.update(value=res[9]),
            res[10])


def build_app():
    cases = list_cases()
    default = "ACCIDENT_001" if "ACCIDENT_001" in cases else (cases[0] if cases else None)

    with gr.Blocks(title="交通事故道路平面空间建模（第四阶段）") as demo:
        gr.Markdown(
            "# 交通事故道路平面空间建模与二维场景重建（第四阶段）\n"
            "由第三阶段融合场景（VLM 语义 + YOLO 检测 + 知识图谱）出发，"
            "通过四点单应性变换把交通参与者投影到道路局部平面坐标系，"
            "并重建道路平面二维场景。**坐标单位 `local_unit`，非米制实测。**"
        )

        with gr.Row():
            case_dd = gr.Dropdown(choices=cases, value=default, label="事故案例")
            run_btn = gr.Button("重新执行投影与重建", variant="primary")
            reload_btn = gr.Button("刷新")

        gr.Markdown("## ① 原始事故图片与检测结果")
        with gr.Row():
            overlay_img = gr.Image(label="检测框 + 地面锚点 + 标定四边形",
                                   height=520, type="filepath")
            overview_md = gr.Markdown()

        gr.Markdown("## ② 道路标定控制点")
        with gr.Row():
            ctrl_table = gr.Dataframe(
                headers=["控制点", "角色", "图像坐标 (u, v)", "道路坐标 (X, Y)"],
                datatype="str", interactive=False, wrap=True, label="四点标定")
            ctrl_note = gr.Markdown()

        gr.Markdown("## ③ 交通参与者道路平面位置")
        obj_table = gr.Dataframe(
            headers=["编号", "类别", "语义组", "匹配状态", "图像锚点 (u, v)",
                     "道路 X", "道路 Y", "投影状态"],
            datatype="str", interactive=False, wrap=True, label="对象道路坐标表")

        gr.Markdown("## ④ 道路平面二维场景重建")
        topdown_img = gr.Image(label="道路平面俯视重建图", height=620,
                               type="filepath")
        dist_md = gr.Markdown()
        with gr.Accordion("空间模型信息（矩阵 / 规则 / 局限）", open=False):
            info_md = gr.Markdown()
        with gr.Accordion("scene_model_spatial.json", open=False):
            spatial_json = gr.JSON()
        files_out = gr.File(label="下载第四阶段结果文件", file_count="multiple")

        gr.Markdown("## ⑤ 含道路坐标的知识图谱（3D 交互）\n"
                    "新增 `道路局部坐标系` 节点与 `L4 道路平面坐标引用` 边："
                    "已成功投影的对象都指向同一个局部坐标系，坐标写在边属性里。")
        graph_html = gr.HTML()

        inputs = [case_dd]
        outputs = [overview_md, ctrl_table, ctrl_note, obj_table, topdown_img,
                   overlay_img, dist_md, info_md, spatial_json, files_out,
                   graph_html]

        run_btn.click(load_case_ui, inputs=inputs, outputs=outputs)
        reload_btn.click(load_case_ui, inputs=inputs, outputs=outputs)
        case_dd.change(load_case_ui, inputs=inputs, outputs=outputs)
        demo.load(load_case_ui, inputs=inputs, outputs=outputs)

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_port=7863, allowed_paths=[str(OUT_DIR)], inbrowser=False,
               show_error=True)
