"""交互版道路平面场景（Plotly 2D），与静态图共用同一套几何与距离口径。

设计要点：
- 底图由 scene_base_topdown.png 转成 JPEG data URI（约 200 KB），随页面一起走，不额外请求、不塞 13 MB 的像素数组；
- 道路几何、车道线可见段/外推段、路缘、标定区域全部用 shape 绘制，与静态图一致；
- 只有 status=projected 的对象之间才画"可靠距离"，外推对象用空心标记并注明不参与测量；
- 点击任一对象会向父页面 postMessage({type:'scene-object', id:...})，供 index.html 联动高亮表格行；
- 悬浮信息区分"图像事实 / 几何计算 / 语义推测"三类字段。
"""

import base64
import io
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from PIL import Image

ASPHALT = "#3d434b"
LANE = "#f1f3f5"
CURB = "#ffd43b"
DIST = "#e8590c"
SEM = "#b197fc"
INK = "#212529"
MUTED = "#868e96"

VEHICLE = dict(symbol="square", size=13, line=dict(width=1.4, color="#1c7ed6"))
PERSON = dict(symbol="circle", size=11, line=dict(width=1.4, color="#37b24d"))

CLICK_JS = """
(function(){
  var gd = document.getElementById('{plot_id}');
  if(!gd) return;
  gd.on('plotly_click', function(ev){
    if(!ev || !ev.points || !ev.points.length) return;
    var cd = ev.points[0].customdata;
    if(!cd || !cd[0]) return;
    var msg = {type:'scene-object', id:String(cd[0])};
    try { parent.postMessage(msg, '*'); } catch(e) {}
  });
})();
"""




def _data_uri(path):
    """把 PNG 读成 data URI：Plotly 对相对路径的 layout image 支持不稳定，
    内嵌可同时解决 file:// 直接打开与 Pages 上的加载问题。"""
    raw = Path(path).read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{b64}"

def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _status_zh(status):
    return {
        "projected": "区内投影（可用）",
        "projected_extrapolated": "外推投影（不参与可靠距离）",
        "outside_calibration_region": "超出标定范围（不给坐标）",
        "excluded_non_ground_object": "悬空设施（不参与地面投影）",
        "no_bbox_available": "无检测框",
    }.get(status, status)


def _line_shape(x0, y0, x1, y1, color, width=1.6, dash="solid", opacity=1.0):
    return dict(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                line=dict(color=color, width=width, dash=dash), opacity=opacity)


def make_base_image(surface_png_path, out_png_path, road_polygon,
                    xlim=(-3.9, 10.0), ylim=(-0.45, 3.3),
                    asphalt=(89, 96, 106), hatch="#aeb6c0", hatch_step=24,
                    texture_alpha=0.55):
    """把"路面底色 + 真实纹理 + 遮挡斜线"预先合成一张不透明底图。

    为什么要预先合成：Plotly 的 layout image 画在最底层，而 shape 默认画在 trace 之上，
    直接叠 layer 会出现"填充色盖住纹理"的层级问题；先合成一张图最稳，也保证与静态图同口径。
    """
    from PIL import Image, ImageDraw

    rgba = Image.open(surface_png_path).convert("RGBA")
    w, h = rgba.size

    def to_px(x, y):
        px = (x - xlim[0]) / (xlim[1] - xlim[0]) * w
        py = (ylim[1] - y) / (ylim[1] - ylim[0]) * h
        return px, py

    poly_px = [to_px(p["x"], p["y"]) for p in road_polygon]
    poly_mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(poly_mask).polygon(poly_px, fill=255)

    # 只在道路多边形内铺沥青底色，其余保持白色（与静态图的画布一致）
    canvas = Image.composite(Image.new("RGB", (w, h), asphalt),
                             Image.new("RGB", (w, h), (255, 255, 255)), poly_mask)
    # 纹理按 texture_alpha 压暗后贴上去：原图照片偏亮，全不透明会让交互版比静态图"发白"，
    # 压到 55% 后与静态图的 fade 口径一致，也更容易压住 JPEG 的块状噪点。
    tex = rgba.copy()
    tex.putalpha(rgba.split()[-1].point(lambda a: int(a * texture_alpha)))
    canvas.paste(tex, (0, 0), tex)            # 只把有纹理的部分贴上来

    # 多边形内、但没有纹理的地方 = 路面被遮挡，用斜线标"仅按几何推断"
    alpha = rgba.split()[-1]
    occluded = Image.eval(alpha, lambda a: 255 if a < 30 else 0)
    occluded = Image.composite(occluded, Image.new("L", (w, h), 0), poly_mask)

    # 斜线只在"遮挡且无纹理"的区域出现：先把线画在透明层，再用遮挡掩膜裁掉其余部分
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    rgb = tuple(int(hatch[i:i + 2], 16) for i in (1, 3, 5))
    for offset in range(-h, w + h, hatch_step):
        d.line([(offset, h), (offset + h, 0)], fill=rgb + (110,), width=1)
    line_alpha = Image.composite(overlay.split()[-1], Image.new("L", (w, h), 0), occluded)
    overlay.putalpha(line_alpha)
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")

    canvas.save(out_png_path, optimize=True)
    return out_png_path, occluded


def build_scene_figure(spatial_scene_path, road_geometry_local_path,
                       base_png_name="scene_base_topdown.png",
                       xlim=(-3.9, 10.0), ylim=(-0.45, 3.3),
                       title="交通事故道路平面二维场景重建（交互版）"):
    scene = _load(spatial_scene_path)
    geom = _load(road_geometry_local_path)
    unit = geom.get("unit", "local_unit")
    ground_quad = scene["road_plane"]["ground_points"]

    fig = go.Figure()

    # ---------------- 底图 ----------------
    # 用 image trace 而不是 layout image：layout image 在不同 plotly 版本里
    # 会被 plot 背景覆盖或纵向塌缩，image trace 作为第一个 trace 最稳，也天然压在其它图元下面。
    # 底图用 data URI 走 source 参数，而不是传 z 像素数组：
    # z 会被 plotly 重新编码成 PNG（浏览器 canvas 不做优化），一张 2224×600 的底图能膨胀到 13 MB，
    # 手机上几乎打不开；改成 JPEG data URI 后整页约 200 KB。
    with Image.open(base_png_name) as im:
        bw, bh = im.size
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=82, optimize=True)
    uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    fig.add_trace(go.Image(
        source=uri, x0=xlim[0], y0=ylim[1],
        dx=(xlim[1] - xlim[0]) / bw, dy=-(ylim[1] - ylim[0]) / bh,
        hoverinfo="skip", name="路面俯视底图", opacity=1.0,
    ))

    # ---------------- 道路多边形 ----------------
    poly = geom["road_polygon"]
    fig.add_shape(dict(type="path",
                       path="M " + " L ".join(f"{p['x']},{p['y']}" for p in poly) + " Z",
                       line=dict(color="#adb5bd", width=1.0)))

    # ---------------- 车道线（可见段实线 / 外推段虚线） ----------------
    for lane in geom["lane_lines"]:
        pts = lane["points_local"]
        y_split = lane.get("visible_split_local", {}).get("y_at_visible_top_row")
        fig.add_shape(_line_shape(pts[0]["x"], pts[0]["y"], pts[1]["x"], pts[1]["y"],
                                  LANE, 3.2, "solid", 0.92))
        if y_split is not None and y_split > pts[0]["y"]:
            xs = pts[0]["x"] + (pts[1]["x"] - pts[0]["x"]) * (
                (y_split - pts[0]["y"]) / max(pts[1]["y"] - pts[0]["y"], 1e-6))
            fig.add_shape(_line_shape(xs, y_split, pts[1]["x"], pts[1]["y"],
                                      LANE, 2.4, "dash", 0.75))
        fig.add_annotation(x=pts[0]["x"], y=(pts[0]["y"] + pts[1]["y"]) / 2,
                           text=f"<b>{lane['id']}</b>", showarrow=False,
                           font=dict(size=10, color=INK),
                           bgcolor="rgba(255,255,255,0.82)", borderpad=2)

    # ---------------- 路缘 / 边界 ----------------
    for b in geom.get("road_boundaries", []):
        pts = b.get("points_local", [])
        if len(pts) >= 2:
            fig.add_shape(_line_shape(pts[0]["x"], pts[0]["y"], pts[-1]["x"], pts[-1]["y"],
                                      CURB, 2.4, "dash"))
            fig.add_annotation(x=pts[-1]["x"], y=pts[-1]["y"] - 0.18,
                               text=f"{b['id']} 路缘（近似）", showarrow=False,
                               font=dict(size=9.5, color="#8a6d00"),
                               bgcolor="rgba(255,244,214,0.9)", borderpad=2)

    # ---------------- 标定区域 ----------------
    qx = [p[0] for p in ground_quad] + [ground_quad[0][0]]
    qy = [p[1] for p in ground_quad] + [ground_quad[0][1]]
    fig.add_trace(go.Scatter(x=qx, y=qy, mode="lines", name="标定区域边界",
                             line=dict(color="#f783ac", width=1.4, dash="dot"),
                             hoverinfo="skip", showlegend=True))

    # ---------------- 对象 ----------------
    groups = {
        ("vehicle", True): dict(name="车辆（区内投影）", color="#1c7ed6", fill="#a5d8ff"),
        ("vehicle", False): dict(name="车辆（外推/不可用于距离）", color="#1c7ed6", fill="rgba(0,0,0,0)"),
        ("person", True): dict(name="行人（区内投影）", color="#37b24d", fill="#b2f2bb"),
        ("person", False): dict(name="行人（外推/不可用于距离）", color="#37b24d", fill="rgba(0,0,0,0)"),
    }
    for (kind, inside), style in groups.items():
        sel = []
        for o in scene["objects"]:
            gp = o.get("ground_position", {})
            st = gp.get("status")
            if st not in ("projected", "projected_extrapolated"):
                continue
            if o.get("object_group") == "facility":
                continue
            is_person = kind == "person"
            cat = o.get("category", "")
            if is_person != (cat == "person"):
                continue
            if (st == "projected") != inside:
                continue
            sel.append((o, gp))
        if not sel:
            continue

        xs = [gp["x"] for _, gp in sel]
        ys = [gp["y"] for _, gp in sel]
        customdata = [[o["id"], _status_zh(gp["status"]), o.get("category_zh", ""),
                       gp["x"], gp["y"], unit,
                       o.get("ground_anchor_image"),
                       (o.get("vlm_description") or o.get("description") or "")[:120],
                       (o.get("semantic_role") or "")] for o, gp in sel]
        hover = ("<b>%{customdata[0]} %{customdata[2]}</b><br>" +
                 "状态：%{customdata[1]}<br>" +
                 "道路坐标：X=%{customdata[3]:.2f}，Y=%{customdata[4]:.2f} %{customdata[5]}<br>" +
                 "地面锚点（像素）：%{customdata[6]}<br>" +
                 "<i>锚点规则：bbox 底边中点</i><br>" +
                 "VLM 描述：%{customdata[7]}<extra></extra>")
        marker = dict(PERSON if kind == "person" else VEHICLE)
        marker = dict(marker)
        marker["color"] = style["fill"] if inside else "rgba(255,255,255,0.9)"
        marker["size"] = 14 if kind == "vehicle" else 11
        if not inside:
            marker["line"] = dict(width=1.6, color=style["color"], dash="dot")

        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", name=style["name"],
            marker=marker,
            customdata=customdata, hovertemplate=hover,
        ))

        # 对象名称用注释而不是 trace 文本：可以带白底，压在深色路面和斜线上也能读
        for o, gp in sel:
            dy = 0.26 if kind == "vehicle" else 0.18
            above = gp["y"] < 0.55 or gp["y"] > 2.35
            fig.add_annotation(
                x=gp["x"] + (0 if above else (0.22 if kind == "person" else 0)),
                y=gp["y"] + dy * 0.7 if above else gp["y"] - dy,
                text=f"{o['id']} {o.get('category_zh', '')}" +
                     ("" if gp["status"] == "projected" else "（外推）"),
                showarrow=False, font=dict(size=9.5, color=INK),
                bgcolor="rgba(255,255,255,0.86)", borderpad=2,
                xanchor="center" if above else ("left" if kind == "person" else "center"),
            )

    # ---------------- 可靠距离（双方均在标定区内） ----------------
    projected = [o for o in scene["objects"]
                 if o["ground_position"]["status"] == "projected"
                 and o.get("object_group") != "facility"]
    pairs = []
    for i in range(len(projected)):
        for j in range(i + 1, len(projected)):
            a, b = projected[i], projected[j]
            ga, gb = a["ground_position"], b["ground_position"]
            d = ((ga["x"] - gb["x"]) ** 2 + (ga["y"] - gb["y"]) ** 2) ** 0.5
            pairs.append((a["id"], b["id"], d, ga, gb))
    pairs.sort(key=lambda t: t[2])

    for k, (ida, idb, d, ga, gb) in enumerate(pairs):
        fig.add_shape(_line_shape(ga["x"], ga["y"], gb["x"], gb["y"],
                                  DIST, 6.5 if k == 0 else 5.0, "solid", 0.22))
        fig.add_shape(_line_shape(ga["x"], ga["y"], gb["x"], gb["y"],
                                  DIST, 2.2 if k == 0 else 1.6, "solid", 0.95))
        fig.add_annotation(x=(ga["x"] + gb["x"]) / 2, y=(ga["y"] + gb["y"]) / 2 + 0.07,
                           text=f"{ida}—{idb} {d:.2f}", showarrow=False,
                           font=dict(size=9.5, color="#a03a00" if k else "#c2410c"),
                           bgcolor="rgba(255,255,255,0.9)", bordercolor=DIST, borderpad=2)

    # ---------------- 语义关系（不做距离） ----------------
    pos_by_id = {o["id"]: o["ground_position"] for o in scene["objects"]}
    for rel in scene.get("semantic_relations", []):
        s, t = rel.get("subject"), rel.get("object")
        ps, pt = pos_by_id.get(s, {}), pos_by_id.get(t, {})
        if ps.get("x") is None or pt.get("x") is None:
            continue
        if ps.get("status") not in ("projected", "projected_extrapolated"):
            continue
        if pt.get("status") not in ("projected", "projected_extrapolated"):
            continue
        fig.add_trace(go.Scatter(
            x=[ps["x"], pt["x"]], y=[ps["y"], pt["y"]], mode="lines",
            name="语义关系（VLM 观察，未验证碰撞）",
            line=dict(color=SEM, width=2, dash="dash"),
            hovertemplate=(f"<b>{s} —{rel.get('predicate')}→ {t}</b><br>"
                           f"来源：{rel.get('source', 'VLM')}<br>"
                           "该关系来自视觉语言模型的语义判断，未经几何验证<extra></extra>"),
            showlegend=True,
        ))

    # ---------------- 布局 ----------------
    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br>"
                        f"<span style='font-size:12px;color:{MUTED}'>单位 {unit}（未做米制标定）｜"
                        "点击任一对象可在页面表格中定位</span>",
                   x=0.012, xanchor="left", font=dict(size=17, color=INK)),
        height=430,
        margin=dict(l=58, r=18, t=64, b=54),
        xaxis=dict(title=f"X 道路横向（{unit}）", range=list(xlim), showgrid=True,
                   gridcolor="rgba(0,0,0,0.06)", zeroline=False, fixedrange=False),
        yaxis=dict(title=f"Y 道路纵向（{unit}）", range=list(ylim), showgrid=True,
                   gridcolor="rgba(0,0,0,0.06)", zeroline=False,
                   scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="top", y=-0.14, x=0,
                    font=dict(size=10), bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#dee2e6", borderwidth=1),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        hovermode="closest",
        dragmode="pan",
    )
    fig.add_annotation(x=xlim[0] + 0.12, y=ylim[0] + 0.14,
                       text=f"比例尺 1 {unit}（未标定为米）", showarrow=False,
                       font=dict(size=9, color=INK),
                       bgcolor="rgba(255,255,255,0.85)", borderpad=2, xanchor="left")
    return fig


def build_scene_html(spatial_scene_path, road_geometry_local_path, output_path,
                     surface_png_name="road_surface_topdown.png",
                     base_png_name="scene_base_topdown.png",
                     include_plotlyjs="directory", **kwargs):
    out_dir = Path(output_path).parent
    geom = _load(road_geometry_local_path)
    make_base_image(out_dir / surface_png_name, out_dir / base_png_name,
                    geom["road_polygon"], kwargs.get("xlim", (-3.9, 10.0)),
                    kwargs.get("ylim", (-0.45, 3.3)))
    fig = build_scene_figure(spatial_scene_path, road_geometry_local_path,
                             base_png_name=str(out_dir / base_png_name), **kwargs)
    plot_id = "scene-plot"
    fig.write_html(str(output_path), include_plotlyjs=include_plotlyjs,
                   full_html=True, div_id=plot_id,
                   config={"displaylogo": False, "responsive": True,
                           "scrollZoom": True},
                   post_script=CLICK_JS.replace("{plot_id}", plot_id))
    return output_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    d = root / "outputs" / "ACCIDENT_001"
    build_scene_html(d / "scene_model_spatial.json", d / "road_geometry_local.json",
                     d / "scene_topdown.html")
    print("ok")
