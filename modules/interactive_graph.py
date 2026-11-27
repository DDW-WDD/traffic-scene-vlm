"""第三阶段：2D 融合图谱的三维化（Plotly）。

思路：
- 用 networkx spring_layout 算 2D 坐标（与 2D 版完全一致）
- 按 node_type 在 z 轴分层：Scene 最高 -> AccidentEvent 次高 -> 参与者中层 -> 设施低层
- 节点位置 = 2D(x, y) + 类型 z 高度
- 旋转视角时能看到 2D 图谱"立起来"的立体感
"""

import networkx as nx
import plotly.graph_objects as go


# 与 2D 版一致的浅色配色（深色文字压在浅底上更清晰）
_NODE_COLOR = {
    "Scene": "#f4a460",          # 沙棕
    "Vehicle": "#add8e6",        # 浅蓝
    "Pedestrian": "#90ee90",     # 浅绿
    "Facility": "#dda0dd",       # 梅红
    "VLMROnly": "#d3d3d3",
    "AccidentEvent": "#fa8072",  # 鲑鱼红
    "CoordinateSystem": "#74c0fc",
    "RoadArea": "#dee2e6",
    "LaneLine": "#ced4da",
    "RoadBoundary": "#ffe066",
}

# z 轴分层高度（压缩后接近 2D 平面，避免纵深过大）
_NODE_Z = {
    "Scene": 1.30,
    "AccidentEvent": 0.95,
    "Pedestrian": 0.60,
    "Vehicle": 0.30,
    "Facility": 0.10,
    "VLMROnly": 0.0,
    "CoordinateSystem": 0.95,
    "RoadArea": 0.45,
    "LaneLine": 0.45,
    "RoadBoundary": 0.45,
}

_NODE_LABEL_ZH = {
    "Scene": "场景",
    "Vehicle": "车辆",
    "Pedestrian": "行人",
    "Facility": "道路设施",
    "VLMROnly": "仅 VLM 对象",
    "AccidentEvent": "事故事件",
    "CoordinateSystem": "道路坐标系",
    "RoadArea": "路面区域",
    "LaneLine": "车道线",
    "RoadBoundary": "道路边界",
}

_EDGE_STYLE = {
    "semantic": {"color": "#e03131", "width": 5, "dash": "solid",
                 "name": "L3 语义关系"},
    "geometry": {"color": "#f4a261", "width": 1.5, "dash": "dot",
                 "name": "L2 几何候选"},
    "accident": {"color": "#ff8787", "width": 1.2, "dash": "dash",
                 "name": "事故关联"},
    "belongs": {"color": "#adb5bd", "width": 0.8, "dash": "dash",
                "name": "属于场景"},
    "position": {"color": "#4dabf7", "width": 1.4, "dash": "dot",
                 "name": "L4 道路坐标"},
}


_CLICK_JS = """
(function(){
  var gd = document.getElementById('{plot_id}');
  if(!gd) return;
  gd.on('plotly_click', function(ev){
    if(!ev || !ev.points || !ev.points.length) return;
    var cd = ev.points[0].customdata;
    if(!cd || !cd[0]) return;
    try { parent.postMessage({type:'graph-node', id:String(cd[0])}, '*'); } catch(e) {}
  });
})();
"""

_VIEWPORT = ('<meta name="viewport" content="width=device-width, initial-scale=1.0, '
             'viewport-fit=cover">')


def write_graph_html(fig, path, div_id="graph-plot"):
    """写 3D 图谱 HTML：共用一份 plotly.min.js + 注入 viewport + 点击联动脚本。"""
    import re
    from pathlib import Path

    path = Path(path)
    fig.write_html(str(path), include_plotlyjs="directory", full_html=True,
                   div_id=div_id,
                   config={"displaylogo": False, "responsive": True},
                   post_script=_CLICK_JS.replace("{plot_id}", div_id))
    html = path.read_text(encoding="utf-8")
    if 'name="viewport"' not in html:
        html = html.replace("<head>", f"<head>\n        {_VIEWPORT}", 1)
    # 让画布跟随 iframe 高度，避免页面里出现上下裁切
    inject = ('<style>html,body{height:100%;margin:0;}'
              '[id^="graph-plot"]{height:100% !important;}</style></head>')
    html = html.replace("</head>", inject, 1)
    html = html.replace("<body>", '<body style="margin:0">', 1)
    path.write_text(html, encoding="utf-8")
    return path


def _short_label(n, data):
    """3D 节点标签要短：VLM 的长描述会把相邻节点盖住，屏幕上只写"编号 + 类别"。

    完整描述（颜色、衣着、损伤）仍保留在 hovertext 里，点开详情也能看到，信息不丢。
    """
    import re

    label = str(data.get("label", n))
    cat = data.get("category_zh")
    if re.fullmatch(r"O\d+", str(n)) and cat:
        return f"{n} {cat}"
    if str(n) in ("L1", "L2"):
        return f"{n} 车道线"
    if str(n) == "RB_R":
        return "RB_R 路缘"
    if str(n) == "ROAD_AREA":
        return "路面区域"
    return label if len(label) <= 12 else label[:11] + "…"


def build_interactive_3d_figure(graph: nx.MultiDiGraph) -> go.Figure:
    """从融合知识图谱构建 2D 布局的 3D 版本。"""
    # 2D spring_layout（与 2D 版相同的 seed 和 k）
    pos2d = nx.spring_layout(graph, seed=42, k=2.4, dim=2)

    # 节点坐标 = 2D(x, y) + 类型 z
    coords = {}
    for n, data in graph.nodes(data=True):
        ntype = data.get("node_type", "VLMROnly")
        x, y = pos2d[n]
        z = _NODE_Z.get(ntype, 0.0)
        coords[n] = (x, y, z)

    fig = go.Figure()

    # ---------- 节点：按类型分 trace ----------
    node_groups = {}
    for n, data in graph.nodes(data=True):
        ntype = data.get("node_type", "VLMROnly")
        node_groups.setdefault(ntype, []).append((n, data))

    for ntype, items in node_groups.items():
        xs, ys, zs, texts, names, sizes = [], [], [], [], [], []
        for n, data in items:
            x, y, z = coords[n]
            xs.append(x); ys.append(y); zs.append(z)
            label = data.get("label", n)
            # 没有 VLM 描述时会退化成类别名（如“小汽车”），此时补上编号避免同名节点混淆
            if data.get("category_zh") and label == data["category_zh"]:
                label = f"{n} {label}"
            names.append(_short_label(n, data))   # 节点上只写短名，长描述留给悬停/详情
            conf = data.get("confidence")
            conf_str = f"<br>YOLO 置信度: {conf}" if conf is not None else ""
            match_status = data.get("match_status", "")
            match_str = (f"<br>对齐状态: {match_status}"
                         if match_status else "")
            extra = ""
            if data.get("position_text"):
                extra += f"<br>道路坐标: {data['position_text']}"
            if data.get("projection_status"):
                extra += f"<br>投影状态: {data['projection_status']}"
            if data.get("line_type"):
                extra += f"<br>线型: {data['line_type']}"
            if data.get("quality"):
                extra += f"<br>标注质量: {data['quality']}"
            texts.append(
                f"<b>{n}</b><br>类型: {ntype}<br>名称: {label}"
                f"{conf_str}{match_str}{extra}"
            )
            if ntype == "AccidentEvent":
                sizes.append(54)
            elif ntype == "Scene":
                sizes.append(46)
            elif match_status == "manually_confirmed":
                sizes.append(38)
            else:
                sizes.append(32)
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers+text",
            name=_NODE_LABEL_ZH.get(ntype, ntype),
            text=names,
            textposition="middle center",
            textfont=dict(size=11, color="#212529"),
            marker=dict(
                size=sizes,
                color=_NODE_COLOR.get(ntype, "#cccccc"),
                line=dict(width=1.8, color="#495057"),
                opacity=1.0,
            ),
            hovertext=texts,
            hoverinfo="text",
            customdata=[[n] for n, _ in items],
        ))

    # ---------- 边：按类型分 trace（带 relation 悬停提示 + 中线标签） ----------
    edge_groups = {k: {"x": [], "y": [], "z": [], "text": [], "mid": []}
                   for k in _EDGE_STYLE}
    for u, v, data in graph.edges(data=True):
        rel = data["relation"]
        source = data.get("source", "")
        if rel == "belongs_to":
            g = "belongs"
        elif rel == "accident_involves":
            g = "accident"
        elif rel == "position_in_coordinate_system":
            g = "position"
        elif source == "vlm":
            g = "semantic"
        else:
            g = "geometry"
        x1, y1, z1 = coords[u]
        x2, y2, z2 = coords[v]
        tip = f"{u} → {v}<br>关系: {rel}<br>来源: {source or '-'}"
        edge_groups[g]["x"].extend([x1, x2, None])
        edge_groups[g]["y"].extend([y1, y2, None])
        edge_groups[g]["z"].extend([z1, z2, None])
        edge_groups[g]["text"].extend([tip, tip, None])
        # 边标签：放在边的 0.62 处，且只给足够长的边（避免中心区域堆叠）
        if g in ("semantic", "geometry"):
            length2d = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            if g == "semantic" or length2d >= 0.55:
                t = 0.62
                edge_groups[g]["mid"].append((
                    x1 + (x2 - x1) * t,
                    y1 + (y2 - y1) * t,
                    z1 + (z2 - z1) * t,
                    rel, tip,
                ))

    for g, style in _EDGE_STYLE.items():
        d = edge_groups[g]
        if not d["x"]:
            continue
        fig.add_trace(go.Scatter3d(
            x=d["x"], y=d["y"], z=d["z"],
            mode="lines",
            name=style["name"],
            line=dict(color=style["color"], width=style["width"],
                      dash=style["dash"]),
            hovertext=d["text"],
            hoverinfo="text",
        ))
        # 中线文字标签
        if d["mid"]:
            fig.add_trace(go.Scatter3d(
                x=[m[0] for m in d["mid"]],
                y=[m[1] for m in d["mid"]],
                z=[m[2] for m in d["mid"]],
                mode="text",
                text=[m[3] for m in d["mid"]],
                textfont=dict(size=8, color=style["color"]),
                hovertext=[m[4] for m in d["mid"]],
                hoverinfo="text",
                showlegend=False,
            ))

    # ---------- 布局：屏幕平面就是 2D 图谱平面，z 轴作为"纵深" ----------
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    fig.update_layout(
        title=dict(
            text=f"交通事故融合知识图谱（{n_nodes} 节点 / {n_edges} 边）",
            font=dict(size=13),
            x=0.5, xanchor="center", y=0.97, yanchor="top",
        ),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(
                eye=dict(x=0.28, y=-0.5, z=2.2),
                up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0),
            ),
            bgcolor="white",
            aspectmode="manual",
            aspectratio=dict(x=1.85, y=1.35, z=0.85),
        ),
        legend=dict(
            itemsizing="constant",
            font=dict(size=9),
            x=0.01, y=1.0, xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="rgba(0,0,0,0.12)",
            borderwidth=1,
            tracegroupgap=2,
        ),
        margin=dict(l=2, r=2, t=32, b=2),
        height=720,
        showlegend=True,
    )
    return fig
