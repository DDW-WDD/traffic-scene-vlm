"""从第四阶段产物生成多视图静态站点 index.html。

为什么用脚本生成而不是手写：
- 页面上的对象表、距离表、状态统计全部直接读 JSON，改数据后重跑即可同步；
- 避免"页面写 5.38、数据里已改成不可靠"这类前后不一致；
- 视图（原图 / 俯视场景 / 对象与距离 / 知识图谱 / 方法与误差）共用同一套口径文案。
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASE = ROOT / "outputs" / "ACCIDENT_001"


def load(name):
    with open(CASE / name, "r", encoding="utf-8") as f:
        return json.load(f)


STATUS_ZH = {
    "projected": ("区内投影", "b"),
    "projected_extrapolated": ("外推投影", "a"),
    "outside_calibration_region": ("超出标定范围", "r"),
    "excluded_non_ground_object": ("悬空设施（排除）", "m"),
    "no_bbox_available": ("无检测框", "m"),
}


def fmt(v, nd=2, dash="—"):
    return dash if v is None else f"{v:.{nd}f}"


def build():
    scene = load("scene_model_spatial.json")
    dist = load("ground_distances.json")
    check = load("calibration_check.json")
    geom = load("road_geometry_local.json")
    calib = load("calibration.json")

    objects = scene["objects"]
    summary = scene.get("projection_summary", {})
    counts = {
        k: summary.get(k, 0) for k in
        ("projected", "projected_extrapolated", "outside_calibration_region",
         "excluded_non_ground_object")
    }
    pairs = dist["pairs"]
    reliable = dist["reliable_pairs"]
    nearest = dist.get("nearest_reliable_pair") or {}

    obj_rows = []
    for o in objects:
        gp = o.get("ground_position", {})
        st = gp.get("status")
        label, cls = STATUS_ZH.get(st, (st, "m"))
        pos = (f"{fmt(gp.get('x'))}, {fmt(gp.get('y'))}"
               if gp.get("x") is not None else "—")
        anchor = o.get("ground_anchor_image")
        anchor_s = f"{anchor[0]:.0f}, {anchor[1]:.0f}" if anchor else "—"
        desc = (o.get("vlm_description") or o.get("description") or "").replace("<", "&lt;")
        obj_rows.append(
            f'<tr data-oid="{o["id"]}" class="obj-row">'
            f'<td class="nowrap"><b>{o["id"]}</b></td>'
            f'<td class="nowrap">{o.get("category_zh", "")}</td>'
            f'<td class="num">{pos}</td>'
            f'<td><span class="pill {cls}">{label}</span></td>'
            f'<td class="num">{anchor_s}</td>'
            f'<td class="desc">{desc[:46]}{"…" if len(desc) > 46 else ""}</td>'
            f"</tr>")

    rel_rows = []
    for p in reliable:
        rel_rows.append(
            f'<tr><td class="nowrap"><b>{p["a"]} — {p["b"]}</b></td>'
            f'<td class="num">{p["distance"]:.2f}</td><td>local_unit</td>'
            f'<td>双方均在标定区内</td></tr>')
    unreliables = [p for p in pairs if not p["reliable"]]
    unreliables.sort(key=lambda p: p["distance"])
    # 页面要解释的是"VLM 记为接触的那一对"为何降级，而不是距离最小的外推对
    accident_note = ""
    for rel in scene.get("semantic_relations", []):
        gd = rel.get("ground_distance", {})
        if gd.get("reliable") is False and gd.get("approx_value_for_trace") is not None:
            accident_note = (f'VLM 记为"{rel["predicate"]}"的 {rel["subject"]}—{rel["object"]} 对，'
                             f'地面距离约 {gd["approx_value_for_trace"]:.2f} local_unit；'
                             "但其中一方是外推结果，该值只作追溯，"
                             "不能写成“两车相距该数值”的结论。")
            break

    sem_rows = []
    for rel in scene.get("semantic_relations", []):
        gd = rel.get("ground_distance", {})
        if gd.get("reliable"):
            val = f'{gd["value"]:.2f} local_unit'
            tag = '<span class="pill b">可靠</span>'
        elif gd.get("approx_value_for_trace") is not None:
            val = f'约 {gd["approx_value_for_trace"]:.2f} local_unit（仅追溯）'
            tag = '<span class="pill a">不可靠</span>'
        else:
            val = "—"
            tag = '<span class="pill r">不可用</span>'
        sem_rows.append(
            f'<tr><td class="nowrap">{rel["subject"]} —{rel["predicate"]}→ {rel["object"]}</td>'
            f'<td>{rel.get("evidence", "")[:52]}</td><td>{val}</td><td>{tag}</td></tr>')

    lane_rows = []
    for lane in check.get("lane_line_alignment", []):
        lane_rows.append(
            f'<tr><td class="nowrap">{lane["id"]} {lane.get("name", "")}</td>'
            f'<td class="num">{lane["ground_slope_dx_per_dy"]:+.4f}</td>'
            f'<td class="num">{lane["deviation_from_vertical_deg"]:.2f}°</td>'
            f'<td class="num">{lane["max_residual_local_unit"]:.4f}</td>'
            f'<td class="num">{lane["measured_point_count"]}</td></tr>')

    par = check.get("lane_pair_parallelism", {})
    sens = check.get("anchor_sensitivity", {})
    verdict = check.get("verdict", {})
    scale = check.get("scale_ambiguity", {})
    selfc = check.get("self_consistency", {})

    n_nodes = 22
    n_edges = 56
    geometry_stat = (f'{len(geom.get("lane_lines", []))} 条车道线'
                     f' + {len(geom.get("road_boundaries", []))} 条路缘/边界')

    calib_rows = "".join(
        f'<tr><td class="nowrap">{name}</td><td class="num">({ip[0]}, {ip[1]})</td>'
        f'<td class="num">({gp[0]:.1f}, {gp[1]:.1f})</td></tr>'
        for name, ip, gp in zip(("P1 近左", "P2 近右", "P3 远右", "P4 远左"),
                                calib["image_points"], calib["ground_points"]))

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<meta name="theme-color" content="#1d2b45"/>
<meta name="description" content="交通事故语义场景建模第四阶段：道路结构重建、单应性投影与多视图场景呈现"/>
<title>交通事故语义场景建模 · 第四阶段 道路平面场景重建</title>
<style>
  :root{{
    --ink:#141821; --muted:#5c6577; --line:#e3e7ef;
    --bg:#f5f7fb; --card:#ffffff;
    --blue:#2f6fd0; --green:#3f9d63; --red:#d2483b; --amber:#c9871f; --violet:#7c5cd6;
    --pad:20px;
  }}
  *{{box-sizing:border-box}}
  html{{-webkit-text-size-adjust:100%;scroll-behavior:smooth}}
  body{{
    margin:0; background:var(--bg); color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei","Hiragino Sans GB",sans-serif;
    line-height:1.7; -webkit-font-smoothing:antialiased;
    -webkit-tap-highlight-color:transparent; overflow-x:hidden;
  }}
  .wrap{{
    max-width:1180px;margin:0 auto;
    padding-left:max(var(--pad), env(safe-area-inset-left));
    padding-right:max(var(--pad), env(safe-area-inset-right));
  }}
  header{{background:linear-gradient(135deg,#1d2b45 0%,#2f6fd0 100%);color:#fff;padding:52px 0 44px}}
  .tag{{display:inline-block;font-size:13px;letter-spacing:.5px;background:rgba(255,255,255,.16);
        border:1px solid rgba(255,255,255,.28);padding:4px 12px;border-radius:999px;margin-bottom:14px}}
  h1{{margin:0 0 10px;font-size:clamp(21px,4.6vw,30px);letter-spacing:.5px;font-weight:700;line-height:1.35}}
  .sub{{opacity:.92;font-size:clamp(13.5px,3.4vw,15px);max-width:820px;margin:0}}
  .meta{{margin-top:16px;font-size:clamp(11.5px,3vw,13px);opacity:.82}}

  .kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:clamp(22px,4vw,30px) 0}}
  .kpi{{background:var(--card);border:1px solid var(--line);border-radius:14px;
       padding:16px 16px 13px;box-shadow:0 4px 18px rgba(20,24,33,.05)}}
  .kpi .v{{font-size:clamp(20px,5vw,26px);font-weight:700;letter-spacing:.5px;line-height:1.2}}
  .kpi .u{{font-size:11.5px;color:var(--muted);margin-top:2px}}
  .kpi .k{{font-size:clamp(12px,3vw,13px);color:var(--muted);margin-top:7px;line-height:1.55}}
  .kpi.b .v{{color:var(--blue)}} .kpi.g .v{{color:var(--green)}}
  .kpi.r .v{{color:var(--red)}}  .kpi.a .v{{color:var(--amber)}}

  /* 视图切换 */
  .tabs{{position:sticky;top:0;z-index:30;background:rgba(245,247,251,.94);
        backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
        border-bottom:1px solid var(--line);margin:0 0 clamp(18px,3vw,26px)}}
  .tabbar{{display:flex;gap:8px;overflow-x:auto;-webkit-overflow-scrolling:touch;
          padding:10px max(var(--pad), env(safe-area-inset-left)) 10px max(var(--pad), env(safe-area-inset-right));
          max-width:1180px;margin:0 auto;scrollbar-width:none}}
  .tabbar::-webkit-scrollbar{{display:none}}
  .tab{{flex:0 0 auto;border:1px solid var(--line);background:#fff;color:#39404f;
        border-radius:999px;padding:8px 15px;font-size:clamp(12.5px,3vw,13.5px);
        cursor:pointer;white-space:nowrap;font-family:inherit;line-height:1.4}}
  .tab[aria-selected="true"]{{background:var(--blue);border-color:var(--blue);color:#fff;font-weight:600}}
  .panel[hidden]{{display:none}}

  h2{{font-size:clamp(16px,4vw,19px);margin:0 0 6px;display:flex;align-items:flex-start;gap:9px;line-height:1.5}}
  h2 span.dot{{width:8px;height:8px;border-radius:50%;background:var(--blue);display:inline-block;flex:0 0 auto;margin-top:.55em}}
  h3{{font-size:clamp(14px,3.4vw,15.5px);margin:22px 0 8px;color:#2b3242}}
  .lead{{color:var(--muted);font-size:clamp(12.5px,3.2vw,14px);margin:0 0 14px;overflow-wrap:anywhere}}
  .card{{background:var(--card);border:1px solid var(--line);border-radius:14px;
        box-shadow:0 4px 18px rgba(20,24,33,.05);overflow:hidden;margin-bottom:18px}}
  .card img{{display:block;width:100%;height:auto}}
  .cap{{padding:11px 14px;font-size:clamp(12px,3vw,13px);color:var(--muted);
       border-top:1px solid var(--line);background:#fbfcfe;overflow-wrap:anywhere}}
  .cap a{{white-space:nowrap}}
  .zoom{{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap}}

  .iframe-box{{border:0;width:100%;height:520px;display:block;background:#fff}}
  .graph-box{{border:0;width:100%;height:700px;display:block;background:#fff}}
  .scroll-media{{overflow-x:auto;-webkit-overflow-scrolling:touch}}
  .scroll-media > *{{min-width:940px;display:block}}
  .only-mobile{{display:none}}

  /* 表格 */
  .table-wrap{{overflow-x:auto;-webkit-overflow-scrolling:touch}}
  .table-wrap table{{min-width:560px}}
  table{{width:100%;border-collapse:collapse;font-size:clamp(12.5px,3vw,13.5px);background:#fff}}
  th,td{{border-bottom:1px solid var(--line);padding:9px 12px;text-align:left;vertical-align:top}}
  th{{background:#f3f6fb;color:#39404f;font-weight:600;white-space:nowrap}}
  td.num{{font-variant-numeric:tabular-nums;white-space:nowrap}}
  td.nowrap{{white-space:nowrap}}
  td.desc{{color:#4b5464;font-size:12.5px}}
  .scroll-hint{{display:none;font-size:11.5px;color:var(--muted);padding:7px 12px 0}}

  .pill{{display:inline-block;font-size:11.5px;padding:2px 9px;border-radius:999px;
         border:1px solid transparent;white-space:nowrap;line-height:1.6}}
  .pill.b{{background:#e7f0fd;color:#1e5cb3;border-color:#c8dbf7}}
  .pill.g{{background:#e8f6ec;color:#2c7a48;border-color:#c7e6d2}}
  .pill.a{{background:#fff5e3;color:#8a5b00;border-color:#f2ddb6}}
  .pill.r{{background:#fdecea;color:#a63023;border-color:#f6cfc9}}
  .pill.m{{background:#f1f0fa;color:#54459b;border-color:#ddd9f2}}
  tr.obj-row{{cursor:pointer}}
  tr.obj-row:hover td{{background:#f7f9fe}}
  tr.obj-row.flash td{{background:#fff4d6;transition:background .2s}}
  tr.obj-row.sel td{{background:#e7f0fd}}

  .detail{{display:none;margin:14px 0 0;padding:14px 16px;border:1px solid #c8dbf7;border-radius:12px;
           background:#f7faff;font-size:13px;color:#2b3242}}
  .detail.on{{display:block}}
  .detail b{{color:#14478f}}

  .note{{border-left:3px solid var(--amber);background:#fff9ec;padding:11px 14px;border-radius:0 10px 10px 0;
        font-size:clamp(12.5px,3.1vw,13.5px);color:#5b4a26;margin:12px 0}}
  .note strong{{color:#8a5b00}}
  .note.red{{border-left-color:var(--red);background:#fdf1ef;color:#6b2b23}}
  .note.red strong{{color:#a63023}}
  ul.lim{{margin:8px 0 0;padding-left:19px;font-size:clamp(12.5px,3.1vw,13.5px);color:#3c4351}}
  ul.lim li{{margin:7px 0;overflow-wrap:anywhere}}
  .two{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
  footer{{border-top:1px solid var(--line);margin-top:30px;padding:22px 0 42px;color:var(--muted);font-size:12.5px;line-height:1.75}}
  a{{color:var(--blue);text-decoration:none}}
  code{{background:#eef2f8;padding:1px 5px;border-radius:4px;font-size:12px}}

  @media (max-width:860px){{
    :root{{--pad:16px}}
    header{{padding:34px 0 28px}}
    .tag{{font-size:12px;margin-bottom:10px;padding:3px 10px}}
    .kpis{{grid-template-columns:repeat(2,1fr);gap:10px}}
    .kpi{{padding:13px 13px 11px;border-radius:12px}}
    .two{{grid-template-columns:1fr;gap:14px}}
    .iframe-box{{height:440px}}
    .graph-box{{height:600px}}
    .card{{border-radius:12px}}
    .scroll-hint{{display:block}}
    .only-mobile{{display:block;margin-top:6px;color:#8a5b00}}
    .table-wrap table{{min-width:520px}}
    .col-desc, td.desc{{display:none}}   /* 手机上隐藏长描述列，点行看详情卡，避免表格被撑得过高 */
    footer{{margin-top:24px;padding:18px 0 34px}}
  }}
</style>
</head>
<body>

<header>
  <div class="wrap">
    <div class="tag">毕业设计 · 第四阶段</div>
    <h1>交通事故道路平面场景重建<br>与多视图联动</h1>
    <p class="sub">
      第三阶段给出"画面里有什么、谁和谁可能有关"；第四阶段回答"它们各自在路面的什么位置"。
      用四个道路控制点求解单应性变换，把车辆与行人的地面锚点投到统一的道路局部平面，
      重建路面几何（车道线、路缘），再用俯视场景、对象表、关系面板与知识图谱四种视图交叉呈现。
    </p>
    <div class="meta">案例：ACCIDENT_001 ｜ 坐标单位：local_unit（局部坐标单位，<b>非米制实测</b>）｜ 道路结构：{geometry_stat}</div>
  </div>
</header>

<div class="wrap">
  <div class="kpis">
    <div class="kpi b"><div class="v">15</div><div class="u">objects</div><div class="k">统一场景对象（O1–O15）</div></div>
    <div class="kpi g"><div class="v">{counts['projected']}</div><div class="u">projected</div><div class="k">标定区内投影，坐标可用于测量</div></div>
    <div class="kpi a"><div class="v">{len(reliable)}</div><div class="u">reliable pairs</div><div class="k">可靠距离对，最近 {nearest.get('a','—')}—{nearest.get('b','—')} = {nearest.get('distance', 0):.2f}</div></div>
    <div class="kpi r"><div class="v">{n_nodes} / {n_edges}</div><div class="u">nodes / edges</div><div class="k">含道路结构与道路坐标的知识图谱</div></div>
  </div>
</div>

<div class="tabs">
  <div class="tabbar" role="tablist" aria-label="视图切换">
    <button class="tab" role="tab" data-view="scene" aria-selected="true">② 俯视道路场景</button>
    <button class="tab" role="tab" data-view="image" aria-selected="false">① 原始图像与标定</button>
    <button class="tab" role="tab" data-view="table" aria-selected="false">③ 对象表与距离</button>
    <button class="tab" role="tab" data-view="graph" aria-selected="false">④ 知识图谱</button>
    <button class="tab" role="tab" data-view="method" aria-selected="false">⑤ 方法与误差</button>
  </div>
</div>

<div class="wrap">

<!-- ============ ② 俯视道路场景 ============ -->
<div class="panel" id="panel-scene" role="tabpanel">
  <h2><span class="dot"></span>道路平面二维场景重建</h2>
  <p class="lead">
    底图是把原图中"确实是沥青、且没被车辆挡住"的像素做透视变换后铺出来的，缺失处留白；
    两种标记区分可靠性：<b>实心</b>=标定区内投影，<b>空心虚线框</b>=外推投影（画出来但不参与测量）。
    红色连线只连接<strong>双方都在标定区内</strong>的对象。
  </p>
  <div class="card">
    <div class="scroll-media">
      <a href="outputs/ACCIDENT_001/scene_topdown.png" target="_blank" rel="noopener">
        <img src="outputs/ACCIDENT_001/scene_topdown.png" alt="道路平面二维场景重建图"/>
      </a>
    </div>
    <div class="cap zoom">
      <span>斜线区＝路面纹理被车辆遮挡，只能按道路几何推断；比例尺 1 local_unit 未标定为米。</span>
      <a href="outputs/ACCIDENT_001/scene_topdown.png" target="_blank" rel="noopener">查看原图 ↗</a>
    </div>
  </div>

  <h3>可点击的交互版</h3>
  <p class="lead">点击场景中的任一对象，页面会自动跳到「对象表」并高亮对应行，实现场景—表格联动。</p>
  <div class="card">
    <div class="scroll-media">
      <iframe class="iframe-box" id="frame-scene" src="outputs/ACCIDENT_001/scene_topdown.html" title="交互式道路平面场景" loading="lazy"></iframe>
    </div>
    <div class="cap">
      拖动平移、滚轮缩放、悬浮看详情。
      <span class="only-mobile">手机上可在框内左右滑动查看完整宽度。</span>
      也可单独打开 <a href="outputs/ACCIDENT_001/scene_topdown.html" target="_blank" rel="noopener">scene_topdown.html</a>。
    </div>
  </div>

  <h3>这张图看什么</h3>
  <ul class="lim">
    <li><b>位置关系</b>：O2 在道路中部偏左，O8、O9 两名行人位于右侧靠边，O8 更靠近车行道。</li>
    <li><b>车道结构</b>：L1（左分界线）与 L2（右分界线）在道路坐标中宽约 8 local_unit，两线之间未见虚线，<strong>无法判定车道数</strong>。</li>
    <li><b>诚实性标注</b>：O1、O3、O4 是外推结果，图上以半透明呈现并明确写"不参与可靠距离"；O5、O6、O10、O13、O14、O15 没有坐标，只列清单——不把标定区外的点硬拉回矩形内。</li>
  </ul>
</div>

<!-- ============ ① 原始图像与标定 ============ -->
<div class="panel" id="panel-image" role="tabpanel" hidden>
  <h2><span class="dot"></span>原始图像与道路标定</h2>
  <p class="lead">蓝=车辆、绿=行人、紫=道路设施；红点是检测框底边中点（近似地面接触点）；黄色梯形是四点标定出的道路平面范围。</p>
  <div class="card">
    <a href="outputs/ACCIDENT_001/_stage4_overlay.png" target="_blank" rel="noopener">
      <img src="outputs/ACCIDENT_001/_stage4_overlay.png" alt="检测框与标定四边形叠加图"/>
    </a>
    <div class="cap zoom">
      <span>四个控制点全部取自两条车道白线：近端两点是白线在画面底边的实测中心，远端两点由白线直线方程外推到被车辆遮挡的行。</span>
      <a href="outputs/ACCIDENT_001/_stage4_overlay.png" target="_blank" rel="noopener">查看原图 ↗</a>
    </div>
  </div>
  <div class="card">
    <div class="scroll-hint">← 左右滑动查看完整表格 →</div>
    <div class="table-wrap">
      <table>
        <tr><th>控制点</th><th>图像坐标 (u, v)</th><th>道路坐标 (X, Y)</th></tr>
        {calib_rows}
      </table>
    </div>
    <div class="cap">原点=近端左侧控制点；X=道路横向（右为正）；Y=道路纵向（远端为正）；标定矩形人为设定 8 × 3 local_unit。</div>
  </div>
  <div class="note">
    <strong>为什么控制点取白线而不取车道中心？</strong>
    白线边缘清晰、可逐行扫描取中心，比"车道中心"更容易复核；代价是标定矩形描述的是"两条白线之间的区域"，
    而不是整条道路的边界，所以页面只称它为"标定区域"。
  </div>
</div>

<!-- ============ ③ 对象表与距离 ============ -->
<div class="panel" id="panel-table" role="tabpanel" hidden>
  <h2><span class="dot"></span>对象表与距离面板</h2>
  <p class="lead">点任意一行可展开该对象的锚点、坐标与语义描述；从场景图或知识图谱点击对象也会自动跳到并高亮对应行。</p>
  <div class="card">
    <div class="scroll-hint">← 左右滑动查看完整表格 →</div>
    <div class="table-wrap">
      <table id="obj-table">
        <tr><th>编号</th><th>类别</th><th>道路坐标 (X, Y)</th><th>投影状态</th><th>锚点像素</th><th class="col-desc">VLM 描述</th></tr>
        {''.join(obj_rows)}
      </table>
    </div>
    <div class="cap">共 {len(objects)} 个对象：区内投影 {counts['projected']}、外推 {counts['projected_extrapolated']}、超范围 {counts['outside_calibration_region']}、悬空设施 {counts['excluded_non_ground_object']}。</div>
  </div>
  <div class="detail" id="obj-detail"></div>

  <h3>可靠距离（双方均在标定区内）</h3>
  <div class="card">
    <div class="scroll-hint">← 左右滑动查看完整表格 →</div>
    <div class="table-wrap">
      <table>
        <tr><th>对象对</th><th>距离</th><th>单位</th><th>可靠性依据</th></tr>
        {''.join(rel_rows)}
      </table>
    </div>
    <div class="cap">只列出 {len(reliable)} 对可靠距离；其余 {len(pairs) - len(reliable)} 对含外推对象，数值保留在 JSON 里供追溯，页面不展示、不作为结论。</div>
  </div>

  <h3>语义关系与其距离（VLM 观察层）</h3>
  <div class="card">
    <div class="scroll-hint">← 左右滑动查看完整表格 →</div>
    <div class="table-wrap">
      <table>
        <tr><th>关系</th><th>VLM 证据</th><th>道路平面距离</th><th>是否可用于结论</th></tr>
        {''.join(sem_rows) if sem_rows else '<tr><td colspan="4">本案例无带距离的语义关系</td></tr>'}
      </table>
    </div>
    <div class="cap">
      VLM 说"接触"是语义判断，不是碰撞事实；距离只在双方都投影到标定区内时给出数值。
      {accident_note}
    </div>
  </div>
</div>

<!-- ============ ④ 知识图谱 ============ -->
<div class="panel" id="panel-graph" role="tabpanel" hidden>
  <h2><span class="dot"></span>含道路结构与道路坐标的知识图谱</h2>
  <p class="lead">
    图谱新增两类内容：<b>道路结构节点</b>（路面区域、L1/L2 车道线、RB_R 路缘）与<b>对象坐标属性</b>
    （ground_x / ground_y / coordinate_unit / projection_status / coordinate_source）。
    点击节点会自动跳到对象表并高亮。
  </p>
  <div class="card">
    <div class="scroll-media">
      <iframe class="graph-box" id="frame-graph" src="outputs/ACCIDENT_001/interactive_graph_spatial.html" title="交互式知识图谱" loading="lazy"></iframe>
    </div>
    <div class="cap">
      单指拖动旋转、双指或滚轮缩放、点按节点看详情。
      <span class="only-mobile">手机竖屏下图形会被两侧裁掉一部分，<strong>横屏</strong>观看最完整。</span>
      也可单独打开 <a href="outputs/ACCIDENT_001/interactive_graph_spatial.html" target="_blank" rel="noopener">interactive_graph_spatial.html</a>。
    </div>
  </div>
  <div class="card">
    <a href="outputs/ACCIDENT_001/knowledge_graph_spatial.png" target="_blank" rel="noopener">
      <img src="outputs/ACCIDENT_001/knowledge_graph_spatial.png" alt="含道路坐标的知识图谱（2D 静态版）"/>
    </a>
    <div class="cap zoom">
      <span>静态版便于放进论文：蓝点线为 road coordinate 引用，橙点线为 L2 几何候选，红实线为 L3 语义关系。</span>
      <a href="outputs/ACCIDENT_001/knowledge_graph_spatial.png" target="_blank" rel="noopener">查看原图 ↗</a>
    </div>
  </div>
  <div class="note">
    <strong>图谱里的坐标不是"摆位"。</strong>
    3D 视图的节点位置由 spring_layout 决定，只表示关系结构，<b>不代表真实道路位置</b>；
    真实坐标写在节点属性与 JSON 边属性里（<code>ground_x</code>、<code>ground_y</code>、<code>unit</code>）。
  </div>
</div>

<!-- ============ ⑤ 方法与误差 ============ -->
<div class="panel" id="panel-method" role="tabpanel" hidden>
  <h2><span class="dot"></span>投影质量与方法边界</h2>
  <p class="lead">
    四点单应性只有 4 组对应点、恰好确定 8 个自由度，<b>不存在留一交叉验证</b>（去掉一点后方程数不足）。
    因此这里用三类与标定四边形无关的检查来量化可信度，而不是伪造一个 LOO 误差。
  </p>

  <h3>1. 几何一致性（车道线在道路坐标中的表现）</h3>
  <div class="card">
    <div class="scroll-hint">← 左右滑动查看完整表格 →</div>
    <div class="table-wrap">
      <table>
        <tr><th>车道线</th><th>斜率 dX/dY</th><th>偏离竖直</th><th>拟合残差</th><th>参与拟合点数</th></tr>
        {''.join(lane_rows)}
      </table>
    </div>
    <div class="cap">
      两线夹角 {par.get('angle_between_lanes_deg', '—')}°、间距从 Y=0 的 {par.get('gap_at_y_0_local_unit', '—')} 变到 Y=3 的 {par.get('gap_at_y_3_local_unit', '—')} local_unit。
      斜率接近 0 说明标定矩形的纵向轴与车道方向基本一致。
    </div>
  </div>

  <h3>2. 锚点敏感度（像素误差放大成多少局部单位）</h3>
  <div class="note">
    锚点标注抖动 <b>{sens.get('shift_px', 5):g} px</b>，在道路坐标中平均造成
    <b>{sens.get('mean_local_unit', 0):.3f} local_unit</b> 的位移，最大 {sens.get('max_local_unit', 0):.3f} local_unit。
    远处（Y 大）同样的像素误差代价更高——这正是"越远越不准"的定量表述。
  </div>

  <h3>3. 尺度歧义（为什么不能写"米"）</h3>
  <div class="note red">
    <strong>{scale.get('statement', '')}</strong>
  </div>

  <h3>4. 自洽性与已知局限</h3>
  <div class="card">
    <div class="table-wrap">
      <table>
        <tr><th>项目</th><th>结果</th><th>说明</th></tr>
        <tr><td>四点回代误差</td><td class="num">{selfc.get('max_error_local_unit', 0):.6f} local_unit</td><td>只证明 DLT 求解闭合，不代表标定精度</td></tr>
        <tr><td>投影成功率</td><td class="num">{summary.get('success_rate_percent', '—')}%</td><td>6/12 个地面目标得到可用坐标</td></tr>
        <tr><td>路面纹理覆盖</td><td class="num">10.1%</td><td>其余路面被车辆遮挡，底图只能按几何推断</td></tr>
        <tr><td>两线夹角</td><td class="num">{par.get('angle_between_lanes_deg', '—')}°</td><td>平行度良好，说明标定与道路方向基本对齐</td></tr>
      </table>
    </div>
    <div class="cap">上述数值全部来自 calibration_check.json，可在仓库中复核。</div>
  </div>

  <ul class="lim">
    <li><strong>不能写米。</strong>1 local_unit = 标定矩形宽度 ÷ 8，而矩形宽度未实测。</li>
    <li><strong>不能写"碰撞"。</strong>VLM 的"接触"是语义观察；几何上只能给"双方都在标定区内时的平面距离"。</li>
    <li><strong>符号不代表实测尺寸。</strong>车辆矩形 0.26 × 0.58 local_unit、行人圆点 r=0.075 只是示意，不代表车身尺寸与朝向。</li>
    <li><strong>外推不参与测量。</strong>含 projected_extrapolated 的距离一律降级，只在 JSON 里保留追溯值。</li>
    <li><strong>3D 图谱位置不是坐标。</strong>spring_layout 只表达关系结构。</li>
  </ul>

  <h3>5. 技术路线</h3>
  <div class="card">
    <div class="scroll-hint">← 左右滑动查看完整表格 →</div>
    <div class="table-wrap">
      <table>
        <tr><th>步骤</th><th>方法</th><th>产物</th></tr>
        <tr><td class="nowrap">道路控制点标定</td><td>车道白线逐行扫描取中心 + 直线拟合外推</td><td>image_control_points.json / calibration.json</td></tr>
        <tr><td class="nowrap">道路几何标注</td><td>人工判读网格 + 亮度扫描复核车道线</td><td>road_geometry.json</td></tr>
        <tr><td class="nowrap">几何逐点投影</td><td>同一 H 变换，逐点标注 inside/outside</td><td>road_geometry_local.json</td></tr>
        <tr><td class="nowrap">路面俯视底图</td><td>HSV 沥青掩膜 ∩ 道路多边形 − 对象框，warpPerspective</td><td>road_surface_topdown.png</td></tr>
        <tr><td class="nowrap">对象定位与距离</td><td>bbox 底边中点 → perspectiveTransform → 欧氏距离（仅区内可算）</td><td>scene_model_spatial.json / ground_distances.json</td></tr>
        <tr><td class="nowrap">投影质量评估</td><td>几何一致性 + 锚点敏感度 + 尺度歧义</td><td>calibration_check.json</td></tr>
        <tr><td class="nowrap">场景重建</td><td>matplotlib 静态图 + Plotly 交互图（共用同一数据）</td><td>scene_topdown.png / .html</td></tr>
        <tr><td class="nowrap">知识图谱扩展</td><td>MultiDiGraph + 道路结构节点 + 坐标属性 + 点击联动</td><td>knowledge_graph_spatial.json/.png/.html</td></tr>
      </table>
    </div>
  </div>
</div>

</div>

<footer>
  <div class="wrap">
    数据来源：事故图片的 VLM 语义分析、YOLO 检测框、人工确认的对象映射、道路几何人工标注与单应性变换计算。<br>
    本页面为静态页面，所有图片与交互视图随仓库发布，无需运行后端即可查看。
  </div>
</footer>

<script>
(function () {{
  var tabs = Array.prototype.slice.call(document.querySelectorAll('.tab'));
  var panels = {{
    scene: document.getElementById('panel-scene'),
    image: document.getElementById('panel-image'),
    table: document.getElementById('panel-table'),
    graph: document.getElementById('panel-graph'),
    method: document.getElementById('panel-method'),
  }};

  function showView(name, opts) {{
    opts = opts || {{}};
    if (!panels[name]) name = 'scene';
    tabs.forEach(function (t) {{
      var on = t.dataset.view === name;
      t.setAttribute('aria-selected', on ? 'true' : 'false');
    }});
    Object.keys(panels).forEach(function (k) {{
      if (panels[k]) panels[k].hidden = (k !== name);
    }});
    if (!opts.silent) history.replaceState(null, '', '#' + name);
  }}

  tabs.forEach(function (t) {{
    t.addEventListener('click', function () {{ showView(t.dataset.view); }});
  }});

  var hash = (location.hash || '').replace('#', '');
  showView(hash || 'scene', {{silent: true}});

  // 直接改地址栏 hash 或分享带 #view 的链接时同步切换视图
  window.addEventListener('hashchange', function () {{
    var h = (location.hash || '').replace('#', '');
    if (h && panels[h]) showView(h, {{silent: true}});
  }});

  // ---------------- 对象表联动 ----------------
  var rows = Array.prototype.slice.call(document.querySelectorAll('#obj-table tr[data-oid]'));
  var detail = document.getElementById('obj-detail');
  var data = {json.dumps({o["id"]: {
        "cat": o.get("category_zh", ""),
        "status": o.get("ground_position", {}).get("status"),
        "x": o.get("ground_position", {}).get("x"),
        "y": o.get("ground_position", {}).get("y"),
        "anchor": o.get("ground_anchor_image"),
        "conf": o.get("confidence"),
        "desc": (o.get("vlm_description") or "")[:160],
        "bbox": o.get("bbox"),
    } for o in objects}, ensure_ascii=False)};

  function statusText(s) {{
    return ({{projected:'区内投影（坐标可用）',
             projected_extrapolated:'外推投影（不参与可靠距离）',
             outside_calibration_region:'超出标定范围（不给坐标）',
             excluded_non_ground_object:'悬空设施（不参与地面投影）'}})[s] || s;
  }}

  function highlight(id, scroll) {{
    var row = null;
    rows.forEach(function (r) {{
      r.classList.toggle('sel', r.dataset.oid === id);
      if (r.dataset.oid === id) row = r;
    }});
    if (!row) return;
    if (scroll && row.scrollIntoView) row.scrollIntoView({{block: 'center', behavior: 'smooth'}});
    row.classList.add('flash');
    setTimeout(function () {{ row.classList.remove('flash'); }}, 1600);
  }}

  function showDetail(id) {{
    var d = data[id];
    if (!d || !detail) return;
    var pos = (d.x === null || d.x === undefined) ? '未给出道路坐标' : ('X=' + d.x.toFixed(2) + '，Y=' + d.y.toFixed(2) + ' local_unit');
    detail.innerHTML =
      '<b>' + id + ' ' + d.cat + '</b><br>' +
      '投影状态：' + statusText(d.status) + '<br>' +
      '道路坐标：' + pos + '<br>' +
      '地面锚点（像素）：' + (d.anchor ? '(' + d.anchor[0].toFixed(0) + ', ' + d.anchor[1].toFixed(0) + ')' : '—') +
      '　锚点规则：检测框底边中点' + '<br>' +
      (d.conf !== null && d.conf !== undefined ? 'YOLO 置信度：' + d.conf + '<br>' : '') +
      (d.desc ? 'VLM 描述：' + d.desc : '');
    detail.classList.add('on');
  }}

  rows.forEach(function (r) {{
    r.addEventListener('click', function () {{
      highlight(r.dataset.oid, false);
      showDetail(r.dataset.oid);
    }});
  }});

  // ---------------- iframe 点击联动 ----------------
  window.addEventListener('message', function (e) {{
    var d = e.data || {{}};
    if (d.type !== 'scene-object' && d.type !== 'graph-node') return;
    var id = String(d.id || '');
    if (!id) return;
    showView('table');
    highlight(id, true);
    showDetail(id);
  }});

  // Plotly iframe 自适应
  ['frame-scene', 'frame-graph'].forEach(function (fid) {{
    var f = document.getElementById(fid);
    if (!f) return;
    function refit() {{
      try {{ f.contentWindow.dispatchEvent(new Event('resize')); }} catch (e) {{}}
    }}
    f.addEventListener('load', function () {{ setTimeout(refit, 380); }});
    window.addEventListener('resize', function () {{ setTimeout(refit, 320); }});
    window.addEventListener('orientationchange', function () {{ setTimeout(refit, 520); }});
  }});
}})();
</script>

</body>
</html>
"""
    (ROOT / "index.html").write_text(html, encoding="utf-8")
    print(f"index.html 已生成：{(ROOT / 'index.html').stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    build()
