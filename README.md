# 交通事故语义场景建模：VLM × YOLO × 知识图谱 × 道路平面重建

> 毕业设计项目：从单张交通事故图片出发，依次完成「视觉语义分析 → 目标检测与几何定位 → 多源融合与知识图谱 → 道路平面空间建模与二维场景重建」四个阶段，最终输出一个可交互的网页系统。

**在线可视化（静态页面，任何人可直接打开）**：<https://ddw-wdd.github.io/traffic-scene-vlm/>
（页面源为仓库根目录的 [`index.html`](index.html)，由 `build_site.py` 依据 outputs 下的 JSON 生成）

---

## 一、四个阶段与技术路线

| 阶段 | 目标 | 核心技术 | 入口 | 端口 |
|---|---|---|---|---|
| 一 | 图片语义理解 | Qwen 视觉语言模型，结构化 JSON（对象、关系、事故类型） | `app.py` | 7860 |
| 二 | 目标检测与几何定位 | YOLO26n 检测框、中心点、IoU、NetworkX 关系图 | `app_stage2.py` | 7861 |
| 三 | 多源融合与知识图谱 | VLM 对象 ↔ YOLO 检测人工/几何匹配、L1/L2/L3 分层关系、2D 图谱三维化 | `app_stage3.py` | 7862 |
| 四 | 道路平面场景重建 | 四点单应性变换（Homography）、道路几何重建、路面掩膜俯视底图、可信度分级距离、多视图联动 | `app_stage4.py` | 7863 |

第四阶段还可脱离网页单独运行：`python run_stage4.py`。

## 二、第四阶段做了什么

```
第三阶段融合场景（15 个统一对象 O1–O15 + VLM 语义 + YOLO 检测框 + L1/L2/L3 关系）
        │
        ├─ ① 道路控制点标定：两条车道白线各取两点（近端实测、远端直线外推）
        ├─ ② cv2.findHomography（4 点 DLT）求解 3×3 单应性矩阵 H
        ├─ ③ 道路几何重建：道路区域 / 路缘 / 车道线，逐点投影并标注区内/区外（road_geometry.py）
        ├─ ④ 路面掩膜透视变换 → 俯视底图（沥青阈值 ∩ 道路多边形 − 对象框）
        ├─ ⑤ 交通对象地面锚点（bbox 底边中点）→ 道路坐标，按可信度分级（区内/外推/超范围）
        ├─ ⑥ 可靠距离：仅双方都在标定区内才算；含外推的距离降级为"仅追溯"
        ├─ ⑦ 静态 + 交互双版俯视场景图（符号化、遮挡斜线、语义关系层）
        └─ ⑧ 知识图谱新增道路结构节点与坐标属性，页面五视图联动
```

### 本案例（ACCIDENT_001）结果

| 指标 | 数值 |
|---|---|
| 场景对象总数 | 15 |
| 标定区内投影（projected） | 3（O2、O8、O9） |
| 外推投影（projected_extrapolated） | 3（O1、O3、O4） |
| 超出标定范围（不给坐标） | 6（O5、O6、O10、O13、O14、O15） |
| 悬空设施排除 | 3（O7、O11、O12 信号灯） |
| 投影成功率（按有效地面目标计） | **50.0%** |
| 道路平面距离对数 | 15 对（其中 `reliable=true` 仅 **3 对**） |
| 事故对 `O1 —接触→ O2` | 约 **5.38 local_unit**（含外推点，仅追溯，不作结论） |
| 可靠距离最近一对 `O8 — O9` | **1.25 local_unit** |
| 其余可靠对 | O2—O9 = 2.77、O2—O8 = 3.71 local_unit |
| 道路几何 | 1 个路面多边形 + 2 条车道线 + 1 条路缘（RB_R） |
| 路面纹理覆盖 | 10.1%（其余被车辆遮挡，底图以斜线标注"仅按几何推断"） |
| 知识图谱规模 | 17/42 → 18/48 → **22 节点 / 56 边**（新增 4 个道路结构节点与坐标属性） |
| 锚点敏感度 | 5 px 抖动 → 平均 0.137 local_unit 位移 |

控制点（全部取自车道白线，单位 `local_unit`）：

| 控制点 | 角色 | 图像坐标 (u, v) | 道路坐标 (X, Y) |
|---|---|---|---|
| P1 | near_left | (365, 764) | (0.0, 0.0) |
| P2 | near_right | (1212, 764) | (8.0, 0.0) |
| P3 | far_right | (1170, 620) | (8.0, 3.0) |
| P4 | far_left | (612, 620) | (0.0, 3.0) |

## 三、目录结构

```
├── app.py                    # 阶段一：VLM 语义分析（7860）
├── app_stage2.py             # 阶段二：YOLO 检测 + 关系图（7861）
├── app_stage3.py             # 阶段三：VLM×YOLO 融合 + 知识图谱（7862）
├── app_stage4.py             # 阶段四：道路平面建模页面（7863）
├── run_stage4.py             # 阶段四命令行一键流程（投影→几何→底图→评估→场景→图谱）
├── build_site.py             # 由 outputs 的 JSON 生成多视图 index.html
├── rebuild_graph_html.py     # 只重建 3D 图谱 HTML
├── scene_model.py            # 场景数据模型与校验
├── modules/
│   ├── scene_fusion.py       # 融合模型：对象对齐、关系分层
│   ├── knowledge_graph.py    # MultiDiGraph 构建 + 2D 可视化 + 道路结构节点
│   ├── interactive_graph.py  # 2D 布局三维化（Plotly）+ 点击联动
│   ├── road_plane_projection.py    # 单应性变换、可信度分级、可靠距离
│   ├── road_geometry.py      # 道路几何投影、路面掩膜、俯视底图
│   ├── scene_topdown.py      # 静态俯视场景图（符号化）
│   ├── scene_interactive.py  # 交互版场景（Plotly）
│   ├── calibration_check.py  # 投影质量评估
│   └── spatial_visualization.py    # 旧版俯视示意图（保留兼容）
├── tools/pick_control_points.py    # 四点人工选取工具（可选）
├── data/                     # 阶段一~三的原始结果 JSON
├── index.html                # 多视图可视化页面（GitHub Pages 源，根目录）
├── outputs/ACCIDENT_001/     # 阶段三、四全部产物（JSON / PNG / HTML）
└── 第四阶段_使用说明.md       # 阶段四详细说明、运行方式与论文表述边界
```

## 四、环境与安装

需要 Python 3.10 及以上。

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
```

阶段二的 YOLO 检测依赖 `ultralytics`（会自动安装 CPU 版 PyTorch，体积较大）。只跑第四阶段或只想看静态页面时不需要装。

### 配置 API Key

复制 `.env.example` 为 `.env`，填入你自己的百炼 API Key：

```text
DASHSCOPE_API_KEY=sk-your-own-api-key-here
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
VLM_MODEL=qwen3.7-plus
```

Key 与 Base URL 的地域、计费方案必须一致。`.env` 已被 `.gitignore` 排除，**不要把真实密钥提交到仓库**。

### 运行

```bash
python app.py            # 阶段一  http://127.0.0.1:7860
python app_stage2.py     # 阶段二  http://127.0.0.1:7861
python app_stage3.py     # 阶段三  http://127.0.0.1:7862
python app_stage4.py     # 阶段四  http://127.0.0.1:7863
python run_stage4.py     # 阶段四命令行版（只产出文件，不起网页）
python build_site.py     # 用最新 JSON 重新生成 index.html（改数据后必跑）
```

需要重新手工点选道路控制点：

```bash
python tools/pick_control_points.py outputs/ACCIDENT_001/original.jpg
# 左键加点（顺序 near_left → near_right → far_right → far_left），B 撤销，R 重置，S 保存，Q 退出
```

## 五、结果文件（`outputs/ACCIDENT_001/`）

| 文件 | 内容 |
|---|---|
| `vlm_result.json` / `yolo_detections.json` / `object_mapping.json` | 阶段一~三的原始结果 |
| `scene_model.json` | 阶段三融合场景（15 个统一对象 + 分层关系） |
| `knowledge_graph.json` / `.png` | 阶段三知识图谱（17 节点 / 42 边） |
| `calibration.json` | 第四阶段标定定义：点序、图像/道路坐标、假设尺寸、外推容差、局限 |
| `image_control_points.json` | 四个控制点的选取结果与来源说明 |
| `scene_model_spatial.json` | 带 `ground_position`（x/y/unit/status/method）的空间场景，不覆盖阶段三文件 |
| `road_geometry.json` / `road_geometry_local.json` | 道路几何标注（像素）/ 投影到道路坐标（逐点状态） |
| `road_surface_topdown.png` / `scene_base_topdown.png` | 路面掩膜俯视底图 / 交互页合成底图 |
| `ground_distances.json` | 15 对距离（含 `reliable` 标记）、可靠子集、最近一对、语义关系距离 |
| `calibration_check.json` | 自洽性、车道线一致性、锚点敏感度、尺度歧义、结论口径 |
| `spatial_model_info.json` | H 矩阵、锚点规则、状态计数、成功率、距离口径、局限 |
| `scene_topdown.png` / `scene_topdown.html` | 道路平面场景重建（静态 / 交互） |
| `knowledge_graph_spatial.json` / `.png` / `.html` | 含道路结构与坐标的知识图谱（22 节点 / 56 边） |
| `plotly.min.js` | 交互页共用的 Plotly 运行库 |
| `_stage4_overlay.png` | 原图上的检测框 + 地面锚点 + 标定四边形 |

## 六、已知边界（写论文时的重要约束）

1. **坐标单位是 `local_unit`，不能写成“米”**。本案例没有实测道路尺寸，8 × 3 只是为了坐标系可复现而假设的局部矩形。
2. 四个控制点同时用于求解与检验，`reprojection_error = 0` 只能证明矩阵求解正确，**不能**当作定位精度；要给出精度必须另留独立检验点。
3. 远端两个控制点位于被车辆/行人遮挡处，由白线直线方程外推得到，是已记录的不确定性来源。
4. 地面锚点用 bbox 底边中点近似轮胎/脚部接地点；对斜向停放的车辆会向车身一侧偏移（本案例 O1 被判为外推的直接原因）。
5. 标定矩形是局部坐标参考区，**不是**道路真实边界。
6. VLM 语义结果（如“接触迹象”）是观察推断，不等于物理碰撞判定；YOLO 检测框重叠也不等于碰撞。

## 七、后续可改进

- 增加 2–3 个已知相对坐标的路面检验点，用 `E=(1/N)Σ√[(Xᵢ−X̂ᵢ)²+(Yᵢ−Ŷᵢ)²]` 给出局部单位误差。
- 实测道路宽度/纵深，把 `unit` 从 `local_unit` 升级为 `m`。
- 用实例分割（轮胎接地点、脚部关键点）替代 bbox 底边中点，降低斜向停放的锚点误差。
- 矢量化车道线与道路多边形，把位置引用关系升级为真正的 `located_on` 空间关系。

---

本项目为课程/毕业设计用途，示例图片与检测结果仅用于方法演示。
