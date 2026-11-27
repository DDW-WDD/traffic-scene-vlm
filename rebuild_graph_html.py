"""只重建交互式 3D 知识图谱 HTML（不动其它产物，不重跑检测与投影）。

用法：
    python rebuild_graph_html.py

读取 outputs/ACCIDENT_001/scene_model_spatial.json
→ 构建 MultiDiGraph → Plotly 3D 图 → 覆盖 interactive_graph_spatial.html

说明：统一走 modules.interactive_graph.write_graph_html，
它负责"共用一份 plotly.min.js + viewport + 画布填满容器 + 点击联动脚本"。
"""

import json
import sys
from pathlib import Path

from modules.knowledge_graph import build_fused_knowledge_graph
from modules.interactive_graph import build_interactive_3d_figure, write_graph_html

ROOT = Path(__file__).resolve().parent
CASE_DIR = ROOT / "outputs" / "ACCIDENT_001"


def main() -> int:
    scene_path = CASE_DIR / "scene_model_spatial.json"
    if not scene_path.exists():
        print(f"[ERR] 找不到 {scene_path}")
        return 1

    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    graph = build_fused_knowledge_graph(scene)
    print(f"节点 {graph.number_of_nodes()} / 边 {graph.number_of_edges()}")

    fig = build_interactive_3d_figure(graph)
    out = CASE_DIR / "interactive_graph_spatial.html"
    write_graph_html(fig, out)

    print(f"已生成 {out}  ({out.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
