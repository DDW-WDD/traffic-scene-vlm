"""通过 GitHub Contents API 把本地产物推送到 Pages 仓库。

为什么不用 git push：本机直连 github.com:443 会被 reset，全局 socks 代理不可用；
Contents API 走 HTTPS 且每次都是独立请求，可在这种网络环境下稳定工作。

用法（PowerShell）：
    $env:GITHUB_TOKEN='ghp_xxx'
    python push_github.py            # 推送默认清单
    python push_github.py --check    # 只看哪些文件有差异，不写
    python push_github.py --only index.html README.md

说明：
- 单文件上限 100 MB（Contents API 限制），本项目最大文件是 plotly.min.js（约 4.7 MB），没有压力；
- 每个文件先 GET 取现有 sha 再 PUT，避免 409；
- 失败自动指数退避重试。
"""

import argparse
import base64
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

OWNER = "DDW-WDD"
REPO = "traffic-scene-vlm"
BRANCH = "main"
API = f"https://api.github.com/repos/{OWNER}/{REPO}/contents"

ROOT = Path(__file__).resolve().parent
CASE = "outputs/ACCIDENT_001"

DEFAULT_FILES = [
    # 页面与文档
    "index.html",
    "README.md",
    "第四阶段_使用说明.md",
    # 代码
    "run_stage4.py",
    "build_site.py",
    "rebuild_graph_html.py",
    "modules/road_plane_projection.py",
    "modules/road_geometry.py",
    "modules/scene_topdown.py",
    "modules/scene_interactive.py",
    "modules/calibration_check.py",
    "modules/knowledge_graph.py",
    "modules/interactive_graph.py",
    # 数据与图
    f"{CASE}/road_geometry.json",
    f"{CASE}/road_geometry_local.json",
    f"{CASE}/scene_model_spatial.json",
    f"{CASE}/ground_distances.json",
    f"{CASE}/spatial_model_info.json",
    f"{CASE}/calibration_check.json",
    f"{CASE}/scene_topdown.png",
    f"{CASE}/road_surface_topdown.png",
    f"{CASE}/scene_base_topdown.png",
    f"{CASE}/scene_topdown.html",
    f"{CASE}/plotly.min.js",
    f"{CASE}/knowledge_graph_spatial.json",
    f"{CASE}/knowledge_graph_spatial.png",
    f"{CASE}/interactive_graph_spatial.html",
]


def request(method, url, token, payload=None, retries=5):
    """带重试的请求：本机网络对该 API 的大响应偶尔会 IncompleteRead。"""
    data = json.dumps(payload).encode("utf-8") if payload else None
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("User-Agent", "doubao-stage4-pusher")
        req.add_header("Accept", "application/vnd.github+json")
        if data:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                raw = resp.read()
                return resp.status, json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError:
            raise                                  # 4xx/5xx 交给上层判断
        except (urllib.error.URLError, http.client.IncompleteRead,
                http.client.HTTPException, TimeoutError, ConnectionError, OSError) as exc:
            last = exc
            time.sleep(1.5 + attempt * 1.5)
    raise RuntimeError(f"网络读取重试 {retries} 次仍失败：{last}")


def tree_index(token):
    """一次请求拿到整棵树（path -> (sha, size)），比逐文件 GET 稳，也省配额。"""
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/git/trees/{BRANCH}?recursive=1"
    _, body = request("GET", url, token)
    return {item["path"]: (item["sha"], item.get("size")) for item in body.get("tree", [])}


def remote_sha(path, token):
    url = f"{API}/{urllib.parse.quote(path)}?ref={BRANCH}"
    try:
        _, body = request("GET", url, token)
        return body.get("sha"), body.get("size")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None, None
        raise


def push_one(path, token, check_only=False, message=None, index=None):
    local = ROOT / path
    if not local.exists():
        return "missing", None
    size = local.stat().st_size
    entry = (index or {}).get(path)
    if entry:
        sha, remote_size = entry
    else:
        sha, remote_size = remote_sha(path, token)

    if check_only:
        state = "same" if remote_size == size else ("new" if sha is None else "changed")
        return state, size

    content_b64 = base64.b64encode(local.read_bytes()).decode("ascii")
    payload = {
        "message": message or f"stage4 upgrade: {path}",
        "content": content_b64,
        "branch": BRANCH,
    }
    if sha:
        payload["sha"] = sha

    url = f"{API}/{urllib.parse.quote(path)}"
    for attempt in range(6):
        try:
            status, body = request("PUT", url, token, payload)
            return "pushed", body.get("content", {}).get("size")
        except urllib.error.HTTPError as e:
            if e.code in (409, 422) and attempt == 0:
                # sha 过期/冲突：重新取一次 sha 再试
                sha, _ = remote_sha(path, token)
                if sha:
                    payload["sha"] = sha
                time.sleep(1)
                continue
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"{path}: HTTP {e.code} {e.read()[:200]!r}")
    raise RuntimeError(f"{path}: 重试 6 次仍失败")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--message")
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("请先设置环境变量 GITHUB_TOKEN")
        return 2

    files = args.only or DEFAULT_FILES
    try:
        index = tree_index(token)
    except Exception as exc:                           # noqa: BLE001
        print(f"无法读取远端目录树（{exc}），改为逐文件查询 sha")
        index = {}

    ok, failed = 0, []
    for i, path in enumerate(files, 1):
        try:
            state, size = push_one(path, token, check_only=args.check,
                                   message=args.message, index=index)
            ok += 1
            shown = f"{size / 1024:.1f} KB" if size else "-"
            print(f"[{i:02d}/{len(files)}] {state:<8} {path}  {shown}")
        except Exception as exc:                       # noqa: BLE001
            failed.append(path)
            print(f"[{i:02d}/{len(files)}] FAIL     {path}  {exc}")
    print(f"\n完成 {ok}/{len(files)}；失败：{failed or '无'}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
