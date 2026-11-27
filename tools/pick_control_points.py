"""道路平面标定：四点人工选取工具。

用法（在项目根目录执行）：
    python tools/pick_control_points.py outputs/ACCIDENT_001/original.jpg

操作：
    鼠标左键：依次点选 4 个控制点（顺序固定：near_left -> near_right -> far_right -> far_left）
    B：撤销上一个点
    R：全部重置
    S：保存（必须已选满 4 个点）
    Q / ESC：退出

说明：
- 四个点必须位于同一个道路平面（路面上），不能取车身角点、车顶、红绿灯等悬空物。
- 保存时若 outputs/<case>/image_control_points.json 已存在，会先备份为
  image_control_points.previous.json，再写入新的点。
- 本工具只保存图像坐标；道路平面坐标（ground_points）需要你在 calibration.json 中给定，
  并把单位明确写为 local_unit（没有实测尺寸时不得声称米制）。
- 支持中文路径（OpenCV 原生 imread/imwrite 不支持，本工具改用 imdecode/imencode）。
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

POINT_ORDER = ["near_left", "near_right", "far_right", "far_left"]
SUGGESTION = {
    "near_left": "画面下方，道路左侧路面/标线",
    "near_right": "画面下方，道路右侧路面/标线",
    "far_right": "画面上方，道路右侧路面/标线",
    "far_left": "画面上方，道路左侧路面/标线",
}
MAX_DISPLAY_W = 1280
MAX_DISPLAY_H = 800


def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"图片无法读取：{path}")
    return img


def imwrite_unicode(path, img):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise IOError(f"图片编码失败：{path}")
    buf.tofile(str(path))


def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            "请提供图片路径，例如：\n"
            "  python tools/pick_control_points.py "
            "outputs/ACCIDENT_001/original.jpg"
        )

    image_path = Path(sys.argv[1]).resolve()
    if not image_path.exists():
        raise SystemExit(f"图片不存在：{image_path}")

    img = imread_unicode(image_path)
    h, w = img.shape[:2]
    scale = min(MAX_DISPLAY_W / w, MAX_DISPLAY_H / h, 1.0)
    disp = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    points = []          # 原图坐标
    win = "pick_control_points  (L=add  B=undo  R=reset  S=save  Q=quit)"

    def redraw():
        canvas = disp.copy()
        for i, (x, y) in enumerate(points):
            px, py = int(x * scale), int(y * scale)
            cv2.circle(canvas, (px, py), 5, (0, 0, 255), -1)
            cv2.putText(canvas, f"{i + 1} {POINT_ORDER[i]}", (px + 8, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            if i > 0:
                qx, qy = int(points[i - 1][0] * scale), int(points[i - 1][1] * scale)
                cv2.line(canvas, (qx, qy), (px, py), (0, 255, 0), 1)
        if len(points) == 4:
            pts = np.array([[[int(x * scale), int(y * scale)] for x, y in points]],
                           dtype=np.int32)
            cv2.polylines(canvas, pts, True, (0, 255, 0), 2)
        hint = (f"next: {POINT_ORDER[len(points)]} ({SUGGESTION[POINT_ORDER[len(points)]]})"
                if len(points) < 4 else "4 points OK -> press S to save")
        cv2.putText(canvas, hint, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (255, 255, 255), 2)
        cv2.putText(canvas, hint, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (0, 0, 0), 1)
        cv2.imshow(win, canvas)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((round(x / scale, 2), round(y / scale, 2)))
            redraw()

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("b") and points:
            points.pop()
            redraw()
        elif key == ord("r"):
            points.clear()
            redraw()
        elif key == ord("s"):
            if len(points) != 4:
                print("需要 4 个点才能保存，当前：", len(points))
                continue
            out_path = image_path.parent / "image_control_points.json"
            if out_path.exists():
                backup = image_path.parent / "image_control_points.previous.json"
                backup.write_text(out_path.read_text(encoding="utf-8"),
                                  encoding="utf-8")
                print("已备份原文件 ->", backup.name)
            data = {
                "image_path": str(image_path),
                "image_width": w,
                "image_height": h,
                "point_order": POINT_ORDER,
                "image_points": [[int(round(x)), int(round(y))] for x, y in points],
                "selection_note": "由 tools/pick_control_points.py 人工点选保存；"
                                  "四个点须严格位于同一道路平面。",
            }
            out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                encoding="utf-8")
            print("已保存 ->", out_path)
            print("下一步：把这些坐标填进 calibration.json 的 image_points，"
                  "并给出对应的 ground_points（单位写 local_unit）。")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
