"""第二阶段 - 第一步：YOLO 独立检测冒烟测试。

运行方式：
    python test_yolo.py

脚本会加载同目录下的 yolo26n.pt 权重，对同目录的 test.jpg 进行目标检测，
逐个打印检测框的类别、置信度和像素坐标，并保存带检测框的 yolo_result.jpg。
"""

from pathlib import Path

import cv2
from ultralytics import YOLO

# 权重、测试图片和输出都基于脚本所在目录定位，不依赖运行时的工作目录
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "yolo26n.pt"
IMAGE_PATH = BASE_DIR / "test.jpg"
OUTPUT_PATH = BASE_DIR / "yolo_result.jpg"


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"未找到测试图片：{IMAGE_PATH}")

    model = YOLO(str(MODEL_PATH))
    results = model.predict(
        source=str(IMAGE_PATH),
        conf=0.25,
        imgsz=640,
    )

    result = results[0]
    print(f"共检测到 {len(result.boxes)} 个目标：")
    for box in result.boxes:
        cls_id = int(box.cls[0])
        class_name = result.names[cls_id]
        confidence = float(box.conf[0])
        x_min, y_min, x_max, y_max = [
            round(float(value), 1) for value in box.xyxy[0]
        ]
        print(
            f"  类别: {class_name:<10} 置信度: {confidence:.3f}"
            f"  bbox: [{x_min}, {y_min}, {x_max}, {y_max}]"
        )

    # result.plot() 返回画好检测框的 BGR numpy 数组，再用 OpenCV 写入文件
    annotated = result.plot()
    cv2.imwrite(str(OUTPUT_PATH), annotated)
    print(f"检测框图片已保存：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
