import cv2
from ultralytics import YOLO
from .utils import sort_and_group_boxes


# ===== LOAD MODEL =====
def load_yolo_model(model_path):
    return YOLO(model_path)


# ===== DETECT TEXT =====
def detect_text(img, model, conf=0.5, imgsz=1024):
    results = model.predict(source=img, conf=conf, imgsz=imgsz, verbose=False)

    boxes = []

    if len(results) == 0 or results[0].boxes is None:
        return boxes

    for b in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, b.tolist())
        boxes.append([x1, y1, x2, y2])

    return boxes


# ===== DETECT + SORT + VISUALIZE =====
def detect_and_visualize(img, model, conf=0.5):
    """
    Returns:
        img_draw: ảnh có bbox (1 ảnh duy nhất)
        lines: list các dòng (đã sort)
        crops: list ảnh crop theo thứ tự đọc (KHÔNG hiển thị)
    """

    boxes = detect_text(img, model, conf)

    # sort theo layout
    lines = sort_and_group_boxes(boxes)

    img_draw = img.copy()
    crops = []

    count = 0

    for line in lines:
        for box in line:
            x1, y1, x2, y2 = box

            # ===== VẼ BBOX =====
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (255, 0, 255), 2)

            # ===== ĐÁNH SỐ THỨ TỰ =====
            cv2.putText(img_draw, str(count),
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)

            # ===== CROP (KHÔNG HIỂN THỊ) =====
            crop = img[y1:y2, x1:x2]
            crops.append(crop)

            count += 1

    return img_draw, lines, crops