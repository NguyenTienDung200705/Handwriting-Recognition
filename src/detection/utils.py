import cv2


# ===== SORT + GROUP DÒNG =====
def sort_and_group_boxes(boxes, y_threshold=30):
    """
    Sort bbox theo:
    - trên → dưới
    - trái → phải trong từng dòng
    """

    # sort theo center y
    boxes = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2)

    lines = []
    current_line = []

    for box in boxes:
        if not current_line:
            current_line.append(box)
            continue

        box_center = (box[1] + box[3]) / 2
        line_center = (current_line[0][1] + current_line[0][3]) / 2

        if abs(box_center - line_center) < y_threshold:
            current_line.append(box)
        else:
            lines.append(current_line)
            current_line = [box]

    if current_line:
        lines.append(current_line)

    # sort trái → phải trong từng dòng
    for i in range(len(lines)):
        lines[i] = sorted(lines[i], key=lambda b: b[0])

    return lines


# ===== CROP BOX =====
def crop_box(img, box, pad=5):
    x1, y1, x2, y2 = box

    h, w = img.shape[:2]

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    return img[y1:y2, x1:x2]