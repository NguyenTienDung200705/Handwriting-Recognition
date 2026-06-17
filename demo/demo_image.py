import cv2
from src.detection.yolo_detect import load_yolo_model, detect_and_visualize
from src.ocr.predict import load_ocr_model, predict_text
from src.ocr.dataset import OCRDataset


# ===== LOAD YOLO =====
yolo_model = load_yolo_model(r"D:\LAB\Handwriting Recognition\models\yolo\best (1).pt")

# ===== LOAD DATASET (để lấy charset) =====
dataset = OCRDataset("data/images", "data/labels")
char_list = dataset.CHARS

# ===== LOAD OCR MODEL =====
ocr_model = load_ocr_model(
    r"D:\LAB\Handwriting Recognition\models\crnn\model_crnn_text.pth",
    num_classes=len(char_list) + 1
)

# ===== LOAD IMAGE =====
img = cv2.imread(r"D:\LAB\Handwriting Recognition\data\images\h6.jpg")

# ===== DETECT =====
img_draw, lines, crops = detect_and_visualize(img, yolo_model)

# ===== OCR =====
final_text = ""

idx = 0
for line in lines:
    for box in line:
        crop = crops[idx]
        text = predict_text(crop, ocr_model, char_list)
        final_text += text + " "
        idx += 1
    final_text += "\n"

# ===== PRINT RESULT =====
print("\n===== OCR RESULT =====")
print(final_text)