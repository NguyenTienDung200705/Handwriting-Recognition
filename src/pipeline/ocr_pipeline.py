import cv2
import yaml

from src.detection.yolo_detect import load_yolo_model, detect_and_visualize
from src.ocr.predict import load_ocr_model, predict_text
from src.ocr.dataset import OCRDataset


class OCRPipeline:
    def __init__(self, config_path="config/config.yaml"):

        # ===== LOAD CONFIG =====
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.device = self.cfg["project"]["device"]

        # ===== LOAD YOLO =====
        self.yolo_model = load_yolo_model(
            self.cfg["yolo"]["model_path"]
        )

        # ===== LOAD DATASET (lấy charset) =====
        self.dataset = OCRDataset(
            self.cfg["dataset"]["image_dir"],
            self.cfg["dataset"]["label_dir"]
        )
        self.char_list = self.dataset.CHARS

        # ===== LOAD OCR MODEL =====
        self.ocr_model = load_ocr_model(
            self.cfg["ocr"]["model_path"],
            num_classes=len(self.char_list) + 1,
            device=self.device
        )

    # ===== RUN FROM FILE PATH =====
    def run(self, image_path):

        img = cv2.imread(image_path)

        if img is None:
            raise ValueError(f"Không đọc được ảnh: {image_path}")

        return self._process(img)

    # ===== RUN FROM NUMPY (CHO WEB) =====
    def run_from_array(self, img):

        if img is None:
            raise ValueError("Ảnh đầu vào bị None")

        return self._process(img)

    # ===== CORE PIPELINE =====
    def _process(self, img):

        # ===== DETECT + SORT =====
        img_draw, lines, crops = detect_and_visualize(
            img,
            self.yolo_model
        )

        # ===== OCR =====
        final_text = ""
        idx = 0

        for line in lines:
            for box in line:
                crop = crops[idx]

                text = predict_text(
                    crop,
                    self.ocr_model,
                    self.char_list,
                    device=self.device,
                    beam_size=self.cfg["ocr"]["beam_size"]
                )

                final_text += text + " "
                idx += 1

            final_text += "\n"

        return img_draw, final_text


# ===== RUN TEST =====
if __name__ == "__main__":

    pipeline = OCRPipeline("config/config.yaml")

    img_draw, text = pipeline.run(
        pipeline.cfg["demo"]["image_path"]
    )

    print("\n===== OCR RESULT =====")
    print(text)

    # show image
    cv2.imshow("OCR Result", img_draw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()