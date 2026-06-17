from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import torch
import torch.nn.functional as F
import yaml
import traceback

from src.detection.yolo_detect import detect_and_visualize
from src.ocr.preprocess import preprocess
from src.ocr.predict import decode_indices, predict_text

# Import tất cả model
from src.ocr.CRNN import CRNN_VGG16, CRNN_VGG19, CRNN

app = Flask(__name__)

# ====================== LOAD CONFIG ======================
config_path = "config/config.yaml"

try:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print("✅ Đọc config thành công")
except Exception as e:
    print(f"❌ Lỗi đọc config: {e}")
    raise

device = torch.device(config.get("project", {}).get("device", "cpu"))

# ====================== LOAD YOLO ======================
from ultralytics import YOLO
yolo_model = YOLO(config["yolo"]["model_path"])
print("✅ Loaded YOLO model")

# ====================== LOAD CHARSET ======================
from src.ocr.dataset import OCRDataset
temp_dataset = OCRDataset(
    img_dir=config["dataset"]["image_dir"],
    label_dir=config["dataset"]["label_dir"]
)
char_list = temp_dataset.CHARS
print(f"✅ Loaded charset: {len(char_list)} ký tự")

# ====================== LOAD OCR MODELS ======================
models = {}
model_charlists = {}   # Lưu charset riêng cho từng model

print("\n🔄 Đang load các model OCR...")

for model_name, model_cfg in config.get("ocr_models", {}).items():
    try:
        model_path = model_cfg["model_path"]
        backbone = model_cfg.get("backbone", "vgg16")

        # ====================== LOAD STATE_DICT TRƯỚC ======================
        state_dict = torch.load(model_path, map_location=device)

        # ====================== XỬ LÝ THEO BACKBONE ======================
        if backbone == "crnn_vgg16":
            # 🔥 Lấy num_classes từ model
            num_classes = state_dict["dense.weight"].shape[0]
            print(f"📦 {model_name}: num_classes từ model = {num_classes}")

            # init model đúng size
            model = CRNN_VGG16(1, 32, 640, num_classes)

            # load weight
            model.load_state_dict(state_dict, strict=True)

            # 🔥 FIX charset (cắt theo model)
            current_char_list = char_list[:num_classes - 1]

        elif backbone == "crnn_base":
            num_classes = len(char_list) + 1
            current_char_list = char_list

            model = CRNN_VGG16(1, 32, 640, num_classes)
            model.load_state_dict(state_dict, strict=True)

        elif backbone == "efficientNet":
            num_classes = len(char_list) + 1
            current_char_list = char_list

            model = CRNN_VGG16(1, 32, 640, num_classes)
            model.load_state_dict(state_dict, strict=True)
            
        elif backbone == "crnn_resnet50":
            num_classes = len(char_list) + 1
            current_char_list = char_list

            model = CRNN_VGG16(1, 32, 640, num_classes)
            model.load_state_dict(state_dict, strict=True)

        else:
            raise ValueError(f"Backbone không hợp lệ: {backbone}")

        # ====================== FINALIZE ======================
        model.to(device)
        model.eval()

        models[model_name] = model
        model_charlists[model_name] = current_char_list

        print(f"✅ Loaded {model_name} ({backbone}) | classes = {num_classes}")

    except Exception as e:
        print(f"❌ Load thất bại {model_name} ({backbone}): {e}")

# ====================== ROUTES ======================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    try:
        file = request.files["image"]
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img_draw, lines, _ = detect_and_visualize(img, yolo_model)

        _, buffer = cv2.imencode(".jpg", img_draw)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "image": img_base64,
            "message": f"✅ Đã detect {len(lines)} dòng văn bản"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ocr", methods=["POST"])
def ocr():
    try:
        arch = request.form.get("architecture")

        if not arch or arch not in models:
            return jsonify({
                "error": f"Model '{arch}' không tồn tại. Các model có: {list(models.keys())}"
            }), 400

        file = request.files.get("image")
        if not file:
            return jsonify({"error": "Không tìm thấy file ảnh"}), 400

        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        img_draw, lines, crops = detect_and_visualize(img, yolo_model)

        print(f"🔍 Detect: {len(lines)} dòng | {len(crops)} crops")

        ocr_lines = []
        crop_idx = 0
        model = models[arch]
        char_list_for_model = model_charlists.get(arch, char_list)

        model_cfg = config["ocr_models"].get(arch, {})
        use_beam = model_cfg.get("decoder", "ctc") == "beam"
        beam_size = model_cfg.get("beam_size", 10)

        print(f"🚀 Đang chạy OCR với model: {arch} | Decoder: {'Beam' if use_beam else 'CTC Greedy'}")

        for line_idx, line_boxes in enumerate(lines):
            line_parts = []

            for _ in line_boxes:
                if crop_idx >= len(crops):
                    break

                crop = crops[crop_idx]

                if use_beam:
                    text = predict_text(
                        crop, model, char_list_for_model,
                        device=device,
                        beam_size=beam_size
                    )
                else:
                    processed = preprocess(crop)
                    tensor = processed.unsqueeze(0).to(device)

                    with torch.no_grad():
                        outputs = model(tensor)

                    log_probs = F.log_softmax(outputs, dim=2)
                    emission = log_probs[:, 0, :].cpu().numpy()
                    best_path = np.argmax(emission, axis=1).tolist()

                    text = decode_indices(best_path, char_list_for_model)

                line_parts.append(text.strip() if text else "")
                crop_idx += 1

            full_line = " ".join(line_parts).strip()
            ocr_lines.append(full_line)

            # In ra terminal theo từng dòng - như bạn muốn
            print(f"Dòng {line_idx + 1}: {full_line[:150]}{'...' if len(full_line) > 150 else ''}")

        _, buffer = cv2.imencode(".jpg", img_draw)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "image": img_base64,
            "lines": ocr_lines
        })

    except Exception as e:
        print("❌ LỖI OCR:\n", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)