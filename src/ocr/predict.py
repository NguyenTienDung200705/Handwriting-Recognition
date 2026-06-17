import torch
import torch.nn.functional as F

# from .CRNN import CRNN
from .decoder import BeamSearchDecoder
from .preprocess import preprocess


# ===== LOAD MODEL =====
def load_ocr_model(model_path, num_classes, device="cpu"):
    """
    Load model CRNN đã train
    """
    device = torch.device("cpu") 
    model = CRNN(
        img_channel=1,
        img_height=32,
        img_width=640,
        num_class=num_classes,
        pretrained=False  # load thì không cần pretrained
    )

    # load weights
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


# ===== DECODE LABEL =====
def decode_indices(indices, char_list):
    text = ""
    prev_idx = -1

    for idx in indices:
        if idx == 0:  # blank
            prev_idx = idx
            continue

        if idx == prev_idx:
            continue

        if idx - 1 < len(char_list):
            text += char_list[idx - 1]

        prev_idx = idx

    return text

# ===== PREDICT =====
# def predict_text(img, model, char_list, device="cpu", beam_size=10):
#     """
#     OCR 1 ảnh (crop từ YOLO)

#     Args:
#         img: numpy image (BGR)
#         char_list: dataset.CHARS
#     """

#     # ===== PREPROCESS =====
#     img = preprocess(img)

#     # (1, 1, H, W)
#     img = img.unsqueeze(0).to(device)

#     # ===== FORWARD =====
#     with torch.no_grad():
#         outputs = model(img)  # [T, B, C]

#     # ===== LOG SOFTMAX =====
#     log_probs = F.log_softmax(outputs, dim=2)

#     # ===== LẤY 1 SAMPLE =====
#     emission = log_probs[:, 0, :].cpu().numpy()  # [T, C]

#     # ===== BEAM SEARCH =====
#     decoder = BeamSearchDecoder(beam_size=beam_size, blank=0)
#     best_path = decoder.decode(emission)

#     # ===== DECODE TEXT =====
#     text = decode_indices(best_path, char_list)

#     return text

def predict_text(img, model, char_list, device="cpu", beam_size=10):

    device = torch.device("cpu")  # 🔥 ÉP CỨNG CPU

    # ===== PREPROCESS =====
    img = preprocess(img)

    # (1, 1, H, W)
    img = img.unsqueeze(0).to(device)

    # ===== FORWARD =====
    with torch.no_grad():
        outputs = model(img)

    log_probs = F.log_softmax(outputs, dim=2)
    emission = log_probs[:, 0, :].cpu().numpy()

    decoder = BeamSearchDecoder(beam_size=beam_size, blank=0)
    best_path = decoder.decode(emission)

    text = decode_indices(best_path, char_list)

    return text