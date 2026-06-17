import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image


# ===== TRANSFORM (GIỐNG TRAIN) =====
transform = T.Compose([
    T.Grayscale(num_output_channels=1),
    T.Resize(32),              # chiều cao = 32 (giống dataset)
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])


# ===== PREPROCESS =====
def preprocess(img, img_width=640):
    """
    Chuẩn hóa ảnh trước khi đưa vào CRNN

    Args:
        img: numpy image (BGR từ cv2)
        img_width: width cố định (giống training)

    Returns:
        tensor: (1, H, W)
    """

    # ===== numpy -> PIL =====
    img = Image.fromarray(img)

    # ===== transform =====
    img = transform(img)

    # ===== padding width =====
    c, h, w = img.shape

    if w < img_width:
        pad = img_width - w
        img = F.pad(img, (0, pad), value=1.0)  # padding trắng
    else:
        img = img[:, :, :img_width]

    return img