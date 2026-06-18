# 🖋️ Vietnamese Handwritten Text Recognition

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Research-yellow?style=for-the-badge)

**Nghiên cứu hiệu quả các kiến trúc học sâu trong bài toán nhận diện văn bản tiếng Việt**

*Môn Đồ án Kỹ thuật Điện tử và Tin học — Đại học Khoa học Tự nhiên, ĐHQGHN*

[📄 Báo cáo](#) · [📊 Kết quả](#kết-quả-thực-nghiệm) · [🚀 Hướng dẫn sử dụng](#cài-đặt-và-sử-dụng)

</div>

---

## 📋 Mục lục

- [Giới thiệu](#giới-thiệu)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Mô hình và phương pháp](#mô-hình-và-phương-pháp)
- [Tập dữ liệu](#tập-dữ-liệu)
- [Kết quả thực nghiệm](#kết-quả-thực-nghiệm)
- [Cài đặt và sử dụng](#cài-đặt-và-sử-dụng)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Hạn chế và hướng phát triển](#hạn-chế-và-hướng-phát-triển)
- [Tài liệu tham khảo](#tài-liệu-tham-khảo)
- [Tác giả](#tác-giả)

---

## Giới thiệu

Trong bối cảnh chuyển đổi số, việc số hóa tài liệu viết tay tiếng Việt là một bài toán quan trọng nhưng đầy thách thức. Tiếng Việt có hệ thống **dấu thanh và dấu phụ phong phú**, độ biến thiên cao về phong cách chữ viết, và sự đa dạng trong nét ký tự — tất cả làm tăng đáng kể độ phức tạp của bài toán nhận diện so với các ngôn ngữ khác.

Project này nghiên cứu và so sánh hiệu quả các kiến trúc học sâu phổ biến trong bài toán **Handwritten Text Recognition (HTR)** cho tiếng Việt, đề xuất pipeline hai giai đoạn sử dụng **YOLO26** để phát hiện dòng chữ và **CRNN + CTC** để nhận diện nội dung.

### Điểm nổi bật

- ✅ Pipeline end-to-end: từ ảnh tài liệu thô đến văn bản số hóa
- ✅ So sánh 6 kiến trúc backbone khác nhau (VGG16/19, ResNet34/50, EfficientNetB0)
- ✅ Tiền xử lý nâng cao: Deskew tự động bằng Hough Transform
- ✅ Tối ưu cho chữ tiếng Việt với dấu phụ phức tạp
- ✅ Đạt **Char Acc 88.8%** với cấu hình CRNN + ResNet50

---

## Kiến trúc hệ thống

Hệ thống được thiết kế theo hướng **two-stage pipeline**:

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT IMAGE                          │
│              (Ảnh tài liệu viết tay gốc)                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     MODULE 1: DETECTION                     │
│                         YOLO26                              │
│  Resize → Backbone → FPN Neck → Regression Head → Crop     │
└──────────────────────────┬──────────────────────────────────┘
                           │  Cropped line images
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  PREPROCESSING (Deskew)                     │
│      Binary → Canny Edge → Hough Transform → Rotate        │
└──────────────────────────┬──────────────────────────────────┘
                           │  Aligned line images
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   MODULE 2: RECOGNITION                     │
│                   CRNN + CTC Loss                           │
│   CNN Features → BiLSTM Sequence → Softmax → CTC Decode    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
                    📝 OUTPUT TEXT
```

---

## Mô hình và phương pháp

### Module 1 — Phát hiện dòng chữ (YOLO26)

**YOLO26** là thế hệ mới nhất của họ YOLO do Ultralytics phát triển, được tối ưu đặc biệt cho edge devices với các cải tiến:

| Cải tiến | Mô tả |
|---|---|
| **NMS-Free** | Loại bỏ bước hậu xử lý Non-Maximum Suppression, giảm độ trễ |
| **No DFL** | Loại bỏ Distribution Focal Loss, dễ dàng export sang TensorRT/ONNX |
| **STAL** | Small-Target-Aware Label Assignment — tăng nhạy cảm với dòng chữ nhỏ |
| **MuSGD** | Optimizer kết hợp SGD + Muon, hội tụ nhanh và ổn định hơn |
| **ProgLoss** | Progressive Loss Balancing — cân bằng động giữa classification và regression loss |

**Hàm mất mát:**

$$L_{total}(t) = \lambda_{cls}(t) \cdot L_{cls} + \lambda_{box}(t) \cdot L_{box}$$

Trong đó $L_{box}$ sử dụng **CIoU Loss** và $L_{cls}$ dùng **Soft Target** thông qua cơ chế STAL.

---

### Module 2 — Nhận diện chữ viết tay (CRNN)

Kiến trúc **CRNN (Convolutional Recurrent Neural Network)** kết hợp:

```
Input Image (H×W×1)
       │
       ▼
┌─────────────┐
│  CNN Block  │  ← Trích xuất đặc trưng không gian
│  (Backbone) │    (nét chữ, hình dạng, cấu trúc ký tự)
└──────┬──────┘
       │ Feature Sequence
       ▼
┌─────────────┐
│   BiLSTM   │  ← Mô hình hóa ngữ cảnh hai chiều
│  (2 layers) │    (forward + backward pass)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  FC + Softmax│ ← Phân loại ký tự tại mỗi time step
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  CTC Decode │  ← Loại bỏ blank và ký tự lặp
└──────┬──────┘
       │
       ▼
   Output Text
```

**Hàm mất mát CTC:**

$$L_{CTC} = -\log P(y|x) = -\log \sum_{\pi \in \mathcal{B}^{-1}(y)} \prod_{t=1}^{T} p_{\pi_t}^{(t)}$$

**Tiền xử lý Deskew (Hough Transform):**

$$\rho = x\cos\theta + y\sin\theta$$

Ảnh được chỉnh thẳng tự động bằng cách tìm góc nghiêng $\theta_{peak}$ qua không gian Hough và xoay bù trừ $-\theta_{peak}$.

---

## Tập dữ liệu

### Detection Dataset (YOLO26)

| Tập | Số ảnh | Số nhãn |
|---|---|---|
| Train | 506 | 506 |
| Validation | 102 | 102 |
| Test | 52 | 52 |
| **Tổng** | **660** | **660** |

- Nguồn: Thu thập từ nhiều nguồn thực tế, đa dạng bố cục và góc chụp
- Gán nhãn: Line-level bounding boxes (PaddleOCR + Manual fix)
- Quy trình: Automatic detection → Quality filter → Human review → Refinement loop

### Recognition Dataset (CRNN)

| Tập | Số ảnh | Số nhãn |
|---|---|---|
| Train | 1,534 | 1,534 |
| Validation | 285 | 285 |
| Test | 94 | 94 |
| **Tổng** | **1,913** | **1,913** |

- Tỉ lệ phân chia: 80% / 15% / 5%
- Ảnh crop từ Module 1, mỗi ảnh chứa một dòng chữ viết tay duy nhất
- Gán nhãn thủ công bằng labeling tool tự phát triển

---

## Kết quả thực nghiệm

### So sánh các Backbone trên CRNN

| Model | Char Acc (%) ↑ | CER (%) ↓ | Word Acc (%) ↑ | Seq Acc (%) ↑ |
|---|:---:|:---:|:---:|:---:|
| CRNN (baseline) | 82.8 | 17.2 | 68.5 | 55.2 |
| CRNN + VGG16 | 85.4 | 14.6 | 74.2 | 58.7 |
| CRNN + VGG19 | 83.6 | 16.4 | 72.1 | 56.8 |
| CRNN + ResNet34 | 82.9 | 17.1 | 68.9 | 55.3 |
| **CRNN + ResNet50** | **88.8** | **11.2** | **77.3** | **61.5** |
| CRNN + EfficientNetB0 | 84.7 | 15.3 | 72.8 | 58.1 |

> 🏆 **CRNN + ResNet50** đạt hiệu năng tốt nhất trên toàn bộ chỉ số đánh giá nhờ cơ chế **residual learning** giúp học đặc trưng sâu mà không bị vanishing gradient.

### Phân tích kết quả

- **ResNet50** vượt trội nhờ Bottleneck blocks — dung hòa giữa độ sâu mạng và ổn định tối ưu
- **VGG19** thua VGG16 do overfitting khi thiếu skip connections
- **ResNet34** (Basic Block) không đủ capacity để phân biệt ký tự tiếng Việt tương đồng
- **EfficientNetB0** cho kết quả khá tốt xét về hiệu quả tham số

---

## Cài đặt và sử dụng

### Yêu cầu hệ thống

```
Python >= 3.8
CUDA >= 11.8 (khuyến nghị)
GPU >= 4GB VRAM
```

### Cài đặt

```bash
# Clone repository
git clone https://github.com/NguyenTienDung200705/Handwriting-Recognition.git

# Tạo môi trường ảo
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### requirements.txt

```
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
albumentations>=1.3.0
paddleocr>=2.7.0
tqdm>=4.65.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
```

### Chuẩn bị dữ liệu

```bash
# Tổ chức dữ liệu detection
data/
├── detection/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/          # YOLO format (.txt)
│       ├── train/
│       ├── val/
│       └── test/
└── recognition/
    ├── train/            # line images
    ├── val/
    ├── test/
    └── labels.txt        # format: image_path\ttext_label
```

### Huấn luyện

```bash
# Module 1: Train YOLO26 (Line Detection)
python train_detection.py \
    --data data/detection/data.yaml \
    --model yolo26n \
    --epochs 100 \
    --imgsz 640 \
    --batch 16

# Module 2: Train CRNN (Text Recognition)
python train_recognition.py \
    --data_dir data/recognition \
    --backbone resnet50 \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-3 \
    --hidden_size 256 \
    --lstm_layers 2
```

### Inference

```bash
# Full pipeline: ảnh tài liệu → văn bản
python predict.py \
    --image path/to/document.jpg \
    --det_model weights/detection/best.pt \
    --rec_model weights/recognition/best.pth \
    --output output/result.txt

# Chỉ nhận diện (dùng ảnh crop sẵn)
python predict_line.py \
    --image path/to/line.jpg \
    --model weights/recognition/best.pth
```

### Đánh giá

```bash
# Đánh giá mô hình nhận diện
python evaluate.py \
    --data_dir data/recognition/test \
    --model weights/recognition/best.pth \
    --backbone resnet50
```

---

## Cấu trúc thư mục

```
vietnamese-htr/
│
├── 📁 data/                    # Dữ liệu (không đưa lên git)
│   ├── detection/
│   └── recognition/
│
├── 📁 models/                  # Định nghĩa kiến trúc
│   ├── crnn.py                 # CRNN model
│   ├── backbone.py             # VGG, ResNet, EfficientNet backbones
│   └── ctc_decoder.py          # CTC beam search decoder
│
├── 📁 datasets/                # Dataset classes & augmentation
│   ├── detection_dataset.py
│   └── recognition_dataset.py
│
├── 📁 utils/                   # Tiện ích
│   ├── deskew.py               # Hough Transform deskew
│   ├── preprocess.py           # Tiền xử lý ảnh
│   ├── metrics.py              # CER, WER, SeqAcc tính toán
│   └── visualize.py            # Trực quan hóa kết quả
│
├── 📁 configs/                 # File cấu hình
│   ├── detection.yaml
│   └── recognition.yaml
│
├── 📁 weights/                 # Model weights (không đưa lên git)
│   ├── detection/
│   └── recognition/
│
├── train_detection.py
├── train_recognition.py
├── predict.py
├── evaluate.py
├── requirements.txt
└── README.md
```

---

## Chỉ số đánh giá

### Detection

| Chỉ số | Công thức | Ý nghĩa |
|---|---|---|
| **IoU** | $\frac{Area(B_{pred} \cap B_{gt})}{Area(B_{pred} \cup B_{gt})}$ | Độ trùng khớp bounding box |
| **Precision** | $\frac{TP}{TP+FP}$ | Tỉ lệ dự đoán đúng |
| **Recall** | $\frac{TP}{TP+FN}$ | Tỉ lệ phát hiện đúng |
| **mAP** | $\frac{1}{C}\sum_{c=1}^{C} AP_c$ | Trung bình AP trên tất cả lớp |

### Recognition

| Chỉ số | Công thức | Ý nghĩa |
|---|---|---|
| **Char Acc** | $\frac{N_{char\_correct}}{N_{char\_total}} \times 100\%$ | Độ chính xác ký tự |
| **CER** | $\frac{S+D+I}{N} \times 100\%$ | Tỉ lệ lỗi ký tự (Levenshtein) |
| **Word Acc** | $\frac{N_{word\_correct}}{N_{word\_total}} \times 100\%$ | Độ chính xác từ |
| **Seq Acc** | $\frac{N_{seq\_correct}}{N_{seq\_total}} \times 100\%$ | Độ chính xác chuỗi đầy đủ |

---

## Hạn chế và hướng phát triển

### Hạn chế hiện tại

- Hiệu năng giảm với chữ viết tháu, dính nét hoặc biến dạng cực đoan
- Tập dữ liệu còn hạn chế về quy mô và độ đa dạng
- Chưa tích hợp Language Model cho hậu xử lý ngữ cảnh

### Hướng phát triển

- [ ] **Mở rộng dataset** — thu thập thêm mẫu chữ khó, điều kiện chụp thực tế (ánh sáng kém, giấy nhăn, mực nhòe)
- [ ] **Language Model Integration** — kết hợp n-gram hoặc BERT tiếng Việt để sửa lỗi chính tả theo ngữ cảnh
- [ ] **Transformer-based Recognition** — thử nghiệm TrOCR/ViT thay thế CRNN
- [ ] **Attention Mechanism** — bổ sung CBAM hoặc self-attention vào CNN backbone
- [ ] **Data Augmentation** — elastic distortion, random perspective để tăng robustness
- [ ] **Quantization** — INT8/FP16 để triển khai trên thiết bị nhúng

---

## Tài liệu tham khảo

1. Shi, B., Bai, X., & Yao, C. (2017). *An End-to-End Trainable Neural Network for Image-Based Sequence Recognition*. IEEE TPAMI.
2. Graves, A. et al. (2006). *Connectionist Temporal Classification*. ICML.
3. He, K. et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
4. Redmon, J. et al. (2016). *You Only Look Once: Unified, Real-Time Object Detection*. CVPR.
5. Sapkota, R. et al. (2025). *YOLO26: Key Architectural Enhancements for Real-Time Object Detection*. arXiv:2509.25164.
6. Simonyan, K. & Zisserman, A. (2014). *Very Deep Convolutional Networks*. arXiv:1409.1556.
7. Tan, M. & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for CNNs*. ICML.

---

## Tác giả

<div align="center">

**Nguyễn Tiến Dũng**

Khoa Vật Lý — Ngành Kỹ thuật Điện tử và Tin học

Trường Đại học Khoa học Tự nhiên, ĐHQG Hà Nội

📧 Email liên hệ : nguyentiendung200705@gmail.com

**Giảng viên hướng dẫn:**<br>
TS. Nguyễn Tiến Cường<br>
CN. Vi Anh Quân

---

*Môn Đồ án Kỹ thuật Điện tử và Tin học — Hà Nội, 2026*

</div>

---
