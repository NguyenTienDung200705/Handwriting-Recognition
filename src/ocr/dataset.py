import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class OCRDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_height=32, img_width=640, augment=False,
                     charset=("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                              "ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặ"
                              "ẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ ")):

        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_height = img_height
        self.img_width = img_width
        self.augment = augment

        # charset
        self.CHARS = charset
        self.CHAR2LABEL = {c: i+1 for i, c in enumerate(self.CHARS)}  # 0 = blank
        self.LABEL2CHAR = {i+1: c for i, c in enumerate(self.CHARS)}

        self.LABEL2CHAR[0] = '<BLANK>'

        # load samples
        self.samples = []
        for fname in os.listdir(img_dir):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                base = os.path.splitext(fname)[0]
                img_path = os.path.join(img_dir, fname)
                label_path = os.path.join(label_dir, base + ".txt")

                if os.path.exists(label_path):
                    self.samples.append((img_path, label_path))

        # transform
        if self.augment:
            self.transform = T.Compose([
                T.Grayscale(num_output_channels=1),
                T.Resize(self.img_height),

                # augmentation
                T.RandomRotation(2),
                T.RandomAffine(
                    degrees=2,
                    translate=(0.02, 0.02),
                    scale=(0.95, 1.05)
                ),
                T.ColorJitter(
                    brightness=0.3,
                    contrast=0.3
                ),

                T.ToTensor(),
                T.Normalize((0.5,), (0.5,))
            ])
        else:
            self.transform = T.Compose([
                T.Grayscale(num_output_channels=1),
                T.Resize(self.img_height),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,))
            ])

    def encode_label(self, text):
        return [self.CHAR2LABEL[c] for c in text if c in self.CHAR2LABEL]

    def decode_label(self, labels):
        return ''.join([self.LABEL2CHAR[l] for l in labels if l in self.LABEL2CHAR])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        # load image
        img = Image.open(img_path).convert("L")
        img = self.transform(img)

        # padding width
        c, h, w = img.shape

        if w < self.img_width:
            pad = self.img_width - w
            img = F.pad(img, (0, pad), value=1.0)
        else:
            img = img[:, :, :self.img_width]

        # load label
        with open(label_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        label = torch.tensor(self.encode_label(text), dtype=torch.long)

        return img, label, text
    

# ===== COLLATE FUNCTION =====
def make_ocr_collate_fn(seq_len):
    def ocr_collate_fn(batch):
        imgs, labels, texts = zip(*batch)
        imgs = torch.stack(imgs, dim=0)

        label_lengths = [len(l) for l in labels]
        labels_concat = torch.cat(labels, dim=0) if len(labels) > 0 else torch.tensor([], dtype=torch.long)

        batch_size = imgs.size(0)
        input_lengths = torch.full(size=(batch_size,), fill_value=seq_len, dtype=torch.long)

        return imgs, labels_concat, input_lengths, torch.tensor(label_lengths, dtype=torch.long), texts
    return ocr_collate_fn