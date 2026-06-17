import math
import torch
import torch.nn as nn
import torchvision.models as models


# ========================
# Positional Encoding
# ========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ========================
# CRNN VGG16
# ========================
class CRNN_VGG16(nn.Module):
    def __init__(self, img_channel, img_height, img_width, num_class,
                 map_to_seq_hidden=256, nhead=8, num_layers=4, leaky_relu=False, pretrained=True):

        super().__init__()

        self.cnn, (output_channel, output_height, output_width) = \
            self._vgg_backbone(img_channel, img_height, img_width, leaky_relu, pretrained)

        self.map_to_seq = nn.Linear(output_channel * output_height, map_to_seq_hidden)
        self.pos_encoder = PositionalEncoding(map_to_seq_hidden)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=map_to_seq_hidden,
            nhead=nhead,
            dim_feedforward=4 * map_to_seq_hidden,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(map_to_seq_hidden, num_class)

    def _vgg_backbone(self, img_channel, img_height, img_width, leaky_relu, pretrained):

        def conv_relu(in_c, out_c):
            layers = [
                nn.Conv2d(in_c, out_c, 3, 1, 1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
            ]
            return layers

        layers = []

        layers.extend(conv_relu(img_channel, 64))
        layers.extend(conv_relu(64, 64))
        layers.append(nn.MaxPool2d(2, 2))

        layers.extend(conv_relu(64, 128))
        layers.extend(conv_relu(128, 128))
        layers.append(nn.MaxPool2d(2, 2))

        layers.extend(conv_relu(128, 256))
        layers.extend(conv_relu(256, 256))
        layers.extend(conv_relu(256, 256))
        layers.append(nn.MaxPool2d((2, 1)))

        layers.extend(conv_relu(256, 512))
        layers.extend(conv_relu(512, 512))
        layers.extend(conv_relu(512, 512))
        layers.append(nn.MaxPool2d((2, 1)))

        layers.extend(conv_relu(512, 512))
        layers.extend(conv_relu(512, 512))
        layers.extend(conv_relu(512, 512))
        layers.append(nn.MaxPool2d((2, 1)))

        cnn = nn.Sequential(*layers)

        if pretrained:
            print("Đang tải VGG16_BN...")
            vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)

            custom_convs = [m for m in cnn.modules() if isinstance(m, nn.Conv2d)]
            pretrained_convs = [m for m in vgg.features.modules() if isinstance(m, nn.Conv2d)]

            custom_bns = [m for m in cnn.modules() if isinstance(m, nn.BatchNorm2d)]
            pretrained_bns = [m for m in vgg.features.modules() if isinstance(m, nn.BatchNorm2d)]

            for i, (c, p) in enumerate(zip(custom_convs, pretrained_convs)):
                if i == 0 and img_channel == 1:
                    c.weight.data = p.weight.data.sum(dim=1, keepdim=True)
                else:
                    c.weight.data = p.weight.data
                if c.bias is not None:
                    c.bias.data = p.bias.data

            for c, p in zip(custom_bns, pretrained_bns):
                c.weight.data = p.weight.data
                c.bias.data = p.bias.data
                c.running_mean.data = p.running_mean.data
                c.running_var.data = p.running_var.data

        with torch.no_grad():
            dummy = torch.zeros(1, img_channel, img_height, img_width)
            _, c, h, w = cnn(dummy).size()

        return cnn, (c, h, w)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()

        x = x.permute(0, 3, 1, 2).reshape(b, w, c * h)
        x = self.map_to_seq(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)

        x = self.dropout(x)
        x = self.dense(x)

        return x.permute(1, 0, 2)


# ========================
# CRNN VGG19
# ========================
class CRNN_VGG19(nn.Module):
    def __init__(self, img_channel, img_height, img_width, num_class,
                 map_to_seq_hidden=256, nhead=8, num_layers=4, leaky_relu=False, pretrained=True):

        super().__init__()

        self.cnn, (output_channel, output_height, output_width) = \
            self._vgg19_backbone(img_channel, img_height, img_width, leaky_relu, pretrained)

        self.map_to_seq = nn.Linear(output_channel * output_height, map_to_seq_hidden)
        self.pos_encoder = PositionalEncoding(map_to_seq_hidden)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=map_to_seq_hidden,
            nhead=nhead,
            dim_feedforward=4 * map_to_seq_hidden,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(map_to_seq_hidden, num_class)

    def _vgg19_backbone(self, img_channel, img_height, img_width, leaky_relu, pretrained):

        def conv_relu(in_c, out_c):
            return [
                nn.Conv2d(in_c, out_c, 3, 1, 1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
            ]

        layers = []

        layers.extend(conv_relu(img_channel, 64))
        layers.extend(conv_relu(64, 64))
        layers.append(nn.MaxPool2d(2, 2))

        layers.extend(conv_relu(64, 128))
        layers.extend(conv_relu(128, 128))
        layers.append(nn.MaxPool2d(2, 2))

        for _ in range(4):
            layers.extend(conv_relu(128 if _ == 0 else 256, 256))
        layers.append(nn.MaxPool2d((2, 1)))

        for _ in range(4):
            layers.extend(conv_relu(256 if _ == 0 else 512, 512))
        layers.append(nn.MaxPool2d((2, 1)))

        for _ in range(4):
            layers.extend(conv_relu(512, 512))
        layers.append(nn.MaxPool2d((2, 1)))

        cnn = nn.Sequential(*layers)

        if pretrained:
            print("Đang tải VGG19_BN...")
            vgg = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT)

            custom_convs = [m for m in cnn.modules() if isinstance(m, nn.Conv2d)]
            pretrained_convs = [m for m in vgg.features.modules() if isinstance(m, nn.Conv2d)]

            custom_bns = [m for m in cnn.modules() if isinstance(m, nn.BatchNorm2d)]
            pretrained_bns = [m for m in vgg.features.modules() if isinstance(m, nn.BatchNorm2d)]

            for i, (c, p) in enumerate(zip(custom_convs, pretrained_convs)):
                if i == 0 and img_channel == 1:
                    c.weight.data = p.weight.data.sum(dim=1, keepdim=True)
                else:
                    c.weight.data = p.weight.data
                if c.bias is not None:
                    c.bias.data = p.bias.data

            for c, p in zip(custom_bns, pretrained_bns):
                c.weight.data = p.weight.data
                c.bias.data = p.bias.data
                c.running_mean.data = p.running_mean.data
                c.running_var.data = p.running_var.data

        with torch.no_grad():
            dummy = torch.zeros(1, img_channel, img_height, img_width)
            _, c, h, w = cnn(dummy).size()

        return cnn, (c, h, w)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()

        x = x.permute(0, 3, 1, 2).reshape(b, w, c * h)
        x = self.map_to_seq(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)

        x = self.dropout(x)
        x = self.dense(x)

        return x.permute(1, 0, 2)


# class CRNN(nn.Module):
#     def __init__(self, img_channel, img_height, img_width, num_class,
#                  map_to_seq_hidden=256, nhead=8, num_layers=4, leaky_relu=False):

#         super(CRNN, self).__init__()

#         # CNN backbone
#         self.cnn, (output_channel, output_height, output_width) = \
#             self._vgg_backbone(img_channel, img_height, img_width, leaky_relu)

#         # CNN -> sequence
#         self.map_to_seq = nn.Linear(output_channel * output_height, map_to_seq_hidden)

#         # positional encoding
#         self.pos_encoder = PositionalEncoding(map_to_seq_hidden)

#         # Transformer encoder
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=map_to_seq_hidden,
#             nhead=nhead,
#             dim_feedforward=4 * map_to_seq_hidden,
#             dropout=0.1,
#             batch_first=True,
#             activation='gelu'
#         )

#         self.transformer = nn.TransformerEncoder(
#             encoder_layer,
#             num_layers=num_layers
#         )

#         # classifier
#         self.dense = nn.Linear(map_to_seq_hidden, num_class)

#     def _vgg_backbone(self, img_channel, img_height, img_width, leaky_relu):

#         def conv_block(in_c, out_c, num_convs):
#             layers = []
#             for i in range(num_convs):
#                 layers.append(nn.Conv2d(in_c, out_c, 3, 1, 1))
#                 layers.append(nn.BatchNorm2d(out_c))

#                 if leaky_relu:
#                     layers.append(nn.LeakyReLU(0.2, inplace=True))
#                 else:
#                     layers.append(nn.ReLU(inplace=True))

#                 in_c = out_c

#             return nn.Sequential(*layers)

#         cnn = nn.Sequential(

#             conv_block(img_channel, 64, 2),
#             nn.MaxPool2d(2, 2),

#             conv_block(64, 128, 2),
#             nn.MaxPool2d(2, 2),

#             conv_block(128, 256, 3),
#             nn.MaxPool2d((2, 1)),

#             conv_block(256, 512, 3),
#             nn.MaxPool2d((2, 1)),

#             conv_block(512, 512, 3)
#         )

#         with torch.no_grad():
#             dummy = torch.zeros(1, img_channel, img_height, img_width)
#             feat = cnn(dummy)
#             _, c, h, w = feat.size()

#         return cnn, (c, h, w)

#     def forward(self, x):

#         # CNN
#         x = self.cnn(x)  # [B, C, H, W]

#         b, c, h, w = x.size()

#         # convert to sequence
#         x = x.permute(0, 3, 1, 2)  # [B, W, C, H]
#         x = x.reshape(b, w, c * h)  # [B, W, C*H]

#         # embedding
#         x = self.map_to_seq(x)

#         # positional encoding
#         x = self.pos_encoder(x)

#         # transformer
#         x = self.transformer(x)

#         # classifier
#         x = self.dense(x)

#         # CTC format
#         x = x.permute(1, 0, 2)  # [T, B, C]

#         return x
    
class CRNN(nn.Module):
    def __init__(self, img_channel, img_height, img_width, num_class,
                 map_to_seq_hidden=256, nhead=8, num_layers=4, leaky_relu=False):

        super(CRNN, self).__init__()

        # CNN backbone
        self.cnn, (output_channel, output_height, output_width) = \
            self._vgg_backbone(img_channel, img_height, img_width, leaky_relu)

        # CNN -> sequence
        self.map_to_seq = nn.Linear(output_channel * output_height, map_to_seq_hidden)

        # positional encoding
        self.pos_encoder = PositionalEncoding(map_to_seq_hidden)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=map_to_seq_hidden,
            nhead=nhead,
            dim_feedforward=4 * map_to_seq_hidden,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # classifier
        self.dense = nn.Linear(map_to_seq_hidden, num_class)

    def _vgg_backbone(self, img_channel, img_height, img_width, leaky_relu):

        def conv_block(in_c, out_c, num_convs):
            layers = []
            for i in range(num_convs):
                layers.append(nn.Conv2d(in_c, out_c, 3, 1, 1))
                layers.append(nn.BatchNorm2d(out_c))

                if leaky_relu:
                    layers.append(nn.LeakyReLU(0.2, inplace=True))
                else:
                    layers.append(nn.ReLU(inplace=True))

                in_c = out_c

            return nn.Sequential(*layers)

        cnn = nn.Sequential(

            conv_block(img_channel, 64, 2),
            nn.MaxPool2d(2, 2),

            conv_block(64, 128, 2),
            nn.MaxPool2d(2, 2),

            conv_block(128, 256, 3),
            nn.MaxPool2d((2, 1)),

            conv_block(256, 512, 3),
            nn.MaxPool2d((2, 1)),

            conv_block(512, 512, 3)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, img_channel, img_height, img_width)
            feat = cnn(dummy)
            _, c, h, w = feat.size()

        return cnn, (c, h, w)

    def forward(self, x):

        # CNN
        x = self.cnn(x)  # [B, C, H, W]

        b, c, h, w = x.size()

        # convert to sequence
        x = x.permute(0, 3, 1, 2)  # [B, W, C, H]
        x = x.reshape(b, w, c * h)  # [B, W, C*H]

        # embedding
        x = self.map_to_seq(x)

        # positional encoding
        x = self.pos_encoder(x)

        # transformer
        x = self.transformer(x)

        # classifier
        x = self.dense(x)

        # CTC format
        x = x.permute(1, 0, 2)  # [T, B, C]

        return x