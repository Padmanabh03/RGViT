import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from timm.models.vision_transformer import vit_base_patch16_224

class RadarEncoder(nn.Module):
    def __init__(self, input_dim=6, embed_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.ReLU()
        )

    def forward(self, radar):
        # radar: (B, N, 6)
        return self.encoder(radar)  # (B, N, embed_dim)

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(self, img_tokens, radar_tokens):
        Q = self.query_proj(img_tokens)
        K = self.key_proj(radar_tokens)
        V = self.value_proj(radar_tokens)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = torch.softmax(attn, dim=-1)
        fused = torch.matmul(attn, V)
        return img_tokens + fused

class DetectionHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, num_classes)
        )
        self.reg_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 4)  # x, y, w, h
        )

    def forward(self, tokens):
        cls_logits = self.cls_head(tokens)
        box_preds = self.reg_head(tokens)
        return cls_logits, box_preds

class RGVisionTransformer(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.image_encoder = vit_base_patch16_224(pretrained=True)
        self.radar_encoder = RadarEncoder(input_dim=6, embed_dim=768)
        self.fusion = CrossAttentionFusion(embed_dim=768)
        self.head = DetectionHead(embed_dim=768, num_classes=num_classes)

    def forward(self, image, radar):
        # image: (B, 3, H, W)
        # radar: (B, N, 6)
        B = image.shape[0]
        img_tokens = self.image_encoder.patch_embed(image)  # (B, num_patches, 768)
        img_tokens = self.image_encoder.pos_drop(img_tokens + self.image_encoder.pos_embed[:, 1:, :])
        for blk in self.image_encoder.blocks:
            img_tokens = blk(img_tokens)

        radar_tokens = self.radar_encoder(radar)  # (B, N, 768)
        fused_tokens = self.fusion(img_tokens, radar_tokens)  # (B, num_patches, 768)

        cls_logits, box_preds = self.head(fused_tokens.mean(dim=1))
        return cls_logits, box_preds

if __name__ == "__main__":
    model = RGVisionTransformer(num_classes=11)
    dummy_image = torch.randn(2, 3, 224, 224)
    dummy_radar = torch.randn(2, 32, 6)
    cls_out, box_out = model(dummy_image, dummy_radar)
    print("Class logits:", cls_out.shape)  # [B, num_classes]
    print("Box preds:", box_out.shape)    # [B, 4]