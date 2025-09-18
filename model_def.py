# model_def.py
# KP+Hands Fusion architecture (CPU friendly)
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

def masked_mean_pool(x: torch.Tensor, m: Optional[torch.Tensor]) -> torch.Tensor:
    if m is None:
        return x.mean(dim=1)
    denom = m.sum(dim=1, keepdim=True).clamp(min=1.0)
    return (x * m.unsqueeze(-1)).sum(dim=1) / denom

class KPTransformer(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 256, nhead: int = 8, num_layers: int = 4, dim_ff: int = 768, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        # Avoid nested tensor path to silence prototype warnings & be CPU-friendly
        try:
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers, enable_nested_tensor=False)
        except TypeError:
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, m: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.proj(x)  # [B,T,D]
        key_padding_mask = (m < 0.5) if m is not None else None  # True = pad
        try:
            h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        except TypeError:
            h = self.encoder(h)
        h = self.ln(h)
        return masked_mean_pool(h, m)

class HandFrameCNN(nn.Module):
    def __init__(self, out_dim: int = 256, bn_eps: float = 1e-5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32, eps=bn_eps), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64, eps=bn_eps), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128, eps=bn_eps), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128, eps=bn_eps), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        h = self.conv(x)
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0).flatten(1)
        h = self.fc(h)
        return torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

class HandsTemporalEncoder(nn.Module):
    def __init__(self, frame_out: int = 256, t_model: int = 256, nhead: int = 8, layers: int = 2, dim_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.frame_encoder = HandFrameCNN(out_dim=frame_out)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=t_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        try:
            self.temporal = nn.TransformerEncoder(enc_layer, num_layers=layers, enable_nested_tensor=False)
        except TypeError:
            self.temporal = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.mix = nn.Linear(frame_out * 2, t_model)
        self.ln = nn.LayerNorm(t_model)

    def forward(self, left: torch.Tensor, right: torch.Tensor, m_left: Optional[torch.Tensor], m_right: Optional[torch.Tensor]) -> torch.Tensor:
        B, T = left.size(0), left.size(1)
        l_flat = left.view(B*T, *left.shape[2:])     # [B*T,3,H,W]
        r_flat = right.view(B*T, *right.shape[2:])
        l_feat = self.frame_encoder(l_flat)          # [B*T,F]
        r_feat = self.frame_encoder(r_flat)          # [B*T,F]
        lr = torch.cat([l_feat, r_feat], dim=1).view(B, T, -1)  # [B,T,2F]
        lr = torch.nan_to_num(lr, nan=0.0, posinf=0.0, neginf=0.0)

        h = self.mix(lr)
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

        if (m_left is not None) and (m_right is not None):
            m_any = ((m_left > 0.5) | (m_right > 0.5)).float()
        elif m_left is not None:
            m_any = (m_left > 0.5).float()
        elif m_right is not None:
            m_any = (m_right > 0.5).float()
        else:
            m_any = torch.ones((B, T), device=h.device, dtype=torch.float32)

        h = h * m_any.unsqueeze(-1)
        key_padding_mask = (m_any < 0.5) if m_any is not None else None
        try:
            h = self.temporal(h, src_key_padding_mask=key_padding_mask)
        except TypeError:
            h = self.temporal(h)
        h = self.ln(h)
        denom = m_any.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (h * m_any.unsqueeze(-1)).sum(dim=1) / denom
        return torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)

class KPHandsFusion(nn.Module):
    def __init__(self, kp_in: int, num_classes: int,
                 kp_d: int = 256, kp_layers: int = 4,
                 hands_fdim: int = 128, hands_tdim: int = 256,
                 hands_layers: int = 2, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.kp_enc = KPTransformer(in_dim=kp_in, d_model=kp_d, nhead=nhead, num_layers=kp_layers, dim_ff=kp_d*3, dropout=dropout)
        self.hands_enc = HandsTemporalEncoder(frame_out=hands_fdim, t_model=hands_tdim, nhead=nhead, layers=hands_layers, dim_ff=hands_tdim*2, dropout=dropout)
        fusion_dim = kp_d + hands_tdim
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes)
        )

    def forward(self, kp: torch.Tensor, m_kp: Optional[torch.Tensor],
                left: torch.Tensor, right: torch.Tensor, m_left: Optional[torch.Tensor], m_right: Optional[torch.Tensor]) -> torch.Tensor:
        kp_repr = self.kp_enc(kp, m_kp)                                   # [B, Dk]
        hands_repr = self.hands_enc(left, right, m_left, m_right)         # [B, Dh]
        fused = torch.cat([kp_repr, hands_repr], dim=1)                   # [B, Dk+Dh]
        logits = self.head(fused)                                         # [B, C]
        return logits
