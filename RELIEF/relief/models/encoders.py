from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import inject_lora


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: int, pool: int):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(c_out)
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.bn(self.conv(x))))


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, channels: Sequence[int], kernel_size: int, pool: int):
        super().__init__()
        blocks, c = [], in_channels
        for i, h in enumerate(channels):
            blocks.append(ConvBlock(c, h, kernel_size, pool if i < len(channels) - 1 else 0))
            c = h
        self.layers = nn.ModuleList(blocks)
        self.out_dim = c

    def layer_modules(self) -> List[nn.Module]:
        return list(self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x.mean(dim=-1)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.wi = nn.Linear(d_model, d_ff)
        self.wo = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        h = self.norm1(x)

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(b, l, self.n_heads, d // self.n_heads).transpose(1, 2)

        att = F.scaled_dot_product_attention(
            heads(self.q(h)), heads(self.k(h)), heads(self.v(h)), dropout_p=self.dropout if self.training else 0.0
        )
        x = x + self.o(att.transpose(1, 2).reshape(b, l, d))
        return x + self.wo(F.gelu(self.wi(self.norm2(x))))


class PatchTSTEncoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        patch_len: int,
        patch_stride: int,
        dropout: float,
        checkpoint: Optional[str] = None,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        n_patches = (seq_len - patch_len) // patch_stride + 1
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_patches, d_model))
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.out_dim = d_model
        if checkpoint:
            self.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=False)

    def layer_modules(self) -> List[nn.Module]:
        return list(self.blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        s = x.reshape(b * c, t)
        s = (s - s.mean(-1, keepdim=True)) / (s.std(-1, keepdim=True) + 1e-5)
        z = self.embed(s.unfold(-1, self.patch_len, self.patch_stride)) + self.pos
        for block in self.blocks:
            z = block(z)
        return self.norm(z).mean(1).view(b, c, -1).mean(1)


class MomentEncoder(nn.Module):
    def __init__(self, name: str, seq_len: int):
        super().__init__()
        from momentfm import MOMENTPipeline

        self.seq_len = seq_len
        self.moment = MOMENTPipeline.from_pretrained(name, model_kwargs={"task_name": "embedding"})
        self.moment.init()
        d_model = getattr(self.moment.config, "d_model", None)
        self.out_dim = int(d_model) if d_model else self.infer_dim()

    @torch.no_grad()
    def infer_dim(self) -> int:
        x = torch.zeros(1, 1, self.seq_len)
        return int(self.moment(x_enc=x, input_mask=torch.ones(1, self.seq_len)).embeddings.shape[-1])

    def layer_modules(self) -> List[nn.Module]:
        return list(self.moment.encoder.block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        s = x.reshape(b * c, 1, t)
        pad = self.seq_len - t
        mask = torch.ones(b * c, self.seq_len, device=x.device)
        if pad > 0:
            s = F.pad(s, (pad, 0))
            mask[:, :pad] = 0
        emb = self.moment(x_enc=s, input_mask=mask).embeddings
        return emb.view(b, c, -1).mean(1)


def build_encoder(cfg, in_channels: int, seq_len: int) -> nn.Module:
    if cfg.backbone == "cnn":
        return CNNEncoder(in_channels, cfg.cnn.channels, cfg.cnn.kernel_size, cfg.cnn.pool)
    if cfg.backbone == "moment":
        encoder = MomentEncoder(cfg.moment.name, cfg.moment.seq_len)
    elif cfg.backbone == "patchtst":
        p = cfg.patchtst
        encoder = PatchTSTEncoder(seq_len, p.d_model, p.n_layers, p.n_heads, p.d_ff, p.patch_len, p.patch_stride, cfg.dropout, p.checkpoint)
    else:
        raise ValueError(f"unknown backbone {cfg.backbone}")
    encoder.requires_grad_(False)
    for layer in encoder.layer_modules():
        inject_lora(layer, cfg.lora_targets, cfg.rho)
    return encoder
