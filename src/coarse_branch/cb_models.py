"""굵은 깊이 가지 모델: 인코더 둘 (DAv2-S 통째 학습 / DepthVLM 의 SigLIP ViT + LoRA) + 공통 머리.

입력은 작은 사진 uint8 (B, 3, H, W) — H · W 는 32 의 배수. 출력은 굵은 칸마다 (굵은 상대 로그 깊이, 하늘 logit).
머리: 네 층 특징 → 256 합 → 트랜스포머 블록 → 굵은 칸 질의 (칸 중심 위치) 가 크로스 어텐션 → 숫자 둘. 마지막 층은 0 에서 시작 (처음엔 평평한 굵은 무늬).
프로젝트 기존 코드는 고치지 않는다. SigLIP ViT 는 DepthVLM 체크포인트의 model.visual.* 만 읽는다 (LLM 은 안 읽음).
"""
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cb_common as C  # noqa: E402

P = C.PLAN
VDR = os.path.expanduser("~/code/vlm-depth-rmse")
VIT_LIN = ("attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2")     # geo_lora.VIT_LIN 과 같은 자리


def pos2d(h, w, d, device):
    """2D sin-cos 위치 (칸 중심을 0–1 로 맞춘 뒤 ×64 — 격자 크기가 달라도 같은 자리 = 같은 값). (h·w, d)"""
    y = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h * 64
    x = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w * 64
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    k = d // 4
    om = 1.0 / (10000 ** (torch.arange(k, device=device, dtype=torch.float32) / k))
    a, b = yy.reshape(-1, 1) * om, xx.reshape(-1, 1) * om
    return torch.cat([a.sin(), a.cos(), b.sin(), b.cos()], 1)


class SelfBlock(nn.Module):
    def __init__(self, d, heads=8):
        super().__init__()
        self.n1, self.attn = nn.LayerNorm(d), nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2, self.mlp = nn.LayerNorm(d), nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class CrossBlock(nn.Module):
    def __init__(self, d, heads=8):
        super().__init__()
        self.n1, self.nk, self.attn = nn.LayerNorm(d), nn.LayerNorm(d), nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2, self.mlp = nn.LayerNorm(d), nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, q, x):
        k = self.nk(x)
        q = q + self.attn(self.n1(q), k, k, need_weights=False)[0]
        return q + self.mlp(self.n2(q))


class CoarseHead(nn.Module):
    def __init__(self, in_dims):
        super().__init__()
        d = int(P["head_dim"])
        self.d = d
        self.proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(c), nn.Linear(c, d)) for c in in_dims])
        self.blocks = nn.ModuleList([SelfBlock(d) for _ in range(int(P["head_blocks"]))])
        self.qmlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.qblocks = nn.ModuleList([CrossBlock(d) for _ in range(int(P["query_blocks"]))])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))
        nn.init.zeros_(self.out[1].weight)
        nn.init.zeros_(self.out[1].bias)
        with torch.no_grad():
            self.out[1].bias[1] = -2.0                                   # 하늘 확률 약 0.12 에서 시작

    def forward(self, feats, gh, gw, ky, kx):
        x = sum(p(f) for p, f in zip(self.proj, feats))                   # (B, N, d)
        assert x.shape[1] == gh * gw, (x.shape, gh, gw)
        x = x + pos2d(gh, gw, self.d, x.device)
        for b in self.blocks:
            x = b(x)
        q = self.qmlp(pos2d(ky, kx, self.d, x.device))[None].expand(x.shape[0], -1, -1)
        for b in self.qblocks:
            q = b(q, x)
        o = self.out(q)
        return o[..., 0].reshape(-1, ky, kx), o[..., 1].reshape(-1, ky, kx)


class Dav2S(nn.Module):
    """DAv2-S 의 DINOv2-S 몸통 (14 화소 패치). 통째로 학습."""
    out_dims = (384,) * 4

    def __init__(self):
        super().__init__()
        from transformers import DepthAnythingForDepthEstimation
        self.bb = DepthAnythingForDepthEstimation.from_pretrained(P["dav2s"]).backbone
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def trainable(self):
        return list(self.bb.parameters())

    def state(self):
        return self.bb.state_dict()

    def load_state(self, sd):
        self.bb.load_state_dict(sd)

    def forward(self, u8):
        H, W = u8.shape[-2:]
        h, w = 14 * round(H / 14), 14 * round(W / 14)
        x = F.interpolate(u8.float() / 255.0, size=(h, w), mode="bilinear", antialias=True, align_corners=False)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=u8.is_cuda):
            fm = self.bb((x - self.mean) / self.std).feature_maps
        return [f[:, 1:].float() for f in fm], h // 14, w // 14


class LoRALinear(nn.Module):
    """y = W x + b + B A x. W · b 는 원래 층 그대로 (고정), A · B 는 fp32 로 두고 계산은 입력 dtype (geo_lora.LoRALinear 와 같은 식, 항상 켬)."""

    def __init__(self, base, r):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype))


def load_visual():
    """DepthVLM 체크포인트에서 비전 인코더 (model.visual.*) 만 읽는다."""
    sys.path.insert(0, os.path.join(VDR, "third_party/DepthVLM"))
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from model.configuration_qwen3_vl import Qwen3VLConfig
    from model.modeling_qwen3_vl import Qwen3VLVisionModel
    snap = snapshot_download(P["depthvlm"][0], revision=P["depthvlm"][1], local_files_only=True)
    cfg = Qwen3VLConfig.from_pretrained(snap)
    vis = Qwen3VLVisionModel(cfg.vision_config)
    pre = "model.visual."
    with safe_open(os.path.join(snap, "model.safetensors"), "pt") as f:
        sd = {k[len(pre):]: f.get_tensor(k) for k in f.keys() if k.startswith(pre)}
    vis.load_state_dict(sd, strict=True)
    return vis


def patchify(x):
    """(B, 3, H, W) 정규화된 사진 → Qwen 프로세서와 같은 순서의 패치 (B·gh·gw, 3·2·16·16): 2×2 묶음 순서, 패치 안은 (통로, 시간 2, 16, 16)."""
    B, _, H, W = x.shape
    gh, gw = H // 16, W // 16
    p = x.reshape(B, 3, gh // 2, 2, 16, gw // 2, 2, 16).permute(0, 2, 5, 3, 6, 1, 4, 7)       # B, gh/2, gw/2, 2, 2, 3, 16, 16
    p = p.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, 2, -1, -1)
    return p.reshape(B * gh * gw, 3 * 2 * 16 * 16)


def unshuffle(h, B, gh, gw):
    """ViT 안의 2×2 묶음 순서 (N, C) → 사진마다 행 우선 순서 (B, gh·gw, C)."""
    return h.reshape(B, gh // 2, gw // 2, 2, 2, -1).permute(0, 1, 3, 2, 4, 5).reshape(B, gh * gw, -1)


class SiglipLoRA(nn.Module):
    """DepthVLM 의 SigLIP ViT (고정, bf16) + LoRA. 패치 단위 숨은 상태 (taps 블록 뒤) 를 행 우선으로 돌려준다."""
    out_dims = (1024,) * 4

    def __init__(self):
        super().__init__()
        vis = load_visual().to(dtype=torch.bfloat16)
        for q in vis.parameters():
            q.requires_grad_(False)
        self.mods = []
        for blk in vis.blocks:
            for path in VIT_LIN:
                *pre, last = path.split(".")
                parent = blk
                for name in pre:
                    parent = getattr(parent, name)
                lo = LoRALinear(getattr(parent, last), int(P["lora_rank"]))
                setattr(parent, last, lo)
                self.mods.append(lo)
        self.vis = vis.eval()
        self.taps = tuple(int(t) for t in P["siglip_taps"])
        self._buf = {}
        for t in self.taps:
            vis.blocks[t].register_forward_hook(lambda m, i, o, t=t: self._buf.__setitem__(t, o))

    def train(self, mode=True):
        super().train(mode)
        self.vis.eval()                                                   # 고정 몸통은 늘 eval
        return self

    def trainable(self):
        return [q for m in self.mods for q in (m.A, m.B)]

    def state(self):
        return dict(A=[m.A.detach().cpu() for m in self.mods], B=[m.B.detach().cpu() for m in self.mods])

    def load_state(self, sd):
        assert len(sd["A"]) == len(self.mods)
        with torch.no_grad():
            for m, a, b in zip(self.mods, sd["A"], sd["B"]):
                m.A.copy_(a)
                m.B.copy_(b)

    def forward(self, u8):
        B, _, H, W = u8.shape
        assert H % 32 == 0 and W % 32 == 0, (H, W)
        gh, gw = H // 16, W // 16
        pv = patchify((u8.float() / 255.0 - 0.5) / 0.5).to(torch.bfloat16)
        grid = torch.tensor([[1, gh, gw]] * B, device=u8.device)
        self._buf.clear()
        self.vis(pv, grid_thw=grid)
        return [unshuffle(self._buf[t], B, gh, gw).float() for t in self.taps], gh, gw


class Branch(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.enc = Dav2S() if kind == "dav2s" else SiglipLoRA()
        self.head = CoarseHead(self.enc.out_dims)

    def forward(self, u8, ky, kx):
        feats, gh, gw = self.enc(u8)
        with torch.autocast("cuda", enabled=False):
            return self.head(feats, gh, gw, ky, kx)

    def state(self):
        return dict(kind=self.kind, enc=self.enc.state(), head=self.head.state_dict())

    def load_state(self, sd):
        assert sd["kind"] == self.kind
        self.enc.load_state(sd["enc"])
        self.head.load_state_dict(sd["head"])

    def n_added(self):
        """추론에 새로 더해지는 매개변수 수 (SigLIP 몸통은 DepthVLM 과 같이 쓰므로 LoRA 만 셈)."""
        return sum(q.numel() for q in self.enc.trainable()) + sum(q.numel() for q in self.head.parameters())
