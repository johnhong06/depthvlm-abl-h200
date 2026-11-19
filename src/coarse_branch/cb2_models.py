"""분리 디코딩 실험 1 단계의 모델 (PROMPT_v2 3.1 · 3.5 · 6.1): 모양 지식을 담는 세 팔 + 공통 머리.

  읽기 전용 토큰 팔  ReadArm    DepthVLM ViT (고정) 의 각 층 패치 토큰 (k · v) 을 읽는 기하 토큰 약 150 개 + 그 자리에만 LoRA (GeoTokens)
  작은 사진 통과 팔  ThumbArm('siglip')  같은 ViT + 모든 토큰 LoRA, 약 15 만 화소 사진을 한 번 더 (cb_models.SiglipLoRA)
  작은 인코더 팔     ThumbArm('dav2s')   DAv2-S 인코더 통째 학습 (cb_models.Dav2S)
공통 머리 GridHead: 네 층 특징마다 LN → Linear(256) 를 더함 → 굵은 격자로 bilinear → [3 × 3 conv · GELU] × 2 → 1 × 1 conv → (굵은 상대 로그 깊이, 하늘 logit).

읽기 전용의 뜻: 본 통과 (ViT 원래 계산) 는 손대지 않는다. 훅으로 읽기만 하고 (0 층 입력 · 층마다 qkv 출력), 기하 토큰은 그 값을 열쇠 · 값으로 받아 따로 계산한다.
패치 토큰은 기하 토큰을 보지 못하므로 본 통과의 출력은 기하 토큰이 있든 없든 같다. LoRA 는 기하 토큰 계산에만 더해진다 (원래 층 가중치는 그대로 함께 씀).
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cb_models as M  # noqa: E402

RANK, TAPS, HEAD_DIM = 64, (5, 11, 17, 23), 256


class GridHead(nn.Module):
    def __init__(self, in_dims):
        super().__init__()
        self.proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(c), nn.Linear(c, HEAD_DIM)) for c in in_dims])
        self.conv = nn.Sequential(nn.Conv2d(HEAD_DIM, HEAD_DIM, 3, padding=1), nn.GELU(), nn.Conv2d(HEAD_DIM, HEAD_DIM, 3, padding=1), nn.GELU(), nn.Conv2d(HEAD_DIM, 2, 1))
        nn.init.zeros_(self.conv[-1].weight)
        nn.init.zeros_(self.conv[-1].bias)
        with torch.no_grad():
            self.conv[-1].bias[1] = -2.0                              # 하늘 확률 약 0.12 에서 시작 (11-10 과 같음)

    def forward(self, feats, gh, gw, ky, kx):
        x = sum(p(f) for p, f in zip(self.proj, feats))               # (B, gh·gw, 256)
        assert x.shape[1] == gh * gw, (x.shape, gh, gw)
        x = x.transpose(1, 2).reshape(x.shape[0], HEAD_DIM, gh, gw)
        x = F.interpolate(x, size=(ky, kx), mode="bilinear", align_corners=False, antialias=(gh > ky or gw > kx))
        o = self.conv(x)
        return o[:, 0], o[:, 1]


class ThumbArm(nn.Module):
    """작은 사진 (uint8, 32 의 배수) 을 따로 통과시키는 두 팔."""

    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.enc = M.Dav2S() if kind == "dav2s" else M.SiglipLoRA()
        self.head = GridHead(self.enc.out_dims)

    def forward(self, u8, ky, kx):
        feats, gh, gw = self.enc(u8)
        with torch.autocast("cuda", enabled=False):
            return self.head(feats, gh, gw, ky, kx)

    def groups(self, lr_lora, lr_new, lr_body, wd):
        body = dict(params=self.enc.trainable(), lr=lr_body, weight_decay=wd) if self.kind == "dav2s" else dict(params=self.enc.trainable(), lr=lr_lora, weight_decay=0.0)
        return [body, dict(params=list(self.head.parameters()), lr=lr_new, weight_decay=wd)]

    def state(self):
        return dict(kind=self.kind, enc=self.enc.state(), head=self.head.state_dict())

    def load_state(self, sd):
        assert sd["kind"] == self.kind
        self.enc.load_state(sd["enc"])
        self.head.load_state_dict(sd["head"])

    def n_added(self):
        return sum(q.numel() for q in self.enc.trainable()) + sum(q.numel() for q in self.head.parameters())


def rope(x, cos, sin):
    """모델의 apply_rotary_pos_emb_vision 과 같은 식 — x (토큰, 헤드, 헤드 차원), cos · sin (토큰, 헤드 차원)."""
    xf = x.float()
    x1, x2 = xf[..., : xf.shape[-1] // 2], xf[..., xf.shape[-1] // 2:]
    return (xf * cos[:, None, :] + torch.cat((-x2, x1), -1) * sin[:, None, :]).to(x.dtype)


class GeoTokens(nn.Module):
    """읽기 전용 기하 토큰. vis = DepthVLM 의 비전 인코더 (Qwen3VLVisionModel, 고정). 사진 한 장씩.
    쓰는 법: vis 를 평소대로 한 번 돌린다 (훅이 0 층 입력과 층마다 qkv 출력을 잡음) → tokens(Hp, Wp, gh, gw) 가 꺼내는 층의 기하 토큰을 돌려준다."""

    def __init__(self, vis, rank=RANK, taps=TAPS):
        super().__init__()
        for q in vis.parameters():
            q.requires_grad_(False)
        self.vis, self.taps = vis.eval(), tuple(taps)
        d, inner, L = vis.config.hidden_size, vis.config.intermediate_size, len(vis.blocks)
        dims = ((d, 3 * d), (d, d), (d, inner), (inner, d))            # attn.qkv · attn.proj · mlp.linear_fc1 · mlp.linear_fc2
        self.A = nn.ParameterList([nn.Parameter(torch.empty(rank, i)) for _ in range(L) for i, _ in dims])
        self.B = nn.ParameterList([nn.Parameter(torch.zeros(o, rank)) for _ in range(L) for _, o in dims])
        for a in self.A:
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        self.e = nn.Parameter(torch.zeros(d))                          # 시작값에 더하는 학습 벡터 (0 에서 시작)
        self._x0, self._qkv = None, [None] * L
        vis.blocks[0].register_forward_pre_hook(lambda m, args, kwargs: setattr(self, "_x0", args[0]), with_kwargs=True)
        for l, blk in enumerate(vis.blocks):
            blk.attn.qkv.register_forward_hook(lambda m, i, o, l=l: self._qkv.__setitem__(l, o))

    def train(self, mode=True):
        super().train(mode)
        self.vis.eval()
        return self

    def trainable(self):
        return list(self.A) + list(self.B) + [self.e]

    def state(self):
        return dict(A=[a.detach().cpu() for a in self.A], B=[b.detach().cpu() for b in self.B], e=self.e.detach().cpu(), taps=self.taps)

    def load_state(self, sd):
        with torch.no_grad():
            for dst, src in zip(list(self.A) + list(self.B), list(sd["A"]) + list(sd["B"])):
                dst.copy_(src)
            self.e.copy_(sd["e"])

    def lora(self, l, j, x):
        return F.linear(F.linear(x, self.A[4 * l + j].to(x.dtype)), self.B[4 * l + j].to(x.dtype))

    def tokens(self, Hp, Wp, gh, gw, self_keys=True, lora=True):
        """잡아 둔 본 통과 값으로 기하 토큰을 계산. → 꺼내는 층마다 (gh·gw, 1024) fp32 (행 우선).
        self_keys = False 면 패치 토큰의 열쇠 · 값만 본다 (시험용 — LoRA 0 · 칸 = 패치 하나일 때 본 통과의 그 패치 값과 같아야 한다)."""
        vis, N, Mg = self.vis, Hp * Wp, gh * gw
        x0 = self._x0
        assert x0 is not None and x0.shape[0] == N, (None if x0 is None else x0.shape, N)
        dt, dev = x0.dtype, x0.device
        ras = M.unshuffle(x0, 1, Hp, Wp)[0].reshape(Hp, Wp, -1).permute(2, 0, 1).float()
        g = F.adaptive_avg_pool2d(ras[None], (gh, gw))[0].flatten(1).T
        g = (g + self.e).to(dt)
        rows = (torch.arange(gh, device=dev, dtype=torch.float32) + 0.5) * Hp / gh - 0.5
        cols = (torch.arange(gw, device=dev, dtype=torch.float32) + 0.5) * Wp / gw - 0.5
        pos = torch.stack(torch.meshgrid(rows, cols, indexing="ij"), -1).reshape(Mg, 2)
        inv = vis.rotary_pos_emb.inv_freq                                # 모델과 같은 자료형 (bf16) 으로 각도 · cos · sin 을 낸 뒤 fp32 로 — 본 통과가 패치에 하는 것과 같은 순서
        fr = (pos.to(inv.dtype)[:, :, None] * inv[None, None, :]).flatten(1)
        emb = torch.cat((fr, fr), -1)
        cg, sg = emb.cos().float(), emb.sin().float()
        rp = vis.rot_pos_emb(torch.tensor([[1, Hp, Wp]], device=dev))
        embp = torch.cat((rp, rp), -1)
        cp, sp = embp.cos().float(), embp.sin().float()
        outs = []
        for l, blk in enumerate(vis.blocks):
            at, nh = blk.attn, blk.attn.num_heads
            h = blk.norm1(g)
            qkv = F.linear(h, at.qkv.weight, at.qkv.bias) + (self.lora(l, 0, h) if lora else 0)    # 모듈을 부르면 훅이 잡아 둔 패치 값을 덮어쓴다 — 함수로 계산
            q, k, v = qkv.reshape(Mg, 3, nh, -1).permute(1, 0, 2, 3).unbind(0)
            q, k = rope(q, cg, sg), rope(k, cg, sg)
            _, kp, vp = self._qkv[l].detach().reshape(N, 3, nh, -1).permute(1, 0, 2, 3).unbind(0)
            kp = rope(kp, cp, sp)
            K, V = (torch.cat((kp, k)), torch.cat((vp, v))) if self_keys else (kp, vp)
            o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], K.transpose(0, 1)[None], V.transpose(0, 1)[None])[0].transpose(0, 1).reshape(Mg, -1)
            g = g + at.proj(o) + (self.lora(l, 1, o) if lora else 0)
            h2 = blk.norm2(g)
            m = blk.mlp.act_fn(blk.mlp.linear_fc1(h2) + (self.lora(l, 2, h2) if lora else 0))
            g = g + blk.mlp.linear_fc2(m) + (self.lora(l, 3, m) if lora else 0)
            if l in self.taps:
                outs.append(g.float())
        return outs


def normalize_patches(u8):
    """uint8 (1, 3, H, W) → DepthVLM 처리기와 같은 패치 (bf16)."""
    return M.patchify((u8.float() / 255.0 - 0.5) / 0.5).to(torch.bfloat16)


class ReadArm(nn.Module):
    """읽기 전용 토큰 팔 (1 단계): ViT 만 올려서 본 통과 + 기하 토큰 + 공통 머리. 사진 한 장씩 (본 통과와 같은 전처리를 거친 패치)."""
    kind = "read"

    def __init__(self):
        super().__init__()
        self.geo = GeoTokens(M.load_visual().to(dtype=torch.bfloat16))
        self.head = GridHead((self.geo.vis.config.hidden_size,) * len(TAPS))
        self.sky = nn.Sequential(nn.LayerNorm(self.geo.vis.config.hidden_size), nn.Linear(self.geo.vis.config.hidden_size, 1))
        nn.init.zeros_(self.sky[1].weight)
        nn.init.constant_(self.sky[1].bias, -2.0)

    def train(self, mode=True):
        super().train(mode)
        self.geo.vis.eval()
        return self

    def forward(self, pv, Hp, Wp, ky, kx, gh, gw):
        """pv = 본 통과가 받는 패치 (Hp·Wp, 1536) — cb2_data.model_pixels."""
        assert pv.shape[0] == Hp * Wp, (pv.shape, Hp, Wp)
        with torch.no_grad():
            self.geo.vis(pv.to(torch.bfloat16), grid_thw=torch.tensor([[1, Hp, Wp]], device=pv.device))
        G = self.geo.tokens(Hp, Wp, gh, gw)
        with torch.autocast("cuda", enabled=False):
            r, sky = self.head([g[None] for g in G], gh, gw, ky, kx)
            sky_tok = self.sky(G[-1])[:, 0].reshape(1, gh, gw)         # 기하 토큰 격자의 하늘 logit (2 단계 · 추론에서 쓰는 것)
        return r, sky, sky_tok

    def groups(self, lr_lora, lr_new, lr_body, wd):
        return [dict(params=list(self.geo.A) + list(self.geo.B), lr=lr_lora, weight_decay=0.0),
                dict(params=[self.geo.e] + list(self.head.parameters()) + list(self.sky.parameters()), lr=lr_new, weight_decay=wd)]

    def state(self):
        return dict(kind=self.kind, geo=self.geo.state(), head=self.head.state_dict(), sky=self.sky.state_dict())

    def load_state(self, sd):
        assert sd["kind"] == self.kind
        self.geo.load_state(sd["geo"])
        self.head.load_state_dict(sd["head"])
        self.sky.load_state_dict(sd["sky"])

    def n_added(self):
        return sum(q.numel() for q in self.geo.trainable()) + sum(q.numel() for q in self.sky.parameters())   # 1 단계 머리는 2 단계에서 안 씀
