"""굵은 깊이 가지 공통 부품: 경로 · 격자 크기 · 굵은 무늬 합치기 · 손실 · 채점. torch · numpy 만 쓴다 (프로젝트 기존 코드 import 없음).

지도는 모두 로그 깊이, 8 화소 작은 칸 해상도 (4·ph × 4·pw, D-114 캐시와 같음). 묶음 안의 사진은 크기가 같다.
  합치기: F = V + up(Δ) − 평균,  Δ = (B − 평균) − (굵은(V) − 평균)  (하늘 아닌 굵은 칸만, 하늘 칸은 0)
  → 하늘 아닌 곳의 평균은 V 와 같다 (배율 유지).
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / "cb_plan.json").read_text())
CACHE = Path("~/data/dvsilog/d114/cache").expanduser()          # 읽기만
DATA = Path("~/data/dvsilog/coarse_branch").expanduser()
CKPT = Path("~/checkpoints/dvsilog/coarse_branch").expanduser()
INDOOR = ("hm3d", "scannetpp", "taskonomy")
DRIVING = ("argoverse", "ddad", "nuscenes", "waymo")
LOG125 = math.log(1.25)


def paths(split, i):
    """작은 사진 · 작은 칸 지도 파일 위치 (cb_prep 가 만듦)."""
    sub = f"{int(i) // 1000:03d}"
    return DATA / "thumbs" / split / sub / f"{int(i)}.jpg", DATA / "maps" / split / sub / f"{int(i)}.npz"


def thumb_grid(ph, pw):
    """가지가 보는 작은 사진의 크기 (32 화소 토큰 단위): 긴 변 = thumb_tokens, 짧은 변은 비율대로 (최소 4)."""
    t = int(PLAN["thumb_tokens"])
    if pw >= ph:
        return max(4, round(t * ph / pw)), t
    return t, max(4, round(t * pw / ph))


def coarse_grid(hs, ws):
    """굵은 칸 수: 짧은 변 k_short 칸, 긴 변은 비율대로."""
    k = int(PLAN["k_short"])
    if ws >= hs:
        return k, max(k, round(k * ws / hs))
    return max(k, round(k * hs / ws)), k


def mpool(x, w, ky, kx):
    """무게 w 로 굵은 칸 평균. x · w = (B, H, W) → (평균, 칸의 무게 평균). 무게가 없는 칸의 평균은 0."""
    num = F.adaptive_avg_pool2d((x * w)[:, None], (ky, kx))[:, 0]
    den = F.adaptive_avg_pool2d(w[:, None], (ky, kx))[:, 0]
    return torch.where(den > 1e-4, num / den.clamp_min(1e-4), torch.zeros_like(num)), den


def up(c, size):
    return F.interpolate(c[:, None], size=size, mode="bilinear", align_corners=False)[:, 0]


def wmean(x, w):
    return (x * w).flatten(1).sum(1) / w.flatten(1).sum(1).clamp_min(1e-8)


def wvar(d, w):
    m = wmean(d, w)
    return wmean((d - m[:, None, None]) ** 2, w), m


def merge(V, B, sky, center="mean"):
    """V (B, H, W) 원래 로그 깊이 · B (B, ky, kx) 가지가 낸 굵은 상대 깊이 · sky (B, H, W) 하늘 비율 [0, 1].
    → F (B, H, W) 와 부가 값 (delta · valid · B0 · CV0).
    center = 보정에서 빼는 상수: 'mean' 하늘 아닌 곳의 평균 (처음 설계) · 'median' 하늘 아닌 곳의 중앙값 (11-10 보충 — 상수만 달라 SILog · 학습 손실은 같음)."""
    ky, kx = B.shape[-2:]
    q = (1.0 - sky).clamp(0, 1)
    CV, Qc = mpool(V, q, ky, kx)
    valid = Qc >= float(PLAN["min_clear"])
    none = ~valid.flatten(1).any(1)                              # 전부 하늘이면 모든 칸을 씀
    valid = valid | none[:, None, None]
    Qw = torch.where(none[:, None, None], torch.ones_like(Qc), Qc * valid)
    q = torch.where(none[:, None, None], torch.ones_like(q), q)
    CV = torch.where(none[:, None, None], F.adaptive_avg_pool2d(V[:, None], (ky, kx))[:, 0], CV)
    CV0 = CV - wmean(CV, Qw)[:, None, None]
    B0 = B - wmean(B, Qw)[:, None, None]
    delta = (B0 - CV0) * valid
    u = up(delta, V.shape[-2:])
    if center == "median":
        c = torch.stack([(u[b][q[b] > 0.5] if bool((q[b] > 0.5).any()) else u[b].flatten()).median() for b in range(u.shape[0])])
    else:
        c = wmean(u, q)
    Fm = V + u - c[:, None, None]
    return Fm, dict(delta=delta, valid=valid, B0=B0, CV0=CV0, q=q)


def coarse_of(X, sky, ky, kx):
    """지도 X 의 굵은 무늬 (하늘 아닌 화소 평균) — 기준선 (전문가 · 정답 굵은 무늬로 바꾸기) 용."""
    return mpool(X, (1.0 - sky).clamp(0, 1), ky, kx)[0]


def gate_and_target(G, w, E):
    """문턱 (정답에 배율을 맞춘 전문가의 δ1) 과 채운 목표. G · w · E = (B, H, W), 정답 없는 칸은 w = 0 (G 값은 안 씀)."""
    has = w.flatten(1).sum(1) > 0
    a = wmean(G - E, w)
    d1 = wmean(((E + a[:, None, None] - G).abs() < LOG125).float(), w)
    T = torch.where(w > 0, G, E + a[:, None, None])
    return has, d1, T


def loss_fn(Fm, aux, sky_logit, G, w, E, sky):
    """사진마다: 정답 칸 √분산 + 0.5 · (문턱 통과) · 채운 목표 √분산 + 0.1 · 목표 없는 굵은 칸의 Δ² + 0.2 · 하늘."""
    ky, kx = aux["delta"].shape[-2:]
    has, d1, T = gate_and_target(G, w, E)
    gate = has & (d1 >= float(PLAN["gate_d1"]))
    l_gt = torch.sqrt(wvar(Fm - G, w)[0] + 1e-6) * has
    l_fill = torch.sqrt(wvar(Fm - T, aux["q"])[0] + 1e-6) * gate
    gtc = F.adaptive_avg_pool2d((w > 0).float()[:, None], (ky, kx))[:, 0] > 0
    nosup = (aux["valid"] & ~gtc & ~gate[:, None, None]).float()
    l_keep = (aux["delta"] ** 2 * nosup).flatten(1).sum(1) / nosup.flatten(1).sum(1).clamp_min(1.0)
    sc = F.adaptive_avg_pool2d(sky[:, None], (ky, kx))[:, 0]
    l_sky = F.binary_cross_entropy_with_logits(sky_logit, sc, reduction="none").flatten(1).mean(1)
    total = l_gt + float(PLAN["w_fill"]) * l_fill + float(PLAN["w_keep"]) * l_keep + float(PLAN["w_sky"]) * l_sky
    return total.mean(), dict(gt=l_gt.mean().item(), fill=l_fill.mean().item(), keep=l_keep.mean().item(), sky=l_sky.mean().item(),
                              gate=gate.float().mean().item())


def score(X, G, w):
    """사진별 (SILog, 평균 로그 오차) ×100 — 정답 있는 칸에서."""
    v, m = wvar(X - G, w)
    return 100 * torch.sqrt(v), 100 * m


def load_maps(path):
    """작은 칸 지도 한 장 (cb_prep 가 만든 npz) → dict of float32 torch (V · E · G · w · sky)."""
    z = np.load(path)
    g = z["gt_ln"].astype(np.float32)
    w = z["gt_frac"].astype(np.float32)
    ok = np.isfinite(g) & (w > 0)
    return dict(V=torch.from_numpy(z["V"].astype(np.float32)), E=torch.from_numpy(z["E"].astype(np.float32)),
                G=torch.from_numpy(np.where(ok, g, 0.0).astype(np.float32)), w=torch.from_numpy(np.where(ok, w, 0.0).astype(np.float32)),
                sky=torch.from_numpy(z["Sky"].astype(np.float32) / 255.0))


def index(split):
    """D-114 캐시 색인 (split 의 줄) — i · ds · ph · pw."""
    import pandas as pd
    t = pd.read_parquet(CACHE / "cache.parquet")
    return t[t.split == split][["i", "ds", "ph", "pw"]].reset_index(drop=True)


def macro(df, col):
    """데이터셋 평균 → 7 종 평균 · 실내 평균 · 주행 평균."""
    m = df.groupby("ds")[col].mean()
    return dict(all=float(m.mean()), indoor=float(m[[d for d in INDOOR if d in m.index]].mean()),
                driving=float(m[[d for d in DRIVING if d in m.index]].mean()), per_ds={k: float(v) for k, v in m.items()})


def boot_diff(df, a, b, groups=None, n=None, seed=0):
    """(a − b) 의 데이터셋 평균의 평균, 데이터셋 안에서 사진을 다시 뽑는 bootstrap 95 % 구간."""
    n = int(n or PLAN["boot"])
    rng = np.random.default_rng(seed)
    dss = [d for d in sorted(df.ds.unique()) if groups is None or d in groups]
    vals = [(df[df.ds == d][a] - df[df.ds == d][b]).to_numpy() for d in dss]
    point = float(np.mean([v.mean() for v in vals]))
    bs = np.mean([v[rng.integers(0, len(v), (n, len(v)))].mean(1) for v in vals], axis=0)
    return point, float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))
