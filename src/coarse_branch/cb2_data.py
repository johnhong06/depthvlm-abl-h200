"""분리 디코딩 실험 1 단계의 자료 (PROMPT_v2 5 · 6.1 절): 학습 순서 (정답 1 : 무라벨 2) · 증강 · 굵은 칸 목표.

학습 순서  자리 k = 3j → 정답 사진 j 번째 (프로젝트 학습 순서 = d113 표의 pos) · 3j + 1, 3j + 2 → 무라벨 사진 2j, 2j + 1 번째 (cb2_common.unlabeled 의 pos, 선생이 못 낸 사진은 건너뜀).
증강      사진 번호로 정해진 난수 (seed, k): 좌우 뒤집기 0.5 · 무작위 자르기 (넓이 0.5–1.0, 가로세로 비 ± 20 %, 토큰 칸 단위 — 크기는 안 바꿈) · 색 흔들기 · 흐림 0.2.
굵은 칸    짧은 변 16 칸. 세 팔이 같은 목표를 쓰도록 격자는 '작은 사진 토큰 격자' (th × tw ≈ 147) 에서 정한다.
목표      16 화소 칸 지도 (정답 · 전문가 · 하늘 · 선생 불확실도) 를 자른 뒤 굵은 칸으로 무게 평균:
           정답 사진   Gc · wg (정답 칸 평균 · 정답 비율) / Tc · qc (정답 없는 칸은 정답에 상수 하나로 맞춘 전문가 · 하늘 아닌 비율) / gate (전문가 δ1 ≥ 0.95)
           무라벨 사진 Tc · qc (선생 · 하늘 아닌 칸 중 선생 불확실도 Q 가 큰 20 % 는 뺌) — 11-11 cb2_qcheck: confidence 는 클수록 오차가 큼
사진      원본에서 필요한 영역만 한 번에 잘라 줄인다 (PIL resize 의 box). kind = 'token' 이면 canonical 크기 (32 화소 × 토큰 수), 'thumb' 이면 작은 사진.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

import cb2_common as C
import cb_common as B

HOME = Path.home()
DATA = HOME / "data/dvtrain/datasets"
ANN = HOME / "data/dvtrain/annotations"
D113 = HOME / "data/dvsilog/d113"
SEED, N_THUMB, K_SHORT, N_GEO, MIN_T = 1111, 147, 16, 150, 8
AREA, ASPECT_JIT, GATE_D1, Q_DROP = (0.5, 1.0), 1.2, 0.95, 0.20
HOLD_FRAC, N_U2 = 2 / 15, C.N8            # 2 단계용으로 떼어 두는 정답 장면의 몫 (5 만 장의 1/3 = 16,667 장) · 무라벨 앞 34,000 장 (8 화소 칸 지도가 있는 것)
MAX_EDGE = {"taskonomy": 1024, "hm3d": 1024}                      # 공식 DATASET_CONFIGS 의 max_canonical_edge (가운데 자르기)
_RECS = {}


def recs(split):
    if split not in _RECS:
        _RECS[split] = [json.loads(l) for l in open(ANN / f"{split}_mix_spp.jsonl")]
    return _RECS[split]


# ---------------------------------------------------------------- 격자
def thumb_tokens(h, w):
    th = max(4, round(math.sqrt(N_THUMB * h / w)))
    return th, max(4, round(N_THUMB / th))


def coarse_grid(th, tw):
    return (K_SHORT, max(K_SHORT, round(K_SHORT * tw / th))) if tw >= th else (max(K_SHORT, round(K_SHORT * th / tw)), K_SHORT)


def geo_grid(h, w):
    gh = max(2, round(math.sqrt(N_GEO * h / w)))
    return gh, max(2, round(N_GEO / gh))


def sample_aug(k, ph, pw, seed=SEED):
    """자리 k 의 증강 (뒤집기, 자르기 상자 y · x · h · w — 토큰 칸)."""
    rng = np.random.default_rng([int(seed), int(k)])
    flip = bool(rng.random() < 0.5)
    area, jit = rng.uniform(*AREA), math.exp(rng.uniform(-math.log(ASPECT_JIT), math.log(ASPECT_JIT)))
    a = pw / ph * jit
    h = math.sqrt(area * ph * pw / a)
    w = h * a
    h, w = int(min(ph, max(min(MIN_T, ph), round(h)))), int(min(pw, max(min(MIN_T, pw), round(w))))
    return flip, int(rng.integers(0, ph - h + 1)), int(rng.integers(0, pw - w + 1)), h, w


# ---------------------------------------------------------------- 학습 순서
def labeled_table():
    e = pd.read_parquet(D113 / "expert/expert.parquet")
    e = e[e.split == "train"].sort_values("pos")[["i", "pos", "ph", "pw"]].reset_index(drop=True)
    assert (e.pos.values == np.arange(len(e))).all() and len(e) == len(recs("train"))
    e["ds"] = [recs("train")[i]["dataset"] for i in e.i]
    return e


def unlabeled_table(allow_partial=False):
    """선생 · 하늘 지도가 끝난 무라벨 사진 (선생이 못 낸 사진은 빠짐) — pos 순."""
    tp, sp = sorted((C.V2 / "teacher/parts").glob("t_*.parquet")), sorted((C.V2 / "sky/parts").glob("s_*.parquet"))
    t, s = pd.concat([pd.read_parquet(p) for p in tp], ignore_index=True), pd.concat([pd.read_parquet(p) for p in sp], ignore_index=True)
    n = min(len(t), len(s))
    assert allow_partial or n == 250_000, f"지도가 아직 덜 됨 (선생 {len(t)} · 하늘 {len(s)})"
    t = t.iloc[:n].merge(s.iloc[:n][["pos", "sky_frac"]], on="pos", validate="one_to_one")
    assert (t.pos.values == np.arange(n)).all()
    t = t[t.ok].reset_index(drop=True)
    for c in ("cw", "ch", "ph", "pw"):
        t[c] = t[c].astype(int)
    return t


def unlabeled_table_b():
    """추가 무라벨 목록 (11-12, pos ≥ 250,000) 중 선생 · 하늘 지도가 끝난 사진 — pos 순. 지도가 덜 됐으면 멈춘다."""
    tp, sp = sorted((C.V2 / "teacher/parts_b").glob("t_*.parquet")), sorted((C.V2 / "sky/parts_b").glob("s_*.parquet"))
    t, s = pd.concat([pd.read_parquet(p) for p in tp], ignore_index=True), pd.concat([pd.read_parquet(p) for p in sp], ignore_index=True)
    n = len(C.unlabeled_b())
    assert len(t) == len(s) == n, f"추가 목록의 지도가 아직 덜 됨 (선생 {len(t)} · 하늘 {len(s)} / {n})"
    t = t.merge(s[["pos", "sky_frac"]], on="pos", validate="one_to_one")
    assert (t.pos.values == C.N_A + np.arange(n)).all()
    t = t[t.ok].reset_index(drop=True)
    for c in ("cw", "ch", "ph", "pw"):
        t[c] = t[c].astype(int)
    return t


def scene_of(rec):
    return str(rec.get("scene") or rec.get("scene_id") or rec["image"].split("/")[1])


def holdout_table():
    """정답 사진 표 (학습 순서) + s2: 2 단계용으로 떼어 둔 장면의 사진이면 True (사용자 결정 11-11).
    데이터셋마다 장면 이름을 정렬해 seed 1111 로 섞고, 그 데이터셋 사진의 2/15 에 가장 가까워질 때까지 앞에서부터 장면을 뗀다 (장면 단위 — 한 장면이 두 쪽에 걸치지 않음)."""
    L = labeled_table()
    L["scene"] = [scene_of(recs("train")[i]) for i in L.i]
    rng, hold = np.random.default_rng(SEED), set()
    for ds, x in L.groupby("ds"):
        cnt = x.scene.value_counts()
        names = sorted(cnt.index)
        target, acc = len(x) * HOLD_FRAC, 0
        for s_ in (names[j] for j in rng.permutation(len(names))):
            if abs(acc + cnt[s_] - target) > abs(acc - target):
                break
            hold.add((ds, s_))
            acc += cnt[s_]
    L["s2"] = [(d, s_) in hold for d, s_ in zip(L.ds, L.scene)]
    return L


def build_stream(variant, allow_partial=False, n_items=None):
    """학습 순서 표: k · kind (L / U) · src (정답 = jsonl 줄 i, 무라벨 = pos) · ph · pw · 증강 · 작은 사진 격자.
    variant  s1      1 단계: 떼어 두지 않은 장면의 정답 사진 + 무라벨 pos ≥ N_U2 (정답 1 : 무라벨 2)
             s1_nou  1 단계의 '무라벨 없는' 대조: s1 이 쓰는 정답 사진만 세 번 (걸음 수는 s1 과 같고 증강은 번마다 다름)
             s2      2 단계: 떼어 둔 장면의 정답 사진 + 무라벨 pos < N_U2 (1 단계가 보지 않은 사진)
             s1x3    s1 을 세 번 (진단용)
             s1b     (11-12) s1 의 사진 전부 + 추가 무라벨 목록 전부를 한 번씩 (_rows_s1b)
             all     떼어 두기 없이 전부 (시험용)"""
    if variant == "s1b":
        return _finish(_rows_s1b(), n_items, k0=K0_S1B)
    L, U = holdout_table(), unlabeled_table(allow_partial)
    if variant in ("s1", "s1_nou", "s1x3"):
        L, U = L[~L.s2], U[U.pos >= N_U2]
    elif variant == "s2":
        L, U = L[L.s2], U[U.pos < N_U2]
    else:
        assert variant == "all", variant
    L, U = L.reset_index(drop=True), U.reset_index(drop=True)
    n = min(len(L), len(U) // 2)
    rows = []
    if variant == "s1_nou":
        for _ in range(3):
            rows += [("L", int(i), int(ph), int(pw)) for i, ph, pw in zip(L.i[:n], L.ph[:n], L.pw[:n])]
    else:
        for j in range(n):
            rows.append(("L", int(L.i[j]), int(L.ph[j]), int(L.pw[j])))
            rows += [("U", int(U.pos[m]), int(U.ph[m]), int(U.pw[m])) for m in (2 * j, 2 * j + 1)]
    if variant == "s1x3":                                             # 진단용 (11-11 19:4x): s1 과 같은 사진을 세 번 (증강은 번마다 다름) — '한 번 보기' 가 모자랐는지 확인
        rows = rows * 3
    return _finish(rows, n_items)


K0_S1B = 1_000_000               # 's1b' 의 자리 번호가 시작하는 값 — 증강 씨앗 (자리 번호) 이 첫 통과 (k < 323,997) · 세 번 보기 (k < 971,991) 와 겹치지 않게


def _rows_s1b():
    """(11-12 사용자 결정) 1 단계 다시: 's1' 이 쓴 사진 전부 (정답 107,999 + 무라벨 215,998, 같은 사진 · 다른 증강) + 추가 무라벨 목록 전부를 한 번씩.
    무라벨은 두 목록을 합쳐 한 번 섞고 (새 사진이 한쪽에 몰리지 않게), 정답 사진은 's1' 의 순서 그대로 그 사이에 고르게 끼운다. 2 단계용 사진은 여전히 안 봄."""
    s1 = build_stream("s1")
    L, Ua, Ub = s1[s1.kind == "L"], s1[s1.kind == "U"], unlabeled_table_b()
    U = pd.DataFrame(dict(src=Ua.src.tolist() + Ub.pos.tolist(), ph=Ua.ph.tolist() + Ub.ph.tolist(), pw=Ua.pw.tolist() + Ub.pw.tolist()))
    assert not U.src.duplicated().any() and (U.src >= N_U2).all()
    U = U.iloc[np.random.default_rng([SEED, 21]).permutation(len(U))].reset_index(drop=True)
    key = np.concatenate([(np.arange(len(L)) + 0.5) / len(L), (np.arange(len(U)) + 0.5) / len(U)])
    rows = [("L", int(i), int(ph), int(pw)) for i, ph, pw in zip(L.src, L.ph, L.pw)] + [("U", int(i), int(ph), int(pw)) for i, ph, pw in zip(U.src, U.ph, U.pw)]
    return [rows[j] for j in np.argsort(key, kind="stable")]


def _finish(rows, n_items=None, k0=0):
    t = pd.DataFrame(rows[: n_items or None], columns=["kind", "src", "ph", "pw"])
    t.insert(0, "k", k0 + np.arange(len(t)))
    aug = [sample_aug(k, ph, pw) for k, ph, pw in zip(t.k, t.ph, t.pw)]
    for c, v in zip(("flip", "y", "x", "h", "w"), zip(*aug)):
        t[c] = v
    tt = [thumb_tokens(h, w) for h, w in zip(t.h, t.w)]
    t["th"], t["tw"] = [a_ for a_, _ in tt], [b_ for _, b_ in tt]
    return t


def batches(stream, bs=8, window=512):
    """작은 사진 크기 (th, tw) 가 같은 자리끼리 bs 장 묶음. 학습 순서를 window 장씩 훑으며 채워진 묶음만 내고 남은 것은 다음 창으로 넘긴다 (끝에서 한 번 털어 냄)."""
    pend, out = {}, []
    keys = list(zip(stream.th.tolist(), stream.tw.tolist()))
    for s in range(0, len(stream), window):
        for k in range(s, min(len(stream), s + window)):
            g = pend.setdefault(keys[k], [])
            g.append(k)
            if len(g) == bs:
                out.append(g[:])
                g.clear()
    out += [g for g in pend.values() if g]
    return out


# ---------------------------------------------------------------- 사진
def _open(path, need_wh, frac_box):
    """원본에서 frac_box (x0, y0, x1, y1 — 0–1) 를 need_wh 크기로. JPEG 는 필요한 만큼만 풀어 읽는다 (draft)."""
    im = Image.open(path)
    W0, H0 = im.size
    bw, bh = (frac_box[2] - frac_box[0]) * W0, (frac_box[3] - frac_box[1]) * H0
    if im.format == "JPEG" and min(bw / need_wh[0], bh / need_wh[1]) >= 2.0:
        s = min(bw / need_wh[0], bh / need_wh[1])
        im.draft("RGB", (math.ceil(W0 / s) + 1, math.ceil(H0 / s) + 1))
    im = im.convert("RGB")
    W, H = im.size
    return im.resize(need_wh, Image.BILINEAR, box=(frac_box[0] * W, frac_box[1] * H, frac_box[2] * W, frac_box[3] * H))


def canon_frac(rec):
    """정답 사진: 공식 로더가 쓰는 영역 (canonical 가운데 자르기) 을 원본의 비율 상자로."""
    cs, me = rec.get("canonical_size"), MAX_EDGE.get(rec["dataset"])
    if not cs or me is None or max(cs) <= me:
        return 0.0, 0.0, 1.0, 1.0
    cw, ch = min(cs[0], me), min(cs[1], me)
    left, top = (cs[0] - cw) // 2, (cs[1] - ch) // 2
    return left / cs[0], top / cs[1], (left + cw) / cs[0], (top + ch) / cs[1]


def view_box(full, ph, pw, y, x, h, w):
    """full (원본 비율 상자) 안에서 토큰 칸 자르기 상자를 원본 비율 상자로."""
    fx, fy = (full[2] - full[0]) / pw, (full[3] - full[1]) / ph
    return full[0] + x * fx, full[1] + y * fy, full[0] + (x + w) * fx, full[1] + (y + h) * fy


def jitter(img, rng, scale=1.0):
    """색 흔들기 (밝기 · 대비 · 채도 0.2, 색조 0.05) + 흐림 0.2 — 학생 입력에만. 흐림 반지름은 canonical 화소 0.5–2.0 (scale = 출력 / canonical)."""
    import torchvision.transforms.functional as TF
    b, c, s = (float(rng.uniform(0.8, 1.2)) for _ in range(3))
    hue = float(rng.uniform(-0.05, 0.05))
    ops = (lambda im: TF.adjust_brightness(im, b), lambda im: TF.adjust_contrast(im, c), lambda im: TF.adjust_saturation(im, s), lambda im: TF.adjust_hue(im, hue))
    for f in rng.permutation(4):
        img = ops[int(f)](img)
    if rng.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.5, 2.0)) * scale))
    return img


def to_u8(img):
    return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).contiguous()


DEPTHVLM = ("JonnyYu828/DepthVLM-4B", "2b2d02fcfe0c89c8aa7d541055e5a078930077c9")
_IP = None


def model_pixels(img):
    """PIL 사진 (32 의 배수) → DepthVLM 이 실제로 받는 패치 (N, 1536) 와 패치 격자 (Hp, Wp).
    본 통과와 같은 전처리: qwen_vl_utils.fetch_image (28 의 배수로 bicubic) → 처리기 (32 의 배수로 bicubic · 정규화 · 패치) — geo_mix.vlm_inputs 의 pixel_values 와 비트까지 같다 (11-11 확인).
    읽기 전용 토큰은 본 통과의 패치 특징을 읽으므로, 학습 때도 본 통과가 실제로 보는 사진을 넣는다."""
    global _IP
    if _IP is None:
        from transformers import AutoProcessor
        _IP = AutoProcessor.from_pretrained(DEPTHVLM[0], revision=DEPTHVLM[1]).image_processor
    from qwen_vl_utils.vision_process import fetch_image
    o = _IP(images=[fetch_image({"type": "image", "image": img})], return_tensors="pt")
    _, Hp, Wp = (int(v) for v in o["image_grid_thw"][0])
    assert (16 * Wp, 16 * Hp) == img.size, ("처리기가 크기를 바꿈", img.size, Hp, Wp)
    return o["pixel_values"], Hp, Wp


# ---------------------------------------------------------------- 지도
def _npy(path):
    return torch.from_numpy(np.load(path).astype(np.float32))


def labeled_maps16(i, ph, pw):
    """정답 사진의 16 화소 칸 지도 (2ph × 2pw): G · w · E · sky."""
    z = np.load(C.map_path(C.V2 / "gt16", i))
    sub = f"{i // 1000:03d}/{i}.npy"
    E = F.avg_pool2d(_npy(D113 / "expert/maps/train" / sub)[None, None], 2)[0, 0]
    sky = F.avg_pool2d(_npy(D113 / "sky/maps/train" / sub)[None, None], 2)[0, 0] / 255.0
    G, w = torch.from_numpy(z["G"].astype(np.float32)), torch.from_numpy(z["w"].astype(np.float32))
    assert G.shape == E.shape == sky.shape == (2 * ph, 2 * pw), (i, G.shape, E.shape, sky.shape, ph, pw)
    return G, w, E, sky


def unlabeled_maps16(pos, ph, pw):
    """무라벨 사진의 16 화소 칸 지도: E · Q · sky."""
    z = np.load(C.map_path(C.V2 / "teacher/m16", pos))
    S = torch.from_numpy(np.load(C.map_path(C.V2 / "sky/maps", pos))["S"].astype(np.float32)) / 255.0
    sky = F.interpolate(S[None, None], size=(2 * ph, 2 * pw), mode="area")[0, 0]
    E, Q = torch.from_numpy(z["E"].astype(np.float32)), torch.from_numpy(z["Q"].astype(np.float32))
    assert E.shape == (2 * ph, 2 * pw), (pos, E.shape, ph, pw)
    return E, Q, sky


def labeled_maps8(i, ph, pw):
    """2 단계용 정답 사진의 8 화소 칸 지도 (4ph × 4pw): G · w (cb2_gt.py --s2) · E · sky (d113 저장분 그대로)."""
    z = np.load(C.map_path(C.V2 / "gt8", i))
    sub = f"{i // 1000:03d}/{i}.npy"
    G, w = torch.from_numpy(z["G"].astype(np.float32)), torch.from_numpy(z["w"].astype(np.float32))
    E, sky = _npy(D113 / "expert/maps/train" / sub), _npy(D113 / "sky/maps/train" / sub) / 255.0
    assert G.shape == E.shape == sky.shape == (4 * ph, 4 * pw), (i, G.shape, E.shape, sky.shape, ph, pw)
    return G, w, E, sky


def unlabeled_maps8(pos, ph, pw):
    """2 단계용 무라벨 사진 (pos < N_U2) 의 8 화소 칸 지도: E · Q · sky."""
    z = np.load(C.map_path(C.V2 / "teacher/m8", pos))
    S = torch.from_numpy(np.load(C.map_path(C.V2 / "sky/maps", pos))["S"].astype(np.float32)) / 255.0
    sky = F.interpolate(S[None, None], size=(4 * ph, 4 * pw), mode="bilinear", align_corners=False)[0, 0].clamp(0, 1)   # 하늘 지도 (원본의 1/8) 가 더 굵어 늘려야 함
    E, Q = torch.from_numpy(z["E"].astype(np.float32)), torch.from_numpy(z["Q"].astype(np.float32))
    assert E.shape == (4 * ph, 4 * pw), (pos, E.shape, ph, pw)
    return E, Q, sky


def coarse_targets(kind, maps, ky, kx):
    """16 화소 칸 지도 (자르고 뒤집은 뒤) → 굵은 칸 목표."""
    if kind == "L":
        G, w, E, sky = (m[None] for m in maps)
        q = (1.0 - sky).clamp(0, 1)
        has, d1, T = B.gate_and_target(G, w, E)
        Gc, wg = B.mpool(G, w, ky, kx)
        Tc, qc = B.mpool(T, q, ky, kx)
        gate = bool(has[0]) and float(d1[0]) >= GATE_D1
        has = bool(has[0])
    else:
        E, Q, sky = (m[None] for m in maps)
        q = (1.0 - sky).clamp(0, 1)
        clear = q > 0.5
        keep = (Q <= torch.quantile(Q[clear], 1.0 - Q_DROP)) if int(clear.sum()) >= 10 else torch.ones_like(clear)
        Tc, qc = B.mpool(E, q * keep.float(), ky, kx)
        Gc, wg, gate, has = torch.zeros_like(Tc), torch.zeros_like(qc), True, False
    skyc = F.adaptive_avg_pool2d(sky[None], (ky, kx))[0]
    return dict(Gc=Gc[0], wg=wg[0], Tc=Tc[0], qc=qc[0], skyc=skyc[0], has=float(has), gate=float(gate), unl=float(kind == "U"))


# ---------------------------------------------------------------- 데이터셋
class StreamSet(torch.utils.data.Dataset):
    """학습 순서의 자리 k → 학생 입력 (uint8) + 굵은 칸 목표. kind: 'token' (canonical 크기) | 'thumb' (작은 사진)."""

    def __init__(self, stream, kind, augment=True):
        self.s, self.kind, self.augment = stream.reset_index(drop=True), kind, augment
        self.U = None

    def _unl(self):
        if self.U is None:
            u = C.unlabeled_all()                                  # 첫 목록 + (있으면) 추가 목록 — 첫 목록의 pos · 경로는 그대로
            self.U = dict(zip(u.pos.tolist(), u.path.tolist()))
        return self.U

    def __len__(self):
        return len(self.s)

    def __getitem__(self, n):
        torch.set_num_threads(1)
        r = self.s.iloc[n]
        k, ph, pw = int(r.k), int(r.ph), int(r.pw)
        flip, y, x, h, w = (bool(r.flip), int(r.y), int(r.x), int(r.h), int(r.w)) if self.augment else (False, 0, 0, ph, pw)
        th, tw = thumb_tokens(h, w)
        ky, kx = coarse_grid(th, tw)
        if r.kind == "L":
            rec = recs("train")[int(r.src)]
            path, full = DATA / rec["image"].lstrip("/"), canon_frac(rec)
            maps = labeled_maps16(int(r.src), ph, pw)
        else:
            path, full = C.PLACES / self._unl()[int(r.src)], (0.0, 0.0, 1.0, 1.0)
            maps = unlabeled_maps16(int(r.src), ph, pw)
        size = (32 * w, 32 * h) if self.kind == "token" else (32 * tw, 32 * th)
        img = _open(path, size, view_box(full, ph, pw, y, x, h, w))
        maps = [m[2 * y: 2 * (y + h), 2 * x: 2 * (x + w)] for m in maps]
        if self.augment:
            img = jitter(img, np.random.default_rng([SEED, k, 7]), size[0] / (32.0 * w))
        if flip:
            img, maps = img.transpose(Image.FLIP_LEFT_RIGHT), [m.flip(-1) for m in maps]
        out = coarse_targets(r.kind, maps, ky, kx)
        out.update(k=k, nh=h, nw=w)                                    # nh · nw = 넣는 사진의 토큰 칸 수 (자른 뒤)
        if self.kind == "token":
            out["pv"] = model_pixels(img)[0]
        else:
            out["u8"] = to_u8(img)
        return out


_PROC = None
PROMPT = "Given this image, estimate the metric depth (in meters, rounded to two decimal places) for every pixel of the image."   # 공식 DEFAULT_DEPTH_PROMPT


def vlm_inputs(img):
    """analysis/geo_mix.vlm_inputs 와 같은 호출 (작업자 안에서 처리기를 한 번 만든다) — 본 통과 입력 (input_ids · attention_mask · pixel_values · image_grid_thw)."""
    global _PROC
    if _PROC is None:
        from transformers import AutoProcessor
        _PROC = AutoProcessor.from_pretrained(DEPTHVLM[0], revision=DEPTHVLM[1])
        _PROC.tokenizer.padding_side = "left"
    from qwen_vl_utils import process_vision_info
    msg = [{"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
           {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
    inp = _PROC(text=[_PROC.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)], images=[process_vision_info(msg)[0]], padding=True, return_tensors="pt")
    return {k: inp[k] for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}


class S2Set(torch.utils.data.Dataset):
    """2 단계 학습 순서의 자리 → 본 통과 입력 + 8 화소 칸 목표 (coarse_targets 를 풀링 없이 — 격자 = 4·nh × 4·nw)."""

    def __init__(self, stream, augment=True):
        self.s, self.augment, self.U = stream.reset_index(drop=True), augment, None

    def __len__(self):
        return len(self.s)

    def __getitem__(self, n):
        torch.set_num_threads(1)
        r = self.s.iloc[n]
        k, ph, pw = int(r.k), int(r.ph), int(r.pw)
        flip, y, x, h, w = (bool(r.flip), int(r.y), int(r.x), int(r.h), int(r.w)) if self.augment else (False, 0, 0, ph, pw)
        if r.kind == "L":
            rec = recs("train")[int(r.src)]
            path, full, maps = DATA / rec["image"].lstrip("/"), canon_frac(rec), labeled_maps8(int(r.src), ph, pw)
        else:
            if self.U is None:
                u = C.unlabeled()
                self.U = dict(zip(u.pos.tolist(), u.path.tolist()))
            path, full, maps = C.PLACES / self.U[int(r.src)], (0.0, 0.0, 1.0, 1.0), unlabeled_maps8(int(r.src), ph, pw)
        img = _open(path, (32 * w, 32 * h), view_box(full, ph, pw, y, x, h, w))
        maps = [m[4 * y: 4 * (y + h), 4 * x: 4 * (x + w)] for m in maps]
        if self.augment:
            img = jitter(img, np.random.default_rng([SEED, k, 7]), 1.0)
        if flip:
            img, maps = img.transpose(Image.FLIP_LEFT_RIGHT), [m.flip(-1) for m in maps]
        out = coarse_targets(r.kind, maps, 4 * h, 4 * w)
        out.update(k=k, nh=h, nw=w, inp=vlm_inputs(img))
        return out


def mp(workers):
    """작업자를 띄우는 방식: forkserver (스레드가 많은 본 프로세스를 그대로 fork 하면 잠긴 채 멈출 수 있다 — 11-11 평가 시험에서 실제로 멈춤) + 15 분 안에 자료가 안 오면 예외 (조용히 멈추지 않게)."""
    return dict(num_workers=workers, multiprocessing_context="forkserver", timeout=900) if workers else dict(num_workers=0)


def collate(items):
    out = {}
    for key in items[0]:
        v = [it[key] for it in items]
        out[key] = torch.stack(v) if torch.is_tensor(v[0]) else torch.tensor(v)
    return out
