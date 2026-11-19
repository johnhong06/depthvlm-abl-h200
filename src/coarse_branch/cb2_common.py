"""분리 디코딩 실험 (PROMPT_v2) 의 공통 부품 — 경로 · 무라벨 사진의 학습 순서 · canonical 크기. numpy · pandas 만 쓴다 (여러 venv 에서 같이 씀).

무라벨 사진의 학습 순서: 목록 (unlabeled_places365.parquet) 을 seed 1111 로 한 번 섞은 자리 pos (0 … 249,999). 선생 · 하늘 지도도 이 순서로 만든다.
canonical 크기 (PROMPT_v2 5.4): 초점 1000 으로 맞춘 크기 → 긴 변을 [896, 1792] 로 → 32 의 배수 (토큰 한 칸 = 32 화소).
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

V2 = Path("~/data/dvsilog/coarse_branch/v2data").expanduser()
PLACES = Path("~/data/unlabeled/places365/images").expanduser()
SEED_ORDER = 1111
N8 = 34_000                      # 8 화소 칸 지도를 함께 저장하는 장 수 = 2 단계가 쓰는 무라벨 33,333 장 + 빠질 사진 여유
CANON_MIN, CANON_MAX = 896, 1792
BLOCK = 5000


def unlabeled():
    """무라벨 목록을 학습 순서로 — 열: pos · idx (목록의 줄) · split · path · W · H."""
    t = pd.read_parquet(V2 / "unlabeled_places365.parquet")
    perm = np.random.default_rng(SEED_ORDER).permutation(len(t))
    t = t.iloc[perm].reset_index(names="idx")
    t.insert(0, "pos", np.arange(len(t)))
    return t


N_A = 250_000                    # 첫 목록의 장 수 — 추가 목록 (b) 의 pos 는 여기서부터 (첫 목록의 지도 · 학습 순서는 그대로)


def unlabeled_b():
    """추가 무라벨 목록 (11-12 사용자 결정: Places365 에서 처음에 안 고른 사진 전부, unlabeled_places365_b.parquet) 을 학습 순서로 — pos 는 250,000 부터."""
    t = pd.read_parquet(V2 / "unlabeled_places365_b.parquet")
    perm = np.random.default_rng(SEED_ORDER + 1).permutation(len(t))
    t = t.iloc[perm].reset_index(names="idx")
    t.insert(0, "pos", N_A + np.arange(len(t)))
    return t


GLDV2 = Path("~/data/unlabeled/gldv2/images").expanduser()
N_C0 = 400_000                   # 추가 목록 c (Google Landmarks v2, 11-12) 의 pos 가 시작하는 값 (a: 0 … 249,999 · b: 250,000 … 363,086)


def unlabeled_c():
    """추가 무라벨 목록 c (11-12 사용자 지시: Google Landmarks v2 를 보고 쓸 만하면 사용, unlabeled_gldv2.parquet) 를 학습 순서로 — pos 는 400,000 부터."""
    t = pd.read_parquet(V2 / "unlabeled_gldv2.parquet")
    perm = np.random.default_rng(SEED_ORDER + 2).permutation(len(t))
    t = t.iloc[perm].reset_index(names="idx")
    t.insert(0, "pos", N_C0 + np.arange(len(t)))
    return t


BDD = Path("~/data/unlabeled/bdd100k/100k").expanduser()
N_D0 = 600_000                   # 추가 목록 d (BDD100K 주행 사진, 11-12) 의 pos 가 시작하는 값 (c: 400,000 … 520,204)


def unlabeled_d():
    """추가 무라벨 목록 d (11-12 사용자 지시: 주행 시점 사진도 포함, unlabeled_bdd100k.parquet) 를 학습 순서로 — pos 는 600,000 부터."""
    t = pd.read_parquet(V2 / "unlabeled_bdd100k.parquet")
    perm = np.random.default_rng(SEED_ORDER + 3).permutation(len(t))
    t = t.iloc[perm].reset_index(names="idx")
    t.insert(0, "pos", N_D0 + np.arange(len(t)))
    return t


SA1B = Path("~/data/unlabeled/sa1b/images").expanduser()
N_E0 = 800_000                   # 추가 목록 e (SA-1B, 11-13) 의 pos 가 시작하는 값 (d: 600,000 … 699,999)


def unlabeled_e():
    """추가 무라벨 목록 e (11-12 사용자 지시: 실내 · 실외 사진이 부족하면 SA-1B, unlabeled_sa1b.parquet) 를 학습 순서로 — pos 는 800,000 부터."""
    t = pd.read_parquet(V2 / "unlabeled_sa1b.parquet")
    perm = np.random.default_rng(SEED_ORDER + 4).permutation(len(t))
    t = t.iloc[perm].reset_index(names="idx")
    t.insert(0, "pos", N_E0 + np.arange(len(t)))
    return t


PLACES_TRAIN = Path("~/data/unlabeled/places365_train/images").expanduser()
N_F0 = 1_000_000                 # 추가 목록 f (Places365-Standard train 에서 고른 것, 11-18) 의 pos 가 시작하는 값 (e: 800,000 … 899,999 안)


def unlabeled_f():
    """추가 무라벨 목록 f (11-18 사용자 결정: 새 사진으로 학습량을 늘린다 — Places365 train 에서 실내 종류 위주 + 실외 조금, unlabeled_places365_train.parquet) 를 학습 순서로 — pos 는 1,000,000 부터."""
    t = pd.read_parquet(V2 / "unlabeled_places365_train.parquet")
    perm = np.random.default_rng(SEED_ORDER + 5).permutation(len(t))
    t = t.iloc[perm].reset_index(names="idx")
    t.insert(0, "pos", N_F0 + np.arange(len(t)))
    return t


COCO = Path("~/data/unlabeled/coco/images").expanduser()   # 로컬 디스크에 자리가 없어 외장하드에 둔다 (11-18)
N_G0 = 1_700_000                 # 추가 목록 g (COCO train2017 + unlabeled2017, 11-18) 의 pos 가 시작하는 값 (f: 1,000,000 … 1,699,999 안)


def unlabeled_g():
    """추가 무라벨 목록 g (11-18 사용자 결정: COCO — 가까운 일상 장면, unlabeled_coco.parquet) 를 학습 순서로 — pos 는 1,700,000 부터."""
    t = pd.read_parquet(V2 / "unlabeled_coco.parquet")
    perm = np.random.default_rng(SEED_ORDER + 6).permutation(len(t))
    t = t.iloc[perm].reset_index(names="idx")
    t.insert(0, "pos", N_G0 + np.arange(len(t)))
    return t


def root(which="a"):
    """목록의 사진이 있는 폴더 (a · b = Places365 val / test, c = Google Landmarks, d = BDD100K, e = SA-1B, f = Places365 train, g = COCO)."""
    return {"c": GLDV2, "d": BDD, "e": SA1B, "f": PLACES_TRAIN, "g": COCO}.get(which, PLACES)


def unlabeled_list(which="a"):
    assert which in ("a", "b", "c", "d", "e", "f", "g"), which
    return {"a": unlabeled, "b": unlabeled_b, "c": unlabeled_c, "d": unlabeled_d, "e": unlabeled_e, "f": unlabeled_f, "g": unlabeled_g}[which]()


def unlabeled_all():
    """첫 목록 + (있으면) 추가 목록 — pos 로 사진 경로를 찾을 때."""
    return pd.concat([unlabeled(), unlabeled_b()], ignore_index=True) if (V2 / "unlabeled_places365_b.parquet").exists() else unlabeled()


def canon_size(W, H, fx):
    """(가로, 세로) 원본 화소 · 초점 fx → canonical (가로, 세로)."""
    if not (math.isfinite(fx) and fx > 0):
        raise ValueError(f"fx {fx}")
    s = 1000.0 / fx
    L = max(W, H) * s
    s *= min(max(CANON_MIN / L, 1.0), CANON_MAX / L) if not (CANON_MIN <= L <= CANON_MAX) else 1.0
    return max(32, round(W * s / 32) * 32), max(32, round(H * s / 32) * 32)


def map_path(root, pos):
    return Path(root) / f"{int(pos) // 1000:03d}" / f"{int(pos)}.npz"


def save_atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def write_part(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.parquet")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(path)
