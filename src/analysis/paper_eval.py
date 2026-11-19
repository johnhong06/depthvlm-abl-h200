"""논문 규칙 (Metric3D 공식 평가) 으로 4 세트를 다시 잰다 (사용자 10-13: "논문과 거의 같으면 논문 수치를 쓰고, 논문에 없는 SILog·전역 스케일 오차는
같은 규칙으로 우리가 채점" · "Metric3Dv2 채점 방식이 정당하면 그 방식으로"). NOTES D-19.

규칙 (Metric3D training/mono 공식 평가 — test_configs_vit 의 clip_depth_range, 예측은 자르지 않음, 모든 지표 사진별 평균):
  NYUv2  표준 654 장 (BTS 형식, depthvlm-finetune 이 만든 것, 정답 png ÷ 1000) · Eigen crop [45:471, 41:601] · 0.1–10 m
  KITTI  Eigen 652 장 annotated (D-18 과 같은 사진) · Eigen crop (0.3324–0.9135 H, 0.0359–0.9640 W) · 0.1–80 m
  iBims-1 100 장 · crop 없음 · 0.1–10 m        DIODE Outdoor 446 장 · crop 없음 · 0.1–150 m
지표: δ1 · AbsRel · RMSE · SILog = 100·√(mean d² − (mean d)²) · 전역 스케일 오차 = 100·√(mean_img (mean d)²), d = ln 예측 − ln 정답.
입력은 모델마다 공식 평가 그대로: Metric3Dv2 = NYU 테두리 6 px 검게 · KITTI 열 43:1197 (depthvlm-finetune predict.py --model metric3d 예측,
F-4·F-5 에서 논문 값 재현), 그 밖의 모델 = 사진 그대로. 비교용으로 Metric3Dv2 를 사진 그대로 넣은 판 (Metric3Dv2-L·raw) 도 함께 잰다.

단계:
  vlm   (~/venv/depthvlm, GPU) NYUv2·iBims-1·DIODE 에 원래 DepthVLM + 재학습 헤드 6 개 (kitti_eval 과 같은 코드) → pred/<세트>/<방법>/<id>.npy
  pv    (모델 환경, GPU)        NYUv2 에 UniDepthV2-L (K 없음) · +K · DAv2 실내 · Metric3Dv2 사진 그대로 (iBims-1·DIODE 는 vlm-depth-rmse 의 저장 지도를 씀)
  score (~/venv/main, CPU)      위 규칙으로 채점 → per_image.parquet · tables/paper_eval.csv (+ 논문 값 대조)
사용: ~/venv/depthvlm/bin/python -I analysis/paper_eval.py vlm · ~/venv/unidepth/bin/python -I analysis/paper_eval.py pv --model UniDepthV2-L (…) · ~/venv/main/bin/python -I analysis/paper_eval.py score
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
import kitti_eval as KE  # noqa: E402

D = os.path.expanduser("~/data/dvsilog/paper_eval")
DVFT = os.path.expanduser("~/data/dvft")
BENCH = os.path.expanduser("~/data/depthvlm_bench")
VDR_MAPS = os.path.expanduser("~/data/vdr_maps")
NYU_K = [518.8579, 519.4691, 325.58245, 253.73617]               # Metric3D NYU 테스트 intrinsics (depthvlm-finetune predict.py M3D_NYU_K)
RANGE = {"nyu_std": (0.1, 10.0), "kitti": (0.1, 80.0), "ibims1": (0.1, 10.0), "diode_outdoor": (0.1, 150.0), "sunrgbd": (0.1, 10.0), "eth3d": (0.1, 200.0)}
SETS = ("nyu_std", "ibims1", "kitti", "diode_outdoor", "sunrgbd", "eth3d")
SUN = os.path.expanduser("~/data/dvsilog/sunrgbd_nonyu.json")   # D-41: 공식 test − NYU 4,396 장 (analysis/prep_sunrgbd.py), 정답 = 센서 원본 depth/
INDOOR = ("nyu_std", "ibims1", "sunrgbd")
ETH = os.path.expanduser("~/data/dvsilog/eth3d_records.json")   # D-46: ETH3D 454 장 (analysis/prep_eth3d.py) — Depth Pro 방식: 왜곡 원본 jpg + 정답 원해상도 6048×4032 · 0.1–200 m
ETH_BENCH = os.path.expanduser("~/data/dvsilog/eth3d_bench_records.json")   # D-47 진단: DepthVLM-Bench 방식 (보정본 jpg, analysis/prep_eth3d_bench.py) — 채점은 analysis/d47_eth3d_bench.py
VALMIX = os.path.expanduser("~/data/dvsilog/val_mix_records.json")   # D-49 확인: 헤드 학습 검증 분할 600 장 (전문가 배율 정확도) — 채점은 analysis/d49_val_scale.py
ETH_DS = 3                                                          # ETH3D 예측은 1/3 크기 (2016×1344) 로 저장, 채점 때 정답 원해상도로 bilinear (Depth Pro 'preds bilinearly resized to GT')
ETH_OUTDOOR = ("courtyard", "electro", "facade", "meadow", "playground", "terrace")   # DAv2-metric 은 장면에 따라 실내 · 실외 모델
PV = ("UniDepthV2-L", "UniDepthV2-L+K", "Metric3Dv2-L", "DAv2-metric-L")
OURS = ("DepthVLM", *KE.HEADS)
OTHER = ("Youtu-VL-4B", "Youtu-VL-4B·NYU")                           # D-25 다른 VLM (analysis/youtu_eval.py 예측, ·NYU = NYUv2 전용 프롬프트 재현 확인)
EXTRA = ("DepthPro", "UniDepth-L", "ZoeDepth-NK", "DA3-metric-L")   # D-41 표 1 빈 칸 모델 (pv --sets, analysis/pv_extra.py · vlm-depth-rmse dense_sparse.depthpro)


def records(ds):
    """[(id, 사진 경로, K, W, H)] — NYU 는 depthvlm-finetune std_eval.records (BTS 공식 테스트 654), iBims-1·DIODE 는 geo_mix 평가 기록과 같은 사진."""
    from PIL import Image
    if ds == "sunrgbd":
        return [(r["id"], r["image"], r["k"], r["W"], r["H"]) for r in json.load(open(SUN))]
    if ds == "eth3d":                                                 # W · H = 원본 크기 (VLM 입력 크기 · 초점 계산에 씀), 저장만 1/3 (save)
        return [(r["id"], r["image"], r["k"], r["W"], r["H"]) for r in json.load(open(ETH))]
    if ds == "val_mix":
        return [(r["id"], r["image"], r["k"], r["W"], r["H"]) for r in json.load(open(VALMIX))]
    if ds == "eth3d_bench":
        return [(r["id"], r["image"], r["k"], r["W"], r["H"]) for r in json.load(open(ETH_BENCH))]
    if ds == "nyu_std":
        sys.path.insert(0, os.path.expanduser("~/code/depthvlm-finetune/eval"))
        import std_eval
        return [(rid, rgb, NYU_K, 640, 480) for rid, rgb, _, _ in std_eval.records("nyu", os.path.join(DVFT, "data"), os.path.join(DVFT, "splits"))]
    out = []
    for r in json.load(open(os.path.expanduser("~/data/dvsilog/records_eval.json"))):
        if r["set"] == ds:
            p = os.path.join(BENCH, r["image"])
            W, H = Image.open(p).size
            out.append((r["image"].rsplit(".", 1)[0].replace("/", "__"), p, r["k"], W, H))
    return out


def save(ds, model, rid, p, H, W):
    import cv2
    if ds in ("eth3d", "eth3d_bench", "val_mix"):                     # D-46 · D-47 · D-49: 저장 1/3 (원해상도 저장은 모델 10 × 454 × 49 MB)
        H, W = H // ETH_DS, W // ETH_DS
    if p.shape != (H, W):
        p = cv2.resize(p.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    os.makedirs(os.path.join(D, "pred", ds, model), exist_ok=True)
    np.save(os.path.join(D, "pred", ds, model, rid + ".npy"), p.astype(np.float16))


def vlm(a):
    import torch
    from PIL import Image
    sys.argv = sys.argv[:1]
    import geo_mix as G
    names = a.heads or list(KE.HEADS)                                 # --heads: 그 헤드만 (원래 DepthVLM·다른 헤드 예측은 다시 저장 안 함)
    proc, base, features = G.load_models(sorted({KE.HEADS[k][3] for k in names if KE.HEADS[k][3]}))
    hd = KE.load_heads(G, base, names)
    need_hr = any(hd[m].hproj is not None for m in names)            # D-24 헤드: 원래 해상도 지도도 받음
    for ds in (a.sets or ("nyu_std", "ibims1", "diode_outdoor")):       # D-41: --sets sunrgbd 등
        rs = records(ds)[:a.limit or None]
        t0 = time.time()
        for j, (rid, path, k, W, H) in enumerate(rs):
            img = G._load_and_resize_image(path, [round(W * 1000.0 / k[0]), round(H * 1000.0 / k[0])])   # geo_mix.evaluate 와 같음
            p0, feats, geo_c, (ph, pw) = features(G.vlm_inputs(proc, img), G.enc_input(img))
            x14n, box = G.unidepth_input(Image.open(path).convert("RGB"), box=True)
            x14t = G.tok_input(img, ph, pw) if any(KE.HEADS[m][4] == "t" for m in names) else None   # D-66 토큰 격자 입력 (같은 canonical 사진)
            _, _, geo_n, _, *gh = features(G.vlm_inputs(proc, img), x14n, need_p0=False, box=box, hr=need_hr, x14t=x14t)
            ghr_n = gh[0] if gh else {}
            preds = {} if a.heads and not a.with_base else {"DepthVLM": p0.cpu().numpy()}
            with torch.no_grad():
                for m in names:
                    _, _, _, enc, res = KE.HEADS[m]
                    pm = G.predict(hd[m], feats, None if enc is None else (geo_c if res == "c" else geo_n)[enc], ph, pw,
                                   ghr_n.get(enc) if hd[m].hproj is not None else None,
                                   G.rgb_input(img) if hd[m].rconv is not None else None)   # D-59 ② 사진 경로 = 같은 canonical 사진
                    preds[m] = (G.vlm_scale(pm, p0) if m in KE.VS else pm).cpu().numpy()   # D-52: 크기는 VLM 인 헤드
            for m, p in preds.items():
                assert np.isfinite(p).all(), (ds, rid, m)
                save(ds, m, rid, p, H, W)
            if j % 100 == 0 or j == len(rs) - 1:
                print(f"[vlm] {ds} {j + 1}/{len(rs)} {(time.time() - t0) / (j + 1):.2f} s/장", flush=True)


def pv(a):
    import torch
    name = a.model
    for ds in (a.sets or ("nyu_std",)):                               # D-41: 기본은 예전처럼 NYU 만, --sets 로 iBims-1 · SUN-RGBD
        if name in ("UniDepth-L", "ZoeDepth-NK", "DA3-metric-L"):
            import pv_extra
            f = pv_extra.FN[name]()
        elif name == "DAv2-metric-L" and ds in INDOOR:              # pv_fn 은 실외 (VKITTI) — 실내 세트는 실내 (Hypersim) 모델
            sys.argv = sys.argv[:1]
            sys.path.insert(0, os.path.join(KE.VDR, "eval"))
            import dense_sparse as DS
            f = DS.MODELS[name]("Indoor")
        elif name == "DAv2-metric-L" and ds == "eth3d":             # D-46: ETH3D 는 실내 + 실외 → 장면에 따라 실내 · 실외 모델 (표의 § 와 같은 쓰임)
            sys.argv = sys.argv[:1]
            sys.path.insert(0, os.path.join(KE.VDR, "eval"))
            import dense_sparse as DS
            f_in, f_out = DS.MODELS[name]("Indoor"), DS.MODELS[name]("Outdoor")
            f = lambda path, k: (f_out if path.split("/")[-4] in ETH_OUTDOOR else f_in)(path, k)
        else:
            f = KE.pv_fn("Metric3Dv2-L" if name == "Metric3Dv2-L·raw" else name)
        rs = (records(ds) if ds != "kitti" else [(r["id"], r["image"], r["K"], r["W"], r["H"]) for r in KE.records()])[:a.limit or None]   # D-44: KITTI = kitti_eval 기록 (652 장, 원본 사진 · 정답 K)
        t0 = time.time()
        for j, (rid, path, k, W, H) in enumerate(rs):
            out = os.path.join(D, "pred", ds, name, rid + ".npy")
            if a.resume and os.path.exists(out):
                continue
            with torch.no_grad():
                p = f(path, k)
            assert np.isfinite(p).all(), (ds, rid, name)
            save(ds, name, rid, p, H, W)
            if j % 100 == 0 or j == len(rs) - 1:
                print(f"[pv] {name} {ds} {j + 1}/{len(rs)} {(time.time() - t0) / (j + 1):.2f} s/장", flush=True)
        del f
        torch.cuda.empty_cache()


def pred_path(ds, m, rid):
    """예측 위치: 우리 방법은 paper_eval (KITTI 는 kitti_eval), 비교 모델은 NYU = paper_eval (Metric3Dv2 공식 = depthvlm-finetune), iBims-1·DIODE = vlm-depth-rmse 지도."""
    if ds == "kitti" and m not in EXTRA:                                 # D-44: 새 비교 모델 (EXTRA) 의 KITTI 는 paper_eval 자기 폴더 (아래 마지막 줄)
        if m == "Metric3Dv2-L":
            return os.path.join(DVFT, "preds", "m3d", "kitti", rid + ".npy")
        return os.path.join(KE.D, "pred", "Metric3Dv2-L" if m == "Metric3Dv2-L·raw" else m, rid + ".npy")
    if ds == "nyu_std" and m == "Metric3Dv2-L":
        return os.path.join(DVFT, "preds", "m3d", "nyu", rid + ".npy")
    if ds in ("ibims1", "diode_outdoor") and m in (*PV, "Metric3Dv2-L·raw"):
        return os.path.join(VDR_MAPS, "Metric3Dv2-L" if m == "Metric3Dv2-L·raw" else m, ds, rid + ".npy")
    return os.path.join(D, "pred", ds, m, rid + ".npy")


def gt_and_mask(ds, rid, rec):
    """정답 (m, 원본 크기) 과 채점 픽셀."""
    from PIL import Image
    lo, hi = RANGE[ds]
    if ds == "sunrgbd":                                               # 16 bit 3 bit 순환 mm (SUNRGBDtoolbox read3dPoints 와 같은 해독), 구멍 = 0
        r = np.array(Image.open(rec)).astype(np.uint16)
        g = ((r >> 3) | (r << 13)).astype(np.uint16).astype(np.float64) / 1000.0
        crop = np.ones_like(g, bool)
    elif ds == "nyu_std":
        g = np.array(Image.open(rec)).astype(np.float64) / 1000.0
        crop = np.zeros_like(g, bool)
        crop[45:471, 41:601] = True
    elif ds == "eth3d":                                               # raw float32 (왜곡 원본 좌표 · z), 무효 = inf, crop 없음
        sp = rec + ".sparse.npz"
        if not os.path.exists(rec) and os.path.exists(sp):            # (H200 데이터 묶음, 2026-11-19) raw 44 GB 대신 유한한 화소의 위치 (비트 마스크) 와 값만 저장한 것.
            z = np.load(sp)                                           #   아래 줄 (유한하지 않으면 0) 뒤의 g 는 raw 로 만든 것과 비트까지 같다 — 454 장 모두 로컬에서 확인 (coarse_branch/cb5_h200_eth3d.py)
            g = np.full(4032 * 6048, np.inf, np.float32)
            g[np.unpackbits(z["mask"], count=g.size).astype(bool)] = z["val"]
            g = g.reshape(4032, 6048).astype(np.float64)
        else:
            g = np.fromfile(rec, dtype=np.float32).reshape(4032, 6048).astype(np.float64)
        g[~np.isfinite(g)] = 0
        crop = np.ones_like(g, bool)
    elif ds == "kitti":
        g = np.array(Image.open(rec)).astype(np.float64) / 256.0
        H, W = g.shape
        crop = np.zeros_like(g, bool)
        crop[int(0.3324324 * H):int(0.91351351 * H), int(0.0359477 * W):int(0.96405229 * W)] = True
    else:
        r = rec
        if r.get("depth_format") == "npy":
            g = np.load(os.path.join(BENCH, r["depth_path"]))[..., 0].astype(np.float64)
            g[np.load(os.path.join(BENCH, r["mask_valid_path"])) == 0] = 0
        else:
            sys.path.insert(0, os.path.join(KE.VDR, "eval"))
            import dense_sparse as DS
            g = DS.gt_map(r, BENCH, ds).astype(np.float64)              # iBims-1: 공식 로더 (무효 마스크 포함) — 0.1–10 m 안은 원본과 같음
        crop = np.ones_like(g, bool)
    return g, crop & (g >= lo) & (g <= hi)


def score(a):
    import pandas as pd
    sys.path.insert(0, os.path.join(KE.VDR, "eval"))
    import common
    rows = []
    for ds in SETS:
        if ds == "kitti":
            rs = [(r["id"], r["gt"]) for r in json.load(open(os.path.join(KE.D, "records.json")))]
        elif ds == "sunrgbd":
            rs = [(r["id"], r["depth"]) for r in json.load(open(SUN))]
        elif ds == "eth3d":
            rs = [(r["id"], r["gt"]) for r in json.load(open(ETH))]
        elif ds == "nyu_std":
            sys.path.insert(0, os.path.expanduser("~/code/depthvlm-finetune/eval"))
            import std_eval
            rs = [(rid, gp) for rid, _, gp, _ in std_eval.records("nyu", os.path.join(DVFT, "data"), os.path.join(DVFT, "splits"))]
        else:
            bench = {r["image"]: r for r in common.load_bench(ds)[0]}
            rs = [(rid, bench[p.replace(BENCH + "/", "")]) for rid, p, _, _, _ in records(ds)]
        models = [*PV, "Metric3Dv2-L·raw", *OURS, *OTHER, *EXTRA]
        for rid, rec in rs:
            g, v = gt_and_mask(ds, rid, rec)
            gg = g[v]
            for m in models:
                pth = pred_path(ds, m, rid)
                if not os.path.exists(pth):
                    continue
                p = np.load(pth).astype(np.float64)
                if ds == "eth3d" and p.shape != g.shape:                     # D-46: 1/3 저장 → 정답 원해상도로 bilinear (Depth Pro 와 같음)
                    import cv2
                    p = cv2.resize(p.astype(np.float32), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float64)
                assert p.shape == g.shape, (ds, m, rid, p.shape, g.shape)
                pp = np.abs(p[v]) + 1e-10                                # Metric3D do_test: 예측 안 자름
                d = np.log(pp) - np.log(gg)
                rows.append(dict(set=ds, image=rid, model=m, n=int(v.sum()), d1=float((np.maximum(pp / gg, gg / pp) < 1.25).mean()),
                                 absrel=float((np.abs(pp - gg) / gg).mean()), rmse=float(np.sqrt(((pp - gg) ** 2).mean())),
                                 mu=float(d.mean()), m2=float((d ** 2).mean())))
        print(f"[score] {ds} {len(rs)} 장", flush=True)
    t = pd.DataFrame(rows)
    os.makedirs(D, exist_ok=True)
    t.to_parquet(os.path.join(D, "per_image.parquet"), index=False)
    out = []
    for (ds, m), y in t.groupby(["set", "model"], sort=False):
        sil = 100 * np.sqrt(np.maximum(y.m2 - y.mu ** 2, 0))
        out.append(dict(set=ds, model=m, n_img=len(y), d1=y.d1.mean(), absrel=y.absrel.mean(), rmse=y.rmse.mean(), silog=sil.mean(),
                        scale_rms=100 * np.sqrt((y.mu ** 2).mean()), scale_abs=100 * y.mu.abs().mean(), bias=100 * y.mu.mean()))   # D-38: scale_abs = 사진별 100·|d̄| 의 평균 (표의 배율 오차)
    O = pd.DataFrame(out)
    O.to_csv(os.path.join(ROOT, "tables", "paper_eval.csv"), index=False)
    pd.set_option("display.width", 200)
    print(O.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["vlm", "pv", "score"])
    ap.add_argument("--model", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--heads", nargs="*", default=[], help="vlm: 이 헤드만 (기본 = kitti_eval.HEADS 전부 + 원래 DepthVLM)")
    ap.add_argument("--sets", nargs="*", default=[], help="D-41: vlm · pv 가 돌 세트 (기본 = 예전과 같음)")
    ap.add_argument("--with_base", action="store_true", help="D-41: --heads 와 함께 원래 DepthVLM 예측도 저장")
    ap.add_argument("--resume", action="store_true", help="D-41 pv: 이미 저장된 예측은 건너뜀")
    a = ap.parse_args()
    {"vlm": vlm, "pv": pv, "score": score}[a.stage](a)
