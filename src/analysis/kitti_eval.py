"""KITTI 평가 (사용자 10-13 "이제부터 nyu,ibims,kitti,diode 4개로 비교", NOTES D-18): Eigen 테스트 652 장 (annotated depth) 에서 모든 방법.

규칙 (D-18, 결과 전 고정): 목록·사진·정답 = depthvlm-finetune 이 만든 것 그대로 (~/data/dvft/data/kitti/{lists/test.txt, full, full_gt}),
입력 = 원본 사진 전체, 카메라 = calib P_rect_02 (fx, fy, cx, cy). 채점 픽셀 = 0.001 m < 정답 < 80 m 이고 Garg crop 안.
예측은 정답 크기로 bilinear, log 에서만 [0.001, 80] 으로 자름 — geo_mix.stats (= vlm-depth-rmse Track B) 와 같은 충분통계.

단계 (환경이 달라 나눔, hypersim_eval.py 와 같은 틀):
  prep  (~/venv/main, CPU)     목록 → records.json (id, 사진, 정답, K, 크기)
  vlm   (~/venv/depthvlm, GPU) 원래 DepthVLM · control · proposed · dino (canonical 인코더 입력) · proposed_native · dino_native (원래 해상도) · udv2only (D-17)
  pv    (각 모델 환경, GPU)     UniDepthV2-L (K 없음) · UniDepthV2-L+K · Metric3Dv2-L (정답 fx) · DAv2-metric-L 실외 (VKITTI) — vlm-depth-rmse 공식 설정 함수
  score (~/venv/main, CPU)     이미지별 충분통계 (+ 헤드마다 +VLMscale) → per_image.parquet
  bench (모델 환경, GPU)       앞 50 장 (예열 5 장 제외): 사진 1 장당 시간 (파일 → 깊이 지도 CPU numpy, 배치 1)·최대 GPU 메모리 → bench/<방법>.json
예측은 정답 해상도 fp16 으로 ~/data/dvsilog/kitti_eval/pred/<방법>/<id>.npy.

사용:
  ~/venv/main/bin/python -I analysis/kitti_eval.py prep
  ~/venv/depthvlm/bin/python -I analysis/kitti_eval.py vlm [--limit N]
  ~/venv/unidepth/bin/python -I analysis/kitti_eval.py pv --model UniDepthV2-L     (UniDepthV2-L+K 도 unidepth, Metric3Dv2-L @metric3d, DAv2-metric-L @dav2)
  ~/venv/main/bin/python -I analysis/kitti_eval.py score
  <환경>/bin/python -I analysis/kitti_eval.py bench --model <방법>               (VLM·헤드 방법은 depthvlm, 비교 모델은 pv 와 같은 환경)
"""
import argparse
import json
import os
import sys
import time
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.environ.get("KITTI_EVAL_DIR", os.path.expanduser("~/data/dvsilog/kitti_eval"))   # 시험 실행은 다른 폴더로
KITTI = os.path.expanduser("~/data/dvft/data/kitti")
RAW = os.path.expanduser("~/data/kitti/raw")
VDR = os.path.expanduser("~/code/vlm-depth-rmse")
CK = os.path.expanduser("~/checkpoints/dvsilog")
CAP = (0.001, 80.0)                                    # BTS KITTI 범위 (정답 유효 범위 = log cap)
GARG = (0.40810811, 0.99189189, 0.03594771, 0.96405229)   # 행 위·아래, 열 왼·오 (H·W 비율)
# 헤드: (주입 여부, VLM 특징 사용, 체크포인트, 인코더, 인코더 입력 c = canonical (해결 실험 2) / n = 원래 해상도 (D-14))
HEADS = {"control": (False, True, "geo_mix/best_control.pt", None, None),
         "proposed": (True, True, "geo_mix/best_proposed.pt", "udv2", "c"),
         "dino": (True, True, "geo_mix/best_dino.pt", "dino", "c"),
         "proposed_native": (True, True, "geo_mix_native/best_proposed.pt", "udv2", "n"),
         "dino_native": (True, True, "geo_mix_native/best_dino.pt", "dino", "n"),
         "udv2only": (True, False, "geo_mix_udv2only/best_udv2only.pt", "udv2", "n"),
         "proposed_spp": (True, True, "geo_mix_spp/best_proposed.pt", "udv2", "n"),             # D-21 (+ScanNet++, 12.5 만 장 끝 = 최고)
         "proposed_spp100k": (True, True, "geo_mix_spp/at_100000_proposed.pt", "udv2", "n"),    # D-21 같은 학습량 비교 (10 만 장 시점)
         "proposed_hr": (True, True, "geo_mix_hr/best_proposed_hr.pt", "udv2", "n"),          # D-24 (+원래 해상도 UD 를 헤드 세밀한 단계에, 12.5 만 장 최고)
         "proposed_hr100k": (True, True, "geo_mix_hr/at_100000_proposed_hr.pt", "udv2", "n"),  # D-24 같은 학습량 비교 (10 만 장 시점)
         "proposed_hr_edge": (True, True, "geo_mix_edge2/best_hr_dav2d.pt", "udv2", "n"),   # D-27 dav2d (D-24 + DAv2 경계 손실, 조밀 정답 3 종만, 2 만 장 끝, F-46)
         "proposed_hr_ctrl": (True, True, "geo_mix_edge2/best_hr_ctrl.pt", "udv2", "n"),    # D-27 대조 (D-24 + 같은 2 만 장, 경계 손실 없음, 2 만 장 끝) — ablation 짝 (D-28)
         "d29_ctrl": (True, True, "geo_mix_hr_edge/best_hr_ctrl.pt", "udv2", "n"),          # D-29 λ 0 (처음부터 12.5 만 장, SILog 만 = D-24 조리법 다시, 12.5 만 장 고름)
         "d29_l045": (True, True, "geo_mix_hr_edge/best_hr_dav2d_lo.pt", "udv2", "n"),      # D-29 λ 0.045 (경계 손실 처음부터, 조밀 정답 3 종만, 12.5 만 장 고름)
         "d29_l090": (True, True, "geo_mix_hr_edge/best_hr_dav2d.pt", "udv2", "n"),         # D-29 λ 0.090 (12.5 만 장 고름)
         "d29_l180": (True, True, "geo_mix_hr_edge/best_hr_dav2d_hi.pt", "udv2", "n"),      # D-29 λ 0.18 (12 만 장 고름) — F-50 주 λ
         "syn_v1_ctrl": (True, True, "geo_mix_syn/best_v1_ctrl.pt", "udv2", "n"),          # D-30 시너지: v1 (토큰 주입만) λ 0
         "syn_v1_l018": (True, True, "geo_mix_syn/best_v1_dav2d.pt", "udv2", "n"),         # D-30 v1 λ 0.18
         "syn_ctrl": (True, True, "geo_mix_syn/best_hr_ctrl.pt", "udv2", "n"),             # D-30 v2 λ 0 (D-29 대조 재현)
         "syn_l018": (True, True, "geo_mix_syn/best_hr_dav2d_hi.pt", "udv2", "n"),         # D-30 v2 λ 0.18 (D-29 주 λ 재현)
         "syn_l036": (True, True, "geo_mix_syn/best_hr_dav2d_hi2.pt", "udv2", "n"),        # D-30 v2 λ 0.36
         "d31_ctrl": (True, True, "geo_mix_d31/best_hr_ctrl.pt", "udv2", "n"),             # D-31 기존 경계 기법 비교: v2 λ 0 (대조)
         "d31_l036": (True, True, "geo_mix_d31/best_hr_dav2d_hi2.pt", "udv2", "n"),        # D-31 우리 (DAv2 교사 경계, λ 0.36)
         "d31_midas": (True, True, "geo_mix_d31/best_hr_midas.pt", "udv2", "n"),           # D-31 MiDaS 기울기 맞추기 (공식 gist)
         "d31_rpnl": (True, True, "geo_mix_d31/best_hr_rpnl.pt", "udv2", "n"),             # D-31 Metric3D v2 RPNL (공식 코드 그대로)
         "d31_rpnl_m": (True, True, "geo_mix_d31/best_hr_rpnl_m.pt", "udv2", "n"),         # D-31 RPNL, MAD 를 조각 안 정답 픽셀로만 (논문 식 8)
         "d31_egssi": (True, True, "geo_mix_d31/best_hr_egssi.pt", "udv2", "n"),           # D-31 UniDepthV2 EG-SSI (공식 코드)
         "d31_dav2d_gt": (True, True, "geo_mix_d31/best_hr_dav2d_gt.pt", "udv2", "n"),     # D-31 분리 ① 우리 손실, 비교 대상만 정답
         "d31_dav2d_rgb": (True, True, "geo_mix_d31/best_hr_dav2d_rgb.pt", "udv2", "n"),   # D-31 분리 ② 우리 손실, 위치만 RGB 경계
         "d31_control": (False, True, "geo_mix_d31/best_control.pt", None, None),          # D-31 인코더 표: 헤드만 다시 학습 (12.5 만 장)
         "d31_dino": (True, True, "geo_mix_d31/best_dino.pt", "dino", "n"),                # D-31 인코더 표: VLM + DINOv2-L (12.5 만 장)
         "d31_udv2only": (True, False, "geo_mix_d31/best_udv2only.pt", "udv2", "n"),       # D-31 인코더 표: 전문가 특징만 (VLM 없음, 12.5 만 장)
         "vs_ctrl036": (True, True, "geo_mix_vs/best_hr_dav2d_hi2.pt", "udv2", "n"),       # D-52 같은 실행 대조 (지금 조리법 = syn_l036 · d31_l036 과 같음)
         "vs_l036": (True, True, "geo_mix_vs/best_vs_dav2d_hi2.pt", "udv2", "n"),          # D-52 크기는 VLM (출력 = geo_mix.vlm_scale, 손실 √Var + 경계)
         "vd_l036": (True, True, "geo_mix_vd/best_vd_dav2d_hi2.pt", "udv2", "n")}          # D-53 C안: 크기 기준 = 원래 DepthVLM 예측 (학습 때, 정답 픽셀), 출력은 헤드 그대로 — VS 아님
HEADS.update({"kr_ctrl": (False, True, "geo_mix_kr/best_control.pt", None, None),        # D-84 국내 논문: 헤드만 다시 학습 (주입 없음, 같은 실행 대조)
              "kr_l018": (False, True, "geo_mix_kr/best_kr_dav2d.pt", None, None)})     # D-84 주입 없는 헤드 + DAv2 경계 손실 λ 0.18
VS = {"vs_l036"}                                                    # D-52: 평가에서도 출력 크기를 원래 DepthVLM 예측에 맞추는 헤드 (geo_mix.VS_HEADS 와 짝)
HEADS.update({"vs_rgb036": (True, True, "geo_mix_d59/best_vs_rgb_dav2d_hi2.pt", "udv2", "n"),    # D-59 ② ⓑ + 원본 사진 경로 (1/2 단계, D-40 A)
              "vs_control": (False, True, "geo_mix_d59/best_vs_control.pt", None, None),        # D-59 ③ 표 3 (b) ⓑ 조리법: 헤드만
              "vs_dino": (True, True, "geo_mix_d59/best_vs_dino.pt", "dino", "n"),              # D-59 ③ (c) VLM + DINOv2-L
              "vs_v1_ctrl": (True, True, "geo_mix_d59/best_vs_v1_ctrl.pt", "udv2", "n"),        # D-59 ③ (d) VLM + 전문가 (토큰 주입만)
              "vs_udv2only": (True, False, "geo_mix_d59/best_vs_udv2only.pt", "udv2", "n"),     # D-59 ③ (e) 전문가만 (크기 기준만 원래 DepthVLM)
              "vs_hr_ctrl": (True, True, "geo_mix_d59/best_vs_hr_ctrl.pt", "udv2", "n")})       # D-59 ③ (f) + 고해상도 경로 (경계 손실 없음)
VS.update({"vs_rgb036", "vs_control", "vs_dino", "vs_v1_ctrl", "vs_udv2only", "vs_hr_ctrl"})
HEADS.update({"d62_crop036": (True, True, "geo_mix_d62/at_20000_hr_dav2d_hi2.pt", "udv2", "n"),   # D-62 파일럿: 예전 Ours 조리법 + 자르기 증강 (2 만 장)
              "d62_cropvd": (True, True, "geo_mix_d62/at_20000_vd_dav2d_hi2.pt", "udv2", "n"),    # D-62 파일럿: C안 손실 + 자르기 증강 (2 만 장)
              "d62_ctrl20k": (True, True, "geo_mix_vs/at_20000_hr_dav2d_hi2.pt", "udv2", "n"),    # D-62 대조: D-52 실행 예전 Ours 조리법 2 만 장 (자르기 없음)
              "d62_vd20k": (True, True, "geo_mix_vd/at_20000_vd_dav2d_hi2.pt", "udv2", "n")})     # D-62 대조: D-53 실행 C안 2 만 장 (자르기 없음)
for _p in (10000, 20000, 50000):                                      # D-66 파일럿: 같은 실행 헤드 여섯의 1 만 · 2 만 장 가중치 (인코더 입력 't' = 토큰 격자, geo_mix.tok_input) · 5 만 장 = F-93 뒤 연장 (보고만)
    _s = f"{_p // 1000}k"
    HEADS.update({f"d66_ref{_s}": (True, True, f"geo_mix_d66/at_{_p}_hr_dav2d_hi2.pt", "udv2", "n"),      # 예전 Ours 조리법 (같은 실행 대조)
                  f"d66_vlm{_s}": (False, True, f"geo_mix_d66/at_{_p}_vlm_dav2d_hi2.pt", None, None),      # VLM 특징만 + 같은 경계 손실
                  f"d66_tk{_s}": (True, True, f"geo_mix_d66/at_{_p}_tk_dav2d_hi2.pt", "udv2t", "t"),       # 토큰 격자 전문가 (평균 풀링 없음)
                  f"d66_tkc{_s}": (True, True, f"geo_mix_d66/at_{_p}_tkc_dav2d_hi2.pt", "udv2t", "t"),     # + 사진 평균 뺀 주입
                  f"d66_nc{_s}": (True, True, f"geo_mix_d66/at_{_p}_nc_dav2d_hi2.pt", "udv2", "n"),        # 원래 해상도 전문가 + 사진 평균 뺀 주입
                  f"d66_tkct{_s}": (True, True, f"geo_mix_d66/at_{_p}_tkct_dav2d_hi2.pt", "udv2t", "t")})  # + 전문가 끈 같은 헤드와 배율 묶기 (학습 때만)
PV = ("UniDepthV2-L", "UniDepthV2-L+K", "Metric3Dv2-L", "DAv2-metric-L")


# ───────────────────────── prep ─────────────────────────
def calib(date):
    """calib_cam_to_cam.txt 의 P_rect_02 → [fx, fy, cx, cy] (image_02 = 왼쪽 컬러, 목록의 사진)."""
    with zipfile.ZipFile(os.path.join(RAW, f"{date}_calib.zip")) as z:
        for l in z.read(f"{date}/calib_cam_to_cam.txt").decode().splitlines():
            if l.startswith("P_rect_02:"):
                P = [float(x) for x in l.split()[1:]]
                return [P[0], P[5], P[2], P[6]]
    raise KeyError(date)


def prep(a):
    from PIL import Image
    rows, cal = [], {}
    for l in open(os.path.join(KITTI, "lists", "test.txt")):
        img, gt, fx = l.split()
        date = img.split("/")[0]
        K = cal.setdefault(date, calib(date))
        assert abs(K[0] - float(fx)) < 1e-3, (img, K, fx)                # 목록 초점 = calib (depthvlm-finetune build_kitti 와 같은 점검)
        ip, gp = os.path.join(KITTI, "full", img), os.path.join(KITTI, "full_gt", gt)
        W, H = Image.open(ip).size
        assert Image.open(gp).size == (W, H), img
        rows.append(dict(id=img.split("/")[1] + "__" + os.path.basename(img)[:-4], image=ip, gt=gp, K=K, W=W, H=H))
    assert len(rows) == 652 and len({r["id"] for r in rows}) == 652
    os.makedirs(D, exist_ok=True)
    json.dump(rows, open(os.path.join(D, "records.json"), "w"))
    print(f"records {len(rows)} | 크기 {sorted({(r['W'], r['H']) for r in rows})} | fx {sorted({round(r['K'][0], 2) for r in rows})}")


def records(limit=0):
    r = json.load(open(os.path.join(D, "records.json")))
    return r[:limit] if limit else r


def save_pred(model, rid, p, H, W):
    import cv2
    if p.shape != (H, W):
        p = cv2.resize(p.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    os.makedirs(os.path.join(D, "pred", model), exist_ok=True)
    np.save(os.path.join(D, "pred", model, rid + ".npy"), p.astype(np.float16))


# ───────────────────────── vlm: DepthVLM · 재학습 헤드 6 개 ─────────────────────────
def load_heads(G, base, names):
    import torch
    hd = {}
    for k in names:
        inj, use_vlm, f, _, _ = HEADS[k]
        sd = torch.load(os.path.join(CK, f), map_location=G.DEV)
        hd[k] = G.Head(base, inj, vlm=use_vlm, hr=any(n.startswith("hproj.") for n in sd),
                       rgb=any(n.startswith("rconv.") for n in sd), center="center_flag" in sd).to(G.DEV)   # D-24 고해상도 경로 · D-59 ② 사진 경로 · D-66 평균 뺀 주입
        hd[k].load_state_dict(sd)
        hd[k].eval()
    return hd


def canonical(G, r):
    fx = r["K"][0]
    return G._load_and_resize_image(r["image"], [round(r["W"] * 1000.0 / fx), round(r["H"] * 1000.0 / fx)])   # DepthVLM 공식: 초점 1000 으로 (geo_mix.evaluate 와 같음)


def vlm(a):
    import torch
    from PIL import Image
    sys.argv = sys.argv[:1]
    sys.path.insert(0, os.path.join(ROOT, "analysis"))
    import geo_mix as G
    names = a.heads or list(HEADS)                                    # --heads: 그 헤드만 (원래 DepthVLM·다른 헤드 예측은 다시 저장 안 함)
    proc, base, features = G.load_models(sorted({HEADS[k][3] for k in names if HEADS[k][3]}))
    hd = load_heads(G, base, names)
    need_hr = any(hd[k].hproj is not None for k in names)
    t0 = time.time()
    rs = records(a.limit)
    for j, r in enumerate(rs):
        img = canonical(G, r)
        p0, feats, geo_c, (ph, pw) = features(G.vlm_inputs(proc, img), G.enc_input(img))                       # 해결 실험 2 와 같은 인코더 입력
        x14n, box = G.unidepth_input(Image.open(r["image"]).convert("RGB"), box=True)                        # D-14 (KITTI 는 위아래 패딩 → 사진 칸만)
        x14t = G.tok_input(img, ph, pw) if any(HEADS[k][4] == "t" for k in names) else None                   # D-66 토큰 격자 입력 (같은 canonical 사진)
        _, _, geo_n, _, *gh = features(G.vlm_inputs(proc, img), x14n, need_p0=False, box=box, hr=need_hr, x14t=x14t)
        ghr_n = gh[0] if gh else {}
        preds = {} if a.heads else {"DepthVLM": p0.cpu().numpy()}
        with torch.no_grad():
            for k in names:
                _, _, _, enc, res = HEADS[k]
                geo = None if enc is None else (geo_c if res == "c" else geo_n)[enc]
                pk = G.predict(hd[k], feats, geo, ph, pw, ghr_n.get(enc) if hd[k].hproj is not None else None,
                               G.rgb_input(img) if hd[k].rconv is not None else None)   # D-59 ② 사진 경로 = 같은 canonical 사진
                preds[k] = (G.vlm_scale(pk, p0) if k in VS else pk).cpu().numpy()   # D-52
        for k, p in preds.items():
            assert np.isfinite(p).all(), (r["id"], k)
            save_pred(k, r["id"], p, r["H"], r["W"])
        if j % 50 == 0 or j == len(rs) - 1:
            print(f"[vlm] {j + 1}/{len(rs)} {(time.time() - t0) / (j + 1):.2f} s/장", flush=True)


# ───────────────────────── pv: 비교 모델 (vlm-depth-rmse 공식 설정 함수) ─────────────────────────
def pv_fn(name):
    import torch
    sys.argv = sys.argv[:1]
    sys.path.insert(0, os.path.join(VDR, "eval"))
    import dense_sparse as DS
    if name == "UniDepthV2-L+K":                      # 정답 K 를 넣는 공식 경로 (scripts/demo.py) — hypersim_eval.pv 와 같음
        sys.path.insert(0, os.path.join(DS.EXT, "UniDepth"))
        from PIL import Image
        from unidepth.models import UniDepthV2
        m = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14", revision="52b349b514bd8b47642f67ac78cb7b5dc5c51dd9")
        m.interpolation_mode = "bilinear"
        m = m.to("cuda").eval()

        def f(path, k):
            K = torch.tensor([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1]], dtype=torch.float32)
            rgb = torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1)
            return m.infer(rgb, K)["depth"][0, 0].float().cpu().numpy()
        return f
    return DS.MODELS[name]("Outdoor")                 # UniDepthV2 = K 없음 (Track B 규칙), Metric3Dv2 = 정답 fx, DAv2 = 실외 (VKITTI) 모델


def pv(a):
    import torch
    f = pv_fn(a.model)
    t0 = time.time()
    rs = records(a.limit)
    for j, r in enumerate(rs):
        with torch.no_grad():
            p = f(r["image"], r["K"])
        assert np.isfinite(p).all(), (r["id"], a.model)
        save_pred(a.model, r["id"], p, r["H"], r["W"])
        if j % 50 == 0 or j == len(rs) - 1:
            print(f"[pv] {a.model} {j + 1}/{len(rs)} {(time.time() - t0) / (j + 1):.2f} s/장", flush=True)


# ───────────────────────── score ─────────────────────────
def stats(p, g, v):
    """geo_mix.stats 와 같은 충분통계 (log 에서만 CAP 으로 자름). v = 채점 픽셀."""
    pp, gg = p[v].astype(np.float64), g[v].astype(np.float64)
    e = pp - gg
    d = np.log(np.clip(pp, *CAP)) - np.log(gg)
    return dict(n=int(v.sum()), se=float((e ** 2).sum()), ar=float((np.abs(e) / gg).sum()),
                d1=int(((pp > 0) & (np.maximum(pp / gg, gg / np.maximum(pp, 1e-12)) < 1.25)).sum()), dl=float(d.sum()), dl2=float((d ** 2).sum()))


def score(a):
    import pandas as pd
    from PIL import Image
    rs = records(a.limit)
    models = sorted(os.listdir(os.path.join(D, "pred")))
    need = ["DepthVLM", *HEADS, *PV]
    missing = [m for m in need if m not in models]
    if missing:
        raise SystemExit(f"!!! 예측 없는 방법: {missing}")
    lm = lambda x: float(np.log(np.clip(x, *CAP)).mean())
    res = []
    for r in rs:
        g = np.array(Image.open(r["gt"])).astype(np.float32) / 256.0
        H, W = g.shape
        crop = np.zeros_like(g, bool)
        crop[int(GARG[0] * H):int(GARG[1] * H), int(GARG[2] * W):int(GARG[3] * W)] = True
        v = crop & (g > CAP[0]) & (g < CAP[1])
        P = {m: np.load(os.path.join(D, "pred", m, r["id"] + ".npy")).astype(np.float32) for m in need}
        for k in HEADS:                                # 사후 보완 (F-11): 사진마다 원래 DepthVLM 의 log 평균 깊이에 맞춤 — 모든 픽셀, GT 안 씀 (geo_mix.evaluate 와 같음)
            P[f"{k}+VLMscale"] = P[k] * np.exp(lm(P["DepthVLM"]) - lm(P[k]))
        for m, p in P.items():
            res.append(dict(set="kitti", image=r["id"], model=m, **stats(p, g, v)))
    t = pd.DataFrame(res)
    t.to_parquet(os.path.join(D, "per_image.parquet"), index=False)
    out = []
    for m, y in t.groupby("model", sort=False):
        n = y.n.values.astype(float)
        mu = y.dl.values / n
        out.append(dict(model=m, n_img=len(y), silog=(100 * np.sqrt(np.maximum(y.dl2.values / n - mu ** 2, 0))).mean(), scale_rms=100 * np.sqrt((mu ** 2).mean()),
                        bias=100 * mu.mean(), absrel=y.ar.sum() / n.sum(), rmse=np.sqrt(y.se.sum() / n.sum()), d1=y.d1.sum() / n.sum()))
    pd.set_option("display.width", 200)
    print(pd.DataFrame(out).round(4).to_string(index=False))


# ───────────────────────── bench: 시간·메모리 ─────────────────────────
def bench(a):
    import torch
    from PIL import Image
    rs = records(50)
    name = a.model
    if name in PV:
        f = pv_fn(name)
        run = lambda r: f(r["image"], r["K"])
    else:
        sys.argv = sys.argv[:1]
        sys.path.insert(0, os.path.join(ROOT, "analysis"))
        import geo_mix as G
        h = HEADS.get(name)
        enc = h[3] if h else None
        proc, base, features = G.load_models([enc] if enc else [], vlm=(h is None or h[1]))
        hd = load_heads(G, base, [name])[name] if h else None

        def run(r):
            img = canonical(G, r)
            x14, box = (None, None) if enc is None else ((G.enc_input(img), None) if h[4] == "c" else G.unidepth_input(Image.open(r["image"]).convert("RGB"), box=True))
            p0, feats, geo, (ph, pw) = features(G.vlm_inputs(proc, img), x14, need_p0=h is None, box=box)
            with torch.no_grad():
                return (p0 if h is None else G.predict(hd, feats, None if enc is None else geo[enc], ph, pw)).cpu().numpy()
    torch.cuda.synchronize()
    w_gb = torch.cuda.memory_allocated() / 1e9
    ts = []
    for j, r in enumerate(rs):
        if j == 5:
            torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.no_grad():
            run(r)
        torch.cuda.synchronize()
        if j >= 5:
            ts.append(time.perf_counter() - t)
    o = dict(model=name, n=len(ts), ms_mean=1000 * float(np.mean(ts)), ms_median=1000 * float(np.median(ts)),
             peak_gb=torch.cuda.max_memory_allocated() / 1e9, weights_gb=w_gb, gpu=torch.cuda.get_device_name())
    os.makedirs(os.path.join(D, "bench"), exist_ok=True)
    json.dump(o, open(os.path.join(D, "bench", f"{name}.json"), "w"))
    print(o)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "vlm", "pv", "score", "bench"])
    ap.add_argument("--model", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--heads", nargs="*", default=[], help="vlm: 이 헤드만 (기본 = HEADS 전부 + 원래 DepthVLM)")
    a = ap.parse_args()
    {"prep": prep, "vlm": vlm, "pv": pv, "score": score, "bench": bench}[a.stage](a)
