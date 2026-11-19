"""평가 6 세트 채점 (2 단계 뒤 한 번 · 보고용 — PROMPT_v2 12.3). cb2_stage2.py eval6 가 부른다.
입력 · 크기 맞춤 · 채점은 analysis/paper_eval.py 와 같다: 초점 1000 canonical 사진 → 본 통과 → 예측 (32ph × 32pw) → 저장 꼴 크기 (ETH3D 는 1/3) 로 bilinear · float16 → 정답 크기 → 사진별 지표.
예측은 디스크에 두지 않고 그 자리에서 채점한다 (디스크 여유 때문). 줄 이름:
  orig     우리 경로로 다시 낸 원래 DepthVLM            stored   paper_eval 에 저장된 원래 DepthVLM (배관 확인: orig 와 같아야 함)
  prop_S · ctrl_S   제안 · 대조 팔, 배율 = 배율 헤드 S (3.4 의 합치기 — 방법 그대로)
  prop_O · ctrl_O   배율 = 원래 헤드 출력의 칸 평균 (S 가 완벽할 때)          prop_G · ctrl_G   배율 = 정답 화소에서 원래 예측과 평균 ln 이 같게 (합치기가 완벽할 때, 정답 위치만 씀)
  prop_X · ctrl_X   배율 = 정답에 맞춤 (사용자 11-11 '모양이 달라지는데 원래 배율에 맞추는 게 의미가 있나' — 모양만으로 얻을 수 있는 최대치, 참고용)
출력  ~/data/dvsilog/coarse_branch/v2runs/<tag>/eval6_per_image.csv
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import cb2_data as D

SETS = ("nyu_std", "ibims1", "sunrgbd", "kitti", "eth3d", "diode_outdoor")


class Items(torch.utils.data.Dataset):
    def __init__(self, ds, rs):
        self.ds, self.rs = ds, rs

    def __len__(self):
        return len(self.rs)

    def __getitem__(self, j):
        import cv2
        cv2.setNumThreads(1)
        torch.set_num_threads(1)
        import cb_eval as CE
        PE = CE.pe()
        sys.path.insert(0, str(Path.home() / "code/vlm-depth-rmse/third_party/DepthVLM"))
        from utils.datasets import _load_and_resize_image
        rid, path, fx, W, H, gtrec = self.rs[j]
        img = _load_and_resize_image(path, [round(W * 1000.0 / fx), round(H * 1000.0 / fx)])          # paper_eval.vlm 과 같은 사진
        g, v = PE.gt_and_mask(self.ds, rid, gtrec)
        stored = np.load(PE.pred_path(self.ds, "DepthVLM", rid)).astype(np.float32)
        return dict(j=j, inp=D.vlm_inputs(img), g=g.astype(np.float32), v=v, stored=stored, W=W, H=H)


def met(p, gg, lg, v, k=1.0):
    """paper_eval.score 와 같은 식 (k = 예측에 곱하는 상수)."""
    pp = (np.abs(p[v]).astype(np.float64) + 1e-10) * k
    d = np.log(pp) - lg
    mu, m2 = float(d.mean()), float((d ** 2).mean())
    return dict(d1=float((np.maximum(pp / gg, gg / pp) < 1.25).mean()), absrel=float((np.abs(pp - gg) / gg).mean()), rmse=float(np.sqrt(((pp - gg) ** 2).mean())),
                mu=mu, m2=m2, silog=100 * math.sqrt(max(m2 - mu * mu, 0.0)), scale=100 * abs(mu))


@torch.no_grad()
def run(a, Net, load_final, compose, cell_log, to_dev, one_item):
    import cv2
    cv2.setNumThreads(0)                                               # 본 프로세스의 cv2 스레드 풀을 끔 (작업자와 잠금이 얽히지 않게)
    import cb_eval as CE
    PE = CE.pe()
    net = Net(a.read)
    R, Sh = load_final(net, a)
    out = Path.home() / "data/dvsilog/coarse_branch/v2runs" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in (a.sets or SETS):
        rs = CE.recs(ds, a.limit)
        t0, empty = time.time(), 0
        for it in torch.utils.data.DataLoader(Items(ds, rs), batch_size=None, collate_fn=one_item, prefetch_factor=1, **D.mp(max(1, min(a.workers, 3)))):
            rid = rs[it["j"]][0]
            g, v = np.asarray(it["g"], dtype=np.float64), np.asarray(it["v"]).astype(bool)
            if not v.any():
                empty += 1
                continue
            gg = g[v]
            lg = np.log(gg)
            H, W = int(it["H"]), int(it["W"])
            Hs, Ws = (H // PE.ETH_DS, W // PE.ETH_DS) if ds == "eth3d" else (H, W)

            def full(p):                                               # paper_eval.save (저장 꼴 · float16) → 채점 때 정답 크기로
                q = p.astype(np.float32)
                if q.shape != (Hs, Ws):
                    q = cv2.resize(q, (Ws, Hs), interpolation=cv2.INTER_LINEAR)
                q = q.astype(np.float16).astype(np.float32)
                if q.shape != g.shape:
                    q = cv2.resize(q, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
                return q.astype(np.float64)

            p0, feats, ph, pw = net.main_pass(to_dev(it["inp"]))
            geo, sky_logit = net.geo_feats(ph, pw)
            m = Sh(feats, ph, pw)
            m0 = F.avg_pool2d(p0.clamp_min(1e-4).log()[None, None], 32)[0, 0]
            P0 = full(p0.cpu().numpy())
            st = np.asarray(it["stored"], dtype=np.float64)
            if st.shape != g.shape:
                st = cv2.resize(st.astype(np.float32), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float64)
            res = dict(orig=met(P0, gg, lg, v), stored=met(st, gg, lg, v))
            info = dict(sky_frac=float((sky_logit > 0).float().mean()), ph=ph, pw=pw)
            for arm in R:
                z = net.predict(R[arm], feats, geo if arm == "prop" else None, ph, pw)
                lz = z.clamp_min(1e-4).log()
                Z = full(z.cpu().numpy())
                sS, sO = float(compose(lz, m, sky_logit)), float(compose(lz, m0, sky_logit))
                sG = float(np.log(np.abs(P0[v]) + 1e-10).mean() - np.log(np.abs(Z[v]) + 1e-10).mean())
                sX = float(lg.mean() - np.log(np.abs(Z[v]) + 1e-10).mean())     # 정답에 맞춘 배율 (모양만으로 얻을 수 있는 최대치 — 참고용)
                for tag, s_ in (("S", sS), ("O", sO), ("G", sG), ("X", sX)):
                    res[f"{arm}_{tag}"] = met(Z, gg, lg, v, math.exp(s_))
                info[f"s_{arm}"] = sS
            rows += [dict(set=ds, image=rid, model=k, n=int(v.sum()), **r, **info) for k, r in res.items()]
        t = pd.DataFrame([r for r in rows if r["set"] == ds])
        piv = t.pivot(index="image", columns="model", values="silog")
        print(f"[eval6] {ds} {len(rs)} 장 · {time.time() - t0:.0f} 초 · 채점 화소 없는 사진 {empty} · SILog 원래 {piv.orig.mean():.2f} (저장된 것 {piv.stored.mean():.2f}, 사진별 차 평균 {(piv.orig - piv.stored).abs().mean():.3f})"
              f" · 제안 {piv.prop_S.mean():.2f} · 대조 {piv.ctrl_S.mean():.2f}", flush=True)
        pd.DataFrame(rows).to_csv(out / "eval6_per_image.csv", index=False)
    print(f"[eval6] 저장 {out / 'eval6_per_image.csv'} · EXIT 0", flush=True)
