"""H200 실행기 — 표 2 · 부록 표 S3 의 세 판 (run.sh 가 부른다). 학습 · 평가 코드는 src/coarse_branch · src/analysis (로컬에서 네 칸 ablation 을 돌린 코드와 같다) 를 그대로 쓰고,
여기서는 그 코드를 부르는 방식만 정한다:
  s2         2 단계 (토큰 있는 팔 + 대조 팔, 합치기 전 특징 conv) — cb4_abl.py s2 와 같은 흐름. 끝의 '검증 700 장' 만 건너뜀
             (검증 데이터를 올리지 않음 — 보고용이라 학습된 모델과 무관)
  head       판 1: 토큰 없이 DepthVLM 의 헤드만 다시 학습 — cb2_stage2.train 에서 대조 팔 부분을 그대로 (같은 손실 · 학습률 · 일정 · 누적 4 · 자르기 · 배율 헤드)
  eval_head  판 1 의 평가 6 세트 — cb2_eval6.run 과 같은 사진 · 입력 · 채점. 토큰이 없어 하늘 판정이 없으므로 배율 헤드로 합친 값 (_S · _O) 은 하늘 없이 —
             표 2 의 SILog 는 배율과 무관 (사진마다 상수 배를 곱해도 같다)
  table      평가 결과 (v2runs/<tag>/eval6_per_image.csv) → 세트별 SILog 평균 (표 2 꼴)
1 단계와 판 2 · 3 의 평가는 run.sh 가 원래 스크립트 (cb4_abl.py s1 · cb3_stage2.py eval6) 를 그대로 부른다.
"""
import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
REF = {   # 네 칸 ablation (로컬, 11-19) 의 1 배 값 — 비교용 (results/abl4.txt): NYUv2 · iBims-1 · SUN · KITTI · DIODE · ETH3D
    "DepthVLM-4B": [10.52, 10.85, 11.45, 12.85, 39.21, 9.50],
    "1x tokens, labeled": [9.22, 9.18, 9.73, 11.42, 36.10, 10.42],
    "1x tokens, labeled + unlabeled": [9.09, 9.06, 9.57, 11.38, 37.54, 9.10],
}
ORDER = ("nyu_std", "ibims1", "sunrgbd", "kitti", "diode_outdoor", "eth3d")


def path_setup():
    sys.path.insert(0, str(SRC / "analysis"))
    sys.path.insert(0, str(SRC / "coarse_branch"))


def gt_only_patch(S1):
    """cb4_abl.py --gt_only 와 같은 것: 손실을 부르기 전에 사진의 gate 를 0 으로 → 선생 채움 항 · 채움 칸끼리의 기울기 항이 0."""
    import torch
    _loss = S1.stage1_loss

    def gt_only_loss(r, sky_logit, b, hp=S1.HP):
        b = dict(b)
        b["gate"] = torch.zeros_like(b["gate"])
        return _loss(r, sky_logit, b, hp)

    S1.stage1_loss = gt_only_loss


# ---------------------------------------------------------------- 2 단계 (판 2 · 3)
def cmd_s2(a):
    argv = ["train", "--read", a.read, "--tag", a.tag, "--stream", a.stream, "--inline", "--n_geo", "600", "--detail", "conv", "--workers", str(a.workers), "--resume"]
    if a.limit:
        argv += ["--limit", str(a.limit)]
    if a.all_lists:
        argv.append("--all_lists")
    sys.argv = sys.argv[:1] + argv                                   # cb3_stage2 · cb2_stage2 가 import 때 인자를 읽는다
    import pandas as pd
    import cb3_stage2 as W
    S2 = W.S2
    S2.val_cells = lambda *x, **k: pd.DataFrame()
    S2.val_line = lambda df: "검증 건너뜀 (H200 — 검증 700 장은 올리지 않음, 학습된 모델과 무관)"
    if a.gt_only:
        gt_only_patch(S2.S1)
    print(f"[h200 s2] {a.tag}: 학습 순서 {a.stream} · 토큰 {a.read} · 라벨만 (선생 채움 끔) {a.gt_only} · 목록 a–d {a.all_lists}", flush=True)
    S2.main()


# ---------------------------------------------------------------- 판 1: 헤드만
def _net_without_tokens(S2):
    """토큰 없는 본 통과 (cb2_stage2.Net · attach 안 함). 원래 출력은 토큰이 있을 때와 비트까지 같다 (읽기 전용 — 로컬 점검)."""
    import torch
    S2.Net.geo_feats = lambda self, ph, pw: (None, torch.full((ph, pw), -1.0, device=S2.DEV))
    return S2.Net(None, attach=False)


def cmd_head(a):
    sys.argv = sys.argv[:1]
    import pandas as pd
    import torch
    import cb2_common as C
    import cb2_data as D
    import cb2_stage2 as S2
    if a.gt_only:
        gt_only_patch(S2.S1)
    HP = S2.HP
    torch.manual_seed(S2.S1.HP["seed"])
    out, ck = S2.RUNS / a.tag, S2.CKPT / a.tag
    out.mkdir(parents=True, exist_ok=True)
    ck.mkdir(parents=True, exist_ok=True)
    stream = pd.read_parquet(C.V2 / f"stream_{a.stream}.parquet")
    if a.limit:
        stream = stream.iloc[: a.limit]
    assert (stream.kind == "L").all() or not a.gt_only, "라벨만 판에 무라벨 줄이 있음"
    net = _net_without_tokens(S2)
    R = {"ctrl": net.G.Head(net.base, False).to(S2.DEV)}
    Sh = S2.ScaleHead(net.base.norm.normalized_shape[0]).to(S2.DEV)
    h = lambda r: [q for n, q in r.named_parameters() if n.startswith("h.")]
    opt = dict(ctrl=torch.optim.AdamW([dict(params=h(R["ctrl"]), lr=HP["lr_head"])], weight_decay=0.0),      # cb2_stage2.optimizers 의 ctrl · S 그대로
               S=torch.optim.AdamW([dict(params=list(Sh.parameters()), lr=HP["lr_scale"])], weight_decay=0.0))
    base = {k: [g["lr"] for g in o.param_groups] for k, o in opt.items()}
    total = len(stream) // HP["accum"]
    warm = min(HP["warmup"], max(1, total // 10))
    sched = lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total))))
    step, log = 0, []
    if (ck / "last.pt").exists():
        st = torch.load(ck / "last.pt", map_location="cpu", weights_only=False)
        assert st["stream"] == a.stream and st["n"] == len(stream), "이어 하기: 학습 순서가 다름"
        R["ctrl"].load_state_dict(st["R"]["ctrl"])
        Sh.load_state_dict(st["S"])
        for k in opt:
            opt[k].load_state_dict(st["opt"][k])
        step, log = st["step"], st["log"]
        print(f"[{a.tag}] 이어 하기: 걸음 {step}/{total}", flush=True)

    def save(path, with_opt):
        tmp = path.with_name(path.stem + ".tmp.pt")
        torch.save(dict(R={"ctrl": R["ctrl"].state_dict()}, S=Sh.state_dict(), opt={k: o.state_dict() for k, o in opt.items()} if with_opt else None,
                        step=step, log=log, stream=a.stream, n=len(stream), hp=HP, gt_only=a.gt_only), tmp)
        tmp.replace(path)

    dl = torch.utils.data.DataLoader(D.S2Set(stream), batch_size=None, sampler=range(step * HP["accum"], total * HP["accum"]), collate_fn=S2.one_item, prefetch_factor=4, **D.mp(a.workers))
    params = dict(ctrl=list(R["ctrl"].parameters()), S=list(Sh.parameters()))
    print(f"[{a.tag}] 헤드만 (토큰 없음) · 학습 순서 {a.stream} · 사진 {len(stream)} 장 · 걸음 {total} · 헤드 매개변수 {sum(q.numel() for q in params['ctrl']) / 1e6:.1f} M · 라벨만 {a.gt_only}", flush=True)
    t0, run, n0, n_in = time.time(), defaultdict(list), step, 0
    for it in dl:
        if n_in == 0:
            for k, o in opt.items():
                for g, lr0 in zip(o.param_groups, base[k]):
                    g["lr"] = lr0 * sched(step)
                o.zero_grad(set_to_none=True)
        lr, ls, parts = S2.losses(net, R, Sh, S2.to_dev(it))
        (lr["ctrl"] / HP["accum"]).backward()
        (ls / HP["accum"]).backward()
        for k2, v in parts.items():
            run[k2].append(v)
        n_in += 1
        if n_in < HP["accum"]:
            continue
        n_in = 0
        for k, o in opt.items():
            if not all(torch.isfinite(q.grad).all() for q in params[k] if q.grad is not None):
                raise RuntimeError(f"기울기가 유한하지 않음 ({k}, 걸음 {step})")
            torch.nn.utils.clip_grad_norm_(params[k], HP["clip"])
            o.step()
        step += 1
        if step % a.log_every == 0 or step == total:
            row = dict(step=step, sec_per_img=(time.time() - t0) / ((step - n0) * HP["accum"]), gpu_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                       **{k2: sum(x for x, _ in v) / max(1.0, sum(c for _, c in v)) for k2, v in run.items()})
            log.append(row)
            run = defaultdict(list)
            print(f"[{a.tag}] {step}/{total} 대조 {row['ctrl_loss']:.4f} (정답 {row['ctrl_gt']:.4f} · 채움 {row['ctrl_fill']:.4f} · 기울기 {row['ctrl_gm']:.4f}) · 배율 헤드 {row['scale']:.4f}"
                  f" | {row['sec_per_img']:.2f} 초/장 · GPU {row['gpu_gb']:.1f} GB · 남은 {(total - step) * HP['accum'] * row['sec_per_img'] / 60:.0f} 분", flush=True)
            pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
        if step % a.ckpt_every == 0 and step < total:
            save(ck / "last.pt", True)
    save(ck / "final.pt", False)
    print(f"[{a.tag}] 끝 · {(time.time() - t0) / 60:.1f} 분 · EXIT 0", flush=True)


def cmd_eval_head(a):
    """cb2_eval6.run 과 같은 흐름 (사진 · 입력 · 크기 맞춤 · 채점 met 그대로), 팔은 대조 하나."""
    sys.argv = sys.argv[:1]
    import cv2
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    import cb2_data as D
    import cb2_eval6 as E6
    import cb2_stage2 as S2
    import cb_eval as CE
    cv2.setNumThreads(0)
    PE = CE.pe()
    net = _net_without_tokens(S2)
    R = {"ctrl": net.G.Head(net.base, False).to(S2.DEV)}
    Sh = S2.ScaleHead(net.base.norm.normalized_shape[0]).to(S2.DEV)
    st = torch.load(S2.CKPT / a.tag / "final.pt", map_location="cpu", weights_only=False)
    R["ctrl"].load_state_dict(st["R"]["ctrl"])
    Sh.load_state_dict(st["S"])
    R["ctrl"].eval(), Sh.eval()
    out = S2.RUNS / a.tag
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        for ds in (a.sets or E6.SETS):
            rs = CE.recs(ds, a.limit)
            t0, empty = time.time(), 0
            for it in torch.utils.data.DataLoader(E6.Items(ds, rs), batch_size=None, collate_fn=S2.one_item, prefetch_factor=1, **D.mp(max(1, min(a.workers, 3)))):
                rid = rs[it["j"]][0]
                g, v = np.asarray(it["g"], dtype=np.float64), np.asarray(it["v"]).astype(bool)
                if not v.any():
                    empty += 1
                    continue
                gg = g[v]
                lg = np.log(gg)
                H, W = int(it["H"]), int(it["W"])
                Hs, Ws = (H // PE.ETH_DS, W // PE.ETH_DS) if ds == "eth3d" else (H, W)

                def full(p):
                    q = p.astype(np.float32)
                    if q.shape != (Hs, Ws):
                        q = cv2.resize(q, (Ws, Hs), interpolation=cv2.INTER_LINEAR)
                    q = q.astype(np.float16).astype(np.float32)
                    if q.shape != g.shape:
                        q = cv2.resize(q, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
                    return q.astype(np.float64)

                p0, feats, ph, pw = net.main_pass(S2.to_dev(it["inp"]))
                _, sky_logit = net.geo_feats(ph, pw)
                m = Sh(feats, ph, pw)
                m0 = F.avg_pool2d(p0.clamp_min(1e-4).log()[None, None], 32)[0, 0]
                P0 = full(p0.cpu().numpy())
                stored = np.asarray(it["stored"], dtype=np.float64)
                if stored.shape != g.shape:
                    stored = cv2.resize(stored.astype(np.float32), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float64)
                res = dict(orig=E6.met(P0, gg, lg, v), stored=E6.met(stored, gg, lg, v))
                z = net.predict(R["ctrl"], feats, None, ph, pw)
                lz = z.clamp_min(1e-4).log()
                Z = full(z.cpu().numpy())
                sS, sO = float(S2.compose(lz, m, sky_logit)), float(S2.compose(lz, m0, sky_logit))
                sG = float(np.log(np.abs(P0[v]) + 1e-10).mean() - np.log(np.abs(Z[v]) + 1e-10).mean())
                sX = float(lg.mean() - np.log(np.abs(Z[v]) + 1e-10).mean())
                for tag, s_ in (("S", sS), ("O", sO), ("G", sG), ("X", sX)):
                    res[f"ctrl_{tag}"] = E6.met(Z, gg, lg, v, math.exp(s_))
                info = dict(ph=ph, pw=pw, s_ctrl=sS, sky="none")
                rows += [dict(set=ds, image=rid, model=k, n=int(v.sum()), **r, **info) for k, r in res.items()]
            t = pd.DataFrame([r for r in rows if r["set"] == ds])
            piv = t.pivot(index="image", columns="model", values="silog")
            print(f"[eval6] {ds} {len(rs)} 장 · {time.time() - t0:.0f} 초 · 채점 화소 없는 사진 {empty} · SILog 원래 {piv.orig.mean():.2f} (저장된 것 {piv.stored.mean():.2f}, 사진별 차 평균 {(piv.orig - piv.stored).abs().mean():.3f})"
                  f" · 헤드만 {piv.ctrl_S.mean():.2f}", flush=True)
            pd.DataFrame(rows).to_csv(out / "eval6_per_image.csv", index=False)
    print(f"[eval6] 저장 {out / 'eval6_per_image.csv'} · EXIT 0", flush=True)


# ---------------------------------------------------------------- 표
def cmd_table(a):
    import pandas as pd
    runs = Path.home() / "data/dvsilog/coarse_branch/v2runs"             # cb2_stage1.RUNS 와 같은 자리
    df = pd.read_csv(runs / a.tag / "eval6_per_image.csv")
    sets = [s for s in ORDER if s in set(df.set)]
    name = {"nyu_std": "NY", "ibims1": "IB", "sunrgbd": "SU", "kitti": "KI", "diode_outdoor": "DI", "eth3d": "ET"}
    lines = [f"[표] {a.title} — 평가 6 세트 SILog (사진별 평균, 낮을수록 좋음){' · smoke (세트마다 몇 장 — 수치는 뜻 없음)' if a.smoke else ''}",
             "[표] " + f"{'':34s}" + "".join(f"{name[s]:>8s}" for s in sets) + f"{'Avg':>8s}"]
    out = {}
    for model in a.models.split(","):
        x = df[df.model == model]
        if x.empty:
            continue
        v = [float(x[x.set == s].silog.mean()) for s in sets]
        out[model] = dict(zip(sets, v))
        lines.append("[표] " + f"{model:34s}" + "".join(f"{u:8.2f}" for u in v) + f"{sum(v) / len(v):8.2f}")
    if not a.smoke and len(sets) == 6:
        for k, v in REF.items():
            lines.append("[표] " + f"(ref) {k:28s}" + "".join(f"{u:8.2f}" for u in v) + f"{sum(v) / 6:8.2f}")
    o, s = df[df.model == "orig"].set_index(["set", "image"]).silog, df[df.model == "stored"].set_index(["set", "image"]).silog
    lines.append("[표] 원래 DepthVLM 을 이 GPU 로 다시 낸 것 − 로컬에 저장된 예측 (사진별 SILog 차 절댓값 평균): " + " · ".join(f"{name[s_]} {float((o.loc[s_] - s.loc[s_]).abs().mean()):.3f}" for s_ in sets))
    print("\n".join(lines), flush=True)
    (runs / a.tag / "table.json").write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("s2", "head", "eval_head", "table"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--stream", default="")
    ap.add_argument("--read", default="")
    ap.add_argument("--gt_only", action="store_true")
    ap.add_argument("--all_lists", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sets", nargs="*", default=[])
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--models", default="orig,prop_S,ctrl_S")
    ap.add_argument("--title", default="")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    path_setup()
    import resource                                                  # 작업자가 넘기는 텐서마다 파일 번호를 하나씩 쓴다 (cb2_stage2.main 과 같게)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (65536 if hard == resource.RLIM_INFINITY else min(hard, 65536), hard))
    dict(s2=cmd_s2, head=cmd_head, eval_head=cmd_eval_head, table=cmd_table)[a.cmd](a)


if __name__ == "__main__":
    main()
