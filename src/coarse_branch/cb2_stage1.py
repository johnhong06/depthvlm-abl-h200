"""분리 디코딩 실험 1 단계 (PROMPT_v2 6.1): 모양 지식을 담는 세 팔을 같은 자료 · 같은 묶음 · 같은 손실로 학습한다 (GPU).

  train  --arm dav2s | siglip | read   학습 순서 표 (--stream) 를 한 번 통과. 묶음 = 작은 사진 크기가 같은 8 장 (세 팔 공통 목록, 읽기 전용 토큰 팔은 같은 8 장을 한 장씩 누적).
         → ~/checkpoints/dvsilog/coarse_branch/v2/<tag>/{last.pt, final.pt} · ~/data/dvsilog/coarse_branch/v2runs/<tag>/{train_log.csv, val_*.csv}
  eval   --arm … [--ckpt 경로]          검증 표만 다시 (--sets val hyper)
손실 (굵은 칸 ky × kx, 사진마다 — 4 절을 11-11 검토로 고친 꼴):
  정답 사진   √Var_wg(r − Gc) + 0.5 · [문턱] · √Var_qc(r − Tc) + 0.5 · L_gm + 0.2 · BCE(하늘)
  무라벨 사진 √Var_qc^trim(r − Tc)  (|d − 중앙값| 큰 10 % 칸 제외) + 0.5 · L_gm + 0.2 · BCE(하늘)
  L_gm = 정답 있는 칸끼리의 이웃 차 (목표 Gc) + 정답 없는 칸끼리의 이웃 차 (목표 Tc — 정답 사진은 문턱 통과 때만). 정답 칸과 채운 칸 사이 (솔기) 는 뺀다. 축척 2 개.
검증 (보고만): 가지의 굵은 무늬로 원래 DepthVLM 예측의 굵은 무늬만 바꾼 SILog (cb_common.merge, 중앙값 맞춤, 하늘은 가지가 낸 것).
사용: HF_HUB_OFFLINE=1 ~/venv/depthvlm/bin/python -I -B -u cb2_stage1.py train --arm dav2s --stream s1 [--limit 2000 --tag smoke]
"""
import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
sys.path.insert(0, str(Path(__file__).resolve().parent))
ARGV, sys.argv = sys.argv[1:], sys.argv[:1]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import cb_common as B  # noqa: E402
import cb2_common as C  # noqa: E402
import cb2_data as D  # noqa: E402

DEV = "cuda"
HP = dict(bs=8, lr_lora=1e-4, lr_new=5e-4, lr_body=5e-5, wd=0.01, warmup=500, clip=1.0, w_fill_l=0.5, w_fill_u=1.0, w_gm=0.5, w_sky=0.2, trim=0.10, gm_scales=2, seed=1111)
RUNS, CKPT = Path.home() / "data/dvsilog/coarse_branch/v2runs", Path.home() / "checkpoints/dvsilog/coarse_branch/v2"


# ---------------------------------------------------------------- 손실
def si(d, w):
    """√(무게 분산 + 1e-6) — 사진별. 무게가 없으면 0."""
    v, _ = B.wvar(d, w)
    return torch.sqrt(v + 1e-6) * (w.flatten(1).sum(1) > 0)


def trim_keep(d, w, frac):
    """무게 있는 칸 중 |d − 중앙값| 이 큰 frac 를 뺀 무게 (사진별)."""
    out = torch.zeros_like(w)
    for b in range(d.shape[0]):
        ok = w[b] > 0
        if int(ok.sum()) < 10:
            out[b] = w[b]
            continue
        dev = (d[b] - d[b][ok].median()).abs()
        out[b] = w[b] * (dev <= torch.quantile(dev[ok], 1.0 - frac))
    return out


def gm(d, w, K):
    """이웃 차 L1 — 두 칸 모두 무게가 있을 때만 (쌍 무게 = 둘 중 작은 쪽), 2 배씩 솎은 K 축척의 평균. 사진별."""
    tot, n = d.new_zeros(d.shape[0]), d.new_zeros(d.shape[0])
    for k in range(K):
        dk, wk = d[:, ::2 ** k, ::2 ** k], w[:, ::2 ** k, ::2 ** k]
        wx, wy = torch.minimum(wk[:, :, 1:], wk[:, :, :-1]), torch.minimum(wk[:, 1:], wk[:, :-1])
        num = ((dk[:, :, 1:] - dk[:, :, :-1]).abs() * wx).flatten(1).sum(1) + ((dk[:, 1:] - dk[:, :-1]).abs() * wy).flatten(1).sum(1)
        den = wx.flatten(1).sum(1) + wy.flatten(1).sum(1)
        ok = den > 0
        tot, n = tot + torch.where(ok, num / den.clamp_min(1e-12), torch.zeros_like(num)), n + ok.float()
    return tot / n.clamp_min(1.0)


def stage1_loss(r, sky_logit, b, hp=HP):
    """r · sky_logit (B, ky, kx). b = cb2_data 의 굵은 칸 목표. → (평균 손실, 부분 값)."""
    unl, gate, has = b["unl"], b["gate"], b["has"]
    l_gt = si(r - b["Gc"], b["wg"]) * has
    d = r - b["Tc"]
    qk = torch.where(unl[:, None, None] > 0, trim_keep(d.detach(), b["qc"], hp["trim"]), b["qc"])
    l_fill = si(d, qk) * gate
    gt_cell = (b["wg"] > 0).float()
    l_gm = gm(r - b["Gc"], b["wg"], hp["gm_scales"]) + gm(d, b["qc"] * (1.0 - gt_cell), hp["gm_scales"]) * gate
    l_sky = F.binary_cross_entropy_with_logits(sky_logit, b["skyc"], reduction="none").flatten(1).mean(1) if sky_logit is not None else torch.zeros_like(l_gt)   # 2 단계는 하늘 항 없음
    w_fill = torch.where(unl > 0, torch.full_like(unl, hp["w_fill_u"]), torch.full_like(unl, hp["w_fill_l"]))
    total = l_gt + w_fill * l_fill + hp["w_gm"] * l_gm + hp["w_sky"] * l_sky
    lab, n = 1.0 - unl, float(r.shape[0])
    return total.mean(), dict(loss=(total.sum().item(), n), gt=(l_gt.sum().item(), has.sum().item()), fill=(l_fill.sum().item(), gate.sum().item()), gm=(l_gm.sum().item(), n),
                              sky=(l_sky.sum().item(), n), gate=((gate * lab).sum().item(), lab.sum().item()))     # 값마다 (합, 사진 수)


# ---------------------------------------------------------------- 모델
def build(arm):
    import cb2_models as M2
    return (M2.ReadArm() if arm == "read" else M2.ThumbArm(arm)).to(DEV)


def to_dev(b):
    return {k: (v.to(DEV, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def forward(model, arm, b):
    """묶음 → (r, sky_logit, 추가 손실). 읽기 전용 토큰 팔은 사진 한 장 (b 의 첫 차원 1)."""
    ky, kx = b["Gc"].shape[-2:]
    if arm != "read":
        r, sky = model(b["u8"], ky, kx)
        return r, sky, 0.0
    nh, nw = int(b["nh"][0]), int(b["nw"][0])
    gh, gw = D.geo_grid(nh, nw)
    r, sky, sky_tok = model(b["pv"][0], 2 * nh, 2 * nw, ky, kx, gh, gw)
    tgt = F.interpolate(b["skyc"][:, None], size=(gh, gw), mode="area")[:, 0]
    return r, sky, HP["w_sky"] * F.binary_cross_entropy_with_logits(sky_tok, tgt)


# ---------------------------------------------------------------- 검증
class EvalSet(torch.utils.data.Dataset):
    """검증 사진 (자르기 · 증강 없음) → 학생 입력 + 8 화소 칸 지도 (V · G · w · sky). name: val (학습 분포 700) | hyper (HyperSim 500) | fit · held (학습에 쓴 장면 / 떼어 둔 장면 표본)."""

    def __init__(self, name, kind, limit=0):
        self.name, self.kind = name, kind
        if name == "val":
            t = B.index("val")
            t["img"] = [str(D.DATA / D.recs("val")[i]["image"].lstrip("/")) for i in t.i]
            t["full"] = [D.canon_frac(D.recs("val")[i]) for i in t.i]
            t["maps"] = [str(B.paths("val", i)[1]) for i in t.i]
            t["id"], t["grp"] = t.i.astype(str), [D.scene_of(D.recs("val")[i]) for i in t.i]
        elif name in ("fit", "held"):                                 # 학습 분포 사진 (D-114 캐시에 원래 예측이 있는 것) 에서 데이터셋마다 150 장 — fit = 1 단계가 학습에 쓴 장면 · held = 떼어 둔 장면
            H = D.holdout_table().set_index("i")
            t = B.index("train")
            t = t[t.i.map(H.s2) == (name == "held")]
            pick = []                                                  # pandas 판과 무관하게 같은 표본: 데이터셋 이름 순 · 사진 번호 순으로 놓고 numpy 난수 (seed 0, 데이터셋 차례) 로 150 장
            for n_ds, (_, x) in enumerate(sorted(t.groupby("ds"), key=lambda kv: kv[0])):
                ids = np.sort(x.i.to_numpy())
                pick += ids[np.sort(np.random.default_rng([0, n_ds]).permutation(len(ids))[:150])].tolist()
            t = t[t.i.isin(pick)].sort_values("i").reset_index(drop=True)
            t["img"] = [str(D.DATA / D.recs("train")[i]["image"].lstrip("/")) for i in t.i]
            t["full"] = [D.canon_frac(D.recs("train")[i]) for i in t.i]
            t["maps"] = [str(B.paths("train", i)[1]) for i in t.i]
            t["id"], t["grp"] = t.i.astype(str), [D.scene_of(D.recs("train")[i]) for i in t.i]
        else:
            t = pd.read_parquet(C.V2 / "hyper/index.parquet")
            t["full"] = [(0.0, 0.0, 1.0, 1.0)] * len(t)
            t["ds"], t["grp"] = "hypersim", t.scene
        self.t = (t.groupby("ds").head(limit) if limit else t).reset_index(drop=True)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, n):
        torch.set_num_threads(1)
        r = self.t.iloc[n]
        ph, pw = int(r.ph), int(r.pw)
        th, tw = D.thumb_tokens(ph, pw)
        size = (32 * pw, 32 * ph) if self.kind == "token" else (32 * tw, 32 * th)
        img = D._open(r.img, size, r.full)
        x = dict(pv=D.model_pixels(img)[0]) if self.kind == "token" else dict(u8=D.to_u8(img))
        return dict(n=n, nh=ph, nw=pw, ky=D.coarse_grid(th, tw)[0], kx=D.coarse_grid(th, tw)[1], **x, **B.load_maps(r.maps))


@torch.no_grad()
def evaluate(model, arm, name, limit=0, workers=6):
    """사진별: 원래 DepthVLM (V) · 가지의 굵은 무늬로 바꾼 것 (F, 하늘 = 가지) · 저장된 하늘로 바꾼 것 — SILog · 평균 로그 오차 (× 100)."""
    model.eval()
    ds = EvalSet(name, "token" if arm == "read" else "thumb", limit)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, collate_fn=D.collate, **D.mp(workers))
    rows = []
    for b in dl:
        b = to_dev(b)
        ky, kx = int(b["ky"][0]), int(b["kx"][0])
        if arm == "read":
            nh, nw = int(b["nh"][0]), int(b["nw"][0])
            r, sky, _ = model(b["pv"][0], 2 * nh, 2 * nw, ky, kx, *D.geo_grid(nh, nw))
        else:
            r, sky = model(b["u8"], ky, kx)
        sky_hat = (B.up(torch.sigmoid(sky), b["V"].shape[-2:]) > 0.5).float()
        Fp, _ = B.merge(b["V"], r, sky_hat, center="median")
        Fc, _ = B.merge(b["V"], r, b["sky"], center="median")
        (s0, m0), (sp, mp), (sc, mc) = B.score(b["V"], b["G"], b["w"]), B.score(Fp, b["G"], b["w"]), B.score(Fc, b["G"], b["w"])
        t = ds.t.iloc[int(b["n"][0])]
        rows.append(dict(id=t.id, ds=t.ds, grp=t.grp, V=s0.item(), F=sp.item(), F_cachesky=sc.item(), mu_V=m0.item(), mu_F=mp.item(), mu_F_cachesky=mc.item()))
    model.train()
    return pd.DataFrame(rows)


def summarize(df, name):
    if name == "hyper":
        return dict(V=float(df.V.mean()), F=float(df.F.mean()), F_cachesky=float(df.F_cachesky.mean()), n=len(df))
    return {c: B.macro(df, c) for c in ("V", "F", "F_cachesky")}


def line(s, name):
    if name == "hyper":
        return f"HyperSim {s['n']} 장: DepthVLM {s['V']:.2f} → 가지 {s['F']:.2f}"
    return (f"{dict(val='검증 700', fit='학습에 쓴 장면', held='떼어 둔 장면')[name]}: DepthVLM {s['V']['all']:.2f} → 가지 {s['F']['all']:.2f} (실내 {s['V']['indoor']:.2f} → {s['F']['indoor']:.2f} · "
            f"주행 {s['V']['driving']:.2f} → {s['F']['driving']:.2f})")


# ---------------------------------------------------------------- 학습
def collate_each(items):
    return [D.collate([it]) for it in items]


def load_stream(name, limit=0):
    t = pd.read_parquet(C.V2 / f"stream_{name}.parquet")
    return t.iloc[:limit].reset_index(drop=True) if limit else t


def train(a):
    torch.manual_seed(HP["seed"])
    tag = a.tag or f"{a.arm}_{a.stream}"
    out, ck = RUNS / tag, CKPT / tag
    out.mkdir(parents=True, exist_ok=True)
    ck.mkdir(parents=True, exist_ok=True)
    stream = load_stream(a.stream, a.limit)
    groups = D.batches(stream, HP["bs"])
    model = build(a.arm)
    if a.init:                                                        # (11-12) 다른 실행의 가중치만 받아 시작 — 걸음 · 학습률 일정 · 옵티마이저는 처음부터
        st0 = torch.load(a.init, map_location="cpu", weights_only=False)
        assert st0.get("arm", a.arm) == a.arm, (st0.get("arm"), a.arm)
        model.load_state(st0["state"])
        print(f"[{a.tag or a.arm + '_' + a.stream}] 가중치를 {a.init} ({st0.get('stream')} · 걸음 {st0.get('step')}) 에서 받아 시작 — 학습률 일정은 처음부터", flush=True)
    model.train()
    pg = model.groups(HP["lr_lora"], HP["lr_new"], HP["lr_body"], HP["wd"])
    opt = torch.optim.AdamW(pg)
    base = [g["lr"] for g in pg]
    total, warm = len(groups), min(HP["warmup"], max(1, len(groups) // 10))
    sched = lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total))))
    step, log = 0, []
    if a.resume and (ck / "last.pt").exists():
        st = torch.load(ck / "last.pt", map_location="cpu", weights_only=False)
        assert st["n_groups"] == total and st["stream"] == a.stream, "이어 하기: 학습 순서가 다름"
        model.load_state(st["state"])
        opt.load_state_dict(st["opt"])
        step, log = st["step"], st["log"]
        print(f"[{tag}] 이어 하기: 걸음 {step}/{total}", flush=True)
    kind = "token" if a.arm == "read" else "thumb"
    dl = torch.utils.data.DataLoader(D.StreamSet(stream, kind), batch_sampler=groups[step:], collate_fn=collate_each if a.arm == "read" else D.collate, prefetch_factor=4, **D.mp(a.workers))
    params = [q for g in pg for q in g["params"]]
    print(f"[{tag}] 사진 {len(stream)} 장 (정답 {(stream.kind == 'L').sum()} · 무라벨 {(stream.kind == 'U').sum()}) · 걸음 {total} · 더해지는 매개변수 {model.n_added() / 1e6:.2f} M", flush=True)
    t0, run, n0 = time.time(), defaultdict(list), step
    for batch in dl:
        for g, lr0 in zip(opt.param_groups, base):
            g["lr"] = lr0 * sched(step)
        opt.zero_grad(set_to_none=True)
        if a.arm == "read":                                            # 같은 8 장을 한 장씩 — 기울기 누적
            for it in batch:
                it = to_dev(it)
                r, sky, extra = forward(model, a.arm, it)
                loss, parts = stage1_loss(r, sky, it)
                ((loss + extra) / len(batch)).backward()
                for k2, v in parts.items():
                    run[k2].append(v)
        else:
            b = to_dev(batch)
            r, sky, _ = forward(model, a.arm, b)
            loss, parts = stage1_loss(r, sky, b)
            loss.backward()
            for k2, v in parts.items():
                run[k2].append(v)
        if not all(torch.isfinite(q.grad).all() for q in params if q.grad is not None):
            raise RuntimeError(f"기울기가 유한하지 않음 (걸음 {step})")
        torch.nn.utils.clip_grad_norm_(params, HP["clip"])
        opt.step()
        step += 1
        if step % a.log_every == 0 or step == total:
            row = dict(step=step, sec_per_step=(time.time() - t0) / (step - n0), gpu_gb=torch.cuda.max_memory_allocated() / 2 ** 30, **{k2: sum(x for x, _ in v) / max(1.0, sum(c for _, c in v)) for k2, v in run.items()})
            log.append(row)
            run = defaultdict(list)
            print(f"[{tag}] {step}/{total} 손실 {row['loss']:.4f} (정답 {row['gt']:.4f} · 채움 {row['fill']:.4f} · 기울기 {row['gm']:.4f} · 하늘 {row['sky']:.4f} · 문턱 {row['gate']:.2f})"
                  f" | {row['sec_per_step']:.3f} 초/걸음 · GPU {row['gpu_gb']:.1f} GB · 남은 {(total - step) * row['sec_per_step'] / 60:.0f} 분", flush=True)
            pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
        if step % a.ckpt_every == 0 and step < total:
            tmp = ck / "last.tmp.pt"
            torch.save(dict(state=model.state(), opt=opt.state_dict(), step=step, log=log, n_groups=total, stream=a.stream, hp=HP), tmp)
            tmp.replace(ck / "last.pt")
        if a.keep_at and step in a.keep_at and step < total:           # (11-18 사용자: 중간중간 체크포인트) 그 시점을 따로 남긴다 — last.pt 와 같은 내용이라 거기서 이어 하기 · 그 시점의 가중치 쓰기 모두 가능
            tmp = ck / f"step{step}.tmp.pt"
            torch.save(dict(state=model.state(), opt=opt.state_dict(), step=step, log=log, n_groups=total, stream=a.stream, hp=HP, arm=a.arm), tmp)
            tmp.replace(ck / f"step{step}.pt")
        if a.val_at and step in a.val_at and step < total:
            df = evaluate(model, a.arm, "val", workers=a.workers)
            df.to_csv(out / f"val_step{step}.csv", index=False)
            print(f"[{tag}] 걸음 {step} (보고만) " + line(summarize(df, "val"), "val"), flush=True)
    torch.save(dict(state=model.state(), step=step, n_groups=total, stream=a.stream, hp=HP, arm=a.arm, init=a.init), ck / "final.pt")
    res = dict(steps=step, minutes=round((time.time() - t0) / 60, 1), images=len(stream), init=a.init)
    for name in a.sets:
        df = evaluate(model, a.arm, name, limit=a.eval_limit, workers=a.workers)
        df.to_csv(out / f"{name}_final.csv", index=False)
        res[name] = summarize(df, name)
        print(f"[{tag}] 끝 " + line(res[name], name), flush=True)
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"[{tag}] 끝 · {res['minutes']} 분 · EXIT 0", flush=True)


def evaluate_only(a):
    tag = a.tag or f"{a.arm}_{a.stream}"
    model = build(a.arm)
    model.load_state(torch.load(a.ckpt or CKPT / tag / "final.pt", map_location="cpu", weights_only=False)["state"])
    for name in a.sets:
        df = evaluate(model, a.arm, name, limit=a.eval_limit, workers=a.workers)
        (RUNS / tag).mkdir(parents=True, exist_ok=True)
        df.to_csv(RUNS / tag / f"{name}_final.csv", index=False)
        print(f"[{tag}] " + line(summarize(df, name), name), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("train", "eval"))
    ap.add_argument("--arm", choices=("dav2s", "siglip", "read"), required=True)
    ap.add_argument("--stream", default="s1")
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval_limit", type=int, default=0)
    ap.add_argument("--sets", nargs="*", default=["val"])
    ap.add_argument("--val_at", type=int, nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--ckpt_every", type=int, default=2000)
    ap.add_argument("--keep_at", type=int, nargs="*", default=[], help="이 걸음들의 체크포인트를 step<걸음>.pt 로 따로 남긴다 (기본: 안 남김)")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", default="")
    a = ap.parse_args(ARGV)
    import resource                                                  # 작업자가 넘기는 텐서마다 파일 번호를 하나씩 쓴다 — 한도를 넉넉히
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (65536 if hard == resource.RLIM_INFINITY else min(hard, 65536), hard))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    (train if a.stage == "train" else evaluate_only)(a)


if __name__ == "__main__":
    main()
