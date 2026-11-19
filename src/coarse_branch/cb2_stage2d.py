"""2 단계 변형 실험 (사용자 11-12 승인): 상대 깊이 헤드에 '2 × 2 로 합치기 전' 의 16 화소 패치 특징을 넣으면 나아지는가, 다듬어 넣어야 하는가.

원래 헤드 · 지금의 상대 깊이 헤드는 2 × 2 로 합친 토큰 (32 화소 칸) 만 본다. 본 통과의 ViT 5 · 11 · 17 번째 블록 뒤 패치 값 (16 화소, 합치기 전) 은 이미 계산돼 있으므로
읽기만 해서 (훅, 본 통과는 그대로) 헤드의 1/4 · 1/8 · 1/16 단계에 더한다.

팔 (모두 토큰 주입 있음 · 같은 사진 · 같은 순서 · cb2_stage2 와 같은 조리법 · 팔마다 옵티마이저 따로 · 토큰 주입 선형층의 초기값은 팔끼리 같음):
  prop    토큰만 (cb2_stage2 의 제안 팔과 같은 구조 — 이 실행 안의 기준)
  raw     + 합치기 전 특징을 그대로: 채널 LN → 1 × 1 투영 → 0 에서 여는 게이트 (geo_mix.Head 의 hr 경로 그대로)
  conv    + 작은 합성곱으로 다듬어서: 1 × 1 (1024 → 128) → 3 × 3 두 층 → 1 × 1 → 게이트
  guide   + 기하 토큰을 같이 보며 다듬어서: conv 팔의 입력에 23 번째 블록 뒤 토큰 특징 (1024 → 128, 패치 격자로 bilinear) 을 이어 붙임
배율 헤드는 여기서 학습하지 않는다 (SILog 는 배율과 무관). 고르는 것은 학습 분포 검증 700 장으로만 — 평가 6 세트는 이 파일에서 쓰지 않는다.

    ~/venv/depthvlm/bin/python -I -B -u cb2_stage2d.py train --read <read 체크포인트> --tag s2d --val_at 4000 8000 --resume
    → ~/checkpoints/dvsilog/coarse_branch/v2/<tag>/{last.pt, final.pt} · ~/data/dvsilog/coarse_branch/v2runs/<tag>/{train_log.csv, val_*.csv}
"""
import argparse
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ARMS = ("prop", "raw", "conv", "guide")
DTAPS = (5, 11, 17)            # 합치기 전 패치 값을 읽는 블록 (헤드 입력 F1–F3 이 나오는 블록과 같음)
SEED = 1111


def build(S2, nn, F, torch):
    """torch 를 불러온 뒤에 만드는 것들 (일꾼 프로세스가 이 파일을 다시 읽어도 무거운 것을 올리지 않게)."""
    M = S2.M2.M

    class Detail:
        """본 통과가 지나갈 때 ViT 블록 출력 (합치기 전 패치 값) 을 잡아 둔다 — 읽기만 (본 통과 값은 그대로)."""

        def __init__(self, vis):
            self.h = [None] * len(DTAPS)
            for j, l in enumerate(DTAPS):
                vis.blocks[l].register_forward_hook(lambda m, i, o, j=j: self.h.__setitem__(j, (o[0] if isinstance(o, (tuple, list)) else o).detach()))

        def maps(self, ph, pw):
            """→ 지도 셋 (1, 1024, 2ph, 2pw) fp32, 채널 LN (학습 계수 없음)."""
            Hp, Wp = 2 * ph, 2 * pw
            out = []
            for h in self.h:
                assert h is not None and h.shape[0] == Hp * Wp, (None if h is None else tuple(h.shape), Hp, Wp)
                r = M.unshuffle(h, 1, Hp, Wp)[0].float()
                out.append(F.layer_norm(r, (r.shape[-1],)).T.reshape(1, -1, Hp, Wp))
            return out

    class HeadD(S2_G(S2).Head):
        """토큰 주입 헤드 + 합치기 전 특징을 작은 합성곱으로 다듬어 넣는 길. guide=True 면 기하 토큰 (23 번째 블록 뒤) 을 같이 본다."""

        def __init__(self, base, guide):
            super().__init__(base, True)
            self.guide = guide
            self.dred = nn.ModuleList([nn.Conv2d(1024, 128, 1) for _ in range(3)])
            self.gred = nn.ModuleList([nn.Linear(1024, 128) for _ in range(3)]) if guide else None
            cin = 256 if guide else 128
            self.dconv = nn.ModuleList([nn.Sequential(nn.Conv2d(cin, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()) for _ in range(3)])
            self.dout = nn.ModuleList([nn.Conv2d(128, base.projects[i].out_channels, 1) for i in range(3)])
            self.dgate = nn.Parameter(torch.zeros(3))

        def forward(self, feats, geo, ph, pw, ghr=None, rgb=None, inject=True):
            out = []
            for i in range(4):
                x = self.h.norm(feats[i])
                x = x + self.gate[i] * self.proj[i](self.gnorm(geo[i]))[None]
                x = x.permute(0, 2, 1).reshape(1, x.shape[-1], ph, pw)
                y = self.h.resize_layers[i](self.h.projects[i](x))
                if i < 3:
                    d = self.dred[i](ghr[i])
                    if self.guide:
                        g = self.gred[i](self.gnorm(geo[3])).T.reshape(1, 128, ph, pw)
                        d = torch.cat((d, F.interpolate(g, size=d.shape[-2:], mode="bilinear", align_corners=False)), 1)
                    y = y + self.dgate[i] * F.interpolate(self.dout[i](self.dconv[i](d)), size=y.shape[-2:], mode="bilinear", align_corners=False)
                out.append(y)
            fused = F.interpolate(self.h._scratch_forward(out), size=(ph * 32, pw * 32), mode="bilinear", align_corners=True)
            return self.h.scratch.output_conv2(fused).squeeze(1)[0]

    def heads(net):
        G = net.G
        R = {}
        for arm in ARMS:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(SEED)                               # 토큰 주입 선형층 (맨 먼저 만들어짐) 의 초기값이 팔끼리 같게
                R[arm] = (G.Head(net.base, True) if arm == "prop" else G.Head(net.base, True, hr=True) if arm == "raw" else HeadD(net.base, arm == "guide")).to(S2.DEV)
        return R

    return Detail, heads


def S2_G(S2):
    import geo_mix as G
    return G


def optimizers(R, hp, torch):
    """cb2_stage2 와 같은 조리법: 헤드 2e-5 · 새 층 1e-4 · 게이트 1e-3 (AdamW, wd 0). 팔마다 따로."""
    opt = {}
    for arm, r in R.items():
        hd = [q for n, q in r.named_parameters() if n.startswith("h.")]
        gt = [q for n, q in r.named_parameters() if n in ("gate", "hgate", "dgate")]
        new = [q for n, q in r.named_parameters() if not n.startswith("h.") and n not in ("gate", "hgate", "dgate")]
        assert len(hd) + len(gt) + len(new) == len(list(r.parameters()))
        opt[arm] = torch.optim.AdamW([dict(params=hd, lr=hp["lr_head"]), dict(params=new, lr=hp["lr_proj"]), dict(params=gt, lr=hp["lr_gate"])], weight_decay=0.0)
    return opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("train", "val"))
    ap.add_argument("--read", required=True)
    ap.add_argument("--tag", default="s2d")
    ap.add_argument("--stream", default="s2")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--val_limit", type=int, default=0)
    ap.add_argument("--val_at", type=int, nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    sys.argv = sys.argv[:1]   # 아래 모듈이 인자를 다시 읽지 않게
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import cb2_common as C
    import cb2_data as D
    import cb2_stage1 as S1
    import cb2_stage2 as S2

    B, HP, DEV = S2.B, S2.HP, S2.DEV
    Detail, heads = build(S2, nn, F, torch)
    out, ck = S2.RUNS / a.tag, S2.CKPT / a.tag
    out.mkdir(parents=True, exist_ok=True)
    ck.mkdir(parents=True, exist_ok=True)
    net = S2.Net(a.read)
    det = Detail(net.vlm.model.visual)
    R = heads(net)
    pred = lambda arm, feats, geo, dm, ph, pw: net.G.predict(R[arm], feats, geo, ph, pw, ghr=None if arm == "prop" else dm)

    def val_cells(limit):
        ds = S2.ValSet(limit)
        rows = []
        for r in R.values():
            r.eval()
        with torch.no_grad():
            for it in torch.utils.data.DataLoader(ds, batch_size=None, collate_fn=S2.one_item, **D.mp(a.workers)):
                it = S2.to_dev(it)
                p0, feats, ph, pw = net.main_pass(it["inp"])
                dm = det.maps(ph, pw)
                geo, _ = net.geo_feats(ph, pw)
                G_, w_ = it["G"][None], it["w"][None]
                t = ds.t.iloc[int(it["n"])]
                row = dict(i=int(t.i), ds=t.ds, V=B.score(S2.cell_log(p0, ph, pw)[None], G_, w_)[0].item())
                for arm in R:
                    row[arm] = B.score(S2.cell_log(pred(arm, feats, geo, dm, ph, pw), ph, pw)[None], G_, w_)[0].item()
                rows.append(row)
        for r in R.values():
            r.train()
        return pd.DataFrame(rows).sort_values("i").reset_index(drop=True)

    def val_line(df):
        f = lambda c: B.macro(df, c)
        s = " · ".join(f"{c} {f(c)['all']:.2f} / {f(c)['indoor']:.2f} / {f(c)['driving']:.2f}" for c in ["V"] + list(R))
        return f"검증 {len(df)} 장 SILog (7 종 / 실내 / 주행): {s}"

    def gates():
        return {arm: [round(float(x), 4) for n in ("gate", "hgate", "dgate") if getattr(r, n, None) is not None for x in getattr(r, n).detach()] for arm, r in R.items()}

    def save(path, opt, step, log):
        tmp = path.with_name(path.stem + ".tmp.pt")
        torch.save(dict(R={k: r.state_dict() for k, r in R.items()}, opt=None if opt is None else {k: o.state_dict() for k, o in opt.items()},
                        step=step, log=log, read=str(a.read), stream=a.stream, hp=HP, arms=ARMS, dtaps=DTAPS), tmp)
        tmp.replace(path)

    if a.stage == "val":
        st = torch.load(ck / "final.pt", map_location="cpu", weights_only=False)
        for k in R:
            R[k].load_state_dict(st["R"][k])
        df = val_cells(a.val_limit)
        df.to_csv(out / "val_final.csv", index=False)
        print(f"[{a.tag}] " + val_line(df), flush=True)
        return

    stream = pd.read_parquet(C.V2 / f"stream_{a.stream}.parquet")
    if a.limit:
        stream = stream.iloc[: a.limit]
    opt = optimizers(R, HP, torch)
    base = {k: [g["lr"] for g in o.param_groups] for k, o in opt.items()}
    params = {k: list(r.parameters()) for k, r in R.items()}
    total = len(stream) // HP["accum"]
    warm = min(HP["warmup"], max(1, total // 10))
    sched = lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total))))
    step, log = 0, []
    if a.resume and (ck / "last.pt").exists():
        st = torch.load(ck / "last.pt", map_location="cpu", weights_only=False)
        assert st["stream"] == a.stream and st["read"] == str(a.read) and tuple(st["arms"]) == ARMS
        for k in R:
            R[k].load_state_dict(st["R"][k])
            opt[k].load_state_dict(st["opt"][k])
        step, log = st["step"], st["log"]
        print(f"[{a.tag}] 이어 하기: 걸음 {step}/{total}", flush=True)
    print(f"[{a.tag}] 사진 {len(stream)} 장 · 걸음 {total} · 매개변수 (M) " + " · ".join(f"{k} {sum(q.numel() for q in v) / 1e6:.2f}" for k, v in params.items()), flush=True)
    dl = torch.utils.data.DataLoader(D.S2Set(stream), batch_size=None, sampler=range(step * HP["accum"], total * HP["accum"]), collate_fn=S2.one_item, prefetch_factor=4, **D.mp(a.workers))
    t0, run, n0, n_in, first = time.time(), defaultdict(list), step, 0, step == 0
    for it in dl:
        it = S2.to_dev(it)
        if n_in == 0:
            for k, o in opt.items():
                for g, lr0 in zip(o.param_groups, base[k]):
                    g["lr"] = lr0 * sched(step)
                o.zero_grad(set_to_none=True)
        p0, feats, ph, pw = net.main_pass(it["inp"])
        assert (ph, pw) == (int(it["nh"]), int(it["nw"])), ("토큰 격자가 다름", ph, pw, it["nh"], it["nw"])
        dm = det.maps(ph, pw)
        geo, _ = net.geo_feats(ph, pw)
        ln0 = S2.cell_log(p0, ph, pw)
        tgt = {k: (it[k][None] if torch.is_tensor(it[k]) and it[k].dim() == 2 else torch.tensor([float(it[k])], device=DEV)) for k in ("Gc", "wg", "Tc", "qc", "skyc", "has", "gate", "unl")}
        q = (1.0 - tgt["skyc"]).clamp(0, 1)
        if float(q.sum()) <= 0:
            q = torch.ones_like(q)
        for arm in R:
            z = pred(arm, feats, geo, dm, ph, pw)
            if first:                                                  # 게이트가 0 이므로 처음 출력은 원래 예측과 같아야 한다 (배관 확인)
                rel = float(((z - p0).abs() / p0.clamp_min(1e-3)).median())
                print(f"[{a.tag}] 처음 출력 − 원래 예측 (상대 차 중앙값) {arm}: {rel:.2e}", flush=True)
                assert rel < 2e-2, (arm, rel)
            r8 = S2.cell_log(z, ph, pw)[None]
            l_shape, p = S1.stage1_loss(r8, None, tgt, HP["shape"])
            l_anchor = HP["w_anchor"] * (B.wmean(r8, q) - B.wmean(ln0[None], q)).pow(2).mean()
            ((l_shape + l_anchor) / HP["accum"]).backward()
            for k2, v in p.items():
                if k2 in ("loss", "gt", "fill", "gm"):
                    run[f"{arm}_{k2}"].append(v)
        first = False
        n_in += 1
        if n_in < HP["accum"]:
            continue
        n_in = 0
        for k, o in opt.items():
            if not all(torch.isfinite(g.grad).all() for g in params[k] if g.grad is not None):
                raise RuntimeError(f"기울기가 유한하지 않음 ({k}, 걸음 {step})")
            torch.nn.utils.clip_grad_norm_(params[k], HP["clip"])
            o.step()
        step += 1
        if step % a.log_every == 0 or step == total:
            row = dict(step=step, sec_per_img=(time.time() - t0) / ((step - n0) * HP["accum"]), gpu_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                       **{k2: sum(x for x, _ in v) / max(1.0, sum(c for _, c in v)) for k2, v in run.items()}, gates=str(gates()))
            log.append(row)
            run = defaultdict(list)
            print(f"[{a.tag}] {step}/{total} 손실 " + " · ".join(f"{arm} {row[arm + '_loss']:.4f}" for arm in R) + f" | 게이트 {row['gates']} | {row['sec_per_img']:.2f} 초/장 · GPU {row['gpu_gb']:.1f} GB · "
                  f"남은 {(total - step) * HP['accum'] * row['sec_per_img'] / 60:.0f} 분", flush=True)
            pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
        if step % a.ckpt_every == 0 and step < total:
            save(ck / "last.pt", opt, step, log)
        if a.val_at and step in a.val_at and step < total:
            df = val_cells(a.val_limit)
            df.to_csv(out / f"val_step{step}.csv", index=False)
            print(f"[{a.tag}] 걸음 {step} (보고만) " + val_line(df), flush=True)
    save(ck / "final.pt", None, step, log)
    df = val_cells(a.val_limit)
    df.to_csv(out / "val_final.csv", index=False)
    print(f"[{a.tag}] 끝 " + val_line(df), flush=True)
    print(f"[{a.tag}] 끝 · {(time.time() - t0) / 60:.1f} 분 · EXIT 0", flush=True)


if __name__ == "__main__":
    main()
