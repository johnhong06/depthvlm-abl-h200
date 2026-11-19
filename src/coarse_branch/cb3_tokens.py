"""기하 토큰을 '메모 없이' 계산 (사용자 11-12: 마지막에 처음부터 다시 학습할 때 메모로 인한 지연을 없애기).

지금 구현 (cb2_models.GeoTokens): 본 통과가 24 층을 다 지난 뒤, 층마다 잡아 둔 qkv 출력 24 개 (메모) 로 토큰을 따로 24 층 계산한다. 패치 열쇠의 위치 회전도 층마다 다시 계산한다.
이 파일 (GeoTokensInline): 본 통과가 한 층을 지날 때마다 그 자리에서 토큰도 한 층 나아간다 (블록 뒤 훅).
  · 패치 열쇠는 본 통과가 이미 회전해 둔 것을 그대로 받는다 (회전 함수가 돌려준 텐서를 가리키기만 함 — 다시 계산하지 않음)
  · 잡아 두는 값은 지금 지나는 층 것 하나뿐 (24 층 치를 쌓아 두지 않음)
  · 본 통과의 계산 경로는 그대로다. 훅과 감싼 함수는 값을 읽기만 한다 → 본 통과 출력은 비트까지 같아야 한다
  · 토큰 값 · 기울기도 지금 구현과 같아야 한다 (같은 식 · 같은 순서) → check 로 확인
매개변수 이름 · 모양 · 만드는 순서는 cb2_models.GeoTokens 와 같다 (state · load_state 호환 · 같은 씨앗이면 같은 초기값).

쓰는 법:  geo.arm(Hp, Wp, gh, gw) → 본 통과 (vis 가 불리면 됨 — 따로든 VLM 안에서든) → geo.take() 가 꺼내는 층의 토큰을 돌려준다. arm 하지 않으면 훅은 아무것도 안 한다.

    ~/venv/depthvlm/bin/python -I -B cb3_tokens.py check [--read <read 체크포인트>] [--per 2]   → results/tokens_inline_check.json  (값 · 기울기가 같은가 — GPU 약 3 GB)
    ~/venv/depthvlm/bin/python -I -B cb3_tokens.py time  [--read …] [--per 4]                    → results/tokens_inline_time.json   (시간 · 메모리 — GPU 가 빈 상태에서만 뜻이 있다)
"""
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import cb2_models as M2  # noqa: E402
import cb_models as M  # noqa: E402


def rope_qk(qk, cos, sin):
    """토큰의 q · k 를 한 번에 회전 — qk (토큰, 2, 헤드, 헤드 차원). 식은 cb2_models.rope (= 모델의 apply_rotary_pos_emb_vision) 와 같다 (원소별 계산이라 값도 같다)."""
    xf = qk.float()
    half = xf.shape[-1] // 2
    return (xf * cos[:, None, None, :] + torch.cat((-xf[..., half:], xf[..., :half]), -1) * sin[:, None, None, :]).to(qk.dtype)


def tap_rotation(vis):
    """본 통과의 위치 회전 함수를 감싸서, 켜져 있을 때만 '들어온 열쇠 · 회전된 열쇠' 를 가리켜 둔다. 원래 함수의 결과를 그대로 돌려주므로 본 통과 값은 바뀌지 않는다."""
    mod = sys.modules[type(vis.blocks[0].attn).__module__]
    fn = mod.apply_rotary_pos_emb_vision
    if not hasattr(fn, "_cb3_tap"):
        tap = dict(on=False, k_in=None, k_out=None)

        def wrapped(q, k, cos, sin, _orig=fn, _tap=tap):
            out = _orig(q, k, cos, sin)
            if _tap["on"]:
                _tap["k_in"], _tap["k_out"] = k, out[1]
            return out

        wrapped._cb3_tap = tap
        mod.apply_rotary_pos_emb_vision = wrapped
    return mod.apply_rotary_pos_emb_vision._cb3_tap


class GeoTokensInline(M2.GeoTokens):
    """읽기 전용 기하 토큰 — 본 통과와 같이 한 층씩 계산 (메모 없음). vis = DepthVLM 의 비전 인코더 (고정). 사진 한 장씩."""

    def __init__(self, vis, rank=M2.RANK, taps=M2.TAPS):
        nn.Module.__init__(self)                                      # GeoTokens.__init__ 은 메모용 훅을 달기 때문에 부르지 않는다 (매개변수는 같은 순서로 만든다)
        for q in vis.parameters():
            q.requires_grad_(False)
        self.vis, self.taps = vis.eval(), tuple(taps)
        d, inner, L = vis.config.hidden_size, vis.config.intermediate_size, len(vis.blocks)
        dims = ((d, 3 * d), (d, d), (d, inner), (inner, d))            # attn.qkv · attn.proj · mlp.linear_fc1 · mlp.linear_fc2
        self.A = nn.ParameterList([nn.Parameter(torch.empty(rank, i)) for _ in range(L) for i, _ in dims])
        self.B = nn.ParameterList([nn.Parameter(torch.zeros(o, rank)) for _ in range(L) for _, o in dims])
        for a in self.A:
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        self.e = nn.Parameter(torch.zeros(d))
        self._plan = self._g = self._cs = self._outs = self._cur = None
        self._tap = tap_rotation(vis)
        vis.blocks[0].register_forward_pre_hook(self._start, with_kwargs=True)
        for l, blk in enumerate(vis.blocks):
            blk.attn.qkv.register_forward_hook(lambda m, i, o, l=l: self._catch(l, o))
            blk.register_forward_hook(lambda m, i, o, l=l: self._step(l, m))

    def tokens(self, *a, **k):
        raise RuntimeError("GeoTokensInline 은 arm() → 본 통과 → take() 로 쓴다")

    def arm(self, Hp, Wp, gh, gw):
        """다음 본 통과에서 토큰을 같이 계산하게 켠다. 기울기를 낼지는 지금의 설정 (torch.is_grad_enabled) 을 따른다 — 본 통과가 no_grad 안에서 돌아도."""
        self._plan = (Hp, Wp, gh, gw, torch.is_grad_enabled())
        self._g = self._cs = self._cur = None
        self._outs = []
        self._tap.update(on=True, k_in=None, k_out=None)

    def take(self):
        outs = self._outs
        self._plan = self._g = self._cs = self._outs = self._cur = None
        self._tap.update(on=False, k_in=None, k_out=None)
        assert outs is not None and len(outs) == len(self.taps), ("본 통과가 돌지 않았거나 도중에 끊김", None if outs is None else len(outs))
        return outs

    def _start(self, m, args, kwargs):
        """0 층 입력 (패치 임베딩 + 위치) 에서 토큰의 시작값과 토큰 자리의 회전 각을 만든다 — cb2_models.GeoTokens.tokens 의 앞부분과 같은 식."""
        if self._plan is None:
            return
        Hp, Wp, gh, gw, grad = self._plan
        x0, vis = args[0], self.vis
        assert x0.shape[0] == Hp * Wp, (tuple(x0.shape), Hp, Wp)
        dev = x0.device
        with torch.autocast("cuda", enabled=False), torch.set_grad_enabled(grad):
            ras = M.unshuffle(x0.detach(), 1, Hp, Wp)[0].reshape(Hp, Wp, -1).permute(2, 0, 1).float()
            g = F.adaptive_avg_pool2d(ras[None], (gh, gw))[0].flatten(1).T
            self._g = (g + self.e).to(x0.dtype)
            rows = (torch.arange(gh, device=dev, dtype=torch.float32) + 0.5) * Hp / gh - 0.5
            cols = (torch.arange(gw, device=dev, dtype=torch.float32) + 0.5) * Wp / gw - 0.5
            pos = torch.stack(torch.meshgrid(rows, cols, indexing="ij"), -1).reshape(gh * gw, 2)
            inv = vis.rotary_pos_emb.inv_freq
            fr = (pos.to(inv.dtype)[:, :, None] * inv[None, None, :]).flatten(1)
            emb = torch.cat((fr, fr), -1)
            self._cs = (emb.cos().float(), emb.sin().float())

    def _catch(self, l, o):
        if self._plan is not None:
            self._cur = (l, o)

    def _step(self, l, blk):
        """본 통과가 l 층을 막 지난 자리 — 그 층의 패치 열쇠 (회전된 것) · 값으로 토큰을 한 층 나아가게 한다. cb2_models.GeoTokens.tokens 의 한 바퀴와 같은 식."""
        if self._plan is None:
            return
        Hp, Wp, gh, gw, grad = self._plan
        N, Mg = Hp * Wp, gh * gw
        lq, qkv_p = self._cur
        k_in, kp = self._tap["k_in"], self._tap["k_out"]
        assert lq == l and qkv_p.shape[0] == N and k_in is not None and k_in.untyped_storage().data_ptr() == qkv_p.untyped_storage().data_ptr(), ("이 층의 패치 열쇠를 못 잡음", l, lq)
        self._cur = None
        self._tap.update(k_in=None, k_out=None)
        at, nh = blk.attn, blk.attn.num_heads
        kp, vp = kp.detach(), qkv_p.detach().reshape(N, 3, nh, -1)[:, 2]
        cg, sg = self._cs
        with torch.autocast("cuda", enabled=False), torch.set_grad_enabled(grad):
            g = self._g
            h = blk.norm1(g)
            qkv = F.linear(h, at.qkv.weight, at.qkv.bias) + self.lora(l, 0, h)      # 모듈을 부르면 본 통과용 훅이 다시 불린다 — 함수로 계산
            qkv = qkv.reshape(Mg, 3, nh, -1)
            qk = rope_qk(qkv[:, :2], cg, sg)
            K, V = torch.cat((kp, qk[:, 1])), torch.cat((vp, qkv[:, 2]))
            o = F.scaled_dot_product_attention(qk[:, 0].transpose(0, 1)[None], K.transpose(0, 1)[None], V.transpose(0, 1)[None])[0].transpose(0, 1).reshape(Mg, -1)
            g = g + at.proj(o) + self.lora(l, 1, o)
            h2 = blk.norm2(g)
            m = blk.mlp.act_fn(blk.mlp.linear_fc1(h2) + self.lora(l, 2, h2))
            g = g + blk.mlp.linear_fc2(m) + self.lora(l, 3, m)
            self._g = g
            if l in self.taps:
                self._outs.append(g.float())


class ReadArmInline(M2.ReadArm):
    """읽기 전용 토큰 팔 (1 단계) 의 메모 없는 판 — 쓰는 법 · 저장 형식은 cb2_models.ReadArm 과 같다."""

    def __init__(self):
        nn.Module.__init__(self)                                      # ReadArm.__init__ 은 메모용 GeoTokens 를 만들기 때문에 부르지 않는다 (만드는 순서는 같게)
        self.geo = GeoTokensInline(M.load_visual().to(dtype=torch.bfloat16))
        d = self.geo.vis.config.hidden_size
        self.head = M2.GridHead((d,) * len(M2.TAPS))
        self.sky = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        nn.init.zeros_(self.sky[1].weight)
        nn.init.constant_(self.sky[1].bias, -2.0)

    def forward(self, pv, Hp, Wp, ky, kx, gh, gw):
        assert pv.shape[0] == Hp * Wp, (pv.shape, Hp, Wp)
        self.geo.arm(Hp, Wp, gh, gw)
        with torch.no_grad():
            self.geo.vis(pv.to(torch.bfloat16), grid_thw=torch.tensor([[1, Hp, Wp]], device=pv.device))
        G = self.geo.take()
        with torch.autocast("cuda", enabled=False):
            r, sky = self.head([g[None] for g in G], gh, gw, ky, kx)
            sky_tok = self.sky(G[-1])[:, 0].reshape(1, gh, gw)
        return r, sky, sky_tok


# ---------------------------------------------------------------- 확인 · 시간
def main():
    import argparse
    import json
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("check", "time"))
    ap.add_argument("--read", default=str(Path.home() / "checkpoints/dvsilog/coarse_branch/v2/read_s1b/final.pt"))
    ap.add_argument("--per", type=int, default=2, help="검증 사진을 데이터셋마다 몇 장 (7 종)")
    ap.add_argument("--rep", type=int, default=7)
    ap.add_argument("--n_geo", type=int, default=0, help="기하 토큰 수 (0 = 기본 150). 가중치는 --read 것을 그대로 — 시간 · 메모리만 볼 때")
    a = ap.parse_args()
    sys.argv = sys.argv[:1]   # 아래 모듈이 인자를 다시 읽지 않게
    import numpy as np
    import cb2_data as D
    import cb2_stage1 as S1
    DEV, MB = "cuda", 2 ** 20
    if a.n_geo:
        D.N_GEO = a.n_geo
    sfx = f"_n{a.n_geo}" if a.n_geo else ""
    st = torch.load(a.read, map_location="cpu", weights_only=False)["state"]["geo"]
    ds = S1.EvalSet("val", "token", a.per)
    items = []
    for n in range(len(ds)):
        it = ds[n]
        nh, nw = int(it["nh"]), int(it["nw"])
        items.append(dict(pv=it["pv"].to(DEV, torch.bfloat16), Hp=2 * nh, Wp=2 * nw, g=D.geo_grid(nh, nw), ds=ds.t.ds.iloc[n]))
    grid = lambda it: torch.tensor([[1, it["Hp"], it["Wp"]]], device=DEV)
    flat = lambda o: [o.last_hidden_state, o.pooler_output] + list(o.deepstack_features)
    loss = lambda T: sum(t.pow(2).mean() for t in T)

    def clock(fn):
        ts = []
        for _ in range(a.rep):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append(1e3 * (time.perf_counter() - t0))
        return float(np.median(ts))

    def peak(fn):
        """fn 을 한 번 돌리는 동안 가중치 위로 더 쓰는 최대 메모리 (MB) — 앞 실행이 남긴 것 (지금 구현의 메모 · 기울기) 은 먼저 비운다."""
        clear()
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        fn()
        torch.cuda.synchronize()
        return (torch.cuda.max_memory_allocated() - base) / MB

    mods = []

    def clear():
        for mod in mods:
            if hasattr(mod, "_qkv"):
                mod._x0 = None
                mod._qkv[:] = [None] * len(mod._qkv)
            for p in mod.trainable():
                p.grad = None

    # ① 아무것도 달지 않은 ViT (회전 함수도 감싸기 전) — 본 통과의 기준값 · 기준 시간
    vis_new = M.load_visual().to(DEV, torch.bfloat16).eval()
    rows = [dict(ds=it["ds"], patches=it["Hp"] * it["Wp"], tokens=it["g"][0] * it["g"][1]) for it in items]
    with torch.no_grad():
        for _ in range(3):
            vis_new(items[0]["pv"], grid_thw=grid(items[0]))
        ref = [[t.cpu() for t in flat(vis_new(it["pv"], grid_thw=grid(it)))] for it in items]
        if a.mode == "time":
            for it, r in zip(items, rows):
                r["vit"] = clock(lambda: vis_new(it["pv"], grid_thw=grid(it)))
                r["peak_vit"] = peak(lambda: vis_new(it["pv"], grid_thw=grid(it)))
    # ② 새 구현은 같은 ViT 에, 지금 구현은 따로 올린 같은 가중치의 ViT 에 (메모용 훅이 새 구현의 메모리 · 시간에 섞이지 않게)
    new = GeoTokensInline(vis_new).to(DEV)
    old = M2.GeoTokens(M.load_visual().to(DEV, torch.bfloat16)).to(DEV)
    new.load_state(st)
    old.load_state(st)
    mods += [old, new]

    def run_old(it, grad=False):
        with torch.no_grad():
            o = old.vis(it["pv"], grid_thw=grid(it))
        with torch.set_grad_enabled(grad):
            return o, old.tokens(it["Hp"], it["Wp"], *it["g"])

    def run_new(it, grad=False):
        with torch.set_grad_enabled(grad):
            new.arm(it["Hp"], it["Wp"], *it["g"])
        with torch.no_grad():
            o = new.vis(it["pv"], grid_thw=grid(it))
        return o, new.take()

    def step(run, mod, it):
        _, T = run(it, True)
        for p in mod.trainable():
            p.grad = None
        loss(T).backward()

    same = lambda xs, ys: all(torch.equal(x.cpu(), y.cpu()) for x, y in zip(xs, ys))
    if a.mode == "check":
        for it, r, rf in zip(items, rows, ref):
            (oo, To), (on, Tn) = run_old(it), run_new(it)
            with torch.no_grad():
                idle = flat(new.vis(it["pv"], grid_thw=grid(it)))        # 켜지 않은 상태의 본 통과
            r.update(main_same_armed=same(flat(on), rf), main_same_idle=same(idle, rf), main_same_oldvit=same(flat(oo), rf),
                     tok_same=same(To, Tn), tok_maxabs=max(float((x - y).abs().max()) for x, y in zip(To, Tn)), tok_scale=float(To[-1].abs().mean()))
            # 기울기: bf16 역전파는 같은 코드를 두 번 돌려도 조금 다르다 (11-12 확인) → 비트 비교가 아니라 '지금 구현을 두 번 돌린 차이' 와 견준다
            old.train()
            new.train()
            gs = []
            for run, mod in ((run_old, old), (run_old, old), (run_new, new)):
                step(run, mod, it)
                gs.append([p.grad.clone() for p in mod.trainable()])
            rel = lambda xs, ys: max(float((x - y).abs().max() / x.abs().max().clamp_min(1e-30)) for x, y in zip(xs, ys))
            vec = lambda xs: torch.cat([x.flatten().double() for x in xs])
            r.update(grad_maxrel=rel(gs[0], gs[2]), grad_maxrel_rerun=rel(gs[0], gs[1]), grad_cos=float(F.cosine_similarity(vec(gs[0]), vec(gs[2]), dim=0)),
                     grad_cos_rerun=float(F.cosine_similarity(vec(gs[0]), vec(gs[1]), dim=0)), grad_nonzero=sum(int(x.abs().max() > 0) for x in gs[0]), grad_n=len(gs[0]))
            r["grad_ok"] = r["grad_cos"] > 0.9999
            print({k: (round(v, 7) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        keys = ("main_same_armed", "main_same_idle", "main_same_oldvit", "tok_same", "grad_ok")
        out = dict(n=len(rows), read=a.read, all={k: all(r[k] for r in rows) for k in keys}, tok_maxabs=max(r["tok_maxabs"] for r in rows),
                   grad_maxrel=max(r["grad_maxrel"] for r in rows), grad_maxrel_rerun=max(r["grad_maxrel_rerun"] for r in rows),
                   grad_cos_min=min(r["grad_cos"] for r in rows), grad_cos_rerun_min=min(r["grad_cos_rerun"] for r in rows), rows=rows)
        print(json.dumps({k: v for k, v in out.items() if k != "rows"}, ensure_ascii=False), flush=True)
        (HERE / f"results/tokens_inline_check{sfx}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
        assert all(out["all"].values()), "같지 않은 것이 있음"
        return
    # ---- 시간 · 메모리 (추론: 본 통과 + 토큰 / 학습: 본 통과 + 토큰 + 역전파)
    old.train()
    new.train()
    for _ in range(3):
        step(run_old, old, items[0])
        step(run_new, new, items[0])
    for it, r in zip(items, rows):
        for name, run, mod in (("old", run_old, old), ("new", run_new, new)):
            r[f"inf_{name}"] = clock(lambda: run(it))
            r[f"peak_inf_{name}"] = peak(lambda: run(it))
            r[f"train_{name}"] = clock(lambda: step(run, mod, it))
            r[f"peak_train_{name}"] = peak(lambda: step(run, mod, it))
        clear()
    med = {k: float(np.median([r[k] for r in rows])) for k in rows[0] if k != "ds"}
    med.update(tok_old=med["inf_old"] - med["vit"], tok_new=med["inf_new"] - med["vit"])
    apps = os.popen("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader").read().strip().splitlines()
    used = os.popen("nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader").read().strip()
    out = dict(n=len(rows), median=med, gpu_now=used, gpu_other_procs=[x for x in apps if not x.startswith(str(os.getpid()))], rows=rows)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1, ensure_ascii=False), flush=True)
    (HERE / f"results/tokens_inline_time{sfx}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
