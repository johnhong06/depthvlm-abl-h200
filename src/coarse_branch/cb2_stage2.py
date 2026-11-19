"""분리 디코딩 실험 2 단계 (PROMPT_v2 3.2–3.4 · 6.3 · 12 절): 본 통과 (ViT → LLM) 고정 · 기하 토큰 (1 단계에서 학습한 LoRA) 고정.
학습되는 것  상대 깊이 헤드 R = 원래 DPT 헤드 사본 (analysis/geo_mix.Head 그대로) — 제안 팔 (기하 토큰 주입, 게이트 0 에서 시작) · 대조 팔 (주입 없음 = 헤드만 다시 학습)
            배율 헤드 S (F4 · F3 → 칸마다 절대 로그 깊이, 목표 = 원래 헤드 출력의 칸 평균). R 과 무관해서 두 팔이 같이 쓴다.
합치기      a_c = m_c − 칸평균_c(ln z) (하늘 아닌 토큰 칸) → 중앙값에서 ln 1.25 안인 칸의 중앙값 s → 깊이 = z · exp(s)      (3.4)
단계
  audit  실행 전 점검 (7 절의 1 · 2 · 3 · 4 · 5 · 9) → v2runs/<tag>/audit.json. 하나라도 어긋나면 예외.
  train  학습 순서 s2 (떼어 둔 장면의 정답 사진 + 1 단계가 안 본 무라벨) 를 한 번. 사진 한 장씩 · 누적 4. 두 팔이 본 통과를 같이 쓴다.
  val    학습 분포 검증 700 장을 공식 로더와 같은 입력으로 (8 화소 칸에서 SILog · 배율 오차 — 보고만)
  eval6  평가 6 세트 한 번 (paper_eval 과 같은 입력 · 크기 맞춤 · 채점). 배율 셋: 배율 헤드 · 원래 헤드 · 정답 위치.
손실 (8 화소 칸, 사진마다)  cb2_stage1.stage1_loss 와 같은 식 (하늘 항 없음 · 기울기 맞춤 4 축척) + 0.05 · (평균 r − 평균 ln D0)² (하늘 아닌 칸) · S: 평균 |m − 칸평균 ln D0|
사용: ~/venv/depthvlm/bin/python -I -B -u cb2_stage2.py audit|train|val|eval6 --read <1 단계 읽기 전용 토큰 팔 final.pt> [--tag s2] [--limit N]
"""
import argparse
import copy
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
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "analysis"))
ARGV, sys.argv = sys.argv[1:], sys.argv[:1]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import cb_common as B  # noqa: E402
import cb2_common as C  # noqa: E402
import cb2_data as D  # noqa: E402
import cb2_models as M2  # noqa: E402
import cb2_stage1 as S1  # noqa: E402

DEV, LN125 = "cuda", math.log(1.25)
HP = dict(accum=4, lr_head=2e-5, lr_proj=1e-4, lr_gate=1e-3, lr_scale=5e-4, warmup=200, clip=1.0, w_anchor=0.05,
          shape=dict(S1.HP, gm_scales=4))                             # 모양 손실: 1 단계와 같은 무게, 기울기 맞춤만 4 축척
RUNS, CKPT = S1.RUNS, S1.CKPT
ARMS = ("prop", "ctrl")


class ScaleHead(nn.Module):
    """배율 헤드 S (3.3): LN(F4) · LN(F3) → 1 × 1 conv 256 씩 → 이어 붙임 → 3 × 3 conv 256 → 3 × 3 conv 128 → 1 × 1 conv → 토큰 칸마다 절대 로그 깊이."""

    def __init__(self, d=2560):
        super().__init__()
        self.p4, self.p3 = nn.Conv2d(d, 256, 1), nn.Conv2d(d, 256, 1)
        self.net = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 1, 1))
        nn.init.constant_(self.net[-1].bias, 1.0)                     # ln 2.7 m 근처에서 시작

    def forward(self, feats, ph, pw):
        f = lambda x: F.layer_norm(x.float(), (x.shape[-1],))[0].T.reshape(1, -1, ph, pw)
        return self.net(torch.cat((self.p4(f(feats[3])), self.p3(f(feats[2]))), 1))[0, 0]


def cell_log(depth, ph, pw):
    """깊이 (32ph × 32pw) → ln 깊이의 8 화소 칸 평균 (analysis/d116_core.cell_log 와 같은 식)."""
    return F.interpolate(depth.float().clamp_min(1e-4).log()[None, None], size=(4 * ph, 4 * pw), mode="area")[0, 0]


def compose(ln_z, m_tok, sky_logit):
    """ln_z (32ph × 32pw) · m_tok (ph, pw) 칸 절대 로그 깊이 · sky_logit (ph, pw) → 더할 상수 s (3.4 의 규칙 그대로)."""
    a = m_tok - F.avg_pool2d(ln_z[None, None], 32)[0, 0]
    clear = sky_logit <= 0
    a = a[clear] if bool(clear.any()) else a.flatten()                # 하늘 아닌 칸이 없으면 모든 칸
    ok = (a - a.median()).abs() < LN125
    return (a[ok] if float(ok.float().mean()) >= 0.05 else a).median()


# ---------------------------------------------------------------- 모델
class Net:
    """본 통과 (고정) + 기하 토큰 (고정) + 헤드들."""

    def __init__(self, read_ckpt, attach=True):
        import geo_mix as G
        self.G = G
        self.proc, self.base, self.features = G.load_models([], vlm=True)
        self.vlm = self.features.model
        self.geo = self.sky = None
        self.read_ckpt = read_ckpt
        if attach:
            self.attach()

    def attach(self):
        """기하 토큰 (읽기 훅 + 1 단계에서 학습한 LoRA · 하늘 탐침) 을 본 통과의 ViT 에 붙인다. 원래 계산은 바뀌지 않는다 (점검 1 · 5)."""
        st = torch.load(self.read_ckpt, map_location="cpu", weights_only=False)["state"]
        self.geo = M2.GeoTokens(self.vlm.model.visual).to(DEV)
        self.geo.load_state(st["geo"])
        d = self.vlm.model.visual.config.hidden_size
        self.sky = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1)).to(DEV)
        self.sky.load_state_dict(st["sky"])
        for q in list(self.geo.parameters()) + list(self.sky.parameters()):
            q.requires_grad_(False)

    def heads(self):
        G = self.G
        return dict(prop=G.Head(self.base, True).to(DEV), ctrl=G.Head(self.base, False).to(DEV)), ScaleHead(self.base.norm.normalized_shape[0]).to(DEV)

    @torch.no_grad()
    def main_pass(self, inp):
        """→ 원래 예측 p0 (32ph × 32pw) · 헤드 입력 F1–F4 · (ph, pw). 기하 토큰이 읽을 값은 훅이 잡아 둔다."""
        p0, feats, _, (ph, pw) = self.features(inp, None, need_p0=True)
        return p0, feats, ph, pw

    @torch.no_grad()
    def geo_feats(self, ph, pw):
        """기하 토큰 → 헤드에 넣을 특징 넷 (토큰 격자로 bilinear, (ph·pw, 1024)) · 토큰 칸의 하늘 logit (ph, pw)."""
        gh, gw = D.geo_grid(ph, pw)
        Gt = self.geo.tokens(2 * ph, 2 * pw, gh, gw)
        up = lambda t: F.interpolate(t.T.reshape(1, -1, gh, gw), size=(ph, pw), mode="bilinear", align_corners=False)[0]
        return [up(g).flatten(1).T.contiguous() for g in Gt], up(self.sky(Gt[-1]))[0]

    def predict(self, head, feats, geo, ph, pw):
        return self.G.predict(head, feats, geo, ph, pw)


def to_dev(x):
    if torch.is_tensor(x):
        return x.to(DEV, non_blocking=True)
    return {k: to_dev(v) for k, v in x.items()} if isinstance(x, dict) else x


def losses(net, R, Sh, it, hp=HP):
    """사진 한 장 → (팔별 R 손실, S 손실, 부분 값). it = cb2_data.S2Set 의 한 줄 (GPU)."""
    p0, feats, ph, pw = net.main_pass(it["inp"])
    assert (ph, pw) == (int(it["nh"]), int(it["nw"])), ("토큰 격자가 다름", ph, pw, it["nh"], it["nw"])
    geo, _ = net.geo_feats(ph, pw)
    ln0 = cell_log(p0, ph, pw)
    tgt = {k: (it[k][None] if torch.is_tensor(it[k]) and it[k].dim() == 2 else torch.tensor([float(it[k])], device=DEV)) for k in ("Gc", "wg", "Tc", "qc", "skyc", "has", "gate", "unl")}
    q = (1.0 - tgt["skyc"]).clamp(0, 1)
    if float(q.sum()) <= 0:
        q = torch.ones_like(q)
    out, parts = {}, {}
    for arm in R:
        z = net.predict(R[arm], feats, geo if arm == "prop" else None, ph, pw)
        r8 = cell_log(z, ph, pw)[None]
        l_shape, p = S1.stage1_loss(r8, None, tgt, hp["shape"])
        l_anchor = hp["w_anchor"] * (B.wmean(r8, q) - B.wmean(ln0[None], q)).pow(2).mean()
        out[arm] = l_shape + l_anchor
        parts.update({f"{arm}_{k}": v for k, v in p.items() if k in ("loss", "gt", "fill", "gm")})
        parts[f"{arm}_anchor"] = (l_anchor.item(), 1.0)
    m = Sh(feats, ph, pw)
    tm, clear = F.avg_pool2d(ln0[None, None], 4)[0, 0], F.avg_pool2d(q[:, None], 4)[0, 0] > 0.5
    if not bool(clear.any()):
        clear = torch.ones_like(clear)
    l_scale = (m - tm).abs()[clear].mean()
    parts["scale"] = (l_scale.item(), 1.0)
    return out, l_scale, parts


def optimizers(R, Sh, hp=HP):
    """팔마다 따로 (서로 영향 없음): 헤드 2e-5 · 주입 선형층 1e-4 · 게이트 1e-3 — F-13 의 조리법 (AdamW, wd 0). S 는 5e-4."""
    h = lambda r: [q for n, q in r.named_parameters() if n.startswith("h.")]
    return dict(prop=torch.optim.AdamW([dict(params=h(R["prop"]), lr=hp["lr_head"]), dict(params=list(R["prop"].proj.parameters()), lr=hp["lr_proj"]),
                                        dict(params=[R["prop"].gate], lr=hp["lr_gate"])], weight_decay=0.0),
                ctrl=torch.optim.AdamW([dict(params=h(R["ctrl"]), lr=hp["lr_head"])], weight_decay=0.0),
                S=torch.optim.AdamW([dict(params=list(Sh.parameters()), lr=hp["lr_scale"])], weight_decay=0.0))


# ---------------------------------------------------------------- 점검
def audit(a):
    res = {}
    out = RUNS / a.tag
    out.mkdir(parents=True, exist_ok=True)
    stream = pd.read_parquet(C.V2 / f"stream_{a.stream}.parquet")
    ds = D.S2Set(stream)
    items = [to_dev(ds[n]) for n in (0, 1, 2, 3)]                     # 정답 · 무라벨 섞임
    assert {float(i["unl"]) for i in items} == {0.0, 1.0}
    net = Net(a.read, attach=False)
    G = net.G
    # (배관) 작업자의 입력이 geo_mix.vlm_inputs 와 같은가
    from PIL import Image
    im = Image.fromarray(np.random.default_rng(0).integers(0, 255, (384, 512, 3), dtype=np.uint8))
    a_, b_ = D.vlm_inputs(im), G.vlm_inputs(net.proc, im)
    res["inputs_same"] = bool(all(torch.equal(a_[k], b_[k]) for k in b_))
    assert res["inputs_same"], "본 통과 입력이 geo_mix.vlm_inputs 와 다름"
    # 점검 1 · 5: 훅을 붙이기 전의 값
    ref = []
    for it in items[:2]:
        with torch.no_grad():
            o = net.vlm(**it["inp"], logits_to_keep=1)
        ref.append((o.depth_pred[0].clone(), [f.clone() for f in net.features(it["inp"], None)[1]], o.logits.clone()))
    msg = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": "What is in this image? Answer briefly."}]}]
    from qwen_vl_utils import process_vision_info
    vq = net.proc(text=[net.proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)], images=[process_vision_info(msg)[0]], padding=True, return_tensors="pt").to(DEV)
    with torch.no_grad():
        vqa0 = net.vlm.model(**{k: vq[k] for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}).last_hidden_state.clone()
    net.attach()
    R, Sh = net.heads()
    same = True
    for it, (p_ref, f_ref, l_ref) in zip(items[:2], ref):
        p0, feats, ph, pw = net.main_pass(it["inp"])
        net.geo_feats(ph, pw)                                          # 기하 토큰을 실제로 계산한 뒤에도
        with torch.no_grad():
            o = net.vlm(**it["inp"], logits_to_keep=1)
        f2 = net.features(it["inp"], None)[1]
        same &= torch.equal(o.depth_pred[0], p_ref) and all(torch.equal(x, y) for x, y in zip(f2, f_ref)) and torch.equal(o.logits, l_ref)
    with torch.no_grad():
        vqa1 = net.vlm.model(**{k: vq[k] for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}).last_hidden_state
    res["check1_main_pass_identical"] = bool(same)
    res["check5_vqa_hidden_identical"] = bool(torch.equal(vqa0, vqa1))
    assert same and res["check5_vqa_hidden_identical"], "기하 토큰을 붙이자 본 통과 값이 달라짐"
    # 점검 2 · 9: 처음 상태의 헤드
    rel, eq9 = {}, True
    for it in items:
        p0, feats, ph, pw = net.main_pass(it["inp"])
        geo, _ = net.geo_feats(ph, pw)
        with torch.no_grad():
            zp, zc = net.predict(R["prop"], feats, geo, ph, pw), net.predict(R["ctrl"], feats, None, ph, pw)
            zn = G.predict(R["prop"], feats, geo, ph, pw, None, None, False)        # 주입을 끈 같은 헤드
        for k, z in (("prop", zp), ("ctrl", zc)):
            rel[k] = max(rel.get(k, 0.0), float(((z - p0).abs() / p0.clamp_min(1e-6)).max()))
        eq9 &= torch.equal(zp, zc) and torch.equal(zp, zn)
    res["check2_zero_init_rel"] = rel
    res["check9_ctrl_equals_gate0"] = bool(eq9)
    assert max(rel.values()) <= 0.02, ("처음 상태가 원래 예측과 2 % 넘게 다름", rel)
    assert eq9, "대조 팔 = 게이트 0 인 제안 팔 이어야 하는데 다름"
    # 점검 3: 기울기가 흐르는 곳
    lr, ls, _ = losses(net, R, Sh, items[0])
    lr["prop"].backward()
    g = lambda ps: float(sum(q.grad.abs().sum() for q in ps if q.grad is not None))
    res["check3_grad"] = dict(prop_head=g(R["prop"].h.parameters()), prop_gate=g([R["prop"].gate]), prop_proj=g(R["prop"].proj.parameters()), ctrl=g(R["ctrl"].parameters()), scale=g(Sh.parameters()),
                              frozen=float(sum(q.grad is not None for q in list(net.vlm.parameters()) + list(net.geo.parameters()) + list(net.sky.parameters()))))
    c3 = res["check3_grad"]
    assert c3["prop_head"] > 0 and c3["prop_gate"] > 0 and c3["ctrl"] == 0 and c3["scale"] == 0 and c3["frozen"] == 0, ("기울기가 엉뚱한 곳으로 흐름", c3)
    for r in R.values():
        r.zero_grad(set_to_none=True)
    # 점검 4: 사진 한 장 32 걸음 (점검용 헤드를 그대로 씀 — 사본을 두지 않아 메모리를 아낌. 점검 뒤에는 버린다)
    del lr, ls
    torch.cuda.empty_cache()
    R2, S2 = R, Sh
    opt = optimizers(R2, S2)
    hist = defaultdict(list)
    for _ in range(32):
        lr, ls, _ = losses(net, R2, S2, items[0])
        for k in ARMS:
            opt[k].zero_grad(set_to_none=True)
            lr[k].backward()
            opt[k].step()
            hist[k].append(lr[k].item())
        opt["S"].zero_grad(set_to_none=True)
        ls.backward()
        opt["S"].step()
        hist["S"].append(ls.item())
    res["check4_overfit"] = {k: [round(v[0], 5), round(v[-1], 5)] for k, v in hist.items()}
    res["check4_gate_after"] = [round(float(x), 5) for x in R2["prop"].gate.detach()]
    (out / "audit.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))
    bad = [k for k, v in res["check4_overfit"].items() if not v[1] < 0.9 * v[0]]
    assert not bad, ("한 장 32 걸음에 손실이 10 % 넘게 줄지 않음", bad)
    print("[audit] 통과", flush=True)


# ---------------------------------------------------------------- 학습
def one_item(x):
    return x


def save_state(path, R, Sh, opt, step, log, a):
    tmp = Path(str(path) + ".tmp")
    torch.save(dict(R={k: v.state_dict() for k, v in R.items()}, S=Sh.state_dict(), opt={k: v.state_dict() for k, v in opt.items()} if opt else None,
                    step=step, log=log, read=str(a.read), stream=a.stream, hp=HP), tmp)
    tmp.replace(path)


def train(a):
    torch.manual_seed(S1.HP["seed"])
    out, ck = RUNS / a.tag, CKPT / a.tag
    out.mkdir(parents=True, exist_ok=True)
    ck.mkdir(parents=True, exist_ok=True)
    stream = pd.read_parquet(C.V2 / f"stream_{a.stream}.parquet")
    if a.limit:
        stream = stream.iloc[: a.limit]
    net = Net(a.read)
    R, Sh = net.heads()
    opt = optimizers(R, Sh)
    base = {k: [g["lr"] for g in o.param_groups] for k, o in opt.items()}
    total = len(stream) // HP["accum"]
    warm = min(HP["warmup"], max(1, total // 10))
    sched = lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total))))
    step, log = 0, []
    if a.resume and (ck / "last.pt").exists():
        st = torch.load(ck / "last.pt", map_location="cpu", weights_only=False)
        assert st["stream"] == a.stream and st["read"] == str(a.read)
        for k in R:
            R[k].load_state_dict(st["R"][k])
        Sh.load_state_dict(st["S"])
        for k in opt:
            opt[k].load_state_dict(st["opt"][k])
        step, log = st["step"], st["log"]
        print(f"[{a.tag}] 이어 하기: 걸음 {step}/{total}", flush=True)
    dl = torch.utils.data.DataLoader(D.S2Set(stream), batch_size=None, sampler=range(step * HP["accum"], total * HP["accum"]), collate_fn=one_item, prefetch_factor=4, **D.mp(a.workers))
    params = dict(prop=[q for q in R["prop"].parameters()], ctrl=[q for q in R["ctrl"].parameters()], S=list(Sh.parameters()))
    print(f"[{a.tag}] 사진 {len(stream)} 장 · 걸음 {total} · 헤드 매개변수 {sum(q.numel() for q in params['prop']) / 1e6:.1f} M · 배율 헤드 {sum(q.numel() for q in params['S']) / 1e6:.2f} M", flush=True)
    t0, run, n0, n_in = time.time(), defaultdict(list), step, 0
    for it in dl:
        if n_in == 0:
            for k, o in opt.items():
                for g, lr0 in zip(o.param_groups, base[k]):
                    g["lr"] = lr0 * sched(step)
                o.zero_grad(set_to_none=True)
        lr, ls, parts = losses(net, R, Sh, to_dev(it))
        for k in ARMS:
            (lr[k] / HP["accum"]).backward()
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
                       gate=[round(float(x), 4) for x in R["prop"].gate.detach()], **{k2: sum(x for x, _ in v) / max(1.0, sum(c for _, c in v)) for k2, v in run.items()})
            log.append(row)
            run = defaultdict(list)
            print(f"[{a.tag}] {step}/{total} 제안 {row['prop_loss']:.4f} (정답 {row['prop_gt']:.4f} · 채움 {row['prop_fill']:.4f} · 기울기 {row['prop_gm']:.4f}) · 대조 {row['ctrl_loss']:.4f} (정답 {row['ctrl_gt']:.4f}) · "
                  f"배율 헤드 {row['scale']:.4f} · 게이트 {row['gate']} | {row['sec_per_img']:.2f} 초/장 · GPU {row['gpu_gb']:.1f} GB · 남은 {(total - step) * HP['accum'] * row['sec_per_img'] / 60:.0f} 분", flush=True)
            pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
        if step % a.ckpt_every == 0 and step < total:
            save_state(ck / "last.pt", R, Sh, opt, step, log, a)
        if a.val_at and step in a.val_at and step < total:
            df = val_cells(net, R, Sh, a.val_limit, a.workers)
            df.to_csv(out / f"val_step{step}.csv", index=False)
            print(f"[{a.tag}] 걸음 {step} (보고만) " + val_line(df), flush=True)
    save_state(ck / "final.pt", R, Sh, None, step, log, a)
    df = val_cells(net, R, Sh, a.val_limit, a.workers)
    df.to_csv(out / "val_final.csv", index=False)
    print(f"[{a.tag}] 끝 " + val_line(df), flush=True)
    print(f"[{a.tag}] 끝 · {(time.time() - t0) / 60:.1f} 분 · EXIT 0", flush=True)


# ---------------------------------------------------------------- 검증 (학습 분포 700, 8 화소 칸)
class ValSet(torch.utils.data.Dataset):
    """공식 로더와 같은 canonical 사진 (BILINEAR 로 canonical 크기 → 가운데 자르기) → 본 통과 입력 + 저장된 8 화소 칸 지도."""

    def __init__(self, limit=0):
        t = B.index("val")
        self.t = (t.groupby("ds").head(limit) if limit else t).reset_index(drop=True)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, n):
        torch.set_num_threads(1)
        sys.path.insert(0, str(Path.home() / "code/vlm-depth-rmse/third_party/DepthVLM"))
        from utils.datasets import DATASET_CONFIGS, _clamp_canonical_size, _load_and_resize_image
        r = self.t.iloc[n]
        rec = D.recs("val")[int(r.i)]
        cs, crop = _clamp_canonical_size(rec.get("canonical_size"), DATASET_CONFIGS[rec["dataset"]].get("max_canonical_edge"))
        img = _load_and_resize_image(str(D.DATA / rec["image"].lstrip("/")), cs, crop)
        return dict(n=n, inp=D.vlm_inputs(img), **B.load_maps(B.paths("val", int(r.i))[1]))


@torch.no_grad()
def val_cells(net, R, Sh, limit=0, workers=4):
    ds = ValSet(limit)
    rows = []
    for it in torch.utils.data.DataLoader(ds, batch_size=None, collate_fn=one_item, **D.mp(workers)):
        it = to_dev(it)
        p0, feats, ph, pw = net.main_pass(it["inp"])
        geo, sky_logit = net.geo_feats(ph, pw)
        G_, w_ = it["G"][None], it["w"][None]
        ln0 = cell_log(p0, ph, pw)
        t = ds.t.iloc[int(it["n"])]
        row = dict(i=int(t.i), ds=t.ds)
        row["V"], row["mu_V"] = (x.item() for x in B.score(ln0[None], G_, w_))
        row["V_cache"] = B.score(it["V"][None], G_, w_)[0].item()      # 저장해 둔 원래 예측 (배관 확인: V 와 같아야 함)
        m = Sh(feats, ph, pw)
        m0 = F.avg_pool2d(ln0[None, None], 4)[0, 0]
        for arm in ARMS:
            z = net.predict(R[arm], feats, geo if arm == "prop" else None, ph, pw)
            lz = z.clamp_min(1e-4).log()
            r8 = cell_log(z, ph, pw)
            for tag, mm in (("", m), ("_head0", m0)):                  # 배율: 배율 헤드 / 원래 헤드
                s_, mu = B.score((r8 + compose(lz, mm, sky_logit))[None], G_, w_)
                row[arm + tag], row["mu_" + arm + tag] = s_.item(), mu.item()
        row["scale_head_err"] = float((m - m0).abs().mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("i").reset_index(drop=True)


def val_line(df):
    f = lambda c: B.macro(df, c)
    ab = lambda c: float(df.groupby("ds")[c].apply(lambda s: s.abs().mean()).mean())
    v, p, c = f("V"), f("prop"), f("ctrl")
    return (f"검증 {len(df)} 장 SILog (7 종 / 실내 / 주행): 원래 {v['all']:.2f} / {v['indoor']:.2f} / {v['driving']:.2f} · 제안 {p['all']:.2f} / {p['indoor']:.2f} / {p['driving']:.2f} · "
            f"대조 {c['all']:.2f} / {c['indoor']:.2f} / {c['driving']:.2f} | 배율 오차 원래 {ab('mu_V'):.2f} · 제안 {ab('mu_prop'):.2f} (원래 헤드로 {ab('mu_prop_head0'):.2f}) · 대조 {ab('mu_ctrl'):.2f} | "
            f"배율 헤드 − 원래 헤드 칸 차 {df.scale_head_err.mean():.4f} | 저장된 원래 예측과의 SILog 차 {float((df.V - df.V_cache).abs().mean()):.3f}")


def load_final(net, a):
    R, Sh = net.heads()
    st = torch.load(a.ckpt or CKPT / a.tag / "final.pt", map_location="cpu", weights_only=False)
    for k in R:
        R[k].load_state_dict(st["R"][k])
        R[k].eval()
    Sh.load_state_dict(st["S"])
    return R, Sh.eval()


def val(a):
    net = Net(a.read)
    R, Sh = load_final(net, a)
    df = val_cells(net, R, Sh, a.val_limit, a.workers)
    df.to_csv(RUNS / a.tag / "val_final.csv", index=False)
    print(f"[{a.tag}] " + val_line(df), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("audit", "train", "val", "eval6"))
    ap.add_argument("--read", required=True)
    ap.add_argument("--tag", default="s2")
    ap.add_argument("--stream", default="s2")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--val_limit", type=int, default=0)
    ap.add_argument("--val_at", type=int, nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--sets", nargs="*", default=[])
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(ARGV)
    import resource                                                  # 작업자가 넘기는 텐서마다 파일 번호를 하나씩 쓴다 — 한도를 넉넉히
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (65536 if hard == resource.RLIM_INFINITY else min(hard, 65536), hard))
    if a.stage == "eval6":
        import cb2_eval6
        return cb2_eval6.run(a, Net, load_final, compose, cell_log, to_dev, one_item)
    dict(audit=audit, train=train, val=val)[a.stage](a)


if __name__ == "__main__":
    main()
