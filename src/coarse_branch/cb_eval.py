"""평가 6 세트 채점 (보고만 — RECORD 2 절 '판정 뒤 한 번'). 저장된 가지 (final.pt) 가 낸 굵은 무늬로 원래 DepthVLM 예측의 굵은 무늬만 바꿔,
프로젝트의 논문 규칙 (analysis/paper_eval.py 의 gt_and_mask · score 와 같은 식) 으로 잰다. 프로젝트 파일 · 저장된 예측은 읽기만 한다.

단계 (산출물은 ~/data/dvsilog/coarse_branch/eval/, 있는 파일은 건너뜀):
  thumbs  (CPU · ~/venv/depthvlm)  DepthVLM 평가와 같은 canonical 사진 · 토큰 격자 (geo_mix.vlm_inputs 와 같은 호출) → 가지가 볼 작은 사진 (cb_prep 과 같은 식)
  branch  (GPU · ~/venv/depthvlm)  가지 → 굵은 상대 깊이 · 하늘 logit
  refs    (GPU · ~/venv/unidepth)  기성 전문가 UniDepthV2-S · -L 을 같은 작은 사진으로 (cb_refs 와 같은 식)
  score   (CPU · ~/venv/depthvlm)  화소 단위로 합쳐 채점 → eval/per_image.csv
  report  (CPU · ~/venv/depthvlm)  표 + 사진 짝 bootstrap → results/eval_report.{json,txt}
줄 이름: V 원래 DepthVLM · dav2s / siglip 가지 · refS / refL 기성 전문가 (작은 사진) 의 굵은 무늬 · expL 저장된 큰 전문가 예측 (원래 크기) 의 굵은 무늬.
  '·median' = 합칠 때 빼는 상수를 중앙값으로 (보충 — SILog 는 같고 배율에 걸린 지표만 다름). 기준 줄 (ref · exp) 은 가지 (dav2s, 없으면 siglip) 가 낸 하늘을 같이 쓴다.
사용: HF_HUB_OFFLINE=1 ~/venv/depthvlm/bin/python -I -B cb_eval.py thumbs|branch|score|report [--limit N] [--sets …] [--kinds …]
      HF_HUB_OFFLINE=1 ~/venv/unidepth/bin/python -I -B cb_eval.py refs
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cb_common as C  # noqa: E402

PROJ = HERE.parent
VDR = os.path.expanduser("~/code/vlm-depth-rmse")
SETS = ("nyu_std", "ibims1", "sunrgbd", "kitti", "eth3d", "diode_outdoor")
KINDS = ("dav2s", "siglip")
NAME = {"V": "원래 DepthVLM", "dav2s": "가지: DAv2-S", "siglip": "가지: SigLIP + LoRA", "refS": "기성 작은 전문가 (S) 굵은 무늬",
        "refL": "기성 큰 전문가 (L) 굵은 무늬 · 작은 사진", "expL": "큰 전문가 굵은 무늬 · 원래 크기"}
SETNAME = {"nyu_std": "NYUv2", "ibims1": "iBims-1", "sunrgbd": "SUN RGB-D", "kitti": "KITTI", "eth3d": "ETH3D", "diode_outdoor": "DIODE Outdoor"}


def pe():
    if str(PROJ / "analysis") not in sys.path:
        sys.path.insert(0, str(PROJ / "analysis"))
    import paper_eval as PE
    return PE


def recs(ds, limit=0):
    """[(id, 사진 경로, fx, W, H, 정답 기록)] — paper_eval.score 가 쓰는 것과 같은 사진 · 정답 짝."""
    PE = pe()
    if ds == "kitti":
        out = [(r["id"], r["image"], r["K"][0], r["W"], r["H"], r["gt"]) for r in PE.KE.records()]
    else:
        base = PE.records(ds)
        if ds == "sunrgbd":
            gt = [r["depth"] for r in json.load(open(PE.SUN))]
        elif ds == "eth3d":
            gt = [r["gt"] for r in json.load(open(PE.ETH))]
        elif ds == "nyu_std":
            import std_eval
            gt = [gp for _, _, gp, _ in std_eval.records("nyu", os.path.join(PE.DVFT, "data"), os.path.join(PE.DVFT, "splits"))]
        else:
            sys.path.insert(0, os.path.join(VDR, "eval"))
            import common
            bench = {r["image"]: r for r in common.load_bench(ds)[0]}
            gt = [bench[p.replace(PE.BENCH + "/", "")] for _, p, _, _, _ in base]
        assert len(gt) == len(base), (ds, len(gt), len(base))
        out = [(rid, path, k[0], W, H, g) for (rid, path, k, W, H), g in zip(base, gt)]
    return out[:limit] if limit else out


def load_u8(path):
    return torch.from_numpy(np.asarray(Image.open(path).convert("RGB")).copy()).permute(2, 0, 1)


# ───────────────────────── thumbs: 작은 사진 + 토큰 격자 ─────────────────────────
_PROC = None


class Thumbs(torch.utils.data.Dataset):
    def __init__(self, ds, rs, out):
        self.ds, self.rs, self.out = ds, rs, out

    def __len__(self):
        return len(self.rs)

    def __getitem__(self, j):
        global _PROC
        sys.path.insert(0, os.path.join(VDR, "third_party/DepthVLM"))
        from qwen_vl_utils import process_vision_info
        from utils.datasets import DEFAULT_DEPTH_PROMPT, _load_and_resize_image
        if _PROC is None:
            from huggingface_hub import snapshot_download
            from transformers import AutoProcessor
            _PROC = AutoProcessor.from_pretrained(snapshot_download(C.PLAN["depthvlm"][0], revision=C.PLAN["depthvlm"][1], local_files_only=True))
        rid, path, fx, W, H, _ = self.rs[j]
        img = _load_and_resize_image(path, [round(W * 1000.0 / fx), round(H * 1000.0 / fx)])        # paper_eval.vlm · kitti_eval.canonical 과 같음
        assert img is not None, path
        msg = [{"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
               {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": DEFAULT_DEPTH_PROMPT}]}]   # geo_mix.chat
        inp = _PROC(text=[_PROC.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)], images=[process_vision_info(msg)[0]],
                    padding=True, return_tensors="pt")                                                # geo_mix.vlm_inputs
        _, h, w = (int(v) for v in inp["image_grid_thw"][0])
        ph, pw = h // 2, w // 2
        th, tw = C.thumb_grid(ph, pw)
        tp = self.out / "thumbs" / self.ds / f"{rid}.jpg"
        tmp = tp.with_name(tp.stem + ".tmp.jpg")
        img.convert("RGB").resize((32 * tw, 32 * th), Image.BICUBIC).save(tmp, "JPEG", quality=95, subsampling=0)   # cb_prep 과 같음
        os.replace(tmp, tp)
        return rid, ph, pw, img.size[0], img.size[1]


def thumbs(a):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for ds in a.sets:
        gf = a.out / f"grid_{ds}.json"
        rs = recs(ds, a.limit)
        if gf.exists() and len(json.loads(gf.read_text())) == len(rs):
            print(f"[thumbs] {ds} 있음 ({len(rs)} 장)", flush=True)
            continue
        (a.out / "thumbs" / ds).mkdir(parents=True, exist_ok=True)
        t0, grid, asp = time.time(), {}, []
        for rid, ph, pw, cw, ch in torch.utils.data.DataLoader(Thumbs(ds, rs, a.out), batch_size=None, shuffle=False, num_workers=a.workers):
            grid[rid] = [int(ph), int(pw)]
            asp.append(abs(float(np.log((int(cw) / int(ch)) / (int(pw) / int(ph))))))
        assert len(grid) == len(rs), (ds, len(grid), len(rs))
        gf.write_text(json.dumps(grid))
        sizes = sorted({(v[0], v[1]) for v in grid.values()})
        print(f"[thumbs] {ds} {len(rs)} 장 · {time.time() - t0:.0f} s · 토큰 격자 {len(sizes)} 종 (예 {sizes[:3]}) · 가로세로 비 차이 최대 {100 * max(asp):.1f} %", flush=True)


# ───────────────────────── branch: 가지 통과 ─────────────────────────
@torch.no_grad()
def branch(a):
    import cb_models as M
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    for kind in a.kinds:
        ck = C.CKPT / kind / "final.pt"
        if not ck.exists():
            print(f"[branch] {kind}: 체크포인트 없음 — 건너뜀", flush=True)
            continue
        model = M.Branch(kind).to("cuda")
        model.load_state(torch.load(ck, map_location="cpu", weights_only=False)["state"])
        model.eval()
        for ds in a.sets:
            grid = json.loads((a.out / f"grid_{ds}.json").read_text())
            d = a.out / "branch" / kind / ds
            d.mkdir(parents=True, exist_ok=True)
            t0, n = time.time(), 0
            for rid, (ph, pw) in grid.items():
                f = d / f"{rid}.npz"
                if f.exists():
                    continue
                ky, kx = C.coarse_grid(4 * ph, 4 * pw)
                B, S = model(load_u8(a.out / "thumbs" / ds / f"{rid}.jpg")[None].to("cuda"), ky, kx)
                assert B.shape == (1, ky, kx) and torch.isfinite(B).all() and torch.isfinite(S).all(), (kind, ds, rid)
                tmp = d / f"{rid}.tmp.npz"
                np.savez(tmp, B=B[0].float().cpu().numpy(), S=S[0].float().cpu().numpy())
                os.replace(tmp, f)
                n += 1
            print(f"[branch] {kind} {ds} 새로 {n} / {len(grid)} 장 · {time.time() - t0:.0f} s", flush=True)
        del model
        torch.cuda.empty_cache()


# ───────────────────────── refs: 기성 전문가 (작은 사진) ─────────────────────────
def refs(a):
    sys.path.insert(0, os.path.join(VDR, "ext/UniDepth"))
    from unidepth.models import UniDepthV2
    for name, spec in (("S", dict(src=os.path.expanduser(C.PLAN["ref_small"]))), ("L", dict(src=C.PLAN["ref_large"][0], revision=C.PLAN["ref_large"][1]))):
        m = UniDepthV2.from_pretrained(spec["src"], **({"revision": spec["revision"]} if "revision" in spec else {}))
        m.interpolation_mode = "bilinear"
        m = m.to("cuda").eval()
        for ds in a.sets:
            grid = json.loads((a.out / f"grid_{ds}.json").read_text())
            d = a.out / "refs" / name / ds
            d.mkdir(parents=True, exist_ok=True)
            t0, n = time.time(), 0
            for rid, (ph, pw) in grid.items():
                f = d / f"{rid}.npy"
                if f.exists():
                    continue
                rgb = load_u8(a.out / "thumbs" / ds / f"{rid}.jpg")
                H, W = rgb.shape[-2:]
                m.shape_constraints["pixels_min"] = m.shape_constraints["pixels_max"] = H * W
                with torch.inference_mode():
                    dep = m.infer(rgb)["depth"][0, 0].float()
                assert dep.shape == (H, W) and torch.isfinite(dep).all() and (dep > 0).all(), (name, ds, rid)
                ln = F.adaptive_avg_pool2d(dep.double().log()[None, None], (4 * ph, 4 * pw))[0, 0]
                tmp = d / f"{rid}.tmp.npy"
                np.save(tmp, ln.cpu().numpy().astype(np.float16))
                os.replace(tmp, f)
                n += 1
            print(f"[refs] {name} {ds} 새로 {n} / {len(grid)} 장 · {time.time() - t0:.0f} s", flush=True)
        del m
        torch.cuda.empty_cache()


# ───────────────────────── score: 화소 단위로 합쳐 논문 규칙으로 채점 ─────────────────────────
class Score(torch.utils.data.Dataset):
    def __init__(self, ds, rs, grid, out, kinds):
        self.ds, self.rs, self.grid, self.out, self.kinds = ds, rs, grid, out, kinds

    def __len__(self):
        return len(self.rs)

    def __getitem__(self, j):
        import cv2
        cv2.setNumThreads(1)
        torch.set_num_threads(1)
        PE = pe()
        ds = self.ds
        rid, _, _, _, _, gtrec = self.rs[j]
        g, v = PE.gt_and_mask(ds, rid, gtrec)
        if not v.any():
            return []
        gg = g[v]
        lg = np.log(gg)

        def met(p, k=1.0):                                           # paper_eval.score 와 같은 식 (k = 예측에 곱하는 상수)
            pp = (np.abs(p[v]) + 1e-10) * k
            d = np.log(pp) - lg
            return dict(d1=float((np.maximum(pp / gg, gg / pp) < 1.25).mean()), absrel=float((np.abs(pp - gg) / gg).mean()),
                        rmse=float(np.sqrt(((pp - gg) ** 2).mean())), mu=float(d.mean()), m2=float((d ** 2).mean()))

        def full(p):                                                 # ETH3D: 1/3 저장 → 정답 원해상도로 bilinear (paper_eval.score 와 같음)
            if p.shape != g.shape:
                assert ds == "eth3d", (ds, rid, p.shape, g.shape)
                p = cv2.resize(p.astype(np.float32), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
            return p.astype(np.float64)

        p0 = np.load(PE.pred_path(ds, "DepthVLM", rid)).astype(np.float64)
        rows, info = {"V": met(full(p0))}, {}
        V = torch.from_numpy(np.log(np.abs(p0) + 1e-10)).float()[None]
        ph, pw = self.grid[rid]
        ky, kx = C.coarse_grid(4 * ph, 4 * pw)

        def add(name, B, sky):
            Fm, _ = C.merge(V, B, sky)
            Fd, _ = C.merge(V, B, sky, center="median")
            p = full(torch.exp(Fm[0].double()).numpy())
            rows[name] = met(p)
            rows[name + "·median"] = met(p, float(torch.exp((Fd - Fm).double().mean())))   # 상수만 다름 → 같은 지도에 상수 배

        ref = None
        for kind in self.kinds:
            f = self.out / "branch" / kind / ds / f"{rid}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            S = torch.from_numpy(z["S"])[None]
            sky = (C.up(torch.sigmoid(S), V.shape[-2:]) > 0.5).float()   # cb_train.evaluate 와 같은 하늘
            add(kind, torch.from_numpy(z["B"])[None], sky)
            info["sky_" + kind] = float(sky.mean())
            if ref is None:
                ref = (sky, S)
        if ref is not None:
            sky, S = ref
            for name in ("S", "L"):
                f = self.out / "refs" / name / ds / f"{rid}.npy"
                if f.exists():
                    R = torch.from_numpy(np.load(f).astype(np.float32))[None]
                    sky_s = (C.up(torch.sigmoid(S), R.shape[-2:]) > 0.5).float()
                    add("ref" + name, C.coarse_of(R, sky_s, ky, kx), sky)
            ef = PE.pred_path(ds, "UniDepthV2-L", rid)
            if os.path.exists(ef):
                E = np.load(ef).astype(np.float64)
                assert E.shape == p0.shape, (ds, rid, E.shape, p0.shape)
                add("expL", C.coarse_of(torch.from_numpy(np.log(np.abs(E) + 1e-10)).float()[None], sky, ky, kx), sky)
        return [dict(set=ds, image=rid, model=m, n=int(v.sum()), **r, **info) for m, r in rows.items()]


def score(a):
    import pandas as pd
    rows = []
    for ds in a.sets:
        rs = recs(ds, a.limit)
        grid = json.loads((a.out / f"grid_{ds}.json").read_text())
        t0, empty = time.time(), 0
        for out in torch.utils.data.DataLoader(Score(ds, rs, grid, a.out, a.kinds), batch_size=None, shuffle=False, num_workers=a.workers):
            empty += not out
            rows += out
        print(f"[score] {ds} {len(rs)} 장 · {time.time() - t0:.0f} s · 채점 화소 없는 사진 {empty}", flush=True)
    t = pd.DataFrame(rows)
    for c in t.columns:
        if t[c].dtype == object and c not in ("set", "image", "model"):
            t[c] = t[c].astype(float)
    t.to_csv(a.out / "per_image.csv", index=False)
    print(f"[score] 저장 {a.out / 'per_image.csv'} ({len(t)} 줄)", flush=True)


# ───────────────────────── report ─────────────────────────
def boot(d, n, seed=0):
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1) if len(d) * n <= 4e7 else np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def report(a):
    import pandas as pd
    t = pd.read_csv(a.out / "per_image.csv")
    t["silog"] = 100 * np.sqrt(np.maximum(t.m2 - t.mu ** 2, 0))
    t["scale"] = 100 * t.mu.abs()
    nb = int(C.PLAN["boot"])
    out, L = {}, ["평가 6 세트 (보고만) — 논문 규칙 채점 · 사진별 평균 · SILog · 배율 오차는 낮을수록, δ1 은 높을수록 좋음",
                  "'(중앙값)' 줄 = 합칠 때 빼는 상수만 중앙값으로 바꾼 것 (SILog 는 같음). 차이 구간 = 사진 짝 bootstrap 95 %.", ""]
    f3 = lambda d: f"{d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}]"
    for ds in [s for s in SETS if s in set(t.set)]:
        y = t[t.set == ds]
        w = y.pivot(index="image", columns="model", values="silog")
        order = [m for b in ("V", "dav2s", "siglip", "refS", "refL", "expL") for m in (b, b + "·median") if m in w.columns]
        L.append(f"[{SETNAME[ds]}] {len(w)} 장 — 열: δ1 · AbsRel · RMSE · SILog · 배율 오차 · (SILog − DepthVLM)")
        out[ds] = {}
        for m in order:
            z = y[y.model == m]
            r = dict(n=len(z), d1=float(z.d1.mean()), absrel=float(z.absrel.mean()), rmse=float(z.rmse.mean()), silog=float(z.silog.mean()), scale=float(z.scale.mean()))
            base = m.split("·")[0]
            label = NAME[base] + (" (중앙값)" if m.endswith("·median") else "")
            diff = ""
            if m != "V" and not m.endswith("·median"):
                r["silog_minus_V"] = boot((w[m] - w["V"]).to_numpy(), nb)
                diff = f3(r["silog_minus_V"])
            for kind in KINDS:
                if kind in y.model.values and f"sky_{kind}" in z and m == kind:
                    r["sky_mean"] = float(z[f"sky_{kind}"].mean())
            out[ds][m] = r
            L.append(f"  {label:44s} {r['d1']:6.3f} {r['absrel']:7.3f} {r['rmse']:7.3f} {r['silog']:6.2f} {r['scale']:6.2f}   {diff}")
        cmp_ = {}
        for x, yv in (("dav2s", "refS"), ("siglip", "refS"), ("dav2s", "siglip"), ("dav2s", "refL"), ("siglip", "refL")):
            if x in w.columns and yv in w.columns:
                cmp_[f"{x}_minus_{yv}"] = boot((w[x] - w[yv]).to_numpy(), nb)
                L.append(f"    비교 {x} − {yv}: {f3(cmp_[f'{x}_minus_{yv}'])}")
        out[ds]["compare"] = cmp_
        L.append("")
    res = HERE / "results"
    res.mkdir(exist_ok=True)
    tag = "" if a.out.name == "eval" else "_" + a.out.name
    (res / f"eval_report{tag}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    (res / f"eval_report{tag}.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("thumbs", "branch", "refs", "score", "report"))
    ap.add_argument("--limit", type=int, default=0, help="세트마다 앞 N 장만 (스모크 — 산출물은 eval_smoke/ 로)")
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--kinds", nargs="*", default=list(KINDS))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    a.out = C.DATA / ("eval_smoke" if a.limit else "eval")
    a.out.mkdir(parents=True, exist_ok=True)
    {"thumbs": thumbs, "branch": branch, "refs": refs, "score": score, "report": report}[a.stage](a)
