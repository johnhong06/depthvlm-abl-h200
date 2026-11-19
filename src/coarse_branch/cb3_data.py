"""마지막 1 단계 학습 (처음부터 · 고른 사진 전부를 여러 번) 의 학습 순서와 사진 읽기 — 무라벨 목록 a–e 를 모두 다룬다. 기존 cb2_data 는 안 고침.

나눔 (11-13 사용자 결정 — 논문에 한 줄로 쓸 수 있게): 모든 출처에서 약 2/15 를 2 단계용으로 떼어 두고 나머지를 1 단계가 쓴다. 두 단계 모두 자기 몫을 여러 번 (세 번) 본다.
  정답 사진  장면 단위로 2/15 (cb2_data.holdout_table 그대로 — 1 단계 108,366 · 2 단계 16,634)
  무라벨     목록마다 학습 순서 (한 번 섞어 둔 pos) 의 앞 2/15 가 2 단계용 — 목록 a 는 지금까지와 같이 앞 34,000 장 (13.6 %), b–e 는 round(2/15 × 장 수)
학습 순서 'f<번>' (1 단계, 예: f3 = 세 번) · 'g<번>' (2 단계, 예: g3 = 세 번):
  1 단계 f  정답 = 떼어 두지 않은 장면 전부 · 무라벨 = 각 목록의 1 단계 몫 중 선생 · 하늘 지도가 있고 거르기를 통과한 사진
  2 단계 g  정답 = 떼어 둔 장면 전부 · 무라벨 = 각 목록의 2 단계 몫 중 거르기를 통과한 사진 (8 화소 칸 선생 지도 m8 이 있어야 한다 — b–e 는 cb3_teacher_m8.py 로 만든다)
  한 번 볼 때마다 무라벨을 새로 섞고, 정답 사진을 그 사이에 고르게 끼운다. 증강 씨앗 (자리 번호 k) 은 번마다 다름 (f 는 2,000,000 부터 · g 는 5,000,000 부터 — 앞선 실행들과 안 겹침).
거르기 (사용자 11-12 · 13, 선생 지도의 사진별 통계와 사진 밝기만 씀 — 정답 · 평가 세트 안 씀):
  모든 목록   깊이 폭 (ln_p95 − ln_p05) < MIN_RANGE 인 사진은 뺌 (화면 전체가 한 평면이거나 전부 아주 먼 풍경 — 배울 모양이 거의 없음)
  목록 d     위쪽 1/3 의 밝기가 MIN_TOP_D 보다 어두운 사진 (밤) 은 뺌 — night=True 로 부르면 안 뺌

    ~/venv/main/bin/python -I -B cb3_data.py report [목록…]        목록별 남는 장 수
    ~/venv/main/bin/python -I -B cb3_data.py build f3 [목록…]      → v2data/stream_f3.parquet (목록을 안 주면 지도가 다 된 목록 전부)
    ~/venv/main/bin/python -I -B cb3_data.py build g3 [목록…]      → v2data/stream_g3.parquet (2 단계)
    ~/venv/main/bin/python -I -B cb3_data.py need8 [목록…]         2 단계 몫 가운데 m8 지도가 아직 없는 사진 수
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import cb2_common as C  # noqa: E402
import cb2_data as D  # noqa: E402

K0_FINAL, MIN_RANGE, MIN_TOP_D = 2_000_000, 0.10, 70.0
K0_FINAL2, HOLD2 = 5_000_000, 2 / 15
LISTS = ("a", "b", "c", "d", "e", "f", "g")                          # f · g = 11-18 에 더한 목록 (Places365 train · COCO)
FILES = {"a": "unlabeled_places365", "b": "unlabeled_places365_b", "c": "unlabeled_gldv2", "d": "unlabeled_bdd100k", "e": "unlabeled_sa1b", "f": "unlabeled_places365_train", "g": "unlabeled_coco"}
POS0 = {"a": 0, "b": C.N_A, "c": C.N_C0, "d": C.N_D0, "e": C.N_E0, "f": C.N_F0, "g": C.N_G0}
STREAMS = {"f": (1, K0_FINAL), "g": (2, K0_FINAL2), "h": (1, 10_000_000), "i": (2, 20_000_000)}   # 학습 순서 이름의 첫 글자 → (단계, 증강 씨앗 자리의 시작). h · i = 11-18 의 새 학습 (목록 a–g) — f · g 의 파일은 그대로 둔다


def n_stage2(which):
    """목록에서 2 단계용으로 떼어 두는 장 수 (학습 순서 pos 의 앞에서부터). a 는 지금까지 쓰던 값 그대로."""
    if which == "a":
        return D.N_U2
    return round(HOLD2 * len(pd.read_parquet(C.V2 / f"{FILES[which]}.parquet", columns=["path"])))


def parts_done(which):
    """목록의 선생 · 하늘 지도가 다 됐는가."""
    sfx = "" if which == "a" else f"_{which}"
    f = FILES[which]
    if not (C.V2 / f"{f}.parquet").exists():
        return False
    n = len(pd.read_parquet(C.V2 / f"{f}.parquet", columns=["path"]))
    cnt = lambda d, pat: sum(len(pd.read_parquet(p, columns=["pos"])) for p in sorted((C.V2 / d).glob(pat)))
    return cnt(f"teacher/parts{sfx}", "t_*.parquet") == n and cnt(f"sky/parts{sfx}", "s_*.parquet") == n


def table(which, night=False):
    """목록 하나: 지도가 있는 사진 + 거르기 표시. 열: pos · path (전체 경로) · ph · pw · range · bad ('' | flat | night) · s2 (2 단계 몫) · why · keep (1 단계가 쓰는가) · keep2 (2 단계가 쓰는가)."""
    sfx = "" if which == "a" else f"_{which}"
    t = pd.concat([pd.read_parquet(p) for p in sorted((C.V2 / f"teacher/parts{sfx}").glob("t_*.parquet"))], ignore_index=True)
    s = pd.concat([pd.read_parquet(p) for p in sorted((C.V2 / f"sky/parts{sfx}").glob("s_*.parquet"))], ignore_index=True)
    t = t.merge(s[["pos", "sky_frac"]], on="pos", validate="one_to_one")
    t = t[t.ok].reset_index(drop=True)
    for c in ("cw", "ch", "ph", "pw"):
        t[c] = t[c].astype(int)
    t["range"] = t.ln_p95 - t.ln_p05
    t["bad"] = ""
    t.loc[t["range"] < MIN_RANGE, "bad"] = "flat"
    if which == "d" and not night:
        b = pd.read_parquet(C.V2 / "bright_d.parquet")
        t = t.merge(b[["pos", "top"]], on="pos", validate="one_to_one")
        t.loc[(t.top < MIN_TOP_D) & (t.bad == ""), "bad"] = "night"
    t["s2"] = t.pos < POS0[which] + n_stage2(which)                   # 2 단계용으로 떼어 둔 것 (학습 순서 앞 2/15)
    t["why"] = np.where(t.s2, "stage2", t.bad)                        # 1 단계에서 빠지는 까닭 (예전 열 이름 그대로)
    t["keep"], t["keep2"] = t.why == "", t.s2 & (t.bad == "")
    t["path"] = [str(C.root(which) / p) for p in t.path]
    t["list"] = which
    return t


def rows_final(passes, lists, night=False):
    return _rows(passes, lists, night, stage=1)


def rows_final2(passes, lists, night=False):
    """2 단계: 떼어 둔 장면의 정답 사진 + 각 목록의 2 단계 몫."""
    return _rows(passes, lists, night, stage=2)


def _rows(passes, lists, night, stage):
    H = D.holdout_table()
    L = H[H.s2 == (stage == 2)].reset_index(drop=True).rename(columns={"i": "src"})
    U = pd.concat([table(w, night) for w in lists], ignore_index=True)
    U = U[U.keep if stage == 1 else U.keep2].reset_index(drop=True)
    if stage == 2:                                                    # 2 단계는 8 화소 칸 선생 지도가 있는 사진만 (다시 추론했을 때 격자가 달라진 드문 사진은 빠진다)
        has8 = np.array([C.map_path(C.V2 / "teacher/m8", p).exists() for p in U.pos])
        if not has8.all():
            print(f"[2 단계] m8 지도가 없어 빠지는 무라벨 사진 {int((~has8).sum())} 장 ({U[~has8]['list'].value_counts().to_dict()})")
        U = U[has8].reset_index(drop=True)
    assert not U.pos.duplicated().any()
    rows = []
    for p in range(passes):
        u = U.iloc[np.random.default_rng([D.SEED, 31 if stage == 1 else 32, p]).permutation(len(U))]
        key = np.concatenate([(np.arange(len(L)) + 0.5) / len(L), (np.arange(len(u)) + 0.5) / len(u)])
        one = [("L", int(i), int(ph), int(pw)) for i, ph, pw in zip(L.src, L.ph, L.pw)] + [("U", int(i), int(ph), int(pw)) for i, ph, pw in zip(u.pos, u.ph, u.pw)]
        rows += [one[j] for j in np.argsort(key, kind="stable")]
    return rows, len(L), U


class StreamSet3(D.StreamSet):
    """cb2_data.StreamSet 과 같되 무라벨 사진의 경로를 목록 a–e 모두에서 찾는다 (전체 경로 — C.PLACES / '전체 경로' 는 그 전체 경로가 된다)."""

    def _unl(self):
        if self.U is None:
            self.U = {}
            for w in LISTS:
                f = FILES[w]
                if (C.V2 / f"{f}.parquet").exists():
                    u = C.unlabeled_list(w)
                    self.U.update(zip(u.pos.tolist(), [str(C.root(w) / p) for p in u.path]))
        return self.U


class S2Set3(D.S2Set):
    """cb2_data.S2Set 과 같되 무라벨 사진의 경로를 목록 a–e 모두에서 찾는다 (전체 경로 — C.PLACES / '전체 경로' 는 그 전체 경로가 된다)."""

    def __getitem__(self, n):
        if self.U is None:
            self.U = {}
            for w in LISTS:
                if (C.V2 / f"{FILES[w]}.parquet").exists():
                    u = C.unlabeled_list(w)
                    u = u[u.pos < POS0[w] + n_stage2(w)]                  # 2 단계 몫만 (표를 작게)
                    self.U.update(zip(u.pos.tolist(), [str(C.root(w) / p) for p in u.path]))
        return super().__getitem__(n)


def main():
    mode = sys.argv[1]
    if mode == "report":
        for w in (sys.argv[2:] or [x for x in LISTS if parts_done(x)]):
            t = table(w)
            print(f"[{w}] 지도 있는 사진 {len(t)} · 1 단계 {int(t.keep.sum())} · 2 단계 {int(t.keep2.sum())} · 거른 것 {t[t.bad != ''].bad.value_counts().to_dict()} · 화소 중앙 {float((t.cw * t.ch).median()) / 1e6:.2f} MP · 토큰 칸 중앙 {int((t.ph * t.pw).median())}")
        return
    if mode == "need8":                                               # 2 단계 몫 가운데 m8 지도가 없는 사진 → v2data/need8.parquet (cb3_teacher_m8.py 가 읽는다)
        need = []
        for w in (sys.argv[2:] or [x for x in LISTS if parts_done(x)]):
            t = table(w)
            t = t[t.keep2]
            miss = t[[not C.map_path(C.V2 / "teacher/m8", p).exists() for p in t.pos]]
            need.append(miss[["pos", "list", "path", "W", "H", "ph", "pw"]])
            print(f"[{w}] 2 단계 몫 {len(t)} 장 · m8 지도 없는 것 {len(miss)}")
        need = pd.concat(need, ignore_index=True)
        need.to_parquet(C.V2 / "need8.parquet", index=False)
        print(f"m8 를 만들 사진 {len(need)} 장 → {C.V2 / 'need8.parquet'}")
        return
    name = sys.argv[2]
    (stage, k0), passes = STREAMS[name[0]], int(name[1:])
    assert not (C.V2 / f"stream_{name}.parquet").exists(), f"stream_{name}.parquet 이 이미 있음 — 덮어쓰지 않는다 (이미 쓴 학습 순서)"
    lists = sys.argv[3:] or [x for x in LISTS if parts_done(x)]
    rows, nL, U = _rows(passes, lists, False, stage)
    t = D._finish(rows, k0=k0)
    tmp = C.V2 / f"stream_{name}.tmp.parquet"
    t.to_parquet(tmp, index=False)
    tmp.replace(C.V2 / f"stream_{name}.parquet")
    print(f"[{name}] 목록 {list(lists)} · 한 번에 정답 {nL} + 무라벨 {len(U)} ({U['list'].value_counts().to_dict()}) · {passes} 번 = {len(t)} 장 → {C.V2 / f'stream_{name}.parquet'}")


if __name__ == "__main__":
    main()
