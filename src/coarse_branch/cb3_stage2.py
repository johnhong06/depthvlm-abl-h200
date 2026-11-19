"""2 단계 (학습 · 검증 · 평가 6 세트) — cb2_stage2 를 그대로 쓰되 (같은 손실 · 조리법 · 인자 · 저장 형식) 마지막 학습에 필요한 것만 바꿀 수 있게 감싼 것.
  --inline        기하 토큰을 메모 없는 계산 (cb3_tokens.GeoTokensInline) 으로 — 본 통과 안에서 한 층씩 같이 계산 (값은 같고 메모리 · 시간만 준다)
  --n_geo N       기하 토큰 수 — 1 단계 (cb3_stage1 --n_geo) 와 같아야 한다. 0 = 병합 칸 (32 화소) 마다 하나
  --detail 방식   제안 팔에 '2 × 2 로 합치기 전 특징' 도 같이 넣는다: raw (그대로) | conv (작은 합성곱으로 다듬어) | guide (토큰을 같이 보며 다듬어) — cb2_stage2d 의 헤드 그대로.
                  주지 않으면 토큰만 (cb2_stage2 와 같음). 대조 팔 (주입 없음) · 배율 헤드는 그대로.
  --all_lists     무라벨 사진을 목록 a–e 모두에서 찾는다 (cb3_data.S2Set3 — 마지막 2 단계 학습 순서 g<번> 용)
  --scale 경로    val · eval6 에서 배율 헤드를 따로 학습한 것 (cb3_scale 의 final.pt) 으로 바꿔 끼운다 — 결과가 덮이지 않게 --ckpt <2 단계 체크포인트> --tag <새 이름> 과 같이 쓴다
나머지 인자 (train | val | eval6 · --read · --tag · --stream · --val_at · --resume …) 는 cb2_stage2 와 같다. audit 은 cb2_stage2 로 (메모 방식 전용).

    ~/venv/depthvlm/bin/python -I -B -u cb3_stage2.py train --read <1 단계 final.pt> --tag f2 --inline --n_geo 600 --detail conv --val_at 4000 8000 --resume
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
_argv = sys.argv[1:]


def _flag(name):
    if name in _argv:
        _argv.remove(name)
        return True
    return False


def _opt(name, cast=str):
    if name not in _argv:
        return None
    i = _argv.index(name)
    v = cast(_argv[i + 1])
    del _argv[i:i + 2]
    return v


INLINE, N_GEO, DETAIL, SCALE, ALL = _flag("--inline"), _opt("--n_geo", int), _opt("--detail"), _opt("--scale"), _flag("--all_lists")
assert DETAIL in (None, "raw", "conv", "guide"), DETAIL
sys.argv = sys.argv[:1] + _argv
import cb2_data as D  # noqa: E402
import cb2_stage2 as S2  # noqa: E402

torch, nn, F = S2.torch, S2.nn, S2.F
if N_GEO == 0:
    D.geo_grid = lambda h, w: (int(h), int(w))                        # 병합 칸마다 토큰 하나 (부르는 곳은 모두 D.geo_grid 로 부른다)
elif N_GEO is not None:
    D.N_GEO = N_GEO
if ALL:
    import cb3_data as F3
    D.S2Set = F3.S2Set3                                               # S2.train · audit 이 D.S2Set 으로 부른다
if DETAIL:
    import cb2_stage2d as S2D
    Detail, _heads4 = S2D.build(S2, nn, F, torch)
    HeadD = next(c.cell_contents for c in _heads4.__closure__ if isinstance(c.cell_contents, type) and c.cell_contents.__name__ == "HeadD")

_attach, _main_pass, _geo_feats, _optimizers, _load_final = S2.Net.attach, S2.Net.main_pass, S2.Net.geo_feats, S2.optimizers, S2.load_final


def attach(self):
    """기하 토큰 (1 단계에서 학습한 LoRA · 하늘 탐침) 을 본 통과의 ViT 에 붙인다 — 메모 없는 판이면 본 통과가 지나갈 때 같이 계산된다."""
    if INLINE:
        import cb3_tokens as T3
        st = torch.load(self.read_ckpt, map_location="cpu", weights_only=False)["state"]
        self.geo = T3.GeoTokensInline(self.vlm.model.visual).to(S2.DEV)
        self.geo.load_state(st["geo"])
        d = self.vlm.model.visual.config.hidden_size
        self.sky = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1)).to(S2.DEV)
        self.sky.load_state_dict(st["sky"])
        for q in list(self.geo.parameters()) + list(self.sky.parameters()):
            q.requires_grad_(False)
    else:
        _attach(self)
    if DETAIL:
        self.det = Detail(self.vlm.model.visual)                       # 합치기 전 패치 값 (5 · 11 · 17 번째 블록 뒤) 을 읽는 훅 — 읽기만


@torch.no_grad()
def main_pass(self, inp):
    if INLINE and self.geo is not None:
        _, Hp, Wp = (int(v) for v in inp["image_grid_thw"][0])
        self.geo.arm(Hp, Wp, *D.geo_grid(Hp // 2, Wp // 2))            # 이번 본 통과에서 토큰을 같이 계산 (no_grad — 2 단계에서 토큰은 고정)
    return _main_pass(self, inp)


@torch.no_grad()
def geo_feats(self, ph, pw):
    """cb2_stage2.Net.geo_feats 와 같은 출력 — 토큰만 본 통과에서 이미 계산된 것을 꺼낸다."""
    if not INLINE:
        return _geo_feats(self, ph, pw)
    gh, gw = D.geo_grid(ph, pw)
    Gt = self.geo.take()
    assert Gt[0].shape[0] == gh * gw, (tuple(Gt[0].shape), gh, gw)
    up = lambda t: F.interpolate(t.T.reshape(1, -1, gh, gw), size=(ph, pw), mode="bilinear", align_corners=False)[0]
    return [up(g).flatten(1).T.contiguous() for g in Gt], up(self.sky(Gt[-1]))[0]


def heads(self):
    G = self.G
    prop = G.Head(self.base, True, hr=True) if DETAIL == "raw" else HeadD(self.base, DETAIL == "guide")
    return dict(prop=prop.to(S2.DEV), ctrl=G.Head(self.base, False).to(S2.DEV)), S2.ScaleHead(self.base.norm.normalized_shape[0]).to(S2.DEV)


def predict(self, head, feats, geo, ph, pw):
    return self.G.predict(head, feats, geo, ph, pw, ghr=None if geo is None else self.det.maps(ph, pw))   # 대조 팔 (geo 없음) 에는 넣지 않는다


def optimizers(R, Sh, hp=S2.HP):
    """제안 팔만 cb2_stage2d 의 묶음 (헤드 2e-5 · 새 층 1e-4 · 게이트 1e-3) 으로 — 합치기 전 특징 쪽 새 층이 들어가게. 대조 팔 · 배율 헤드는 그대로."""
    o = _optimizers(R, Sh, hp)
    o["prop"] = S2D.optimizers({"prop": R["prop"]}, hp, torch)["prop"]
    return o


def load_final(net, a):
    R, Sh = _load_final(net, a)
    st = torch.load(SCALE, map_location="cpu", weights_only=False)
    Sh.load_state_dict(st["S"])
    print(f"[cb3_stage2] 배율 헤드를 바꿔 끼움: {SCALE} (학습 사진 {st.get('n')} 장 · {st.get('info')})", flush=True)
    return R, Sh.eval()


S2.Net.attach, S2.Net.main_pass, S2.Net.geo_feats = attach, main_pass, geo_feats
if DETAIL:
    S2.Net.heads, S2.Net.predict, S2.optimizers = heads, predict, optimizers
if SCALE:
    S2.load_final = load_final

if __name__ == "__main__":
    assert not (S2.ARGV and S2.ARGV[0] == "audit" and (INLINE or DETAIL)), "audit 은 cb2_stage2 로"
    print(f"[cb3_stage2] 메모 없는 계산 {INLINE} · 기하 토큰 수 {'병합 칸마다 하나' if N_GEO == 0 else D.N_GEO} · 합치기 전 특징 {DETAIL or '없음 (토큰만)'} · 배율 헤드 바꿔 끼움 {SCALE or '없음'} · 목록 a–e {ALL}"
          f" · 예: 토큰 격자 (28 × 38 칸 사진) {D.geo_grid(28, 38)}", flush=True)
    S2.main()
