"""1 단계 학습 — cb2_stage1 을 그대로 쓰되 (같은 손실 · 같은 조리법 · 같은 인자) 두 가지만 바꿀 수 있게 감싼 것.
  --inline    읽기 전용 토큰 팔을 메모 없는 계산 (cb3_tokens.ReadArmInline) 으로
  --n_geo N   기하 토큰 수 (기본 150 — cb2_data.N_GEO). 0 이면 병합 칸 (32 화소) 마다 토큰 하나 — 토큰 격자 = 헤드가 받는 특징 격자 (사진 크기에 따라 수가 달라짐)
  --mem_gb X  이 프로세스가 쓸 GPU 메모리 한도 (다른 학습과 같이 돌 때 — 넘으면 이 프로세스만 멈춘다)
  --all_lists 무라벨 사진을 목록 a–e 모두에서 찾는다 (cb3_data.StreamSet3 — 마지막 학습 순서 f<번> 용)
나머지 인자는 cb2_stage1 과 같다. 저장 형식도 같다 (cb2_models.ReadArm 이 그대로 읽는다 — 단 토큰 수가 다르면 쓰는 쪽도 같은 --n_geo 여야 한다).

    ~/venv/depthvlm/bin/python -I -B -u cb3_stage1.py train --arm read --stream s1 --limit 48000 --tag p_tok600 --inline --n_geo 600 --sets val hyper
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
_argv = sys.argv[1:]
INLINE = "--inline" in _argv
if INLINE:
    _argv.remove("--inline")
ALL = "--all_lists" in _argv
if ALL:
    _argv.remove("--all_lists")
N_GEO = None
if "--n_geo" in _argv:
    _i = _argv.index("--n_geo")
    N_GEO = int(_argv[_i + 1])
    del _argv[_i:_i + 2]
MEM_GB = None
if "--mem_gb" in _argv:
    _i = _argv.index("--mem_gb")
    MEM_GB = float(_argv[_i + 1])
    del _argv[_i:_i + 2]
sys.argv = sys.argv[:1] + _argv
import cb2_data as D  # noqa: E402
import cb2_stage1 as S1  # noqa: E402

if N_GEO == 0:
    D.geo_grid = lambda h, w: (int(h), int(w))                        # 병합 칸마다 토큰 하나 (부르는 곳은 모두 D.geo_grid 로 부른다 — 학습 · 검증 모두 본 프로세스)
elif N_GEO is not None:
    D.N_GEO = N_GEO                                                   # geo_grid 가 읽는 값 (학습 · 검증 모두 본 프로세스에서 부름)
if MEM_GB is not None:
    import torch
    torch.cuda.set_per_process_memory_fraction(min(1.0, MEM_GB * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory))
if ALL:
    import cb3_data as F3
    D.StreamSet = F3.StreamSet3                                       # S1.train 이 D.StreamSet 으로 부른다
if INLINE:
    _build = S1.build

    def build(arm):
        if arm != "read":
            return _build(arm)
        import cb3_tokens as T3
        return T3.ReadArmInline().to(S1.DEV)

    S1.build = build

if __name__ == "__main__":
    print(f"[cb3_stage1] 메모 없는 계산 {INLINE} · 목록 a–e {ALL} · 기하 토큰 수 {'병합 칸마다 하나' if N_GEO == 0 else D.N_GEO} · GPU 한도 {MEM_GB} GB · 예: 토큰 격자 (28 × 38 칸 사진) {D.geo_grid(28, 38)}", flush=True)
    S1.main()
