"""네 칸 ablation (사용자 11-18: '기존 구조 / 우리 구조' × '라벨만 / 라벨 + 무라벨 (선생)') 의 학습을 띄우는 감싸개 — cb3_stage1 · cb3_stage2 를 그대로 쓰고, '라벨만' 일 때만 손실에서 선생 몫을 끈다.
  cb4_abl.py s1 [--gt_only] <cb3_stage1 의 인자 …>     1 단계 (기하 토큰)
  cb4_abl.py s2 [--gt_only] <cb3_stage2 의 인자 …>     2 단계 (상대 깊이 헤드 둘 — 토큰 있는 제안 팔 · 토큰 없는 대조 팔)
--gt_only  (사용자: 'gt 만 쓸 때는 전문가 정답을 채우는 것을 안 해야') cb2_stage1.stage1_loss 를 부르기 전에 사진의 gate 를 0 으로 바꾼다 →
           선생 채움 항 (l_fill) 과 채움 칸끼리의 기울기 항이 0 이 되고, 남는 것은 정답 칸 항 (l_gt) · 정답 칸끼리의 기울기 항 · (1 단계만) 하늘 항.
           하늘 목표는 깊이 선생이 아니라 하늘 분할 (Mask2Former) 지도라 그대로 둔다 (네 칸 모두 같음). 2 단계의 '평균을 원래 예측에 묶는 항' 도 그대로 (선생이 아니라 원래 DepthVLM).
자료 읽기 일꾼 (spawn) 이 이 파일을 다시 읽으므로 모든 일은 본 프로세스에서만 한다.
    ~/venv/depthvlm/bin/python -I -B -u cb4_abl.py s1 --gt_only train --arm read --stream s1_nou --tag abl_L_s1 --inline --n_geo 600 --sets val hyper
"""
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    stage = sys.argv.pop(1)
    assert stage in ("s1", "s2"), stage
    gt_only = "--gt_only" in sys.argv
    if gt_only:
        sys.argv.remove("--gt_only")
    if stage == "s1":
        import cb3_stage1 as W
        S1, run = W.S1, W.S1.main
    else:
        import cb3_stage2 as W
        S1, run = W.S2.S1, W.S2.main
    if gt_only:
        import torch
        _loss = S1.stage1_loss

        def gt_only_loss(r, sky_logit, b, hp=S1.HP):
            b = dict(b)
            b["gate"] = torch.zeros_like(b["gate"])
            return _loss(r, sky_logit, b, hp)

        S1.stage1_loss = gt_only_loss
    print(f"[cb4_abl] 단계 {stage} · 라벨만 (선생 채움 끔) {gt_only}", flush=True)
    run()
