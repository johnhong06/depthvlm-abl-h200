"""D-41 표 1 빈 칸 비교 모델 — 각자 공식 추론 경로 (README · hubconf) 그대로, f(path, k) → 깊이 (m, 아무 크기 — paper_eval.save 가 정답 크기로 bilinear).
설정은 D-33 (각자 공식 설정): 카메라를 스스로 예측하는 모델 (UniDepth · ZoeDepth) 은 정답 K 를 안 줌, 초점이 필요한 모델 (DA3-metric) 은 정답 fx (†).
모델마다 자기 환경에서 돈다 (paper_eval pv 를 그 환경의 python 으로):
  UniDepth-L   ~/venv/unidepth  (vlm-depth-rmse/ext/UniDepth 의 공식 코드, 읽기만)
  ZoeDepth-NK  ~/venv/zoedepth  (torch.hub isl-org/ZoeDepth, timm 0.6.7)
  DA3-metric-L ~/venv/da3       (ByteDance-Seed/Depth-Anything-3 공식 패키지)
"""
import os
import sys

import numpy as np
import torch
from PIL import Image

VDR_EXT = os.path.expanduser("~/code/vlm-depth-rmse/ext")


def unidepth_l():
    """UniDepth v1 ViT-L — README 'Get Started': UniDepthV1.from_pretrained → infer(uint8 RGB CHW), 카메라 없음 (UniDepthV2 와 같은 D-33 설정). 출력 depth = z (원본 크기).
    환경 ~/venv/unidepth1: v1 의 NystromAttention 이 xformers.components (0.0.28.post3 이 마지막) 를 요구 → triton 커널은 끄고 (XFORMERS_FORCE_DISABLE_TRITON),
    이 xformers 의 CUDA 연산은 torch 2.7 용으로 빌드되지 않아 DINOv2 백본의 XFORMERS_AVAILABLE 만 실행 때 꺼서 PyTorch attention 으로 (저장소 주석: 'new pytorch have good attn efficient')."""
    os.environ.setdefault("XFORMERS_FORCE_DISABLE_TRITON", "1")
    sys.path.insert(0, os.path.join(VDR_EXT, "UniDepth"))
    import unidepth.models.backbones.metadinov2.attention as _att
    import unidepth.models.backbones.metadinov2.block as _blk
    _att.XFORMERS_AVAILABLE = _blk.XFORMERS_AVAILABLE = False
    from unidepth.models import UniDepthV1
    m = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14").to("cuda").eval()
    return lambda path, k: m.infer(torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1))["depth"][0, 0].float().cpu().numpy()


def zoedepth_nk():
    """ZoeDepth ZoeD_NK — 공식 hubconf (torch.hub.load('isl-org/ZoeDepth', 'ZoeD_NK', pretrained=True)) → infer_pil, 카메라 안 받음.
    pad_input=False = 공식 평가 evaluate.py 와 같음 (flip 평균만, 패딩 없음). infer_pil 기본값 (반사 패딩 ≈ 46·53 px) 은 시야를 넓혀 NYU 에서 깊이를 약 10 % 크게 냄
    (NYU δ1 89.7 → 패딩 없음 95.3 · AbsRel 7.7 = 논문 표 1 그대로, D-41 10-20 확인 — scratchpad diag/zoe_nopad*.py).
    config_mode="eval" = 공식 평가 · 학습 설정 (hub 기본 'infer' 는 force_keep_ar=True 로 종횡비를 지켜 KITTI 를 384×1280 으로 넣음 — 학습 · 평가는 384×512 로 늘려 넣음).
    4:3 사진 (NYU · iBims · DIODE) 은 두 설정이 같고, KITTI 100 장 δ1 87.7 → 95.4 · SILog 12.06 → 8.89 (D-44 10-20, scratchpad diag/zoe_kitti_mode*.py)."""
    m = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True, trust_repo=True, config_mode="eval").to("cuda").eval()
    return lambda path, k: np.asarray(m.infer_pil(Image.open(path).convert("RGB"), pad_input=False, output_type="numpy"), dtype=np.float32)


def da3_metric_l():
    """Depth Anything 3 DA3METRIC-LARGE — 공식 API (DepthAnything3.from_pretrained → inference). README: metric depth = focal × 출력 / 300 (초점 = 정답 fx, 처리 해상도 기준)."""
    from depth_anything_3.api import DepthAnything3
    m = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE").to("cuda").eval()

    def f(path, k):
        img = Image.open(path).convert("RGB")
        W, H = img.size
        # 짧은 변 504 (lower_bound_resize) — API 기본 (긴 변 504, upper_bound) 은 KITTI 를 504×154 로 줄임. 짧은 변 504 가 원 논문 표 11 의 KITTI 95.3 · NYU AbsRel 7.0 을 재현
        # (100 장씩: KITTI 90.8 → 95.3, NYU 96.3/7.2 → 96.7/7.0, SUN · DIODE 같거나 조금 좋음 — D-44 10-20, scratchpad diag/da3_*res*.py). 모든 세트에 같게.
        pr = m.inference([img], process_res=504, process_res_method="lower_bound_resize")
        out = np.asarray(pr.depth[0], dtype=np.float32)          # 처리 해상도의 출력
        Hp, Wp = out.shape
        focal = (k[0] * Wp / W + k[1] * Hp / H) / 2.0             # README 'focal = fx · fy 평균' · utils/alignment.apply_metric_scaling 과 같게, 처리 해상도 기준
        return focal * out / 300.0
    return f


FN = {"UniDepth-L": unidepth_l, "ZoeDepth-NK": zoedepth_nk, "DA3-metric-L": da3_metric_l}
