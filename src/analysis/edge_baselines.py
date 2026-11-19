"""D-31 준비 (사용자 10-17 "기존 엣지기법을 우리한테도 똑같이 적용한 ablation은 필수", "무작위로 코드 작성하지말고 논문에서 제공하는 코드를 확인해가며 정당성을 파악할 것"):
기존 경계·국소 손실 셋을 공식 코드 그대로 불러 쓰는 얇은 감싸개. 손실 수식은 여기서 다시 쓰지 않는다 — 공식 파일을 경로로 읽어 그 함수·클래스를 부르고,
감싸개는 공식 학습 코드가 넘기는 것과 같은 입력 (공간·모양·mask·정규화) 만 만든다. 우리 학습에서는 예측을 정답 크기로 bilinear (align_corners=True, official_loss 와 같음) 한 뒤 부른다.

출처 (2026-10-17 확인 — 손실 파일과 설정은 GitHub main 과 바이트 단위로 같음):
- UniDepthV2 EG-SSI: vlm-depth-rmse/ext/UniDepth (lpiccinelli-eth/UniDepth, 로컬 사본 @8d8cfe4) unidepth/ops/losses/local_ssi.py EdgeGuidedLocalSSI.
  설정 = configs/config_v2_vitl14.json training.losses.ssi (weight 1.0 · output_fn sqrt · input_fn log1i · use_global true · min_samples 6) → EdgeGuidedLocalSSI.build.
  호출 = unidepth/models/unidepthv2/unidepthv2.py compute_losses: (예측 깊이, target=정답 깊이, mask=정답 mask, image=ImageNet 정규화 사진 (data.normalization imagenet), validity_mask).
  논문 (§III-D) 과 다른 곳은 공식 코드를 따름: 조각 중심 = RGB Sobel 크기를 1/14 로 줄인 격자에서 0.90 분위 넘는 칸 전부 (논문 '상위 5 %'), 정규화 = 안정 점 (평균 ± 1.96 σ) 의
  평균·표준편차 (논문 중앙값·MAD — 코드 주석 'median/MAD gradients are really weird'), 조각 오차에 sqrt, 화면 전체 항 (use_global) 을 조각 항과 평균.
  패치 추출 CUDA 확장은 빌드하지 않음 → 공식 코드가 스스로 TorchScript extract_patches 로 내려감 (같은 값, 느림).
- Metric3D v2 RPNL: vlm-depth-rmse/ext/Metric3D (YvanYin/Metric3D) training/mono/model/losses/HDSNL_random.py HDSNRandomLoss (논문 식 8 의 무작위 조각 정규화 = 이 클래스).
  설정 = training/mono/configs/RAFTDecoder/vit.raft5.large.py decoder_losses 의 HDSNRandomLoss(loss_weight 0.5, random_num 20, batch_limit 4) — 가중치만 우리 규칙으로 밖에서 곱하므로 1.
  호출 = training/mono/model/__base_model__.py branch_loss: (prediction=예측 깊이, target=정답 깊이, mask=정답 > 1e-8, dataset, sem_mask 없음).
  논문과 다른 곳은 공식을 따름: 조각 20 개 (논문 32), 화면 전체 항 포함, 조각·전체를 픽셀 수로 가중 평균, 중앙값은 예측을 detach 해서.
  같은 설정의 HDNRandomLoss (깊이 범위 무작위 정규화) 는 논문의 RPNL 이 아니라 넣지 않음.
  mask 고친 판 (D-31, 사용자 10-18 "둘다 해"): 공식 ssi_mae 는 조각의 MAD 합에 mask 를 곱하지 않아 조각 밖·정답 없는 픽셀까지 더함. 논문 식 8 의 분모는
  조각 안 N 픽셀의 |d − 중앙값| 평균 → 그 두 줄에만 '* mask_valid' 를 붙인 판 (HDNRandomLoss.ssi_mae 의 줄과 같은 글자) 을 함께 씀 — metric3d_rpnl_masked.
- MiDaS 기울기 맞추기: 공저자 David Hafner 의 gist 'Loss function of MiDaS' (gist.github.com/dvdhfnr/732c26b61a0e63a0abc8a5d769dbebd0, 판 f417b71d, 2019-07-16) 를
  analysis/third_party/midas_loss.py 에 그대로 둠 (sha256 a3cf810af4d13dcc4f8253e814d25cfc5757d371ba9bae166086912fc643fd5c). ScaleAndShiftInvariantLoss 의 정규화 항 (alpha 항)
  = compute_scale_and_shift 로 최소제곱 정렬한 예측 disparity 와 정답 disparity 의 GradientLoss(scales 4, batch-based). 데이터 항 (SSI MSE) 은 우리 SILog 가 맡으므로 뺌.
  정답 disparity 는 논문 학습 설정대로 사진마다 [0, 1] 로 옮기고 늘림 (MiDaS §5 'We shift and scale the ground-truth disparity to the range [0, 1] for all datasets',
  10-18 원문 재확인 — gist 는 손실만이고 이 전처리는 데이터 쪽). 최소제곱 정렬이라 결과는 사진마다 1 / (정답 disparity 범위) 배.
  MegaDepth 는 공개 저장소 (zl548/MegaDepth) 에 학습 손실이 없어 (models/ 에 시험 코드만) 공식 코드로 확인할 수 있는 MiDaS 를 씀 (DAv2 교사 학습도 이 항, MiDaS 인용).
"""
import importlib.util
import json
import os
import sys
import types

import torch

VDR = os.path.expanduser("~/code/vlm-depth-rmse")
UD = os.path.join(VDR, "ext/UniDepth")
M3D = os.path.join(VDR, "ext/Metric3D/training")
MIDAS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "midas_loss.py")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]     # unidepth/utils/constants.py IMAGENET_DATASET_MEAN·STD 와 같은 값
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
_cache = {}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _unidepth_local_ssi():
    """공식 파일 셋 (utils/geometric.py · ops/losses/utils.py · ops/losses/local_ssi.py) 만 실행 — 패키지 __init__ (matplotlib 등을 부름) 은 건너뜀."""
    root = os.path.join(UD, "unidepth")
    for pkg, path in (("unidepth", root), ("unidepth.utils", f"{root}/utils"), ("unidepth.ops", f"{root}/ops"), ("unidepth.ops.losses", f"{root}/ops/losses")):
        if pkg not in sys.modules:
            p = types.ModuleType(pkg)
            p.__path__ = [path]
            sys.modules[pkg] = p
    _load("unidepth.utils.geometric", f"{root}/utils/geometric.py")
    _load("unidepth.ops.losses.utils", f"{root}/ops/losses/utils.py")
    return _load("unidepth.ops.losses.local_ssi", f"{root}/ops/losses/local_ssi.py")


def fast_extract_patches(tensor, sample_coords, patch_size: tuple):
    """공식 local_ssi.extract_patches (TorchScript, 조각마다 Python 반복) 와 같은 값을 고급 색인 한 번으로 (C = 1 — EG-SSI 는 깊이·mask 만 넘김).
    같은 패딩 (k//2, 0) · 같은 좌표 변환 (+패딩 뒤 .int()) · 같은 출력 모양 (N, 조각 수, k²). 공식 함수와 같음: edge_baselines_check.py (무작위 20 번 torch.equal)."""
    N, C, H, W = tensor.shape
    assert C == 1, C
    pw, ph = patch_size[0] // 2, patch_size[1] // 2
    tp = torch.nn.functional.pad(tensor, (pw, pw, ph, ph), mode="constant", value=0.0)
    c = sample_coords + torch.tensor([ph, pw], dtype=tensor.dtype, device=tensor.device).reshape(1, 1, 2)
    yc, xc = c[:, :, 0].int().long(), c[:, :, 1].int().long()
    dy, dx = torch.arange(-ph, ph + 1, device=tensor.device), torch.arange(-pw, pw + 1, device=tensor.device)
    out = tp[torch.arange(N, device=tensor.device)[:, None, None, None], 0, yc[:, :, None, None] + dy[:, None], xc[:, :, None, None] + dx[None, :]]
    return out.reshape(N, -1, patch_size[1] * patch_size[0])


def unidepth_egssi_module(fast=True):
    """UniDepthV2-L 학습 설정 그대로 만든 EdgeGuidedLocalSSI (train 모드 = 공식 학습처럼 조각 크기 무작위).
    fast=True: 조각 모으기만 fast_extract_patches (공식 CUDA 확장 RandomPatchExtractor 를 빌드하지 않아 내려가는 TorchScript 판과 같은 값, 약 25 배 빠름)."""
    key = f"egssi_{fast}"
    if key not in _cache:
        cfg = json.load(open(os.path.join(UD, "configs", "config_v2_vitl14.json")))["training"]["losses"]["ssi"]
        assert cfg["name"] == "EdgeGuidedLocalSSI", cfg
        mod = _unidepth_local_ssi().EdgeGuidedLocalSSI.build(cfg).train()
        if fast:
            mod.random_patch_extractor = fast_extract_patches
        _cache[key] = mod
    return _cache[key]


def unidepth_egssi(pred, gt, rgb):
    """pred·gt = (H, W) 깊이 m (같은 크기, gt 0 = 정답 없음), rgb = (H, W, 3) uint8 같은 크기 사진 → UniDepthV2 EG-SSI (공식 값, 가중치 1)."""
    img = ((rgb.permute(2, 0, 1).float() / 255 - IMAGENET_MEAN.to(rgb.device)) / IMAGENET_STD.to(rgb.device))[None]
    m = (gt > 0)[None, None]
    return unidepth_egssi_module()(pred[None, None], target=gt[None, None], mask=m, image=img, validity_mask=torch.ones_like(m)).mean()


def metric3d_rpnl_module():
    if "rpnl" not in _cache:
        hd = _load("m3d_hdsnl_random", os.path.join(M3D, "mono/model/losses/HDSNL_random.py"))
        _cache["rpnl"] = hd.HDSNRandomLoss(loss_weight=1.0, random_num=20, batch_limit=4)   # vit.raft5.large.py 의 random_num·batch_limit (loss_weight 는 밖에서)
    return _cache["rpnl"]


def metric3d_rpnl(pred, gt):
    """pred·gt = (H, W) 깊이 m → Metric3D v2 RPNL (공식 HDSNRandomLoss 값, 가중치 1). 무작위 조각은 공식대로 numpy 전역 난수."""
    t = gt[None, None]
    return metric3d_rpnl_module()(prediction=pred[None, None], target=t, mask=t > 1e-8, dataset=["dvsilog"])


def metric3d_rpnl_masked_module():
    """공식 HDSNL_random.py 를 읽어 ssi_mae 의 MAD 두 줄에만 '* mask_valid' 를 붙여 실행 (나머지는 공식 그대로). 고칠 줄이 공식 파일에 꼭 한 번 있고
    고친 줄이 같은 폴더 HDNL_random.py 에 글자 그대로 있는지 확인한 뒤에만 만든다."""
    if "rpnl_m" not in _cache:
        losses = os.path.join(M3D, "mono/model/losses")
        src, hdn = open(os.path.join(losses, "HDSNL_random.py")).read(), open(os.path.join(losses, "HDNL_random.py")).read()
        for old, new in (("gt_diff = (torch.abs(target - gt_median) ).reshape((B, C, -1))", "gt_diff = (torch.abs(target - gt_median) * mask_valid).reshape((B, C, -1))"),
                         ("pred_diff = (torch.abs(prediction - pred_median)).reshape((B, C, -1))", "pred_diff = (torch.abs(prediction - pred_median) * mask_valid).reshape((B, C, -1))")):
            assert src.count(old) == 1 and new in hdn, old
            src = src.replace(old, new)
        m = types.ModuleType("m3d_hdsnl_random_masked")
        exec(compile(src, os.path.join(losses, "HDSNL_random.py"), "exec"), m.__dict__)
        _cache["rpnl_m"] = m.HDSNRandomLoss(loss_weight=1.0, random_num=20, batch_limit=4)
    return _cache["rpnl_m"]


def metric3d_rpnl_masked(pred, gt):
    """metric3d_rpnl 과 같은 입력 → MAD 를 조각 안 정답 픽셀로만 계산한 RPNL (논문 식 8 의 정규화)."""
    t = gt[None, None]
    return metric3d_rpnl_masked_module()(prediction=pred[None, None], target=t, mask=t > 1e-8, dataset=["dvsilog"])


def midas_module():
    if "midas" not in _cache:
        _cache["midas"] = _load("midas_loss_official", MIDAS)
    return _cache["midas"]


def midas_gradient(pred, gt):
    """pred·gt = (H, W) 깊이 m → MiDaS 기울기 맞추기 (공식 gist: 최소제곱 정렬한 disparity 의 GradientLoss scales 4, 가중치 1). 정답 disparity 는 사진마다 [0, 1]."""
    g = midas_module()
    v = gt > 0
    if not bool(v.any()):
        return None
    m = v.float()[None]
    pd = 1.0 / pred[None]
    td = torch.where(v, 1.0 / gt.clamp(min=1e-8), torch.zeros_like(gt))
    lo, hi = td[v].min(), td[v].max()
    td = torch.where(v, (td - lo) / (hi - lo).clamp(min=1e-8), torch.zeros_like(td))[None]   # 논문 학습 설정: 정답 disparity 를 [0, 1] 로
    s, t = g.compute_scale_and_shift(pd, td, m)                          # ScaleAndShiftInvariantLoss.forward 와 같은 정렬 (기울기도 그대로 흐름)
    return g.GradientLoss(scales=4, reduction="batch-based")(s.view(-1, 1, 1) * pd + t.view(-1, 1, 1), td, m)


def rgb_edges(rgb):
    """D-31 분리 ② (우리 손실의 조각 위치만 RGB 경계로): UniDepthV2 공식 get_edge (ImageNet 정규화 사진의 Sobel, 테두리 3 px 0) 그대로. rgb = (H, W, 3) uint8 → (H, W)."""
    img = ((rgb.permute(2, 0, 1).float() / 255 - IMAGENET_MEAN.to(rgb.device)) / IMAGENET_STD.to(rgb.device))[None]
    return unidepth_egssi_module().get_edge(img, torch.ones_like(img[:, :1]))[0, 0]
