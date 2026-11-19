"""해결 실험 2 (NOTES D-12): DepthVLM 원 학습 데이터 6 종 10 만 장 (D-11) 으로 깊이 헤드를 다시 학습하고 NYUv2·iBims-1·DIODE Outdoor 에서 평가.

- DepthVLM-4B 는 고정 (vlm-depth-rmse eval/dense_full 과 같은 가중치 커밋·프롬프트·bf16·sdpa). 깊이 헤드 입력 4 개
  (DeepStack 5·11·17, LLM 마지막 층; 토큰 격자 ph × pw, 2560) 를 hook 으로 받는다.
- 기하 특징 (고정): udv2 = UniDepthV2-L pixel encoder (F-8·F-12 와 같은 가중치·추출), dino = 원래 DINOv2-L (같은 구조, 깊이 학습 없음).
  canonical 사진을 14 의 배수로 bicubic → 블록 6·12·18 출력 + 24 블록 정규화 출력 → DepthVLM 토큰 격자로 면적 평균 (층마다 1024).
- 헤드 (모두 DepthVLM 깊이 헤드 가중치에서 시작, 같은 사진·같은 순서로 함께 학습):
    control  = 헤드만 다시 학습 (추가 학습 효과)
    proposed = 헤드 + 층마다 gate × Linear(LayerNorm(udv2 특징)) 를 헤드 LayerNorm 뒤에 더함 (gate 0 에서 시작 = 원래 모델에서 출발)
    dino     = proposed 와 같고 특징만 원래 DINOv2-L (선택, '특징이 늘어서' vs '깊이 지식 덕분')
    udv2only = (D-17) VLM 없이 Linear(LayerNorm(udv2 특징)) 를 VLM 특징 자리에 넣음 → 헤드 LayerNorm → 나머지 (게이트 없음).
               실행 헤드가 모두 udv2only 면 VLM 을 올리지도 돌리지도 않는다 (헤드 초기값은 safetensors 의 depth_head 만, 격자는 image_grid_thw 로)
- 데이터·손실은 공식 학습 코드 그대로: utils/datasets.py 의 canonical 크기 resize·center crop·깊이 범위 (_clamp_canonical_size·
  _load_and_resize_image·_load_depth 를 그대로 부름), 손실 = modeling_qwen3_vl._silog_loss (λ 0.5, +1e-8), 예측을 GT 크기로
  bilinear(align_corners=True), valid = GT > 0 & 예측 > 0.
- 최적화는 F-10 (geo_inject) 과 같음: AdamW (wd 0), lr 헤드 2e-5 (공식 Stage-2)·Linear 1e-4·gate 1e-3, 사진 1 장 × 누적 4,
  워밍업 200 걸음 뒤 cosine, clip 1.0. 순서 = 학습 목록을 seed 0 으로 섞은 것 1 회.
- 검증: 6 데이터셋 공식 val 분할에서 100 장씩 (600 장), VAL_EVERY 장마다 이미지별 SILog (λ 1, ×100) 의 데이터셋 평균 →
  헤드마다 가장 좋은 상태를 best_<헤드>.pt 로 남긴다. 평가 세트는 학습·선택에 쓰지 않는다.
- 안전장치 (2026-10-10 독립 검토 반영):
  · CKPT_EVERY 장마다 last.pt (임시 파일 → 교체, 직전 것은 prev.pt). 저장 전에 가중치가 유한한지 확인. 최고 상태도 체크포인트 안에 둔다.
  · SIGTERM·SIGINT → 누적 경계에서 저장하고 끝냄. --resume → last → prev 순으로 읽어 같은 순서의 다음 사진부터 (설정·목록이 다르면 멈춤).
  · 읽기 실패 사진은 모든 헤드가 건너뜀. NaN 손실은 그 헤드만 (GT 가 비면 모든 헤드가 같이). 기울기 inf/NaN 이면 그 헤드의 그 걸음만 건너뜀.
    CUDA OOM 이면 그 누적 묶음을 모든 헤드에서 버리고 30 초 쉼 (헤드끼리 같은 사진만 배우게).
  · 최근 2,000 장 중 실패 5 % 초과 또는 가중치 NaN → abort.pt 에 남기고 종료 코드 3 (재시작 안 함, last/prev 는 깨끗하게 둠).
    최근 250 누적 묶음 중 OOM 20 % 초과 (GPU 경합) → 종료 코드 4 (유닛이 5 분 뒤 이어 학습). 집계 창은 저장하지 않아 재시작하면 새로 셈.
  · 실행은 systemd 유닛 + analysis/run_bg.sh (KillMode=mixed, Restart=on-failure, RestartPreventExitStatus=3, TimeoutStopSec=1200,
    HF_HUB_OFFLINE=1, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True). 명령은 NOTES 실행 로그.
- 평가 (학습이 끝나면 자동, 또는 --eval_only): ~/data/dvsilog/records_eval.json (NYUv2 테스트 652 장 Kinect 원측정·iBims-1 100 장·
  DIODE Outdoor 446 장), vlm-depth-rmse Track B 규칙 (geo_inject 와 같은 충분통계 — pooled RMSE·AbsRel·δ1, 이미지별 SILog,
  예측은 log 에서만 cap). 원래 DepthVLM·각 헤드 + 사후 보완 +VLMscale (F-11, 사진마다 원래 DepthVLM 의 log 평균 깊이에 맞춤), 이미지 bootstrap (B 1,000, seed 0).

사용: ~/venv/depthvlm/bin/python analysis/geo_mix.py [--heads control proposed dino] [--resume] [--smoke N] [--eval_only]
      (D-17: --heads udv2only --enc_res native --tag _udv2only — VLM 없이, 판정은 analysis/compare_udv2only.py)
출력: ~/checkpoints/dvsilog/geo_mix/{last,prev,best_<헤드>}.pt, ~/data/dvsilog/geo_mix/{train_log,val_log}.csv·per_image.parquet,
      tables/geo_mix.csv·geo_mix_comp.csv
"""
import argparse
import copy
import json
import math
import os
import random
import signal
import sys
import time
from collections import Counter, deque

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VDR = os.path.expanduser("~/code/vlm-depth-rmse")
sys.path.insert(0, os.path.join(VDR, "eval"))
sys.path.insert(0, os.path.join(VDR, "third_party/DepthVLM"))
sys.path.insert(0, os.path.join(VDR, "ext/Depth-Anything-V2/metric_depth"))
from breakdown import CAP                    # noqa: E402
from dense_full import DEPTHVLM              # noqa: E402
from utils.datasets import (DATASET_CONFIGS, DEFAULT_DEPTH_PROMPT, _clamp_canonical_size,  # noqa: E402
                            _load_and_resize_image, _load_depth)

DATA = os.path.expanduser("~/data/dvtrain/datasets")
ANN = os.path.expanduser("~/data/dvtrain/annotations")
BENCH = os.path.expanduser("~/data/depthvlm_bench")
D = os.path.expanduser("~/data/dvsilog")
CK = os.path.expanduser("~/checkpoints/dvsilog/geo_mix")
OUT = os.path.join(D, "geo_mix")
DEV = "cuda"
B = 1000
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
ENC_OF = {"control": None, "proposed": "udv2", "dino": "dino", "udv2only": "udv2", "proposed_hr": "udv2", "hr_ctrl": "udv2", "hr_gt": "udv2", "hr_dav2": "udv2",
          "hr_dav2b": "udv2", "hr_dav2d": "udv2"}
HR_HEADS = {"proposed_hr", "hr_ctrl", "hr_gt", "hr_dav2", "hr_dav2b", "hr_dav2d"}   # F-39 (10-15): 원래 해상도 기하 특징을 헤드의 세밀한 단계에도 넣는 헤드
EDGE_OF = {"hr_gt": "gt", "hr_dav2": "dav2"}                     # D-26 경계 손실 팔 (hr_ctrl = 같은 이어 학습, 손실만 SILog)
EDGE_OF.update(hr_dav2b="dav2b", hr_dav2d="dav2d")              # D-27: 새 경계 선택 (절대 문턱 + 하늘 제외) — b = 모든 사진, d = 조밀 정답 데이터셋만
ENC_OF.update(hr_dav2d_lo="udv2", hr_dav2d_hi="udv2")            # D-29: λ 민감도 헤드 — dav2d 와 같은 경계, 가중치만 다름 (--edge_w 에 헤드 이름=값)
HR_HEADS.update({"hr_dav2d_lo", "hr_dav2d_hi"})
EDGE_OF.update(hr_dav2d_lo="dav2d", hr_dav2d_hi="dav2d")
ENC_OF.update(v1_ctrl="udv2", v1_dav2d="udv2", hr_dav2d_hi2="udv2")   # D-30: 시너지 (고해상도 주입 없는 v1 = proposed 구조 × 경계 손실) · λ 0.36 — 가중치는 --edge_w 이름=값
HR_HEADS.update({"hr_dav2d_hi2"})
EDGE_OF.update(v1_dav2d="dav2d", hr_dav2d_hi2="dav2d")
ENC_OF.update(hr_midas="udv2", hr_rpnl="udv2", hr_egssi="udv2", hr_dav2d_gt="udv2", hr_dav2d_rgb="udv2")   # D-31: 기존 경계 손실 (공식 코드 — edge_baselines.py) · 우리 손실의 대상/위치 분리
HR_HEADS.update({"hr_midas", "hr_rpnl", "hr_egssi", "hr_dav2d_gt", "hr_dav2d_rgb"})
EDGE_OF.update(hr_midas="midas", hr_rpnl="rpnl", hr_egssi="egssi", hr_dav2d_gt="dav2d_gt", hr_dav2d_rgb="dav2d_rgb")
RGB_EDGE = {"egssi", "dav2d_rgb"}                                  # D-31: 사진 (canonical, 정답 크기 uint8) 이 필요한 경계 손실
ENC_OF.update(hr_rpnl_m="udv2")                                   # D-31 (사용자 10-18 "둘다 해"): RPNL 의 MAD 를 조각 안 정답 픽셀로만 (논문 식 8) — 공식 판 hr_rpnl 과 함께
HR_HEADS.update({"hr_rpnl_m"})
EDGE_OF.update(hr_rpnl_m="rpnl_m")
VS_HEADS = {"vs_dav2d_hi2"}                                       # D-52: 크기는 VLM — 출력 = 헤드 예측 × exp(mean ln p_DepthVLM − mean ln p_헤드), 손실 = SILog 크기 가중치 0 (√Var) + 경계 손실
ENC_OF.update(vs_dav2d_hi2="udv2")                                 # vs_dav2d_hi2 = hr_dav2d_hi2 (D-30 λ 0.36, 표의 Ours 조리법) 와 같은 구조 · 경계 손실, 크기만 VLM
EDGE_OF.update(vs_dav2d_hi2="dav2d")
HR_HEADS.update({"vs_dav2d_hi2"})
VD_HEADS = {"vd_dav2d_hi2"}                                       # D-53 (C안): 크기를 학습으로 VLM 에 — 손실의 크기 항 기준만 정답 → 원래 DepthVLM 예측 (정답 있는 픽셀), 출력은 헤드 그대로 (곱하기 없음)
ENC_OF.update(vd_dav2d_hi2="udv2")                                 # vd_dav2d_hi2 = hr_dav2d_hi2 (D-30 λ 0.36, 표의 Ours 조리법) 와 같은 구조 · 경계 손실, 크기 기준만 VLM
EDGE_OF.update(vd_dav2d_hi2="dav2d")
HR_HEADS.update({"vd_dav2d_hi2"})
EDGE_ABS, EDGE_SKY = math.log(1.05), 0.01                       # D-27: 후보 = 이웃 |Δ log disparity| ≥ log 1.05 (Depth Pro 가림 경계 5 %), 하늘 = disparity < 최댓값의 1 %
DENSE = {"taskonomy", "hm3d", "scannetpp"}                       # D-26: 정답이 조밀한 데이터셋 (팔 A 손실 · 검증 경계 F1)
DAV2_DIR = os.path.join(os.path.expanduser("~/data/dvsilog"), "dav2_teacher")   # D-26 팔 B 교사 (analysis/dav2_teacher.py, 학습 목록 번호.npy)
VLM_FREE = {"udv2only"}                                          # VLM 특징을 쓰지 않는 헤드 (D-17)
# D-59 (ⓑ 를 작업 기준 최종 방법으로): 모두 VS (출력 크기 = VLM, 손실 √Var), 처음 상태 · 데이터 · 순서 · 일정은 D-52 와 같음. 기존 헤드 동작은 그대로 (D-51).
#   ② vs_rgb_dav2d_hi2 = vs_dav2d_hi2 + canonical 사진 → 얕은 conv → 1×1 → 0 게이트 → 1/2 단계 (refinenet1 뒤, output_conv1 앞) — D-40 A
#   ③ 표 3 ablation 을 ⓑ 조리법으로: vs_control (b) · vs_dino (c) · vs_v1_ctrl (d) · vs_udv2only (e) · vs_hr_ctrl (f) = control · dino · v1_ctrl · udv2only · hr_ctrl 와 같은 구조, 경계 손실 없음
RGB_HEADS = {"vs_rgb_dav2d_hi2"}
D59_HEADS = {"vs_rgb_dav2d_hi2", "vs_control", "vs_dino", "vs_v1_ctrl", "vs_udv2only", "vs_hr_ctrl"}
VS_HEADS.update(D59_HEADS)
ENC_OF.update(vs_rgb_dav2d_hi2="udv2", vs_control=None, vs_dino="dino", vs_v1_ctrl="udv2", vs_udv2only="udv2", vs_hr_ctrl="udv2")
HR_HEADS.update({"vs_rgb_dav2d_hi2", "vs_hr_ctrl"})
EDGE_OF.update(vs_rgb_dav2d_hi2="dav2d")
VLM_FREE.add("vs_udv2only")
# D-66 (10-24, D-63 새 설계 파일럿): 기존 헤드 동작은 그대로 (D-51), 새 이름만 더함. 모두 D-30 조리법 (손실 λ 0.5 · DAv2 경계 · 같은 처음 상태) 위에서 하나씩만 바꿈.
#   tk   = 토큰 격자 전문가: canonical 사진을 (14·pw, 14·ph) 로 → UniDepthV2 패치 격자 = DepthVLM 토큰 격자 (평균 풀링 없음), 고해상도 경로도 같은 지도
#   tkc  = tk + 사진 평균 뺀 주입 (토큰 경로 · 고해상도 경로 모두, 주입 직전에 사진 안 평균을 채널마다 뺌)
#   nc   = 예전 조리법 (원래 해상도 전문가) + 사진 평균 뺀 주입 — 해상도 몫과 평균 빼기 몫을 가름
#   tkct = tkc + 학습 때만: 전문가를 끈 같은 헤드 (공식 손실) 와 사진 log 배율 (중앙값) 을 묶음
#   vlm  = VLM 특징만 헤드 + 같은 경계 손실 (모양 · 누출의 바닥선)
TOK_HEADS = {"tk_dav2d_hi2", "tkc_dav2d_hi2", "tkct_dav2d_hi2"}
CEN_HEADS = {"tkc_dav2d_hi2", "tkct_dav2d_hi2", "nc_dav2d_hi2"}
TIE_HEADS = {"tkct_dav2d_hi2"}
W_TIE = 1.0                                                       # 묶기 항 = |median(ln 헤드 − ln 전문가 끈 헤드)| (log 배율 단위, 결과 전 고정)
D66_HEADS = TOK_HEADS | CEN_HEADS | {"vlm_dav2d_hi2"}
ENC_OF.update({k: "udv2t" for k in TOK_HEADS}, nc_dav2d_hi2="udv2", vlm_dav2d_hi2=None)
HR_HEADS.update(TOK_HEADS | {"nc_dav2d_hi2"})
EDGE_OF.update({k: "dav2d" for k in D66_HEADS})
BASE_ENC = {"udv2t": "udv2"}                                      # 토큰 격자 입력도 같은 UniDepthV2-L 인코더 (특징 이름만 다름)
ENC_OF.update(kr_dav2d=None)                                       # D-84 (국내 논문): 주입 없는 헤드 (control 구조) + DAv2 경계 손실 dav2d — 가중치는 --edge_w kr_dav2d=값
EDGE_OF.update(kr_dav2d="dav2d")
MERGE = 2                                                         # DepthVLM spatial_merge_size (config) — 토큰 격자 = image_grid_thw 의 h · w / 2
MONITOR = {"scannetpp"}                                          # D-21: 검증에서 기록만 하고 고르기에는 안 쓰는 데이터셋
SETS = ("nyu_test", "ibims1", "diode_outdoor")
CAP_OF = {"nyu_test": CAP["nyuv2"], "ibims1": CAP["ibims1"], "diode_outdoor": CAP["diode_outdoor"]}


# ───────────────────────── 입력 만들기 (학습·검증은 DataLoader 작업자, 평가는 본 프로세스) ─────────────────────────
def chat(img):
    return [{"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": DEFAULT_DEPTH_PROMPT}]}]


def vlm_inputs(proc, img):
    from qwen_vl_utils import process_vision_info
    msg = chat(img)
    inp = proc(text=[proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)], images=[process_vision_info(msg)[0]],
               padding=True, return_tensors="pt")
    return {k: inp[k] for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}


ENC_RES = "canonical"                                            # 기하 인코더 입력: canonical (해결 실험 2) | native (해결 실험 3, D-14)
HR = False                                                       # 실행 헤드에 HR_HEADS 가 있으면 True (features 가 원래 해상도 지도도 돌려줌)


def unidepth_input(img, box=False):
    """UniDepthV2 공식 infer 전처리 (ext/UniDepth unidepthv2.py get_paddings·get_resize_factor, resolution_level 미설정 = 0.2–0.6 MP), enc_probe.unidepth_input 과 같음:
    ImageNet 정규화 → 가로세로비 [0.5, 2.5] 패딩 → 화소 0.2–0.6 MP 로 배율·14 배수 올림 → bilinear (align_corners=False). fp16 (3, H, W).
    box=True 면 패딩을 뺀 사진 영역을 14 px 칸 단위 (위, 아래, 왼, 오) 로 함께 돌려준다 — features(box=) 가 그 칸만 토큰 격자로 평균 (D-18: KITTI 처럼
    가로세로비가 [0.5, 2.5] 밖일 때만 차이. 학습·검증·기존 평가 사진은 모두 안이라 패딩 없음)."""
    W, H = img.size
    W0, H0, pt, pl = W, H, 0, 0
    x = ((torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255 - MEAN) / STD)[None]
    r = min(2.5, max(0.5, W / H))
    if W / H > r:
        Hn = int(W / r); pt = (Hn - H) // 2; x = Fn.pad(x, (0, 0, pt, Hn - H - pt)); H = Hn
    elif W / H < r:
        Wn = int(H * r); pl = (Wn - W) // 2; x = Fn.pad(x, (pl, Wn - W - pl, 0, 0)); W = Wn
    f = (min(6e5, max(2e5, W * H)) / (W * H)) ** 0.5
    nh, nw = math.ceil(int(H * f) / 14) * 14, math.ceil(int(W * f) / 14) * 14
    x = Fn.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)[0].half()
    if not box:
        return x
    sy, sx = nh / H / 14, nw / W / 14
    return x, (round(pt * sy), round((pt + H0) * sy), round(pl * sx), round((pl + W0) * sx))


def native_input(path, cs=None, crop=None, box=None):
    """원래 해상도 입력: 원본 사진에서 (공식 로더가 canonical 크기에서 가운데를 잘랐다면 같은 영역을 원본 좌표로 옮겨 자른 뒤) unidepth_input.
    D-62: box = (왼, 위, 폭, 높이) — 공식 처리 뒤 canonical 사진 좌표의 자르기 증강 상자. 그 영역을 원본 좌표로 옮겨 자름 (box 가 없으면 예전과 같음)."""
    img = Image.open(path).convert("RGB")
    if crop or box:
        W0, H0 = img.size
        sx, sy = W0 / cs[0], H0 / cs[1]
        left, top = ((cs[0] - crop[0]) // 2, (cs[1] - crop[1]) // 2) if crop else (0, 0)   # _load_and_resize_image 의 center crop 과 같은 식
        cw, ch = crop if crop else cs
        if box:
            left, top, cw, ch = left + box[0], top + box[1], box[2], box[3]
        img = img.crop((round(left * sx), round(top * sy), round((left + cw) * sx), round((top + ch) * sy)))
    return unidepth_input(img)


def enc_input(img):
    """F-8·geo_inject 와 같은 기하 인코더 입력: canonical 사진 → 14 의 배수로 bicubic → ImageNet 정규화 (fp16)."""
    w14, h14 = max(14, round(img.size[0] / 14) * 14), max(14, round(img.size[1] / 14) * 14)
    x = torch.from_numpy(np.asarray(img.resize((w14, h14), Image.BICUBIC)).copy()).permute(2, 0, 1).float() / 255
    return ((x - MEAN) / STD).half()


def tok_input(img, ph, pw):
    """D-66 토큰 격자 전문가 입력: VLM 과 같은 canonical 사진을 (14·pw, 14·ph) 로 bicubic → ImageNet 정규화 (fp16).
    UniDepthV2 패치 격자 (14 px) 가 DepthVLM 토큰 격자 (ph × pw) 와 칸마다 1:1 — 평균 풀링이 항등이 됨."""
    x = torch.from_numpy(np.asarray(img.resize((14 * pw, 14 * ph), Image.BICUBIC)).copy()).permute(2, 0, 1).float() / 255
    return ((x - MEAN) / STD).half()


def tok_grid(inp):
    """vlm_inputs 의 image_grid_thw → DepthVLM 토큰 격자 (ph, pw) (모델 코드와 같은 식)."""
    _, h, w = (int(v) for v in inp["image_grid_thw"][0])
    return h // MERGE, w // MERGE


class MixSet(torch.utils.data.Dataset):
    """학습·검증 목록 (data_prep/build_mix.py) 한 줄 → 공식 dataset_pixel_depth_train.__getitem__ 과 같은 사진·깊이 + 모델 입력.
    공식은 읽기 실패 시 다음 사진으로 넘어가지만, 여기서는 'bad' 로 돌려 본 루프가 세고 건너뛴다 (목록은 미리 전부 읽어 확인함)."""

    def __init__(self, recs, enc_res="canonical", dav2=False, rgb=False, crop_aug=None, tok=False):
        self.recs, self.proc, self.enc_res, self.dav2, self.rgb = recs, None, enc_res, dav2, rgb
        self.crop_aug = crop_aug                                      # D-62: (확률, 최대 배율, seed) 또는 None (예전과 같음)
        self.tok = tok                                                # D-66: 토큰 격자 전문가 입력 x14t 도 만듦 (기본 꺼짐 = 예전과 같음)

    def crop_box(self, i, g):
        """D-62 자르기 증강 상자 (canonical 좌표 왼 · 위 · 폭 · 높이) 와 배율 s, 아니면 (None, 1.0). 사진 번호로 정해진 난수 — 이어 학습해도 같음."""
        p, s_max, seed = self.crop_aug
        rng = np.random.default_rng([int(seed), int(i)])
        if rng.random() >= p:
            return None, 1.0
        H, W = g.shape
        n_all = int((g > 0).sum())
        for _ in range(5):
            s = float(np.exp(rng.uniform(0.0, np.log(s_max))))
            w, h = max(32, round(W / s)), max(32, round(H / s))
            l, t = int(rng.integers(0, W - w + 1)), int(rng.integers(0, H - h + 1))
            n = int((g[t:t + h, l:l + w] > 0).sum())
            if n >= max(50, 0.5 * n_all / s ** 2):                    # 상자 안 정답이 기대의 절반 이상
                return (l, t, w, h), s
        return None, 1.0

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        if self.proc is None:
            from transformers import AutoProcessor
            self.proc = AutoProcessor.from_pretrained(DEPTHVLM[0], revision=DEPTHVLM[1])
            self.proc.tokenizer.padding_side = "left"
        r = self.recs[i]
        cfg = DATASET_CONFIGS[r["dataset"]]
        cs, crop = _clamp_canonical_size(r.get("canonical_size"), cfg.get("max_canonical_edge"))
        img = _load_and_resize_image(os.path.join(DATA, r["image"].lstrip("/")), cs, crop)
        p = lambda k: os.path.join(DATA, r[k]) if r.get(k) else None
        g = _load_depth(p("depth_path"), r.get("depth_scale", 1000.0), cfg["min_depth"], cfg["max_depth"], cs, crop,
                        p("mask_valid_path"), p("mask_transp_path"), r.get("depth_format")) if img is not None else None
        if img is None or g is None:
            return dict(i=i, bad=True)
        box, s_aug = self.crop_box(i, g) if self.crop_aug else (None, 1.0)
        H_full, W_full = g.shape
        if box is not None:                                           # D-62: canonical 사진 · 정답을 같은 상자로 (VLM 은 초점 그대로 크기만 작아짐)
            l, t, w, h = box
            img, g = img.crop((l, t, l + w, t + h)), np.ascontiguousarray(g[t:t + h, l:l + w])
        x14 = native_input(os.path.join(DATA, r["image"].lstrip("/")), cs, crop, box) if self.enc_res == "native" else enc_input(img)
        out = dict(i=i, ds=r["dataset"], g=torch.from_numpy(g), inp=vlm_inputs(self.proc, img), x14=x14)
        if self.tok:                                                  # D-66: 같은 canonical 사진 → 토큰 격자 크기
            out["x14t"] = tok_input(img, *tok_grid(out["inp"]))
        if self.crop_aug:
            out["crop_s"] = s_aug
        if self.rgb:                                                  # D-31: 정답과 같은 canonical 크기의 사진 (UniDepthV2 EG-SSI 의 RGB 경계)
            out["rgb"] = torch.from_numpy(np.asarray(img).copy())
        f = os.path.join(DAV2_DIR, f"{i}.npy")
        if self.dav2 and os.path.exists(f):                          # D-26 팔 B: 같은 canonical 사진의 DAv2 disparity (절반 크기)
            dv = np.load(f).astype(np.float32)
            if box is not None:                                       # D-62: 교사 지도도 같은 영역 (교사는 정답 크기의 약 절반)
                fy, fx = dv.shape[0] / H_full, dv.shape[1] / W_full
                dv = np.ascontiguousarray(dv[round(t * fy):round((t + h) * fy), round(l * fx):round((l + w) * fx)])
            out["dv"] = torch.from_numpy(dv)
        return out


def interleave(a, b):
    """a 의 순서를 그대로 두고 b 를 고르게 끼움 (a 를 i 장 지난 뒤 b 는 floor(i·|b|/|a|) 장) — D-21."""
    out, j = [], 0
    for i, x in enumerate(a, 1):
        out.append(x)
        while j < len(b) and (j + 1) * len(a) <= i * len(b):
            out.append(b[j])
            j += 1
    out.extend(b[j:])
    return np.array(out)


class OrderSampler(torch.utils.data.Sampler):
    def __init__(self, idx):
        self.idx = list(idx)

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)


def _worker_init(_):
    torch.set_num_threads(1)                                  # 작업자 여럿이 CPU 를 나눠 쓰게
    cv2.setNumThreads(1)


def loader(ds, idx, workers):
    return torch.utils.data.DataLoader(ds, batch_size=1, sampler=OrderSampler(idx), num_workers=workers, collate_fn=lambda b: b[0],
                                       prefetch_factor=4 if workers else None, persistent_workers=False, worker_init_fn=_worker_init)


# ───────────────────────── 모델 ─────────────────────────
def depth_head_only():
    """D-17: VLM 을 올리지 않고 DepthVLM 깊이 헤드만 — 모델 코드 (modeling_qwen3_vl 1353–1357) 와 같은 구조에 safetensors 의 depth_head.* 를 strict 로 넣고
    저장 dtype (bf16) 으로 둔다 = from_pretrained(torch_dtype=bf16) 로 받은 헤드와 같은 값. → (헤드 (CPU, bf16), spatial_merge_size)."""
    from huggingface_hub import hf_hub_download
    from model.dpt_depth_head import DPTDepthHead
    from safetensors import safe_open
    cfg = json.load(open(hf_hub_download(DEPTHVLM[0], "config.json", revision=DEPTHVLM[1])))
    assert os.environ.get("DEPTHLM_DEPTH_HEAD_TYPE", cfg.get("depth_head_type", "dpt")) == "dpt"
    base = DPTDepthHead(dim_in=cfg["text_config"]["hidden_size"], features=256, out_channels=[256, 512, 1024, 1024])
    with safe_open(hf_hub_download(DEPTHVLM[0], "model.safetensors", revision=DEPTHVLM[1]), "pt") as f:
        sd = {k[len("depth_head."):]: f.get_tensor(k) for k in f.keys() if k.startswith("depth_head.")}
    base.load_state_dict(sd, strict=True)
    return base.to(torch.bfloat16).eval(), cfg["vision_config"]["spatial_merge_size"]


def load_models(encs, vlm=True):
    encs = sorted({BASE_ENC.get(k, k) for k in encs})                # D-66: 'udv2t' (토큰 격자 입력) 는 같은 udv2 인코더
    from depth_anything_v2.dinov2 import DINOv2
    from model import Qwen3VLForConditionalGeneration
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(DEPTHVLM[0], revision=DEPTHVLM[1])
    proc.tokenizer.padding_side = "left"
    cap = {"skip": False}
    if vlm:
        m = Qwen3VLForConditionalGeneration.from_pretrained(DEPTHVLM[0], revision=DEPTHVLM[1], torch_dtype=torch.bfloat16,
                                                            attn_implementation="sdpa", device_map="cuda:0").eval()
        for q in m.parameters():
            q.requires_grad_(False)
        orig_fwd = m.depth_head.forward

        def head_fwd(features_list, patch_h, patch_w, output_size=None):
            """헤드 입력을 잡아 두고, 학습 중(skip)에는 원래 헤드 계산을 건너뛴다 (원래 예측은 검증·평가에서만 씀)."""
            cap.update(f=[t.detach() for t in features_list], hw=(int(patch_h), int(patch_w)))
            if cap["skip"]:
                return features_list[0].new_zeros(1, 1, 1)
            return orig_fwd(features_list, patch_h, patch_w, output_size=output_size)
        m.depth_head.forward = head_fwd
        base, merge = m.depth_head, None
    else:                                                             # D-17: VLM 없는 실행 (헤드가 모두 udv2only)
        m, (base, merge) = None, depth_head_only()

    from depth_anything_v2.dinov2_layers import attention as dino_attn

    def sdpa(self, x, attn_bias=None):
        """DINOv2 attention 을 PyTorch SDPA 로 (xFormers 가 없으면 공식 구현은 N×N 행렬을 만든다). 같은 식, 특징 상대차 < 0.5 % (2026-10-10),
        인코더 3.5–4.5 배 빠름."""
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x = Fn.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2]).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))
    dino_attn.Attention.forward = dino_attn.MemEffAttention.forward = sdpa

    def make(sd):
        e = DINOv2(model_name="vitl")
        e.load_state_dict(sd, strict=True)
        return e.half().to(DEV).eval()
    src = {"udv2": lambda: torch.load(os.path.join(D, "unidepthv2_vitl14_pixel_encoder.pt"), map_location="cpu"),   # F-12 에서 꺼내 둔 것
           "dino": lambda: torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
                                                             map_location="cpu")}
    enc = {k: make(src[k]()) for k in encs}

    @torch.no_grad()
    def features(inp, x14, need_p0=True, box=None, hr=False, x14t=None):
        """→ (원래 DepthVLM 예측 (ph·32 × pw·32) 또는 None, 헤드 입력 4 개 fp32 (VLM 없는 실행은 None), {인코더: 특징 4 개 (ph·pw, 1024)}, (ph, pw))
        [hr=True 면 + {인코더: 블록 6·12·18 특징 지도 (1, 1024, h, w) — 토큰 격자로 평균하지 않은 원래 해상도, 채널 LayerNorm} (F-39)]."""
        if m is None:                                                 # D-17: 격자만 모델 코드 (modeling_qwen3_vl 1506–1510) 와 같은 식으로
            _, h, w = (int(v) for v in inp["image_grid_thw"][0])
            p, (ph, pw) = None, (h // merge, w // merge)
        else:
            inp = {k: v.to(DEV) for k, v in inp.items()}
            cap.update(skip=not need_p0, f=None, hw=None)
            p = m(**inp, logits_to_keep=1).depth_pred[0].float()     # 로짓은 마지막 토큰만 (깊이 헤드 입력은 전체 은닉 상태라 영향 없음)
            if cap["f"] is None:
                raise RuntimeError("깊이 헤드가 불리지 않음 — 헤드 입력을 못 잡음")
            while p.dim() > 2:
                p = p[0]
            p = p if need_p0 else None
            ph, pw = cap["hw"]
        geo, ghr = {}, {}
        if enc:
            x = x14[None].to(DEV)
            h14, w14 = x.shape[-2] // 14, x.shape[-1] // 14
            crop = (lambda t: t) if box is None else (lambda t: t[..., box[0]:box[1], box[2]:box[3]])   # unidepth_input(box=True) 의 사진 영역 칸만
            for k, e in enc.items():
                hs = list(e.get_intermediate_layers(x, [5, 11, 17, 23], norm=False))
                hs[-1] = e.norm(hs[-1])
                maps = [crop(h[0].float().T.reshape(1, -1, h14, w14)) for h in hs]
                geo[k] = [Fn.adaptive_avg_pool2d(t, (ph, pw))[0].reshape(1024, -1).T for t in maps]
                if hr:
                    ghr[k] = [Fn.layer_norm(t.permute(0, 2, 3, 1), (1024,)).permute(0, 3, 1, 2) for t in maps[:3]]
            if x14t is not None and "udv2" in enc:                    # D-66: 토큰 격자 입력 — 패치 격자 = 토큰 격자라 평균 풀링 없음 (특징 이름 'udv2t')
                xt = x14t[None].to(DEV)
                assert (xt.shape[-2] // 14, xt.shape[-1] // 14) == (ph, pw), (tuple(xt.shape), ph, pw)
                hs = list(enc["udv2"].get_intermediate_layers(xt, [5, 11, 17, 23], norm=False))
                hs[-1] = enc["udv2"].norm(hs[-1])
                maps = [h[0].float().T.reshape(1, -1, ph, pw) for h in hs]
                geo["udv2t"] = [t[0].reshape(1024, -1).T for t in maps]
                if hr:
                    ghr["udv2t"] = [Fn.layer_norm(t.permute(0, 2, 3, 1), (1024,)).permute(0, 3, 1, 2) for t in maps[:3]]
        out = (p, (None if m is None else [f.float() for f in cap["f"]]), geo, (ph, pw))
        return out + (ghr,) if hr else out
    features.model = m                                              # VQA 확인 (vqa_eval.py) 이 같은 모델 객체로 답을 생성하도록
    return proc, base, features


class Head(torch.nn.Module):
    """DepthVLM DPT 깊이 헤드 사본 (+ 선택: 기하 특징 주입). forward 는 dpt_depth_head.DPTDepthHead.forward 와 같고
    (output_size = 토큰 격자 × 32, 모델 forward 와 같음) LayerNorm 뒤 더하기만 다르다 — geo_inject.Head 와 같은 코드.
    vlm=False (udv2only, D-17): VLM 특징 대신 Linear(LayerNorm(기하 특징)) 이 헤드 입력 (헤드 LayerNorm 부터 그대로)."""

    def __init__(self, base, inject, vlm=True, hr=False, rgb=False, center=False):
        super().__init__()
        self.center = center                                          # D-66: 주입 직전에 전문가 특징의 사진 안 평균 (채널마다) 을 뺌
        if center:
            self.register_buffer("center_flag", torch.ones(1))        # 체크포인트 키로 평가 쪽 (kitti_eval.load_heads) 이 알아봄
        self.h = copy.deepcopy(base).float()
        for q in self.h.parameters():
            q.requires_grad_(True)
        self.proj = torch.nn.ModuleList([torch.nn.Linear(1024, base.norm.normalized_shape[0]) for _ in range(4)]) if inject else None
        self.gnorm = torch.nn.LayerNorm(1024, elementwise_affine=False)      # DINOv2 중간층 값 크기가 커서 주입 전에 정규화
        self.gate = torch.nn.Parameter(torch.zeros(4)) if inject and vlm else None   # 0 에서 여는 층별 게이트 (udv2only 는 지킬 원래 신호가 없어 없음)
        self.vlm = vlm
        # F-39 (10-15): 원래 해상도 기하 특징 (블록 6·12·18) 을 헤드의 1/4·1/8·1/16 단계 (resize 뒤) 에도 더함 — 1×1 투영 + 0 에서 여는 게이트 (proj 뒤에 만들어 proj 초기값은 같음)
        self.hproj = torch.nn.ModuleList([torch.nn.Conv2d(1024, base.projects[i].out_channels, 1) for i in range(3)]) if hr else None
        self.hgate = torch.nn.Parameter(torch.zeros(3)) if hr else None
        # D-59 ② (D-40 A): canonical 사진 → 얕은 conv (3 → 32 → 64 → 64, 3×3 · ReLU) → 1×1 (→ 256) → 0 에서 여는 게이트 → 1/2 단계에 더함.
        # 초기값은 따로 고정한 난수 (fork_rng · seed 59) 라 다른 헤드 · 헤드 수와 무관, 게이트 0 이라 처음 출력은 이 경로가 없는 헤드와 같음.
        self.rconv = self.rproj = self.rgate = None
        if rgb:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(59)
                self.rconv = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(),
                                                 torch.nn.Conv2d(64, 64, 3, padding=1), torch.nn.ReLU())
                self.rproj = torch.nn.Conv2d(64, base.scratch.output_conv1.in_channels, 1)
            self.rgate = torch.nn.Parameter(torch.zeros(1))

    def _scratch_rgb(self, feats, rgb):
        """D-59 ②: dpt_depth_head.DPTDepthHead._scratch_forward 와 같은 계산 + refinenet1 출력 (1/2 단계) 에 사진 경로 (사진은 그 크기로 area)."""
        s = self.h.scratch
        l1, l2, l3, l4 = s.layer1_rn(feats[0]), s.layer2_rn(feats[1]), s.layer3_rn(feats[2]), s.layer4_rn(feats[3])
        x = s.refinenet4(l4, size=l3.shape[2:])
        x = s.refinenet3(x, l3, size=l2.shape[2:])
        x = s.refinenet2(x, l2, size=l1.shape[2:])
        x = s.refinenet1(x, l1)
        x = x + self.rgate * self.rproj(self.rconv(Fn.interpolate(rgb, size=x.shape[-2:], mode="area")))
        return s.output_conv1(x)

    def forward(self, feats, geo, ph, pw, ghr=None, rgb=None, inject=True):
        """inject=False (D-66 묶기의 '전문가를 끈 같은 헤드', 학습 때만): 주입 항을 빼고 같은 가중치로 계산 (게이트 0 일 때와 같은 꼴)."""
        out = []
        for i in range(4):
            if self.vlm:
                x = self.h.norm(feats[i])
                if self.proj is not None and inject:
                    gi = geo[i] - geo[i].mean(0, keepdim=True) if self.center else geo[i]   # D-66: 토큰 평균을 뺀 뒤 토큰마다 LN
                    x = x + self.gate[i] * self.proj[i](self.gnorm(gi))[None]
            else:                                                     # udv2only (D-17): 투영한 기하 특징이 VLM 특징 자리에 그대로
                x = self.h.norm(self.proj[i](self.gnorm(geo[i]))[None])
            x = x.permute(0, 2, 1).reshape(1, x.shape[-1], ph, pw)
            y = self.h.resize_layers[i](self.h.projects[i](x))
            if self.hproj is not None and i < 3 and inject:           # 투영은 원래 해상도에서 하고 이 단계 크기로 bilinear (둘 다 선형이라 순서와 무관, 가벼움)
                gh = ghr[i] - ghr[i].mean((2, 3), keepdim=True) if self.center else ghr[i]   # D-66: 위치 평균을 뺌 (채널 LN 뒤 지도)
                y = y + self.hgate[i] * Fn.interpolate(self.hproj[i](gh), size=y.shape[-2:], mode="bilinear", align_corners=False)
            out.append(y)
        if self.rconv is not None:
            assert rgb is not None, "D-59 사진 경로 헤드에는 canonical 사진 (rgb_input) 이 필요"
        fused = Fn.interpolate(self.h._scratch_forward(out) if self.rconv is None else self._scratch_rgb(out, rgb), size=(ph * 32, pw * 32),
                               mode="bilinear", align_corners=True)
        return self.h.scratch.output_conv2(fused).squeeze(1)[0]


def rgb_input(img):
    """D-59 ②: canonical 사진 (PIL 또는 H×W×3 uint8 텐서) → ImageNet 정규화 (1, 3, H, W) fp32 (DEV). 1/2 단계 크기로 줄이는 것은 헤드 안에서."""
    x = (img if torch.is_tensor(img) else torch.from_numpy(np.asarray(img).copy())).permute(2, 0, 1).float() / 255
    return ((x - MEAN) / STD)[None].to(DEV)


def vlm_scale(pred, p0):
    """D-52: 헤드 예측의 전역 크기를 원래 DepthVLM 예측의 것으로 — pred · exp(mean ln p0 − mean ln pred) (사진 전체 픽셀, 정답 안 씀). 미분 가능 (헤드 쪽 전역 크기로는 기울기 0).
    eval 의 +VLMscale (F-11) 과 같은 식 (그쪽은 numpy · 세트 범위 clip). p0 는 고정 모델 출력, 크기가 다르면 pred 크기로 bilinear."""
    if p0.shape != pred.shape:
        p0 = Fn.interpolate(p0[None, None], size=pred.shape, mode="bilinear", align_corners=True)[0, 0]
    lp, l0 = torch.log(pred.clamp(min=1e-3)), torch.log(p0.detach().clamp(min=1e-3))
    return pred * torch.exp(l0.mean() - lp.mean())


def predict(hd, feats, geo, ph, pw, ghr=None, rgb=None, inject=True):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return hd(feats, geo, ph, pw, ghr, rgb, inject).float()        # rgb 는 D-59 사진 경로 헤드만 씀 (다른 헤드는 무시) · inject=False 는 D-66 묶기만


def official_loss(pred, gt, lam=0.5):
    """modeling_qwen3_vl 의 학습 손실 그대로: GT 크기로 bilinear(align_corners=True), valid = GT > 0 & 예측 > 0, SILog λ 0.5 + 1e-8 (fp32).
    D-52: lam=1.0 = 크기 가중치 0 (√Var(d)) — VS_HEADS 만."""
    p = Fn.interpolate(pred[None, None], size=gt.shape, mode="bilinear", align_corners=True)[0, 0]
    v = (gt > 0) & (p > 0)
    if not v.any():
        return None
    d = torch.log(p[v]) - torch.log(gt[v])
    return torch.sqrt(torch.clamp((d ** 2).mean() - lam * d.mean() ** 2, min=0) + 1e-8)


def vlm_target_loss(pred, gt, p0, w=0.5):
    """D-53 (C안): official_loss (λ 0.5) 의 크기 항 기준만 정답 → 원래 DepthVLM 예측 p0 — √( Var(d) + w·mean(d0)² ), d = ln 예측 − ln 정답, d0 = ln 예측 − ln p0.
    둘 다 같은 픽셀 (GT 크기로 bilinear(align_corners=True), valid = GT > 0 & 예측 > 0 & p0 > 0) → 정답 없는 픽셀 (하늘 등) 은 크기 기준에도 안 들어감. p0 는 고정 모델 출력 (기울기 없음).
    p0 = 정답이면 official_loss(lam=0.5) 와 같은 값 (Var(d) + 0.5·mean(d)² = mean(d²) − 0.5·mean(d)²)."""
    p = Fn.interpolate(pred[None, None], size=gt.shape, mode="bilinear", align_corners=True)[0, 0]
    q = Fn.interpolate(p0.detach().float()[None, None], size=gt.shape, mode="bilinear", align_corners=True)[0, 0]
    v = (gt > 0) & (p > 0) & (q > 0)
    if not v.any():
        return None
    d = torch.log(p[v]) - torch.log(gt[v])
    d0 = torch.log(p[v]) - torch.log(q[v])
    return torch.sqrt(torch.clamp((d ** 2).mean() - d.mean() ** 2 + w * d0.mean() ** 2, min=0) + 1e-8)


def grad_loss(pred, gt, K=4):
    """D-26 팔 A: MegaDepth (Li & Snavely 2018) 식 다중 크기 log 깊이 기울기 맞추기 — R = log 예측 − log GT 의 x·y 차분 L1, 2 배씩 줄인 K 크기의 평균.
    official_loss 와 같이 GT 크기로 bilinear(align_corners=True), valid = GT > 0 & 예측 > 0, 차분은 두 픽셀이 다 valid 일 때만."""
    p = Fn.interpolate(pred[None, None], size=gt.shape, mode="bilinear", align_corners=True)[0, 0]
    v = (gt > 0) & (p > 0)
    R = torch.where(v, torch.log(p.clamp(min=1e-8)) - torch.log(gt.clamp(min=1e-8)), torch.zeros_like(p))
    out = []
    for k in range(K):
        r, m = R[::2 ** k, ::2 ** k], v[::2 ** k, ::2 ** k]
        mx, my = m[:, 1:] & m[:, :-1], m[1:, :] & m[:-1, :]
        nk = mx.sum() + my.sum()
        if nk > 0:
            out.append(((r[:, 1:] - r[:, :-1]).abs()[mx].sum() + (r[1:, :] - r[:-1, :]).abs()[my].sum()) / nk)
    return torch.stack(out).mean() if out else None


def egssi_loss(pred, dv, M=32, q=0.95, smin=0.04, smax=0.08, abs_thr=None, sky=None, tgt=None, cand_map=None):
    """D-26 팔 B: UniDepthV2 EG-SSI 식, 교사만 DAv2 — 교사 log disparity 기울기 상위 (1 − q) 위치에서 조각 M 개 (한 변 = 짧은 변의 smin–smax),
    조각마다 예측 역깊이와 교사 disparity 를 각자 (− 중앙값) / 평균 절대 편차 로 정규화한 L1 의 평균. 정규화로 교사의 배율·이동·큰 규모 모양은 넘어오지 않음.
    교사는 절반 크기라 예측을 그 크기로 bilinear (align_corners=True). 위치·크기 뽑기는 torch 난수 (체크포인트에 상태 저장).
    D-27: abs_thr 를 주면 상위 (1 − q) 대신 |Δ log d| ≥ abs_thr 인 곳만 후보. sky 를 주면 disparity < sky × 최댓값 인 픽셀 (하늘처럼 아주 먼 곳) 과
    그 이웃 1 px 는 후보에서 빼고, 조각 안에서도 비교에서 뺌 (남는 픽셀이 절반 미만인 조각은 버림) — 교사가 하늘 깊이를 끌고 가지 못하게.
    D-31 분리: tgt (dv 와 같은 격자의 정답 역깊이, 0 = 없음) 를 주면 조각 비교 대상만 정답 (정답 없는 픽셀은 하늘처럼 뺌) — 위치는 그대로 교사 경계.
    cand_map (같은 격자의 RGB 경계 크기, edge_baselines.rgb_edges) 을 주면 후보 위치만 UniDepthV2 식 (0.90 분위 이상) — 대상은 그대로 교사."""
    H, W = dv.shape
    ip = 1.0 / Fn.interpolate(pred[None, None], size=(H, W), mode="bilinear", align_corners=True)[0, 0]
    lt = torch.log(dv.clamp(min=float(1e-3 * dv.max().clamp(min=1e-6))))   # 하늘 (disparity ≈ 0) 은 최댓값의 1/1000 로 바닥
    mag = torch.hypot(Fn.pad(lt[:, 1:] - lt[:, :-1], (0, 1)), Fn.pad(lt[1:, :] - lt[:-1, :], (0, 0, 0, 1)))
    cand = mag >= (torch.quantile(mag.flatten()[::max(1, mag.numel() // 200000)], q) if abs_thr is None else abs_thr)
    if cand_map is not None:                                          # D-31 분리 ②: UniDepthV2 sample_strong_edges 와 같은 0.90 분위 (그쪽은 1/14 격자에서, 여기는 픽셀)
        cand = cand_map >= torch.quantile(cand_map.flatten()[::max(1, cand_map.numel() // 200000)], 0.90)
    keep = None
    if sky is not None:
        far = dv < sky * dv.max()
        cand &= ~(Fn.max_pool2d(far[None, None].float(), 3, 1, 1)[0, 0] > 0)
        keep = ~far
    if tgt is not None:                                               # D-31 분리 ①: 대상 = 정답 역깊이, 정답 없는 픽셀은 비교에서 뺌
        keep = (tgt > 0) if keep is None else keep & (tgt > 0)
        dv = tgt
    ys, xs = torch.nonzero(cand, as_tuple=True)
    if len(ys) == 0:
        return None
    sel = torch.randint(len(ys), (M,), device=dv.device)
    size = torch.randint(max(4, int(smin * min(H, W))), max(5, int(smax * min(H, W)) + 1), (M,)).tolist()
    out = []
    for y, x, s in zip(ys[sel].tolist(), xs[sel].tolist(), size):
        y0, x0 = min(max(y - s // 2, 0), H - s), min(max(x - s // 2, 0), W - s)
        a_, b_ = ip[y0:y0 + s, x0:x0 + s].flatten(), dv[y0:y0 + s, x0:x0 + s].flatten()
        if keep is not None:
            kk = keep[y0:y0 + s, x0:x0 + s].flatten()
            if kk.float().mean() < 0.5:
                continue
            a_, b_ = a_[kk], b_[kk]
        na = (a_ - a_.median()) / ((a_ - a_.median()).abs().mean() + 1e-6 + 1e-3 * a_.median().abs())
        nb = (b_ - b_.median()) / ((b_ - b_.median()).abs().mean() + 1e-6 + 1e-3 * b_.median().abs())
        out.append((na - nb).abs().mean())
    return torch.stack(out).mean() if out else None


def edge_baseline(kind, pred, g, rgb):
    """D-31: 기존 경계 손실 셋 (edge_baselines.py — 공식 코드 그대로). 예측은 official_loss 와 같이 정답 크기로 bilinear(align_corners=True)."""
    import edge_baselines as EB
    pg = Fn.interpolate(pred[None, None], size=g.shape, mode="bilinear", align_corners=True)[0, 0]
    if kind == "midas":
        return EB.midas_gradient(pg, g)
    if kind == "rpnl":
        return EB.metric3d_rpnl(pg, g)
    if kind == "rpnl_m":
        return EB.metric3d_rpnl_masked(pg, g)
    return EB.unidepth_egssi(pg, g, rgb.to(pred.device))


def edge_split(kind, pred, g, dv, rgb):
    """D-31 분리: 우리 손실 (dav2d) 에서 하나만 바꿈 — dav2d_gt = 비교 대상만 정답 역깊이 (교사 격자로 nearest), dav2d_rgb = 후보 위치만 RGB 경계 (교사 격자로 area)."""
    if kind == "dav2d_gt":
        gd = Fn.interpolate(g[None, None], size=dv.shape, mode="nearest")[0, 0]
        return egssi_loss(pred, dv, abs_thr=EDGE_ABS, sky=EDGE_SKY, tgt=torch.where(gd > 0, 1.0 / gd.clamp(min=1e-8), torch.zeros_like(gd)))
    import edge_baselines as EB
    rh = Fn.interpolate(rgb.to(pred.device).permute(2, 0, 1)[None].float(), size=dv.shape, mode="area")[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8)
    return egssi_loss(pred, dv, abs_thr=EDGE_ABS, sky=EDGE_SKY, cand_map=EB.rgb_edges(rh))


def _nms_rows(c, ratio):
    """각 행에서 이어진 경계 픽셀 묶음마다 비가 가장 큰 것만 남김 (Depth Pro boundary_metrics 의 nms_horizontal 과 같은 뜻 — 번진 경계는 한 픽셀로)."""
    out = np.zeros_like(c)
    r, cc = np.nonzero(c)
    if len(r) == 0:
        return out
    start = np.r_[True, (r[1:] != r[:-1]) | (cc[1:] != cc[:-1] + 1)]
    v = ratio[r, cc]
    top = np.maximum.reduceat(v, np.flatnonzero(start))[np.cumsum(start) - 1]
    k = v == top
    out[r[k], cc[k]] = True
    return out


def boundary_f1(p, g, ts=np.linspace(1.05, 1.25, 10)):
    """D-26 검증 경계 지표: Depth Pro (Bochkovskii et al. 2025) 의 scale-invariant boundary F1. 역깊이에서 이웃 픽셀 비 > t 를 가림 경계로 (4 방향),
    예측 경계는 그 방향 묶음마다 최댓값만, 방향마다 정밀도·재현율 → 평균 → F1, t = 1.05–1.25 열 개 (가중치 ∝ t).
    정답이 없는 픽셀이 낀 짝은 뺀다 (공식은 정답이 다 있는 합성 데이터용). p·g = GT 크기 numpy (m)."""
    ip, ig, v = 1 / np.clip(p, 1e-6, None), 1 / np.clip(g, 1e-6, None), g > 0
    pairs = []                                                        # (예측 비, 정답 비, valid, 방향이 세로인가)
    for a_, b_, vv, vert in ((ip[:, 1:] / ip[:, :-1], ig[:, 1:] / ig[:, :-1], v[:, 1:] & v[:, :-1], False),
                             (ip[1:, :] / ip[:-1, :], ig[1:, :] / ig[:-1, :], v[1:, :] & v[:-1, :], True)):
        pairs += [(a_, b_, vv, vert), (1 / a_, 1 / b_, vv, vert)]
    f = []
    for t in ts:
        P, R = [], []
        for a_, b_, vv, vert in pairs:
            cp = (_nms_rows((a_ > t).T, a_.T).T if vert else _nms_rows(a_ > t, a_)) & vv
            cg = (b_ > t) & vv
            tp = np.count_nonzero(cp & cg)
            P.append(tp / max(np.count_nonzero(cp), 1))
            R.append(tp / max(np.count_nonzero(cg), 1))
        pr, rc = np.mean(P), np.mean(R)
        f.append(0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc))
    return float(np.sum(np.array(f) * ts / ts.sum()))


def edge_dbe(p, g, theta=10):
    """D-26 검증 경계 지표 (판정용): iBims-1 DBE (Koch et al.) 식, analysis/boundary_check.dbe 와 같은 Canny (0–1 정규화 깊이, σ = √2, 문턱 0.1/0.2)·θ = 10 px.
    검증 세트에는 손으로 그린 경계가 없어 정답 경계도 정답 깊이에서 같은 Canny 로 (빈 곳은 가장 가까운 값으로 채우고, 빈 곳 2 px 안 경계는 뺌).
    → (ε_acc, ε_comp) px, 낮을수록 좋음 (ε_comp = 정답 경계를 빠짐없이 내는가 — 번지면 커짐). 정답 경계가 없으면 nan."""
    from scipy import ndimage
    from skimage import feature
    v = g > 0
    gf = g[tuple(ndimage.distance_transform_edt(~v, return_distances=False, return_indices=True))] if not v.all() else g
    keep = ndimage.distance_transform_edt(v) > 2

    def edges(x):
        x = (x - x.min()) / max(x.max() - x.min(), 1e-9)
        return feature.canny(x, sigma=np.sqrt(2), low_threshold=0.1, high_threshold=0.2) & keep
    eg, ep = edges(gf.astype(np.float64)), edges(p.astype(np.float64))
    if not eg.any():
        return float("nan"), float("nan")
    Dg, Dp = ndimage.distance_transform_edt(~eg), ndimage.distance_transform_edt(~ep)
    near = ep & (Dg < theta)
    return (float(Dg[near].mean()) if near.any() else float(theta)), float(np.minimum(Dp[eg], theta).mean())


def silog_eval(pred, gt, cap):
    """검증 지표: (이미지 SILog (λ 1, ×100), 사진 log 평균 오차 = 배율 ×100). 예측은 GT 크기로 bilinear(align_corners=True)·log 에서만 cap."""
    p = Fn.interpolate(pred[None, None], size=gt.shape, mode="bilinear", align_corners=True)[0, 0]
    v = gt > 0
    d = torch.log(p[v].clamp(*cap)) - torch.log(gt[v])
    return float(100 * torch.sqrt(torch.clamp((d ** 2).mean() - d.mean() ** 2, min=0))), float(100 * d.mean())


# ───────────────────────── 체크포인트 ─────────────────────────
def save_atomic(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def save_ckpt(state):
    last, prev = os.path.join(CK, "last.pt"), os.path.join(CK, "prev.pt")
    torch.save(state, last + ".tmp")
    if os.path.exists(last):
        os.replace(last, prev)
    os.replace(last + ".tmp", last)


def load_ckpt():
    for n in ("last.pt", "prev.pt"):
        p = os.path.join(CK, n)
        if os.path.exists(p):
            try:
                return torch.load(p, map_location="cpu", weights_only=False), n
            except Exception as e:
                print(f"!!! {n} 읽기 실패 ({e}) — 다음 후보", flush=True)
    return None, None


# ───────────────────────── 학습 ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", nargs="+", default=["control", "proposed"], choices=list(ENC_OF))
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--val_every", type=int, default=10000)
    ap.add_argument("--ckpt_every", type=int, default=2000)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--expect", type=int, nargs=2, default=[100000, 600], help="학습·검증 목록 장 수 (스모크 목록이 남아 있으면 멈추게)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", type=int, default=0, help="N > 0: 섞은 순서의 앞 N 장·검증 데이터셋마다 2 장·평가 세트마다 3 장 (출력은 *_smoke)")
    ap.add_argument("--stop_after", type=int, default=0, help="이어 학습 시험용: 이 장 수까지 하고 저장 후 끝냄")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--enc_res", choices=["canonical", "native"], default="canonical", help="기하 인코더 입력 해상도 (D-14)")
    ap.add_argument("--tag", default="", help="출력 꼬리표 (체크포인트·로그·표 폴더/파일 이름에 붙음)")
    ap.add_argument("--mix", default="", help="D-21: 목록 꼬리표 (train_mix<mix>·val_mix<mix>.jsonl). 기존 목록 뒤에 붙은 줄은 기존 순서를 그대로 두고 고르게 끼움")
    ap.add_argument("--save_at", type=int, nargs="*", default=[], help="D-21: 이 장 수에서 헤드 가중치를 at_<장 수>_<헤드>.pt 로 따로 저장 (같은 걸음 비교용)")
    ap.add_argument("--init", default="", help="D-26: 모든 헤드를 이 가중치 (헤드 state_dict) 에서 시작 (이어 학습 실험)")
    ap.add_argument("--limit_train", type=int, default=0, help="D-26: 학습 순서의 앞 N 장만 (일정도 N 장 기준)")
    ap.add_argument("--edge_w", nargs="+", default=["0", "0"], help="D-26: 경계 손실 가중치 'GT DAV2' (팔 A hr_gt, 팔 B hr_dav2) 또는 종류=값 (D-27: dav2b=… dav2d=…)")
    ap.add_argument("--calib", type=int, default=0, help="D-26: 학습 순서 앞 N 장에서 시작 헤드의 SILog·경계 손실 크기만 재고 끝냄 (가중치 정하기)")
    ap.add_argument("--crop_aug", type=float, nargs=2, default=None, metavar=("P", "S_MAX"),
                    help="D-62: 학습 사진만 확률 P 로 변 1/s (s ~ logU[1, S_MAX]) 상자를 VLM · 정답 · 교사 · 전문가 입력에 같이 자름 (기본 없음 = 예전과 같음)")
    a = ap.parse_args()
    global CK, OUT, ENC_RES, HR
    ENC_RES = a.enc_res
    HR = any(k in HR_HEADS for k in a.heads)
    CK, OUT = CK + a.tag, OUT + a.tag
    if a.smoke:
        CK, OUT = CK + "_smoke", OUT + "_smoke"
        a.val_every, a.ckpt_every, a.log_every = max(a.accum, a.smoke // 2), max(a.accum, a.smoke // 4), max(1, a.smoke // 4)
    os.makedirs(CK, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    assert a.ckpt_every % a.accum == 0 and a.val_every % a.accum == 0

    train = [json.loads(l) for l in open(os.path.join(ANN, f"train_mix{a.mix}.jsonl"))]
    val = [json.loads(l) for l in open(os.path.join(ANN, f"val_mix{a.mix}.jsonl"))]
    order = np.random.default_rng(0).permutation(len(train))          # 학습 순서 = seed 0 으로 한 번 섞은 것
    if a.mix:                                                         # D-21: 기존 목록 (앞 n_old 줄) 의 순서는 그대로, 붙은 줄 (seed 1 섞음) 을 고르게 끼움
        n_old = sum(1 for _ in open(os.path.join(ANN, "train_mix.jsonl")))
        assert all(r["dataset"] != "scannetpp" for r in train[:n_old]) and all(r["dataset"] == "scannetpp" for r in train[n_old:])
        order = interleave(np.random.default_rng(0).permutation(n_old), n_old + np.random.default_rng(1).permutation(len(train) - n_old))
    if a.limit_train:
        order = order[:a.limit_train]                                 # D-26: 앞 N 장 (DAv2 교사도 이 장들만 계산해 둠)
    if a.smoke:
        order = order[:a.smoke]                                       # 섞은 순서에서 뽑아 모든 데이터셋·crop 경로를 지나게
        val = [r for d in sorted({r["dataset"] for r in val}) for r in [x for x in val if x["dataset"] == d][:2]]
    else:
        print("학습", dict(Counter(r["dataset"] for r in train)), "검증", dict(Counter(r["dataset"] for r in val)), flush=True)
        if [len(train), len(val)] != a.expect:
            raise SystemExit(f"!!! 목록 장 수 {len(train)}·{len(val)} ≠ 기대 {a.expect} — build_mix.py 결과가 맞는지 확인")
    n = len(order)
    sig = dict(heads=a.heads, accum=a.accum, warmup=a.warmup, n_train=n, n_val=len(val), train_head=[train[i]["image"] for i in order[:5]])
    if a.enc_res != "canonical":                                      # 해결 실험 2 체크포인트와의 호환을 위해 기본값일 때는 서명에 넣지 않음
        sig["enc_res"] = a.enc_res
    if a.mix:
        sig["mix"] = a.mix
    if a.init:                                                        # D-26 (기본값이면 서명에 안 넣어 기존 체크포인트와 호환)
        sig.update(init=a.init, edge_w=a.edge_w)
    elif any(h in EDGE_OF for h in a.heads):                          # D-29: 처음부터 경계 손실 — 가중치도 서명에 (이어 학습 때 바뀌면 멈추게)
        sig["edge_w"] = a.edge_w
    if a.crop_aug:                                                    # D-62 (기본값이면 서명에 안 넣음)
        sig["crop_aug"] = list(a.crop_aug)
    done_path = os.path.join(CK, "train_done.json")
    if a.resume and not a.eval_only and os.path.exists(done_path):    # 학습이 끝난 뒤 죽었으면 평가만 다시 (같은 설정일 때만)
        if json.load(open(done_path)).get("sig") != sig:
            raise SystemExit("!!! train_done.json 의 설정이 지금 목록·설정과 다름")
        a.eval_only = True

    encs = sorted({ENC_OF[h] for h in a.heads if ENC_OF[h]})
    need_vlm = any(h not in VLM_FREE for h in a.heads)
    proc, base, features = load_models(encs, vlm=need_vlm)
    torch.manual_seed(0)                                              # 주입 Linear 초기화 고정 (이어 학습은 저장된 값을 씀)
    heads = {k: Head(base, ENC_OF[k] is not None, vlm=k not in VLM_FREE, hr=k in HR_HEADS, rgb=k in RGB_HEADS, center=k in CEN_HEADS).to(DEV) for k in a.heads}
    assert not (VS_HEADS & VLM_FREE & set(a.heads)) or need_vlm, "D-59: VLM 없는 VS 헤드 (vs_udv2only) 는 크기 기준 (원래 DepthVLM 예측) 이 필요 — VLM 쓰는 헤드와 같이"
    if a.init:                                                        # D-26: 같은 가중치에서 출발 (이어 학습 체크포인트가 있으면 아래에서 덮음)
        sd = torch.load(a.init, map_location="cpu")
        for k, hd in heads.items():                                   # D-30: 고해상도 경로가 없는 헤드 (v1) 는 그 키 (hproj·hgate) 만 빼고, 나머지는 strict
            extra = {n for n in sd if n not in hd.state_dict()}       # D-31 인코더 표 헤드: control 은 주입 키 전부 (proj·gate 도), udv2only 는 gate 도 (지킬 VLM 신호가 없어 게이트 없음)
            assert all(n.startswith(("hproj.", "proj.")) or n in ("hgate", "gate") for n in extra), (k, sorted(extra)[:5])
            if k in RGB_HEADS:                                        # D-59 ②: 사진 경로는 처음 상태에 없음 → 그 키만 빠진 것 허용 (나머지는 strict 와 같은 점검)
                r = hd.load_state_dict({n: v for n, v in sd.items() if n not in extra}, strict=False)
                assert not r.unexpected_keys and r.missing_keys and all(n.startswith(("rconv.", "rproj.")) or n == "rgate" for n in r.missing_keys), (k, r)
            elif k in CEN_HEADS:                                      # D-66: center_flag 버퍼는 처음 상태에 없음 → 그 키만 빠진 것 허용
                r = hd.load_state_dict({n: v for n, v in sd.items() if n not in extra}, strict=False)
                assert not r.unexpected_keys and r.missing_keys == ["center_flag"], (k, r)
            else:
                hd.load_state_dict({n: v for n, v in sd.items() if n not in extra})
        ref = {}
        for k, hd in heads.items():                                   # D-30: 모든 헤드의 공통 가중치가 똑같은 처음 상태인지
            for nm, v in hd.state_dict().items():
                assert torch.equal(v, ref.setdefault(nm, v)), (k, nm)
    use_dv = any(EDGE_OF.get(k, "").startswith("dav2") for k in heads)
    w_edge = ({k: float(v) for k, v in (x.split("=") for x in a.edge_w)} if all("=" in x for x in a.edge_w)
              else dict(zip(("gt", "dav2"), map(float, a.edge_w))))
    if a.calib:                                                       # D-26: 시작 헤드 (모든 헤드가 같음) 의 손실 크기 — 학습 사진만
        hd, rows = next(iter(heads.values())), []
        with torch.no_grad():
            for it in loader(MixSet(train, ENC_RES, dav2=True, rgb=True), order[:a.calib], a.workers):
                if it.get("bad"):
                    continue
                _, feats, geo, (ph, pw), *gh = features(it["inp"], it["x14"], need_p0=False, hr=HR)
                p = predict(hd, feats, geo.get("udv2"), ph, pw, (gh[0] if gh else {}).get("udv2"))
                g = it["g"].to(DEV)
                sl, gl = official_loss(p, g), grad_loss(p, g) if it["ds"] in DENSE else None
                el = egssi_loss(p, it["dv"].to(DEV)) if "dv" in it else None
                eb = egssi_loss(p, it["dv"].to(DEV), abs_thr=EDGE_ABS, sky=EDGE_SKY) if "dv" in it else None   # D-27 새 경계 선택
                rows.append(dict(ds=it["ds"], silog=None if sl is None else float(sl), grad=None if gl is None else float(gl), egssi=None if el is None else float(el),
                                 egssi_new=None if eb is None else float(eb)))
                if it["ds"] in DENSE:                                 # D-31: 기존 경계 손실 · 분리 두 가지 (조밀 정답 사진만 — 학습에서도 거기에만 검)
                    for kd in ("midas", "rpnl", "rpnl_m", "egssi"):
                        if kd == "rpnl":
                            rs = np.random.get_state()
                        if kd == "rpnl_m":                            # 공식 판과 같은 무작위 조각으로, 뒤 난수 (egssi) 는 rpnl_m 이 없을 때와 같게
                            rs2 = np.random.get_state()
                            np.random.set_state(rs)
                        v = edge_baseline(kd, p, g, it["rgb"])
                        if kd == "rpnl_m":
                            np.random.set_state(rs2)
                        rows[-1][kd] = None if v is None else float(v)
                    if "dv" in it:
                        for kd in ("dav2d_gt", "dav2d_rgb"):
                            v = edge_split(kd, p, g, it["dv"].to(DEV), it["rgb"])
                            rows[-1][kd] = None if v is None else float(v)
        t = pd.DataFrame(rows)
        print(t.groupby("ds")[["silog", "grad", "egssi", "egssi_new"]].mean().round(4).to_string(), flush=True)
        sd_, sa = t[t.ds.isin(DENSE)].silog.mean(), t.silog.mean()
        print(f"[calib] {len(t)} 장: SILog 전체 {sa:.4f} · 조밀 {sd_:.4f}, grad (조밀) {t.grad.mean():.4f}, egssi {t.egssi.mean():.4f} → "
              f"시작 때 경계 항 = SILog 의 절반이 되는 가중치: GT {0.5 * sd_ / t.grad.mean():.3f}, DAV2 {0.5 * sa / t.egssi.mean():.3f}", flush=True)
        td = t[t.ds.isin(DENSE)]
        print(f"[calib D-27] 새 선택 egssi: 전체 {t.egssi_new.mean():.4f} (없는 사진 {t.egssi_new.isna().sum()}) · 조밀 {td.egssi_new.mean():.4f} → "
              f"가중치 dav2b {0.5 * sa / t.egssi_new.mean():.3f} · dav2d {0.5 * sd_ / td.egssi_new.mean():.3f}", flush=True)
        ks = [kd for kd in ("midas", "rpnl", "rpnl_m", "egssi", "dav2d_gt", "dav2d_rgb") if kd in td]
        if ks:                                                        # D-31: 시작 때 (경계 항 / 조밀 SILog) 비율 r → 가중치 = r × 조밀 SILog / 손실 (r 는 우리 주 λ 의 비율)
            r = {f"{lam:g}": lam * td.egssi_new.mean() / sd_ for lam in (0.18, 0.36)}
            print(f"[calib D-31] 조밀 {len(td)} 장 손실 평균: " + " · ".join(f"{kd} {td[kd].mean():.4f}" for kd in ["egssi_new"] + ks) +
                  " | 우리 λ 의 비율 r " + " · ".join(f"λ {lam} → {v:.3f}" for lam, v in r.items()) + " | 같은 r 의 가중치 " +
                  " / ".join(f"(r {v:.3f}) " + " ".join(f"{kd} {v * sd_ / td[kd].mean():.4f}" for kd in ks) for v in r.values()), flush=True)
        return
    if a.eval_only:
        evaluate(proc, features, heads, a)
        return

    total = math.ceil(n / a.accum)
    sched = lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))
    groups = {k: [(hd.h.parameters(), 2e-5)] + ([(hd.proj.parameters(), 1e-4)] if hd.proj is not None else []) +
              ([([hd.gate], 1e-3)] if hd.gate is not None else []) +
              ([(hd.hproj.parameters(), 1e-4), ([hd.hgate], 1e-3)] if hd.hproj is not None else []) +
              ([(list(hd.rconv.parameters()) + list(hd.rproj.parameters()), 1e-4), ([hd.rgate], 1e-3)] if hd.rconv is not None else [])   # D-59 ② 사진 경로 = 같은 규칙
              for k, hd in heads.items()}   # 고해상도 경로 = proj·gate 와 같은 lr
    opts = {k: torch.optim.AdamW([{"params": list(ps), "lr": lr} for ps, lr in g], weight_decay=0.0) for k, g in groups.items()}
    base_lr = {k: [g["lr"] for g in o.param_groups] for k, o in opts.items()}

    st = dict(pos=0, step=0, best={k: (float("inf"), -1) for k in heads}, cnt=dict(bad=0, nan=0, oom=0, nan_grad=0), log=[], vlog=[])
    best_state = {k: None for k in heads}                             # 최고 상태는 체크포인트 안에도 둔다 → best_<헤드>.pt 와 st 가 항상 맞음
    if a.resume:
        ck, name = load_ckpt()
        if ck is None:
            print("이어 학습할 체크포인트 없음 — 처음부터", flush=True)
        else:
            if ck["sig"] != sig:
                raise SystemExit(f"!!! 체크포인트 설정이 다름: {ck['sig']} vs {sig}")
            for k in heads:
                heads[k].load_state_dict(ck["heads"][k])
                opts[k].load_state_dict(ck["opts"][k])
            st, best_state = ck["st"], ck["best_state"]
            torch.set_rng_state(ck["rng"]["torch"])
            torch.cuda.set_rng_state(ck["rng"]["cuda"])
            np.random.set_state(ck["rng"]["np"])
            random.setstate(ck["rng"]["py"])
            for k, s in best_state.items():                           # 끊기기 전 검증이 best 파일만 바꿔 놓았을 수 있어 체크포인트 쪽으로 맞춤
                if s is not None:
                    save_atomic(s, os.path.join(CK, f"best_{k}.pt"))
            print(f"이어 학습: {name}, {st['pos']} / {n} 장, 걸음 {st['step']}, 건너뜀 {st['cnt']}", flush=True)

    def finite():
        return [k for k, h in heads.items() if not all(bool(torch.isfinite(q).all()) for q in h.parameters())]

    def abort(msg):
        """복구할 수 없는 문제 → last/prev 는 건드리지 않고 abort.pt 에 남기고 종료 코드 3 (유닛은 RestartPreventExitStatus=3 로 재시작 안 함)."""
        torch.save(dict(sig=sig, st=st, heads={k: h.state_dict() for k, h in heads.items()}), os.path.join(CK, "abort.pt"))
        print(f"!!! 중단: {msg} — last/prev 는 그대로, 상태는 abort.pt", flush=True)
        sys.exit(3)

    def snapshot():
        bad = finite()
        if bad:
            abort(f"헤드 {bad} 가중치에 NaN/inf")
        save_ckpt(dict(sig=sig, st=st, best_state=best_state, heads={k: h.state_dict() for k, h in heads.items()},
                       opts={k: o.state_dict() for k, o in opts.items()},
                       rng=dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(), np=np.random.get_state(), py=random.getstate())))
        for k, s in best_state.items():
            if s is not None and st["best"][k][1] == st["pos"]:           # 이번 검증에서 새로 좋아진 헤드만 다시 씀
                save_atomic(s, os.path.join(CK, f"best_{k}.pt"))

    stop = {"sig": None}
    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, lambda num, frm: stop.update(sig=num))
    t0, tlog, run, run_e = time.time(), time.time(), {k: [] for k in heads}, {k: [] for k in heads}
    lvl = {k: [] for k in heads if k in VS_HEADS | VD_HEADS}           # D-52 · D-53: 헤드 자체 전역 크기 − VLM (ln, 사진 전체) — 학습 중 떠다님 감시 (기록만)
    need_vs = bool(lvl)                                               # D-52 · D-53: 그 헤드가 있을 때만 학습 중 원래 DepthVLM 예측을 계산 (없으면 지금과 똑같음)
    bf1_on = bool({"hr_ctrl", "hr_gt", "hr_dav2", "hr_dav2b", "hr_dav2d", "hr_dav2d_lo", "hr_dav2d_hi", "v1_ctrl", "v1_dav2d", "hr_dav2d_hi2", "hr_midas", "hr_rpnl", "hr_rpnl_m", "hr_egssi", "hr_dav2d_gt", "hr_dav2d_rgb", "vd_dav2d_hi2"} & set(heads))   # D-26: 조밀 정답 검증 세트에서 경계 F1 도 (다른 실행의 검증은 그대로) · D-53 vd 단독 실행도 사진별 기록
    bf1_on = bf1_on or bool(D59_HEADS & set(heads))                   # D-59: 사진별 검증 기록 (vs_l036 과 짝 비교)
    bf1_on = bf1_on or bool(D66_HEADS & set(heads))                   # D-66: 같은 사진별 검증 기록
    bf1_on = bf1_on or "kr_dav2d" in heads                            # D-84: 같은 사진별 검증 기록 (control 과 짝 비교)
    use_rgb = bool(RGB_HEADS & set(heads))                            # D-59 ②: 사진 경로 헤드가 있을 때만 canonical 사진을 읽음
    use_tok = bool(TOK_HEADS & set(heads))                            # D-66: 토큰 격자 전문가 헤드가 있을 때만 x14t 를 만듦
    tiel = {k: [] for k in heads if k in TIE_HEADS}                   # D-66: 묶기 항 크기 (기록)

    def bf1(pred, g):
        """(Depth Pro 경계 F1 — 기록만, DBE ε_acc, DBE ε_comp — 판정용) — 예측은 GT 크기로 bilinear(align_corners=True)."""
        p = Fn.interpolate(pred[None, None], size=g.shape, mode="bilinear", align_corners=True)[0, 0].float().cpu().numpy()
        gn = g.cpu().numpy()
        return (boundary_f1(p, gn), *edge_dbe(p, gn))

    def validate():
        res = {k: {} for k in list(heads) + (["DepthVLM"] if need_vlm else [])}   # VLM 없는 실행은 원래 DepthVLM 줄 없음 (값은 다른 실행과 같음)
        bres, vimg = {k: {} for k in res}, []
        for k in heads:
            heads[k].eval()
        with torch.no_grad():
            for it in loader(MixSet(val, ENC_RES, rgb=use_rgb, tok=use_tok), range(len(val)), 0):   # D-30: 본 프로세스에서 읽음 — 검증마다 작업자를 새로 fork 하면 커진 본 프로세스의 copy-on-write 가 쌓여 메모리 초과 (10-17 10:10)
                if it.get("bad"):
                    continue
                p0, feats, geo, (ph, pw), *gh = features(it["inp"], it["x14"], hr=HR, x14t=it.get("x14t"))
                ghr = gh[0] if gh else {}
                g = it["g"].to(DEV)
                c = (DATASET_CONFIGS[it["ds"]]["min_depth"], DATASET_CONFIGS[it["ds"]]["max_depth"])
                rt = rgb_input(it["rgb"]) if use_rgb else None                # D-59 ②
                preds = {"DepthVLM": p0} if p0 is not None else {}
                preds.update({k: predict(hd, feats, geo.get(ENC_OF[k]), ph, pw, ghr.get(ENC_OF[k]), rt if k in RGB_HEADS else None) for k, hd in heads.items()})
                preds.update({k: vlm_scale(preds[k], p0) for k in heads if k in VS_HEADS})   # D-52
                for k, pk in preds.items():
                    s = silog_eval(pk, g, c)
                    res[k].setdefault(it["ds"], []).append(s)
                    b = bf1(pk, g) if bf1_on and it["ds"] in DENSE else (float("nan"),) * 3
                    if bf1_on and it["ds"] in DENSE:
                        bres[k].setdefault(it["ds"], []).append(b)
                    if bf1_on:
                        vimg.append(dict(pos=st["pos"], model=k, ds=it["ds"], i=it["i"], silog=s[0], bias=s[1], bf1=b[0], eacc=b[1], ecomp=b[2]))
        for k in heads:
            heads[k].train()
        if bf1_on:                                                    # D-26: 사진별 값 (팔끼리 같은 사진 bootstrap 용)
            pd.DataFrame(vimg).to_csv(os.path.join(OUT, f"val_images_{st['pos']}.csv"), index=False)
        row = dict(pos=st["pos"], time=time.time() - t0)
        for k, r in res.items():
            per = {ds: float(np.mean([x[0] for x in v])) for ds, v in sorted(r.items())}
            row[k] = float(np.mean([v for ds, v in per.items() if ds not in MONITOR]))   # 고르기 기준 = 데이터셋 평균 SILog (기록만 하는 세트 제외)
            row[f"{k}/scale_rms"] = float(np.sqrt(np.mean([x[1] ** 2 for ds, v in r.items() if ds not in MONITOR for x in v])))   # 보기용 (F-10 같은 크기 쏠림)
            row.update({f"{k}/{ds}": v for ds, v in per.items()})
            row.update({f"{k}/{ds}/bias": float(np.mean([x[1] for x in v])) for ds, v in sorted(r.items())})   # D-21: 데이터셋별 쏠림 (사진 평균 log 오차의 평균 ×100)
            if bres[k]:                                               # D-26: 경계 지표 = 조밀 세트 평균 (데이터셋마다 평균 뒤 평균, 정답 경계 없는 사진 nan 제외)
                for j, nm in enumerate(("bf1", "eacc", "ecomp")):
                    row.update({f"{k}/{ds}/{nm}": float(np.nanmean([x[j] for x in v])) for ds, v in sorted(bres[k].items())})
                    row[f"{k}/{nm}"] = float(np.mean([np.nanmean([x[j] for x in v]) for v in bres[k].values()]))
        st["vlog"].append(row)
        for k in heads:
            if row[k] < st["best"][k][0]:
                st["best"][k] = (row[k], st["pos"])
                best_state[k] = {nm: q.detach().cpu().clone() for nm, q in heads[k].state_dict().items()}
        mon = " ".join(f"{k}/{ds} SILog {row[f'{k}/{ds}']:.2f} 쏠림 {row[f'{k}/{ds}/bias']:+.2f}" for k in res for ds in sorted(MONITOR) if f"{k}/{ds}" in row)
        print(f"[val {st['pos']}] " + " ".join(f"{k} {row[k]:.3f} (배율 {row[f'{k}/scale_rms']:.1f}" +
                                               (f", ε_comp {row[f'{k}/ecomp']:.3f} ε_acc {row[f'{k}/eacc']:.3f} F1 {row[f'{k}/bf1']:.4f}" if f"{k}/bf1" in row else "") + ")"
                                               for k in res) + (f" | 기록만: {mon}" if mon else "") +
              f" | 최고 {({k: (round(v[0], 3), v[1]) for k, v in st['best'].items()})}", flush=True)
        pd.DataFrame(st["vlog"]).to_csv(os.path.join(OUT, "val_log.csv"), index=False)

    # D-30: 학습 작업자를 0 장 검증보다 먼저 fork — 검증을 본 프로세스에서 읽게 바꾼 뒤로는, 본 프로세스가 사진 처리에 스레드를 쓴 다음 fork 한 작업자가
    # 교착 (스모크: 작업자 8 개 모두 futex 대기). 먼저 만들면 학습 끝까지 더 fork 하지 않음.
    train_it = iter(loader(MixSet(train, ENC_RES, dav2=use_dv, rgb=any(EDGE_OF.get(k) in RGB_EDGE for k in heads) or use_rgb,
                                  crop_aug=(*a.crop_aug, 0) if a.crop_aug else None, tok=use_tok), order[st["pos"]:], a.workers))   # D-62: 학습만 자름 · D-66 토큰 격자 입력
    if st["pos"] == 0:
        validate()                                                    # 0 장 = 원래 DepthVLM 헤드 (VLM 쓰는 헤드는 모두 같은 값, udv2only 는 무작위 Linear 라 다름)
        snapshot()
    for k in heads:
        heads[k].train()
    recent = deque(maxlen=2000)                                       # 최근 사진 실패 (읽기·NaN) — 저장하지 않음 (재시작하면 새로 셈)
    oom_recent = deque(maxlen=250)                                    # 최근 누적 묶음의 OOM 여부
    group_bad = False                                                 # 이 누적 묶음에서 OOM → 모든 헤드 기울기를 버리고 걸음을 건너뜀 (헤드끼리 같은 사진만 배우게)
    for it in train_it:
        st["pos"] += 1
        fail = 0
        if it.get("bad"):
            st["cnt"]["bad"] += 1
            fail = 1
        elif not group_bad:
            try:
                p0, feats, geo, (ph, pw), *gh = features(it["inp"], it["x14"], need_p0=need_vs, hr=HR, x14t=it.get("x14t"))   # D-52: VS 헤드가 있을 때만 원래 예측 (크기 기준)
                ghr = gh[0] if gh else {}
                g = it["g"].to(DEV)
                rt = rgb_input(it["rgb"]) if use_rgb else None            # D-59 ②
                for k, hd in heads.items():                           # GT 가 비면 모든 헤드가 같이 None (예측은 Softplus 라 > 0)
                    pred = predict(hd, feats, geo.get(ENC_OF[k]), ph, pw, ghr.get(ENC_OF[k]), rt if k in RGB_HEADS else None)
                    if k in VS_HEADS:                                 # D-52: 크기는 VLM, 정답은 모양만 (√Var)
                        lvl[k].append(float(torch.log(pred.detach().clamp(min=1e-3)).mean() - torch.log(p0.clamp(min=1e-3)).mean()))   # 헤드 자체 크기의 떠다님 (기록용)
                        pred = vlm_scale(pred, p0)
                    if k in VD_HEADS:                                 # D-53: 출력은 그대로, 크기 기준만 원래 DepthVLM 예측 (정답 있는 픽셀)
                        lvl[k].append(float(torch.log(pred.detach().clamp(min=1e-3)).mean() - torch.log(p0.clamp(min=1e-3)).mean()))   # 헤드 자체 크기 − VLM (사진 전체, 기록용)
                    loss = vlm_target_loss(pred, g, p0) if k in VD_HEADS else official_loss(pred, g, lam=1.0 if k in VS_HEADS else 0.5)
                    if loss is None or not torch.isfinite(loss):
                        st["cnt"]["nan"] += 1
                        fail = 1
                        continue
                    run[k].append(float(loss))                        # 기록 = SILog 만 (팔끼리 같은 눈금)
                    e = None                                          # D-26 경계 손실: 팔 A 는 조밀 정답 데이터셋만, 팔 B 는 교사가 있는 사진 모두
                    if EDGE_OF.get(k) == "gt" and it["ds"] in DENSE:
                        e = grad_loss(pred, g)
                    elif EDGE_OF.get(k) == "dav2" and "dv" in it:
                        e = egssi_loss(pred, it["dv"].to(DEV))
                    elif EDGE_OF.get(k) in ("dav2b", "dav2d") and "dv" in it and (EDGE_OF[k] == "dav2b" or it["ds"] in DENSE):   # D-27
                        e = egssi_loss(pred, it["dv"].to(DEV), abs_thr=EDGE_ABS, sky=EDGE_SKY)
                    elif EDGE_OF.get(k) in ("midas", "rpnl", "rpnl_m", "egssi") and it["ds"] in DENSE:   # D-31: 기존 경계 손실 (공식 코드), 정답 크기에서
                        e = edge_baseline(EDGE_OF[k], pred, g, it.get("rgb"))
                    elif EDGE_OF.get(k) in ("dav2d_gt", "dav2d_rgb") and "dv" in it and it["ds"] in DENSE:   # D-31 분리 ①·②
                        e = edge_split(EDGE_OF[k], pred, g, it["dv"].to(DEV), it.get("rgb"))
                    if e is not None and torch.isfinite(e):
                        loss = loss + w_edge.get(k, w_edge.get(EDGE_OF[k])) * e   # D-29: 헤드 이름=값 이 있으면 그것 (λ 민감도), 없으면 종류=값
                        run_e[k].append(float(e))
                    if k in TIE_HEADS:                                # D-66: 전문가를 끈 같은 헤드 (학습 때만) — 공식 손실로 배우고, 켠 헤드의 사진 log 배율을 거기에 묶음
                        poff = predict(hd, feats, None, ph, pw, None, inject=False)
                        lo = official_loss(poff, g)
                        tie = (torch.log(pred.clamp(min=1e-3)) - torch.log(poff.detach().clamp(min=1e-3))).median().abs()
                        if lo is not None and torch.isfinite(lo) and torch.isfinite(tie):
                            loss = loss + lo + W_TIE * tie
                            tiel[k].append(float(tie))
                    (loss / a.accum).backward()
            except torch.cuda.OutOfMemoryError:
                st["cnt"]["oom"] += 1
                group_bad = True
                feats = geo = g = loss = None
                for o in opts.values():
                    o.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(f"!!! CUDA OOM — 사진 {it['i']} ({it.get('ds')}): 이 누적 묶음을 모든 헤드에서 버리고 30 초 쉼", flush=True)
                time.sleep(30)
        recent.append(fail)
        if st["pos"] % a.accum == 0:
            for k, o in opts.items():
                if group_bad:
                    o.zero_grad(set_to_none=True)
                    continue
                for gi, gr in enumerate(o.param_groups):
                    gr["lr"] = base_lr[k][gi] * sched(st["step"])
                gn = torch.nn.utils.clip_grad_norm_([q for gr in o.param_groups for q in gr["params"]], 1.0)
                if not torch.isfinite(gn):                            # 기울기가 inf/NaN 이면 그 헤드의 이 걸음만 건너뜀
                    st["cnt"]["nan_grad"] += 1
                    o.zero_grad(set_to_none=True)
                    continue
                o.step()
                o.zero_grad(set_to_none=True)
            oom_recent.append(int(group_bad))
            group_bad = False
            st["step"] += 1
            if st["pos"] in a.save_at:                                # D-21: 이 장 수의 가중치를 따로 (이어 학습으로 다시 지나가도 같은 값)
                for k, hd in heads.items():
                    save_atomic(hd.state_dict(), os.path.join(CK, f"at_{st['pos']}_{k}.pt"))
                print(f"[save_at] {st['pos']} 장 가중치 저장", flush=True)
        if len(recent) >= 200 and sum(recent) > 0.05 * len(recent):
            abort(f"최근 {len(recent)} 장 중 {sum(recent)} 장 실패 (읽기·NaN) {st['cnt']}")
        if len(oom_recent) >= 50 and sum(oom_recent) > 0.2 * len(oom_recent):
            print(f"!!! 최근 {len(oom_recent)} 누적 묶음 중 {sum(oom_recent)} 개 OOM — GPU 를 다른 작업이 쓰는 듯. 종료 코드 4 (유닛이 5 분 뒤 이어 학습)", flush=True)
            sys.exit(4)
        if st["pos"] % a.log_every == 0:
            tl = time.time() - tlog
            row = dict(pos=st["pos"], step=st["step"], sec_per_img=tl / a.log_every, gpu_gb=torch.cuda.max_memory_allocated() / 1e9,
                       **{f"loss_{k}": float(np.mean(v)) if v else float("nan") for k, v in run.items()},
                       **{f"edge_{k}": float(np.mean(v)) for k, v in run_e.items() if v},
                       **{f"level_{k}": float(np.mean(v)) for k, v in lvl.items() if v},
                       **{f"tie_{k}": float(np.mean(v)) for k, v in tiel.items() if v},   # D-66
                       **{f"gate_{k}": [round(float(x), 4) for x in heads[k].gate] for k in heads if heads[k].gate is not None},
                       **{f"hgate_{k}": [round(float(x), 4) for x in heads[k].hgate] for k in heads if heads[k].hgate is not None},
                       **{f"rgate_{k}": round(float(heads[k].rgate), 4) for k in heads if heads[k].rgate is not None})   # D-59 ②
            st["log"].append(row)
            eta = (n - st["pos"]) * tl / a.log_every / 3600
            print(f"train {st['pos']}/{n} " + " ".join(f"{k} {row[f'loss_{k}']:.4f}" + (f" (경계 {row[f'edge_{k}']:.4f})" if f"edge_{k}" in row else "") for k in heads) +
                  f" | {row['sec_per_img']:.2f} s/장, GPU 최대 {row['gpu_gb']:.1f} GB, 남은 {eta:.1f} h, 건너뜀 {st['cnt']}", flush=True)
            pd.DataFrame(st["log"]).to_csv(os.path.join(OUT, "train_log.csv"), index=False)
            run, run_e, tlog = {k: [] for k in heads}, {k: [] for k in heads}, time.time()
            lvl = {k: [] for k in lvl}
            tiel = {k: [] for k in tiel}
        if st["pos"] % a.val_every == 0 and st["pos"] < n:
            validate()
        if st["pos"] % a.ckpt_every == 0 or st["pos"] % a.val_every == 0:
            snapshot()
        if (stop["sig"] is not None or (a.stop_after and st["pos"] >= a.stop_after)) and st["pos"] % a.accum == 0:
            snapshot()                                                # 누적 경계(최대 accum − 1 장 더)에서 저장하고 끝냄
            print(f"멈춤 (신호 {stop['sig']}) — {st['pos']} 장에서 저장", flush=True)
            return
    validate()
    snapshot()
    json.dump(dict(sig=sig, pos=st["pos"], step=st["step"], cnt=st["cnt"], best=st["best"]), open(done_path, "w"))
    print(f"학습 끝: {st['pos']} 장, 걸음 {st['step']}, 건너뜀 {st['cnt']}, 고른 상태 {st['best']} ({(time.time() - t0) / 3600:.2f} h)", flush=True)
    a.eval_only = True                                                # best_<헤드>.pt 를 읽어 평가
    evaluate(proc, features, heads, a)


# ───────────────────────── 평가 ─────────────────────────
def stem(image):
    return image.rsplit(".", 1)[0].replace("/", "__")


def stats(p, g, cap):
    """vlm-depth-rmse weak_stats.dense_rows 와 같은 충분통계 (valid = GT > 0, log 에서만 예측을 cap 으로 자름) — geo_inject 와 같음."""
    v = g > 0
    pp, gg = p[v].astype(np.float64), g[v].astype(np.float64)
    e = pp - gg
    d = np.log(np.clip(pp, *cap)) - np.log(gg)
    return dict(n=int(v.sum()), se=float((e ** 2).sum()), ar=float((np.abs(e) / gg).sum()),
                d1=int(((pp > 0) & (np.maximum(pp / gg, gg / np.maximum(pp, 1e-12)) < 1.25)).sum()), dl=float(d.sum()), dl2=float((d ** 2).sum()))


def evaluate(proc, features, heads, a):
    if a.eval_only:
        for k in heads:
            heads[k].load_state_dict(torch.load(os.path.join(CK, f"best_{k}.pt"), map_location=DEV))
    for k in heads:
        heads[k].eval()
    recs = [r for r in json.load(open(os.path.join(D, "records_eval.json"))) if r["set"] in SETS]
    if a.smoke:
        recs = [r for s in SETS for r in [x for x in recs if x["set"] == s][:3]]
    res, t0 = [], time.time()
    for j, r in enumerate(recs):
        cap = CAP_OF[r["set"]]
        g = np.load(os.path.join(D, "gt", r["set"], stem(r["image"]) + ".npz"))["g"]
        path = os.path.join(BENCH, r["image"])
        W0, H0 = Image.open(path).size
        img = _load_and_resize_image(path, [round(W0 * 1000.0 / r["k"][0]), round(H0 * 1000.0 / r["k"][0])])   # dense_full.canonical 과 같음
        vi = vlm_inputs(proc, img)
        p0, feats, geo, (ph, pw), *gh = features(vi, native_input(path) if ENC_RES == "native" else enc_input(img), hr=HR,
                                                 x14t=tok_input(img, *tok_grid(vi)) if TOK_HEADS & set(heads) else None)   # D-66
        ghr = gh[0] if gh else {}
        preds = {} if p0 is None else {"DepthVLM": p0.cpu().numpy()}   # VLM 없는 실행 (D-17) 은 원래 예측·+VLMscale 없음
        with torch.no_grad():
            for k, hd in heads.items():
                pk = predict(hd, feats, geo.get(ENC_OF[k]), ph, pw, ghr.get(ENC_OF[k]), rgb_input(img) if k in RGB_HEADS else None)   # D-59 ② 사진 경로
                preds[k] = (vlm_scale(pk, p0) if k in VS_HEADS else pk).cpu().numpy()   # D-52
        preds = {k: cv2.resize(p, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR) for k, p in preds.items()}
        lm = lambda x: float(np.log(np.clip(x, *cap)).mean())
        for k in (list(heads) if "DepthVLM" in preds else []):   # 사후 보완 (F-11): 사진마다 원래 DepthVLM 의 log 평균 깊이에 맞춤 — 모든 픽셀, GT 안 씀
            preds[f"{k}+VLMscale"] = preds[k] * np.exp(lm(preds["DepthVLM"]) - lm(preds[k]))
        for k, p in preds.items():
            res.append(dict(set=r["set"], image=r["image"], model=k, **stats(p, g, cap)))
        if j % 200 == 0:
            print(f"eval {j}/{len(recs)} ({time.time() - t0:.0f}s)", flush=True)
    t = pd.DataFrame(res)
    tag = a.tag + ("_smoke" if a.smoke else "")
    t.to_parquet(os.path.join(OUT, f"per_image{tag}.parquet"), index=False)
    summary(t, list(heads), tag)


def summary(t, heads, tag=""):
    out, comp = [], []
    for s in SETS:
        x = t[t.set == s]
        if x.empty:
            continue
        imgs = sorted(x.image.unique())
        N = len(imgs)
        w = np.vstack([np.ones(N), np.random.default_rng(0).multinomial(N, np.full(N, 1 / N), B)])
        P = {m: x[x.model == m].set_index("image").loc[imgs] for m in x.model.unique()}
        sil = {}
        for m, y in P.items():
            nn_ = y.n.values
            mu = y.dl.values / nn_
            sil[m] = 100 * np.sqrt(np.maximum(y.dl2.values / nn_ - mu ** 2, 0))
            out.append(dict(set=s, model=m, n_img=N, silog=sil[m].mean(), rmse=np.sqrt(y.se.sum() / nn_.sum()), absrel=y.ar.sum() / nn_.sum(),
                            d1=y.d1.sum() / nn_.sum(), scale_rms=100 * np.sqrt((mu ** 2).mean())))
        pairs = [(h, "DepthVLM") for h in heads] + [(h, "control") for h in heads if h != "control"] + \
                [(f"{h}+VLMscale", "DepthVLM") for h in heads] + [(f"{h}+VLMscale", "control+VLMscale") for h in heads if h != "control"] + \
                ([("proposed", "dino")] if {"proposed", "dino"} <= set(heads) else [])
        for a_, b_ in [(x, y) for x, y in pairs if x in sil and y in sil]:   # 이번 실행에 있는 헤드끼리만 (대조군 1 없는 실행 = 해결 실험 3)
            dd = w @ (sil[a_] - sil[b_]) / N
            comp.append(dict(set=s, comp=f"{a_} − {b_}", d_silog=dd[0], lo=np.percentile(dd[1:], 2.5), hi=np.percentile(dd[1:], 97.5)))
    O, C = pd.DataFrame(out), pd.DataFrame(comp)
    O.to_csv(os.path.join(ROOT, "tables", f"geo_mix{tag}.csv"), index=False)
    C.to_csv(os.path.join(ROOT, "tables", f"geo_mix_comp{tag}.csv"), index=False)
    pd.set_option("display.width", 220)
    print(O.round(4).to_string(index=False))
    print(C.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
