#!/usr/bin/env bash
# 외부 코드 받기 (git 없이 GitHub 압축본): 공개 저장소 셋을 고정 커밋으로 <code_root> 아래에 풀고 setup/code.sha256 과 대조한다 (하나라도 다르면 실패).
#   vlm-depth-rmse     @0a1f2f9  평가 코드 (eval/common.py · dense_sparse.py) · 평가 목록 (bench/) · DepthVLM 공식 코드 사본 (third_party/DepthVLM)
#   depthvlm-finetune  @3a5c217  NYUv2 평가 목록 (eval/std_eval.py)
#   Depth-Anything-V2  @a561b84  → vlm-depth-rmse/ext/Depth-Anything-V2 (모델을 불러오는 함수가 import 만 한다 — 학습 · 평가에는 안 쓰임)
# 이미 맞게 받은 것은 건너뛴다 (.commit). 네트워크가 끊기면 30 초 쉬고 3 번까지.
# 사용: bash setup/fetch_code.sh <code_root>
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
C=${1:?"code_root 를 준다"}
mkdir -p "$C"; C=$(cd "$C" && pwd)
export PYTHONDONTWRITEBYTECODE=1

get() {  # 폴더 저장소 커밋 — 맨 위 폴더 (<저장소>-<커밋>/) 를 떼고 푼다
  [ "$(cat "$C/$1/.commit" 2>/dev/null)" = "$3" ] && { echo "[code] $1 @ ${3:0:7} (이미 있음)"; return 0; }
  local a
  for a in 1 2 3; do
    rm -rf "${C:?}/${1:?}"
    python3 -B - "$C/$1" "https://codeload.github.com/$2/tar.gz/$3" <<'PY' && break
import sys, tarfile, urllib.request
dst, url = sys.argv[1:]
with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "depthvlm-abl-h200"}), timeout=300) as r, tarfile.open(fileobj=r, mode="r|gz") as t:
    for m in t:
        parts = m.name.split("/", 1)
        if len(parts) < 2 or not parts[1] or not (m.isfile() or m.isdir()):
            continue
        m.name = parts[1]
        t.extract(m, dst, filter="data") if hasattr(tarfile, "data_filter") else t.extract(m, dst)
PY
    [ "$a" = 3 ] && { echo "!!! [code] $1 받기 실패 3 번 ($2 @ ${3:0:7})"; return 1; }
    echo "[code] $1 받기 실패 ($a/3) — 30 초 뒤 다시"; sleep 30
  done
  echo "$3" > "$C/$1/.commit"
  echo "[code] $1 @ ${3:0:7}"
}

get vlm-depth-rmse johnhong06/vlm-depth-rmse 0a1f2f940b23c0bcaa93eb851e63bba111852b36
get depthvlm-finetune johnhong06/depthvlm-finetune 3a5c21792af4a29da23fbff6b96afe85f7dd0c09
mkdir -p "$C/vlm-depth-rmse/ext"
get vlm-depth-rmse/ext/Depth-Anything-V2 DepthAnything/Depth-Anything-V2 a561b849ebae10a6f5ef49e26c83cbbcd36c71bf

python3 -B - "$C" "$HERE/code.sha256" <<'PY'
import hashlib, os, sys
root, lst = sys.argv[1:]
want = [l.split(None, 1) for l in open(lst, encoding="utf-8") if l.strip() and not l.startswith("#")]
bad = []
for h, p in want:
    p = p.strip()
    f = os.path.join(root, p)
    if not os.path.isfile(f):
        bad.append("없음 " + p)
        continue
    with open(f, "rb") as fh:
        if hashlib.sha256(fh.read()).hexdigest() != h:
            bad.append("다름 " + p)
if bad:
    print(f"!!! [code] 받은 코드가 목록과 다름 {len(bad)} 개: " + " · ".join(bad[:5]))
    sys.exit(1)
print(f"[code] 대조 통과 — 파일 {len(want)} 개 (sha256)")
PY
