#!/usr/bin/env bash
# H200 진입점 (gpu-request 이슈의 실행 명령 한 줄) — DepthVLM-4B 의 읽기 전용 기하 토큰 ablation (논문 표 2 · 부록 표 S3) 세 판. 판마다 작업 하나, GPU 1 장.
#   bash run.sh check  [LINKS=…]          1–2 분: 파이썬 · GPU · CPU · RAM · 여유 공간 · 외부 접속 · LINKS 목록. 설치 안 함
#   bash run.sh smoke  LINKS=…            세 판의 흐름을 아주 작게 끝까지 (학습 순서 몇 줄 · 평가 세트마다 2 장) — 환경 · 코드 · 데이터 · 가중치 · 학습 · 평가 · 표
#   bash run.sh run1   LINKS=…            판 1  토큰 없이 DepthVLM 의 헤드만 다시 학습 · 정답만 · 1 배 (373,899 장) → 평가 6 세트
#   bash run.sh run2   LINKS=…            판 2  기하 토큰 · 정답만 · 2 배 (1 단계 647,994 장 → 2 단계 99,804 장) → 평가 6 세트
#   bash run.sh run3   LINKS=…            판 3  기하 토큰 · 정답 + 무라벨 · 2 배 (1 단계 647,994 장 → 2 단계 99,804 장) → 평가 6 세트
# KEY=값 (이슈 한 줄 명령에는 환경 변수를 못 넣으므로 인자로):
#   LINKS=<Drive 파일 ID>  데이터 조각을 받을 곳의 목록 파일. 데이터와 받는 주소는 저장소에 없다 — 이 값은 실행 명령에만 적는다 (/app/data 에 조각이 있으면 없어도 됨)
#   WORKERS=8 (학습 사진을 읽는 작업자 수)  EVAL_WORKERS=3  STALL_MIN=60 (로그가 이만큼 그대로면 멈춘 것으로 보고 끔)  PROGRESS_SEC=1800  RETRY=2 (학습이 죽으면 last.pt 에서 다시)
#   CKPT=keep|none  학습한 가중치 (final.pt) 를 results.zip 에 넣을지 (기본 keep — 평가를 다시 할 수 있게)   JOB=<결과 폴더 이름>
#   (로컬 시험) WORK=<작업 폴더>  PY=<쓸 파이썬>  APP_ROOT=<가짜 /app>  DATA_DIR=<조각이 있는 폴더>  HF_DIR=<HF 캐시>
# 경로 (H200): /app/scratch/depthvlm_abl = 작업 폴더 (환경 · 캐시 · 코드 · 데이터 · 체크포인트) — 그 안의 home/ 를 HOME 으로 삼아 코드의 ~/data · ~/checkpoints · ~/code 가 맞게 한다.
#   /app/output/<작업>/ = 결과만: logs/ · runs/ · ckpt/ · results.zip
# 화면 출력 = 짧은 요약 (앞 150 줄 + 끝 50 줄) — 결과 표가 늘 끝에 오게. 이름과 숫자만 찍는다 (이슈 댓글은 공개).
set -uo pipefail
cd "$(dirname "$0")" || exit 1
REPO=$PWD
MODE=${1:-}
[ $# -gt 0 ] && shift
MODES="check smoke run1 run2 run3"
[[ " $MODES " == *" $MODE "* ]] && [ -n "$MODE" ] || { echo "!!! 모드: ${MODES// / | } (받은 값: '$MODE'). 사용법은 run.sh 맨 위"; exit 2; }
KEYS="LINKS WORKERS EVAL_WORKERS STALL_MIN PROGRESS_SEC RETRY CKPT JOB WORK PY APP_ROOT DATA_DIR HF_DIR"
for k in $KEYS; do unset "$k"; done
for a in "$@"; do
  k=${a%%=*}
  if [ "$k" = "$a" ] || [[ " $KEYS " != *" $k "* ]]; then echo "!!! 모르는 인자: ${a%%=*} (KEY=값 꼴, KEY = $KEYS)"; exit 2; fi
  printf -v "$k" '%s' "${a#*=}"
done
INT='^[0-9]+$'; NAME='^[A-Za-z0-9_.+-]+$'
chk() { [ -z "${!1:-}" ] || [[ ${!1} =~ $2 ]] || { echo "!!! $1=${!1} — 형식이 맞지 않음"; exit 2; }; }
chk WORKERS "$INT"; chk EVAL_WORKERS "$INT"; chk STALL_MIN "$INT"; chk PROGRESS_SEC "$INT"; chk RETRY "$INT"; chk JOB "$NAME"
[ -z "${LINKS:-}" ] || [[ $LINKS =~ ^[A-Za-z0-9_-]{20,80}$ ]] || { echo "!!! LINKS — Drive 파일 ID 꼴이 아님"; exit 2; }
case ${CKPT:-keep} in keep|none) ;; *) echo "!!! CKPT = keep | none"; exit 2;; esac
WORKERS=${WORKERS:-8}; EVAL_WORKERS=${EVAL_WORKERS:-3}; STALL_MIN=${STALL_MIN:-60}; PROGRESS_SEC=${PROGRESS_SEC:-1800}; RETRY=${RETRY:-2}; CKPT=${CKPT:-keep}
PY=${PY:-}; DATA_DIR=${DATA_DIR:-}; LINKS=${LINKS:-}
STALL=$((STALL_MIN * 60))

# ───────────────────────── 경로 ─────────────────────────
APP=${APP_ROOT:-/app}
if [ -d "$APP" ]; then H200=1; else H200=0; fi
if [ -n "${WORK:-}" ]; then OUTROOT=$WORK/output
elif [ $H200 = 1 ]; then WORK=$APP/scratch/depthvlm_abl; OUTROOT=$APP/output
else WORK=$REPO/.work; OUTROOT=$REPO/results; fi
[ $H200 = 1 ] && [ -z "${APP_ROOT:-}" ] && OUTROOT=$APP/output
mkdir -p "$WORK" || { echo "!!! 작업 폴더를 못 만듦"; exit 1; }
WORK=$(cd "$WORK" && pwd)
JOB=${JOB:-${MODE}_$(date +%m%d_%H%M%S)}
OUTJ=$OUTROOT/$JOB; LOGS=$OUTJ/logs
mkdir -p "$LOGS" || { echo "!!! 결과 폴더를 못 만듦: $OUTJ"; exit 1; }
CONSOLE=$LOGS/console.log
exec > >(tee -a "$CONSOLE" | awk -v HL=150 -v HC=40000 -v TL=50 -v W=400 '
  { if (length($0) > W) $0 = substr($0, 1, W) " …"
    if (!cut && n < HL && c + length($0) + 1 <= HC) { print; fflush(); n++; c += length($0) + 1; next }
    cut = 1; k++; buf[k % TL] = $0 }
  END { if (k > TL) print "[… " (k - TL) " 줄 생략 — 전체는 logs/console.log]"
        for (i = (k > TL ? k - TL + 1 : 1); i <= k; i++) print buf[i % TL] }') 2>&1
TEE=$!
CUR=""; STOP=0; T_START=$(date +%s)
finish() {
  local rc=$?
  echo "[run] $(date '+%F %T') 끝 — 모드 $MODE, 종료 코드 $rc, $(( ($(date +%s) - T_START) / 60 )) 분. 결과 $OUTJ"
  exec >&- 2>&-
  wait "$TEE" 2>/dev/null   # H200 의 bash 5.1 은 인자 없는 wait 가 tee 까지 기다려 멈춘다 → PID 를 준다
  exit "$rc"
}
trap finish EXIT
trap 'STOP=1; echo "[run] 멈춤 신호 — 지금 단계를 정리하고 결과를 묶는다"; [ -n "$CUR" ] && kill -TERM "$CUR" 2>/dev/null' TERM INT
SYSPY=$(command -v python3 || echo python3)
SETSID=$(command -v setsid || true)
export HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false TRANSFORMERS_VERBOSITY=error
export HF_HOME=${HF_DIR:-$WORK/hf} PIP_CACHE_DIR=$WORK/pip UV_CACHE_DIR=$WORK/uv/cache UV_PYTHON_INSTALL_DIR=$WORK/uv/python XDG_CACHE_HOME=$WORK/cache
HOMEW=$WORK/home                                                  # 코드가 쓰는 ~ (데이터 · 체크포인트 · 외부 코드가 여기 아래)
mkdir -p "$XDG_CACHE_HOME"
STATE=$WORK/state; mkdir -p "$HOMEW" "$STATE"
under() { local p q; p=$(realpath -m "$1"); q=$(realpath -m "$2"); [ "$p" = "$q" ] || [[ $p == "$q"/* ]]; }
under "$WORK" "$OUTROOT" && { echo "!!! 작업 폴더가 결과 경로 아래 — 중단"; exit 2; }
echo "[run] $(date '+%F %T') 모드 $MODE · $([ $H200 = 1 ] && echo "H200 ($APP)" || echo 로컬) · 커밋 $(git rev-parse --short HEAD 2>/dev/null || echo '?')" \
     "· 작업 폴더 여유 $(df -h "$WORK" 2>/dev/null | awk 'NR==2 {print $4}') · 결과 $OUTJ"
[ "$MODE" = check ] || echo "[run] WORKERS=$WORKERS · EVAL_WORKERS=$EVAL_WORKERS · STALL_MIN=$STALL_MIN · RETRY=$RETRY · CKPT=$CKPT"

# ───────────────────────── 단계 실행기: 멈춤 감지 ─────────────────────────
brief_fail() {  # 실패한 로그에서 마지막 예외 한 줄과 위치만 (공개 댓글)
  "$SYSPY" -I -B - "$1" <<'PY'
import re, sys
try:
    L = open(sys.argv[1], errors="replace").read().splitlines()
except OSError:
    sys.exit()
tb = [i for i, x in enumerate(L) if x.startswith("Traceback")]
if not tb:
    for x in [x for x in L if x.strip()][-3:]:
        print("  로그 끝: " + x.strip()[:300])
    sys.exit()
body = L[tb[-1] + 1:]
frames = [re.sub(r'^\s*File "(?:.*/)?([^/"]+)", line (\d+), in (.*)$', r"\1:\2 (\3)", x) for x in body if x.lstrip().startswith('File "')]
exc = [x for x in body if x and not x.startswith((" ", "\t"))]
print("  예외: " + (exc[-1][:300] if exc else "?"))
for f in frames[-4:]:
    print("  위치: " + f[:200])
PY
}
SHOW='^(\[(h[0-9]|sm[0-9]|eval6|표|data|code|weights|env|check|h200|cb4_abl|cb3_stage[12])|!!!)'
PROG_LAST=""
run_step() {  # 이름 로그 보이기(show|quiet) 명령… → 종료 코드 (124 = 멈춤 감지로 끔, 130 = 멈춤 신호). 진행 줄 ('남은') 은 PROGRESS_SEC 마다 하나만
  local name=$1 log=$2 show=$3; shift 3
  local tp t=0 idle=0 last=-1 sz rc="" seen n lastshow=0 new prog=""
  [ "$STOP" = 1 ] && { echo "[$name] 건너뜀 (멈춤 신호)"; return 130; }
  touch "$log"; seen=$(wc -l < "$log")
  HOME=$HOMEW $SETSID "$@" >> "$log" 2>&1 < /dev/null &
  tp=$!; CUR=$tp
  while kill -0 "$tp" 2>/dev/null; do
    sleep 5; t=$((t + 5))
    sz=$(stat -c %s "$log" 2>/dev/null || echo 0)
    if [ "$sz" = "$last" ]; then idle=$((idle + 5)); else idle=0; last=$sz; fi
    n=$(wc -l < "$log")
    if [ "$n" -gt "$seen" ]; then
      new=$(tail -n "+$((seen + 1))" "$log" | head -n $((n - seen))); seen=$n
      if [ "$show" = show ]; then
        printf '%s\n' "$new" | grep -aE "$SHOW" | grep -av '남은 ' | cut -c1-400
        prog=$(printf '%s\n' "$new" | grep -a '남은 ' | tail -n 1 | cut -c1-300)
        [ -z "$prog" ] || PROG_LAST=$prog
      fi
    fi
    if [ $((t - lastshow)) -ge "$PROGRESS_SEC" ]; then echo "[$name] $(date +%H:%M) 진행 $((t / 60)) 분 · 로그 변화 없음 $((idle / 60)) 분${PROG_LAST:+ · ${PROG_LAST}}"; lastshow=$t; fi
    if [ "$STOP" = 1 ]; then rc=130; break; fi
    if [ "$idle" -ge "$STALL" ]; then
      echo "[$name] !!! 로그가 ${STALL_MIN} 분 동안 그대로 — 멈춘 것으로 보고 끈다"
      kill -TERM -- "-$tp" 2>/dev/null; kill -TERM "$tp" 2>/dev/null; sleep 30
      kill -KILL -- "-$tp" 2>/dev/null; kill -KILL "$tp" 2>/dev/null; rc=124; break
    fi
  done
  if [ "${rc:-}" = 130 ]; then
    for ((n = 0; n < 120; n += 5)); do kill -0 "$tp" 2>/dev/null || break; sleep 5; done
    kill -KILL -- "-$tp" 2>/dev/null; kill -KILL "$tp" 2>/dev/null
  fi
  wait "$tp" 2>/dev/null; n=$?
  CUR=""; PROG_LAST=""; rc=${rc:-$n}
  [ "$rc" = 0 ] || { echo "[$name] $(date +%H:%M:%S) 실패 — 종료 $rc ($((t / 60)) 분, 기록 ${log#"$OUTJ"/})"; brief_fail "$log"; }
  return "$rc"
}

# ───────────────────────── 준비: 환경 · 외부 코드 · 가중치 ─────────────────────────
UV_VER=0.12.13; PY_VER=3.12.14; PIP_VER=26.2.1   # thermal-shape-scale-h200 과 같은 고정 판 (2026-11 H200 에서 확인)
uv_bin() {
  local u
  for u in "$(command -v uv 2>/dev/null)" "$WORK/uv/pkg/bin/uv"; do
    [ -n "$u" ] && [ -x "$u" ] && [ "$("$u" --version 2>/dev/null | awk '{print $2}')" = "$UV_VER" ] && { echo "$u"; return 0; }
  done
  "$SYSPY" -m pip install -q --upgrade --target "$WORK/uv/pkg" "uv==$UV_VER" >&2 || { echo "!!! [env] uv==$UV_VER 설치 실패" >&2; return 1; }
  [ -x "$WORK/uv/pkg/bin/uv" ] && echo "$WORK/uv/pkg/bin/uv"
}
mkenv_body() {
  local req=$1 E=$2 h=$3 u
  echo "[env] 학습 환경 만들기 ($req)"
  u=$(uv_bin) || return 1
  rm -rf "$E"
  "$u" venv -q -p "$PY_VER" "$E" || { echo "!!! [env] uv venv (Python $PY_VER) 실패"; return 1; }
  env -u PIP_CONSTRAINT -u UV_CONSTRAINT "$u" pip install -q -p "$E/bin/python" "pip==$PIP_VER" || { echo "!!! [env] pip==$PIP_VER 실패"; return 1; }
  env -u PIP_CONSTRAINT -u UV_CONSTRAINT "$E/bin/python" -m pip install -q --no-input -r "$req" || { echo "!!! [env] 패키지 설치 실패"; return 1; }
  "$E/bin/python" -m pip freeze > "$E/freeze.txt" 2>/dev/null
  echo "[env] $("$E/bin/python" -V 2>&1) · $(grep -E '^(torch|transformers)==' "$E/freeze.txt" | tr '\n' ' ')"
  echo "$h" > "$E/.ok"
}
TPY=""
setup_env() {
  local E=$WORK/envs/train h rc a n0 L=$LOGS/setup_env.log
  if [ -n "$PY" ]; then TPY=$PY; echo "[env] 학습 파이썬 (주어진 것)"; return 0; fi
  h=$({ cat envs/train.txt; echo "uv $UV_VER python $PY_VER pip $PIP_VER"; } | sha256sum | cut -c1-16)
  if [ ! -f "$E/.ok" ] || [ "$(cat "$E/.ok")" != "$h" ]; then
    for a in 1 2 3; do
      touch "$L"; n0=$(wc -l < "$L")
      mkenv_body envs/train.txt "$E" "$h" >> "$L" 2>&1; rc=$?
      tail -n "+$((n0 + 1))" "$L" | grep -aE '^\[env\]|^!!!' | tail -n 2
      [ $rc = 0 ] && break
      tail -n "+$((n0 + 1))" "$L" | grep -avE '^\[env\]|^!!!|^[[:space:]]*$' | tail -n 4 | cut -c1-300 | sed 's/^/  로그: /'
      [ $a -lt 3 ] && { echo "[env] 60 초 뒤 다시 ($a/3)"; sleep 60; }
    done
    [ $rc = 0 ] || return 1
  fi
  TPY=$E/bin/python
  mkdir -p "$OUTJ/meta"; cp "$E/freeze.txt" "$OUTJ/meta/pip_freeze.txt" 2>/dev/null
}
setup_all() {
  [ -f "$STATE/.ok_setup" ] && { setup_env >/dev/null; export HF_HUB_OFFLINE=1; return 0; }
  setup_env || { echo "!!! [setup] 환경 실패 — logs/setup_env.log"; return 1; }
  run_step code "$LOGS/setup_code.log" show bash setup/fetch_code.sh "$HOMEW/code" || return 1
  run_step weights "$LOGS/setup_weights.log" show "$TPY" -I -B setup/get_weights.py --only depthvlm || \
    run_step weights "$LOGS/setup_weights.log" show env HF_HUB_DISABLE_XET=1 "$TPY" -I -B setup/get_weights.py --only depthvlm || return 1
  export HF_HUB_OFFLINE=1
  run_step env_check "$LOGS/setup_check.log" show "$TPY" -I -B setup/check.py env || return 1
  touch "$STATE/.ok_setup"
}
data_step() {  # 판 이름 (smoke | run1 | run2 | run3)
  local s=(--search)
  [ -d "$APP/data" ] && s+=("$APP/data")
  [ -n "$DATA_DIR" ] && s+=("$DATA_DIR")
  run_step data "$LOGS/data.log" show "$SYSPY" -I -B -u setup/get_data.py --kind "$1" --dest "$HOMEW" --dl "$WORK/dl" ${LINKS:+--links "$LINKS"} "${s[@]}"
}

# ───────────────────────── 학습 · 평가 ─────────────────────────
CKD=$HOMEW/checkpoints/dvsilog/coarse_branch/v2; RUND=$HOMEW/data/dvsilog/coarse_branch/v2runs
keep() {  # 태그 — 기록 (train_log · eval · 표) 은 결과 폴더 runs/ 로, final.pt 는 CKPT=keep 이면 ckpt/ 로
  mkdir -p "$OUTJ/runs" && [ -d "$RUND/$1" ] && cp -r "$RUND/$1" "$OUTJ/runs/"
  if [ "$CKPT" = keep ] && [ -f "$CKD/$1/final.pt" ]; then mkdir -p "$OUTJ/ckpt" && cp "$CKD/$1/final.pt" "$OUTJ/ckpt/$1.pt"; fi
  return 0
}
with_retry() {  # 이름 태그 명령… — 끝난 것 (final.pt) 은 건너뜀, 죽으면 last.pt 에서 다시 (RETRY 번)
  local name=$1 tag=$2 a; shift 2
  [ -f "$CKD/$tag/final.pt" ] && { echo "[$name] 이미 끝남 — 건너뜀"; return 0; }
  for ((a = 0; a <= RETRY; a++)); do
    run_step "$name" "$LOGS/$tag.log" show "$@" && [ -f "$CKD/$tag/final.pt" ] && { keep "$tag"; return 0; }
    [ "$STOP" = 1 ] && return 130
    [ $a -lt "$RETRY" ] && echo "[$name] 다시 시작 ($((a + 1))/$RETRY) — last.pt 에서"
  done
  keep "$tag"; return 1
}
stage1() {  # 태그 학습순서 [--gt_only] [--all_lists]
  local tag=$1 st=$2; shift 2
  local g="" al=""
  for x in "$@"; do case $x in --gt_only) g=--gt_only;; --all_lists) al=--all_lists;; esac; done
  with_retry "1 단계 $tag" "$tag" "$TPY" -I -B -u src/coarse_branch/cb4_abl.py s1 $g train --arm read --stream "$st" --tag "$tag" --inline --n_geo 600 $al \
    --sets --workers "$WORKERS" --resume
}
stage2() {  # 태그 학습순서 1단계태그 [--gt_only] [--all_lists]
  local tag=$1 st=$2 s1=$3; shift 3
  with_retry "2 단계 $tag" "$tag" "$TPY" -I -B -u src/h200/abl.py s2 --tag "$tag" --stream "$st" --read "$CKD/$s1/final.pt" --workers "$WORKERS" "$@"
}
eval_tok() {  # 2단계태그 1단계태그 [--limit N]
  local tag=$1 s1=$2; shift 2
  run_step "평가 $tag" "$LOGS/${tag}_eval6.log" show "$TPY" -I -B -u src/coarse_branch/cb3_stage2.py eval6 --read "$CKD/$s1/final.pt" --tag "$tag" \
    --inline --n_geo 600 --detail conv --workers "$EVAL_WORKERS" "$@" || return 1
  keep "$tag"
}
table() {  # 태그 모델들 제목 [--smoke]
  run_step "표 $1" "$LOGS/${1}_table.log" show "$TPY" -I -B -u src/h200/abl.py table --tag "$1" --models "$2" --title "$3" ${4:-} || return 1
  keep "$1"
}
zip_results() {
  "$SYSPY" -I -B - "$OUTJ/results.zip" "$OUTJ" <<'PY'
import os, sys, zipfile
dst, root = sys.argv[1], sys.argv[2]
items = []
for d, _, fs in os.walk(root):
    for f in fs:
        p = os.path.join(d, f)
        if not p.startswith(dst):
            items.append((p, os.path.relpath(p, root)))
with zipfile.ZipFile(dst + ".tmp", "w", allowZip64=True) as z:
    for src, arc in items:
        z.write(src, arc, compress_type=zipfile.ZIP_STORED if arc.endswith(".pt") else zipfile.ZIP_DEFLATED)
os.replace(dst + ".tmp", dst)
print(f"[zip] results.zip — 파일 {len(items)} 개, {os.path.getsize(dst) / 2**20:.1f} MiB")
PY
}
fail() { echo "!!! [$MODE] $1"; zip_results; exit 1; }

case "$MODE" in
  check) "$SYSPY" -I -B setup/check.py box --app "$APP" --work "$WORK" ${LINKS:+--links "$LINKS"};;
  smoke)
    setup_all || fail "준비 실패"
    data_step smoke || fail "데이터 실패"
    with_retry "판 1 smoke 학습" sm1 "$TPY" -I -B -u src/h200/abl.py head --tag sm1 --stream abl_smoke_head --gt_only --workers 2 --log_every 1 || fail "판 1 smoke 학습 실패"
    run_step "판 1 smoke 평가" "$LOGS/sm1_eval6.log" show "$TPY" -I -B -u src/h200/abl.py eval_head --tag sm1 --workers 2 --limit 2 || fail "판 1 smoke 평가 실패"
    table sm1 orig,ctrl_S "smoke 판 1 (헤드만)" --smoke || fail "판 1 smoke 표 실패"
    stage1 sm2_s1 abl_smoke_s1_nou --gt_only || fail "판 2 smoke 1 단계 실패"
    stage2 sm2 abl_smoke_s2_nou sm2_s1 --gt_only --log_every 1 || fail "판 2 smoke 2 단계 실패"
    eval_tok sm2 sm2_s1 --limit 2 || fail "판 2 smoke 평가 실패"
    table sm2 orig,prop_S,ctrl_S "smoke 판 2 (토큰 · 정답만)" --smoke || fail "판 2 smoke 표 실패"
    stage1 sm3_s1 abl_smoke_s1 --all_lists || fail "판 3 smoke 1 단계 실패"
    stage2 sm3 abl_smoke_s2 sm3_s1 --all_lists --log_every 1 || fail "판 3 smoke 2 단계 실패"
    eval_tok sm3 sm3_s1 --limit 2 || fail "판 3 smoke 평가 실패"
    table sm3 orig,prop_S,ctrl_S "smoke 판 3 (토큰 · 정답 + 무라벨)" --smoke || fail "판 3 smoke 표 실패"
    zip_results;;
  run1)
    setup_all || fail "준비 실패"
    data_step run1 || fail "데이터 실패"
    with_retry "판 1 학습" h1 "$TPY" -I -B -u src/h200/abl.py head --tag h1 --stream abl_head --gt_only --workers "$WORKERS" || fail "판 1 학습 실패"
    run_step "판 1 평가" "$LOGS/h1_eval6.log" show "$TPY" -I -B -u src/h200/abl.py eval_head --tag h1 --workers "$EVAL_WORKERS" || fail "판 1 평가 실패"
    table h1 orig,ctrl_S "판 1 — 토큰 없이 헤드만 · 정답만 · 1 배 (표 2 의 첫 줄)" || fail "판 1 표 실패"
    zip_results;;
  run2)
    setup_all || fail "준비 실패"
    data_step run2 || fail "데이터 실패"
    stage1 h2_s1 abl_s1_nou_x2 --gt_only || fail "판 2 1 단계 실패"
    stage2 h2 abl_s2_nou_x2 h2_s1 --gt_only || fail "판 2 2 단계 실패"
    eval_tok h2 h2_s1 || fail "판 2 평가 실패"
    table h2 orig,prop_S,ctrl_S "판 2 — 토큰 · 정답만 · 2 배 (부록 표 S3, prop_S 가 토큰 판)" || fail "판 2 표 실패"
    zip_results;;
  run3)
    setup_all || fail "준비 실패"
    data_step run3 || fail "데이터 실패"
    stage1 h3_s1 abl_s1_x2 --all_lists || fail "판 3 1 단계 실패"
    stage2 h3 abl_s2_x2 h3_s1 --all_lists || fail "판 3 2 단계 실패"
    eval_tok h3 h3_s1 || fail "판 3 평가 실패"
    table h3 orig,prop_S,ctrl_S "판 3 — 토큰 · 정답 + 무라벨 · 2 배 (부록 표 S3, prop_S 가 토큰 판)" || fail "판 3 표 실패"
    zip_results;;
esac
