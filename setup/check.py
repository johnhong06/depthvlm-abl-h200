"""점검 (설치 없음).
  box  (시스템 python3, 표준 라이브러리만): 파이썬 · GPU · CPU · RAM · 작업 폴더와 결과 경로의 여유 · /app/data 의 묶음 · 외부 접속 (GitHub · HF · Drive) · LINKS 목록을 읽을 수 있는지
  env  (학습 환경 python): torch · CUDA · GPU 이름 · transformers 판 · 우리 코드 import (HOME=작업 폴더 홈, 코드 · 가중치가 받아진 뒤)
화면에는 이름과 숫자만 (이슈 댓글은 공개). LINKS 값 · 파일 ID 는 찍지 않는다.
    python3 setup/check.py box --app /app --work <작업 폴더> [--links <ID>]
    <학습 python> setup/check.py env
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def gib(n):
    return f"{n / 2 ** 30:.0f} GiB"


def box(a):
    ok = True
    print(f"[check] 파이썬 {sys.version.split()[0]} · CPU {os.cpu_count()} 개", flush=True)
    try:
        mem = {l.split(":")[0]: int(l.split()[1]) * 1024 for l in open("/proc/meminfo") if l.startswith(("MemTotal", "MemAvailable"))}
        print(f"[check] RAM {gib(mem['MemTotal'])} (쓸 수 있는 것 {gib(mem['MemAvailable'])})", flush=True)
    except OSError:
        pass
    try:
        g = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
        print(f"[check] GPU {g or '없음'}", flush=True)
    except Exception as ex:
        print(f"!!! [check] nvidia-smi 실패 ({type(ex).__name__})", flush=True)
        ok = False
    for name, p in (("작업 폴더", a.work), ("결과 경로", os.path.join(a.app, "output")), ("올려 둔 데이터", os.path.join(a.app, "data"))):
        q = p
        while q and not os.path.exists(q):
            q = os.path.dirname(q)
        if q:
            u = shutil.disk_usage(q)
            print(f"[check] {name} {p}: 여유 {gib(u.free)} / 전체 {gib(u.total)}", flush=True)
    need = json.load(open(os.path.join(HERE, "data.json")))
    for kind in ("smoke", "run1", "run2", "run3"):
        tot = sum(p["size"] for s in need.values() if kind in s["runs"] for p in s["parts"])
        print(f"[check] 판 {kind} 이 받는 데이터 {gib(tot)} (풀면 같은 크기 · 받는 동안 조각 하나 8 GB 가 더)", flush=True)
    d = os.path.join(a.app, "data")
    if os.path.isdir(d):
        names = {p["file"] for s in need.values() for p in s["parts"]}
        have = sorted(f for _, _, fs in os.walk(d) for f in fs if f in names)
        print(f"[check] /app/data 에 올라가 있는 우리 조각 {len(have)} 개", flush=True)
    for name, url in (("GitHub", "https://codeload.github.com/johnhong06/vlm-depth-rmse/tar.gz/0a1f2f940b23c0bcaa93eb851e63bba111852b36"),
                      ("Hugging Face", "https://huggingface.co/api/models/JonnyYu828/DepthVLM-4B/revision/2b2d02fcfe0c89c8aa7d541055e5a078930077c9"),
                      ("PyTorch whl", "https://download.pytorch.org/whl/cu128/torch/"),
                      ("Google Drive", "https://drive.usercontent.google.com/")):
        t0 = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "depthvlm-abl-h200", "Range": "bytes=0-1023"})
            with urllib.request.urlopen(req, timeout=30) as r:
                r.read(1024)
            print(f"[check] 접속 {name}: 됨 ({time.time() - t0:.1f} 초)", flush=True)
        except Exception as ex:
            code = getattr(ex, "code", None)
            fine = name == "Google Drive" and code in (400, 404)             # 주소만 찔러 본 것 — 서버가 답하면 접속은 된다
            print(f"[check] 접속 {name}: {'됨 (서버 응답 ' + str(code) + ')' if fine else '!!! 안 됨 (' + type(ex).__name__ + (' ' + str(code) if code else '') + ')'}", flush=True)
            ok = ok and fine
    if a.links:
        sys.path.insert(0, HERE)
        import get_data
        try:
            L = get_data.read_links(a.links)
            names = {p["file"] for s in need.values() for p in s["parts"]}
            print(f"[check] LINKS 목록: 조각 {len(L)} 개 · data.json 의 조각 중 목록에 없는 것 {len(names - set(L))} 개", flush=True)
        except Exception as ex:
            print(f"!!! [check] LINKS 목록을 읽지 못함 ({type(ex).__name__})", flush=True)
            ok = False
    print(f"[check] {'통과' if ok else '!!! 문제 있음'}", flush=True)
    return 0 if ok else 1


def env(a):
    import torch
    import transformers
    print(f"[env] torch {torch.__version__} · CUDA {torch.version.cuda} · GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '없음'} · transformers {transformers.__version__}", flush=True)
    assert torch.cuda.is_available(), "GPU 를 못 봄"
    src = os.path.join(os.path.dirname(HERE), "src")
    sys.path.insert(0, os.path.join(src, "analysis"))
    sys.path.insert(0, os.path.join(src, "coarse_branch"))
    sys.argv = sys.argv[:1]
    import cb2_data  # noqa: F401
    import cb2_stage2  # noqa: F401
    import cb3_data  # noqa: F401
    import cb_eval
    cb_eval.pe()
    import geo_mix  # noqa: F401
    print("[env] 우리 코드 import 통과 (cb2_stage2 · cb3_data · cb_eval · paper_eval · geo_mix)", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("box", "env"))
    ap.add_argument("--app", default="/app")
    ap.add_argument("--work", default="")
    ap.add_argument("--links", default="")
    a = ap.parse_args()
    return box(a) if a.mode == "box" else env(a)


if __name__ == "__main__":
    sys.exit(main())
