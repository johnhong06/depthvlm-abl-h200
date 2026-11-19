"""HF 공개 가중치를 고정 리비전으로 HF_HOME 에 받고, 파일마다 크기와 해시 (LFS = sha256, 작은 파일 = git blob sha1) 를 확인한다.
목록 = setup/weights.json (DepthVLM-4B 2b2d02fc… 9 파일). 토큰은 쓰지 않는다.

사용:
  python setup/get_weights.py [--only depthvlm]   # 받기 (huggingface_hub, 학습 환경) + 확인. 이미 받은 파일은 확인만
  python3 setup/get_weights.py --dry_run [--only …]         # 받지 않고 HF API (리비전 JSON) 로 리비전·파일·크기·해시만 대조 (표준 라이브러리, 몇 초)
  python3 setup/get_weights.py --path depthvlm/config.json # HF 캐시 안 그 파일 경로를 찍는다 (네트워크 없음, 없으면 종료 1)
환경: HF_HOME (run.sh 가 /app/scratch 아래로 정함). 받기가 실패하면 30 초 쉬고 3 번까지.
종료 코드: 0 통과, 1 실패 (받기·크기·해시·API 대조).
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")       # 진행 막대가 로그를 채우지 않게
HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "weights.json")


def manifest(path=MANIFEST):
    with open(path, encoding="utf-8") as fh:
        return {k: v for k, v in json.load(fh).items() if not k.startswith("_")}


def hub_dir():
    """huggingface_hub 와 같은 규칙: HF_HUB_CACHE → HF_HOME/hub → ~/.cache/huggingface/hub."""
    if os.environ.get("HF_HUB_CACHE"):
        return os.path.expanduser(os.environ["HF_HUB_CACHE"])
    return os.path.join(os.path.expanduser(os.environ.get("HF_HOME") or "~/.cache/huggingface"), "hub")


def cache_path(m, f):
    """캐시 안 스냅숏 경로 (models--<org>--<이름>/snapshots/<리비전>/<파일>) — 있으면 그 경로, 없으면 None."""
    p = os.path.join(hub_dir(), "models--" + m["repo"].replace("/", "--"), "snapshots", m["revision"], f)
    return p if os.path.exists(p) else None


def file_hash(path, size, spec):
    """spec 의 해시 종류대로 계산 → (종류, 값)."""
    if "sha256" in spec:
        h = hashlib.sha256()
    else:                                                       # git blob id = sha1("blob <크기>\0" + 내용)
        h = hashlib.sha1(b"blob %d\0" % size)
    with open(path, "rb") as f:
        while c := f.read(16 << 20):
            h.update(c)
    return ("sha256" if "sha256" in spec else "git_sha1"), h.hexdigest()


def check_file(path, spec):
    """→ 문제 문장 또는 None."""
    if not path or not os.path.exists(path):
        return "파일 없음"
    n = os.path.getsize(path)
    if n != spec["size"]:
        return f"크기 {n:,} ≠ {spec['size']:,}"
    kind, got = file_hash(path, n, spec)
    return None if got == spec[kind] else f"{kind} {got[:12]}… ≠ {spec[kind][:12]}…"


def api_check(m, timeout=30):
    """HF API 리비전 JSON (blobs=true) 과 목록 대조 → 문제 목록 (받는 양 = JSON 몇 KB)."""
    url = f"https://huggingface.co/api/models/{m['repo']}/revision/{m['revision']}?blobs=true"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "depthvlm-abl-h200"}), timeout=timeout) as r:
        d = json.load(r)
    errs = [] if d.get("sha") == m["revision"] else [f"리비전 {str(d.get('sha'))[:8]}"]
    if d.get("gated") or d.get("private"):
        errs.append("gated/private — 토큰 없이 못 받음")
    sib = {s["rfilename"]: s for s in d.get("siblings", [])}
    for f, spec in m["files"].items():
        s = sib.get(f)
        if s is None:
            errs.append(f"{f}: 저장소에 없음")
            continue
        got = (s.get("lfs") or {}).get("sha256") if "sha256" in spec else s.get("blobId")
        if s.get("size") != spec["size"] or got != (spec.get("sha256") or spec.get("git_sha1")):
            errs.append(f"{f}: 크기 {s.get('size')} 해시 {str(got)[:12]}")
    return errs


def download(name, m):
    from huggingface_hub import hf_hub_download
    out, t0 = {}, time.time()
    for f in m["files"]:
        for k in (1, 2, 3):
            try:
                out[f] = hf_hub_download(m["repo"], f, revision=m["revision"])
                break
            except Exception as e:                              # 네트워크 끊김 등 — 예외 종류·앞부분만 (토큰 같은 것은 쓰지 않음)
                print(f"[weights] {name}/{f} 받기 실패 ({k}/3): {type(e).__name__}: {str(e)[:200]}", flush=True)
                if k == 3:
                    return None, time.time() - t0
                time.sleep(30)
    return out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="쉼표로 이름 (weights.json 의 키). 기본 = 전부")
    ap.add_argument("--dry_run", action="store_true", help="받지 않고 HF API 로 대조만")
    ap.add_argument("--path", default="", help="<이름>/<파일> 의 캐시 경로를 찍고 끝 (오프라인)")
    ap.add_argument("--manifest", default=MANIFEST)
    a = ap.parse_args()
    M = manifest(a.manifest)

    if a.path:
        name, f = a.path.split("/", 1)
        p = cache_path(M[name], f) if name in M else None
        if p is None:
            print(f"!!! [weights] 캐시에 없음: {a.path} ({hub_dir()})", file=sys.stderr)
            return 1
        print(p)
        return 0

    names = [x for x in a.only.split(",") if x] or list(M)
    unknown = [x for x in names if x not in M]
    if unknown:
        print(f"!!! [weights] 모르는 이름 {unknown} (있는 것: {list(M)})", flush=True)
        return 1
    bad = 0
    for name in names:
        m = M[name]
        tag = f"{name} {m['repo']} @{m['revision'][:8]}"
        gb = sum(s["size"] for s in m["files"].values()) / 1e9
        if a.dry_run:
            try:
                errs = api_check(m)
            except Exception as e:
                errs = [f"API 실패 {type(e).__name__}"]
            print(f"[weights] {tag}: " + (f"!!! 대조 실패 {errs[:3]}" if errs else f"파일 {len(m['files'])} 개 {gb:.2f} GB — 리비전·크기·해시가 HF 와 같음 (받지 않음)"), flush=True)
            bad += bool(errs)
            continue
        paths, dt = download(name, m)
        if paths is None:
            print(f"!!! [weights] {tag}: 받기 실패", flush=True)
            bad += 1
            continue
        errs = [f"{f}: {e}" for f, spec in m["files"].items() if (e := check_file(paths.get(f), spec))]
        if errs:
            print(f"!!! [weights] {tag}: 확인 실패 {errs[:3]}", flush=True)
            bad += 1
        else:
            print(f"[weights] {tag}: 파일 {len(m['files'])} 개 {gb:.2f} GB, 크기·해시 확인 ({dt:.0f} 초)", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
