"""데이터 준비: 판 (--kind) 에 필요한 묶음 (tar) 을 찾거나 받아서 sha256 을 확인하고 --dest (작업 폴더의 HOME) 에 푼다. 목록 = setup/data.json.

묶음 안의 경로는 홈 기준 (data/…) 이라, HOME=<dest> 로 돌리면 코드의 ~/data/… 가 그대로 맞는다.
묶음은 이 저장소에 없고 받는 주소도 저장소에 적지 않는다 (다시 배포하면 안 되는 데이터가 섞여 있다). 묶음 조각마다 찾는 순서:
  1. --search 폴더 (H200 = /app/data) 아래 5 단계 안에서 크기가 꼭 맞고 sha256 이 같은 파일 — 올려 둔 사본
  2. 없으면 --links 로 받은 목록 (JSON: {조각 파일 이름: Drive 파일 ID}) 의 Google Drive 파일에서 받는다 (이어받기, 3 번까지)
     --links = 그 목록 파일의 Drive 파일 ID (run.sh 의 LINKS= 인자 — 실행 명령에만 적는 값)
풀 때는 정규 파일만, 이름이 data/ 로 시작하고 '..' 이 없는 것만. 조각마다 푼 파일 수가 data.json 과 같아야 한다.
다 풀린 조각은 <dest>/.packs/<조각> 에 표시 (같은 작업 안에서 다시 부르면 건너뜀). 받은 tar 는 풀고 나서 지운다.
화면에 주소 · 파일 ID 를 찍지 않는다. 표준 라이브러리만 쓴다.

사용: python3 setup/get_data.py --kind smoke|run1|run2|run3 --dest <HOME> --dl <받을 폴더> [--search /app/data] [--links <Drive 파일 ID>]
      python3 setup/get_data.py --kind run3 --list      # 받을 묶음 · 크기만
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = json.load(open(os.path.join(HERE, "data.json"), encoding="utf-8"))
ID_OK = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-")


def is_id(s):
    return isinstance(s, str) and 20 <= len(s) <= 80 and set(s) <= ID_OK


def drive_url(fid):
    return f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t"


def read_links(fid):
    """목록 파일 (JSON, 수 KB) → {조각 이름: 파일 ID}. 주소 · ID 는 찍지 않는다."""
    if not is_id(fid):
        raise ValueError("LINKS 값의 꼴이 Drive 파일 ID 가 아님")
    req = urllib.request.Request(drive_url(fid), headers={"User-Agent": "depthvlm-abl-h200"})
    with urllib.request.urlopen(req, timeout=60) as r:
        if "text/html" in (r.headers.get("Content-Type") or ""):
            raise ValueError("목록 파일 대신 웹 페이지가 옴 (공유가 꺼졌거나 ID 가 틀림)")
        d = json.loads(r.read(1 << 20).decode("utf-8"))
    if not isinstance(d, dict) or not all(isinstance(k, str) and is_id(v) for k, v in d.items()):
        raise ValueError("목록 파일의 꼴이 다름")
    return d


def download(name, size, fid, dst):
    """Google Drive → dst (이어받기). 받은 크기가 맞으면 True."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    part = dst + ".part"
    for t in range(3):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        req = urllib.request.Request(drive_url(fid), headers={"User-Agent": "depthvlm-abl-h200", "Range": f"bytes={have}-"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if "text/html" in (r.headers.get("Content-Type") or ""):
                    print(f"!!! [data] {name}: Drive 가 파일 대신 웹 페이지를 줌 (받기 한도 · 공유 꺼짐)", flush=True)
                    return False
                mode = "ab" if r.status == 206 else "wb"
                got, mark = (have if mode == "ab" else 0), time.time()
                with open(part, mode) as fh:
                    while c := r.read(16 << 20):
                        fh.write(c)
                        got += len(c)
                        if time.time() - mark > 120:              # 2 분마다 한 줄 (멈춤 감지가 진행으로 보게)
                            print(f"[data] {name} 받는 중 {got / 2**30:.2f} / {size / 2**30:.2f} GiB", flush=True)
                            mark = time.time()
        except Exception as ex:
            print(f"[data] {name} 받기 끊김 ({t + 1}/3, {type(ex).__name__}) — 30 초 뒤 이어받기", flush=True)
            time.sleep(30)
            continue
        if os.path.getsize(part) == size:
            os.replace(part, dst)
            return True
        if os.path.getsize(part) > size:
            os.remove(part)
    return False


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while c := fh.read(32 << 20):
            h.update(c)
    return h.hexdigest()


def search(roots, size, depth=5):
    for root in roots:
        if not os.path.isdir(root):
            continue
        base = root.rstrip("/").count("/")
        for d, subs, files in os.walk(root):
            if d.count("/") - base >= depth:
                subs[:] = []
            for f in files:
                p = os.path.join(d, f)
                try:
                    if os.path.getsize(p) == size:
                        yield p
                except OSError:
                    pass


def unpack(tar, dest):
    """정규 파일만, data/ 아래 상대 경로만 → 푼 개수 (어긋난 이름이 하나라도 있으면 예외)."""
    n, mark = 0, time.time()
    with tarfile.open(tar, "r:") as t:
        for m in t:
            if not m.isreg():
                continue
            name = m.name
            if os.path.isabs(name) or ".." in name.split("/") or not name.startswith("data/"):
                raise ValueError(f"묶음 안에 허용하지 않는 이름: {name[:80]}")
            p = os.path.join(dest, name)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with t.extractfile(m) as fi, open(p, "wb") as fo:
                shutil.copyfileobj(fi, fo, 4 << 20)
            n += 1
            if time.time() - mark > 120:
                print(f"[data] 푸는 중 {os.path.basename(tar)} 파일 {n:,}", flush=True)
                mark = time.time()
    return n


def needed(kind):
    return [(name, p) for name, s in SPEC.items() if kind in s["runs"] for p in s["parts"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["smoke", "run1", "run2", "run3"], required=True)
    ap.add_argument("--dest", default="")
    ap.add_argument("--dl", default="")
    ap.add_argument("--search", nargs="*", default=[])
    ap.add_argument("--links", default="")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    parts = needed(a.kind)
    total = sum(p["size"] for _, p in parts)
    if a.list:
        for name, p in parts:
            print(f"{p['file']}\t{p['size'] / 2**30:.2f} GiB\t파일 {p['n_files']:,}")
        print(f"합 {total / 2**30:.1f} GiB · 조각 {len(parts)}")
        return 0
    mk = os.path.join(a.dest, ".packs")
    os.makedirs(mk, exist_ok=True)
    todo = [(n, p) for n, p in parts if not os.path.exists(os.path.join(mk, p["file"]))]
    print(f"[data] 판 {a.kind}: 묶음 {len({n for n, _ in parts})} 개 · 조각 {len(parts)} 개 · {total / 2**30:.1f} GiB (남은 조각 {len(todo)}) · 받는 폴더 여유 "
          f"{shutil.disk_usage(a.dest).free / 2**30:.0f} GiB", flush=True)
    links = {}
    t_all = time.time()
    for name, p in todo:
        t0 = time.time()
        src = next((c for c in search(a.search, p["size"]) if sha256(c) == p["sha256"]), None)
        how = "올려 둔 사본"
        if src is None:
            if not links:
                if not a.links:
                    print(f"!!! [data] {p['file']} 이 없음 — 실행 명령에 LINKS=<목록 파일 ID> 를 주거나 묶음을 /app/data 아래에 올린다", flush=True)
                    return 1
                try:
                    links = read_links(a.links)
                except Exception as ex:
                    print(f"!!! [data] 받을 곳 목록을 읽지 못함 ({type(ex).__name__}: {str(ex)[:80] if isinstance(ex, ValueError) else ''})", flush=True)
                    return 1
            if p["file"] not in links:
                print(f"!!! [data] 목록에 없는 조각: {p['file']}", flush=True)
                return 1
            src, how = os.path.join(a.dl, p["file"]), "Google Drive"
            if not download(p["file"], p["size"], links[p["file"]], src):
                print(f"!!! [data] {p['file']} 을 받지 못함", flush=True)
                return 1
            if sha256(src) != p["sha256"]:
                print(f"!!! [data] {p['file']}: 받은 파일의 sha256 이 다름", flush=True)
                os.remove(src)
                return 1
        t1 = time.time()
        n = unpack(src, a.dest)
        ok = n == p["n_files"]
        print(f"[data] {p['file']} {p['size'] / 2**30:.2f} GiB — {how} {t1 - t0:.0f} 초 · 풀기 {time.time() - t1:.0f} 초 · 파일 {n:,} 개 "
              + ("(맞음)" if ok else f"!!! 기대 {p['n_files']:,}") + f" · 여유 {shutil.disk_usage(a.dest).free / 2**30:.0f} GiB", flush=True)
        if how == "Google Drive":
            os.remove(src)
        if not ok:
            return 1
        open(os.path.join(mk, p["file"]), "w").close()
    print(f"[data] 끝 — {(time.time() - t_all) / 60:.1f} 분", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
