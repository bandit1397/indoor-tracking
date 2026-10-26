"""휴대폰 동영상들 → 역 내부 길 지도(graph.json) 자동 생성.

    python mapper/build_map.py --station seoul

입력:  stations/<역>/videos/*.mp4   걸으면서 촬영한 동영상들
       stations/<역>/labels.csv     목적지 이름 (어느 영상 몇 초 지점이 '1번 출구'인지)
출력:  web/data/<역>/graph.json, web/data/<역>/frames/*.jpg

동작:
 1. 영상에서 일정 간격으로 사진(지점)을 뽑는다. 흔들린 사진, 제자리 사진은 버린다.
 2. 같은 영상에서 연속된 지점끼리 연결한다 (걸어간 길).
 3. 서로 다른 영상(또는 같은 영상의 다른 시점)에서 같은 장소를 찍은 지점을
    AI 특징 벡터로 후보를 찾고, ORB 특징점 + RANSAC 으로 검증한 뒤 연결한다 (교차점).
 4. 얼굴은 흐리게 처리해 저장한다 (공개 웹에 올라가므로).
"""
import argparse
import base64
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import DEFAULT_MODEL, ROOT, Embedder

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
WEB_IMG_WIDTH = 480
ORB_WIDTH = 640
LANGS = ["ko", "en", "zh", "ja"]


def sample_frames(path: Path, fps: float, blur_min: float):
    cap = cv2.VideoCapture(str(path))
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(vfps / fps))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            gray = cv2.cvtColor(cv2.resize(frame, (320, int(320 * frame.shape[0] / frame.shape[1]))),
                                cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() >= blur_min:
                yield i / vfps, frame
        i += 1
    cap.release()


def resize_w(img, width):
    h, w = img.shape[:2]
    return img if w <= width else cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)


def blur_faces(img, cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for (x, y, w, h) in cascade.detectMultiScale(gray, 1.1, 5, minSize=(24, 24)):
        roi = img[y:y + h, x:x + w]
        img[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (0, 0), max(w, h) / 6)
    return img


def blur_regions(img, regions):
    """privacy.csv 에 적은 구역(화면 비율 0~1)을 흐리게 한다. 자동으로 못 찾은 얼굴/사람용."""
    h, w = img.shape[:2]
    for x0, y0, x1, y1 in regions:
        xa, ya, xb, yb = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
        if xb > xa and yb > ya:
            img[ya:yb, xa:xb] = cv2.GaussianBlur(img[ya:yb, xa:xb], (0, 0), max(xb - xa, yb - ya) / 6)
    return img


def orb_verify(a, b, matcher, min_inliers):
    (ka, da), (kb, db) = a, b
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return 0
    good = []
    for m in matcher.knnMatch(da, db, k=2):
        if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
            good.append(m[0])
    if len(good) < min_inliers:
        return 0
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    _, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 3.0, 0.99)
    return int(mask.sum()) if mask is not None else 0


def load_labels(path: Path):
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    for r in csv.DictReader(lines):
        rows.append(r)
    return rows


def components(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _ in edges:
        parent[find(a)] = find(b)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return sorted(groups.values(), key=len, reverse=True)


def main():
    ap = argparse.ArgumentParser(description="동영상으로 역 내부 길 지도 만들기")
    ap.add_argument("--station", default="seoul", help="stations/ 아래 역 폴더 이름")
    ap.add_argument("--station-dir", help="역 폴더 경로를 직접 지정")
    ap.add_argument("--out", help="출력 폴더 (기본: web/data/<역>)")
    ap.add_argument("--name", help="역 표시 이름 (기본: station.json 또는 폴더 이름)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--fps", type=float, default=1.0, help="초당 뽑을 지점 수 (기본 1)")
    ap.add_argument("--blur-min", type=float, default=30.0, help="이보다 흐린 사진은 버림")
    ap.add_argument("--dedup", type=float, default=0.97, help="직전 지점과 이만큼 비슷하면 버림")
    ap.add_argument("--sim", type=float, default=0.75, help="교차점 후보 유사도 기준")
    ap.add_argument("--min-inliers", type=int, default=30, help="교차점 확정에 필요한 특징점 수")
    ap.add_argument("--revisit-sec", type=float, default=30.0,
                    help="같은 영상 안에서 이 시간(초) 이상 떨어져 다시 지나간 곳만 교차점으로 연결")
    ap.add_argument("--no-face-blur", action="store_true")
    args = ap.parse_args()

    sdir = Path(args.station_dir) if args.station_dir else ROOT / "stations" / args.station
    out = Path(args.out) if args.out else ROOT / "web" / "data" / sdir.name
    videos = sorted(p for p in (sdir / "videos").glob("*") if p.suffix.lower() in VIDEO_EXT)
    if not videos:
        raise SystemExit(f"동영상이 없습니다: {sdir / 'videos'}")

    meta_path = sdir / "station.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    names = meta.get("name") or {lang: args.name or sdir.name for lang in LANGS}

    embedder = Embedder(Path(args.model))
    orb = cv2.ORB_create(1500)
    cascade = None if args.no_face_blur else cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # privacy.csv: video,start,end,x0,y0,x1,y1 → 그 영상의 start~end 초 장면에서 해당 구역을 흐리게
    privacy = []
    for r in load_labels(sdir / "privacy.csv"):
        privacy.append((r["video"].strip(), (float(r["start"]), float(r["end"])),
                        float(r["x0"]), float(r["y0"]), float(r["x1"]), float(r["y1"])))

    if (out / "frames").exists():
        shutil.rmtree(out / "frames")
    (out / "frames").mkdir(parents=True)

    nodes, embs, orbs, edges = [], [], [], []
    for vi, vpath in enumerate(videos):
        prev_emb, prev_id, prev_t, kept = None, None, 0.0, 0
        for t, frame in sample_frames(vpath, args.fps, args.blur_min):
            emb = embedder.embed([frame])[0]
            if prev_emb is not None and float(emb @ prev_emb) > args.dedup:
                continue
            nid = len(nodes)
            web_img = resize_w(frame, WEB_IMG_WIDTH).copy()
            if cascade is not None:
                web_img = blur_faces(web_img, cascade)
            web_img = blur_regions(web_img, [r[2:] for r in privacy
                                             if r[0] == vpath.name and r[1][0] <= t <= r[1][1]])
            cv2.imwrite(str(out / "frames" / f"{nid}.jpg"), web_img, [cv2.IMWRITE_JPEG_QUALITY, 72])
            gray = cv2.cvtColor(resize_w(frame, ORB_WIDTH), cv2.COLOR_BGR2GRAY)
            orbs.append(orb.detectAndCompute(gray, None))
            nodes.append({"id": nid, "img": f"frames/{nid}.jpg", "v": vpath.name, "t": round(t, 1)})
            embs.append(emb)
            if prev_id is not None:
                edges.append((prev_id, nid, round(max(0.5, t - prev_t), 1)))  # 걸린 시간(초)
            prev_emb, prev_id, prev_t = emb, nid, t
            kept += 1
        print(f"[{vi + 1}/{len(videos)}] {vpath.name}: 지점 {kept}개")

    E = np.stack(embs)
    sim = E @ E.T
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    vid = [n["v"] for n in nodes]
    linked = set()
    verified = []
    for i in range(len(nodes)):
        for j in np.argsort(-sim[i])[1:8]:
            j = int(j)
            if sim[i, j] < args.sim:
                break
            if vid[i] == vid[j] and abs(nodes[i]["t"] - nodes[j]["t"]) < args.revisit_sec:
                continue  # 같은 영상의 가까운 지점은 이미 길로 연결됨 (지름길이 생기지 않게)
            key = (min(i, j), max(i, j))
            if key in linked:
                continue
            linked.add(key)
            inliers = orb_verify(orbs[i], orbs[j], matcher, args.min_inliers)
            if inliers >= args.min_inliers:
                verified.append((key[0], key[1], inliers))
    # 서로가 상대 영상에서 가장 잘 맞는 짝일 때만 연결 (조금만 겹친 장면끼리 지름길이 생기지 않게)
    best = defaultdict(int)
    for a, b, inl in verified:
        best[(a, vid[b])] = max(best[(a, vid[b])], inl)
        best[(b, vid[a])] = max(best[(b, vid[a])], inl)
    cross = 0
    for a, b, inl in verified:
        if inl == best[(a, vid[b])] and inl == best[(b, vid[a])]:
            edges.append((a, b, 0.5))  # 같은 장소 → 거의 0초
            cross += 1
    print(f"교차점(겹치는 장면) 연결 {cross}개")

    places = []
    for r in load_labels(sdir / "labels.csv"):
        cands = [n for n in nodes if n["v"] == r["video"].strip()]
        if not cands:
            print(f"경고: labels.csv 의 영상 '{r['video']}' 을 찾을 수 없음")
            continue
        t = float(r["time"])
        node = min(cands, key=lambda n: abs(n["t"] - t))
        places.append({"node": node["id"],
                       "name": {lang: (r.get(lang) or r.get("en") or r["ko"]).strip() for lang in LANGS}})

    comps = components(len(nodes), edges)
    if len(comps) > 1:
        print(f"주의: 지도가 {len(comps)}조각으로 나뉘어 있습니다 (크기: {[len(c) for c in comps[:5]]}).")
        print("      조각을 잇는 구간을 겹치게 다시 촬영하면 자동으로 연결됩니다.")
    main_comp = set(comps[0])
    for p in places:
        if p["node"] not in main_comp:
            print(f"주의: 목적지 '{p['name']['ko']}' 가 가장 큰 지도 조각과 연결되어 있지 않습니다.")

    q = np.clip(np.round(E * 127), -127, 127).astype(np.int8)
    graph = {
        "version": 1,
        "name": names,
        "dim": int(E.shape[1]),
        "nodes": nodes,
        "emb": base64.b64encode(q.tobytes()).decode(),
        "edges": [[a, b, w] for a, b, w in edges],
        "places": places,
    }
    (out / "graph.json").write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
    print(f"완료: 지점 {len(nodes)}개, 길 {len(edges)}개, 목적지 {len(places)}개 → {out / 'graph.json'}")


if __name__ == "__main__":
    main()
