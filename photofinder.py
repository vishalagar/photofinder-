"""Face finder for a folder of photos.

  python photofinder.py index PHOTO_DIR    # detect faces, build the index (once)
  python photofinder.py serve PHOTO_DIR    # open the web UI on http://127.0.0.1:5000

Everything runs locally; no photo ever leaves your machine.
"""
import argparse
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DATA_DIR = Path("index_data")
MATCH_THRESHOLD = 0.40   # cosine similarity; ~0.35-0.45 is a sensible range for ArcFace
CLUSTER_EPS = 0.55       # cosine distance for grouping faces into people
MIN_DET_SCORE = 0.5
THUMB = 112


def load_image(path):
    """Load as BGR array, honoring EXIF rotation (phone photos)."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def get_model():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(1024, 1024))  # large det_size helps small faces in group shots
    return app


def crop_thumb(img, bbox):
    x1, y1, x2, y2 = bbox.astype(int)
    pad = int(0.25 * max(x2 - x1, y2 - y1))
    h, w = img.shape[:2]
    face = img[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
    return cv2.resize(face, (THUMB, THUMB), interpolation=cv2.INTER_AREA)


def build_index(photo_dir):
    photo_dir = Path(photo_dir)
    files = sorted(p for p in photo_dir.rglob("*") if p.suffix.lower() in EXTS)
    if not files:
        sys.exit(f"No images found in {photo_dir}")
    (DATA_DIR / "faces").mkdir(parents=True, exist_ok=True)
    model = get_model()
    embs, meta = [], []
    for i, path in enumerate(files, 1):
        rel = str(path.relative_to(photo_dir))
        try:
            img = load_image(path)
            faces = model.get(img)
        except Exception as e:
            print(f"  skip {rel}: {e}")
            continue
        n = 0
        for f in faces:
            if f.det_score < MIN_DET_SCORE:
                continue
            fid = len(meta)
            cv2.imwrite(str(DATA_DIR / "faces" / f"{fid}.jpg"), crop_thumb(img, f.bbox))
            embs.append(f.normed_embedding)
            meta.append({"id": fid, "photo": rel})
            n += 1
        print(f"[{i}/{len(files)}] {rel}: {n} face(s)")
    embs = np.array(embs, dtype=np.float32)
    np.save(DATA_DIR / "embeddings.npy", embs)
    (DATA_DIR / "meta.json").write_text(json.dumps(meta))
    print(f"\nIndexed {len(meta)} faces across {len(files)} photos.")


def cluster_people(embs):
    from sklearn.cluster import DBSCAN
    if len(embs) == 0:
        return []
    labels = DBSCAN(eps=CLUSTER_EPS, min_samples=2, metric="cosine").fit_predict(embs)
    groups = {}
    for idx, lab in enumerate(labels):
        if lab >= 0:
            groups.setdefault(int(lab), []).append(idx)
    return sorted(groups.values(), key=len, reverse=True)


def serve(photo_dir, host, port):
    from flask import Flask, abort, jsonify, render_template, request, send_from_directory

    photo_dir = Path(photo_dir).resolve()
    if not (DATA_DIR / "embeddings.npy").exists():
        sys.exit("No index found. Run: python photofinder.py index PHOTO_DIR")
    embs = np.load(DATA_DIR / "embeddings.npy")
    meta = json.loads((DATA_DIR / "meta.json").read_text())
    people = cluster_people(embs)
    model = None
    app = Flask(__name__)

    def photos_for(idxs):
        return sorted({meta[i]["photo"] for i in idxs})

    @app.route("/")
    def home():
        return render_template("index.html", n_photos=len({m["photo"] for m in meta}), n_faces=len(meta),
                               people=[{"id": k, "count": len(photos_for(g)), "thumb": g[0]}
                                       for k, g in enumerate(people)])

    @app.route("/faces/<int:fid>.jpg")
    def face(fid):
        return send_from_directory(DATA_DIR.resolve() / "faces", f"{fid}.jpg")

    @app.route("/photo/<path:rel>")
    def photo(rel):
        return send_from_directory(photo_dir, rel)

    @app.route("/thumb/<path:rel>")
    def thumb(rel):
        p = (photo_dir / rel).resolve()
        if photo_dir not in p.parents:
            abort(404)
        img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        img.thumbnail((400, 400))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=80)
        return buf.getvalue(), 200, {"Content-Type": "image/jpeg", "Cache-Control": "max-age=86400"}

    @app.route("/api/person/<int:pid>")
    def person(pid):
        if pid >= len(people):
            abort(404)
        return jsonify(photos=photos_for(people[pid]))

    @app.route("/api/search", methods=["POST"])
    def search():
        nonlocal model
        upload = request.files.get("selfie")
        if not upload:
            return jsonify(error="No file uploaded"), 400
        model = model or get_model()
        img = cv2.cvtColor(np.array(ImageOps.exif_transpose(Image.open(upload.stream)).convert("RGB")),
                           cv2.COLOR_RGB2BGR)
        faces = [f for f in model.get(img) if f.det_score >= MIN_DET_SCORE]
        if not faces:
            return jsonify(error="No face found in that picture. Try a clearer, front-facing one."), 400
        # Use the largest face in the upload as the query.
        q = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])).normed_embedding
        sims = embs @ q
        hits = np.where(sims >= MATCH_THRESHOLD)[0]
        best = {}
        for i in hits:
            p = meta[i]["photo"]
            best[p] = max(best.get(p, 0), float(sims[i]))
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        return jsonify(photos=[p for p, _ in ranked], scores=[round(s, 2) for _, s in ranked])

    print(f"Open http://{host}:{port}")
    app.run(host=host, port=port)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index").add_argument("photo_dir")
    s = sub.add_parser("serve")
    s.add_argument("photo_dir")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=5000)
    a = ap.parse_args()
    if a.cmd == "index":
        build_index(a.photo_dir)
    else:
        serve(a.photo_dir, a.host, a.port)
