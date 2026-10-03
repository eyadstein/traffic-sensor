"""Turn one UA-DETRAC sequence (folder of JPEG frames inside the big zip) into an mp4, without extracting the whole zip.

  python src/ua_detrac_to_video.py --zip ua_detrac_test_set.zip --list
  python src/ua_detrac_to_video.py --zip ua_detrac_test_set.zip --seq MVI_39031 --out data/MVI_39031.mp4
"""
import argparse, os, re, zipfile
from collections import defaultdict

import cv2
import numpy as np


def sequences(zf):
    seqs = defaultdict(list)
    for n in zf.namelist():
        if n.lower().endswith((".jpg", ".jpeg", ".png")):
            m = re.search(r"(MVI_\d+)", n)
            seqs[m.group(1) if m else os.path.dirname(n)].append(n)
    return {k: sorted(v) for k, v in seqs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--seq")
    ap.add_argument("--out")
    ap.add_argument("--fps", type=float, default=25.0)   # UA-DETRAC is recorded at 25 fps
    a = ap.parse_args()
    zf = zipfile.ZipFile(a.zip)
    seqs = sequences(zf)
    if a.list or not a.seq:
        for k, v in sorted(seqs.items()):
            print(f"{k}: {len(v)} frames ({len(v) / a.fps:.0f} s)")
        return
    if a.seq not in seqs:
        raise SystemExit(f"{a.seq} not found. Use --list to see names.")
    names = seqs[a.seq]
    first = cv2.imdecode(np.frombuffer(zf.read(names[0]), np.uint8), cv2.IMREAD_COLOR)
    h, w = first.shape[:2]
    out = a.out or f"data/{a.seq}.mp4"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
    for n in names:
        vw.write(cv2.imdecode(np.frombuffer(zf.read(n), np.uint8), cv2.IMREAD_COLOR))
    vw.release()
    print(f"wrote {out}: {len(names)} frames, {w}x{h} @ {a.fps} fps")


if __name__ == "__main__":
    main()
