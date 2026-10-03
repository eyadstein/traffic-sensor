"""Fix km/h using ONE real distance you measured along the road.

  python src/scale_fix.py --video data/clip.mp4 --config configs/clip.yaml --meters 15
      -> click 2 road points that are exactly 15 m apart (along the road); the config is rescaled in place (.bak kept)
  python src/scale_fix.py --config configs/clip.yaml --meters 15 --points 410,300,455,210
      -> same without a window

The correction is a single uniform scale on the calibration rectangle, so it fixes the overall size of the
metres but not a wrong width/length ratio of the original rectangle.
"""
import argparse
import math
import os
import shutil
import sys

import cv2
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from geometry import GroundPlane


def rescale(cfg, p1, p2, meters):
    """Return (new_cfg, factor, model_distance). Speeds scale by the same factor."""
    h = cfg["homography"]
    plane = GroundPlane(h["image_points"], h["world_points"])
    a, b = plane.to_world(p1), plane.to_world(p2)
    d = math.hypot(a[0] - b[0], a[1] - b[1])
    if d < 1e-6:
        raise ValueError("the two points map to the same ground position")
    k = meters / d
    cfg["homography"]["world_points"] = [[x * k, y * k] for x, y in h["world_points"]]
    return cfg, k, d


def pick(video):
    cap = cv2.VideoCapture(video)
    ok, f = cap.read()
    if not ok:
        raise SystemExit("cannot read video")
    pts = []

    def cb(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y))

    cv2.namedWindow("scale")
    cv2.setMouseCallback("scale", cb)
    while True:
        v = f.copy()
        for p in pts:
            cv2.circle(v, p, 5, (0, 255, 255), -1)
        if len(pts) == 2:
            cv2.line(v, pts[0], pts[1], (0, 255, 255), 1)
        cv2.putText(v, "click 2 road points the known distance apart (ESC cancels)", (10, 25), 0, 0.6, (0, 255, 255), 2)
        cv2.imshow("scale", v)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            raise SystemExit("cancelled")
        if len(pts) == 2:
            cv2.waitKey(400)
            cv2.destroyAllWindows()
            return pts[0], pts[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--meters", type=float, required=True, help="real distance between the two points")
    ap.add_argument("--video")
    ap.add_argument("--points", help="x1,y1,x2,y2 in pixels (skips the window)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if "homography" not in cfg:
        raise SystemExit("config has no homography")
    if a.points:
        x1, y1, x2, y2 = (float(v) for v in a.points.split(","))
        p1, p2 = (x1, y1), (x2, y2)
    elif a.video:
        p1, p2 = pick(a.video)
    else:
        raise SystemExit("give --video (to click) or --points")
    cfg, k, d = rescale(cfg, p1, p2, a.meters)
    print(f"model said {d:.2f} m, you said {a.meters:.2f} m -> speeds x{k:.2f}")
    if not 0.2 < k < 5:
        raise SystemExit("factor looks implausible; nothing written. Re-check the two points and the distance.")
    shutil.copy(a.config, a.config + ".bak")
    yaml.safe_dump(cfg, open(a.config, "w"), sort_keys=False)
    print("saved", a.config, "(backup .bak)")


if __name__ == "__main__":
    main()
