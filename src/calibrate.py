"""Interactive calibration: click ground-reference points + zone polygons on the first frame of YOUR video.

  python src/calibrate.py --video data/my_clip.mp4 --out configs/my_clip.yaml

Step 1  click 4 points on the ROAD SURFACE forming a rectangle of known size
        (lane markings, crosswalk edges, measured on Google Earth). Order: top-left, top-right, bottom-right, bottom-left.
        Then type the real width and length of that rectangle in metres in the terminal.
Step 2  for each zone (N, E, S, W) click polygon corners, press ENTER to finish it. Press S to skip a zone.
        Zones should cover where vehicles ENTER/LEAVE the frame on each arm (extend them past the frame edge if needed).
Keys: ENTER finish polygon | BACKSPACE undo last point | ESC quit
"""
import argparse
import cv2
import numpy as np
import yaml


def click_points(win, img, n=None, color=(0, 255, 255)):
    pts, done = [], [False]

    def cb(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))

    cv2.setMouseCallback(win, cb)
    while True:
        view = img.copy()
        for p in pts:
            cv2.circle(view, p, 4, color, -1)
        if len(pts) > 1:
            cv2.polylines(view, [np.int32(pts)], n is None and False, color, 1)
        cv2.imshow(win, view)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            raise SystemExit("cancelled")
        if k == 8 and pts:
            pts.pop()
        if k in (13, 10) and (n is None and len(pts) >= 3):
            return pts
        if k in (ord("s"), ord("S")) and n is None:
            return None
        if n is not None and len(pts) == n:
            view = img.copy()                       # show the finished shape before returning
            for p in pts:
                cv2.circle(view, p, 4, color, -1)
            cv2.polylines(view, [np.int32(pts)], True, color, 2)
            cv2.imshow(win, view)
            cv2.waitKey(300)
            print("4 points recorded. The window may look frozen while the terminal waits for you to type.")
            return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--zones", default="N,E,S,W")
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("cannot read video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    win = "calibrate"
    cv2.namedWindow(win)

    print("STEP 1: click 4 road-surface points (TL, TR, BR, BL) of a rectangle with known size")
    img_pts = click_points(win, frame, n=4)
    w = float(input("rectangle WIDTH in metres (TL->TR): "))
    l = float(input("rectangle LENGTH in metres (TL->BL): "))
    world = [[0, 0], [w, 0], [w, l], [0, l]]

    zones, canvas = {}, frame.copy()
    cv2.polylines(canvas, [np.int32(img_pts)], True, (0, 255, 0), 2)
    for name in a.zones.split(","):
        print(f"STEP 2: zone {name}: click corners, ENTER to finish, S to skip")
        cv2.putText(canvas, f"zone {name}", (10, 30), 0, 0.9, (0, 255, 255), 2)
        poly = click_points(win, canvas, n=None)
        if poly:
            zones[name] = [list(map(int, p)) for p in poly]
            cv2.polylines(canvas, [np.int32(poly)], True, (255, 200, 0), 2)
            cv2.putText(canvas, name, tuple(np.int32(poly).mean(0).astype(int)), 0, 0.8, (255, 255, 255), 2)
        canvas[:40] = frame[:40]
    cfg = {"fps": float(fps),
           "homography": {"image_points": [list(map(float, p)) for p in img_pts], "world_points": world},
           "zones": zones,
           "tracker": {"buffer_frames": int(fps * 2), "min_frames": 3},
           "detector": {"conf": 0.3, "iou": 0.5},
           "trips": {"dwell_s": 0.4, "lost_after_s": 0.8, "ghost_ttl_s": 4.0, "gate": 6.0},
           "traffic": {"window_s": 30, "eval_every_s": 5, "hold": 2, "min_samples": 20}}
    yaml.safe_dump(cfg, open(a.out, "w"), sort_keys=False)
    cv2.imwrite(a.out.replace(".yaml", "_calib.jpg"), canvas)
    print("saved", a.out, "- edit traffic.free_flow_kmh if you know the posted speed limit")


if __name__ == "__main__":
    main()
