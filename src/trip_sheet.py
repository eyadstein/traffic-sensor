import argparse, csv
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--video", required=True)
ap.add_argument("--trips", nargs="+", type=int, required=True)
a = ap.parse_args()
rows = list(csv.DictReader(open(f"{a.run}/tracks.csv")))
cap = cv2.VideoCapture(a.video)
W, H = int(cap.get(3)), int(cap.get(4))
sheet = []
for tid in a.trips:
    r = [x for x in rows if int(x["trip_id"]) == tid]
    if not r:
        print("trip", tid, "not found"); continue
    ya, yb = float(r[0]["y2"]), float(r[-1]["y2"])
    print(f"trip {tid}: {float(r[0]['t']):.1f}-{float(r[-1]['t']):.1f}s, bottom y {ya:.0f} -> {yb:.0f}: "
          + ("TOWARD camera" if yb > ya else "AWAY from camera"))
    tiles = []
    for i in np.linspace(0, len(r) - 1, 5).astype(int):
        x = r[i]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(x["frame"]))
        ok, f = cap.read()
        if not ok:
            continue
        x1, y1, x2, y2 = (float(x[k]) for k in ("x1", "y1", "x2", "y2"))
        cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1, 40) * 0.9
        t = cv2.resize(f[int(max(cy - s, 0)):int(min(cy + s, H)), int(max(cx - s, 0)):int(min(cx + s, W))], (180, 180))
        cv2.putText(t, f"#{tid} t={float(x['t']):.1f}s", (4, 16), 0, 0.5, (0, 255, 255), 1)
        tiles.append(t)
    if tiles:
        sheet.append(np.hstack(tiles))
out = f"{a.run}/trip_check.jpg"
cv2.imwrite(out, np.vstack(sheet))
print("saved", out)
