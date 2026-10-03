"""Synthetic 4-way intersection with exact ground truth.

World = 80 x 60 m top-down, rendered through a perspective camera (known homography).
  python src/synth.py --out data/synth       -> synth.mp4, synth_gt.json, synth.yaml
"""
import argparse, json, math, os
import cv2, numpy as np, yaml

WW, WH, PPM = 80.0, 60.0, 10
CAM_W, CAM_H, FPS = 960, 540, 15
WORLD_C = np.float32([[0, 0], [WW, 0], [WW, WH], [0, WH]])
IMG_C = np.float32([[250, 40], [710, 40], [940, 520], [20, 520]])      # strong perspective
H_W2I = cv2.getPerspectiveTransform(WORLD_C, IMG_C)

# right-hand traffic. entry (point, dir) / exit (point, dir) per arm
ENTRY = {"N": ((37, 0), (0, 1)), "S": ((43, 60), (0, -1)), "W": ((0, 33), (1, 0)), "E": ((80, 27), (-1, 0))}
EXIT = {"N": ((43, 0), (0, -1)), "S": ((37, 60), (0, 1)), "W": ((0, 27), (-1, 0)), "E": ((80, 33), (1, 0))}
# zones extend 5 m past the road end so vehicles entering at the frame border are inside them
ZONES_W = {"N": [[34, -5], [46, -5], [46, 10], [34, 10]], "E": [[70, 24], [85, 24], [85, 36], [70, 36]],
           "S": [[34, 50], [46, 50], [46, 65], [34, 65]], "W": [[-5, 24], [10, 24], [10, 36], [-5, 36]]}
PHASES = [(0, 30, 1.0, "Smooth"), (30, 60, 0.6, "Moderate"), (60, 95, 0.22, "Heavy"), (95, 1e9, 1.0, "Smooth")]
SPAWN_END, DURATION = 110.0, 140.0


def to_img(pts):
    p = np.float32(pts).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, H_W2I).reshape(-1, 2)


def line_intersection(p, d, q, e):
    A = np.array([[d[0], -e[0]], [d[1], -e[1]]], float)
    s, _ = np.linalg.solve(A, np.array([q[0] - p[0], q[1] - p[1]], float))
    return np.array(p, float) + s * np.array(d, float)


def make_path(o, d):
    (p0, d0), (p1, d1) = ENTRY[o], EXIT[d]
    p0, p1 = np.array(p0, float), np.array(p1, float)
    if (d0[0], d0[1]) == (d1[0], d1[1]):                       # straight through
        pts = [p0, p1]
    else:
        c = line_intersection(p0, d0, p1, d1)
        a = c - 7 * np.array(d0, float)
        b = c + 7 * np.array(d1, float)
        arc = [(1 - u) ** 2 * a + 2 * u * (1 - u) * c + u ** 2 * b for u in np.linspace(0, 1, 15)]
        pts = [p0] + arc + [p1]
    pts = np.array(pts)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    return pts, np.concatenate([[0], np.cumsum(seg)])


def pos_at(path, s):
    pts, cum = path
    s = min(max(s, 0), cum[-1])
    k = min(np.searchsorted(cum, s, side="right") - 1, len(pts) - 2)
    u = (s - cum[k]) / max(cum[k + 1] - cum[k], 1e-9)
    p = pts[k] + u * (pts[k + 1] - pts[k])
    h = (pts[k + 1] - pts[k]) / max(np.linalg.norm(pts[k + 1] - pts[k]), 1e-9)
    return p, h


def factor(t):
    for a, b, f, _ in PHASES:
        if a <= t < b:
            return f
    return 1.0


def base_world():
    img = np.full((int(WH * PPM), int(WW * PPM), 3), (70, 120, 70), np.uint8)
    cv2.rectangle(img, (0, 240), (800, 360), (60, 60, 60), -1)
    cv2.rectangle(img, (340, 0), (460, 600), (60, 60, 60), -1)
    for x in range(0, 800, 40):
        if not 330 < x < 470: cv2.line(img, (x, 300), (x + 20, 300), (200, 200, 200), 2)
    for y in range(0, 600, 40):
        if not 230 < y < 370: cv2.line(img, (400, y), (400, y + 20), (200, 200, 200), 2)
    return img


def simulate(seed=7):
    rng = np.random.default_rng(seed)
    t, dt, nxt, vehicles, frames, info = 0.0, 1.0 / FPS, 0, [], [], {}
    next_spawn = {a: rng.exponential(2) for a in ENTRY}
    while t < DURATION:
        f = factor(t)
        if t < SPAWN_END:
            for arm in ENTRY:
                if t >= next_spawn[arm]:
                    r = rng.random()
                    dest = {"N": {"S": "S", "E": "E", "W": "W"}, "S": {"N": "N", "E": "E", "W": "W"},
                            "W": {"E": "E", "N": "N", "S": "S"}, "E": {"W": "W", "N": "N", "S": "S"}}[arm]
                    keys = list(dest)
                    straight = {"N": "S", "S": "N", "W": "E", "E": "W"}[arm]
                    probs = [0.6 if k == straight else 0.2 for k in keys]
                    d = rng.choice(keys, p=probs)
                    cls = rng.choice(["car", "truck", "bus"], p=[0.85, 0.10, 0.05])
                    L, Wd = {"car": (4.5, 1.9), "truck": (8.0, 2.5), "bus": (11.0, 2.6)}[cls]
                    last = [v for v in vehicles if v["o"] == arm and v["s"] - v["L"] / 2 < 12]
                    if not last:
                        path = make_path(arm, d)
                        vehicles.append({"id": nxt, "o": arm, "d": d, "cls": cls, "L": L, "W": Wd, "path": path,
                                         "s": L / 2, "base": rng.uniform(9.0, 13.5), "t0": t, "v": 0.0, "init": True})
                        info[nxt] = {"origin": arm, "dest": d, "cls": cls, "t0": round(t, 2)}
                        nxt += 1
                        lam = 0.28 if f > 0.5 else 0.22
                        next_spawn[arm] = t + rng.exponential(1 / lam)
                    else:
                        next_spawn[arm] = t + 0.5
        row = []
        for v in sorted(vehicles, key=lambda v: -v["s"]):
            if v.pop("init", False): v["v"] = v["base"] * f        # vehicles enter the view already at cruising speed
            lead = [u for u in vehicles if u["o"] == v["o"] and u["s"] > v["s"] and u["id"] != v["id"]]
            vd = v["base"] * f
            if lead:
                u = min(lead, key=lambda u: u["s"])
                vd = min(vd, max(0.0, (u["s"] - u["L"] / 2 - v["L"] / 2 - 2.5) / 0.5))
            v["v"] += np.clip(vd - v["v"], -6 * dt, 3 * dt)
            v["s"] += v["v"] * dt
        for v in list(vehicles):
            p, h = pos_at(v["path"], v["s"])
            nrm = np.array([-h[1], h[0]])
            corners = np.array([p + h * v["L"] / 2 + nrm * v["W"] / 2, p + h * v["L"] / 2 - nrm * v["W"] / 2,
                                p - h * v["L"] / 2 - nrm * v["W"] / 2, p - h * v["L"] / 2 + nrm * v["W"] / 2])
            ic = to_img(corners)
            x1, y1 = ic.min(0); x2, y2 = ic.max(0)
            row.append({"id": v["id"], "cls": v["cls"], "box": [float(x1), float(y1), float(x2), float(y2)],
                        "kmh": float(v["v"] * 3.6), "xy": [float(p[0]), float(p[1])], "corners": corners.tolist()})
            if v["s"] + v["L"] / 2 >= v["path"][1][-1]:
                vehicles.remove(v)
        frames.append(row)
        t += dt
    # occlusions: ~15% of vehicles vanish 1.5-2.5 s while crossing the middle of their path
    occ = {}
    for vid, meta in info.items():
        if rng.random() < 0.15:
            t_in = [i for i, r in enumerate(frames) if any(x["id"] == vid for x in r)]
            if len(t_in) > 60:
                mid = t_in[int(len(t_in) * rng.uniform(0.35, 0.55))]
                occ[vid] = [mid / FPS, mid / FPS + rng.uniform(1.5, 2.5)]
    return frames, info, occ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    frames, info, occ = simulate(a.seed)
    base = base_world()
    M = H_W2I @ np.diag([1 / PPM, 1 / PPM, 1.0])
    vw = cv2.VideoWriter(a.out + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (CAM_W, CAM_H))
    rng = np.random.default_rng(3)
    cols = {}
    for i, row in enumerate(frames):
        canvas = base.copy()
        for r in row:
            c = cols.setdefault(r["id"], tuple(int(x) for x in rng.integers(60, 255, 3)))
            cv2.fillPoly(canvas, [(np.array(r["corners"]) * PPM).astype(np.int32)], c)
        vw.write(cv2.warpPerspective(canvas, M, (CAM_W, CAM_H)))
    vw.release()
    gt = {"fps": FPS, "frames": [[{k: r[k] for k in ("id", "cls", "box", "kmh", "xy")} for r in row] for row in frames],
          "vehicles": {str(k): v for k, v in info.items()}, "occlusions": {str(k): v for k, v in occ.items()},
          "phases": [{"t0": p[0], "t1": min(p[1], DURATION), "label": p[3]} for p in PHASES]}
    json.dump(gt, open(a.out + "_gt.json", "w"))
    # calibration: 4 corners of the central 12x12 m box, "clicked" with 0.5 px noise
    wp = [[34, 24], [46, 24], [46, 36], [34, 36]]
    ip = (to_img(wp) + np.random.default_rng(5).normal(0, 0.5, (4, 2))).round(1).tolist()
    cfg = {"fps": FPS,
           "homography": {"image_points": ip, "world_points": wp},
           "zones": {k: to_img(v).round(0).astype(int).tolist() for k, v in ZONES_W.items()},
           "tracker": {"buffer_frames": 20, "min_frames": 2},
           "trips": {"dwell_s": 0.3, "lost_after_s": 0.7, "ghost_ttl_s": 4.0, "gate": 6.0},
           "traffic": {"window_s": 20, "eval_every_s": 5, "free_flow_kmh": 45, "hold": 2, "min_samples": 20},
           "det_noise": {"dropout": 0.02, "jitter_px": 1.5}}
    yaml.safe_dump(cfg, open(a.out + ".yaml", "w"))
    n_veh = len(info)
    print(f"frames={len(frames)} vehicles={n_veh} occluded={len(occ)}")


class SyntheticDetector:
    """Ground-truth boxes + noise: jitter, random drop-outs, scripted occlusions (stand-in for YOLO)."""
    def __init__(self, gt_path, seed=1, dropout=0.02, jitter_px=1.5):
        import supervision as sv
        self.sv = sv
        gt = json.load(open(gt_path))
        self.frames, self.fps = gt["frames"], gt["fps"]
        self.occ = {int(k): v for k, v in gt["occlusions"].items()}
        self.rng, self.dropout, self.jit = np.random.default_rng(seed), dropout, jitter_px
        self.cid = {"car": 2, "bus": 5, "truck": 7}

    def __call__(self, frame, i):
        t = i / self.fps
        xyxy, cls = [], []
        for r in self.frames[i]:
            o = self.occ.get(r["id"])
            if (o and o[0] <= t <= o[1]) or self.rng.random() < self.dropout:
                continue
            xyxy.append(np.array(r["box"]) + self.rng.normal(0, self.jit, 4))
            cls.append(self.cid[r["cls"]])
        if not xyxy:
            return self.sv.Detections.empty()
        return self.sv.Detections(xyxy=np.array(xyxy, np.float32), confidence=np.full(len(xyxy), 0.9, np.float32),
                                  class_id=np.array(cls))


if __name__ == "__main__":
    main()
