"""CCTV traffic sensor: detect -> track -> speed -> trips/OD -> traffic condition.

python src/pipeline.py --video data/x.mp4 --config configs/x.yaml --out output/run1
"""
import argparse, csv, json, os, sys, time
import cv2, numpy as np, supervision as sv, yaml

sys.path.insert(0, os.path.dirname(__file__))
from detector import COCO_VEHICLES, YoloOnnx
from geometry import GroundPlane, Zones, inside
from traffic import TrafficClassifier
from trips import TripManager

COLORS = {"Smooth": (80, 200, 80), "Moderate": (0, 190, 255), "Heavy": (60, 60, 230)}


def build(cfg, args):
    zones = Zones(cfg["zones"])
    plane = None
    if "homography" in cfg:
        plane = GroundPlane(cfg["homography"]["image_points"], cfg["homography"]["world_points"])
    cents = {k: (plane.to_world(c) if plane else c) for k, c in zones.centroids_px().items()}
    tcfg = dict(cfg.get("trips", {}))
    if plane is None:                                   # pixel mode: no metres -> scale gates to pixels
        tcfg.setdefault("gate", 80.0); tcfg.setdefault("guess_slack", 150.0); tcfg.setdefault("min_move", 20.0)
    tm = TripManager(cents, metric=plane is not None, stitch=not args.no_stitch, **tcfg)
    tr = cfg.get("tracker", {})
    tracker = sv.ByteTrack(track_activation_threshold=tr.get("activation", 0.25),
                           lost_track_buffer=tr.get("buffer_frames", 30),
                           minimum_matching_threshold=tr.get("match_thresh", 0.8),
                           frame_rate=int(round(cfg["fps"])), minimum_consecutive_frames=tr.get("min_frames", 2))
    tc = TrafficClassifier(**cfg.get("traffic", {}))
    return zones, plane, tm, tracker, tc


def draw(frame, zones, tracks_info, tm, label, metrics, t):
    ov = frame.copy()
    for name, poly in zones.polys.items():
        cv2.fillPoly(ov, [poly], (255, 200, 0))
    frame = cv2.addWeighted(ov, 0.18, frame, 0.82, 0)
    for name, poly in zones.polys.items():
        cv2.polylines(frame, [poly], True, (255, 200, 0), 1)
        cx, cy = poly.reshape(-1, 2).mean(0).astype(int)
        cv2.putText(frame, name, (cx - 6, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for (x1, y1, x2, y2), trip, tid in tracks_info:
        c = (0, 255, 0) if trip.status == "ACTIVE" else (200, 200, 200)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), c, 1)
        sp = "" if trip.speed_kmh is None else f" {trip.speed_kmh:.0f}km/h"
        cv2.putText(frame, f"#{trip.id} {trip.cls}{sp}", (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
    names = list(tm.od)
    lines = ["OD   " + " ".join(f"{n:>3}" for n in names)] + \
            [f"{o:>3}  " + " ".join(f"{tm.od[o][d]:>3}" for d in names) for o in names]
    cv2.rectangle(frame, (6, 6), (230, 6 + 20 * (len(lines) + 2)), (0, 0, 0), -1)
    for i, ln in enumerate(lines):
        cv2.putText(frame, ln, (12, 24 + 20 * i), cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 255, 255), 1)
    y = 24 + 20 * len(lines)
    cv2.putText(frame, f"{label}", (12, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS.get(label, (255, 255, 255)), 2)
    if metrics:
        cv2.putText(frame, f"{metrics['mean_kmh']} km/h  t={t:.0f}s", (100, y + 4), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--detector", default="onnx", choices=["onnx", "gt"])
    ap.add_argument("--model", default="data/yolo11n.onnx")
    ap.add_argument("--gt", help="gt json (for --detector gt)")
    ap.add_argument("--no-stitch", action="store_true", help="ablation: disable ghost re-identification")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = yaml.safe_load(open(args.config))
    cap = cv2.VideoCapture(args.video)
    cfg["fps"] = cap.get(cv2.CAP_PROP_FPS) or cfg.get("fps", 30)
    fps, W, H = cfg["fps"], int(cap.get(3)), int(cap.get(4))
    zones, plane, tm, tracker, tc = build(cfg, args)
    roi = cfg.get("valid_roi")
    if args.detector == "gt":
        from synth import SyntheticDetector
        det = SyntheticDetector(args.gt, seed=cfg.get("det_seed", 1), **cfg.get("det_noise", {}))
    else:
        det = YoloOnnx(args.model, **cfg.get("detector", {}))
    writer = None if args.no_video else cv2.VideoWriter(os.path.join(args.out, "annotated.mp4"),
                                                        cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    f_tracks = open(os.path.join(args.out, "tracks.csv"), "w", newline="")
    wt = csv.writer(f_tracks)
    wt.writerow(["frame", "t", "track_id", "trip_id", "cls", "x1", "y1", "x2", "y2", "zone", "speed_kmh"])

    i, t_wall, label, metrics, prev_done = 0, time.time(), "Smooth", None, 0
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and i >= args.max_frames):
            break
        t = i / fps
        dets = det(frame, i)
        tracks = tracker.update_with_detections(dets)
        obs, boxes = [], []
        for xyxy, cid, tid in zip(tracks.xyxy, tracks.class_id, tracks.tracker_id):
            if tid is None or tid < 0:
                continue
            a = (float(np.clip((xyxy[0] + xyxy[2]) / 2, 0, W - 1)), float(np.clip(xyxy[3], 0, H - 1)))  # bottom-centre = ground contact
            zone = zones.lookup(a)
            xy = plane.to_world(a) if plane else a
            obs.append((int(tid), COCO_VEHICLES.get(int(cid), "car"), zone, xy, roi is None or inside(roi, a)))
            boxes.append((xyxy, zone))
        res = tm.update(t, obs)
        info = []
        for (xyxy, zone), (tid, *_ ) in zip(boxes, obs):
            trip = res[tid]
            info.append((xyxy, trip, tid))
            wt.writerow([i, round(t, 3), tid, trip.id, trip.cls, *np.round(xyxy, 1), zone,
                         "" if trip.speed_kmh is None else round(trip.speed_kmh, 2)])
        speeds = [tr.speed_kmh for _, tr, _ in info if tr.speed_kmh is not None] if plane else []
        row = tc.add(t, speeds, len(info), tm.n_completed - prev_done)
        prev_done = tm.n_completed
        if row:
            label, metrics = row["label"], row
        if writer is not None:
            writer.write(draw(frame, zones, info, tm, label, metrics, t))
        i += 1
        if i % 300 == 0:
            print(f"frame {i}  t={t:.0f}s  {i / (time.time() - t_wall):.1f} fps  completed={tm.n_completed}", flush=True)
    cap.release()
    f_tracks.close()
    if writer:
        writer.release()

    trips = tm.finalize()
    with open(os.path.join(args.out, "trips.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trips[0].keys()) if trips else ["trip_id"])
        w.writeheader(); w.writerows(trips)
    names = list(tm.od)
    with open(os.path.join(args.out, "od_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["origin\\dest"] + names)
        for o in names:
            w.writerow([o] + [tm.od[o][d] for d in names])
    with open(os.path.join(args.out, "traffic_timeline.csv"), "w", newline="") as f:
        if tc.timeline:
            w = csv.DictWriter(f, fieldnames=list(tc.timeline[0].keys())); w.writeheader(); w.writerows(tc.timeline)
    json.dump(tm.events, open(os.path.join(args.out, "merge_events.json"), "w"), indent=1)
    st = {}
    for t_ in trips:
        st[t_["status"]] = st.get(t_["status"], 0) + 1
    summary = {"frames": i, "fps": fps, "stitching": not args.no_stitch, "trips_by_status": st,
               "merges": len(tm.events), "od": tm.od,
               "origin_guessed": sum(1 for t_ in trips if t_["origin_source"] == "guess")}
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
