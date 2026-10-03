"""Score a pipeline run against synthetic ground truth.
python src/evaluate.py --gt data/synth_gt.json --run output/synth_stitch
"""
import argparse, csv, json
from collections import Counter, defaultdict
import numpy as np


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0


def evaluate(gt_path, run, verbose=True):
    gt = json.load(open(gt_path))
    rows = list(csv.DictReader(open(f"{run}/tracks.csv")))
    trips = list(csv.DictReader(open(f"{run}/trips.csv")))
    by_frame = defaultdict(list)
    for r in rows:
        by_frame[int(r["frame"])].append(r)

    track2vid, trip2vid_votes, vid_tracks = defaultdict(Counter), defaultdict(Counter), defaultdict(list)
    speed_err, speed_rows = [], 0
    for f, rs in by_frame.items():
        g = gt["frames"][f]
        for r in rs:
            box = [float(r[k]) for k in ("x1", "y1", "x2", "y2")]
            sc = sorted(((iou(box, x["box"]), x) for x in g), key=lambda z: -z[0])
            if not sc or sc[0][0] < 0.4:
                continue
            best, strict = sc[0][1], not (len(sc) > 1 and sc[1][0] > 0.15)   # strict = unambiguous match
            vid = best["id"]
            track2vid[r["track_id"]][vid] += 1
            trip2vid_votes[r["trip_id"]][vid] += 1
            if strict:
                vid_tracks[vid].append((f, r["track_id"], r["trip_id"]))
            if r["speed_kmh"]:
                speed_err.append(float(r["speed_kmh"]) - best["kmh"]); speed_rows += 1

    # ID switches: tracker-level vs trip-level (after stitching)
    def switches(idx):
        n = 0
        for vid, seq in vid_tracks.items():
            seq = sorted(seq)
            n += sum(1 for a, b in zip(seq, seq[1:]) if a[idx] != b[idx])
        return n
    sw_track, sw_trip = switches(1), switches(2)

    # trip -> vehicle, OD per vehicle
    vehicles = gt["vehicles"]
    vid_best = {}
    for tr in trips:
        votes = trip2vid_votes.get(tr["trip_id"])
        if not votes:
            continue
        vid = votes.most_common(1)[0][0]
        if tr["status"] == "COMPLETED":
            vid_best.setdefault(vid, []).append(tr)
    ok = wrong_o = wrong_d = missing = 0
    for vid, meta in vehicles.items():
        c = vid_best.get(int(vid), [])
        if not c:
            missing += 1
        elif any(t["origin"] == meta["origin"] and t["dest"] == meta["dest"] for t in c):
            ok += 1
        else:
            wrong_o += c[0]["origin"] != meta["origin"]; wrong_d += c[0]["dest"] != meta["dest"]
    names = ["N", "E", "S", "W"]
    truth = {o: {d: 0 for d in names} for o in names}
    for v in vehicles.values(): truth[v["origin"]][v["dest"]] += 1
    od = json.load(open(f"{run}/summary.json"))["od"]
    abs_err = sum(abs(truth[o][d] - od[o][d]) for o in names for d in names)
    n_true = len(vehicles)
    n_done = sum(od[o][d] for o in names for d in names)

    # traffic labels
    tl = list(csv.DictReader(open(f"{run}/traffic_timeline.csv"))) if True else []
    def phase(t):
        for p in gt["phases"]:
            if p["t0"] <= t < p["t1"]: return p["label"]
    margin, hit, tot = 12, 0, 0
    for r in tl:
        t = float(r["t"])
        if any(abs(t - p["t0"]) < margin for p in gt["phases"][1:]):   # skip around phase transitions
            continue
        tot += 1; hit += r["label"] == phase(t)
    e = np.array(speed_err) if speed_err else np.array([0.0])
    res = {"true_vehicles": n_true, "completed_trips": n_done, "vehicles_with_correct_OD": ok,
           "OD_per_vehicle_accuracy": round(ok / n_true, 3), "missing": missing, "wrong_origin": int(wrong_o),
           "wrong_dest": int(wrong_d), "OD_matrix_abs_error": abs_err,
           "id_switches_raw_tracker": sw_track, "id_switches_after_stitch": sw_trip,
           "speed_samples": speed_rows, "speed_MAE_kmh": round(float(np.abs(e).mean()), 2),
           "speed_bias_kmh": round(float(e.mean()), 2), "speed_P90_abs_kmh": round(float(np.percentile(np.abs(e), 90)), 2),
           "traffic_label_accuracy": round(hit / tot, 3) if tot else None, "traffic_eval_points": tot,
           "truth_OD": truth, "pred_OD": od}
    if verbose:
        for k, v in res.items():
            if not isinstance(v, dict): print(f"{k:32s} {v}")
        print("truth OD", {o: list(truth[o].values()) for o in names}); print("pred  OD", {o: list(od[o].values()) for o in names})
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--gt", required=True); ap.add_argument("--run", required=True)
    a = ap.parse_args(); evaluate(a.gt, a.run)
