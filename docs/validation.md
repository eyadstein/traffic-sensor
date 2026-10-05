# Validation

## 1. Synthetic intersection (exact ground truth)

`src/synth.py` renders a 4-way intersection through a perspective camera. It has 106 vehicles, three classes,
four traffic phases (smooth, moderate, heavy, smooth), and 23 vehicles hidden for 1.5-2.5 s. The detector is the ground-truth boxes with
jitter and random drop-outs, so this tests the tracking and trip logic, not YOLO.

| | Stitching off | Stitching on |
|---|---|---|
| Vehicles with correct origin and destination | 77.4% (82/106) | 90.6% (96/106) |
| OD matrix error (sum of absolute cell errors) | 25 | 7 |
| Tracker ID breaks after the trip layer | 37 | 9 |
| Speed MAE (km/h) | 1.02 | 1.02 |

The speed error is small because this scene's calibration geometry is known. It says nothing about speed on
real footage with an estimated calibration.

`tests/test_synthetic_benchmark.py` re-runs this comparison on every change and fails if stitching stops helping.

## 2. Real footage: UA-DETRAC MVI_39031

Run: `yolo11n`, FAR/NEAR zones across the road, 59 s (1470 frames), about 22 fps on CPU.
Result: 30 completed trips (18 toward the camera, 12 away), 9 incomplete at the end of the clip, 2 lost.

Hand check: I counted two 10 s windows in the raw video, and compared with
`python src/window_counts.py --run output/<run> --start S --end E`. A trip is counted at completion time.

| Window | Toward camera (hand / pipeline) | Away (hand / pipeline) |
|---|---|---|
| 20-30 s | 5 / 4 | 1 / 1 |
| 40-50 s | 2 / 2 | 2 / 1 |

Within 1 in every cell. Total 10 by hand vs 8 by the pipeline, so it may undercount slightly.

I also looked at crops of three "away" trips: all were real cars driving away. One track appeared to switch
to a different car partway.

### Not validated

- km/h on this clip: the calibration rectangle was estimated, and the implied speeds (about 7 km/h mean) are
  implausibly low. Do not quote them.
- The Smooth/Moderate/Heavy label, which depends on those speeds.
- Anything beyond this one clip and about 20 hand-counted vehicles.

### What would make this stronger

A measured long stretch of road (about 40 m or more) for calibration, a second and third real clip, and the
dataset's own ground-truth annotations for a proper count and ID-switch score.
