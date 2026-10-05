"""End-to-end regression check on the synthetic intersection (exact ground truth).

Fails if a change makes stitching stop helping, or makes OD / speed accuracy collapse."""
import os
import subprocess
import sys

import pytest

from evaluate import evaluate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(*args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True, capture_output=True)


@pytest.mark.slow
def test_stitching_improves_od_accuracy_on_synthetic_scene(tmp_path):
    base = str(tmp_path / "synth")
    run("src/synth.py", "--out", base)
    res = {}
    for name, extra in (("stitch", []), ("nostitch", ["--no-stitch"])):
        out = str(tmp_path / name)
        run("src/pipeline.py", "--video", base + ".mp4", "--config", base + ".yaml", "--detector", "gt",
            "--gt", base + "_gt.json", "--out", out, "--no-video", *extra)
        res[name] = evaluate(base + "_gt.json", out, verbose=False)
    s, n = res["stitch"], res["nostitch"]
    assert s["OD_per_vehicle_accuracy"] >= 0.85
    assert s["OD_per_vehicle_accuracy"] > n["OD_per_vehicle_accuracy"] + 0.05
    assert s["id_switches_after_stitch"] < s["id_switches_raw_tracker"]
    assert s["speed_MAE_kmh"] < 3.0
