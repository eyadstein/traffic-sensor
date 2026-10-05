"""How many trips did the pipeline complete in a time window? Use it to compare with a hand count.

  python src/window_counts.py --run output/MVI_39031_v4 --start 20 --end 30

Counts a trip at the moment it was completed (t_complete), not when the vehicle was last seen.
"""
import argparse

import pandas as pd


def window_counts(trips, start, end):
    """Return {(origin, dest): n} for trips completed in [start, end) seconds."""
    if "t_complete" not in trips:
        raise ValueError("trips.csv has no t_complete column; rerun the pipeline with the current code")
    w = trips[(trips["status"] == "COMPLETED") & (trips["t_complete"] >= start) & (trips["t_complete"] < end)]
    return {k: int(v) for k, v in w.groupby(["origin", "dest"]).size().items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    a = ap.parse_args()
    counts = window_counts(pd.read_csv(f"{a.run}/trips.csv"), a.start, a.end)
    print(f"{a.start:g}-{a.end:g} s:", counts if counts else "no completed trips")


if __name__ == "__main__":
    main()
