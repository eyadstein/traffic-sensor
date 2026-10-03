"""Export one pipeline run (output/<run>/) into a single SQLite file, optionally Parquet too.

  python src/export_db.py --run output/MVI_39031_v4
  python src/export_db.py --run output/MVI_39031_v4 --parquet
"""
import argparse
import json
import os
import sqlite3

import pandas as pd


def read(path, **kw):
    try:
        return pd.read_csv(path, **kw)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--parquet", action="store_true")
    a = ap.parse_args()
    s = json.load(open(os.path.join(a.run, "summary.json")))
    od = read(os.path.join(a.run, "od_matrix.csv"), index_col=0)
    od_long = od.stack().rename("trips").reset_index()
    od_long.columns = ["origin", "dest", "trips"]
    tables = {
        "trips": read(os.path.join(a.run, "trips.csv")),
        "tracks": read(os.path.join(a.run, "tracks.csv")),
        "traffic": read(os.path.join(a.run, "traffic_timeline.csv")),
        "od": od_long,
        "run": pd.DataFrame([{"frames": s["frames"], "fps": s["fps"], "stitching": int(s["stitching"]),
                              "merges": s["merges"], "origin_guessed": s["origin_guessed"]}]),
    }
    db = os.path.join(a.run, "results.sqlite")
    if os.path.exists(db):
        os.remove(db)
    con = sqlite3.connect(db)
    for name, df in tables.items():
        if len(df):
            df.to_sql(name, con, index=False)
            print(f"{name}: {len(df)} rows")
    con.close()
    print("saved", db)
    if a.parquet:
        os.makedirs(os.path.join(a.run, "parquet"), exist_ok=True)
        for name, df in tables.items():
            if len(df):
                df.to_parquet(os.path.join(a.run, "parquet", f"{name}.parquet"), index=False)
        print("saved parquet files (needs pyarrow)")


if __name__ == "__main__":
    main()
