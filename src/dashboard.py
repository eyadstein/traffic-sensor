"""Small Streamlit dashboard for a pipeline run in output/<run>/.

  streamlit run src/dashboard.py
"""
import json
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
st.set_page_config(page_title="Traffic Sensor", layout="wide")
st.title("CCTV traffic sensor")

runs = []
if os.path.isdir(OUT):
    runs = sorted(d for d in os.listdir(OUT) if os.path.exists(os.path.join(OUT, d, "summary.json")))
if not runs:
    st.error("No runs found in output/. Run src/pipeline.py first.")
    st.stop()
run = st.sidebar.selectbox("Run", runs, index=len(runs) - 1)
base = os.path.join(OUT, run)


@st.cache_data
def load(base, stamp):
    def rd(name, **kw):
        try:
            return pd.read_csv(os.path.join(base, name), **kw)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return pd.DataFrame()
    s = json.load(open(os.path.join(base, "summary.json")))
    return s, rd("trips.csv"), rd("od_matrix.csv", index_col=0), rd("traffic_timeline.csv"), rd("tracks.csv")


s, trips, od, tl, tracks = load(base, os.path.getmtime(os.path.join(base, "summary.json")))
st.sidebar.write(f"Stitching: {'on' if s.get('stitching') else 'off'}")
done = trips[trips.status == "COMPLETED"] if len(trips) else trips
by = s.get("trips_by_status", {})

c = st.columns(5)
c[0].metric("Duration", f"{s['frames'] / s['fps']:.0f} s")
c[1].metric("Completed trips", int(by.get("COMPLETED", 0)))
c[2].metric("Incomplete / lost", int(by.get("INCOMPLETE", 0) + by.get("LOST", 0)))
c[3].metric("ID merges", s.get("merges", 0))
c[4].metric("Guessed origins", s.get("origin_guessed", 0))

left, right = st.columns(2)
with left:
    st.subheader("Origin-destination matrix")
    long = od.stack().rename("trips").reset_index()
    long.columns = ["origin", "dest", "trips"]
    b = alt.Chart(long).encode(x=alt.X("dest:N", title="destination"), y=alt.Y("origin:N", title="origin"))
    heat = b.mark_rect().encode(color=alt.Color("trips:Q", scale=alt.Scale(scheme="blues")))
    st.altair_chart((heat + b.mark_text(size=18).encode(text="trips:Q")).properties(width="container", height=260))
with right:
    st.subheader("Traffic condition over time")
    if len(tl):
        st.line_chart(tl.set_index("t")[["mean_kmh", "flow_veh_min"]])
        st.caption("Label counts: " + ", ".join(f"{k} {v}" for k, v in tl["label"].value_counts().items()))
    else:
        st.info("Not enough samples for a traffic label in this run.")
st.caption("Speeds and the traffic label depend on the calibration rectangle size in the config. "
           "If that size was estimated rather than measured, treat km/h and the label as approximate.")

left, right = st.columns(2)
with left:
    st.subheader("Trip speeds (km/h, mean per trip)")
    sp = done["mean_kmh"].dropna() if len(done) else pd.Series(dtype=float)
    if len(sp) > 1:
        h, e = np.histogram(sp, bins=10)
        st.bar_chart(pd.DataFrame({"trips": h}, index=np.round(e[:-1], 1)))
    else:
        st.info("No speed data (run without a homography).")
with right:
    st.subheader("Where vehicles travel (footpoints)")
    if len(tracks):
        pts = tracks.assign(x=(tracks.x1 + tracks.x2) / 2, y=-tracks.y2)
        st.scatter_chart(pts.sample(min(len(pts), 4000), random_state=0), x="x", y="y", color="cls")

vid = os.path.join(base, "annotated.mp4")
st.subheader("Annotated video")
if os.path.exists(vid):
    import cv2
    cap = cv2.VideoCapture(vid)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    t = st.slider("Time (s)", 0.0, max(dur - 0.1, 1.0), 0.0, 0.5)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, frame = cap.read()
    if ok:
        st.image(frame, channels="BGR")
else:
    st.caption("No annotated.mp4 in this run (it was run with --no-video).")

st.subheader("Trips")
if len(trips):
    pick = st.multiselect("Status", sorted(trips.status.unique()), default=sorted(trips.status.unique()))
    view = trips[trips.status.isin(pick)]
    st.dataframe(view)
    st.download_button("Download CSV", view.to_csv(index=False), file_name=f"{run}_trips.csv")
