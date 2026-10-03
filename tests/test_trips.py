from collections import Counter

from trips import TripManager

CENT = {"A": (0.0, 0.0), "B": (100.0, 0.0)}      # two zones 100 m apart; A is x<10, B is x>90


def zone(x):
    return "A" if x < 10 else ("B" if x > 90 else None)


def feed(tm, t, tid, x, cls="car"):
    return tm.update(t, [(tid, cls, zone(x), (x, 0.0), True)])


def tick(tm, t):
    return tm.update(t, [])


def test_trip_counts_once_with_correct_od():
    tm = TripManager(CENT)
    for i in range(0, 111):                         # 10 m/s from x=0 to x=110
        feed(tm, i * 0.1, 1, i * 1.0)
    for i in range(111, 140):
        tick(tm, i * 0.1)
    rows = tm.finalize()
    assert tm.od["A"]["B"] == 1 and tm.n_completed == 1
    assert sum(v for row in tm.od.values() for v in row.values()) == 1
    assert [r["status"] for r in rows] == ["COMPLETED"]
    assert rows[0]["origin"] == "A" and rows[0]["dest"] == "B"


def test_brief_touch_of_exit_zone_is_not_a_trip():
    tm = TripManager(CENT)                          # dwell is 0.3 s
    for i in range(0, 89):
        feed(tm, i * 0.1, 1, i * 1.0)
    feed(tm, 8.9, 1, 95.0)
    feed(tm, 9.0, 1, 95.0)                          # only 0.1 s inside B
    feed(tm, 9.1, 1, 50.0)
    for i in range(92, 120):
        tick(tm, i * 0.1)
    tm.finalize()
    assert tm.n_completed == 0
    assert all(v == 0 for row in tm.od.values() for v in row.values())


def occluded_run(stitch, new_cls="car"):
    tm = TripManager(CENT, stitch=stitch)
    for i in range(0, 41):                          # track 1 until x=40
        feed(tm, i * 0.1, 1, i * 1.0)
    for i in range(41, 56):                         # 1.5 s occlusion
        tick(tm, i * 0.1)
    for i in range(56, 111):                        # track 2 shows up where track 1 should be
        feed(tm, i * 0.1, 2, i * 1.0, cls=new_cls)
    for i in range(111, 140):
        tick(tm, i * 0.1)
    return tm, tm.finalize()


def test_ghost_stitching_keeps_one_vehicle_identity():
    tm, rows = occluded_run(stitch=True)
    assert Counter(r["status"] for r in rows) == {"COMPLETED": 1}
    assert len(tm.events) == 1 and rows[0]["n_merges"] == 1
    assert rows[0]["track_ids"] == "1|2" and rows[0]["origin_source"] == "zone"
    assert tm.od["A"]["B"] == 1


def test_without_stitching_the_vehicle_splits_in_two():
    tm, rows = occluded_run(stitch=False)
    assert Counter(r["status"] for r in rows) == {"COMPLETED": 1, "LOST": 1}
    done = [r for r in rows if r["status"] == "COMPLETED"][0]
    assert done["origin_source"] == "guess"         # origin had to be guessed from heading


def test_stitching_refuses_a_different_vehicle_class():
    tm, rows = occluded_run(stitch=True, new_cls="motorcycle")
    assert len(tm.events) == 0


def test_speed_estimate_matches_constant_velocity():
    tm = TripManager(CENT)
    for i in range(0, 31):
        feed(tm, i * 0.1, 1, i * 1.0)               # 10 m/s = 36 km/h
    trip = next(iter(tm.trips.values()))
    assert abs(trip.speed_kmh - 36.0) < 0.5
