from geometry import GroundPlane, Zones, inside
from traffic import TrafficClassifier


def test_homography_maps_pixels_to_metres():
    g = GroundPlane([[0, 0], [100, 0], [100, 100], [0, 100]], [[0, 0], [10, 0], [10, 20], [0, 20]])
    x, y = g.to_world((50, 50))
    assert abs(x - 5) < 1e-3 and abs(y - 10) < 1e-3


def test_zone_lookup():
    z = Zones({"A": [[0, 0], [10, 0], [10, 10], [0, 10]]})
    assert z.lookup((5, 5)) == "A" and z.lookup((50, 50)) is None
    assert inside([[0, 0], [10, 0], [10, 10], [0, 10]], (5, 5))


def test_traffic_label_needs_consecutive_evidence():
    tc = TrafficClassifier(window_s=10, eval_every_s=2, free_flow_kmh=50, hold=2, min_samples=5)
    rows = []
    for i in range(0, 81):
        r = tc.add(i * 0.5, [10.0] * 4, 4)
        if r:
            rows.append(r)
    assert rows[0]["raw"] == "Heavy" and rows[0]["label"] == "Smooth"   # one reading is not enough
    assert rows[1]["label"] == "Heavy"                                    # second one flips it
