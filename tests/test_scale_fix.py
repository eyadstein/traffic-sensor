import math

from geometry import GroundPlane
from scale_fix import rescale

CFG = {"homography": {"image_points": [[0, 0], [100, 0], [100, 100], [0, 100]],
                      "world_points": [[0, 0], [10, 0], [10, 20], [0, 20]]}}


def test_rescale_makes_model_distance_match_real_distance():
    cfg, k, d = rescale(CFG, (50, 0), (50, 100), 40.0)      # model thinks this is 20 m, you measured 40 m
    assert abs(d - 20) < 1e-3 and abs(k - 2.0) < 1e-3
    g = GroundPlane(cfg["homography"]["image_points"], cfg["homography"]["world_points"])
    a, b = g.to_world((50, 0)), g.to_world((50, 100))
    assert abs(math.hypot(a[0] - b[0], a[1] - b[1]) - 40.0) < 1e-3
