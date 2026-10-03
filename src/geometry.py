"""Ground-plane homography (pixels -> meters) and zone lookup."""
import cv2
import numpy as np


class GroundPlane:
    def __init__(self, image_pts, world_pts):
        self.H = cv2.getPerspectiveTransform(np.float32(image_pts), np.float32(world_pts))

    def to_world(self, xy):
        p = np.array([[[float(xy[0]), float(xy[1])]]], np.float32)
        x, y = cv2.perspectiveTransform(p, self.H)[0, 0]
        return float(x), float(y)


class Zones:
    def __init__(self, polys: dict):
        self.polys = {k: np.int32(v).reshape(-1, 1, 2) for k, v in polys.items()}
        self.names = list(polys)

    def lookup(self, xy):
        for name, poly in self.polys.items():
            if cv2.pointPolygonTest(poly, (float(xy[0]), float(xy[1])), False) >= 0:
                return name
        return None

    def centroids_px(self):
        return {k: tuple(p.reshape(-1, 2).mean(0)) for k, p in self.polys.items()}


def inside(poly, xy):
    return cv2.pointPolygonTest(np.int32(poly).reshape(-1, 1, 2), (float(xy[0]), float(xy[1])), False) >= 0
