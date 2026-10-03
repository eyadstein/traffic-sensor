"""YOLO11 ONNX detector (CPU, no torch). Output: supervision.Detections of vehicles."""
import cv2
import numpy as np
import onnxruntime as ort
import supervision as sv

COCO_VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class YoloOnnx:
    def __init__(self, path, conf=0.30, iou=0.5, imgsz=640, classes=tuple(COCO_VEHICLES)):
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.conf, self.iou, self.imgsz, self.classes = conf, iou, imgsz, np.array(classes)

    def __call__(self, frame, frame_idx=None):
        h, w = frame.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nh, nw = round(h * r), round(w * r)
        top, left = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(frame, (nw, nh))
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.sess.run(None, {self.inp: blob})[0][0].T          # (anchors, 84)
        scores_all = out[:, 4:]
        cls = scores_all.argmax(1)
        conf = scores_all.max(1)
        keep = np.isin(cls, self.classes) & (conf >= self.conf)
        if not keep.any():
            return sv.Detections.empty()
        b, cls, conf = out[keep, :4], cls[keep], conf[keep]
        x1 = (b[:, 0] - b[:, 2] / 2 - left) / r
        y1 = (b[:, 1] - b[:, 3] / 2 - top) / r
        x2 = (b[:, 0] + b[:, 2] / 2 - left) / r
        y2 = (b[:, 1] + b[:, 3] / 2 - top) / r
        xyxy = np.stack([x1, y1, x2, y2], 1).clip([0, 0, 0, 0], [w, h, w, h])
        idx = cv2.dnn.NMSBoxes([[float(a), float(b_), float(c - a), float(d - b_)] for a, b_, c, d in xyxy],
                               conf.tolist(), self.conf, self.iou)
        idx = np.array(idx).reshape(-1)
        return sv.Detections(xyxy=xyxy[idx].astype(np.float32), confidence=conf[idx].astype(np.float32),
                             class_id=cls[idx].astype(int))
