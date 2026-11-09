"""Lightweight YOLO26 detector running on ONNX Runtime (no PyTorch, no Ultralytics).

    det = Detector("models/yolo26n.onnx")
    results = det.predict([frame1, frame2])     # one Detections per frame

`predict` is thread-safe: several threads can share one Detector (one model in RAM).
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

PERSON_CLASS_ID = 0


@dataclass
class Detections:
    xyxy: np.ndarray   # (n, 4) float32, pixel coordinates of the original frame
    conf: np.ndarray   # (n,)
    cls: np.ndarray    # (n,) int

    def __len__(self) -> int:
        return len(self.cls)

    @property
    def persons(self) -> int:
        return int((self.cls == PERSON_CLASS_ID).sum())

    def filter(self, min_conf: float = 0.0, classes: list[int] | None = None) -> Detections:
        m = self.conf >= min_conf
        if classes is not None:
            m &= np.isin(self.cls, classes)
        return Detections(self.xyxy[m], self.conf[m], self.cls[m])

    def to_list(self, names: dict[int, str]) -> list[dict]:
        return [{"box": [round(float(v), 1) for v in box], "conf": round(float(c), 3),
                 "class_id": int(k), "label": names.get(int(k), str(k))}
                for box, c, k in zip(self.xyxy, self.conf, self.cls)]


# ----------------------------------------------------------------------------- pre / post processing
def letterbox(img: np.ndarray, size: tuple[int, int]):
    """Resize keeping the aspect ratio, pad to size=(h, w) with gray (same as Ultralytics)."""
    h, w = img.shape[:2]
    sh, sw = size
    r = min(sh / h, sw / w)
    nw, nh = round(w * r), round(h * r)
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    left, top = (sw - nw) // 2, (sh - nh) // 2
    out = np.full((sh, sw, 3), 114, np.uint8)
    out[top:top + nh, left:left + nw] = img
    return out, r, (left, top)


def to_blob(imgs: list[np.ndarray]) -> np.ndarray:
    """BGR HWC uint8 -> RGB NCHW float32 [0, 1]."""
    x = np.stack(imgs)[..., ::-1].transpose(0, 3, 1, 2)
    return np.ascontiguousarray(x, dtype=np.float32) / 255.0


def postprocess(out: np.ndarray, r: float, pad: tuple[int, int], shape, conf: float, iou: float,
                classes: list[int] | None, max_det: int = 300) -> Detections:
    """`out` is the output for ONE image: raw (4+nc, anchors) or end-to-end (max_det, 6)."""
    if out.ndim == 2 and out.shape[-1] == 6:           # end2end: x1, y1, x2, y2, score, cls
        out = out[out[:, 4] >= conf]
        boxes, scores, cls = out[:, :4].copy(), out[:, 4], out[:, 5].astype(int)
        if classes is not None:
            m = np.isin(cls, classes)
            boxes, scores, cls = boxes[m], scores[m], cls[m]
    else:                                               # raw: needs NMS
        p = out.T                                       # (anchors, 4 + nc)
        cls = p[:, 4:].argmax(1)
        scores = p[np.arange(len(p)), 4 + cls]
        m = scores >= conf
        if classes is not None:
            m &= np.isin(cls, classes)
        p, cls, scores = p[m], cls[m], scores[m]
        xy, wh = p[:, :2], p[:, 2:4]
        boxes = np.c_[xy - wh / 2, xy + wh / 2]
        if len(boxes):
            keep = cv2.dnn.NMSBoxesBatched(np.c_[boxes[:, :2], wh].tolist(), scores.tolist(), cls.tolist(), conf, iou)
            keep = np.asarray(keep, dtype=int).reshape(-1)
            keep = keep[np.argsort(-scores[keep])][:max_det]
            boxes, scores, cls = boxes[keep], scores[keep], cls[keep]

    boxes = (boxes - [pad[0], pad[1], pad[0], pad[1]]) / r
    h, w = shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return Detections(boxes.astype(np.float32), scores.astype(np.float32), cls.astype(int))


def _hw(v) -> tuple[int, int]:
    v = list(v) if isinstance(v, (list, tuple)) else [v]
    return int(v[0]), int(v[-1])


# ----------------------------------------------------------------------------- detector
class Detector:
    def __init__(self, model_path: str | Path, *, conf: float = 0.25, iou: float = 0.7,
                 classes: list[int] | None = None, threads: int | None = None, device: str = "CPU"):
        self.path = Path(model_path)
        if self.path.suffix != ".onnx" or not self.path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.path} (run export_model.py first)")
        self.conf, self.iou, self.classes = conf, iou, classes

        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = int(threads)
            so.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        if device.upper() in ("CUDA", "GPU") and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        self._sess = ort.InferenceSession(str(self.path), so, providers=providers)

        inp = self._sess.get_inputs()[0]
        self._in_name = inp.name
        self._in_batch = inp.shape[0] if isinstance(inp.shape[0], int) else None   # None = dynamic
        meta = self._sess.get_modelmeta().custom_metadata_map
        self.imgsz = _hw(ast.literal_eval(meta["imgsz"])) if "imgsz" in meta else _hw(inp.shape[2:4])
        names = ast.literal_eval(meta["names"]) if "names" in meta else {}
        self.names: dict[int, str] = {int(k): v for k, v in names.items()} or {i: str(i) for i in range(80)}

    @property
    def provider(self) -> str:
        return self._sess.get_providers()[0]

    def _infer(self, blob: np.ndarray) -> np.ndarray:
        if self._in_batch is None or self._in_batch == len(blob):
            return self._sess.run(None, {self._in_name: blob})[0]
        # fixed batch-1 model: one call per image
        return np.concatenate([self._sess.run(None, {self._in_name: blob[i:i + 1]})[0] for i in range(len(blob))])

    def predict(self, frames: list[np.ndarray] | np.ndarray) -> list[Detections]:
        if isinstance(frames, np.ndarray):
            frames = [frames]
        boxed = [letterbox(f, self.imgsz) for f in frames]
        out = self._infer(to_blob([b[0] for b in boxed]))
        return [postprocess(out[i], r, pad, f.shape, self.conf, self.iou, self.classes)
                for i, (f, (_, r, pad)) in enumerate(zip(frames, boxed))]

    def warmup(self) -> None:
        self.predict(np.zeros((*self.imgsz, 3), np.uint8))
