"""Drawing helpers (boxes + small info panel) shared by the scripts and the API."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from engine import PERSON_CLASS_ID, Detections

ROOT = Path(__file__).resolve().parent
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v", ".mpeg", ".mpg"}

PERSON_COLOR = (0, 220, 0)
_PALETTE = [(255, 56, 56), (255, 112, 31), (255, 178, 29), (0, 0, 255), (255, 0, 255), (0, 128, 255),
            (203, 56, 255), (0, 212, 255), (128, 0, 128), (44, 153, 168), (52, 69, 147), (100, 115, 255)]
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def list_videos(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def color_for(cls_id: int) -> tuple[int, int, int]:
    return PERSON_COLOR if cls_id == PERSON_CLASS_ID else _PALETTE[cls_id % len(_PALETTE)]


def draw_detections(frame: np.ndarray, det: Detections, names: dict) -> None:
    t = max(1, round(sum(frame.shape[:2]) / 600))
    fs = t / 3
    for (x1, y1, x2, y2), conf, c in zip(det.xyxy.astype(int), det.conf, det.cls):
        color = color_for(int(c))
        label = f"{names.get(int(c), c)} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, t, cv2.LINE_AA)
        (tw, th), base = cv2.getTextSize(label, _FONT, fs, max(1, t - 1))
        yt = y1 - 4 if y1 - th - 6 > 0 else y1 + th + 4
        cv2.rectangle(frame, (x1, yt - th - 4), (x1 + tw + 4, yt + base - 2), color, -1)
        cv2.putText(frame, label, (x1 + 2, yt - 2), _FONT, fs, (255, 255, 255), max(1, t - 1), cv2.LINE_AA)


def draw_hud(frame: np.ndarray, lines: list[str]) -> None:
    s = max(0.5, frame.shape[0] / 1000)
    th = int(28 * s)
    w = max(cv2.getTextSize(line, _FONT, s, 2)[0][0] for line in lines) + 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, th * len(lines) + 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, th * (i + 1)), _FONT, s, (255, 255, 255), 2, cv2.LINE_AA)


def annotate(frame: np.ndarray, det: Detections, names: dict, hud: list[str] | None = None) -> np.ndarray:
    """Draw boxes (+ optional info panel) in place and return the frame."""
    draw_detections(frame, det, names)
    if hud:
        draw_hud(frame, hud)
    return frame
