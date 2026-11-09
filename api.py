"""FastAPI interface for YOLO ONNX detection: image, video and webcam.

    python api.py                                   # http://127.0.0.1:8000
    python api.py --model models/yolo26n_384x640.onnx --host 0.0.0.0 --port 8000
    uvicorn api:app                                 # model from the MODEL_PATH env variable

Endpoints:
    GET  /                          web interface (static/index.html)
    GET  /api/health                model info
    POST /api/image                 image -> detections (+ annotated JPEG in base64)
    POST /api/video                 upload a video -> {"id"}
    GET  /api/video/{id}/stream     annotated MJPEG stream while the video is processed
    GET  /api/video/{id}            processing status / stats
    GET  /api/video/{id}/download   annotated .mp4 once processing is done

One Detector is shared by all requests (ONNX Runtime sessions are thread-safe).
The webcam page sends frames to POST /api/image?annotate=false and draws boxes in the browser.
"""
from __future__ import annotations

import argparse
import base64
import os
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from drawing import ROOT, VIDEO_EXTS, annotate
from engine import PERSON_CLASS_ID, Detector

DEFAULT_MODEL = ROOT / "models" / "yolo26n.onnx"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "yolo_onnx_uploads"
OUTPUT_DIR = ROOT / "output_videos"


@dataclass
class VideoJob:
    src: Path
    out: Path
    status: str = "uploaded"          # uploaded -> processing -> done | error
    frames: int = 0
    total_frames: int = 0
    max_persons: int = 0
    infer_ms: float = 0.0
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def stats(self) -> dict:
        return {"status": self.status, "frames": self.frames, "total_frames": self.total_frames,
                "max_persons": self.max_persons, "error": self.error,
                "avg_inference_ms": round(self.infer_ms / max(1, self.frames), 1)}


state: dict = {"det": None, "jobs": {}}


@asynccontextmanager
async def lifespan(_: FastAPI):
    det = Detector(os.environ.get("MODEL_PATH", DEFAULT_MODEL), conf=float(os.environ.get("CONF", 0.25)))
    det.warmup()
    state["det"] = det
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Model {det.path.name} loaded ({det.provider}, input {det.imgsz})")
    yield


app = FastAPI(title="YOLO ONNX detection", lifespan=lifespan)


def detector() -> Detector:
    return state["det"]


def classes_for(persons_only: bool) -> list[int] | None:
    return [PERSON_CLASS_ID] if persons_only else None


def get_job(job_id: str) -> VideoJob:
    job = state["jobs"].get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown video id")
    return job


# ----------------------------------------------------------------------------- pages
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health():
    det = detector()
    return {"model": det.path.name, "provider": det.provider, "input_size": det.imgsz,
            "base_conf": det.conf, "classes": len(det.names)}


# ----------------------------------------------------------------------------- image / webcam frame
@app.post("/api/image")
def detect_image(file: UploadFile = File(...),
                 conf: float = Query(0.25, ge=0, le=1, description="minimum confidence"),
                 persons_only: bool = False,
                 annotate_image: bool = Query(True, alias="annotate", description="return an annotated JPEG")):
    frame = cv2.imdecode(np.frombuffer(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Unreadable image")
    det = detector()
    t = time.perf_counter()
    res = det.predict(frame)[0].filter(conf, classes_for(persons_only))
    infer_ms = (time.perf_counter() - t) * 1000

    body = {"width": frame.shape[1], "height": frame.shape[0], "inference_ms": round(infer_ms, 1),
            "count": len(res), "persons": res.persons, "detections": res.to_list(det.names)}
    if annotate_image:
        annotate(frame, res, det.names)
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        body["image"] = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()
    return body


# ----------------------------------------------------------------------------- video
@app.post("/api/video")
def upload_video(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(400, f"Unsupported video type '{ext}'")
    job_id = uuid.uuid4().hex[:12]
    src = UPLOAD_DIR / f"{job_id}{ext}"
    with src.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    state["jobs"][job_id] = VideoJob(src=src, out=OUTPUT_DIR / f"{job_id}_{Path(file.filename).stem}.mp4")
    return {"id": job_id}


@app.get("/api/video/{job_id}")
def video_status(job_id: str):
    return get_job(job_id).stats()


@app.get("/api/video/{job_id}/stream")
def video_stream(job_id: str, conf: float = Query(0.25, ge=0, le=1), persons_only: bool = False):
    job = get_job(job_id)
    if not job.lock.acquire(blocking=False):
        raise HTTPException(409, "This video is already being processed")
    det = detector()
    classes = classes_for(persons_only)

    def frames():
        cap = cv2.VideoCapture(str(job.src))
        writer = None
        job.status, job.frames, job.max_persons, job.infer_ms, job.error = "processing", 0, 0, 0.0, None
        job.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                t = time.perf_counter()
                res = det.predict(frame)[0].filter(conf, classes)
                job.infer_ms += (time.perf_counter() - t) * 1000
                job.frames += 1
                job.max_persons = max(job.max_persons, res.persons)
                annotate(frame, res, det.names, [f"Frame {job.frames}/{job.total_frames}",
                                                 f"Persons: {res.persons}"])
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(job.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(frame)
                ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            job.status = "done"
        except Exception as e:
            job.status, job.error = "error", repr(e)
        finally:
            if job.status == "processing":    # client disconnected before the end
                job.status = "stopped"
            cap.release()
            if writer:
                writer.release()
            job.lock.release()

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/video/{job_id}/download")
def video_download(job_id: str):
    job = get_job(job_id)
    if job.status != "done" or not job.out.exists():
        raise HTTPException(409, f"Video not ready (status: {job.status})")
    return FileResponse(job.out, media_type="video/mp4", filename=job.out.name)


# ----------------------------------------------------------------------------- run
def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description="YOLO ONNX detection API")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--conf", type=float, default=0.25, help="base confidence (requests can only raise it)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    os.environ["MODEL_PATH"] = str(args.model)
    os.environ["CONF"] = str(args.conf)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
