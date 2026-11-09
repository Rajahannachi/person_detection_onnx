"""Compare models on the same video: PyTorch (.pt) reference vs ONNX Runtime exports.

Each engine runs in its own fresh process, so the RAM figure is the real cost of that engine.
Every engine sees exactly the same frames; detections are compared with the first model
(the .pt baseline) to check that accuracy did not change.

Examples:
    python benchmark_engines.py
    python benchmark_engines.py --video input_videos/cam.mp4 --frames 300 --threads 4
    python benchmark_engines.py --models models/yolo26n.pt models/yolo26n_384x640.onnx
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MB = 1024 ** 2


# ----------------------------------------------------------------------------- worker (one engine)
def worker(model: str, video: str, frames: int, imgsz: int, conf: float, threads: int | None):
    if threads:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = str(threads)
    import cv2
    import numpy as np
    import psutil

    proc = psutil.Process()
    rss_start = proc.memory_info().rss
    t = time.perf_counter()

    if model.endswith(".pt"):
        import torch
        from ultralytics import YOLO
        if threads:
            torch.set_num_threads(threads)
        y = YOLO(model)

        def run(frame):
            b = y.predict(frame, imgsz=imgsz, conf=conf, device="cpu", verbose=False)[0].boxes
            return int((b.cls == 0).sum()), len(b)
        backend = "pytorch"
    else:
        from engine import Detector
        d = Detector(model, conf=conf, threads=threads)

        def run(frame):
            r = d.predict(frame)[0]
            return r.persons, len(r)
        backend = "onnxruntime"

    warm = np.zeros((imgsz, imgsz, 3), np.uint8)
    for _ in range(3):
        run(warm)
    load_s = time.perf_counter() - t
    rss_loaded = proc.memory_info().rss

    cap = cv2.VideoCapture(video)
    lat, persons, objects = [], [], []
    t0 = time.perf_counter()
    while len(lat) < frames:
        ok, frame = cap.read()
        if not ok:
            break
        ts = time.perf_counter()
        p, o = run(frame)
        lat.append(time.perf_counter() - ts)
        persons.append(p)
        objects.append(o)
    wall = time.perf_counter() - t0
    cap.release()

    mi = proc.memory_info()
    peak = getattr(mi, "peak_wset", None) or mi.rss
    lat_ms = sorted(x * 1000 for x in lat)
    print("RESULT " + json.dumps({
        "model": model, "backend": backend, "frames": len(lat), "load_s": round(load_s, 2),
        "fps_end_to_end": round(len(lat) / wall, 2), "fps_inference": round(len(lat) / sum(lat), 2),
        "lat_avg_ms": round(sum(lat_ms) / len(lat_ms), 1), "lat_p95_ms": round(lat_ms[int(0.95 * (len(lat_ms) - 1))], 1),
        "ram_model_mb": round((rss_loaded - rss_start) / MB, 1), "ram_process_mb": round(rss_loaded / MB, 1),
        "ram_peak_mb": round(peak / MB, 1), "torch_loaded": "torch" in sys.modules,
        "persons": persons, "objects": objects,
    }), flush=True)


# ----------------------------------------------------------------------------- main
def default_models():
    m = ROOT / "models"
    cands = [m / "yolo26n.pt", m / "yolo26n.onnx"]
    cands += sorted(p for p in m.glob("*.onnx") if p not in cands)
    return [str(p) for p in cands if p.exists()]


def default_video():
    for folder in (ROOT / "input_videos", ROOT.parent / "person_detection" / "input_videos"):
        vids = sorted(p for p in folder.glob("*") if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"})
        if vids:
            return str(vids[0])
    return None


def main():
    ap = argparse.ArgumentParser(description="Compare PyTorch / ONNX Runtime models on the same video")
    ap.add_argument("--models", nargs="+", default=None, help="first one is the reference for detections")
    ap.add_argument("--video", default=None)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--threads", type=int, default=None, help="inference threads per engine")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        worker(args.models[0], args.video, args.frames, args.imgsz, args.conf, args.threads)
        return

    models = args.models or default_models()
    video = args.video or default_video()
    if not models:
        print("No models found. Run: python export_model.py")
        return
    if not video:
        print("No video. Put one in input_videos/ or use --video.")
        return
    print(f"Video: {video} | {args.frames} frames | imgsz {args.imgsz} | threads {args.threads or 'auto'}\n")

    results = []
    for m in models:
        print(f"> {m} ...", flush=True)
        cmd = [sys.executable, str(Path(__file__)), "--worker", "--models", m, "--video", video,
               "--frames", str(args.frames), "--imgsz", str(args.imgsz), "--conf", str(args.conf)]
        if args.threads:
            cmd += ["--threads", str(args.threads)]
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        if not line:
            print(f"  FAILED:\n{p.stderr[-2000:]}")
            continue
        results.append(json.loads(line[7:]))

    if not results:
        return
    ref = results[0]
    for r in results:
        n = min(len(r["persons"]), len(ref["persons"]))
        diffs = [abs(a - b) for a, b in zip(r["persons"][:n], ref["persons"][:n])]
        r["persons_total"] = sum(r["persons"])
        r["frames_same_person_count_pct"] = round(100 * sum(d == 0 for d in diffs) / max(1, n), 1)
        r["person_count_mean_abs_diff"] = round(sum(diffs) / max(1, n), 3)
        r["speedup_vs_ref"] = round(r["fps_inference"] / ref["fps_inference"], 2)

    for r in results:
        r["name"] = Path(r["model"]).name
    cols = [("name", "model"), ("backend", "engine"), ("fps_inference", "FPS infer"), ("fps_end_to_end", "FPS e2e"),
            ("lat_avg_ms", "lat ms"), ("lat_p95_ms", "p95 ms"), ("speedup_vs_ref", "speedup"),
            ("ram_process_mb", "RAM MB"), ("ram_peak_mb", "peak MB"), ("torch_loaded", "torch"),
            ("persons_total", "persons"), ("frames_same_person_count_pct", "same count %")]
    widths = [max(len(h), *(len(str(r[k])) for r in results)) for k, h in cols]
    print("\n" + "  ".join(h.ljust(w) for (_, h), w in zip(cols, widths)))
    for r in results:
        print("  ".join(str(r[k]).ljust(w) for (k, _), w in zip(cols, widths)))
    print(f"\nReference for detections: {ref['model']}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    summary = [{k: v for k, v in r.items() if k not in ("persons", "objects")} for r in results]
    (out_dir / f"engines_{stamp}.json").write_text(
        json.dumps({"video": video, "frames": args.frames, "imgsz": args.imgsz, "threads": args.threads,
                    "results": summary}, indent=2), encoding="utf-8")
    with open(out_dir / f"engines_{stamp}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"Reports: reports/engines_{stamp}.json / .csv")


if __name__ == "__main__":
    main()
