"""Stress test: process several videos in parallel with ONE shared ONNX model.

One thread per stream: it decodes its video and runs the detector. ONNX Runtime releases the
GIL during inference, so the streams really run in parallel on the CPU cores.

Examples:
    python stress_test.py                                  # every video in input_videos/, once
    python stress_test.py --streams 8 --duration 60        # 8 parallel streams for 1 minute
    python stress_test.py --streams 4 --realtime           # pace files at their FPS, like cameras
    python stress_test.py --sources a.mp4 rtsp://cam/sub --save --persons-only
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import psutil

from drawing import ROOT, annotate, list_videos
from engine import PERSON_CLASS_ID, Detector

MB = 1024 ** 2


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[int(p * (len(s) - 1))]


class StreamWorker(threading.Thread):
    """Reads one source and runs detection on every frame (or every frame allowed by `max_fps`)."""

    def __init__(self, sid: int, src: str, det: Detector, stop: threading.Event, *,
                 loop: bool, realtime: bool, out_path: Path | None = None):
        super().__init__(daemon=True, name=f"stream-{sid}")
        self.sid, self.src, self.det, self.stop = sid, src, det, stop
        self.loop, self.realtime, self.out_path = loop, realtime, out_path
        self.latencies_ms: list[float] = []
        self.persons = 0
        self.fps_src = 0.0
        self.error: str | None = None

    def run(self) -> None:
        cap = cv2.VideoCapture(self.src)
        if not cap.isOpened():
            self.error = f"cannot open {self.src}"
            return
        self.fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
        period = 1.0 / self.fps_src
        writer = None
        is_file = "://" not in self.src
        try:
            next_t = time.perf_counter()
            while not self.stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    if is_file and self.loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                t = time.perf_counter()
                res = self.det.predict(frame)[0]
                self.latencies_ms.append((time.perf_counter() - t) * 1000)
                self.persons = res.persons

                if self.out_path:
                    annotate(frame, res, self.det.names, [f"Stream {self.sid}", f"Persons: {res.persons}"])
                    if writer is None:
                        h, w = frame.shape[:2]
                        writer = cv2.VideoWriter(str(self.out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 self.fps_src, (w, h))
                    writer.write(frame)

                if self.realtime and is_file:      # behave like a camera: one frame per period
                    next_t += period
                    delay = next_t - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        next_t = time.perf_counter()
        except Exception as e:                     # keep the other streams running
            self.error = repr(e)
        finally:
            cap.release()
            if writer:
                writer.release()


def main() -> None:
    ap = argparse.ArgumentParser(description="Parallel multi-video stress test (YOLO ONNX)")
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "yolo26n.onnx")
    ap.add_argument("--sources", nargs="*", help="video files or rtsp:// urls (default: input_videos/)")
    ap.add_argument("--streams", type=int, help="number of parallel streams (sources are repeated)")
    ap.add_argument("--duration", type=float, help="stop after N seconds (videos loop); default: until they end")
    ap.add_argument("--realtime", action="store_true", help="read files at their FPS instead of max speed")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--persons-only", action="store_true")
    ap.add_argument("--threads", type=int, help="ONNX threads per inference (default: cores / streams)")
    ap.add_argument("--save", action="store_true", help="save annotated videos to output_videos/")
    args = ap.parse_args()

    sources = args.sources or [str(p) for p in list_videos(ROOT / "input_videos")]
    if not sources:
        raise SystemExit("No sources: put videos in input_videos/ or use --sources.")
    n = args.streams or len(sources)
    sources = [sources[i % len(sources)] for i in range(n)]
    threads = args.threads or max(1, (os.cpu_count() or 1) // n)

    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    t = time.perf_counter()
    det = Detector(args.model, conf=args.conf, threads=threads,
                   classes=[PERSON_CLASS_ID] if args.persons_only else None)
    det.warmup()
    print(f"Model {args.model.name} ({det.provider}) loaded in {time.perf_counter() - t:.1f}s, "
          f"+{(proc.memory_info().rss - rss0) / MB:.0f} MB | {n} stream(s) x {threads} thread(s)")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "output_videos"
    stop = threading.Event()
    workers = [StreamWorker(i, s, det, stop, loop=args.duration is not None, realtime=args.realtime,
                            out_path=out_dir / f"{stamp}_s{i}_{Path(s).stem}.mp4" if args.save else None)
               for i, s in enumerate(sources)]

    cpu_samples: list[float] = []
    rss_peak = proc.memory_info().rss
    proc.cpu_percent(None)
    t0 = last_print = time.perf_counter()
    for w in workers:
        w.start()

    try:
        while any(w.is_alive() for w in workers):
            time.sleep(1)
            now = time.perf_counter()
            el = now - t0
            cpu_samples.append(proc.cpu_percent(None) / psutil.cpu_count())
            rss_peak = max(rss_peak, proc.memory_info().rss)
            if now - last_print >= 5:
                last_print = now
                fps = " ".join(f"{len(w.latencies_ms) / el:.1f}" for w in workers)
                print(f"[{el:5.0f}s] FPS/stream: {fps} | CPU {cpu_samples[-1]:.0f}% | "
                      f"RAM {proc.memory_info().rss / MB:.0f} MB | persons {[w.persons for w in workers]}",
                      flush=True)
            if args.duration and el >= args.duration:
                break
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        stop.set()
        for w in workers:
            w.join(timeout=5)

    el = time.perf_counter() - t0
    all_lat = [x for w in workers for x in w.latencies_ms]
    per_stream = [{
        "source": w.src, "source_fps": round(w.fps_src, 1), "frames": len(w.latencies_ms),
        "fps": round(len(w.latencies_ms) / el, 2),
        "latency_ms_avg": round(sum(w.latencies_ms) / max(1, len(w.latencies_ms)), 1),
        "latency_ms_p95": round(percentile(w.latencies_ms, 0.95), 1),
        "error": w.error,
    } for w in workers]
    report = {
        "date": stamp, "model": str(args.model), "provider": det.provider, "streams": n,
        "threads_per_inference": threads, "realtime": args.realtime, "duration_s": round(el, 1),
        "total_fps": round(len(all_lat) / el, 2),
        "min_stream_fps": min(s["fps"] for s in per_stream),
        "latency_ms_avg": round(sum(all_lat) / max(1, len(all_lat)), 1),
        "latency_ms_p95": round(percentile(all_lat, 0.95), 1),
        "cpu_avg_pct": round(sum(cpu_samples) / max(1, len(cpu_samples)), 1),
        "ram_peak_mb": round(rss_peak / MB, 1),
        "per_stream": per_stream,
    }

    print("\n=== Summary")
    for s in per_stream:
        status = f"ERROR {s['error']}" if s["error"] else f"{s['fps']} FPS, lat {s['latency_ms_avg']} ms"
        print(f"  {Path(s['source']).name:<45} {status}")
    print(f"total {report['total_fps']} FPS over {n} stream(s) | min stream {report['min_stream_fps']} FPS | "
          f"latency avg {report['latency_ms_avg']} ms, p95 {report['latency_ms_p95']} ms | "
          f"CPU {report['cpu_avg_pct']}% | RAM peak {report['ram_peak_mb']} MB")
    out = ROOT / "reports" / f"stress_{stamp}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
