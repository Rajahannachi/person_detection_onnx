# Person Detection (yolo26n, ONNX)

yolo26n run by **ONNX Runtime** (no PyTorch at inference time): a stress test for several
videos in parallel, and a FastAPI web interface (image, video, webcam).

## Installation

```bash
pip install -r requirements.txt
```

## 1. Export the model (once)

```bash
python export_model.py                                 # models/yolo26n.onnx (640x640)
python export_model.py --imgsz 384 640 --name 384x640  # rectangular input for 16:9 cameras (faster)
```

## 2. Stress test: several videos in parallel

```bash
python stress_test.py                                  # every video in input_videos/, once
python stress_test.py --streams 8 --duration 60        # 8 parallel streams for 1 minute (videos loop)
python stress_test.py --streams 4 --realtime           # read files at their FPS, like cameras
python stress_test.py --sources a.mp4 rtsp://cam/sub --save --persons-only
```

- one thread per stream, **one model shared** by all of them (ONNX Runtime runs them in parallel)
- `--threads N`: ONNX threads per inference (default: CPU cores / streams)
- output: FPS per stream, latency avg / p95, CPU, RAM peak → `reports/stress_<date>.json`
- `--save`: annotated videos in `output_videos/`

## 3. Web interface (FastAPI)

```bash
python api.py                                          # http://127.0.0.1:8000
python api.py --model models/yolo26n_384x640.onnx --host 0.0.0.0
```

| Tab    | How it works |
|--------|--------------|
| Image  | `POST /api/image` → detections + annotated image |
| Video  | `POST /api/video` (upload) then `GET /api/video/{id}/stream` (annotated live MJPEG); the annotated `.mp4` can be downloaded at the end |
| Webcam | the browser sends frames to `POST /api/image?annotate=false` and draws the boxes itself |

Interactive API docs: `http://127.0.0.1:8000/docs`.
The browser only allows the webcam on `localhost` or over HTTPS.

## 4. Compare with the PyTorch reference

```bash
python benchmark_engines.py --frames 200
```

Each model runs in its own process on the same frames: FPS, latency, RAM, and how often the
person count matches the `.pt` reference. Reports go to `reports/`.
