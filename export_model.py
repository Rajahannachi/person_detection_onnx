"""Export yolo26n.pt to ONNX (run once; needs ultralytics + torch).

Examples:
    python export_model.py                                  # models/yolo26n.onnx, FP32, 640
    python export_model.py --imgsz 384 640 --name 384x640   # rectangular input for 16:9 cameras
    python export_model.py --dynamic                        # dynamic batch (true batched calls)

Files are written next to the weights in models/.
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description="Export yolo26n to ONNX")
    ap.add_argument("--weights", type=Path, default=ROOT / "models" / "yolo26n.pt")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640],
                    help="640 (square) or H W, e.g. 384 640 for 16:9 cameras (fewer pixels = faster)")
    ap.add_argument("--name", help="suffix added to the exported file name, e.g. 384x640")
    ap.add_argument("--dynamic", action="store_true", help="dynamic input batch")
    args = ap.parse_args()

    import shutil

    from ultralytics import YOLO

    imgsz = args.imgsz[0] if len(args.imgsz) == 1 else args.imgsz[:2]
    weights = args.weights
    if args.name:   # export from a renamed copy so the file does not overwrite the default export
        weights = args.weights.with_name(f"{args.weights.stem}_{args.name}{args.weights.suffix}")
        shutil.copy(args.weights, weights)
    try:
        out = YOLO(str(weights)).export(format="onnx", imgsz=imgsz, simplify=True, dynamic=args.dynamic)
        print(f"-> {out}")
    finally:
        if args.name:
            weights.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
