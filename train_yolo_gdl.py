"""
YOLO-GDL coal/gangue detection training script.

Model:
    ultralytics/cfg/models/v8/yolov8n-ghostneck-dwdown.yaml
Dataset:
    煤块和石块/data.yaml
Output:
    runs/train/YOLO-GDL
"""

from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main():
    train_args = {
        "data": str(ROOT / "煤块和石块" / "data.yaml"),
        "epochs": 100,
        "imgsz": 1280,
        "batch": 8,
        "workers": 4,
        "device": 0,
        "amp": True,
        "patience": 25,
        "project": str(ROOT / "runs" / "train"),
        "name": "YOLO-GDL",
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "auto",
        "seed": 0,
        "deterministic": True,
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "close_mosaic": 10,
        "hsv_h": 0.01,
        "hsv_s": 0.35,
        "hsv_v": 0.65,
        "translate": 0.1,
        "scale": 0.5,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "plots": True,
        "save": True,
    }

    model_cfg = ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8n-ghostneck-dwdown.yaml"
    model = YOLO(str(model_cfg))
    return model.train(**train_args)


if __name__ == "__main__":
    main()
