"""
煤矿履带杂质智能识别 - YOLO26n 训练脚本
数据集: 煤块和石块 (120张, 96/24 train/val)
模型: yolo26n.pt (预训练权重).
"""

import multiprocessing
import os

from ultralytics import YOLO


def main():
    ROOT = os.path.dirname(os.path.abspath(__file__))

    # ==================== 训练配置 ====================
    TRAIN_CONFIG = {
        # --- 模型 ---
        "model": os.path.join(ROOT, "yolov8n.yaml"),
        # --- 数据集 ---
        "data": os.path.join(ROOT, "煤块和石块", "data.yaml"),
        # --- 训练超参 ---
        "epochs": 100,
        "imgsz": 1280,
        "batch": 8,
        # --- Windows 特化 ---
        "workers": 4,
        "device": 0,
        "amp": True,
        # --- 保存与日志 ---
        "project": os.path.join(ROOT, "runs", "train"),
        "name": "coal-gangue-yolov8n",
        "exist_ok": True,
        "save": True,
        "patience": 25,
    }

    # ==================== 开始训练 ====================
    print("=" * 60)
    print("  煤矿履带杂质识别 - YOLOv8n 训练")
    print(f"  平台: Windows | 设备: GPU:{TRAIN_CONFIG['device']}")
    print("  数据集: 煤块和石块 (96 train / 24 val)")
    print(f"  模型: yolov8n.yaml | epochs={TRAIN_CONFIG['epochs']}")
    print("=" * 60)

    model = YOLO(TRAIN_CONFIG["model"])

    # 提取 model 和 data，其余参数传给 train()
    train_args = {k: v for k, v in TRAIN_CONFIG.items() if k not in ("model", "data")}
    results = model.train(data=TRAIN_CONFIG["data"], **train_args)

    print(f"\n✅ 训练完成！最佳模型保存在 {TRAIN_CONFIG['project']}\\{TRAIN_CONFIG['name']}\\weights\\best.pt")
    return results


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
