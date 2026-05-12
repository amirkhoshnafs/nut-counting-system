"""Train YOLO-NAS on the combined nut dataset (single-class nut detection).

num_classes=2 is used to avoid a PPYoloE loss edge case with num_classes=1.
Class 0 = nut, Class 1 = unused placeholder.

Usage:
    conda run -n ml2 python scripts/train.py
    conda run -n ml2 python scripts/train.py --epochs 150 --batch-size 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import super_gradients
from super_gradients.training import Trainer, models
from super_gradients.training.datasets.detection_datasets import COCODetectionDataset
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050, DetectionMetrics_050_095
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training import dataloaders


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data/processed/combined"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def build_dataloader(split: str, batch_size: int, workers: int):
    """Create a DataLoader using SuperGradients' factory (handles target format correctly)."""
    json_path = DATA_DIR / "annotations" / f"{split}.json"
    subdir = f"images/{split}"

    if not json_path.exists():
        raise FileNotFoundError(f"COCO JSON not found: {json_path}")

    is_train = (split == "train")

    dataset_params = {
        "data_dir": str(DATA_DIR),
        "json_file": f"{split}.json",
        "subdir": subdir,
        "with_crowd": not is_train,   # train=False, val=True (matches SG defaults)
        "class_inclusion_list": ["nut"],
    }
    dataloader_params = {
        "batch_size": batch_size,
        "num_workers": workers,
        "shuffle": is_train,
        "drop_last": is_train,
        "pin_memory": True,
    }

    return dataloaders.get_data_loader(
        config_name="coco_detection_yolo_nas_dataset_params",
        dataset_cls=COCODetectionDataset,
        train=is_train,
        dataset_params=dataset_params,
        dataloader_params=dataloader_params,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo_nas_s", choices=["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print("=" * 60)
    print("Training YOLO-NAS on Nut Counting Dataset")
    print(f"  Model:      {args.model}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print(f"  Device:     {args.device}")
    print("=" * 60)

    # ── DataLoaders ────────────────────────────────
    print("\n[1/4] Loading datasets ...")
    train_loader = build_dataloader("train", args.batch_size, args.workers)
    val_loader = build_dataloader("val", args.batch_size, args.workers)
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")

    # ── Model ───────────────────────────────────────
    print("\n[2/4] Creating model ...")
    model = models.get(
        args.model,
        pretrained_weights="coco",
        num_classes=2,  # 2 avoids edge case in PPYoloE loss; class 1 is unused
    )
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # ── Training params ─────────────────────────────
    print("\n[3/4] Configuring trainer ...")
    train_params = {
        "max_epochs": args.epochs,
        "average_best_models": False,
        "lr_mode": "cosine",
        "initial_lr": args.lr,
        "warmup_epochs": 3,
        "warmup_initial_lr": 1e-6,
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "mixed_precision": True,
        "loss": PPYoloELoss(
            use_static_assigner=False,
            num_classes=2,
            reg_max=16,
        ),
        "sg_logger": "base_sg_logger",
        "metric_to_watch": "mAP@0.50",
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300,
                num_cls=2,
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.7,
                ),
            ),
            DetectionMetrics_050_095(
                score_thres=0.1,
                top_k_predictions=300,
                num_cls=2,
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.7,
                ),
            ),
        ],
        "ema": True,
        "ema_decay": 0.9999,
        "save_ckpt_epoch_list": [],
        "launch_tensorboard": False,
    }

    super_gradients.setup_device(device=args.device)
    trainer = Trainer(
        experiment_name="nut_counting_phase1",
        ckpt_root_dir=str(CHECKPOINT_DIR),
    )

    # ── Train ───────────────────────────────────────
    print("\n[4/4] Training ...")
    trainer.train(
        model=model,
        training_params=train_params,
        train_loader=train_loader,
        valid_loader=val_loader,
    )

    print("\nTraining complete. Best checkpoint saved to:", CHECKPOINT_DIR.resolve())


if __name__ == "__main__":
    main()
