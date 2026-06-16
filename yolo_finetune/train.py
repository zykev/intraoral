"""Fine-tune angle-specific SegmentAnyTooth YOLO11 detectors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolo_finetune.dataset import (  # noqa: E402
    InMemoryToothYOLODataset,
    MODEL_VIEWS,
    ToothSample,
    discover_samples,
    normalize_class_names,
    split_samples,
    summarize_samples,
)
from yolo_finetune.model import YOLO11ToothDetector  # noqa: E402

from ultralytics.models.yolo.detect.train import DetectionTrainer  # noqa: E402
from ultralytics.utils import colorstr  # noqa: E402
from ultralytics.utils.torch_utils import unwrap_model  # noqa: E402


CHECKPOINT_NAMES = {
    view: f"segmentanytooth_yolo11_{view}.pt" for view in MODEL_VIEWS
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune the front/upper/lower/right YOLO11 tooth detectors "
            "without copying source images."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(".datasets/intraoral"),
        help="Intraoral root containing any number of people and *_process folders.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(".checkpoints/segtooth_model"),
        help="Directory containing the original SegmentAnyTooth YOLO checkpoints.",
    )
    parser.add_argument(
        "--view",
        choices=(*MODEL_VIEWS, "all"),
        default="all",
        help="Detector to train. There is no separate left checkpoint.",
    )
    parser.add_argument(
        "--annotation-dir",
        type=str,
        default="tooth_bbox_revise",
        help="Annotation directory inside each *_process folder.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument(
        "--project",
        type=Path,
        default=Path("exp"),
        help="Training runs and compatible checkpoints are stored here.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and audit the dataset without starting optimization.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="intraoral",
        help="Weights & Biases entity (user or team).",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="yolo_finetune",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default="yolo_finetune_v1",
        help="Weights & Biases run name. View is appended when --view all.",
    )
    return parser.parse_args(argv)


def build_tooth_trainer(
    train_samples: Sequence[ToothSample],
    val_samples: Sequence[ToothSample],
    class_names: Mapping[int, str] | Sequence[str],
    wandb_run,
):
    """Bind the in-memory dataset and optional W&B logger to DetectionTrainer."""

    normalized_names = normalize_class_names(class_names)
    splits = {
        "train": list(train_samples),
        "val": list(val_samples),
    }

    def log_epoch(trainer) -> None:
        """Log train losses, validation losses, detection metrics, and LR."""

        metrics = {
            "epoch": trainer.epoch + 1,
            **trainer.label_loss_items(trainer.tloss, prefix="train"),
            **(trainer.metrics or {}),
            **trainer.lr,
        }
        wandb_run.log(metrics)

    def log_training_summary(trainer) -> None:
        """Store final detection metrics and the best fitness in run summary."""

        for key, value in (trainer.metrics or {}).items():
            wandb_run.summary[f"final/{key}"] = float(value)
        wandb_run.summary["best_fitness"] = float(trainer.best_fitness)

    class ToothDetectionTrainer(DetectionTrainer):
        """Ultralytics trainer using source images and revised JSON directly."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # A manually initialized run is used so entity/project/name are explicit.
            for event, callbacks in self.callbacks.items():
                self.callbacks[event] = [
                    callback
                    for callback in callbacks
                    if callback.__module__ != "ultralytics.utils.callbacks.wb"
                ]
            self.add_callback("on_fit_epoch_end", log_epoch)
            self.add_callback("on_train_end", log_training_summary)

        def get_dataset(self):
            """Provide class metadata without creating a dataset YAML file."""

            return {
                "train": "in_memory_train",
                "val": "in_memory_val",
                "nc": len(normalized_names),
                "names": normalized_names,
                "channels": 3,
            }

        def build_dataset(self, img_path, mode="train", batch=None):
            """Create the requested train or validation dataset."""

            del img_path
            stride = max(int(unwrap_model(self.model).stride.max()), 32)
            return InMemoryToothYOLODataset(
                samples=splits[mode],
                class_names=normalized_names,
                img_path=mode,
                imgsz=self.args.imgsz,
                batch_size=batch,
                augment=mode == "train",
                hyp=self.args,
                rect=self.args.rect or mode == "val",
                cache=False,
                single_cls=self.args.single_cls or False,
                stride=stride,
                pad=0.0 if mode == "train" else 0.5,
                prefix=colorstr(f"{mode}: "),
                task="detect",
                classes=self.args.classes,
                data=self.data,
                fraction=self.args.fraction if mode == "train" else 1.0,
            )

    return ToothDetectionTrainer


def _train_one_view(args: argparse.Namespace, view: str) -> Path | None:
    """Validate one view, fine-tune its checkpoint, and export ``best.pt``."""

    checkpoint = (args.checkpoint_dir / CHECKPOINT_NAMES[view]).expanduser().resolve()
    run_name = f"yolo11_{view}_finetune"
    project = args.project.expanduser().resolve()

    print(f"\n=== {view.upper()} detector ===")
    print(f"Loading pretrained checkpoint: {checkpoint}")
    detector = YOLO11ToothDetector(checkpoint)
    print(f"Checkpoint classes: {json.dumps(detector.class_names, ensure_ascii=False)}")

    samples = discover_samples(
        data_root=args.data_root,
        model_view=view,
        annotation_dir=args.annotation_dir,
    )
    if not samples:
        raise RuntimeError(
            f"No {view} samples found below {args.data_root}. "
            f"Checked annotation directory: {args.annotation_dir}"
        )
    train_samples, val_samples = split_samples(
        samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(
        f"Discovered {len(samples)} images ({len(train_samples)} train, "
        f"{len(val_samples)} val)"
    )

    train_summary = summarize_samples(train_samples, detector.class_names)
    val_summary = summarize_samples(val_samples, detector.class_names)
    print(f"Train label audit: {json.dumps(train_summary, ensure_ascii=False)}")
    print(f"Val label audit: {json.dumps(val_summary, ensure_ascii=False)}")
    if train_summary["boxes"] == 0 or val_summary["boxes"] == 0:
        raise RuntimeError("Train and validation splits must both contain boxes")

    if args.dry_run:
        print("Dry run complete; no files were generated and optimization was not started.")
        return None

    import wandb

    project.mkdir(parents=True, exist_ok=True)
    wandb_name = (
        f"{args.wandb_name}_{view}" if args.view == "all" else args.wandb_name
    )
    wandb_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    wandb_config.update(
        {
            "model_view": view,
            "checkpoint": str(checkpoint),
            "train_images": train_summary["images"],
            "train_boxes": train_summary["boxes"],
            "val_images": val_summary["images"],
            "val_boxes": val_summary["boxes"],
            "class_names": detector.class_names,
        }
    )
    wandb_run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=wandb_name,
        config=wandb_config,
        dir=str(project),
    )
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

    trainer_class = build_tooth_trainer(
        train_samples=train_samples,
        val_samples=val_samples,
        class_names=detector.class_names,
        wandb_run=wandb_run,
    )
    train_args: dict[str, Any] = {
        "data": "in_memory_tooth_dataset",
        "trainer": trainer_class,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "cache": False,
        "project": str(project),
        "name": run_name,
        "seed": args.seed,
        "plots": False,
        # Horizontal/vertical flips corrupt side-specific FDI class semantics.
        "fliplr": 0.0,
        "flipud": 0.0,
    }
    if args.device is None:
        train_args.pop("device")

    try:
        detector.ultralytics_model.train(**train_args)
    finally:
        wandb_run.finish()

    trainer = detector.ultralytics_model.trainer
    compatible_name = CHECKPOINT_NAMES[view]
    exported = detector.export_segtooth_checkpoint(
        project / "segtooth_model" / compatible_name
    )
    print(f"Best Ultralytics checkpoint: {Path(trainer.best).resolve()}")
    print(f"SegmentAnyTooth-compatible checkpoint: {exported}")
    return exported


def main(argv: Sequence[str] | None = None) -> None:
    """Train one detector or all four angle-specific detectors."""

    args = parse_args(argv)
    views = MODEL_VIEWS if args.view == "all" else (args.view,)
    exported: list[Path] = []
    for view in views:
        result = _train_one_view(args, view)
        if result is not None:
            exported.append(result)

    if exported:
        print("\nFinished checkpoints:")
        for checkpoint in exported:
            print(f"  {checkpoint}")


if __name__ == "__main__":
    main()
