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
    DEFAULT_ANNOTATION_ROOT,
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
from ultralytics.utils.torch_utils import strip_optimizer, unwrap_model  # noqa: E402


CHECKPOINT_NAMES = {
    view: f"segmentanytooth_yolo11_{view}.pt" for view in MODEL_VIEWS
}


def _resolve_annotation_root(
    value: str,
    data_root: Path | None = None,
) -> Path | None:
    """Return the flat annotation tree, or ``None`` for the legacy layout.

    ``DEFAULT_ANNOTATION_ROOT`` is written relative to the dataset volume, but
    the repository keeps symlinks for it under ``.datasets/``. A relative value
    is therefore searched next to the dataset root first, then in the current
    directory and under ``.datasets/``, so both ``--data-root
    .datasets/intraoral`` (repo-local symlinks, no ``cd``) and ``--data-root
    /disk1/.../intraoral`` work with the default.
    """

    text = value.strip()
    if text.lower() in {"", "none", "-"}:
        return None

    raw = Path(text).expanduser()
    if raw.is_absolute():
        return raw

    candidates: list[Path] = []
    if data_root is not None:
        root = Path(data_root).expanduser()
        candidates.append(root.parent / raw)
        candidates.append(root / raw)
    candidates.extend((raw, Path(".datasets") / raw))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return raw


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
        "--annotation-root",
        type=str,
        default=DEFAULT_ANNOTATION_ROOT,
        help=(
            "Flat ground-truth tree that lives outside the *_process folders, e.g. "
            "intraoral_adjust/tooth_bbox_adj. A relative path is searched in the "
            "current directory, under .datasets/, and next to --data-root, so the "
            "repo-local symlinks work without changing directory. Images are matched "
            "through the annotation's own orig_size. Pass 'none' to fall back to "
            "--annotation-dir instead."
        ),
    )
    parser.add_argument(
        "--annotation-dir",
        type=str,
        default="tooth_bbox_revise",
        help="Annotation directory inside each *_process folder (legacy layout). "
             "Only used when --annotation-root none.",
    )
    parser.add_argument(
        "--strict-fdi",
        action="store_true",
        help=(
            "Fail on an FDI the checkpoint has no class for instead of dropping "
            "that box and reporting it in the label audit."
        ),
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    # Hyperparameters. An intraoral view holds up to 16 teeth, and a tooth's FDI
    # number is decided by its position along the whole arch, so the defaults
    # below replace the Ultralytics COCO ones in two places that matter here:
    # the number of optimizer steps per epoch, and mosaic.
    parser.add_argument(
        "--epochs",
        type=int,
        default=120,
        help="Epochs to train. Read together with --nbs: with nbs == batch this "
             "works out to ~1440 optimizer steps for the 90-image views.",
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument(
        "--nbs",
        type=int,
        default=None,
        help="Nominal batch size Ultralytics uses to derive gradient "
             "accumulation. Defaults to --batch, i.e. one optimizer step per "
             "batch. Ultralytics defaults to nbs=64, which with batch=8 gives "
             "accumulate=8 and therefore only (images / 64) optimizer steps per "
             "epoch -- just 1.4 for the 90-image views.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early-stop after this many epochs without validation improvement. "
             "It must be smaller than --epochs to ever trigger (the Ultralytics "
             "default of 100 never fires at epochs=100).",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=10,
        help="Validate every N epochs, plus always the final epoch. Ultralytics "
             "validates every single epoch, which on these views costs roughly a "
             "quarter of a training epoch (22 val images against 90 train "
             "images) even though only best.pt selection and early stopping "
             "read the result. Pass 1 to restore the Ultralytics behaviour.",
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=-1,
        help="Also snapshot every Nth epoch as weights/epochN.pt. The Ultralytics "
             "default of -1 keeps only last.pt and best.pt.",
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=None,
        help="Freeze the first N modules, i.e. model.0 .. model.{N-1}, BatchNorm "
             "running stats included, so those weights stay bit-for-bit "
             "untouched. yolo11's backbone is exactly modules 0-10, so "
             "--freeze 11 trains only the neck and the Detect head (47.3%% of "
             "the parameters). The DFL projection inside Detect is always "
             "frozen. Does not change the learning rate: 'optimizer=auto' "
             "derives it from the class count alone. Default None trains all.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=float,
        default=None,
        help="Ultralytics warmup length in epochs. Careful: the trainer floors "
             "it at max(round(warmup_epochs * batches_per_epoch), 100), so any "
             "value in (0, 8.3] gives exactly 100 iterations, which is ~8.3 "
             "epochs at the 12 batches/epoch of the upper/lower/front views. "
             "Only 0.0 actually disables warmup. Default None keeps the "
             "Ultralytics default of 3.0.",
    )
    parser.add_argument(
        "--mosaic",
        type=float,
        default=0.0,
        help="Probability of 4/9-image mosaic. Tiling images both destroys the "
             "global arch layout that decides an FDI number and halves the "
             "resolution of already-dense occlusal views, so it is off by "
             "default here. Raise it only with a replacement for that context.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.3,
        help="Random scale gain (+-). The cameras are fixed-distance and "
             "auto-flip is disabled, so there is less room for resize jitter "
             "than the 0.5 Ultralytics default.",
    )
    parser.add_argument(
        "--cos-lr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cosine LR schedule. Pass --no-cos-lr for the linear default.",
    )
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
        "--swanlab-workspace",
        type=str,
        default="zykev",
        help="SwanLab workspace (user or team) that owns the project.",
    )
    parser.add_argument(
        "--swanlab-project",
        type=str,
        default="dentmap",
        help="SwanLab project name.",
    )
    parser.add_argument(
        "--swanlab-name",
        type=str,
        default="yolo_finetune_v1",
        help="SwanLab run name. View is appended when --view all.",
    )
    args = parser.parse_args(argv)
    if args.nbs is None:
        # Resolve here so the effective value reaches both swanlab config and
        # train_args as a plain int.
        args.nbs = args.batch
    return args


def build_tooth_trainer(
    train_samples: Sequence[ToothSample],
    val_samples: Sequence[ToothSample],
    class_names: Mapping[int, str] | Sequence[str],
    swanlab_run,
    unknown_fdi_policy: str = "skip",
    val_interval: int = 1,
):
    """Bind the in-memory dataset and optional SwanLab logger to DetectionTrainer."""

    normalized_names = normalize_class_names(class_names)
    splits = {
        "train": list(train_samples),
        "val": list(val_samples),
    }

    def log_epoch(trainer) -> None:
        """Log train losses, validation metrics, LR, and best fitness."""

        metrics = {
            "epoch": trainer.epoch + 1,
            **trainer.label_loss_items(trainer.tloss, prefix="train"),
            **(trainer.metrics or {}),
            **trainer.lr,
            "best_fitness": float(trainer.best_fitness or 0.0),
        }
        swanlab_run.log(metrics)

    def log_training_summary(trainer) -> None:
        """Log the final metrics a second time under a ``final/`` prefix.

        SwanLab has no run-summary object like ``wandb_run.summary``, so the
        values that used to go there are logged as ordinary scalars at the step
        of the last epoch, which keeps them aligned with the per-epoch curves.
        """

        final = {
            f"final/{key}": float(value)
            for key, value in (trainer.metrics or {}).items()
        }
        if trainer.best_fitness is not None:
            final["final/best_fitness"] = float(trainer.best_fitness)
        if final:
            swanlab_run.log(final, step=trainer.epoch + 1)

    class ToothDetectionTrainer(DetectionTrainer):
        """Ultralytics trainer using source images and revised JSON directly."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # A manually initialized run is used so workspace/project/name are
            # explicit. Ultralytics ships no SwanLab callback but does register a
            # W&B one, which the tracker no longer uses here; dropping it keeps
            # Ultralytics from creating a stray W&B run behind our back.
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
                unknown_fdi=unknown_fdi_policy,
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

        def validate(self):
            """Measure only on the configured interval instead of every epoch.

            ``BaseTrainer`` validates whenever ``args.val`` is set (its default),
            and on these views a validation pass is roughly a quarter of a
            training epoch: 22 held-out images against 90 training images. Only
            ``best.pt`` selection and early stopping consume the result here --
            the project's real per-FDI numbers come from ``estimate_accuracy.py``
            after the export -- so a sparse schedule is enough. Skipped epochs
            keep the previous values, which still yields one ``results.csv`` row
            and one SwanLab point per epoch, and ``EarlyStopping`` simply sees no
            improvement.

            The sentinel fitness of ``-1.0`` on skipped epochs is deliberate:
            ``BaseTrainer.save_model`` writes ``best.pt`` whenever
            ``best_fitness == fitness``. Returning the cached (hence equal)
            fitness would overwrite the best checkpoint with unvalidated weights,
            while ``-1.0`` can never equal a real mAP-based fitness.
            """

            epoch = self.epoch + 1
            # ``possible_stop`` turns true one epoch before the patience window
            # closes, and the epoch loop lets the validator through on those
            # epochs; honoring it here means the final decision is always made on
            # fresh weights instead of a remembered mAP.
            if (
                val_interval > 1
                and epoch % val_interval
                and epoch < self.epochs
                and not self.stopper.possible_stop
            ):
                return self.metrics, -1.0
            return super().validate()

        def final_eval(self):
            """Finalize the run without Ultralytics' data re-lookup.

            ``BaseTrainer.final_eval`` re-validates ``best.pt`` through
            ``self.validator(model=...)``, whose non-training branch resolves
            ``args.data`` with ``check_det_dataset``.  Here ``args.data`` is the
            ``"in_memory_tooth_dataset"`` sentinel, so the stock implementation
            raises ``FileNotFoundError: Dataset 'in_memory_tooth_dataset' for
            task=detect not found`` at the very end of every run -- after all
            epochs are done, which also skips the checkpoint stripping and the
            ``on_train_end`` callback.  ``validate`` already reports the detection
            metrics on its own schedule and ``estimate_accuracy.py`` gives the
            project's per-FDI numbers for the exported weights, so only the
            stripping is reproduced here.
            """

            model = self.best if self.best.exists() else None
            ckpt = strip_optimizer(self.last) if self.last.exists() else {}
            if model:
                # Carry the training metrics collected in last.pt over to best.pt.
                strip_optimizer(self.best, updates={"train_results": ckpt.get("train_results")})

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

    annotation_root = _resolve_annotation_root(args.annotation_root, args.data_root)
    print(
        "Annotation source: "
        + (
            str(annotation_root)
            if annotation_root is not None
            else f"<*_process>/{args.annotation_dir}/<subject>/<view>.json"
        )
    )
    # Cross-view teeth are kept on purpose; only FDI the checkpoint cannot
    # express is dropped (and reported) unless --strict-fdi is given.
    unknown_policy = "raise" if args.strict_fdi else "skip"

    samples = discover_samples(
        data_root=args.data_root,
        model_view=view,
        annotation_dir=args.annotation_dir,
        annotation_root=annotation_root,
    )
    if not samples:
        source = (
            str(annotation_root)
            if annotation_root is not None
            else f"<*_process>/{args.annotation_dir}"
        )
        raise RuntimeError(
            f"No {view} samples found below {args.data_root} (annotations: {source})."
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

    train_summary = summarize_samples(
        train_samples, detector.class_names, unknown=unknown_policy
    )
    val_summary = summarize_samples(
        val_samples, detector.class_names, unknown=unknown_policy
    )
    print(f"Train label audit: {json.dumps(train_summary, ensure_ascii=False)}")
    print(f"Val label audit: {json.dumps(val_summary, ensure_ascii=False)}")
    print(
        f"Hyperparameters: epochs={args.epochs} batch={args.batch} nbs={args.nbs} "
        f"imgsz={args.imgsz} patience={args.patience} "
        f"val_interval={args.val_interval} save_period={args.save_period} "
        f"mosaic={args.mosaic} scale={args.scale} cos_lr={args.cos_lr} "
        f"freeze={args.freeze} warmup_epochs={args.warmup_epochs}"
    )
    if train_summary["boxes"] == 0 or val_summary["boxes"] == 0:
        raise RuntimeError("Train and validation splits must both contain boxes")
    dropped = {
        "train": train_summary["dropped_unknown_fdi"],
        "val": val_summary["dropped_unknown_fdi"],
    }
    if any(dropped.values()):
        print(
            "WARNING dropped boxes whose FDI the checkpoint cannot express: "
            f"{json.dumps(dropped, ensure_ascii=False)}"
        )

    if args.dry_run:
        print("Dry run complete; no files were generated and optimization was not started.")
        return None

    import swanlab

    # One folder per view holds everything about that experiment:
    #   exp/yolo11_<view>_finetune/
    #       args.yaml  results.csv  weights/
    #       swanlab/run-<timestamp>-<hash>/     <- this run's local records
    # Without the explicit log_dir SwanLab drops its run-* folders straight
    # into exp/ and nothing tells them apart.
    project.mkdir(parents=True, exist_ok=True)
    run_dir = project / run_name
    swanlab_dir = run_dir / "swanlab"
    swanlab_dir.mkdir(parents=True, exist_ok=True)
    swanlab_name = (
        f"{args.swanlab_name}_{view}" if args.view == "all" else args.swanlab_name
    )
    swanlab_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    swanlab_config.update(
        {
            "model_view": view,
            "checkpoint": str(checkpoint),
            "annotation_root": str(annotation_root),
            "unknown_fdi_policy": unknown_policy,
            "train_images": train_summary["images"],
            "train_boxes": train_summary["boxes"],
            "train_cross_view_boxes": train_summary["cross_view_boxes"],
            "val_images": val_summary["images"],
            "val_boxes": val_summary["boxes"],
            "val_cross_view_boxes": val_summary["cross_view_boxes"],
            "class_names": detector.class_names,
        }
    )
    swanlab_run = swanlab.init(
        workspace=args.swanlab_workspace,
        project=args.swanlab_project,
        name=swanlab_name,
        config=swanlab_config,
        log_dir=str(swanlab_dir),
        # --view all starts one run per view in the same process; reinit makes
        # sure a leftover run from the previous view can never block this one.
        reinit=True,
    )

    trainer_class = build_tooth_trainer(
        train_samples=train_samples,
        val_samples=val_samples,
        class_names=detector.class_names,
        swanlab_run=swanlab_run,
        unknown_fdi_policy=unknown_policy,
        val_interval=args.val_interval,
    )
    train_args: dict[str, Any] = {
        "data": "in_memory_tooth_dataset",
        "trainer": trainer_class,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        # nbs == batch keeps gradient accumulation at 1, so every batch is an
        # optimizer step instead of one step per 64 images.
        "nbs": args.nbs,
        "patience": args.patience,
        "cos_lr": args.cos_lr,
        # -1 keeps only last.pt / best.pt; >0 also snapshots weights/epochN.pt.
        "save_period": args.save_period,
        "device": args.device,
        "cache": False,
        "project": str(project),
        "name": run_name,
        # swanlab_dir above is created before this call, so Ultralytics'
        # increment_path would see exp/yolo11_<view>_finetune/ already there and
        # silently move the run to ..._finetune2, splitting the weights away
        # from the run records. exist_ok pins the trainer to that exact folder,
        # which also means re-running a view refreshes it in place instead of
        # piling up ..._finetune2, ..._finetune3.
        "exist_ok": True,
        "seed": args.seed,
        "plots": False,
        # Horizontal/vertical flips corrupt side-specific FDI class semantics.
        "fliplr": 0.0,
        "flipud": 0.0,
        # Mosaic would hash the arch layout the FDI label is derived from.
        "mosaic": args.mosaic,
        "scale": args.scale,
    }
    if args.device is None:
        train_args.pop("device")
    # Both are valid DEFAULT_CFG keys, so unlike 'val_interval' (which only
    # reaches the trainer as a closure variable) they can travel through
    # Model.train. Omitted entirely when unset, so the default run is unchanged.
    if args.freeze is not None:
        train_args["freeze"] = args.freeze
    if args.warmup_epochs is not None:
        train_args["warmup_epochs"] = args.warmup_epochs

    try:
        detector.ultralytics_model.train(**train_args)
    finally:
        swanlab_run.finish()

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
