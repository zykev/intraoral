"""Explicit YOLO11 model wrapper used by the fine-tuning entry point."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


class YOLO11ToothDetector:
    """Load a YOLO11 checkpoint and expose its actual detection network.

    The network remains owned by Ultralytics so trained ``.pt`` files stay
    directly compatible with ``YOLO(checkpoint_path)`` in ``segtooth_new.py``.
    """

    def __init__(self, checkpoint: str | Path) -> None:
        from ultralytics import YOLO

        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"YOLO checkpoint does not exist: {self.checkpoint}")

        self.ultralytics_model = YOLO(str(self.checkpoint), task="detect")
        self.network = self.ultralytics_model.model

    @property
    def class_names(self) -> dict[int, str]:
        names = self.ultralytics_model.names
        if isinstance(names, dict):
            return {int(index): str(name) for index, name in names.items()}
        return {index: str(name) for index, name in enumerate(names)}

    def architecture_text(self) -> str:
        """Return every backbone, neck, and detection-head layer."""

        lines = [
            f"checkpoint: {self.checkpoint}",
            f"classes: {self.class_names}",
            "network:",
        ]
        for index, layer in enumerate(self.network.model):
            parameter_count = sum(parameter.numel() for parameter in layer.parameters())
            lines.append(
                f"  [{index:02d}] {layer.__class__.__name__}"
                f" | parameters={parameter_count:,} | {layer}"
            )
        return "\n".join(lines)

    def export_segtooth_checkpoint(
        self,
        destination: str | Path,
    ) -> Path:
        """Copy an Ultralytics ``best.pt`` to SegmentAnyTooth's filename."""

        trainer = self.ultralytics_model.trainer
        if trainer is None:
            raise RuntimeError("No trained best checkpoint is available to export")

        source_path = Path(trainer.best).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Trained checkpoint does not exist: {source_path}")
        destination_path = Path(destination).expanduser().resolve()
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        return destination_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the explicit layer structure of a YOLO11 tooth checkpoint."
    )
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    print(YOLO11ToothDetector(args.checkpoint).architecture_text())


if __name__ == "__main__":
    main()
