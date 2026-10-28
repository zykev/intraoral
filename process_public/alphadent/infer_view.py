#!/usr/bin/env python3
"""Infer the intraoral view (F / R / L / U / D) of every AlphaDent image with a Qwen-VL API.

AlphaDent file names look like ``p002_F_52_001.jpg`` where the second token is the
patient **sex** (F/M), *not* the photographic view, so the view has to be inferred
from the image content.

For every image a 2x2 grid is built from the 4 possible 90-degree rotations of the
photo (panel A = as-is, B = rotated 90 CW, C = 180, D = 270 CW).  The VLM is asked

  1. which anatomical view the photo shows, and
  2. which of the four panels is the correctly *upright* orientation.

Panel ``k`` is the original image rotated clockwise by ``90 * k`` degrees, so
``rotation = 90 * k`` is exactly the number of degrees that must be applied to the
original file to make it upright.  Answering with a grid makes the question much
easier for the model than "how many degrees should I rotate", which a VLM answers
very unreliably.

The result is written to ``<base_dir>/view_labels.json``::

    {
      "train/p002_F_52_001": {
        "view": "F",
        "rotation": 0,
        "confidence": 1.0,
        "upright_panel": "A",
        "reason": "...",
        "model": "qwen3.7-plus",
        "image_size": [5472, 3648],
        "ts": "2026-01-01T00:00:00"
      },
      ...
    }

Usage
-----
    python process_public/alphadent/infer_view.py --limit 8 --dry_run
    python process_public/alphadent/infer_view.py --splits train valid
    python process_public/alphadent/infer_view.py --splits train valid test --num_workers 4
    python process_public/alphadent/infer_view.py --stats

The key can be supplied with ``--api_key`` or the ``QWEN_API_KEY`` environment
variable; a default (workspace style) key is baked in for convenience.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_DIR = REPO_ROOT / ".datasets" / "intraoral_public" / "alphadent"
DEFAULT_OUT = "view_labels.json"

# Alibaba Model Studio (Bailian) workspace key -> Hong Kong compatible-mode endpoint.
DEFAULT_BASE_URL = "https://cn-hongkong.dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.7-plus"
DEFAULT_API_KEY = os.environ.get(
    "QWEN_API_KEY",
    "sk-ws-H.RMPRRPY.ygV3.MEUCIFspjkBjTLLGlvb_iPrm8hIHgQzzqtrvgtc8QkBT2mbgAiEA"
    "--1tVrxQl-J8Zmo3PdNiCgW7H9AZNhW5kAzJ1BYZH2c",
)

VALID_VIEWS = ("F", "R", "L", "U", "D")

# ``A`` = original, ``B`` = +90 CW, ``C`` = +180, ``D`` = +270 CW
PANEL_LABELS = "ABCD"

VIEW_PROMPT = (
    "The image is a 2x2 grid of four panels (A = top-left, B = top-right, "
    "C = bottom-left, D = bottom-right). All four panels are the SAME intraoral "
    "(dental) photograph, each one rotated by a different multiple of 90 degrees.\n\n"
    "Task 1 - identify the anatomical photographic view:\n"
    "  F = frontal / buccal: the anterior teeth of BOTH arches are visible, upper and "
    "lower incisors face each other, the dental midline is present, upper arch above "
    "lower arch.\n"
    "  R = right buccal / lateral: the teeth of ONE side are seen in profile from "
    "outside the cheek; upper arch above lower arch; no frontal midline.\n"
    "  L = left buccal / lateral: same as R but the other side of the mouth.\n"
    "  U = maxillary (upper) occlusal: the hard palate fills the middle of the image "
    "and the whole upper dental arch forms a ring / horseshoe shape.\n"
    "  D = mandibular (lower) occlusal: the tongue and floor of the mouth fill the "
    "middle and the whole lower dental arch forms a ring / horseshoe shape.\n"
    "  Note: the photo may be rotated, use the anatomy (palate vs tongue, upper vs "
    "lower arch) and NOT the image borders to decide the view.\n\n"
    "Task 2 - choose the ONE panel that shows the photo in the correct upright "
    "orientation for that view:\n"
    "  upright F: upper teeth along the TOP edge, lower teeth along the BOTTOM edge, "
    "midline vertical.\n"
    "  upright U: anterior (incisor) teeth along the TOP edge, molars along the "
    "BOTTOM edge.\n"
    "  upright D: anterior (incisor) teeth along the BOTTOM edge, molars along the "
    "TOP edge.\n"
    "  upright R: posterior teeth along the LEFT edge, anterior teeth along the "
    "RIGHT edge.\n"
    "  upright L: posterior teeth along the RIGHT edge, anterior teeth along the "
    "LEFT edge.\n"
    "  If several panels look plausible (e.g. a symmetric frontal shot), pick the "
    "panel whose orientation matches the description above most closely.\n\n"
    "Answer with ONE compact JSON object and nothing else:\n"
    '{"view": "F|R|L|U|D", "upright_panel": "A|B|C|D", '
    '"confidence": 0.0-1.0, "reason": "<= 20 words"}'
)


# --------------------------------------------------------------------------- #
# Geometry / encoding helpers
# --------------------------------------------------------------------------- #

def rotate_image(image: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate ``image`` clockwise by a multiple of 90 degrees."""
    degrees = int(degrees) % 360
    if degrees == 0:
        return image
    if degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"rotation must be a multiple of 90, got {degrees}")


def _fit_panel(image: np.ndarray, size: int) -> np.ndarray:
    """Letterbox ``image`` onto a black ``size x size`` canvas."""
    h, w = image.shape[:2]
    scale = size / max(h, w)
    resized = cv2.resize(image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def build_rotation_grid(image: np.ndarray, panel_size: int = 640, border: bool = True) -> np.ndarray:
    """Build the 2x2 grid used by the prompt: A=0, B=90 CW, C=180, D=270 CW."""
    panels = []
    for idx, degrees in enumerate((0, 90, 180, 270)):
        panel = _fit_panel(rotate_image(image, degrees), panel_size)
        if border:
            cv2.rectangle(panel, (1, 1), (panel_size - 2, panel_size - 2), (255, 255, 255), 3)
        cv2.putText(panel, PANEL_LABELS[idx], (14, 62), cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, (0, 0, 255), 5, cv2.LINE_AA)
        panels.append(panel)
    top = np.hstack(panels[:2])
    bottom = np.hstack(panels[2:])
    return np.vstack([top, bottom])


def encode_jpeg_data_url(image: np.ndarray, quality: int = 88) -> str:
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("failed to encode image as JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


# --------------------------------------------------------------------------- #
# Model call
# --------------------------------------------------------------------------- #

def parse_json_object(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from a (possibly fenced) model answer."""
    if not text:
        raise ValueError("empty model answer")
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in model answer: {text[:200]!r}")
    return json.loads(match.group(0))


def call_qwen(
    base_url: str,
    api_key: str,
    model: str,
    image: np.ndarray,
    prompt: str,
    timeout: float = 180.0,
    max_tokens: int = 512,
    temperature: float = 0.0,
    thinking: bool = False,
) -> str:
    """Send one image + prompt to the OpenAI-compatible chat endpoint."""
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": encode_jpeg_data_url(image)}},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "enable_thinking": bool(thinking),
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"].get("content") or ""


def normalise_answer(answer: Dict[str, Any]) -> Tuple[str, int, float, str, str]:
    """Turn the raw model JSON into ``(view, rotation, confidence, panel, reason)``."""
    view = str(answer.get("view", "")).strip().upper()[:1]
    if view not in VALID_VIEWS:
        raise ValueError(f"invalid view in model answer: {answer.get('view')!r}")

    panel = str(answer.get("upright_panel", "")).strip().upper()[:1]
    if panel not in PANEL_LABELS:
        # Fall back to "no rotation" but flag it through the confidence value.
        panel = "A"
        confidence = 0.0
    else:
        try:
            confidence = float(answer.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    rotation = 90 * PANEL_LABELS.index(panel)
    reason = str(answer.get("reason", ""))[:200]
    return view, rotation, confidence, panel, reason


def infer_one(
    image_path: Path,
    args: argparse.Namespace,
    panel_size: int,
) -> Dict[str, Any]:
    """Run the whole inference for a single image (with retries)."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    grid = build_rotation_grid(image, panel_size=panel_size)

    last_error: Optional[Exception] = None
    for attempt in range(args.retries + 1):
        try:
            raw = call_qwen(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                image=grid,
                prompt=VIEW_PROMPT,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                thinking=args.thinking,
            )
            answer = parse_json_object(raw)
            view, rotation, confidence, panel, reason = normalise_answer(answer)
            height, width = image.shape[:2]
            return {
                "view": view,
                "rotation": rotation,
                "confidence": confidence,
                "upright_panel": panel,
                "reason": reason,
                "model": args.model,
                "image_size": [int(width), int(height)],
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
        except Exception as exc:  # noqa: BLE001 - network/parse errors are expected
            last_error = exc
            if attempt < args.retries:
                time.sleep(args.retry_wait * (attempt + 1))
    raise RuntimeError(f"{image_path.name}: {last_error}")


# --------------------------------------------------------------------------- #
# Dataset plumbing
# --------------------------------------------------------------------------- #

def collect_images(base_dir: Path, splits: Iterable[str], limit: Optional[int]) -> List[Tuple[str, Path]]:
    """Return ``(split, path)`` for every image file of the requested splits."""
    tasks: List[Tuple[str, Path]] = []
    for split in splits:
        image_dir = base_dir / "images" / split
        if not image_dir.is_dir():
            print(f"[warn] missing split directory: {image_dir}")
            continue
        files = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if limit is not None:
            files = files[:limit]
        tasks.extend((split, path) for path in files)
    return tasks


def load_results(out_path: Path) -> Dict[str, Any]:
    if out_path.is_file():
        try:
            with out_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            print(f"[warn] could not parse {out_path}, starting from scratch")
    return {}


def save_results(out_path: Path, results: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(sorted(results.items())), handle, indent=1, ensure_ascii=False)
    tmp_path.replace(out_path)


def print_stats(results: Dict[str, Any]) -> None:
    per_split: Dict[str, Dict[str, int]] = {}
    errors = 0
    low_conf = []
    for key, entry in sorted(results.items()):
        split = key.split("/", 1)[0]
        if "error" in entry:
            errors += 1
            continue
        view = entry.get("view", "?")
        per_split.setdefault(split, {}).setdefault(view, 0)
        per_split[split][view] += 1
        if float(entry.get("confidence", 1.0)) < 0.7:
            low_conf.append((key, entry.get("view"), entry.get("confidence")))

    print("\n=== view statistics ===")
    for split, counts in sorted(per_split.items()):
        total = sum(counts.values())
        detail = "  ".join(f"{view}:{counts.get(view, 0)}" for view in VALID_VIEWS)
        print(f"  {split:<6} total={total:<5} {detail}")
    if errors:
        print(f"  errors: {errors}")
    if low_conf:
        print(f"  low confidence (<0.7): {len(low_conf)}")
        for key, view, conf in low_conf[:20]:
            print(f"    {key} -> {view} ({conf})")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer F/R/L/U/D views for AlphaDent images with the Qwen-VL API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base_dir", type=Path, default=DEFAULT_BASE_DIR,
                        help="AlphaDent dataset root.")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"],
                        help="Dataset splits to process.")
    parser.add_argument("--out", type=Path, default=None,
                        help=f"Output JSON path (default: <base_dir>/{DEFAULT_OUT}).")
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL, help="OpenAI-compatible endpoint.")
    parser.add_argument("--api_key", default=DEFAULT_API_KEY, help="API key (env QWEN_API_KEY).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name.")
    parser.add_argument("--panel_size", type=int, default=640, help="Pixel size of each grid panel.")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max completion tokens.")
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-request timeout (s).")
    parser.add_argument("--retries", type=int, default=3, help="Retries per image.")
    parser.add_argument("--retry_wait", type=float, default=3.0, help="Base wait between retries (s).")
    parser.add_argument("--num_workers", type=int, default=4, help="Concurrent API requests.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N images per split.")
    parser.add_argument("--overwrite", action="store_true", help="Re-infer images already in the output.")
    parser.add_argument("--thinking", action="store_true", help="Do not disable model reasoning.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only write the grid previews, do not call the API.")
    parser.add_argument("--preview_dir", type=Path, default=None,
                        help="Where --dry_run writes the grids (default <base_dir>/view_grid_preview).")
    parser.add_argument("--stats", action="store_true", help="Only print statistics of the existing output.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out_path = args.out or (args.base_dir / DEFAULT_OUT)

    if args.stats:
        print_stats(load_results(out_path))
        return 0

    tasks = collect_images(args.base_dir, args.splits, args.limit)
    if not tasks:
        print("no images found")
        return 1

    results = load_results(out_path)
    pending = [(split, path) for split, path in tasks
               if args.overwrite or f"{split}/{path.stem}" not in results]
    skipped = len(tasks) - len(pending)
    print(f"{len(tasks)} images, {skipped} already labelled, {len(pending)} to infer")
    print(f"endpoint: {args.base_url}  model: {args.model}")

    if args.dry_run:
        preview_dir = args.preview_dir or (args.base_dir / "view_grid_preview")
        preview_dir.mkdir(parents=True, exist_ok=True)
        for index, (split, path) in enumerate(pending, 1):
            image = cv2.imread(str(path))
            if image is None:
                print(f"[warn] cannot read {path}")
                continue
            grid = build_rotation_grid(image, panel_size=args.panel_size)
            cv2.imwrite(str(preview_dir / f"{split}__{path.stem}.jpg"), grid,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if index % 25 == 0 or index == len(pending):
                print(f"  wrote {index}/{len(pending)} previews")
        print(f"dry run done -> {preview_dir}")
        return 0

    if not pending:
        print_stats(results)
        return 0

    lock = threading.Lock()
    done = 0
    failures: List[str] = []

    def worker(split: str, path: Path) -> Tuple[str, Dict[str, Any]]:
        key = f"{split}/{path.stem}"
        return key, infer_one(path, args, args.panel_size)

    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as pool:
        futures = {pool.submit(worker, split, path): (split, path) for split, path in pending}
        for future in as_completed(futures):
            split, path = futures[future]
            key = f"{split}/{path.stem}"
            try:
                _, entry = future.result()
            except Exception as exc:  # noqa: BLE001
                entry = {"error": str(exc), "ts": datetime.now().isoformat(timespec="seconds")}
                failures.append(key)
                print(f"[error] {key}: {exc}")
            with lock:
                results[key] = entry
                done += 1
                if done % 10 == 0 or done == len(pending):
                    save_results(out_path, results)
            if "error" not in entry:
                print(f"[{done}/{len(pending)}] {key} -> {entry['view']} "
                      f"rot={entry['rotation']} conf={entry['confidence']:.2f}")

    save_results(out_path, results)
    print(f"\nresults written to {out_path}  ({len(results)} entries, {len(failures)} failures)")
    print_stats(results)
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
