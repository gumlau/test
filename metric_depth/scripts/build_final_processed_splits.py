#!/usr/bin/env python3
"""Generate train/validation split files for the final_processed dataset."""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency at authoring time
    Image = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("..") / "final_version_processed",
        help="Root directory that contains s*_processed folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "train_test_inputs",
        help="Destination directory for the generated split files.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of samples to reserve for validation (0-1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for shuffling before splitting.",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["left"],
        help="Camera folders to include (e.g., left right).",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="final_processed_train.txt",
        help="Filename for the training split within the output directory.",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default="final_processed_val.txt",
        help="Filename for the validation split within the output directory.",
    )
    parser.add_argument(
        "--default-focal",
        type=float,
        default=1.0,
        help="Fallback focal length (in pixels) when intrinsics are missing.",
    )
    return parser.parse_args()


def normalise_path(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return "/" + str(rel).replace("\\", "/")


def load_intrinsics(scene_root: Path) -> Dict[str, Dict[str, dict]]:
    intr_path = scene_root / "instrincs.json"
    if not intr_path.exists():
        return {}
    with intr_path.open("r") as fh:
        data = json.load(fh)
    results: Dict[str, Dict[str, dict]] = {}
    for entry in data:
        name = entry.get("name")
        if not name:
            continue
        results[name] = {
            "left": entry.get("camera_l", {}),
            "right": entry.get("camera_r", {}),
        }
    return results


def focal_from_intrinsics(
    intr: Dict[str, dict],
    img_size: Tuple[int, int],
    default_focal: float,
) -> float:
    lens = intr.get("lens")
    sensor_width = intr.get("sensor_width")
    if lens is None or sensor_width in (None, 0):
        return default_focal
    img_width, _ = img_size
    return float(lens) / float(sensor_width) * float(img_width)


def iter_scene_dirs(dataset_root: Path) -> List[Path]:
    """Return available scene directories even if dataset_root is itself a scene."""
    if not dataset_root.exists():
        return []

    candidates: List[Path] = []
    if dataset_root.is_dir():
        if dataset_root.name.startswith("s") and dataset_root.name.endswith("_processed"):
            candidates.append(dataset_root)
        candidates.extend(sorted(p for p in dataset_root.glob("s*_processed") if p.is_dir()))

    seen: Set[Path] = set()
    ordered: List[Path] = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def collect_samples(
    dataset_root: Path,
    cameras: Iterable[str],
    default_focal: float,
) -> List[Tuple[str, str, str, float]]:
    samples: List[Tuple[str, str, str, float]] = []
    for scene_dir in iter_scene_dirs(dataset_root):
        intrinsics = load_intrinsics(scene_dir)
        for camera in cameras:
            cam_dir = scene_dir / camera
            img_dir = cam_dir / "imgs"
            depth_dir = cam_dir / "depth"
            mask_dir = cam_dir / "valid_region_mask"
            if not (img_dir.exists() and depth_dir.exists() and mask_dir.exists()):
                continue

            # Cache image size for focal conversion
            try:
                first_image = next(iter(sorted(img_dir.glob("*"))))
            except StopIteration:
                continue
            if Image is not None:
                with Image.open(first_image) as ref_img:
                    img_size = ref_img.size
            else:
                img_size = (512, 512)

            for img_path in sorted(img_dir.glob("*")):
                stem = img_path.name
                depth_path = depth_dir / stem
                mask_path = mask_dir / stem
                if not (depth_path.exists() and mask_path.exists()):
                    continue
                intr = intrinsics.get(stem, {}).get(
                    "left" if camera == "left" else "right", {})
                focal = focal_from_intrinsics(intr, img_size, default_focal)

                samples.append(
                    (
                        normalise_path(img_path, dataset_root),
                        normalise_path(depth_path, dataset_root),
                        normalise_path(mask_path, dataset_root),
                        focal,
                    )
                )
    return samples


def split_samples(samples: List[Tuple[str, str, str, float]], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    if not samples:
        return [], []
    random.Random(seed).shuffle(samples)
    val_count = int(len(samples) * val_ratio)
    val_count = max(1, val_count) if len(samples) > 1 else len(samples)
    val_samples = samples[:val_count]
    train_samples = samples[val_count:] if val_count < len(samples) else samples[val_count:]

    def format_sample(sample: Tuple[str, str, str, float]) -> str:
        img_rel, depth_rel, mask_rel, focal = sample
        return f"{img_rel} {depth_rel} {mask_rel} {focal:.6f}"

    return [format_sample(s) for s in train_samples], [format_sample(s) for s in val_samples]


def main() -> None:
    args = parse_args()
    samples = collect_samples(args.dataset_root, args.cameras, args.default_focal)
    if not samples:
        raise SystemExit("No samples found. Check dataset_root and camera names.")

    train_lines, val_lines = split_samples(samples, args.val_ratio, args.seed)
    if not train_lines:
        raise SystemExit("Training split is empty. Adjust --val-ratio or dataset content.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / args.train_file).write_text("\n".join(train_lines) + "\n")
    (args.output_dir / args.val_file).write_text("\n".join(val_lines) + "\n")

    print(f"Wrote {len(train_lines)} train samples and {len(val_lines)} val samples to {args.output_dir}.")


if __name__ == "__main__":
    main()
