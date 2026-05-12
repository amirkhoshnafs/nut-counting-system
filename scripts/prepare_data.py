"""Prepare datasets for YOLO-NAS training.

Merges one or more Roboflow COCO-format datasets into a combined
four-way split (train / val / calibration / test).

Usage:
    python scripts/prepare_data.py           # all datasets combined
    python scripts/prepare_data.py --single nut_counting   # single dataset
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

# ── Project root ─────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data/raw"

# ── Split fractions ──────────────────────────────────

TRAIN_FRAC = 0.70
VAL_FRAC = 0.10
CAL_FRAC = 0.10
TEST_FRAC = 0.10

# ── Category normalisation ───────────────────────────
# Canonical names and IDs.  Map dataset-specific names to these.

CANONICAL_CLASSES = {
    "nut": 0,
    "bolt": 1,
}

# per-dataset name → canonical name
NAME_TO_CANONICAL = {
    # Nut Counting System
    "nut": "nut", "nuts": "nut", "Nuts": "nut",
    # Nuts and Bolts Detection
    "bolt": "bolt",
    # Nuts and Bolts Segmentation
    "Nut": "nut", "Bolt": "bolt",
}

# ── Dataset definitions ──────────────────────────────

ALL_DATASETS = {
    "nut_counting": {
        "raw_dir": RAW_DIR / "nut_counting",
        "splits": ["train", "valid"],
    },
    "nuts_and_bolts_det": {
        "raw_dir": RAW_DIR / "nuts and bolts.v1i.coco",
        "splits": ["train", "valid"],
    },
    "nuts_and_bolts_seg": {
        "raw_dir": RAW_DIR / "Nuts and Bolts.v1i.coco",
        "splits": ["train", "valid", "test"],
    },
}

# ── Helpers ───────────────────────────────────────────

RF_PATTERN = re.compile(r"^(.*)\.rf\.[0-9a-f]+\.jpg$", re.IGNORECASE)


def extract_source_key(filename: str) -> str:
    """Strip Roboflow augmentation hash for grouping.

    'istockphoto-123_jpg.rf.abc123def456.jpg' → 'istockphoto-123_jpg'
    """
    m = RF_PATTERN.match(filename)
    return m.group(1) if m else Path(filename).stem


def load_coco(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ── Merge logic ──────────────────────────────────────

def merge_all_datasets(dataset_names: list[str]) -> tuple[dict, dict[str, list[dict]]]:
    """Merge multiple COCO datasets into one.

    Returns (merged_coco_dict, source_groups).
    source_groups maps "dataset_name/source_key" → list of image dicts.
    """
    merged_images: list[dict] = []
    merged_annotations: list[dict] = []
    merged_categories: list[dict] = []

    # (dataset_name, old_cat_id) → new_cat_id
    cat_id_map: dict[tuple[str, int], int] = {}

    img_id_offset = 0
    ann_id_offset = 0

    for ds_name in dataset_names:
        ds = ALL_DATASETS[ds_name]

        for split_name in ds["splits"]:
            json_path = ds["raw_dir"] / split_name / "_annotations.coco.json"
            if not json_path.exists():
                print(f"  SKIP: {json_path} not found")
                continue

            data = load_coco(json_path)

            # Identify used category IDs in this split
            used_ids = {a["category_id"] for a in data.get("annotations", [])}

            # Build category mapping for this dataset
            for cat in data.get("categories", []):
                if cat["id"] not in used_ids:
                    continue  # skip supercategory placeholders
                canonical_name = NAME_TO_CANONICAL.get(cat["name"])
                if canonical_name is None:
                    print(f"  WARNING: unknown class '{cat['name']}' in {ds_name}/{split_name}, skipping")
                    continue
                new_id = CANONICAL_CLASSES[canonical_name]
                key = (ds_name, cat["id"])
                if key not in cat_id_map:
                    cat_id_map[key] = new_id

            # Remap images
            img_id_map: dict[int, int] = {}
            for img in data.get("images", []):
                new_img_id = img["id"] + img_id_offset
                img_id_map[img["id"]] = new_img_id
                merged_images.append({
                    **img,
                    "id": new_img_id,
                    "_ds": ds_name,
                    "_split": split_name,
                })

            # Remap annotations
            for ann in data.get("annotations", []):
                key = (ds_name, ann["category_id"])
                new_cat_id = cat_id_map.get(key)
                if new_cat_id is None:
                    continue
                merged_annotations.append({
                    **ann,
                    "id": ann["id"] + ann_id_offset,
                    "image_id": img_id_map[ann["image_id"]],
                    "category_id": new_cat_id,
                })

            img_id_offset += len(data.get("images", []))
            ann_id_offset += len(data.get("annotations", []))

    # Build merged categories list (sorted by ID)
    seen = set()
    for (ds_name, old_id), new_id in sorted(cat_id_map.items(), key=lambda x: x[1]):
        if new_id not in seen:
            seen.add(new_id)
            name = [k for k, v in CANONICAL_CLASSES.items() if v == new_id][0]
            merged_categories.append({"id": new_id, "name": name})

    merged = {
        "info": {},
        "licenses": [],
        "categories": merged_categories,
        "images": merged_images,
        "annotations": merged_annotations,
    }

    # Group images by source key (prefixed with dataset name to avoid collisions)
    groups: dict[str, list[dict]] = defaultdict(list)
    for img in merged_images:
        ds_name = img.get("_ds", "")
        key = f"{ds_name}/{extract_source_key(img['file_name'])}"
        groups[key].append(img)

    return merged, groups


# ── Split logic ──────────────────────────────────────

def split_groups(
    groups: dict,
    train_f: float, val_f: float, cal_f: float, test_f: float,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str], list[str]]:
    keys = sorted(groups.keys())

    train_val_cal, test_keys = train_test_split(keys, test_size=test_f, random_state=seed)
    remaining = train_f + val_f
    cal_frac = cal_f / remaining
    train_val, cal_keys = train_test_split(train_val_cal, test_size=cal_frac, random_state=seed)
    val_frac_of_tv = val_f / train_f
    train_keys, val_keys = train_test_split(train_val, test_size=val_frac_of_tv, random_state=seed)

    return train_keys, val_keys, cal_keys, test_keys


# ── Write logic ──────────────────────────────────────

def write_split(
    name: str,
    keys: list[str],
    groups: dict[str, list[dict]],
    merged: dict,
    out_dir: Path,
):
    """Write images/<name>/ and annotations/<name>.json for one split."""
    img_dir = out_dir / "images" / name
    ann_dir = out_dir / "annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    key_set = set(keys)
    split_images = []
    split_ids = set()
    for key in key_set:
        for img in groups[key]:
            split_images.append(img)
            split_ids.add(img["id"])

    split_annotations = [a for a in merged["annotations"] if a["image_id"] in split_ids]

    # Copy images from their source locations
    for img in split_images:
        fname = img["file_name"]
        ds_name = img.get("_ds", "")
        ds = ALL_DATASETS.get(ds_name)
        if ds is None:
            print(f"  WARNING: unknown dataset {ds_name} for {fname}")
            continue
        found = False
        for split_name in ds["splits"]:
            src = ds["raw_dir"] / split_name / fname
            if src.exists():
                shutil.copy2(src, img_dir / fname)
                found = True
                break
        if not found:
            print(f"  WARNING: {fname} not found in any split of {ds_name}")

    # Write COCO JSON — strip internal fields
    clean_images = [{k: v for k, v in img.items() if not k.startswith("_")} for img in split_images]
    coco = {
        "info": merged.get("info", {}),
        "licenses": merged.get("licenses", []),
        "categories": merged["categories"],
        "images": clean_images,
        "annotations": split_annotations,
    }
    json_path = ann_dir / f"{name}.json"
    with open(json_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"  {name}: {len(clean_images)} images, {len(split_annotations)} annotations")


# ── Main ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", default=None, help="Process only one dataset (by key)")
    parser.add_argument("--output", default="combined", help="Output directory name under data/processed/")
    args = parser.parse_args()

    dataset_names = [args.single] if args.single else list(ALL_DATASETS.keys())
    out_dir = PROJECT_ROOT / "data/processed" / args.output

    print("=" * 60)
    print(f"Data Preparation — {', '.join(dataset_names)}")
    print(f"Output: {out_dir}")
    print("=" * 60)

    # 1. Merge
    print("\n[1/4] Merging datasets ...")
    merged, groups = merge_all_datasets(dataset_names)
    print(f"  Total images:       {len(merged['images'])}")
    print(f"  Total annotations:  {len(merged['annotations'])}")
    print(f"  Unique sources:     {len(groups)}")
    print(f"  Categories:         {merged['categories']}")

    # Per-class counts
    for cat in merged["categories"]:
        count = sum(1 for a in merged["annotations"] if a["category_id"] == cat["id"])
        print(f"    {cat['name']}: {count} annotations")

    if len(groups) == 0:
        sys.exit("ERROR: No source groups found. Check dataset paths.")

    # 2. Group by source
    print("\n[2/4] Grouping by source key (pre-augmentation) ...")
    avg_aug = len(merged["images"]) / len(groups)
    print(f"  Avg augmentations per source: {avg_aug:.1f}")

    # 3. Split
    print(f"\n[3/4] Splitting (train={TRAIN_FRAC:.0%} val={VAL_FRAC:.0%} cal={CAL_FRAC:.0%} test={TEST_FRAC:.0%}) ...")
    train_keys, val_keys, cal_keys, test_keys = split_groups(
        groups, TRAIN_FRAC, VAL_FRAC, CAL_FRAC, TEST_FRAC,
    )
    print(f"  Source groups → train: {len(train_keys)}, val: {len(val_keys)}, "
          f"cal: {len(cal_keys)}, test: {len(test_keys)}")

    # 4. Write
    print("\n[4/4] Writing splits ...")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    write_split("train", train_keys, groups, merged, out_dir)
    write_split("val", val_keys, groups, merged, out_dir)
    write_split("calibration", cal_keys, groups, merged, out_dir)
    write_split("test", test_keys, groups, merged, out_dir)

    print("\n" + "=" * 60)
    print(f"Done. Output: {out_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
