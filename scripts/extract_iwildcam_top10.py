#!/usr/bin/env python3
"""
Robust JPEG extractor  –  iWildCam 2022  →  Brazil-10 × 2 700 subset
"""

from __future__ import annotations
import json, pathlib, random, shutil
from collections import Counter, defaultdict
from typing import List
from tqdm import tqdm

ROOT    = pathlib.Path("data/iwildcam_full")
IMG_DIR = next((ROOT / d for d in ("train", "train_images") if (ROOT / d).is_dir()), None)
if IMG_DIR is None:
    raise SystemExit("❌ Could not find train/ or train_images/; check unzip.")

META = ROOT / "metadata/iwildcam2022_train_annotations.json"
GPS  = ROOT / "metadata/gps_locations.json"        # optional
OUT  = ROOT / "jpg"
OUT.mkdir(parents=True, exist_ok=True)

TARGET = 2_700
random.seed(0)

# ─── 1. load annotation + gps ───────────────────────────────────────
print("• Loading metadata …")
meta = json.loads(META.read_text())
id2img = {img["id"]: img for img in meta["images"]}
id2cat = {c["id"]: c["name"] for c in meta["categories"]}

# BBOX = (-75, -34, -35, 6)  # BRAZIL  lon[min,max], lat[min,max]
BBOX = (-82, -34, -55, 13)   # toda América do Sul + Panamá meridional
try:
    gps = json.loads(GPS.read_text())
    br_locs = {loc for loc, v in gps.items()
               if BBOX[2] <= v["latitude"] <= BBOX[3]
               and BBOX[0] <= v["longitude"] <= BBOX[1]}
    print(f"• GPS filter: {len(br_locs)} locations in Brazil")
except FileNotFoundError:
    br_locs = set()
    print("• gps_locations.json missing → Brazil filter disabled")

# ─── 2. gather file lists per species ───────────────────────────────
files_global: dict[str, List[pathlib.Path]] = defaultdict(list)
files_brazil: dict[str, List[pathlib.Path]] = defaultdict(list)

for ann in meta["annotations"]:
    img = id2img.get(ann["image_id"])
    if not img:                               # should not happen
        continue
    sp  = id2cat[ann["category_id"]]
    if sp == "empty":
        continue
    p = IMG_DIR / img["file_name"]
    if not p.exists():
        continue

    files_global[sp].append(p)
    if br_locs and img["location"] in br_locs:
        files_brazil[sp].append(p)

# ─── 3. pick 10 most frequent non-empty species ─────────────────────
freq = Counter({sp: len(files_global[sp]) for sp in files_global})
top10 = [sp for sp, _ in freq.most_common()[:10]]
print("• Selected species:", top10)

# ─── 4. copy / oversample to OUT/jpg/<class-id>/ ────────────────────
def ensure_2700(src_pool: List[pathlib.Path]) -> List[pathlib.Path]:
    if len(src_pool) >= TARGET:
        return random.sample(src_pool, TARGET)
    # oversample with replacement
    picks = src_pool.copy()
    while len(picks) < TARGET:
        picks += random.sample(src_pool, min(TARGET - len(picks), len(src_pool)))
    return picks[:TARGET]

for sp_id, sp in enumerate(top10):
    cls_dir = OUT / str(sp_id)
    shutil.rmtree(cls_dir, ignore_errors=True)
    cls_dir.mkdir(parents=True, exist_ok=True)

    pool = files_brazil[sp] or files_global[sp]  # prefer Brazil subset
    if not pool:
        print(f"⚠️  {sp} has 0 local JPEGs — skipped.")
        shutil.rmtree(cls_dir, ignore_errors=True)
        continue

    for src in ensure_2700(pool):
        dst = cls_dir / src.name
        if not dst.exists():
            shutil.copy(src, dst)

# ─── 5. summary ─────────────────────────────────────────────────────
total = sum(1 for _ in OUT.glob("*/*.jpg"))
print(f"✓ Extracted {total} JPEGs into {OUT}")
print("  classes:", [d.name for d in OUT.iterdir() if d.is_dir()])
print("\n👉  Loader path:\n    fp = root / 'jpg' / str(label_id) / pathlib.Path(img['file_name']).name")