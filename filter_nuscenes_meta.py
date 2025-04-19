# File: filter_nuscenes_meta.py
# Place this at the same level as nuscenes_meta/ and nuscenes_blobs/

import os
import json
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────
META_ROOT   = Path("nuscenes_meta/v1.0-trainval")
OUT_ROOT    = Path("nuscenes_meta/subset_trainval01")
SAMPLES_JSON = Path("nuscenes_blobs/train_samples.json")

# ─── UTILS ────────────────────────────────────────────────────────
def load_json(p):    return json.load(open(p,  "r"))
def dump_json(o,p):  json.dump(o, open(p,"w"), indent=2)

# ─── STEP 1: collect all sample_tokens from your paired-dataset ───
paired = load_json(SAMPLES_JSON)
sample_tokens = {s["sample_token"] for s in paired}

# ─── STEP 2: filter every table that references sample_token ──────
#    We'll keep only those entries whose 'token' is in sample_tokens,
#    then also gather all downstream tokens (e.g. scene, annotation).
#
#  Tables we need to slim:
#   • sample.json
#   • sample_data.json
#   • sample_annotation.json
#   • instance.json
#   • calibrated_sensor.json
#   • ego_pose.json
#   • scene.json
#   • log.json
#
#  The 'static' tables you can copy wholesale:
#   • category.json, attribute.json, sensor.json, visibility.json

# Load full tables
sample_table     = load_json(META_ROOT / "sample.json")
sd_table         = load_json(META_ROOT / "sample_data.json")
ann_table        = load_json(META_ROOT / "sample_annotation.json")
inst_table       = load_json(META_ROOT / "instance.json")
cs_table         = load_json(META_ROOT / "calibrated_sensor.json")
ego_table        = load_json(META_ROOT / "ego_pose.json")
scene_table      = load_json(META_ROOT / "scene.json")
log_table        = load_json(META_ROOT / "log.json")

# 2.1 filter samples
subset_samples = [s for s in sample_table     if s["token"] in sample_tokens]
# collect scene & log tokens
scene_tokens = {s["scene_token"] for s in subset_samples}
log_tokens   = {sc["log_token"]   for sc in scene_table if sc["token"] in scene_tokens}

# 2.2 filter sample_data
subset_sd = [sd for sd in sd_table 
             if sd["sample_token"] in sample_tokens]
# collect all sensor & calibrated_sensor & ego_pose tokens
calib_tokens = {sd["calibrated_sensor_token"] for sd in subset_sd}
ego_tokens   = {sd["ego_pose_token"]          for sd in subset_sd}

# 2.3 filter annotations
subset_ann   = [a for a in ann_table 
                if a["sample_token"] in sample_tokens]
# collect instance tokens
instance_tokens = {a["instance_token"] for a in subset_ann}

# 2.4 filter instance / calibrated_sensor / ego_pose
subset_inst = [i for i in inst_table    if i["token"] in instance_tokens]
subset_cs   = [cs for cs in cs_table    if cs["token"] in calib_tokens]
subset_ego  = [e  for e  in ego_table   if e["token"] in ego_tokens]
subset_scene= [sc for sc in scene_table if sc["token"] in scene_tokens]
subset_log  = [l  for l  in log_table   if l["token"] in log_tokens]

# ─── STEP 3: copy static tables wholesale ─────────────────────────
static_tables = ["category.json","attribute.json","sensor.json","visibility.json","map.json"]

# ─── STEP 4: write out everything under OUT_ROOT/v1.0-trainval ───
for tbl_name, tbl_data in [
    ("sample.json",            subset_samples),
    ("sample_data.json",       subset_sd),
    ("sample_annotation.json", subset_ann),
    ("instance.json",          subset_inst),
    ("calibrated_sensor.json", subset_cs),
    ("ego_pose.json",          subset_ego),
    ("scene.json",             subset_scene),
    ("log.json",               subset_log),
]:
    out_p = OUT_ROOT / tbl_name
    out_p.parent.mkdir(parents=True, exist_ok=True)
    dump_json(tbl_data, out_p)

# copy static
for name in static_tables:
    src, dst = META_ROOT / name, OUT_ROOT / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "r") as f, open(dst, "w") as g:
        g.write(f.read())

print(f"📦 Created filtered metadata under {OUT_ROOT}")
print(f"   • {len(subset_samples)} samples")
print(f"   • {len(subset_sd)} sample_data entries")
print(f"   • {len(subset_ann)} annotations")
print(f"   • + other tables")
