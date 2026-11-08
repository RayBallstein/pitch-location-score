"""Step 2: Clean the raw monthly files and normalize pitch location.

Reads ./statcast_monthly/*.csv one month at a time (to keep memory low),
and writes clean_pitches_v2.pkl.

Normalization:
  x_norm = plate_x / 0.7083                         (plate edges at -1 and +1)
  z_norm = (plate_z - sz_bot) / (sz_top - sz_bot)   (0 = bottom, 1 = top of THAT batter's zone)
  x_norm_io: x_norm flipped for left-handed batters (positive = outside for everyone)
  pitch_value = -delta_run_exp                      (higher = better for the pitcher)
"""
import glob
import os

import numpy as np
import pandas as pd

cols = [
    "game_year", "game_pk", "pitcher", "batter", "stand", "pitch_type",
    "balls", "strikes", "plate_x", "plate_z", "sz_top", "sz_bot",
    "release_speed", "pfx_x", "pfx_z", "delta_run_exp",
]
float_cols = ["plate_x", "plate_z", "sz_top", "sz_bot",
              "release_speed", "pfx_x", "pfx_z", "delta_run_exp"]
dtypes = {c: "float32" for c in float_cols}

# Fixed pitch-type list keeps stacking months light on memory
PITCHES = ["FF", "SI", "FC", "SL", "ST", "SV", "CU", "KC", "CS",
           "CH", "FS", "FO", "KN", "EP", "SC"]

files = sorted(glob.glob("statcast_monthly/*.csv"))
print(f"Found {len(files)} files")

tables = []
removed = 0

for f in files:
    if os.path.getsize(f) <= 10:  # skip empty offseason files
        continue

    month = pd.read_csv(f, usecols=cols, dtype=dtypes)

    before = len(month)
    month = month.dropna(subset=["plate_x", "plate_z", "sz_top", "sz_bot",
                                 "pitch_type", "delta_run_exp"])
    zone_h = month["sz_top"] - month["sz_bot"]
    month = month[(zone_h > 0.5) & (zone_h < 4)].copy()

    month["x_norm"] = month["plate_x"] / 0.7083
    month["z_norm"] = (month["plate_z"] - month["sz_bot"]) / (month["sz_top"] - month["sz_bot"])

    # Remove tracking glitches
    month = month[month["x_norm"].between(-4, 4) & month["z_norm"].between(-1.5, 3)].copy()
    removed += before - len(month)

    month["stand_R"] = (month["stand"] == "R").astype("int8")
    month["x_norm_io"] = month["x_norm"] * np.where(month["stand_R"] == 1, 1, -1).astype("float32")
    month["pitch_value"] = -month["delta_run_exp"]
    month = month.drop(columns=["stand"])

    month["pitch_type"] = pd.Categorical(month["pitch_type"], categories=PITCHES)
    month["game_year"] = month["game_year"].astype("int16")
    month["balls"] = month["balls"].astype("int8")
    month["strikes"] = month["strikes"].astype("int8")

    tables.append(month)
    print(f"Loaded {os.path.basename(f)}: {len(month):,} pitches")

df = pd.concat(tables, ignore_index=True)
del tables

# Drop pitch types outside our list (pitchouts, automatic balls, etc.)
df = df.dropna(subset=["pitch_type"])

print(f"\nRemoved {removed:,} rows in total")
print(df.shape)
print(df["pitch_type"].value_counts())

df.to_pickle("clean_pitches_v2.pkl")
print("Saved clean_pitches_v2.pkl")
