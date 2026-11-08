"""Step 3: Train the location model, score every pitch, and roll scores up.

Model: gradient boosting regression predicting pitch_value (-delta_run_exp)
from location, pitch type, batter handedness, count, and an ABS-era flag.

loc_score = loc_value minus the league average for that pitch type and year,
so 0 = league average. runs_per_100 = mean loc_score * 100.
"""
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# ---------- 1. Load and set up ----------
df = pd.read_pickle("clean_pitches_v2.pkl")
df["pt"] = df["pitch_type"].cat.codes
df["abs_era"] = (df["game_year"] >= 2026).astype("int8")  # 1 = ABS challenge era

features = ["x_norm_io", "z_norm", "pt", "stand_R", "balls", "strikes", "abs_era"]

# ---------- 2. Train on all data ----------
print("Training model... (a few minutes)")
model = HistGradientBoostingRegressor(
    max_iter=200, learning_rate=0.1, categorical_features=[2]
)
model.fit(df[features], df["pitch_value"])
joblib.dump(model, "location_model_b.pkl")
print("Saved location_model_b.pkl")

# ---------- 3. Score every pitch ----------
df["loc_value"] = model.predict(df[features])
df["loc_score"] = df["loc_value"] - df.groupby(
    ["game_year", "pitch_type"], observed=True)["loc_value"].transform("mean")

df[["game_pk", "game_year", "pitcher", "batter", "pitch_type",
    "x_norm_io", "z_norm", "loc_value", "loc_score"]].to_pickle("scored_pitches.pkl")
print("Saved scored_pitches.pkl")

# ---------- 4. Roll up ----------
def roll(keys):
    out = (df.groupby(keys, observed=True)
             .agg(pitches=("loc_score", "size"), loc_score=("loc_score", "mean"))
             .reset_index())
    out["runs_per_100"] = (out["loc_score"] * 100).round(3)
    return out

outing = roll(["pitcher", "game_pk", "game_year"])
season = roll(["pitcher", "game_year"])
career = roll(["pitcher"])
season_pt = roll(["pitcher", "game_year", "pitch_type"])

# ---------- 5. Add player names (needs internet) ----------
try:
    from pybaseball import playerid_reverse_lookup
    ids = df["pitcher"].unique().tolist()
    names = playerid_reverse_lookup(ids, key_type="mlbam")
    names["name"] = names["name_first"].str.title() + " " + names["name_last"].str.title()
    names = names[["key_mlbam", "name"]].rename(columns={"key_mlbam": "pitcher"})
    outing, season, career, season_pt = [
        t.merge(names, on="pitcher", how="left") for t in (outing, season, career, season_pt)
    ]
except Exception as e:
    print("Could not add names (need internet):", e)

# ---------- 6. Save ----------
outing.to_csv("scores_by_outing.csv", index=False)
season.to_csv("scores_by_season.csv", index=False)
career.to_csv("scores_by_career.csv", index=False)
season_pt.to_csv("scores_by_season_pitchtype.csv", index=False)

print("\nAverage score by year (about 0 by construction):")
print(df.groupby("game_year")["loc_score"].mean().round(5))
print("\nSaved scores_by_outing / season / career / season_pitchtype (.csv)")
