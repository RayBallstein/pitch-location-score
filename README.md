# Pitch Location Score

A model that grades the **location** of every MLB pitch (2022-2026) and rolls the grades up to pitcher outings, seasons, and careers.

> **Not what happened to the pitch, but how good the spot was.**

## What it measures

For each pitch, the model estimates how many runs a pitch at that location is typically worth to the pitcher, given:

- where it crossed the plate, **normalized to the batter's own strike zone**
- pitch type
- count (balls, strikes)
- batter handedness
- whether it was thrown in the ABS challenge era (2026 onward)

Each pitch is then compared with the league average for the **same pitch type and season**:

```
loc_score    = predicted value of the pitch's location - league average for that pitch type and year
runs_per_100 = average loc_score * 100
```

**+0.5 runs_per_100** means the pitcher's locations were worth about half a run more than average per 100 pitches, given the pitch types he threw.

The model ignores velocity, movement, and the actual result. A pitcher can score poorly and still be effective (hard throwers often do).

## Method

1. **Data:** regular-season Statcast pitch data from Baseball Savant, downloaded with [pybaseball](https://github.com/jldbc/pybaseball). About 3.5 million pitches after cleaning.
2. **Normalization:**
   - `x_norm = plate_x / 0.7083` (plate edges at -1 and +1)
   - `z_norm = (plate_z - sz_bot) / (sz_top - sz_bot)` (0 = bottom, 1 = top of that batter's zone)
   - Flipped for left-handed batters so "outside" is always positive.
3. **Target:** `-delta_run_exp` (run value of each pitch, from the pitcher's side).
4. **Model:** gradient boosting (scikit-learn `HistGradientBoostingRegressor`).
5. **Centering:** scores are centered at 0 for each season and pitch type.

## How to run

**Rebuild the scores from scratch:**

```
pip install -r requirements-pipeline.txt
python 01_download_data.py     # downloads Statcast data (slow, run once)
python 02_build_dataset.py     # cleans and normalizes
python 03_train_and_score.py   # trains the model and writes score files
python 04_leaderboard.py       # prints a leaderboard
```

Output files: `scores_by_outing.csv`, `scores_by_season.csv`, `scores_by_career.csv`, `scores_by_season_pitchtype.csv`.

**Run the interactive app** (uses the included `app_data.csv`, no download needed):

```
pip install -r requirements.txt
python -m shiny run app.py
```

Then open the address it prints (usually http://127.0.0.1:8000). The app has a date-range filter, pitcher search, minimum-pitches filter, and a by-pitch-type view.

`app_data.csv` has one row per pitcher, game, and pitch type (pitch count and summed location score). Any date window is an exact average of the underlying pitch scores.

## Results

*(Paste your 2026 leaderboard here once you finalize the model.)*

## Model fit

On held-out 2026 games, the model's R-squared was about **0.036** at the single-pitch level. That is expected: run value on one pitch is mostly luck, and location explains only a small share. The scores are meant to be averaged over many pitches.

## ABS and the 2026 strike zone

The 2026 season introduced the ABS challenge system. In Statcast data, the average zone height (`sz_top - sz_bot`) fell from about 1.83 ft in 2025 to about 1.59 ft in 2026, mostly because `sz_top` dropped, which is consistent with a zone based on measured player height. Called-strike rates at the edges of the plate also fell, starting in 2025 for the sides. The model includes an ABS-era flag and centers scores within each season so pitchers are compared with their own season's league average.

## Limitations

- **Validation is in progress.** Year-to-year stability and predictive value of the scores have not been tested yet.
- Only one ABS season is in the data, and the called zone may keep shifting. The model should be retrained each offseason.
- Scores for 2022-2025 are in-sample (the model trained on those pitches). This is fine for describing past seasons but not for out-of-sample claims.
- Location only: no velocity, movement, or sequencing.
- Small samples are noisy. Use a minimum pitch count (for example 1,000-1,500 per season) when ranking, and treat single-game scores with caution.
- Not compared against FanGraphs' Location+ or similar public metrics.

## Data

Raw data is not included. It comes from Baseball Savant; check MLB's terms of use before redistributing pitch-level data.

## Credits

Data: Baseball Savant / Statcast via pybaseball. This project is independent and is not affiliated with MLB, FanGraphs, or Baseball Savant.
