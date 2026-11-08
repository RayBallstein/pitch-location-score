"""Step 4: Print a season leaderboard. Change YEAR and MIN_PITCHES below."""
import pandas as pd

YEAR = 2026
MIN_PITCHES = 1500

season = pd.read_csv("scores_by_season.csv")
s = season[(season["game_year"] == YEAR) & (season["pitches"] >= MIN_PITCHES)]
s = s.sort_values("runs_per_100", ascending=False)

cols = ["name", "pitches", "runs_per_100"]
print(f"Pitchers in {YEAR} with {MIN_PITCHES:,}+ pitches: {len(s)}")
print("\nTOP 10 (best locations)")
print(s[cols].head(10).to_string(index=False))
print("\nBOTTOM 10 (worst locations)")
print(s[cols].tail(10).iloc[::-1].to_string(index=False))
