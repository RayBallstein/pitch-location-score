"""Step 1: Download regular-season Statcast pitch data, one month at a time.

Saves one CSV per month in ./statcast_monthly/. Months already downloaded are
skipped, so it is safe to re-run. Offseason months are saved as empty files.
"""
import os
import time

import pandas as pd
from pybaseball import cache, statcast

FIRST_YEAR, LAST_YEAR = 2022, 2026

cache.enable()
os.makedirs("statcast_monthly", exist_ok=True)
failed_months = []

for year in range(FIRST_YEAR, LAST_YEAR + 1):
    for month in range(1, 13):
        start_date = pd.Timestamp(year=year, month=month, day=1)
        end_date = start_date + pd.offsets.MonthEnd(1)
        start = start_date.strftime("%Y-%m-%d")
        end = end_date.strftime("%Y-%m-%d")
        filename = f"statcast_monthly/{year}_{month:02d}.csv"

        if os.path.exists(filename):
            print(f"Already downloaded: {year}-{month:02d}")
            continue

        print(f"\nDownloading {year}-{month:02d}...")
        success = False

        for attempt in range(1, 4):
            try:
                data = statcast(start_dt=start, end_dt=end, verbose=False, parallel=False)
                if not data.empty:
                    data = data[data["game_type"] == "R"].copy()  # regular season only
                data.to_csv(filename, index=False)
                print(f"Success: {len(data):,} pitches")
                success = True
                break
            except Exception as e:
                print(f"Attempt {attempt} failed: {e}")
                time.sleep(10 * attempt)

        if not success:
            failed_months.append(f"{year}-{month:02d}")
            print(f"FAILED: {year}-{month:02d}")

print("\nDOWNLOAD FINISHED")
if failed_months:
    print("Months to retry:", failed_months)
else:
    print("All months downloaded successfully.")
