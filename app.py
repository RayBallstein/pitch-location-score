"""Pitch Location Score: a Shiny for Python app.

Reads app_data.csv (one row per pitcher, game, and pitch type) and lets the user
pick a date range, search pitchers, and filter pitch types.

runs_per_100 = (sum of loc_score) / (pitches) * 100, so any date window is an
exact average of the underlying pitch scores.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from shiny import App, reactive, render, req, ui

# ---------- Load data ----------
data = pd.read_csv("app_data.csv", parse_dates=["game_date"])
data["name"] = data["name"].fillna(data["pitcher"].astype(str))
data["year"] = data["game_date"].dt.year

MIN_DATE = data["game_date"].min().date()
MAX_DATE = data["game_date"].max().date()
START_DATE = max(MIN_DATE, pd.Timestamp(MAX_DATE.year, 1, 1).date())  # default: latest season

PITCH_TYPES = data["pitch_type"].value_counts().index.tolist()

names = data.drop_duplicates("pitcher").set_index("pitcher")["name"]
PITCHER_CHOICES = {str(pid): nm for pid, nm in sorted(names.items(), key=lambda kv: kv[1])}
DEFAULT_PITCHER = str(data.groupby("pitcher")["pitches"].sum().idxmax())

ABOUT = """
### What this is
A model grades the **location** of every MLB pitch (2022-2026) by how many runs a pitch
at that spot is typically worth to the pitcher. Location is measured relative to each
batter's own strike zone, and the model accounts for pitch type, count, batter
handedness, and the 2026 ABS era.

**runs_per_100** is a pitcher's average location advantage over the league average for
the *same pitch types in the same season*, per 100 pitches. **0 = league average.**
+0.5 means about half a run better than average per 100 pitches.

### How to read it
- It measures **where the pitch went**, not what happened to it.
- It ignores velocity and movement, so a pitcher can score poorly and still be effective.
- **Short date ranges are noisy.** Always check the pitch count. Use the minimum-pitches filter.

### Limitations
- The scores have **not yet been validated** for year-to-year stability or predictive value.
- Only one ABS season is in the data, and the called strike zone may keep shifting.
- Scores for 2022-2025 come from a model trained on those seasons (fine for description).
- Independent project; not affiliated with MLB, FanGraphs, or Baseball Savant.
"""

# ---------- UI ----------
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_date_range(
            "dates", "Date range",
            start=START_DATE, end=MAX_DATE, min=MIN_DATE, max=MAX_DATE,
        ),
        ui.input_numeric("min_pitches", "Minimum pitches", value=300, min=1),
        ui.input_text("search", "Search pitcher name", placeholder="e.g. Nola"),
        ui.input_checkbox_group(
            "ptypes", "Pitch types", choices=PITCH_TYPES, selected=PITCH_TYPES
        ),
        width=300,
    ),
    ui.navset_tab(
        ui.nav_panel(
            "Leaderboard",
            ui.output_text("summary"),
            ui.output_data_frame("board"),
        ),
        ui.nav_panel(
            "Pitcher",
            ui.input_selectize(
                "pitcher", "Pitcher", choices=PITCHER_CHOICES, selected=DEFAULT_PITCHER
            ),
            ui.h5("By season (uses the pitch-type filter, ignores the date range)"),
            ui.output_plot("trend"),
            ui.h5("By pitch type (uses the date range)"),
            ui.output_data_frame("by_pitch"),
        ),
        ui.nav_panel("About", ui.markdown(ABOUT)),
    ),
    title="Pitch Location Score",
)


# ---------- Server ----------
def server(input, output, session):
    @reactive.calc
    def window():
        start, end = input.dates()
        df = data[
            (data["game_date"] >= pd.Timestamp(start))
            & (data["game_date"] <= pd.Timestamp(end))
        ]
        return df[df["pitch_type"].isin(list(input.ptypes()))]

    @reactive.calc
    def leaderboard():
        df = window()
        g = df.groupby(["pitcher", "name"], as_index=False).agg(
            games=("game_pk", "nunique"),
            pitches=("pitches", "sum"),
            score_sum=("score_sum", "sum"),
        )
        g["runs_per_100"] = g["score_sum"] / g["pitches"] * 100
        g = g[g["pitches"] >= input.min_pitches()]
        q = (input.search() or "").strip().lower()
        if q:
            g = g[g["name"].str.lower().str.contains(q, regex=False)]
        g = g.sort_values("runs_per_100", ascending=False).reset_index(drop=True)
        g.insert(0, "rank", g.index + 1)
        g["runs_per_100"] = g["runs_per_100"].round(2)
        return g[["rank", "name", "games", "pitches", "runs_per_100"]]

    @render.text
    def summary():
        df = window()
        n = len(leaderboard())
        return (
            f"{n:,} pitchers shown from {int(df['pitches'].sum()):,} pitches in this window. "
            "Short windows are noisy, so check each pitcher's pitch count."
        )

    @render.data_frame
    def board():
        return render.DataGrid(leaderboard(), width="100%")

    @render.plot
    def trend():
        req(input.pitcher())
        pid = int(input.pitcher())
        df = data[
            (data["pitcher"] == pid) & data["pitch_type"].isin(list(input.ptypes()))
        ]
        g = df.groupby("year").agg(pitches=("pitches", "sum"), s=("score_sum", "sum"))
        fig, ax = plt.subplots(figsize=(7, 3.5))
        if g.empty:
            ax.text(0.5, 0.5, "No pitches for this selection", ha="center")
            ax.axis("off")
            return fig
        g["rp100"] = g["s"] / g["pitches"] * 100
        labels = [f"{y}\n{int(n):,} pitches" for y, n in zip(g.index, g["pitches"])]
        ax.bar(labels, g["rp100"])
        ax.axhline(0, color="gray", linewidth=1)
        ax.set_ylabel("runs per 100 pitches")
        ax.set_title(f"{names[pid]}: 0 = league average")
        fig.tight_layout()
        return fig

    @render.data_frame
    def by_pitch():
        req(input.pitcher())
        pid = int(input.pitcher())
        df = window()
        df = df[df["pitcher"] == pid]
        g = df.groupby("pitch_type", as_index=False).agg(
            pitches=("pitches", "sum"), score_sum=("score_sum", "sum")
        )
        g["runs_per_100"] = (g["score_sum"] / g["pitches"] * 100).round(2)
        g = g.sort_values("pitches", ascending=False)[["pitch_type", "pitches", "runs_per_100"]]
        return render.DataGrid(g, width="100%")


app = App(app_ui, server)
