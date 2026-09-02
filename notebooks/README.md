🌐 **English** · [Español](README.es.md)

# Notebooks

Curated, visual narrations of four of the project's investigations. These are companions to the actual
source of truth — the tested scripts in [`scripts/experiments/`](../scripts/experiments/),
[`src/backtesting.py`](../src/backtesting.py), and [`src/shot_chart_projection.py`](../src/shot_chart_projection.py)
— not a replacement for them. Each notebook re-runs real analysis against `data/processed/` (already
committed to this repo) and renders the same numbers the project's own README and `CLAUDE.md` document.

- [`01_lineup_synergy_investigation.ipynb`](01_lineup_synergy_investigation.ipynb) — does a hand-tuned
  "lineup synergy" bonus predict real 2-man lineup net rating? (No — 5 candidate effects tested, none survive
  leave-one-season-out validation.)
- [`02_contract_year_effect.ipynb`](02_contract_year_effect.ipynb) — do players outperform in the final year
  of their contract? (No measurable effect across 126 real contracts, with a player-fixed-effects regression
  and an age control.)
- [`03_backtest_calibration_story.ipynb`](03_backtest_calibration_story.ipynb) — how a 4-case "superteam
  friction" finding turned out to be mostly a calibration artifact once backtesting scaled to 480 real
  team-seasons. The project's most consequential methodological lesson.
- [`04_shot_chart_projection_walkthrough.ipynb`](04_shot_chart_projection_walkthrough.ipynb) — how the
  player popup's projected shot chart is actually built: the NBA's own court zones (`SHOT_ZONE_BASIC`,
  never redrawn by this project), the recency-weighted resampling across seasons, and the jitter fix that
  took duplicate-coordinate rates from up to 39.5% down to 0%.

To re-run one:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_lineup_synergy_investigation.ipynb
```

`02_contract_year_effect.ipynb` needs `data/raw/contract_data/` (two Kaggle CSVs, not redistributed in this
repo — see the README's "contract year" section for where to get them) to re-run from scratch; the other
three only need `data/processed/` (and, for `04`, the cached `data/raw/shot_charts/`), all already checked in.
