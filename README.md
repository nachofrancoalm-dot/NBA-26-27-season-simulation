🌐 **English** · [Español](README.es.md)

# NBA Superteam Simulator

Monte Carlo simulation of the expected performance of an NBA team that
stacks multiple high-usage players, validated by backtesting against
comparable historical cases (2010-11 Heat, 2016-17 Warriors, 2020-21
Nets, 2022-23 Suns).

**Designed to be 100% reproducible with any team**: everything
roster-specific lives in `config/team_config.yaml`, never in the code.

[![Tests](https://github.com/nachofrancoalm-dot/NBA-26-27-season-simulation/actions/workflows/tests.yml/badge.svg)](https://github.com/nachofrancoalm-dot/NBA-26-27-season-simulation/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-465%20passing-brightgreen)

🔗 **Live demo:** [nba-superteam-sim.onrender.com](https://nba-superteam-sim.onrender.com) (free tier — first load can take ~30-60s if it's been idle) · 📄 [Full architecture walkthrough](ARQUITECTURA.md)

## What this is

A Monte Carlo simulation engine that projects how an NBA roster —
including hypothetical ones you build yourself — would perform over a
full season and playoffs: individual player projections (aging curves,
injury risk, fatigue), lineup synergy, a full 30-team regular season +
playoffs simulation, and MVP/DPOY/All-NBA style awards, all served
through a self-built FastAPI + vanilla JS web app.

What makes this more than a stats toy is the validation discipline
behind it: every modeling assumption gets checked against real
historical data before it's trusted, and **assumptions that don't hold
up get documented and thrown out, not quietly kept**. A few concrete
examples:

- **Calibrated against 480 real season-long cases** (30 teams × 16
  seasons, via `nba_api`), not eyeballed. The projected win total
  correlates **0.75** with actual wins, MAE **6.78 wins/season** — down
  from an initial MAE of 13.2 wins before two calibration bugs were
  found and fixed (see the "Backtesting" section below).
- **Investigated whether a "lineup synergy" bonus (usage clash,
  playmaking + spacing, and three more candidates: post scorer +
  creator, on/off-ball shooters, drive + interior presence) actually
  predicts real 2-man lineup net rating.** Tested all 5 against
  `leaguedashlineups`/tracking data with leave-one-season-out
  validation. Result: none of them hold up (R² ≈ 0.02, wrong sign in
  most folds) — reported as a negative result instead of forcing it in.
- **Investigated the "contract year" effect** (do players statistically
  outperform in the final year of their contract?) using real
  salary/contract data. Paired comparison + player-fixed-effects
  regression on 126 contracts found no measurable effect
  (coefficient ≈ 0, p = 0.89) — not incorporated into the model.
- **Found and fixed a real bug via user-reported inconsistency**: a
  hypothetical-roster simulation reported 26 average wins through one
  code path and 42 through another, for the *same* roster — traced to
  an inconsistent synergy baseline between the two simulation modes.

Every one of these investigations lives in the repo as a runnable,
tested script (`scripts/experiments/`) plus a written account of the
result — including the ones that came back negative. Three of them also
have a curated, visual notebook: [`notebooks/`](notebooks/).

**Screenshots** (web app, `webapp/`):

| Splash | Roster & projections |
|---|---|
| ![Splash](docs/screenshots/01_splash.png) | ![Roster](docs/screenshots/02_roster.png) |

| Monte Carlo distribution | League standings & playoff odds |
|---|---|
| ![Monte Carlo](docs/screenshots/03_simulacion.png) | ![League](docs/screenshots/04_liga.png) |

| Individual awards | Real shot chart (player detail popup) |
|---|---|
| ![Awards](docs/screenshots/05_premios.png) | ![Shot chart](docs/screenshots/06_shot_chart.png) |

**All-NBA / All-Defensive teams on a real half court** — each 5-man
lineup drawn with real player photos in a classic 2-2-1 formation
(`webapp/static/js/court.js::courtLineup`), no table at all: click a
player (or hover for a quick stat line) to open their full profile.
MVP/DPOY/ROY/MIP/6th Man dropped their tables too, in favor of a
ranked leaderboard (`leaderboard.js`) — see the awards screenshot above.

![All-NBA lineups on the court](docs/screenshots/07_all_nba_courts.png)

**Pipeline:** every stage below reuses the one before it — the full
30-team league simulation and backtesting call the *same*
projection/risk/synergy functions as the single-team engine, never a
re-implementation. Full diagram (all intermediate files): [`ARQUITECTURA.md`](ARQUITECTURA.md).

![Architecture](docs/screenshots/architecture.png)

**Stack:** Python (pandas, numpy, scipy, statsmodels/scikit-learn) for
the modeling · `nba_api` for real data · FastAPI + vanilla JS for the
web app · pytest (465 tests) + GitHub
Actions CI.

---

> **First time in the project?** This README explains *what* each
> piece is and *why* it's designed that way. For the full end-to-end
> flow — diagram, exact command order, and how a game gets simulated
> step by step with the actual functions — see
> [`ARQUITECTURA.md`](ARQUITECTURA.md).

## Current status

- [x] Project structure
- [x] Config-driven design (`team_config.yaml`), full roster (starters + bench)
- [x] Automatic `player_id` resolution without network calls (`scripts/resolve_player_ids.py`)
- [x] Data ingestion pipeline (`src/data_pipeline.py`) via `nba_api`, offline-first with local cache (regular season + playoffs)
- [x] Season context layer — the 6 roadmap submodules implemented and validated against real data (see detail below)
- [x] Individual aging curve / projection model (`src/aging_curve.py`) — see detail below
- [x] Lineup synergy model (`src/lineup_synergy.py`) — see detail below
- [x] Monte Carlo simulation engine (`src/simulation.py`) — see detail below
- [x] Backtesting against historical comparables (`src/backtesting.py`) — see detail below
- [x] Web frontend (`webapp/`, FastAPI + HTML/CSS/JS) — the project's only interface, see detail below
- [x] Full league simulation (30 real teams) + playoffs (`src/league_simulation.py`) — see detail below

## Roadmap: season context layer (`src/context/`)

A model that only looks at career averages is statistically naive.
These submodules are added progressively, each one independent and
separately testable:

1. **`schedule_strength.py`** ✅ — `difficulty_score` (0-1) per game on
   the schedule (not per player, unlike the other submodules): opponent
   strength, back-to-backs, and travel. See detail below.
2. **`performance_curve.py`** ✅ — estimated Net Rating in rolling
   windows over the historical comparables (regular season +
   playoffs), to detect slow integration starts and playoff form
   peaks. See detail below.
3. **`injury_model.py`** ✅ — per-player injury `risk_score` (0-1),
   combining missed-game load history, a recency component (a recent
   absence weighs more than an old one, configurable exponential
   decay), and an age-based risk curve. See detail below.
4. **`fatigue_accumulation.py`** ✅ — `fatigue_score` (0-1) for
   accumulated minutes wear: a long career, heavy recent usage, and
   streaks of consecutive seasons without rest. Combines regular
   season and playoffs (especially relevant for LeBron, 41 years old).
   See detail below.
5. **`opponent_weighting.py`** ✅ — weighs games against
   contenders/direct rivals more than games against rebuilding teams,
   for backtesting. See detail below.
6. **`conference_adjustment.py`** ✅ — normalizes relative East/West
   strength per season before comparing records between historical
   comparables.

### `injury_model.py` — detail

Computes, for each player on the roster, a `risk_score` between 0 and 1
from `data/processed/roster_career_stats.csv` (generated by
`data_pipeline.py`). It doesn't use real injury data because `nba_api`
doesn't expose it (`CommonPlayerInfo` only carries biographical data) —
instead it uses historical availability as a proxy: games played (GP)
vs. scheduled games that season (with known exceptions: 66 games in the
2011-12 lockout, ~72 in the 2019-20/2020-21 COVID seasons).

The score combines three components, with weights configurable in
`config/team_config.yaml` (`injury_model` block), never hardcoded:

| Component | Default weight | What it measures |
|---|---|---|
| `historical_load_score` | 0.45 | Average % of games missed over the last N seasons |
| `recency_score` | 0.35 | The same, but weighted with exponential decay: a recent absence weighs more than an old one |
| `age_score` | 0.20 | Age-based risk curve — rises and flattens, doesn't grow unbounded |

The age weight is deliberately low relative to history (0.80 between
the other two components): epidemiological evidence ([Ruddy et al.,
PMC6176657](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6176657/))
points to recent injury history as the single strongest predictor of
a future injury, more than age in the abstract. Also, the age curve
flattens (doesn't decrease, but also stops climbing) past
`peak_start_age` (32 by default), reflecting the survivorship bias
documented in very long careers ([Mack et al.,
PMC11569584](https://pmc.ncbi.nlm.nih.gov/articles/PMC11569584/)): a
41-year-old player with a clean recent history (LeBron James's case)
doesn't get an inflated `risk_score` just for his age.

```bash
python -c "from src.context.injury_model import build_injury_risk_dataset; \
from src.config_loader import load_config; \
print(build_injury_risk_dataset(load_config()))"
```

Generates `data/processed/injury_risk.csv`. Requires
`roster_career_stats.csv` to already exist (run `data_pipeline.py`
first). Tests in `tests/test_injury_model.py` — no network required,
use synthetic DataFrames matching `roster_career_stats.csv`'s schema.

> **Data-correction note:** when a player is traded mid-season,
> `nba_api` includes a `TOT` (total) row in addition to one row per
> team, all with the same season. `injury_model.py` (and
> `fatigue_accumulation.py`) dedupe this by prioritizing the `TOT` row
> — without that step, a season with a trade would count two or more
> times within the N-season window.

### `fatigue_accumulation.py` — detail

Computes, for each player, a `fatigue_score` between 0 and 1 from
`roster_career_stats.csv` (regular season) and
`roster_playoff_career_stats.csv` (playoffs) — both generated by
`data_pipeline.py`. Unlike `injury_model.py` (which measures risk from
games *missed*), this measures accumulated wear from games *played*
under heavy minutes load.

| Component | Default weight | What it measures |
|---|---|---|
| `cumulative_load_score` | 0.35 | Total career minutes (regular + playoffs) against a configurable "long career" ceiling (35,000 min by default) |
| `recent_intensity_score` | 0.35 | Minutes/game over the last N seasons against a "heavy usage" threshold (34 min/game by default), weighted by recency |
| `sustained_streak_score` | 0.30 | Number of consecutive recent seasons without a load drop (no rest/reduced season), with diminishing returns |

Deliberately **has no explicit age component**: wear from a long
career already emerges naturally in `cumulative_load_score` (more years
in the league = more accumulated minutes), so adding age separately
would duplicate that signal — unlike `injury_model.py`, where injury
history and age are genuinely distinct signals.

Unlike `injury_model.py`, there's no published literature here that
justifies a clear hierarchy between the three components, so the
default weights are similar to each other (all configurable in
`config/team_config.yaml`, `fatigue_model` block).

```bash
python -c "from src.context.fatigue_accumulation import build_fatigue_dataset; \
from src.config_loader import load_config; \
print(build_fatigue_dataset(load_config()))"
```

Generates `data/processed/fatigue_risk.csv`. Requires
`roster_career_stats.csv` (and optionally
`roster_playoff_career_stats.csv` — if it doesn't exist, fatigue_score
is computed from regular season only). Tests in
`tests/test_fatigue_accumulation.py` — no network required.

**Result on the real roster (2026-27):** LeBron James tops the
`fatigue_score` (0.89, highest on the team) due to his extremely high
`cumulative_load_score` — but in `injury_model.py` his `risk_score` is
only moderate (0.33, tied with Kentavious Caldwell-Pope). This is
exactly the intended behavior: two submodules measuring different
things can give opposite readings for the same player — lots of
accumulated wear, but low recent injury risk.

### `schedule_strength.py` — detail

Unlike `injury_model.py` and `fatigue_accumulation.py`, this submodule
is **not per player** — it computes a `difficulty_score` (0-1) **per
game** on the team's schedule, from `team_schedule.csv` and
`prior_season_standings.csv` (both generated by `data_pipeline.py`).

| Component | Default weight | What it measures |
|---|---|---|
| `opponent_strength_score` | 0.40 | Opponent's WinPCT in the **prior** season (already on a 0-1 scale) |
| `back_to_back_score` | 0.30 | 1.0 if the team's previous game was the day before, else 0.0 |
| `travel_score` | 0.30 | Distance (haversine) from the previous game's city, normalized against a "long travel" ceiling (3000 km by default) |

**Two important data limitations, also documented in the module's
docstring:**

1. **Opponent strength via prior season.** `team_config.yaml` can point
   to a season that hasn't been played yet (2026-27) — there are no
   real results from that season to measure each opponent's strength.
   Prior season WinPCT is used as a proxy, the same way any real
   schedule preview does.
2. **Travel via a static coordinate table.** `nba_api` doesn't expose
   distances between cities. `ARENA_COORDS` in the module has
   approximate coordinates for the 30 NBA franchise cities (a league
   geographic fact, not something specific to this team) and computes
   geodesic (haversine) distance between consecutive games. Neutral-site
   games outside those 30 cities (Mexico City, London, Paris) have no
   coordinate — that travel leg is treated as 0 km and a warning is
   printed to console, instead of failing.

```bash
python -c "from src.context.schedule_strength import build_schedule_difficulty_dataset; \
from src.config_loader import load_config; \
print(build_schedule_difficulty_dataset(load_config()))"
```

Generates `data/processed/schedule_difficulty.csv`. Requires
`team_schedule.csv` and `prior_season_standings.csv` (run
`data_pipeline.py` first). Tests in `tests/test_schedule_strength.py`
— no network required.

> **Note on real data availability:** when testing this against the
> real API in August 2026, `ScheduleLeagueV2`'s 2026-27 schedule only
> returned 2 games (both preseason) — the NBA hadn't published the full
> regular-season schedule yet. This is a real, expected state, not a
> bug: as soon as the league publishes the full schedule,
> `python src/data_pipeline.py --refresh` will pick it up.

### `performance_curve.py` — detail

Operates on the `historical_comparables` (2010-11 Heat, 2016-17
Warriors, 2020-21 Nets, 2022-23 Suns) — not on the simulated roster. It
estimates per-game Net Rating and computes a configurable rolling
average to detect whether a "superteam" starts slow (takes time to gel)
and improves in the playoffs, which is the central narrative the
project seeks to validate via backtesting.

**Data approximation:** NBA.com doesn't expose official per-game Net
Rating without additional calls. It's estimated as:

```
net_rating_estimate = PLUS_MINUS / estimated_possessions * 100
estimated_possessions ≈ FGA - OREB + TOV + 0.44 * FTA   (standard basketball analytics formula)
```

`PLUS_MINUS` comes from `TeamGameLogs` (the **plural** endpoint, which
has `PLUS_MINUS`) — different from `TeamGameLog` (singular, no
`PLUS_MINUS`, the one `historical_comparables_game_logs.csv` already
used). That's why `data_pipeline.py` generates a new, separate CSV,
`historical_comparables_advanced_game_logs.csv`, instead of touching
the existing one.

```bash
python -c "from src.context.performance_curve import build_performance_curve_dataset; \
from src.config_loader import load_config; \
print(build_performance_curve_dataset(load_config())['summary'])"
```

Generates `data/processed/performance_curve_by_game.csv` (full series)
and `data/processed/performance_curve_summary.csv` (a per-case summary:
`early_season_net_rating`, `playoff_boost`, `trend_slope`, etc.). Tests
in `tests/test_performance_curve.py` — no network required.

**Result on the 4 real cases:** the 2022-23 Suns show
`playoff_boost = -4.36` (they collapsed in the playoffs — swept in the
second round in real life) while the 2016-17 Warriors show
`playoff_boost = +1.62` (they raised their level in the playoffs — that
year's champions). The model, with real data, directionally reproduces
known narratives from those two historical cases.

### `opponent_weighting.py` — detail

Complements `performance_curve.py`: weighs each game of a historical
comparable by its opponent's REAL strength **that same season** (not a
prior-season proxy — unlike `schedule_strength.py`, these 4 seasons
have already been played in full, so real contemporary strength can be
pulled via `historical_comparables_standings.csv`).

- The opponent is resolved from the `MATCHUP` column ("MIA @ TOR") with
  a static table of the 30 NBA franchises, plus aliases for historical
  abbreviations that changed within the project's season range (Nets
  "NJN" before moving to Brooklyn in 2012-13; Hornets/Pelicans "NOH"
  before the 2013-14 rename).
- The numeric weight is **continuous** (`win_pct ** steepness`), not a
  fixed threshold — an opponent at 0.54 WinPCT isn't qualitatively
  different from one at 0.56, so any binary cutoff would be arbitrary.
  It does also offer a descriptive categorical view
  (contender/mid/rebuilding, with configurable thresholds) because the
  roadmap asks for it as a readable summary.
- Reuses `compute_net_rating_estimate()` from `performance_curve.py` via
  direct import, instead of duplicating the formula.

```bash
python -c "from src.context.opponent_weighting import build_opponent_weighting_dataset; \
from src.config_loader import load_config; \
print(build_opponent_weighting_dataset(load_config()))"
```

Generates `data/processed/opponent_weighting_summary.csv`. Requires
`historical_comparables_advanced_game_logs.csv` and
`historical_comparables_standings.csv` (run `data_pipeline.py` first).
Tests in `tests/test_opponent_weighting.py` — no network required.

**Result on the 4 real cases:** the 2022-23 Suns have
`contender_net_rating = -5.35` — **they lost on average against strong
opponents** — while steamrolling rebuilding teams
(`reconstruccion_net_rating = +6.62`). Their overall Net Rating (1.55)
was inflated by easy games; against real competition, the team was net
negative. This matches their real playoff collapse. The 2010-11 Heat
show a similar pattern to a lesser degree (1.47 vs. contenders, 13.17
vs. weak teams) — consistent with the real narrative of a "superteam"
that took time to gel against opponents at its own level.

### `conference_adjustment.py` — detail

Closes out the `src/context/` roadmap. Normalizes each historical
comparable's WinPCT and Net Rating by the relative East/West strength
that season, to be able to compare cases that played in different
conferences and seasons against each other.

**Statistical basis (no additional data needed):** both WinPCT and
point differential (`DiffPointsPG`) are zero-sum at the full LEAGUE
level (30 teams) by construction — but NOT separately within each
conference, because inter-conference games aren't zero-sum within a
single group. If the West beats the East more than it loses to it that
season, the West's average WinPCT/DiffPointsPG rises above the
baseline (0.5 / 0) and the East's drops by the same magnitude. That
deviation from the conference average **is** the relative strength
index for that season.

```
conference_index = mean(metric across all teams in that conference that season) - baseline
adjusted_value    = team's raw value - conference_index
```

Playing in a conference with a positive index (tougher) subtracts less
(or adds) to the adjusted value; playing in a weak conference subtracts
more — it gives credit for context.

```bash
python -c "from src.context.conference_adjustment import build_conference_adjustment_dataset; \
from src.config_loader import load_config; \
print(build_conference_adjustment_dataset(load_config()))"
```

Generates `data/processed/conference_adjustment_summary.csv`. Unlike
`opponent_weighting.py` (which recomputes its own per-game metric to
stay self-contained), this module **does** depend on
`performance_curve.py`'s already-aggregated summary — comparing
summaries across cases is exactly its purpose, there's no point
recomputing the full game series again. Requires
`historical_comparables_standings.csv` and
`performance_curve_summary.csv` (run `data_pipeline.py` and
`performance_curve.py` first). Tests in
`tests/test_conference_adjustment.py` — no network required.

**Result on the 4 real cases:** the East had a negative index in
2010-11 (-0.75) and 2020-21 (-0.35) — the West dominated the
inter-conference matchup those years, consistent with the real
narrative of those seasons. That's why the 2010-11 Heat's Net Rating
gets adjusted UPWARD (8.05 → 8.80) and so does the 2020-21 Nets'
(4.43 → 4.78) — credit for competing in a tougher context. The 2016-17
Warriors, on the other hand, played in a dominant West (+0.77) and
their Net Rating adjusts slightly downward (11.39 → 10.62).

## Individual aging curve / projection model (`src/aging_curve.py`)

Outside `src/context/` — not part of the season context roadmap, it's
the other pending piece from "Current status". Projects each roster
player's per-36-minute production for the config's season, combining:

1. **Baseline** — recency-weighted average (exponential decay, same
   pattern as `injury_model.py`) of the player's own last N seasons
   per-36.
2. **Age adjustment** — a multiplicative factor from TWO distinct
   curves, not one:
   - **General curve** (PTS, AST, REB, STL, BLK, TOV) — peaks ~26-27
     years old.
   - **Shooting curve** (three-point volume: FG3M, FG3A) — peaks ~30
     years old, because outside shooting ages better than the rest of
     the game.

**Why two curves, and where the breakpoints come from:** this project
doesn't have enough of its own data (only 10 players) to fit a
population age curve with statistical rigor — that would require full
careers of hundreds of historical players, beyond the current
`data_pipeline.py`'s scope. Instead, the LOCATION of the breakpoints
comes from public research: average performance rises fastest between
ages 19-20 and 23-25, drops fastest between 29-31 and 36-38, with an
overall peak ~26-27; 2-point/free-throw shooting peaks ~25 followed by
a marked decline, while three-point shooting peaks later, ~30
([Large data and Bayesian modeling — aging curves of NBA players,
PubMed 30684225](https://pubmed.ncbi.nlm.nih.gov/30684225/)). The exact
MAGNITUDES of the yearly change within each stretch are this project's
own estimate calibrated to fit that shape — they don't come literally
from a paper — and are exposed as config (`config["aging_curve"]`),
not hardcoded, precisely because they're an estimate.

**Reuses existing config instead of inventing new fields:** scales the
per-36 projection to season totals using the `minutes_projection` each
player already has in `team_config.yaml`'s `roster`, and
`simulation.games_per_season`.

```bash
python -c "from src.aging_curve import build_aging_projection_dataset; \
from src.config_loader import load_config; \
print(build_aging_projection_dataset(load_config()))"
```

Generates `data/processed/aging_curve_projection.csv`. Requires
`roster_career_stats.csv` (run `data_pipeline.py` first). Tests in
`tests/test_aging_curve.py` — no network required.

**Result on the real roster:** the model doesn't reset a veteran player
to a "generic old player" level — LeBron James (41→42) still projects
high production because his BASELINE (his own recent seasons) is
already high; the age adjustment only applies a marginal one-year
discount, not a population ceiling. This is intentional: the
player-specific baseline dominates, the age curve is the adjustment,
not the starting point — same design principle as "history rules over
age" in `injury_model.py`.

## Lineup synergy model (`src/lineup_synergy.py`)

Adjusts the team Game Score used by `simulation.py` based on how well
the players sharing the floor statistically fit together, instead of
summing their contributions as if they were independent.

**Data limitation:** this roster has never shared the floor together —
it's hypothetical for 2026-27. `nba_api` does have a real-lineup stats
endpoint (`leaguedashlineups`), but it doesn't help here: there are no
shared minutes to query for players who never overlapped. So the
module does NOT measure empirical synergy — it DERIVES it from the
statistical profiles `aging_curve.py` projects (usage, playmaking,
spacing, interior presence).

**Also deliberately doesn't use `role_expected`** from
`team_config.yaml`: the file's original intent (a comment that
`resolve_player_ids.py --fill-config` itself ends up erasing when it
rewrites the YAML) is that this field is descriptive, not a
calculation input. The "roles" this module uses come from real
`nba_api` stats, not a hand-written text label.

**Two effects, both grounded in public usage-rate analytics:**

| Effect | What it measures | Basis |
|---|---|---|
| `usage_clash` | Penalty when 2+ high-usage players share the floor | A player's efficiency drops as their own usage rises, and concentrating usage in few stars benefits role players — the opposite (several high-usage stars at once) creates friction ("there's only one ball") |
| `playmaking_spacing_synergy` | Bonus when a playmaker shares the floor with a shooter | Widely accepted basketball wisdom — shooters' "gravity" effect opening driving lanes |

Both are weighted by `pair_weight = min(minutes_i, minutes_j) / 48` —
an approximation of how much they could share the floor, since there's
no real rotation data to know whether two specific players play at the
same time or in shifts.

`compute_game_synergy_adjustment()` recomputes synergy
**dynamically per game** based on who's available that day (an injured
player neither generates nor suffers clash/synergy that night), via a
vectorized quadratic form (`einsum`) — no per-game loops.

```bash
python -c "from src.lineup_synergy import build_lineup_synergy_dataset; \
from src.config_loader import load_config; \
print(build_lineup_synergy_dataset(load_config()))"
```

Generates `data/processed/lineup_synergy_pairs.csv` (one row per pair
of players, sorted from most synergistic to most conflicting). Tests
in `tests/test_lineup_synergy.py` — no network required.

**Result on the real roster:** the most conflicting pair is Jaylen
Brown + Joel Embiid (`net_pair_score = -1.69`, two high-usage profiles
competing for the ball), while Tyrese Maxey + VJ Edgecombe leads
positive synergy (`+0.75`, complementary playmaking + spacing). The
total sum with the healthy roster is around **+6.5** in the pair table
— which, applied as a quadratic form in `simulation.py` (each pair
counted twice by symmetry), raises the simulation's mean estimated Net
Rating from ~1.7 to ~8.9: the roster has real friction between its
stars, but the rest of the roster's complementary fit more than
compensates for it.

## Monte Carlo simulation engine (`src/simulation.py`)

The final piece: consumes the 6 signals from `src/context/` plus
`aging_curve.py`'s projection and `lineup_synergy.py`'s adjustment to
simulate `simulation.n_seasons` (10,000 by default) hypothetical
82-game seasons.

**Key data limitation:** the config season's real schedule may not
exist yet (see `schedule_strength.py` — as of this writing,
`team_schedule.csv` only had 2 preseason games for 2026-27). You can't
simulate against a schedule that doesn't exist, so each game uses a
**representative synthetic schedule**: the opponent is sampled from the
real league-wide WinPCT distribution (`prior_season_standings.csv`),
and back-to-backs are drawn with a configurable probability (~18% by
default, approximating a modern NBA schedule's typical rate). Once the
NBA publishes the full schedule, the synthetic sample can be swapped
for the real schedule without changing the rest of the engine.

**Per simulated season mechanics:**
1. **Availability** — for each player, the number of missed games is
   drawn from a negative binomial with mean `risk_score * 82` (from
   `injury_model.py`), grouped into ONE contiguous stretch — real
   injuries are a streak, not random scattered games.
2. **Per-game contribution** — each player's per-36 Game Score (from
   `aging_curve.py`) scaled to their projected minutes, with a fatigue
   penalty on back-to-backs and progressive season wear (both
   proportional to `fatigue_score` from `fatigue_accumulation.py`),
   plus game-to-game noise.
3. **Outcome** — team Game Score minus an opponent-strength adjustment,
   converted to a win probability via a logistic function, and drawn as
   a win/loss.

**A calibration decision worth documenting:** team Game Score (sum of
players) is NOT a point differential by itself — without subtracting an
"average team" baseline, this engine's first version projected an
**81-1** season, a clear sign of a calibration bug, not an excellent
team. The fix uses Hollinger's own calibration (an average player is
around ~10 Game Score per 36 minutes) to derive that baseline:
`10 × 240/36 ≈ 66.7` (240 = 5 positions × 48 minutes). Only the EXCESS
of team Game Score over that baseline is read as net rating.

```bash
python -c "from src.simulation import build_simulation_dataset; \
from src.config_loader import load_config; \
print(build_simulation_dataset(load_config()).describe())"
```

Generates `data/processed/simulation_results.csv` (one row per
simulated season: `wins`, `losses`, `net_rating_estimate_mean`,
`total_games_missed`). Requires `aging_curve_projection.csv`,
`injury_risk.csv`, `fatigue_risk.csv`, and `prior_season_standings.csv`
(the four previous modules already run). Tests in
`tests/test_simulation.py` — no network required.

**Result on the real roster (with `lineup_synergy.py` already
integrated):** the mean estimated Net Rating (≈8.9) lands in the same
order of magnitude as the 4 real `historical_comparables` computed by
`performance_curve.py` (Heat 8.05, Warriors 11.39, Nets 4.43, Suns
2.07) — close to the 2016-17 Warriors' level, the most dominant
comparable — a sign the calibration isn't unreasonable. The average win
total (~50.5 of 82) is still more modest than the "superteam" label
would suggest, because the model takes the roster's real injury risk
seriously — Embiid alone, with his `injury_model.py` `risk_score`,
projects losing on average more than 50 games per simulated season. The
model doesn't inflate the result just for having stars; it penalizes
them for their real history, and rewards them (via `lineup_synergy.py`)
only when their skill mix genuinely fits.

## Full league simulation and playoffs (`src/league_simulation.py`)

`simulation.py` (above) pits the config's team against a generic
opponent WinPCT — it doesn't answer "would we beat the real Celtics?"
or "would we make the Finals?". This is a **different** engine that
projects the 30 real NBA teams (with their current rosters, not
`team_config.yaml`'s hypothetical one) and pits them against each other
directly, with a round-robin schedule and a full playoff bracket
(play-in included, real NBA format).

**Why a different engine instead of extending `simulation.py`:**
comparing one real team's Game Score against another real team's
doesn't need the "average team baseline" approximation that's needed
when the opponent is an abstract number (generic WinPCT) — the baseline
cancels out when comparing two real teams directly.

**Real cost, deliberately opt-in:** projecting 30 teams requires each
one's real roster (~450 players) and career history — ~900 new calls to
`stats.nba.com`, the project's most expensive ingestion (20-30+ min the
first time, cached afterward). That's why it's NOT part of the normal
pipeline:

```bash
# Ingestion (opt-in, see cost warning above)
python src/data_pipeline.py --league

# Project the 30 teams + simulate regular season + playoffs
python -c "from src.league_simulation import build_league_simulation_dataset; \
from src.config_loader import load_config; \
print(build_league_simulation_dataset(load_config()))"
```

**How any team gets projected (`project_team_roster`):** the same
`aging_curve`/`injury_model`/`fatigue_accumulation`/`lineup_synergy` as
always, but since the other 29 teams don't have a hand-curated
`minutes_projection` like this project's own roster, minutes are
assumed to be each player's REAL minutes/game from their most recent
season (role continuity) — a data-driven approximation, not made up.

**Schedule:** uses the REAL schedule published by the NBA
(`data_pipeline.build_league_schedule_dataset`, inside `--league`,
saved to `league_schedule_full.csv`) when it exists — real dates,
opponents, rest (real back-to-backs, not sampled), and home-court
advantage (`home_court_advantage`, calibrated at 2.41 points).
`league_simulation.real_schedule_to_games` converts the schedule into a
list of games with each team's SEQUENTIAL game index within its own
season — unlike a synthetic round-robin, where "the same day" implies
every team plays at once, a real schedule doesn't have that property
(rest varies by team), so each team needs its own index, not a shared
one. Real, documented, not-hidden temporal limitation: while the NBA
Cup knockout stage isn't fully resolved, each team has fewer games than
`games_per_season` (currently 80 of 82) — they're used as-is, the
missing ones aren't invented. If `league_schedule_full.csv` doesn't
exist yet (a season without an official published schedule), it falls
back to a balanced synthetic round-robin (classic circle-method
tournament scheduling, each team against each opponent ~2-3 times until
reaching `games_per_season`) — the same "degrade, don't fail"
criterion as the rest of the project.

**Playoffs — real format, with documented simplifications:** real
play-in (7 vs 8, loser vs winner of 9 vs 10), 1v8/4v5/3v6/2v7 bracket
but WITHOUT re-seeding between rounds, best-of-7 series, no
back-to-backs. Availability **is** still sampled in the playoffs
(per-game Bernoulli with the same regular-season `risk_score`); the
only thing not replicated is the *contiguous* injury stretch, because
in 4-7 game series the difference between a streak and a per-game draw
is small.

> **This used to be a bug, not a simplification.** The model assumed a
> fully healthy roster in the playoffs, justified as a "marginal
> benefit." It wasn't: it produced a team with a **worse** regular
> season being **more** favored for the title. Real case (2026-27): San
> Antonio won 56.4 games — best in the league — with a 10.8% title
> chance, while Philadelphia won 45.5 with **23.7%**. The reason: PHI
> loses 31% of its production to injuries (Embiid, 0.65 risk) vs. SAS's
> 17.9%, so it got penalized for all 82 games and then reached the
> playoffs miraculously healthy. Fixed: SAS 10.8% → **37.9%**, PHI
> 23.7% → **3.3%**, and the correlation between regular-season wins and
> title probability rose to **0.683**. Lesson: a simplification being
> *documented* doesn't prove its effect is marginal — it has to be
> measured.

A bug in the per-round metrics was also fixed: they were shifted by one
round (`conf_semis_pct` counted teams that *won* the semifinals) and
`conf_finals_pct` was identical to `finals_pct` across all 30 teams.
Probabilities now decrease monotonically, which is the obvious sanity
check: playoffs ≥ semis ≥ conference finals ≥ Finals ≥ title.

**Home-court advantage** (added after measuring it in real data): the
model didn't have it at all. Measured over 15 seasons: **+2.41 points
at home** in the regular season (57.4% home win rate) and **+3.98 in
the playoffs** (60.3%). Now applied in the regular season (exactly 41
home games and 41 away) and in the playoffs with the real
**2-2-1-1-1** format — the higher seed hosts games 1, 2, 5, and 7. A
non-obvious detail: from the semifinals onward the higher seed isn't
necessarily the first one listed in the matchup (if the 8 upsets the
1), so home court is decided by the real seed; and in the Finals it's
the league's best record that decides, not conference seed.

### Validation against real champions, and a measured limitation

Comparing which seed real champions come from vs. simulated ones (see
the dashboard's **Campeones reales** sub-tab):

| | Real (16 seasons) | Simulated |
|---|---|---|
| Champion from seed 1 | 56% | 47.6% |
| Champion from seed 1-3 | **100%** | 75.5% |
| Champion from **seed ≥4** | **0%** | **24.5%** |

Across 16 real seasons, **no** champion came from a seed worse than 3
(16 of 16). The simulator hands out titles to low seeds 24.5% of the
time (home-court advantage only brought it down from 27.0%). The cause
is **not** the games — measured, they're actually even more
deterministic than reality (effective scale 4.25 points vs. 7.23 real)
— it's the **seeding**:

| | Talent (between-team differences) | Noise (season-to-season variation) | Signal/noise |
|---|---|---|---|
| Real | 11.27 wins | 4.53 | **2.49** |
| Model | 6.52 | 7.70 | **0.85** |

The model compresses talent differences by half and nearly doubles the
noise, so in any simulated season the seeding comes out almost drawn at
random. The interesting part is that **the compressed talent isn't a
prediction error**: the projection regresses to the mean because that's
what minimizes error (MAE 7.75). The flaw is using that regressed
estimate *as if it were the true talent* when simulating — confusing
the predictive distribution with a "plug-in" simulation. Fixing it
properly requires separating estimation uncertainty from season noise,
an architectural change; and simply lowering the noise would narrow
the P10-P90 bands and worsen backtesting calibration. **It's documented
as measured instead of forcing the parameters to make the result look
good** — the mistake this project already made once before.

## Real champions: context and validation (`src/champion_profiles.py`)

**Descriptive** analysis of real champions, reusing backtest sweep data
(doesn't download anything new):

```bash
python -c "from src.champion_profiles import build_champion_analysis_dataset; \
from src.config_loader import load_config; \
build_champion_analysis_dataset(load_config())"
```

- **Title path** (`champion_title_paths.csv`): starting seed,
  regular-season record, and the opponents eliminated in order with
  their seed. Real example: *OKC 2024-25, seed 1, 68 wins, 8 → 4 → 6 →
  4*.
- **Roster composition** (`champion_roster_profiles.csv`): minutes
  breakdown by position, minutes-weighted experience and age, and what
  % of minutes is concentrated in the 2 stars. The 16 champions average
  ~25% of minutes in their two most-used players, ~7 years of
  experience, and ~29 years of age.
- **Seed trajectory** (`champion_seed_trajectories.csv`): each
  franchise's standing in its conference, season by season.

> **Descriptive, not predictive.** These are 16 champions: any "title
> recipe" drawn from that has a sample of 16. This project already got
> burned once drawing a strong conclusion from 4 cases (see above). The
> part that **is** statistically solid is the seed validation: **0 of
> 16** champions from seed 4+ is a real constraint to measure the
> simulator against.

Recent parity data point: across the 16 seasons there are **12
different champion franchises**, and the **last 8 seasons had 8
different champions** (TOR, LAL, MIL, GSW, DEN, BOS, OKC, NYK) — an
unprecedented parity streak.

> **Season coverage.** The sweep needs to reach the most recently
> completed full season (the one projections use as a base). It fell
> one short — up to 2024-25 when 2025-26 already existed — which left
> the most recent champion out of the analysis and calibration, even
> though its individual stats were already in. There's a test
> (`test_backtest_sweep_includes_the_most_recent_completed_season`)
> that checks this against `config["team"]["season"]`, so it doesn't
> happen again as the season advances.

Generates `data/processed/league_regular_season_summary.csv` (30
teams' average wins), `data/processed/league_playoff_summary.csv`
(% of times each team makes the playoffs / reaches each round / wins
the title) and `data/processed/league_player_projections.csv`
(individual projection for the league's ~450 players) — visible in the
dashboard's "Liga y Playoffs" tab, with a selector to browse any of the
30 teams. Tests in `tests/test_league_simulation.py` — no network
required (they use synthetic team projections).

**Real bug found running against the 30 real teams:**
`simulate_playoffs_once` passed each conference's full 15 seeds to
`resolve_play_in()` (which requires exactly 10 — in the real NBA seeds
11-15 are eliminated from the regular season). The original tests used
10-team conferences for simplicity, which hid the bug until running
against real data (15 per conference). Fixed, with a regression test
that deliberately uses 15 teams per conference.

**Second real bug, more important, found by manually inspecting the
results:** reviewing the standings, results kept popping up that made
no basketball sense — Oklahoma City (one of the league's strongest
cores) nearly last in the West, Boston and the Knicks out of the
playoffs. The cause: `project_team_roster()` assigns each player their
REAL minutes/game from their most recent season, but the team-level SUM
of those minutes was never normalized to the 240 minutes that actually
exist in a game (5 positions × 48 min). A roster with a historically
deep rotation could sum well above that — Utah reached **449 "raw"
minutes** (almost double what's possible) and so led title odds, while
OKC (262 summed minutes, closer to reality) came out penalized in the
comparison despite individually having the strongest core. Fixed by
scaling each player's minutes so the team total sums to exactly 240,
preserving relative proportions within the roster — with a regression
test (`test_project_team_roster_normalizes_total_minutes_to_240`).

**Third real bug, found by the user's manual review of individual
stats:** the previous fix (scaling the ENTIRE roster to 240) had an
unforeseen side effect — it diluted real STARS on rosters with heavy
roster churn. Luka Dončić (~35.8 real min/game on the Lakers) ended up
projected at only 26.98, because several deep-bench players (1-15
games played due to injuries/two-way call-ups, not real merit) inflated
the team's raw sum (318 minutes) and that dilution was spread evenly
across everyone, star included. **Fixed by restricting normalization to
a realistic rotation**: only the 10 players with the most "raw" minutes
(`rotation_size`, configurable) participate in the normalization to
240; the rest of the roster stays at 0 minutes — they don't dilute the
real allocation. With this, the Lakers' real rotation (top 10) already
summed to ~257.5 raw, very close to 240, and Luka lands at a realistic
~33.3 min/game. Regression test:
`test_project_team_roster_does_not_dilute_star_minutes_with_bench_churn`.

**Result on the real league, after both fixes:** Oklahoma City clearly
leads title odds (30.8%) and San Antonio leads average wins (56.0) —
both consistent with their real-world perception as top league cores.
Utah, with a young rebuilding roster, drops to a realistic 31.9 average
wins (it used to lead the title odds under the bug). Chicago, which
used to unrealistically lead the East, is now a reasonable mid-table
team (3rd, 45.8 average wins, behind Orlando and Atlanta). The config's
76ers end up with a 64.3% playoff probability and 16.8% title chance.

## Backtesting against historical comparables (`src/backtesting.py`)

The project's real validation: runs the full engine (`aging_curve` +
`injury_model` + `fatigue_accumulation` + `lineup_synergy` +
`simulation`) **retrospectively** over the 4 `historical_comparables`,
with their **real** roster and schedule — not `team_config.yaml`'s
hypothetical roster nor a sampled synthetic schedule.

**No-look-ahead rule (the module's most important one):** each
historical player's projection can only use their seasons *before* the
one being predicted — never the case's own season or later ones.
`filter_seasons_before()` is the sole entry point to a player's career
data across the entire module. The player's real age that season and
their real minutes/game ARE used as external inputs (the same way
`minutes_projection` is for the hypothetical roster in the config) —
what the model predicts is performance, risk, and wear, not the minutes
allocation.

**Methodological advantage over forward simulation:** since these
seasons already happened, the REAL schedule is built (each game's
opponent, real back-to-backs, resolved the same way as in
`opponent_weighting.py`) instead of sampling a synthetic one.

This required extending `data_pipeline.py` with a new endpoint
(`CommonTeamRoster`) and downloading real career stats for ~60 players
across the 4 historical teams — far more API call volume than any
previous module.

```bash
python -c "from src.backtesting import build_backtest_dataset; \
from src.config_loader import load_config; \
print(build_backtest_dataset(load_config()))"
```

Generates `data/processed/backtest_summary.csv` (one row per case: real
wins, simulated distribution, and which percentile of that distribution
the real result falls in). Requires the historical comparables +
rosters datasets (run `data_pipeline.py` first). Tests in
`tests/test_backtesting.py` — no network required.

### Real result, and how it changed after calibrating the model

**First reading (uncalibrated model).** With the original parameters,
the 4 cases gave this:

| Case | Real wins | Simulated median | Real percentile |
|---|---|---|---|
| Miami Heat 2010-11 | 58 | 68 | 7.7% |
| Golden State Warriors 2016-17 | 67 | 70 | 33.9% |
| Brooklyn Nets 2020-21 | 48 | 67 | **0.25%** |
| Phoenix Suns 2022-23 | 45 | 74 | **0.05%** |

3 of 4 cases extremely overestimated. The natural reading — and the one
this README defended for a while — was that the model captured the
talent ceiling but not the **human friction** of superteams (ego,
hierarchy, integration), and that this gap was the project's central
finding.

**Second reading (after systematic backtesting).** Scaling backtesting
from 4 cases to 450 (30 teams × 15 seasons) surfaced three real
calibration bugs — see the next section — that caused the model to
overestimate **every** team, not just superteams. Fixed, the same 4
cases come out like this:

| Case | Real wins | Simulated mean | Real percentile |
|---|---|---|---|
| Miami Heat 2010-11 | 58 | 47.6 | 97.9% (underestimated) |
| Golden State Warriors 2016-17 | 67 | 56.9 | 98.9% (underestimated) |
| Brooklyn Nets 2020-21 | 48 | 47.9 | **52.7%** (nearly exact) |
| Phoenix Suns 2022-23 | 45 | 50.9 | 18.8% (slight overestimation) |

The pattern flips: the two superteams that **actually performed**
(Heat 58 wins, Warriors 67) are now *underestimated*, and the two with
documented friction (Nets, Suns) land in reasonable percentiles.

**The honest conclusion:** a good chunk of that "finding" was a
**calibration artifact**, not evidence of superteam friction. The model
overestimated everyone for mechanical reasons (a fixed baseline against
a league that scores more every year, un-normalized roster minutes, and
a made-up Game Score→differential scale). A residual overestimation
bias still exists (≈3.5 wins on average across the 450 cases), and it's
still true that the box score doesn't capture chemistry or hierarchy —
but the evidence no longer supports the strong claim that superteams
systematically underperform their projection.

> This is, in itself, the project's most valuable methodological
> finding: **with 4 data points it was impossible to distinguish "the
> model has a mechanical bias" from "there's a real friction
> phenomenon"**. It took 450 cases to separate them, and the answer was
> mostly the former. A small sample can produce a convincing and wrong
> narrative.

### Large-scale systematic backtesting (30 teams × 16 seasons)

The 4 `historical_comparables` above are hand-picked narrative cases
(well-known "superteams") — useful for the locker-room friction finding
above, but 4 data points don't say whether the engine works well **in
general**, only with stacked-star teams. To answer that,
`config["backtest_sweep"]` automatically generates the 30 NBA teams for
every season from 2010-11 to 2025-26 (480 cases: contenders, tankers,
mediocre teams — the whole spectrum), via
`config_loader.resolve_backtest_sweep_cases()` (the same static
franchise table as `league_simulation.py`/`opponent_weighting.py` —
nba_api's `team_id` is stable across relocations/name changes).

```bash
# Ingestion (WARNING: the project's most expensive — on the order of
# thousands of API calls, 1.5-3 hours the first time; cached afterward)
python src/data_pipeline.py --backtest-sweep

# Simulation + calibration summary
python -c "from src.backtesting import build_backtest_sweep_dataset; \
from src.config_loader import load_config; \
build_backtest_sweep_dataset(load_config())"
```

Generates `data/processed/backtest_sweep_summary.csv` (one row per
case, same schema as `backtest_summary.csv`) and
`backtest_sweep_calibration.csv` (aggregated summary via
`backtesting.compute_calibration_summary()`): % of cases where the real
result falls within the simulated P10-P90 range (should be around 80%
in a well-calibrated model), mean/median percentile (should be around
50, no systematic bias), mean error in wins (positive = underestimate,
negative = overestimate — same sign as the locker-room friction finding
above, but measured at full-league scale instead of 4 cases), and the
correlation between real and predicted wins (measures whether the
model at least ORDERS teams well, even if the absolute level is
shifted). Visualized in the dashboard's Backtesting tab: KPIs, a
percentile histogram, and a real-vs-simulated wins scatter plot. An
individual case with incomplete data (a real gap in 15 years of NBA
history) is skipped with a warning instead of aborting the whole sweep
— see `backtesting._run_backtest_cases()`.

Reuses the same `run_backtest_case()` and the same no-look-ahead rule
as the 4 narrative comparables — it's the same engine, more cases, not
a different model. Player career stats are downloaded ONCE per unique
player (not per case) — a player who appears in several of the 450
cases (same franchise across multiple seasons) reuses the same
already-cached full career.

### What the sweep found: two calibration bugs

The sweep's first run gave bad results with an unmistakable pattern:
only **36.7%** of cases fell within the P10-P90 range (should be
~80%), the **median** real percentile was 4.5 (should be ~50), and the
mean error was **-13.2 wins**. The breakdown by season revealed the
cause:

| Season | Real mean | PREDICTED mean | Excess |
|---|---|---|---|
| 2010-11 | 41.0 | 49.8 | +8.8 |
| 2016-17 | 41.0 | 49.3 | +8.3 |
| 2020-21 | 36.0 | 53.4 | +17.4 |
| 2024-25 | 41.0 | **66.0** | **+25.0** |

In a real 30-team league the average number of wins is *always* exactly
41 over 82 games (every win is another team's loss). The model violated
that zero-sum constraint, and the violation **grew monotonically over
time**. Two causes, both fixed:

1. **Era inflation.** The NBA's Game Score level rose from ~10.7
   per-36 in 2010-11 to ~13.4 in 2024-25 (more pace, the three-point
   revolution), but the comparison baseline was a fixed constant
   (10.0). An *average* 2024-25 team showed up +22 Game Score points
   above the reference — a credit to its era, not to itself. The
   correlation between era inflation and excess predicted wins is
   **0.926**. Fixed with
   `aging_curve.compute_league_game_score_baseline()`: each team is now
   compared against the mean of **its own season**.
2. **Un-normalized roster minutes.** The backtest summed the Game
   Score of the **entire** roster (14-18 players → 283-343 min/game)
   instead of a game's real 240 (5 positions × 48 min), inflating each
   team's strength by 18-43% and penalizing teams with heavy roster
   churn more. It was exactly the same bug already fixed in
   `league_simulation.py`, which `backtesting.py` had never received.
   Fixed by extracting the logic into
   `simulation.normalize_rotation_minutes()`, now **shared** by both
   modules so it can't diverge again.
3. **Game Score → differential scale miscalibrated by ~3.5x.** The
   `game_score_to_net_rating_scale` parameter was set to 1.0 with the
   comment *"1 Game Score point ≈ 1 differential point"* — an
   assumption never verified. Regressing the **real** point
   differential (from the 450 cases' `PLUS_MINUS` game logs) against
   the projected Game Score, normalized and era-adjusted, the empirical
   slope is **0.29**. The model was amplifying between-team differences
   ~3.5x, which also explains the distributions' overconfidence.

The baseline also stopped being the mean of the league's *players* and
became **the mean of that season's projected teams**
(`backtesting.compute_projected_league_baselines()`), which is the only
thing that satisfies the zero-sum constraint by construction: the
average team lands exactly at `net_rating = 0`.

### Result of the fixes (same cases)

| Metric | Before | After | Ideal |
|---|---|---|---|
| % within P10-P90 | 36.7% | **55.3%** | ~80% |
| Mean real percentile | 18.6 | **38.8** | ~50 |
| Median real percentile | 4.5 | **30.2** | ~50 |
| Mean error (wins) | −13.2 | **−3.5** | ~0 |
| Mean absolute error | 15.0 | **7.75** | low |
| Real vs. predicted correlation | 0.538 | **0.690** | high |

And the era bias disappeared: excess predicted wins went from growing
from +5.8 (2012-13) to +25.0 (2024-25), to staying flat between +1.6
and +4.6 across every season.

When the sweep was expanded from 450 to 480 cases (adding 2025-26), the
metrics came out practically identical (55.4% within P10-P90, MAE 7.78,
correlation 0.690), confirming the calibration is stable and didn't
depend on which specific seasons were included.

### Second round: NET_RATING and true zero-sum

When `NET_RATING` was integrated (see the advanced stats section), the
correlation rose but **calibration got worse**, which uncovered a
deeper bug: the baseline violated the zero-sum constraint through two
off-center terms that, until then, had been **canceling out by
coincidence**.

- The baseline used the team's Game Score at **full health** (88.7
  average in 2024-25) while simulated teams have injury absences
  (66.6): **−4.65** net rating for the average team.
- The synergy adjustment is **always positive** (+4.4 to +11.9,
  average +9.67) and gets added to every team's net rating equally:
  **+9.67**.
- Total: **+5.02** for the *average* team, when it should be 0 by
  definition.

With the old scale (0.29) those two came out to −6.42 + 9.67 = **+3.25**,
which explains almost exactly the −3.5-win residual bias that had been
documented for a while as "unexplained." Recalibrating the scale to
0.21 shrank the negative term and uncovered the positive one. Two large
errors of opposite sign looked like one small one.

| Metric | Before | +NET_RATING | **+ zero-sum** | Ideal |
|---|---|---|---|---|
| % within P10-P90 | 55.4% | 51.0% | **61.3%** | ~80% |
| Mean real percentile | 30.0 | 30.0 | **52.1** | ~50 |
| Median real percentile | 30.8 | 17.5 | **54.6** | ~50 |
| Mean error (wins) | −3.50 | −5.79 | **−0.04** | ~0 |
| Mean absolute error | 7.78 | 8.29 | **6.78** | low |
| Real vs. predicted correlation | 0.690 | 0.734 | **0.750** | high |

**Where the model stands:** with correlation **0.750** and MAE **6.78**,
it now **beats the honest baseline on both metrics** (0.619 and 7.39) —
before it only beat it on correlation. The systematic bias disappeared
(−0.04 wins) and the simulated mean per season is exactly 41, as the
arithmetic of a 30-team league demands.

### Empirical references obtained (useful for future calibration)

From the regression over the 450 real-data cases:

| Reference | Value |
|---|---|
| 1 real differential point = | **2.48 wins** over 82 games |
| corr(real differential, win %) | **0.966** (theoretical ceiling, outcome already known) |
| corr(PRIOR year's differential, win %) | **0.619**, MAE 7.39 wins |

That last one is the **honest baseline**: what the dumbest possible
forecast gets you (*"assume the team will perform like last year"*).
Any projection model that doesn't beat it isn't adding anything.

### Advanced stats: do they improve the forecast?

Investigated over the same 450 cases, with `leaguedashplayerstats`
(measure type *Advanced*) across the 15 seasons — 15 API calls, cheap,
because each one returns the whole league at once. Each metric was
aggregated per roster weighted by minutes, using **only the prior
season** (same no-look-ahead rule) and era-adjusted:

| Metric (prior season, era-adjusted) | corr. with real differential |
|---|---|
| `NET_RATING` | **+0.635** |
| `PIE` (Player Impact Estimate) | **+0.624** |
| `OFF_RATING` | +0.585 |
| `TS_PCT` | +0.524 |
| **Game Score (what the model uses today)** | **+0.529** |
| `DEF_RATING` | −0.396 (negative = correct: lower is better) |
| `USG_PCT` | +0.197 |

### Definitive measurement (and a correction to the one above)

That first reading used a crude aggregation. Repeated with the
pipeline's **real** projection (minutes normalized to 240, 3-season
recency, no look-ahead) and validated **leaving out one entire season
at a time** (LOSO) over the 480 cases:

| Model | Out-of-sample R |
|---|---|
| Game Score alone | 0.702 |
| Game Score + `PIE` | 0.702 — *adds nothing* |
| **Game Score + `NET_RATING`** | **0.754** |
| Game Score + `PIE` + `NET_RATING` | 0.753 — *PIE hurts* |

**`PIE` got dropped**, correcting the earlier reading that had called
it useful. Two reasons that earlier reading was misleading:
`corr(PIE, NET_RATING) = 0.64` at the team level (alone, PIE correlates
because it's riding on the same signal), and with all three variables
its coefficient comes out **negative** (−46.8), a classic collinearity
artifact — nobody would defend the claim that a higher share of
production predicts *worse* outcomes.

**Conclusion:** `NET_RATING` does help, a lot. The underlying reason is
that Game Score is a purely offensive box-score metric: it sees nothing
of defense beyond steals, blocks, and defensive rebounds. `NET_RATING`
(the team's rating while the player is on the floor) does capture it.

Nice side validation: the same regression returns a slope of
**0.2946** for Game Score alone. `game_score_to_net_rating_scale` had
been calibrated to **0.29** through a completely different route.

> ⚠️ **Important caveat for this project's use case.** `NET_RATING`
> and `PIE` are *contaminated by team context*: LeBron's `NET_RATING`
> measures how **the Lakers** performed with him on the floor, not what
> he'd contribute on a hypothetical superteam with other teammates.
> They predict very well for **real** rosters that stay in a similar
> context, but transplanting that number onto a made-up roster is
> exactly the kind of assumption this project already showed fails (see
> the superteam friction section). Game Score, being purely individual,
> is more transferable even though it predicts worse.
>
> **Even so, the weight isn't shrunk.** Shrinking it was tested (0.25,
> 0.4, 0.5, 0.6, 0.75) and out-of-sample R improves *monotonically* up
> to the full weight: there's no empirical overfitting signal that
> would justify it. Shrinking it "just in case" would be forcing a
> parameter against the evidence to make the result look like what's
> expected — the mistake this project already made once. The limitation
> is documented; the parameter isn't manipulated. Anyone who wants to
> measure its effect has the lever: `advanced_impact: {enabled: false}`
> in the YAML recovers exactly pure Game Score (remember to set
> `game_score_to_net_rating_scale` back to 0.29).

**Already integrated** in `src/advanced_impact.py`, as an additive
adjustment in Game Score units:

```
impact/36 = game_score/36
             + 0.42 * (NET_RATING − its season's mean)
             − 57.03 * (PCT_PLUSMINUS − its season's mean)
```

Centering by season preserves the zero-sum constraint by construction
and adjusts for era inflation for free. Requires
`league_advanced_player_stats.csv` (generated by `--backtest-sweep`,
one call per season); without that CSV the project keeps running on
pure Game Score.

**Third metric: `PCT_PLUSMINUS`** (tracking-based defense,
`leaguedashptdefend` / Second Spectrum, available since 2013-14) — how
much worse the real opponent's shooting % gets when this player is the
closest defender, compared to their normal shooting %. Unlike *hustle
stats* (contested shots, deflections... investigated and discarded, no
signal — see `scripts/experiments/hustle_stats_signal.py`), this one
IS a direct defensive-impact signal, validated leave-one-season-out
(out-of-sample ΔR² +0.016 over the backtest sweep's 480 cases, a real
but modest improvement). Weight derived with the same technique as
`net_rating_weight` (coefficient ratio from a joint regression, not a
made-up scale) — see `scripts/experiments/pt_defend_signal.py` and
`advanced_impact.py`'s docstring for the full detail, including the
recalibration of `game_score_to_net_rating_scale` it required (0.172 →
0.1617).

## Web frontend (`webapp/`)

The project's only interface: pure HTML/CSS/JS (no frameworks or
third-party dependencies) served by a FastAPI backend. There was also a
Streamlit dashboard (`dashboard/app.py`) running in parallel during
development — it was retired once `webapp/` covered all its tabs, to
avoid maintaining two interfaces with the same information.
`dashboard/data_loader.py` is still alive: it's the data
loading/combination layer (testable,
`tests/test_dashboard_data_loader.py`) that every router in
`webapp/routers/` reuses without duplicating any transformation, same
as `src/awards_projection.py` / `src/champion_profiles.py` /
`src/llm_explainer.py` in Liga NBA.

```bash
uvicorn webapp.main:app --reload
# opens http://localhost:8000
```

Three top-level tabs:

- **🏀 Mi equipo** (sub-tabs):
  - **Roster y proyecciones** — combined table from `aging_curve.py` +
    `injury_model.py` + `fatigue_accumulation.py` per player, with
    `role_expected`/`minutes_projection` from `team_config.yaml`, plus
    REAL **GP** (games played) and **MPG** (minutes/game) from each
    player's most recent registered season — different from
    `minutes_projection`, which is the ASSUMED minutes load for the
    simulated season. Includes the editable hypothetical roster
    (add/remove/swap any player from the 30 teams, see the league
    sandbox section above). Each player's detail popup (double-click
    their name) includes a **real shot chart** over a hand-drawn SVG
    half court (`webapp/static/js/court.js`, no external library):
    exact location of every shot from their most recent real season
    (`ShotChartDetail` via
    `data_pipeline.build_roster_shot_charts_dataset`), green/red for
    made/missed. Cached in `data/processed/roster_shot_charts.csv` —
    the router never calls `nba_api` from an HTTP request (same
    principle as the rest of `webapp/routers/players.py`).
  - **Simulación Monte Carlo** — win and Net Rating distribution from
    `simulation_results.csv`.
  - **Sinergia de alineación** — full table from
    `lineup_synergy_pairs.csv`.
  - **Backtesting** — `backtest_summary.csv`, with an automatic warning
    when a case falls in an extreme percentile (<5% or >95%) (see the
    Backtesting section above).
- **🏆 Liga NBA** (sub-tabs):
  - **Liga y Playoffs** — average wins and playoff/championship
    probabilities for the 30 teams, with/without injuries scenario
    selector, team explorer (with the same GP/MPG columns, including
    your hypothetical roster if a simulated league is active), a
    playoff bracket simulator with a visual tree (one concrete
    realization — play-in, round 1, conference semis and finals —
    different every time it's clicked), and a full season schedule
    simulator (`league_simulation.run_single_league_season_simulation`):
    the result of EVERY game of one concrete realization, browsable and
    filterable by team, with an ILLUSTRATIVE per-player boxscore
    (already-projected per-game season average + noise, independent
    categories — not a joint play-by-play simulation) and a
    head-to-head widget between any two teams on that same schedule.
    Uses the REAL schedule published by the NBA when it exists
    (`data_pipeline.build_league_schedule_dataset`) — real dates,
    opponents, rest, and home-court advantage; otherwise falls back to
    a synthetic round-robin schedule.
  - **Premios individuales** — MVP, DPOY, 6th Man, ROY, MIP, and COY
    heuristics over the already-computed projections, via
    `src/awards_projection.py`. **These are NOT a prediction of the
    real media vote** — each formula (projected Game Score weighted by
    team wins for MVP, a steals/blocks/defensive-rebound proxy for
    DPOY, real improvement between the last two seasons for MIP, the
    team that most exceeded its real record from the prior year as a
    COY proxy since this project doesn't model coaches...) is
    documented with its limitations in the module's docstring.
    MVP/DPOY/ROY/MIP/6th Man are shown as a **visual ranking**
    (`leaderboard.js`, real photo + bar proportional to the season
    value, no table) — click any row to open the player's detail
    popup; hovering shows a preview card (`player-preview.js`) with the
    same set of stats for all 4 awards and for the lineups, per the
    user's request: PPG/RPG/APG/SPG/BPG/FG%/3P%, team record, and the
    "value" that actually ranks THAT award (mvp_score/dpoy_score/
    season_value/defensive_value). **MIP is the exception**: instead of
    just the projection, it compares each stat "real previous season →
    projected" (`awards_projection.compute_latest_real_season_stats`),
    since MIP is voted on improvement that has ALREADY happened, not on
    a projection. **Entrenador del Año** and **All-Star** are also
    visual rankings, not tables: COY (a TEAM award, no player — this
    project doesn't model coaches at all) uses `teamLeaderboardChart()`,
    with the team's badge instead of a player photo, opening the TEAM
    popup on click; All-Star reuses the regular `leaderboardChart()`,
    split into two East/West columns, with a "Starter"/"Reserve" tag on
    each row. The **All-NBA** and **All-Defensive** lineups
    (`compute_all_nba_teams`/`compute_all_defensive_teams`, classic
    2 guards + 2 forwards + 1 center format) are drawn directly on a
    real half court (`court.js::courtLineup`, the same half court with
    real physical dimensions as the shot chart) with each player's
    photo — no table next to it, same click/hover pattern and the same
    set of stats as the leaderboard. The exact position within each
    G/F group is illustrative only (the
    model doesn't distinguish point guard from shooting guard, or
    small forward from power forward — see the notice on the tab
    itself).
  - **Campeones reales** — historical comparables
    (`champion_profiles.py`).
- **🤖 Explicador (IA)** — natural-language chat over ALL the data
  already computed in the other tabs (`src/llm_explainer.py`, via the
  Groq API — open-weight models served with very fast inference). The
  model receives a text snapshot with the real numbers already
  generated by the pipeline (roster projections, injury/wear risk,
  simulation results, backtesting, league standings) as a `system`
  message, with explicit instructions not to make up numbers and to
  flag when a question depends on data that hasn't been generated yet.
  It doesn't run any new simulation or replace any existing calculation
  — it only narrates over what's already been computed. Requires the
  `GROQ_API_KEY` environment variable; without it, the tab shows a
  notice and doesn't attempt to call the API.

  **Recent news RAG (optional, two phases):** the statistical pipeline
  can't see today's news (last-minute injuries, coaching changes)
  because it doesn't come from any computed CSV. Phase 1: a text box
  where you paste articles/headlines by hand — TF-IDF
  (`src/llm_explainer.py::retrieve_relevant_news_snippets`, no new
  dependencies) retrieves the fragments relevant to the question and
  adds them to the prompt in a section clearly labeled as NOT
  verified, never mixed with the pipeline's data. Phase 2: a "Search
  recent news" button (`src/news_search.py`, Tavily API) that fills
  that same box on the user's explicit demand — it's the project's
  ONLY live network call outside of `nba_api`, never automatic.
  Requires `TAVILY_API_KEY`; without it, that specific button shows a
  notice but you can keep pasting text by hand (phase 1 doesn't depend
  on this variable).

Season **total** stats are shown rounded to a whole number (they're
continuous projections, but "1600 PTS" reads better than "1600.34" —
decimals in a total don't add real precision, just visual noise).
**Per-game** stats keep 1 decimal (there it matters: 22.7 PPG vs.
23 PPG is a real difference over a season).

Charts (histograms, Net Rating line, calibration scatter) are
hand-made SVG, no external library. Team/NBA logos are loaded live from
`cdn.nba.com` (never saved in the repo, which is public) and fall back
to a badge with the team's initials if the image fails to load. Fixed
dark theme with a brand gradient (navy → black → very dark red) in
`webapp/static/css/tokens.css`. Tests in `tests/test_webapp_api.py`.

## Installation

```bash
git clone <your-repo>
cd nba-superteam-sim
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # optional: only if you'll use the "Explicador (IA)" tab
```

Edit `.env` and fill in `GROQ_API_KEY` with your key from
[console.groq.com/keys](https://console.groq.com/keys). The `.env`
file is in `.gitignore` — never committed. Without this variable, the
rest of the dashboard works the same; only the "Explicador (IA)" tab
is disabled. `TAVILY_API_KEY` ([tavily.com](https://tavily.com)) is
optional within that same tab — without it, the "Search recent news"
button shows a notice but you can keep pasting news text by hand.

## Usage: downloading the data

```bash
cd src
python data_pipeline.py
```

This downloads (caching to `data/raw/`) each roster player's career
history (regular season and playoffs) and the full game logs for the
comparable teams defined in `config/team_config.yaml`, and leaves
consolidated datasets in `data/processed/`.

> **Default behavior: offline-first.** The first run downloads and
> caches everything to `data/raw/*.csv`. **From the second run
> onward, the pipeline does NOT call the API again** — it reads
> directly from local CSVs (you'll see `[local cache]` in the console
> instead of `[stats.nba.com API]`). If you need to force a refresh
> (e.g. after new games have been played), use:
> ```bash
> python data_pipeline.py --refresh
> ```
> `stats.nba.com` applies aggressive rate-limiting, so the pipeline
> includes pauses between calls and retries with backoff — but thanks
> to the cache, you'll barely need it in normal day-to-day use.

## Reusing the pipeline for another team

Edit `config/team_config.yaml`:

1. Change `team.team_id` and `team.name` (NBA team IDs are documented
   in the `nba_api` library itself, or can be obtained with
   `nba_api.stats.static.teams`).
2. Replace `roster` with the players you want to simulate, with their
   `player_id` (obtainable via `nba_api.stats.static.players`).
3. You can leave `historical_comparables` as-is (they serve as a
   general "superteam effect" benchmark) or adapt them to cases more
   relevant to your comparison.
4. Run `python src/data_pipeline.py` again — no need to touch a single
   line of code.

## Project structure

```
nba-superteam-sim/
├── config/
│   └── team_config.yaml            # single source of truth for "which team"
├── data/
│   ├── raw/                        # cached nba_api responses, unprocessed
│   └── processed/                  # consolidated datasets ready for modeling
├── src/
│   ├── config_loader.py            # YAML reading and validation
│   ├── data_pipeline.py            # ingestion with caching and retries
│   ├── season_utils.py             # shared traded-season dedupe
│   ├── aging_curve.py              # individual per-36 projection with age adjustment
│   ├── lineup_synergy.py           # Game Score adjustment for lineup fit
│   ├── simulation.py               # Monte Carlo simulation engine (one team vs. generic WinPCT)
│   ├── league_simulation.py        # 30 real teams, regular season + playoffs
│   ├── backtesting.py              # retrospective backtest against real comparables
│   └── context/                    # season context layer (full roadmap)
│       ├── injury_model.py         # per-player injury risk_score
│       ├── fatigue_accumulation.py # fatigue_score from minutes wear
│       ├── schedule_strength.py    # difficulty_score per schedule game
│       ├── performance_curve.py    # estimated Net Rating in rolling windows
│       ├── opponent_weighting.py   # Net Rating weighted by opponent strength
│       └── conference_adjustment.py # East/West normalization between comparables
├── tests/
│   ├── test_config_loader.py
│   ├── test_aging_curve.py
│   ├── test_lineup_synergy.py
│   ├── test_simulation.py
│   ├── test_league_simulation.py
│   ├── test_backtesting.py
│   ├── test_dashboard_data_loader.py
│   ├── test_webapp_api.py
│   ├── test_injury_model.py
│   ├── test_fatigue_accumulation.py
│   ├── test_schedule_strength.py
│   ├── test_performance_curve.py
│   ├── test_opponent_weighting.py
│   └── test_conference_adjustment.py
├── scripts/
│   ├── resolve_player_ids.py       # resolves player_id without network calls
│   └── experiments/                # exploratory investigations, OUTSIDE the pipeline
│       ├── requirements-experiments.txt  # pymc/arviz/statsmodels/lifelines -- not in requirements.txt
│       ├── bayesian_calibration.py       # Bayesian recalibration of game_score_to_net_rating_scale
│       ├── aging_curve_shrinkage.py      # discarded: shrinkage doesn't explain the talent compression
│       ├── team_quality_uncertainty.py   # team-quality uncertainty calibration
│       ├── hustle_stats_signal.py        # discarded: hustle stats add no signal
│       ├── pt_defend_signal.py           # tracking-based defense -- integrated into advanced_impact.py
│       ├── injury_survival_model.py      # discarded: Cox doesn't improve injury_model.py's heuristic
│       ├── game_win_predictor.py         # discarded: GBT doesn't improve compute_win_probabilities's logistic model
│       └── game_win_predictor_injury_signal.py  # positive: key-player availability improves the Brier score, even in a deployable pregame version
├── notebooks/                      # 4 investigations narrated visually (complement, don't replace, scripts/experiments/)
├── dashboard/
│   └── data_loader.py              # CSV loading/combination, testable -- webapp/'s data layer (the Streamlit dashboard that used to live here was retired)
├── webapp/                         # the project's only interface: HTML/CSS/JS + FastAPI
│   ├── main.py                     # FastAPI: mounts routers + serves static/
│   ├── serializers.py              # DataFrame -> JSON (NaN/NaT -> None)
│   ├── routers/                    # /api/status, /api/roster, /api/simulation, ...
│   └── static/                     # index.html, css/, js/ (no external dependencies)
└── requirements.txt
```

## Next steps

See the "Current status" section above. With every piece of the
original roadmap implemented, including the dashboard and the full
league simulation, the project is functionally complete end to end
(data → context → projection → simulation → league/playoffs →
backtesting → visualization). Two reasonable directions remain:

1. **A "locker-room friction" proxy** — backtesting showed the engine
   systematically overestimates when there's known real conflict
   (Heat, Nets, Suns). There's no box-score data point that captures
   this directly, but an indirect signal could be explored (e.g.
   game-to-game performance volatility of the stars) before assuming
   it can't be modeled.
2. **Real re-seeding between playoff rounds** — `league_simulation.py`
   still uses a FIXED bracket (no re-seeding after each round, unlike
   the real NBA). The official 2026-27 schedule has already been
   integrated (`real_schedule_to_games`, see the section above) — this
   is now the only one of the two still pending.
