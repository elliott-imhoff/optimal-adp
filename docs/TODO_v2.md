# TODO_v2

## Immediate Work (Highest Priority)

- [ ] Gather historical data (10 years) into `data/` in consistent CSV format
  - Details: ensure each file is named or contains a year column; columns must include: Player, Pos, Team, AVG, TTL (or documented mapping)
  - Outcome: raw per-year CSVs in `data/` ready for aggregation

- [ ] Add `optimal_adp/historical.py` implementing the historical aggregation pipeline
  - Functions to implement:
    - `load_historical_data(data_dir: str) -> dict[int, list[Player]]` — read all CSVs in `data/` and return mapping year -> players (use `load_player_data` or a wrapper to parse rows into `Player`)
    - `compute_pos_rankings_for_year(players: list[Player]) -> dict[str, list[Player]]` — sort players within position by `avg`
    - `compute_baseline_points_by_year(players_by_year: dict[int, list[Player]], baseline_positions: dict[str,int]) -> dict[int, dict[str, float]]` — for each year and position, get baseline player avg (use `optimal_adp.config.BASELINE_POSITIONS`)
    - `average_baseline_points(baseline_points_by_year: dict[int, dict[str,float]]) -> dict[str, float]` — compute mean baseline per position across years
    - `compute_multi_year_vbr(players_by_year: dict[int, list[Player]], averaged_baselines: dict[str,float]) -> dict[int, list[tuple[Player, float]]]` — compute per-year VBRs using averaged baselines
    - `compute_mean_adp_across_years(vbr_rankings_by_year: dict[int, list[tuple[Player, float]]]) -> dict[str, float]` — compute average ADP per player across years
    - `compute_adp_from_vbr(players: list[Player], baselines: dict[str,float]) -> list[tuple[Player, float, int]]` — helper to compute ADP from VBR (reuse existing `compute_initial_adp` logic or call it with adjusted baselines)
  - Artifacts to write:
    - `artifacts/historical/averaged_baselines.csv`
    - `artifacts/historical/vbr_by_year/<year>_vbr.csv`
    - `artifacts/historical/avg_adp_across_years.csv`
  - Implementation notes:
    - Use deterministic file ordering when reading `data/` (sorted by filename) so runs are reproducible
    - Use `csv` library for all IO
    - Add `logging` statements and minimal docstrings for each new function
    - No global state — pass configuration explicitly (e.g., baseline positions)

- [ ] Add unit tests for the historical pipeline
  - File: `tests/test_historical.py`
  - Use a tiny fixture (2 years, 3 players per year) to validate:
    - baseline averaging
    - per-year VBR computation using averaged baselines
    - mean ADP aggregation across years

- [ ] Add CLI or small runner for historical pipeline
  - File: `scripts/compute_historical.py` or add `optimal_adp/cli.py` command
  - CLI args: `--data-dir`, `--artifacts-dir`, `--baseline-positions-file` (optional)
  - Behavior: run pipeline and write artifacts to `artifacts/historical/`

- [ ] Write README section documenting historical pipeline and how to run it
  - Update `README.md` with a short example command

- [ ] Run tests and verify artifact CSVs are created correctly
  - Command (local): `poetry run pytest tests/test_historical.py`


## Follow-up Work (Next Priority)

- [ ] Compute ADP on averaged data and compare to this year's ADP
  - Steps:
    - Use `compute_adp_from_vbr` with `averaged_baselines` to produce `optimized_adp_from_averages.csv`
    - Load this year's ADP (from current pipeline / provided ADP source)
    - Create comparison report `artifacts/historical/adp_comparison_this_year.csv` listing: player, position, team, avg, total, optimized_adp, this_year_adp, delta
    - Add a short summary function that highlights most under/over-valued players by delta and by position
  - Tests:
    - Add tests for the comparison logic using small fixtures

- [ ] Integrate averaged-baselines ADP as an optional input to optimizer flow
  - Add an argument to `run_optimization_loop` or a wrapper that can accept `baseline_overrides` so `compute_initial_adp` uses averaged baselines


## Future / Harder Tasks (deferred; see `docs/FUTURE_TASKS.md`)

- [ ] Experiment with different drafting strategies (zero-RB, early-QB, safety-first)
- [ ] Add bench players to draft simulation and scoring
- [ ] Bake in risk: injury and underperformance metrics into `Player` model and VBR calculations


## Notes & Conventions
- Artifacts directory: `artifacts/historical/`
- Use `optimal_adp.config.BASELINE_POSITIONS` as default baseline ranks for computing per-year baseline points unless explicitly overridden
- Keep functions small, pure where possible, and add type hints and docstrings (Python 3.10 style)
- Add logging statements; keep verbose/debug behind a `--verbose` flag in any CLI
- Use `csv` library for reading/writing all CSV artifacts


---

Mark items with the checkbox as progress is made. I will not modify code until you ask me to start implementing any checked item.
