# Requirements — Odysse Driver Analysis Dashboard v2

## Source of Truth
All page specifications are in `DASHBOARD_RECONSTRUCTION.md` at the project root.

## Non-Negotiables
- Single `app.py` file (same run command: `python -m streamlit run app.py`)
- Dark theme throughout (`#1e1e2e` card backgrounds, `carto-darkmatter` maps)
- All 20 sidebar pages implemented
- `config.py`, `db.py`, `zones.py` left unchanged
- Credentials must stay in `config.py` only (not hardcoded in app.py)

## 20 Pages (in sidebar order)
1. Final Findings — 9-section presentation page (primary deliverable)
2. Patterns Summary
3. Good vs Bad
4. Fleet Map
5. Driver Day — open to any driver via search
6. Gap Analysis — includes Outlier Spotlight section
7. Trip Flow
8. Zone 1 Selectivity
9. Zone 1: The Why
10. Trip Strategy DNA
11. Airport Run Model
12. Zone 3 Deep Dive
13. Overview
14. Map View
15. Time Patterns
16. Daily Trends
17. Trip Economics
18. Zone Analysis
19. Day Patterns
20. Shift Behaviour

## Shared Helpers (top of app.py)
- `_metric_card(label, value, delta, color)` — dark styled stat card
- `_ff_zone_matrix(flow_raw)` — 6×6 zone pivot % matrix
- `_safe_cell(mat, r, c)` — safe pivot lookup
- `_flow_west_pct(flow_df)` — % pickups west of -0.12
- `_compute_gaps(df)` — inter-trip gap series
- `_gap_buckets(s)` — <25m / 25–75m / >75m bucket dict
- `_ew_parse_and_flag(flow_df)` — add is_west bool col
- `_ping_stats(acc_df, dec_df, n_drivers)` — normalised ping volume/quality stats
- `_scorecard(col, label, color, ...)` — renders scorecard card

## Category Colours
| Cat | Hex |
|-----|-----|
| A | `#22c55e` |
| B1 | `#4ade80` |
| B2 | `#60a5fa` |
| C1 | `#f59e0b` |
| C2 | `#fb923c` |
| D | `#ef4444` |
| — | `#94a3b8` |
