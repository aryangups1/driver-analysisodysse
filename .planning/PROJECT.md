# Odysse Driver Analysis Dashboard

## Overview
A Streamlit analytics dashboard for analysing Odysse taxi driver performance data from Azure PostgreSQL. Provides 20 pages of analysis covering fleet performance, driver behaviour patterns, zone strategies, and actionable insights for fleet management.

## Stack
- **Python** + **Streamlit** (web framework)
- **Plotly Express / Graph Objects** (charts)
- **Folium** (interactive maps)
- **PostgreSQL** on Azure (`dev-odysse.postgres.database.azure.com`, db: `unifieddwh`)

## Key Data
- **Tables:** `rep_fact_trips`, `rep_fact_driver_performance`
- **Accepted status:** `status IN ('completed', 'Finished')`
- **Declined status:** `'Driver did not respond', 'Driver rejected'`
- **Max trip filter:** 60 miles (removes Cambridge/Southampton outliers)
- **West boundary:** -0.12° lon (Charing Cross line)

## Driver Groups
- **Top 10 (TOP_DRIVER_IDS):** 128, 81, 155, 123, 180, 230, 228, 130, 182, 195
- **Comparison/Worst 5 (BAD_DRIVER_IDS):** 82, 178, 72, 36, 32
- **Outlier Spotlight:** Yousuf (219), Mukhtar (215), Abdi (180), Plummer (223)

## Fleet Baselines
- Fleet RPH: £23.04 | Accept: 57.2% | Util: 68.2%

## Files to Keep (unchanged)
- `config.py` — DB credentials, driver ID lists, names
- `db.py` — all DB query functions with caching
- `zones.py` — geo utilities (parse_dms, assign_zone, calc_true_rph, etc.)
- `driver_categories.csv` — dim_driver_id → A/B1/B2/C1/C2/D category
- `flow_data.parquet` — pre-built trip flow data
- `london_zones.geojson` — London zone boundaries

## Rebuild Goal
Rebuild `app.py` clean from scratch — single well-structured file with clear page sections, shared helpers at the top, no dead code, consistent styling throughout.

## Run Command
```
python -m streamlit run app.py
```
