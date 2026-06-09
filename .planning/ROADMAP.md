# Roadmap — Odysse Driver Analysis Dashboard v2

## Phase 1: App Shell + Final Findings (CURRENT)
Core infrastructure and the primary presentation page.

**Deliverables:**
- Clean `app.py` skeleton with sidebar routing, shared CSS, shared helper functions
- Final Findings page — all 9 sections fully implemented
- Shared `_metric_card`, `_scorecard`, zone matrix helpers

**Reference:** DASHBOARD_RECONSTRUCTION.md §§ 1–9 (Final Findings sections)

---

## Phase 2: Comparison Pages
The core analytical comparison pages.

**Deliverables:**
- Good vs Bad Drivers page (7 sections)
- Gap Analysis page (Group Comparison + Outlier Spotlight + per-driver tabs)

**Reference:** DASHBOARD_RECONSTRUCTION.md §§ Gap Analysis, Good vs Bad

---

## Phase 3: Fleet Intelligence
Fleet-level views.

**Deliverables:**
- Fleet Map page (scatter mapbox by category + west % bar + summary table)
- Driver Day page (any-driver search + Folium map + timeline + earnings curve)
- Patterns Summary page

**Reference:** DASHBOARD_RECONSTRUCTION.md §§ Fleet Map, Driver Day

---

## Phase 4: Deep Zone Analysis
Zone-specific analytical pages.

**Deliverables:**
- Zone 1 Selectivity
- Zone 1: The Why
- Zone 3 Deep Dive
- Trip Flow
- Trip Strategy DNA
- Airport Run Model

---

## Phase 5: Performance Overview Pages
Standard performance dashboards.

**Deliverables:**
- Overview
- Map View
- Time Patterns
- Daily Trends
- Trip Economics
- Zone Analysis
- Day Patterns
- Shift Behaviour
