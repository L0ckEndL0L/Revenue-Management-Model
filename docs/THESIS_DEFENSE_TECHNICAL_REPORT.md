# RateAnchor Revenue Management System

## Master's Thesis Defense Technical Report

**Implementation baseline:** repository state inspected 15 July 2026  
**Verification baseline:** 71 automated tests passed; 3 non-failing date-parsing warnings  
**System classification:** explainable, configurable hotel revenue-management decision-support prototype  
**Primary entry points:** `app.py` (Streamlit) and `main.py` (CLI/programmatic pipeline)

> **Evidence rule used in this report.** Every statement about implemented behavior is grounded in the repository's Python source, tests, data, or generated output. The repository contains monthly (31-row), February (28-row output), and larger historical periods (65 validated rows and several 92-row legacy runs), but it contains no identifiable annual experiment and no artifact that identifies a 13-room hotel. Those requested study conditions are therefore documented as proposed or unverified, not reported as completed experiments. This distinction is essential to a defensible research claim.

---

# 1. Executive Summary

## 1.1 What RateAnchor is

RateAnchor is a modular hotel revenue-management decision-support system that transforms property-management-system (PMS) reports and optional contextual data into daily demand forecasts, baseline rate recommendations, tailored property-specific rate recommendations, backtest metrics, explanation fields, warnings, charts, and downloadable artifacts. It is delivered through a Streamlit user interface and a command-line/programmatic pipeline.

The implementation is deliberately hybrid. It combines:

- deterministic PMS data repair, schema mapping, normalization, and validation;
- hotel KPIs and year-over-year (YoY) comparisons;
- day-of-week/month seasonal demand forecasting with sparse-history protection;
- an optional regression/random-forest forecasting path selected only when history is sufficient and validation shows improvement;
- a simple occupancy-threshold baseline pricing control;
- an elasticity-based candidate-rate simulator;
- a tailored recommendation layer using property type, segment posture, occupancy, pace, seasonality, events, and date-level competitive median rates;
- budget and month-end revenue context;
- chronological demand backtesting and historical ADR recommendation backtesting; and
- explicit reasoning, confidence, warning, and status fields.

It is not a production autonomous RMS. It has no live PMS, rate-shop, weather, event, or channel-manager API; it does not execute rates; and it does not demonstrate causal revenue lift. It is a research prototype for transparent, reviewable pricing support.

## 1.2 Problem solved

Hotel room inventory is perishable: an unsold room night cannot be stored for later sale. Pricing must balance rate and conversion under uncertain demand. Smaller or heterogeneous hotel portfolios also face inconsistent report formats, sparse histories, varying inventory, incomplete competitor data, and limited trust in black-box recommendations.

RateAnchor addresses the software problem of converting imperfect operational reports into a consistent decision workflow. It gives a revenue manager an auditable recommendation with its inputs, constraints, contextual signals, and limitations. Its core research contribution is not a claim of globally optimal price; it is the integration of data-quality controls, sparse-data fallback, configurable property behavior, competitive anchoring, and explanation into one testable system.

## 1.3 Why it was developed

The project tests whether an explainable, configurable workflow can be more operationally useful than a once-daily rule-based baseline when property history and market context vary. The design favors visible assumptions and conservative fallbacks because an opaque optimizer trained on insufficient data could produce precise-looking but unreliable recommendations.

## 1.4 Overall workflow

```text
User / CLI
   |
   +--> historical PMS CSV/XLS/XLSX (required)
   +--> future on-books CSV/XLS/XLSX (optional but operationally expected)
   +--> events / budget / comp set / intraday updates (optional)
   +--> mappings, inventory override, rate limits, elasticity,
        property type, segment focus, priority weights
            |
            v
Read and repair report structure
            |
            v
Map source columns to canonical schema
            |
            v
Normalize dates, currency, percentages, and numeric types
            |
            v
Validate business rules; remove critical invalid rows
            |
            v
Calculate daily KPIs and select current/prior-year frames
            |
            +--> YoY and pace/event context
            +--> future demand forecast
            +--> month forecast and budget context
            |
            v
Baseline occupancy-rule recommendation
            |
            +--> elasticity candidate-rate recommendation
            +--> tailored property/segment/comp-anchor recommendation
            |
            v
Chronological forecast backtest + historical ADR rate backtest
            |
            v
CSV + TXT + PNG + ZIP + Streamlit dashboard
```

## 1.5 Primary contribution

The primary contribution is an end-to-end, implementation-tested architecture for explainable hotel pricing support under imperfect data. It makes four separable research objects observable:

1. **Data validity:** whether the operational input can safely support a recommendation.
2. **Demand prediction:** whether a calibrated forecast improves on a day-of-week baseline.
3. **Rate behavior:** how baseline, elasticity, and tailored rules transform demand/context into a bounded rate.
4. **Operational reviewability:** whether the output exposes warnings, confidence, reasoning, median-rate source, budget context, and ranked attention lists.

---

# 2. Complete System Architecture

## 2.1 Architectural layers

```text
Presentation layer
  app.py
  ui/upload_panel.py, dataset_panel.py, budget_panel.py,
  tailored_panel.py, results_panel.py, chart_helpers.py
                    |
Application orchestration
  main.py
  pipeline_config.py, pipeline_inputs.py,
  pipeline_budget_forecast.py, pipeline_outputs.py,
  pipeline_reporting.py
                    |
Domain services
  ingest.py, schema.py, validate.py, metrics.py, yoy.py,
  pace.py, events.py, forecast.py, baseline.py, pricing.py,
  tailored.py, elasticity.py, budget.py, intraday.py,
  evaluation.py, dataset_manager.py
                    |
Persistence / artifacts
  data/, datasets/, outputs/run_<timestamp>/
```

The presentation layer holds session state and user controls. `main.run_pipeline` is the application service that coordinates domain modules. Domain modules are mostly dataframe-in/dataframe-out functions. Persistence is file-based; there is no database or external service boundary.

## 2.2 Stage-by-stage processing pipeline

### Stage A — CSV/Excel acquisition

**Purpose.** Accept a PMS export despite metadata preambles, embedded headers, repeated print headers, malformed delimiters, or shifted currency columns.

**Inputs.** Historical report path; optional future report path; CSV, XLS, or XLSX file; UI-uploaded file-like object.

**Outputs.** Raw but structurally cleaned `pandas.DataFrame`.

**Files/functions.** `src/ingest.py`: `read_table_source`, `load_file`, `read_excel_with_report_header`, `_read_aligned_report_csv`, `clean_report_dataframe`.

**Algorithm.** Detect suffix; for CSV, try aligned PMS-row parsing, normal `read_csv`, delimiter inference, semicolon, tab, then a permissive malformed-quote fallback. For Excel, read without a header, scan the first 25 rows for a likely metric header, promote it, and clean artifacts.

**Why it exists.** A model cannot be validated if parsing silently shifts revenue into ADR or treats report footer totals as daily observations.

**Connection.** The resulting source columns pass to schema mapping.

### Stage B — Structural cleaning and header repair

**Purpose.** Remove non-observation rows and repair common PMS report geometry.

**Inputs.** Source dataframe.

**Outputs.** Cleaned source-shaped dataframe with unique column names.

**Files/functions.** `src/ingest.py`: `_drop_report_artifact_rows`, `_promote_embedded_header_row`, `_make_unique_column_names`, `_repair_shifted_currency_column`.

**Logic.** Drop all-empty rows/columns; remove repeated headers, `Totals:` rows, and duplicate `Date Occupancy` headers; repair `Room Revenue` when fill ratio is below 0.85 or median scale is implausible; repair ADR when incomplete or outside a practical 20–500 median range.

**Connection.** Clean source fields become candidates for canonical mapping.

### Stage C — Canonical column mapping

**Purpose.** Decouple downstream algorithms from vendor-specific column names.

**Inputs.** Cleaned dataframe; automatic alias table; optional saved/user mapping; required-field set appropriate to historical or future data.

**Outputs.** Canonical dataframe containing `stay_date`, `rooms_available`, `rooms_sold`, `room_revenue`, and optional fields such as `occupancy_percent`, `adr`, `current_rate`, `booking_date`, `room_type`, `rate_code`, and `channel`.

**Files/functions.** `src/schema.py`: `find_column_match`, `auto_map_columns`, `apply_column_mapping`; `src/ingest.py`: `map_columns`.

**Algorithm.** Case-insensitive exact matching against alias lists; user mapping overrides valid automatic mappings; currency-column heuristics repair revenue/ADR mapping; interactive CLI or Streamlit mapping resolves remaining fields. Required fields cannot share a source column. `rooms_available` may be deferred if `rooms_sold` and occupancy are present.

**Why it exists.** A canonical contract makes metrics, validation, forecast, and recommendation modules testable independently of PMS format.

**Connection.** Mapped data is normalized.

### Stage D — Type normalization and feature recovery

**Purpose.** Convert display strings into computational types and reconstruct inventory when permitted.

**Inputs.** Canonical mapped dataframe.

**Outputs.** Datetime/numeric dataframe plus `rooms_available_derived_from_occupancy` provenance flag.

**Files/functions.** `src/ingest.py`: `parse_dates`, `convert_numeric_columns`, `normalize_data`.

**Formulas.** If occupancy is expressed as percent, `occ = occupancy_percent / 100`; otherwise it is already decimal. When inventory is absent or zero:

```text
rooms_available = round(rooms_sold / occ), for occ > 0
```

Currency symbols, commas, percent signs, and spaces are removed. Invalid dates become `NaT`; invalid counts become zero; revenue, ADR, and rate retain missing values for later date-aware validation.

**Connection.** Normalized rows enter business validation.

### Stage E — Validation

**Purpose.** Prevent structurally invalid observations from contaminating KPIs and recommendations while preserving recoverable future rows.

**Inputs.** Normalized historical/future data, `allow_overbooking`, as-of date, optional default rate.

**Outputs.** Valid-row dataframe and `ValidationResult`; merged text report.

**Files/functions.** `src/validate.py`: `validate_data`, `check_data_quality`, `save_validation_report`; `main.py`.

**Critical row-removal rules.** Null date; negative available rooms; negative rooms sold; disallowed overbooking; negative revenue; missing revenue on non-future dates.

**Recoverable rule.** Missing/non-positive current rate is filled in order from ADR, revenue/rooms sold, or a positive median/default fallback (ultimately 120.00). Only the final fallback generates `CURRENT_RATE_FILLED`; that issue does not remove the row.

**Connection.** Valid historical rows become KPI/forecast history; valid future rows become on-books recommendation dates.

### Stage F — KPI calculation

**Purpose.** Create a consistent daily analytical basis.

**Inputs.** Valid canonical historical data.

**Outputs.** Daily metrics containing occupancy, occupancy percent, ADR, and RevPAR.

**Files/functions.** `src/metrics.py`: `calculate_daily_metrics`, `export_metrics`.

**Formulas.** See Section 6. Division-by-zero produces 0.0 in this module.

**Connection.** Metrics feed forecasting, YoY, budget history, and evaluation.

### Stage G — Future on-books preparation

**Purpose.** Establish future stay dates, inventory, booked rooms, and a rate reference.

**Inputs.** Explicit future report, or future-dated rows from historical input; optional manual inventory.

**Outputs.** Sorted future dataframe.

**Files/functions.** `src/pipeline_inputs.py`: `prepare_future_dataset`.

**Fallback order for `current_rate`.** Existing positive current rate, then ADR, then `room_revenue / rooms_sold`, then validation fallback.

**Connection.** Future data is separately validated and sent to demand forecast.

### Stage H — Demand forecast

**Purpose.** Estimate final rooms sold while respecting on-books rooms and physical capacity.

**Inputs.** Historical daily metrics, future on-books data, optional events.

**Outputs.** `base_demand`, `forecast_rooms_sold`, `forecast_occ`, history depth, and forecast method.

**Files/functions.** `src/forecast.py`: `build_future_forecast`.

**Algorithm.** Average day-of-week demand and month demand; blend the seasonal estimate with an inventory/on-books fallback according to history depth; apply event multiplier; floor by on-books and cap by inventory. Sparse histories under 14 days are additionally capped near the fallback.

**Connection.** Forecast fields merge with future on-books fields before pricing.

### Stage I — YoY and pace/event context

**Purpose.** Compare current performance with a valid prior-year reference and expose directional booking context.

**Inputs.** Current and prior frames selected by `select_user_comparison_frames`; repository historical file as fallback; events.

**Outputs.** YoY variance table; pace variance; event name/impact/percentage.

**Files/functions.** `src/yoy.py`, `src/pace.py`, `src/events.py`, `src/pipeline_inputs.py`.

**Algorithm.** Aggregate by stay date; match exact prior-year date first, then latest earlier observation with the same month-day; calculate KPI absolute and percent variances. Pace is current occupancy minus STLY occupancy. Event impacts are low 2%, medium 4%, and high 7% for rule pricing.

**Connection.** Pace and event values enrich the future context used by pricing and tailored logic.

### Stage J — Baseline recommendation

**Purpose.** Provide a simple, independent control model.

**Inputs.** Stay date, derivable occupancy, derivable ADR; optional prior history by day of week.

**Outputs.** Baseline recommendation, adjustment, reason, status, and model identifier.

**Files/functions.** `src/baseline.py`: `generate_baseline_pricing_recommendations`.

**Algorithm.** Select current ADR, else day-of-week median historical ADR, as anchor. Apply +5% at occupancy >= 85%, -5% at occupancy <= 55%, otherwise 0%; clip to configured floor/ceiling. Although the standalone dataclass defaults to 40–600, the application pipeline constructs it with the shared operational defaults 99–399 unless overridden.

**Connection.** It is exported directly and becomes the starting price for the tailored model.

### Stage K — Budget/month forecast context

**Purpose.** Estimate month-end revenue and quantify budget pressure.

**Inputs.** Future context, history, STLY, budget file, as-of date, configuration.

**Outputs.** Actual/on-books revenue, pickup revenue, month-end forecast, budget variance, remaining budget, remaining capacity, forecast remaining occupancy, required ADR.

**Files/functions.** `src/pipeline_budget_forecast.py`, `src/budget.py`.

**Algorithm.** Anchor to the earliest future month; estimate stable inventory with a median; estimate STLY pickup with default close factor 1.15 and pace multiplier clipped to 0.5–2.0; blend projected and STLY revenue with weight 0.35; cap to 110% of STLY revenue and inventory × historical 90th-percentile ADR. If a budget exists, recompute budget progress from daily/monthly targets.

**Connection.** Budget fields are included in elasticity explanations and priority scoring. They do not directly alter tailored recommendations in `tailored.py`.

### Stage L — Elasticity candidate-rate recommendation

**Purpose.** Search a small explainable rate set for maximum modeled revenue.

**Inputs.** Future demand, current rate, rooms on books, inventory, elasticity, event percentage, pricing constraints.

**Outputs.** Recommended rate, expected rooms/revenue, uplift, signal labels, confidence, and explanation.

**Files/functions.** `src/pricing.py`: `simulate_elasticity_pricing`; `src/elasticity.py`.

**Algorithm.** Simulate percentage steps `(-10%, -5%, 0%, +5%, +10%, +15%)`, add event percentage, constrain each candidate, calculate constant-elasticity expected rooms, and select the candidate with greatest modeled revenue.

**Connection.** Exported as `rate_recommendations.csv`; feeds raise/rescue/monitor priority lists. It is a third policy path, separate from the simple baseline and tailored recommendation.

### Stage M — Tailored recommendation

**Purpose.** Modify the baseline using property, strategy, demand, event, seasonality, and competitive median context.

**Inputs.** Future context, baseline result, tailored settings, optional comp set.

**Outputs.** Tailored rate, median source, adjustment, status, confidence, reasoning, warnings.

**Files/functions.** `src/tailored.py`: `build_daily_median_rate_table`, `build_tailored_recommendations`, `build_tailored_summary`.

**Algorithm.** Resolve a daily median anchor; compute clipped demand and strategy indexes; compute an anchor weight; blend baseline toward the median; apply a bounded percentage posture; impose high/low occupancy behavior and property min/max rate limits.

**Connection.** Tailored results are exported, shown in the dashboard, and replayed for intraday comp updates.

### Stage N — Backtesting and evaluation

**Purpose.** Measure historical predictive/recommendation error without confusing it with projected revenue.

**Inputs.** Historical metrics, events, STLY, tailored settings, as-of date.

**Outputs.** Demand forecast MAE/RMSE/MAPE/directional accuracy; subgroup metrics; historical ADR recommendation MAE/RMSE; projected elasticity-policy uplift.

**Files/functions.** `src/forecast.py`, `src/evaluation.py`, `src/pipeline_reporting.py`.

**Algorithms.** Expanding-window chronological demand backtest; baseline-versus-calibrated forecast comparison; full-history ADR recommendation comparison; constant-elasticity policy scenario. These are different evaluations and must not be merged into one claim.

### Stage O — Reporting and dashboard

**Purpose.** Make the workflow inspectable and portable.

**Inputs.** All stage outputs.

**Outputs.** Timestamped CSV/TXT/PNG artifacts, ZIP download, Streamlit tabs and charts.

**Files/functions.** `src/pipeline_outputs.py`, `src/pipeline_reporting.py`, `ui/results_panel.py`, `ui/chart_helpers.py`.

**Connection.** This is the terminal stage; no recommendation is automatically pushed to a selling system.

---

# 3. Source Code Walkthrough

## 3.1 Root entry points

| File | Purpose and design | Principal inputs/outputs and interactions |
|---|---|---|
| `app.py` | Streamlit entry point. Detects Streamlit execution, saves uploads to a run-scoped upload area, renders pricing controls, initializes session state, validates tailored settings, calls `run_pipeline`, and sends paths/summary to results UI. | Inputs: UI session and uploads. Output: interactive dashboard. Depends on `main.run_pipeline` and all `ui.*` panels. It contains presentation orchestration, not domain formulas. |
| `main.py` | Authoritative end-to-end application service and CLI. `run_pipeline` executes ingestion through reporting. `parse_arguments`, `_print_summary`, and `main` expose CLI behavior and error codes. | Inputs: historical path plus optional future/budget/events and config. Outputs: path dictionary and summary dictionary. Calls every major domain pipeline. |

## 3.2 Core domain modules

| File | Functions and responsibility | Inputs → outputs / design decisions |
|---|---|---|
| `src/schema.py` | `find_column_match`, `auto_map_columns`, `get_missing_required_columns`, `interactive_column_mapping`, `apply_column_mapping`. Defines canonical schema and aliases. | Vendor columns → canonical dataframe. Exact normalized alias matching is intentionally explainable; manual mapping is the fallback. |
| `src/ingest.py` | Report-header discovery, aligned CSV parsing, artifact removal, currency repair, CSV/Excel loading, mapping, date parsing, numeric conversion, normalization, `process_file`, `process_dataframe`. | PMS export → normalized canonical dataframe. Multiple controlled parser fallbacks improve compatibility; permissive malformed parsing warns about skipped rows. |
| `src/validate.py` | `ValidationResult`, `validate_data`, `check_data_quality`, report writer, YoY field coverage check. | Normalized rows → retained valid rows + issue ledger. Critical validity and recoverable rate imputation are separated. |
| `src/metrics.py` | `calculate_daily_metrics`, `export_metrics`. | Canonical rows → occupancy, ADR, RevPAR dataframe/CSV. Zero denominators produce zero. |
| `src/yoy.py` | Current aggregation, prior normalization, exact/calendar fallback matching, variance calculation, `summarize_yoy`. | Current + prior frames → row-level and summary YoY evidence with alignment/status provenance. |
| `src/pace.py` | `_daily_base`, repository historical loader, `calculate_pace_analysis`. | Current and STLY → occupancy pace flags. Repository historical files are deduplicated by date; pace retains missing STLY as `NaN`. |
| `src/events.py` | Validate event file and keep the highest-impact event per date; merge event percentage. | Event CSV + stay dates → event context. Fixed low/medium/high mapping supports transparent sensitivity. |
| `src/forecast.py` | Feature frame, baseline forecast, enhanced model, calibration-gated tailored forecast, backtest splits/windows, future forecast. | History/future/context → demand predictions and chronological evaluation rows. Short-history paths fall back rather than fit unstable ML. |
| `src/baseline.py` | Baseline input checks, KPI derivation, day-of-week ADR anchor, baseline pricing. | Current inputs + history → simple once-daily control recommendation. It intentionally ignores RMS-only contextual features. |
| `src/elasticity.py` | `expected_rooms_sold`. | Demand/rate/elasticity/on-books/capacity → bounded expected rooms. One formula isolates the pricing assumption for testing. |
| `src/pricing.py` | Rule pricing, elasticity search, signal/confidence labels, priority lists. | Pace/future/budget context → rule recommendation or elasticity policy recommendation and ranked dates. Candidate enumeration is auditable. |
| `src/tailored.py` | Settings schema/validation, median-rate inference and provenance, property/segment presets, demand/strategy indexes, tailored rate and summary. | Future + baseline + comp/settings → date-level tailored outputs. The module deliberately produces explanations and warnings with the rate. |
| `src/budget.py` | Budget-format detection, monthly-to-daily expansion, required ADR, budget progress. | Monthly/daily budget + history/forecast → daily targets and budget metrics. Monthly weighting uses historical day-of-week revenue where possible. |
| `src/intraday.py` | Intraday update validation and sequential replay. | Timestamped comp updates → old/new tailored rates and change log. Replay is deterministic and ordered by stay date/timestamp. |
| `src/evaluation.py` | Forecast error metrics, leakage warning, comparison/subgroup metrics, rate backtest, plots, policy output. | Actual/predicted/recommendation frames → MAE/RMSE/MAPE and chart files. Forecast and rate evaluations remain separate. |
| `src/dataset_manager.py` | Save/load/list/delete named datasets and budget profiles with metadata/mappings/settings. | Dataframes/settings ↔ CSV/JSON filesystem persistence. Names are sanitized; optional stale files are removed when saving. |
| `src/utils.py` | Directory, timestamp, formatting, path, printing, safe division, truncation helpers. | Cross-cutting utilities; no domain decision algorithm. |
| `src/__init__.py` | Package marker and package docstring. | Defines no runtime functions or decisions. |

## 3.3 Application pipeline modules

| File | Responsibility | Interaction |
|---|---|---|
| `src/pipeline_config.py` | Converts free-form config to immutable `PipelineRuntimeConfig`, `PricingConfig`, and `BaselinePricingConfig`. | Centralizes defaults and numeric coercion before orchestration. |
| `src/pipeline_inputs.py` | Prepares future data, normalizes uploaded STLY, and chooses adjacent-year comparison frames. | Prevents same-year data from being used as prior-year forecast context. |
| `src/pipeline_budget_forecast.py` | Encapsulates month forecast and budget calculations in `MonthForecastBudgetContext`. | Receives future/history/STLY and returns all month/budget intermediates. |
| `src/pipeline_outputs.py` | Creates timestamped directory, writes elasticity/tailored/priority outputs, collects public paths, builds summary payload. | Separates filesystem/report assembly from `main.run_pipeline`. |
| `src/pipeline_reporting.py` | Builds elasticity policy comparison, writes forecast/rate evaluation artifacts, and writes chart outputs. | Connects `forecast`, `evaluation`, `pricing`, and `elasticity`. |

## 3.4 User-interface modules

| File | Purpose |
|---|---|
| `ui/upload_panel.py` | Reads uploads, merges saved and automatic mappings, renders required/optional file and mapping controls. |
| `ui/dataset_panel.py` | Loads project-relative demo files; applies saved datasets; supports property/dataset switching and persistence. |
| `ui/budget_panel.py` | Coerces uploaded/manual budgets, builds a daily template, renders monthly/daily budget controls and saved profiles. |
| `ui/tailored_panel.py` | Manages tailored session keys, property/segment settings, comp-rate modes, daily median editor, update timestamps, and future preview. |
| `ui/results_panel.py` | Reads outputs safely, renders forecast/baseline/tailored/YoY/budget/backtest tabs, builds ZIP bytes, formats model metrics. |
| `ui/chart_helpers.py` | Altair/Streamlit line and bar chart helpers plus static image display. |
| `ui/__init__.py` | Package marker. |

## 3.5 Test files

The test suite covers baseline rules, backtest windows, demo integration, elasticity, model comparison, history-depth behavior, intraday replay, KPI formulas, budget context, input-frame selection, schema/ingestion repairs, tailored settings/presets/median precedence, validation, and YoY logic. Root `test_hand_checked.py` and `test_end_to_end.py` are executable verification scripts but do not define pytest test functions; the 71 passing pytest cases arise from `tests/`, including parametrized cases.

---

# 4. Data Pipeline

## 4.1 Import and file-format handling

CSV handling is layered because PMS exports are not reliable rectangular datasets. RateAnchor first attempts a row-alignment parser that can insert a missing blank placeholder after the date and remove extra blanks adjacent to currency. If no report header is found, it uses pandas and then delimiter/quote fallbacks. Excel is first read with `header=None`; a likely header row is detected by the presence of `date` and a room metric.

This protects research validity by reducing silent column shifts. The permissive final CSV fallback is not equivalent to strict validity: it warns that malformed rows may have been skipped, so the validation report and row counts must be reviewed.

## 4.2 Column mapping

The mapping contract is canonical-to-source. Historical data normally requires date, inventory, rooms sold, and revenue. Future data requires date and rooms sold, plus inventory unless manually supplied. Automatic mapping is deterministic exact alias matching; it does not use fuzzy semantic inference. User or saved mappings override automatic choices only when the source column exists.

Revenue and ADR receive additional scale/fill heuristics because printed PMS exports can split values across unnamed columns. Mapping safeguards reject one source column being used for both occupancy and revenue, and reject duplicate sources across required fields.

## 4.3 Cleaning and normalization

The following transformations occur before business validation:

1. Promote embedded metric header.
2. Make duplicate/blank column names unique.
3. Drop empty rows/columns.
4. Remove repeated headers and totals.
5. Repair shifted revenue and ADR values.
6. Rename to canonical columns.
7. Add missing optional columns as null.
8. Parse dates with coercion.
9. Remove `$`, commas, `%`, and spaces from numerics.
10. Convert rooms to integers; keep rate/revenue nulls for validation.
11. Derive missing inventory from occupancy when possible and record provenance.

There is no general statistical normalization such as z-scoring or min-max scaling in the operational pipeline. The word “normalization” refers to schema and type standardization.

## 4.4 Feature engineering

Engineered fields include:

- occupancy, occupancy percent, ADR, and RevPAR;
- day of week, month, weekend flag, and day type;
- event flag/factor and pricing event percentage;
- STLY occupancy and pace variance;
- on-books rooms, seasonal base demand, history trust, and forecast method;
- forecast rooms sold and occupancy;
- month capacity, pickup, budget variance, and required ADR;
- property/segment preset biases;
- demand index, strategy index, median source, anchor weight, and tailored adjustment;
- compression, under-pacing, event, budget, and priority scores.

## 4.5 Missing values

Missingness is handled by field meaning:

- invalid/missing date: critical, row removed;
- missing past revenue: critical, row removed;
- missing future revenue: allowed because an on-books report can omit realized revenue;
- missing current rate: ADR → revenue/rooms → median/default fallback;
- missing rooms available: derive from sold/occupancy or use manual override;
- missing STLY: preserve `NaN`, mark historical unavailable, fall back to current-only logic;
- missing comp median: baseline-centered tailored recommendation with warning;
- missing events/budget/comp set: optional branch is skipped or defaulted;
- missing enough history: baseline/seasonal fallback, not forced ML.

## 4.6 Duplicate handling

Duplicate stay dates in the primary validated dataframe are warned about but not removed because they may represent room types or segments. KPI functions operating directly on that dataframe do not aggregate before calculation, while YoY and pace explicitly aggregate by stay date. Repository historical files are deduplicated across files by date before STLY aggregation. Event duplicates retain the highest impact. Tailored median settings reject duplicate dates. Intraday duplicate date/timestamp slots generate validation errors.

This context-dependent behavior is intentional but is also a limitation: a duplicated full-property daily row could be mistaken for legitimate segmentation and double-counted downstream.

## 4.7 Historical and YoY matching

When uploaded files are adjacent years, the later future file becomes “current” YoY and the historical upload becomes prior. Forecast backtesting does not use that same historical file as STLY context for itself; it uses a repository reference only if it is genuinely older. YoY alignment tries `current_date - 1 year`, then same month-day from the latest earlier year. Each row records `exact_prior_date`, `calendar_date_fallback`, or `no_prior_match`.

## 4.8 Budget handling

A daily budget has `stay_date,budget_revenue`; a monthly budget has `year,month,budget_revenue`. Monthly targets are distributed equally or by historical day-of-week revenue weights. A budget with no row for the target month generates a non-blocking month/year warning; malformed budget structure remains an error.

## 4.9 Fallback chain

```text
Inventory:
source inventory -> manual override -> sold / occupancy -> invalid/zero

Current rate:
source current_rate -> ADR -> revenue / rooms_sold
-> positive median/default -> 120.00

Competitive median by date:
manual daily -> comp-set/dataset daily median
-> global monthly fallback -> missing-with-warning

Demand model:
strong seasonal history -> blended moderate history
-> sparse-history fallback -> on-books floor

Calibrated backtest model:
validated candidate improvement -> chosen candidate
-> baseline if no candidate improves -> baseline for short history
```

---

# 5. Data Validation

## 5.1 Validation rule matrix

| Rule | Rationale | Failure handling | Effect on recommendation |
|---|---|---|---|
| `stay_date` parses and is non-null | All matching, forecasting, and daily pricing is date-keyed. | `NULL_DATE`; row removed. | No recommendation for that record. |
| `rooms_available >= 0` | Negative capacity is physically invalid. | `NEGATIVE_VALUE`; row removed. | Excluded from KPIs/forecast. |
| `rooms_sold >= 0` | Negative demand is invalid. | `NEGATIVE_VALUE`; row removed. | Excluded. |
| `rooms_sold <= rooms_available` unless allowed | Prevents accidental overbooking/column mis-map. | `OVERBOOKING`; row removed unless flag enabled. | Excluded or retained under explicit policy. |
| revenue non-negative | Negative room revenue invalidates ADR/RevPAR in this model. | `NEGATIVE_VALUE`; row removed. | Excluded. |
| missing revenue allowed only for future date | Future on-books reports need not contain realized revenue. | `MISSING_REVENUE_PAST_DATE`; past row removed. | Future recommendation remains possible. |
| current rate positive or recoverable | Pricing needs a rate reference. | Derive from ADR or revenue/sold; otherwise fill fallback and log `CURRENT_RATE_FILLED`. | Row retained, but fallback provenance is in validation report. |
| required mapping completeness | Prevents semantic ambiguity before validation. | `ValueError` before model execution. | Entire run stops until mapping is supplied. |
| unique required mapping sources | Prevents same values serving incompatible meanings. | `ValueError`. | Entire run stops. |
| tailored factors in [0,2] | Bounds user sensitivities/priorities. | User-facing settings error; tailored run stops. | No tailored output until corrected. |
| min/max rate positive and ordered | Enforces coherent guardrails. | Settings error. | No tailored output. |
| median/comp rates positive | Avoids invalid anchor. | Invalid setting row rejected or ignored; warning/fallback. | Uses next anchor source or baseline. |
| event schema and impact level valid | Ensures deterministic event adjustment. | `ValueError`. | Run stops for malformed supplied event file. |
| budget schema detectable | Ensures correct month/daily interpretation. | `ValueError`; only missing target month is softened to warning. | Budget branch omitted or run stops. |

## 5.2 Critical versus advisory issues

`ValidationResult.passed` becomes false for every issue, including recoverable `CURRENT_RATE_FILLED`, but only rows marked with critical `has_issues` are removed. `check_data_quality` separately prints advisory counts for zero inventory, zero sold, zero revenue, date range, and duplicate dates. These advisories are not added to the saved issue ledger.

## 5.3 Invalid-record propagation

Historical invalid rows are removed before KPI and forecast training. Future invalid rows are removed before forecast/recommendation generation. Therefore, validation changes both sample size and model inputs. If every historical row is removed, `main.run_pipeline` raises a diagnostic error with the three most frequent issue types. There is no implicit continuation with an empty training set at that point.

## 5.4 Research-validity contribution

Validation supports construct validity by ensuring that “occupancy,” “ADR,” and “revenue” are computed from semantically distinct, physically plausible fields. It supports internal reproducibility by writing a row-level issue ledger. It does not establish external validity: passing validation means data conforms to the prototype's rules, not that it represents all hotels or that missingness is random.

## 5.5 Validation limitations

- Counts coerced from unparseable text become zero before validation, so parse failure can appear as a plausible zero.
- Duplicate dates are warnings only.
- A default rate can keep a future row alive but can bias pricing.
- Allowing overbooking is a global flag rather than a property/date-specific policy.
- No validation checks currency, tax inclusion, net/gross revenue, time zone, snapshot lead time, or whether future rooms sold and revenue share the same cutoff.

---

# 6. KPI Calculations

Let, for date or record `t`:

- `A_t` = rooms available;
- `S_t` = rooms sold;
- `R_t` = room revenue;
- `O_t` = occupancy ratio;
- `ADR_t` = average daily rate;
- `RevPAR_t` = revenue per available room.

## 6.1 Occupancy

```text
O_t = S_t / A_t, if A_t > 0; else 0
OccupancyPercent_t = 100 * O_t
```

Implementation: `src/metrics.calculate_daily_metrics`. Purpose: measure inventory utilization and drive baseline thresholds, forecast occupancy, tailored demand, and priority scoring.

Example: `A=100`, `S=80` gives `O=0.80`, or 80%.

Effect: baseline increases at >=85% and decreases at <=55%; tailored demand centers occupancy around 72%; elasticity and priority use forecast occupancy.

## 6.2 ADR

```text
ADR_t = R_t / S_t, if S_t > 0; else 0
```

Example: `R=$12,000`, `S=80` gives `ADR=$150`.

Effect: serves as current-rate fallback, forecasted revenue component, baseline anchor, actual rate in rate backtesting, and tailored fallback.

## 6.3 RevPAR

```text
RevPAR_t = R_t / A_t, if A_t > 0; else 0
           = ADR_t * O_t when definitions share the same aggregation
```

Example: `$12,000 / 100 = $120`; equivalently `$150 × 0.80 = $120`.

Effect: exposed as a KPI and a user priority input. In the tailored model, `revpar_priority` affects strategy index, not a separately optimized RevPAR objective.

## 6.4 Rooms sold and revenue

Rooms sold and room revenue are source measures, aggregated by sum where YoY/pace requires one daily row. For modeled candidate rate `p`:

```text
ExpectedRevenue_t(p) = p * ExpectedRoomsSold_t(p)
```

## 6.5 YoY variances

For any metric `X`:

```text
AbsoluteVariance_X = X_current - X_prior
PercentVariance_X  = 100 * (X_current - X_prior) / X_prior,
                     only when X_prior != 0; else NaN
```

Occupancy absolute variance is expressed in percentage points because current and prior occupancy-percent values are subtracted. Example: 80% versus 75% gives +5 percentage points and `6.667%` relative variance.

The project computes these for occupancy, ADR, RevPAR, rooms sold, and revenue.

## 6.6 Pace variance

```text
PaceVariance_t = CurrentOccupancy_t - STLYOccupancy_t
```

Positive means over-pacing; negative means under-pacing. This is occupancy difference, not booking-curve pickup at matched lead time unless the input snapshots were produced at matched cutoffs.

## 6.7 Budget variance

```text
MonthEndForecast = ActualRevenueToDate + ForecastRevenueRemaining
VarianceToBudgetAbs = MonthEndForecast - MonthlyBudget
VarianceToBudgetPct = 100 * VarianceToBudgetAbs / MonthlyBudget
```

Example: forecast `$310,000`, budget `$300,000` gives `+$10,000`, `+3.33%`.

## 6.8 Required ADR remaining

```text
ForecastRemainingOcc = ForecastRemainingRoomsSold / RemainingRoomsAvailable
RequiredADRRemaining = RemainingBudget /
                       (RemainingRoomsAvailable * ForecastRemainingOcc)
```

Returns zero if the budget is already met, capacity is zero, or forecast occupancy is non-positive.

## 6.9 Forecast variance and error

The code does not define a field literally named `forecast_variance`; it defines residual/error `e_i = prediction_i - actual_i`, plus:

```text
MAE  = (1/n) * sum |e_i|
RMSE = sqrt((1/n) * sum e_i^2)
MAPE = 100 * mean(|e_i| / |actual_i|), excluding actual_i = 0
DirectionalAccuracy = 100 * mean(sign(Delta actual) = sign(Delta predicted))
```

MAE is interpretable in rooms or dollars/rate units. RMSE penalizes large misses more strongly. MAPE is scale-relative but unstable around zero. Directional accuracy tests movement direction, not magnitude.

## 6.10 Custom operational KPIs

```text
CompressionScore = max((ForecastOcc - 0.80) * 5, 0)
UnderPacingScore = max((TargetOcc - ForecastOcc) * 5, 0)
EventScore = {none:0, low:0.5, med:1.0, high:1.5}
BudgetScore = clip(|BudgetGap| / 50,000, 0, 2)

PriorityScore = 1.2*CompressionScore
              + 1.0*UnderPacingScore
              + 1.0*EventScore
              + 1.3*BudgetScore
```

This ranks attention; it is not a probability or statistically calibrated score.

---

# 7. Baseline Recommendation Model

## 7.1 Role as the control model

The baseline is `simple_once_daily_rule_based_v1`. It is appropriate as a control because it is deterministic, low-dimensional, explainable, and independent of RateAnchor-only signals. A unit test explicitly verifies that additional RMS features do not change it. This isolates the incremental behavior of the tailored layer.

It should not be described as an industry-standard optimizer. It is a project-defined occupancy rule comparator.

## 7.2 Inputs and derivation

Required: `stay_date`, a derivable occupancy, and a derivable ADR.

```text
occupancy := supplied occupancy
             else rooms_sold / rooms_available

current_ADR := supplied ADR
               else current_rate
               else room_revenue / rooms_sold

historical_ADR_dow := median historical ADR by weekday
rate_anchor := current_ADR else historical_ADR_dow
```

## 7.3 Decision thresholds and rate calculation

Pipeline defaults from `PipelineRuntimeConfig`:

| Condition | Adjustment | Reason |
|---|---:|---|
| occupancy >= 0.85 | +5% | high occupancy |
| occupancy <= 0.55 | -5% | low occupancy |
| otherwise | 0% | moderate occupancy |

```text
baseline_rate = clip(rate_anchor * (1 + adjustment), rate_floor, rate_ceiling)
```

The standalone baseline dataclass defaults to 40–600, but `build_pipeline_config` passes the application's shared `rate_floor` and `rate_ceiling`, which default to 99–399. Thus the normal CLI/Streamlit pipeline uses 99–399 for both baseline and elasticity pricing unless configuration overrides those fields.

## 7.4 Guardrails and output

If occupancy or anchor is missing/non-positive, the status is `UNAVAILABLE` and no rate is produced. Otherwise the rate is clipped to floor/ceiling. Output includes current occupancy, current ADR, recommended rate, dollar adjustment, percentage adjustment, reason, model type, and status.

## 7.5 Pseudocode

```text
function baseline(input, historical, config):
    require stay_date and derivable occupancy and ADR
    derive occupancy and current ADR for every date
    compute median historical ADR by weekday

    for date in chronological input:
        anchor = current ADR else weekday historical median
        if occupancy or anchor invalid:
            output UNAVAILABLE
            continue

        if occupancy >= high threshold:
            delta = +5%
        else if occupancy <= low threshold:
            delta = -5%
        else:
            delta = 0%

        recommendation = clip(anchor * (1 + delta), floor, ceiling)
        emit recommendation, delta, reason, OK
```

## 7.6 Design implications

The baseline often scores well in historical ADR backtesting because actual ADR is itself used as its anchor and the recommendation is only ±5% away. This is not a genuinely out-of-sample price forecast. It is a contemporaneous policy-distance test and must be interpreted accordingly.

---

# 8. Tailored Recommendation Model

## 8.1 Inputs

### Date-level operational inputs

- stay date;
- occupancy or forecast occupancy or sold/available-derived occupancy;
- current rate/ADR/revenue-derived ADR;
- baseline recommendation;
- rooms sold and room revenue;
- pace variance;
- event percentage and impact level;
- date-level competitor/dataset median and provenance.

### Property settings

- property type;
- segment focus;
- baseline occupancy sensitivity;
- ADR sensitivity;
- RevPAR priority;
- rooms-sold priority;
- revenue priority;
- demand adjustment factor;
- seasonality adjustment factor;
- event impact factor;
- minimum acceptable and maximum recommended rates;
- comp-rate input mode, fallback, daily values, timestamps, and review frequency.

`adr_sensitivity` is validated and persisted but is not used in `build_tailored_recommendations`. This is an implementation gap, not an active weight.

## 8.2 Competitive median precedence

For daily mode:

```text
manual daily median
    > median of supplied comp-set rates for date
      (or median of current_rate/ADR/recommended/derived rates)
    > global monthly fallback
    > missing
```

For monthly mode, date-level manual and dataset medians are bypassed, so the global fallback is used. Each date records source and timestamp. Missing, dataset-derived, fallback, and stale fallback conditions generate warnings.

## 8.3 Property presets

Property presets add demand, strategy, anchor, and rate biases and scale seasonality, events, and movement.

| Type | Character of implemented posture |
|---|---|
| Full Service | mild revenue confidence; unit multipliers |
| Limited Service | occupancy-aware, lower anchor/rate/movement |
| Select Service | restrained movement, slightly negative demand/rate bias |
| Luxury | strong premium/comp anchor and event confidence |
| Resort | stronger seasonality and event compression response |
| Boutique | premium positioning and comp influence |
| Extended Stay | slower, steadier, reduced event/seasonality response |
| Economy | strongest conversion/price-sensitivity posture |

Exact preset constants are in `PROPERTY_TYPE_PRESETS` and are included in the formula below. Unknown types fall back to Full Service.

## 8.4 Segment presets

Balanced, Revenue, Occupancy, Corporate, Group, Premium, and Leisure presets alter demand/strategy/anchor/rate biases and movement. For example, Revenue uses strategy bias `+0.14`, anchor bias `+0.08`, rate bias `+0.012`, and movement multiplier `1.10`; Occupancy uses `-0.20`, `-0.06`, `-0.014`, and `0.90`. Unknown segment focus falls back to Balanced.

## 8.5 Seasonality and event indexes

```text
SeasonalityIndex(date):
    +0.25 in Jun, Jul, Aug, Dec
    -0.15 in Jan, Feb
    +0.05 otherwise
    +0.08 on Friday or Saturday

EventIndex = event_pct + {low:0.05, medium:0.10, high:0.18, none:0}
```

This means event context can enter twice: `event_pct` from `events.py` plus an impact-level signal. That is a deliberate implemented compound signal, though it may overstate events without empirical calibration.

## 8.6 Demand index

Let `P` be the property preset and `G` the segment preset:

```text
DemandIndex = clip(
    (occupancy - 0.72) * 0.80 * occupancy_sensitivity
  + pace_variance * 0.60 * demand_adjustment_factor
  + seasonality_index * 0.40 * seasonality_factor * P.seasonality_multiplier
  + event_index * 0.45 * event_factor * P.event_multiplier
  + P.demand_bias
  + G.demand_bias,
  -0.50, +0.60)
```

Missing occupancy becomes 0.0 for this calculation; missing pace/event becomes zero. Centering at 72% encodes a project assumption, not an estimated property optimum.

## 8.7 Strategy index

```text
StrategyIndex = clip(
    (revenue_priority - 1) * 0.55
  + (revpar_priority - 1) * 0.35
  - (rooms_sold_priority - 1) * 0.45
  + P.strategy_bias
  + G.strategy_bias,
  -0.35, +0.35)
```

Revenue/RevPAR priorities push rate posture upward; rooms-sold priority pushes toward occupancy.

## 8.8 Median anchoring

When a median exists:

```text
AnchorWeight = clip(
    0.45
  + 0.25*StrategyIndex
  + 0.30*DemandIndex
  + P.anchor_bias
  + G.anchor_bias,
  0.15, 0.95)

AnchoredRate = BaselineRate
             + (MedianRate - BaselineRate) * AnchorWeight
```

Without a median, `AnchoredRate = BaselineRate`. A baseline-median difference greater than 20% of median generates a warning.

## 8.9 Percentage adjustment and guardrails

```text
AdjustmentPct = (
    0.12*DemandIndex
  + 0.05*StrategyIndex
  + P.rate_bias
  + G.rate_bias)
  * P.movement_multiplier
  * G.movement_multiplier

TailoredRate_unbounded = AnchoredRate * (1 + AdjustmentPct)
```

Additional rules:

```text
if occupancy >= 0.90 and median exists:
    TailoredRate = max(TailoredRate, 1.02 * MedianRate)
else if occupancy <= 0.55 and median exists:
    TailoredRate = min(TailoredRate, 0.99 * MedianRate)

TailoredRate = clip(TailoredRate, property_min_rate, property_max_rate)
```

Unlike elasticity pricing, tailored pricing has no explicit maximum percentage move from baseline/current rate. Its protection is indirect through anchor/index clips and the absolute property min/max.

## 8.10 Status and confidence

- no baseline or ADR: `REVIEW_REQUIRED`;
- no median: `WARNING_MISSING_MEDIAN`;
- stale global fallback: `WARNING_STALE_MEDIAN`;
- otherwise `OK` if no warnings, else `WARNING`.

Confidence is low if baseline and both median/ADR completeness fail; medium when there are two or more warnings; otherwise high. This is a rule label, not calibrated predictive confidence.

## 8.11 Budget logic and priority scoring

Budget is not an input to `build_tailored_recommendations`. It influences the separate elasticity output's explanation and the priority ranking. Therefore, the statement “the tailored rate is budget optimized” would be false for this repository. RateAnchor is budget-aware at workflow/reporting level, not directly in the tailored rate equation.

## 8.12 Complete pseudocode

```text
function tailored(future, baseline, settings, comp_set):
    settings = validate_and_sanitize(settings)
    merge future with baseline by stay_date
    derive occupancy, ADR, revenue, RevPAR
    medians = resolve_date_level_medians(comp_set, dataset, manual, global)

    for each stay_date:
        warnings = median/source/staleness/range checks
        baseline_rate = baseline recommendation else ADR
        if baseline_rate unavailable:
            emit REVIEW_REQUIRED
            continue

        demand = clipped weighted occupancy + pace + season + event
                 + property bias + segment bias
        strategy = clipped revenue + RevPAR - rooms-sold priorities
                   + property bias + segment bias

        if median exists:
            weight = clipped base + demand + strategy
                     + property anchor bias + segment anchor bias
            anchored = baseline + weight * (median - baseline)
        else:
            anchored = baseline

        adjustment = weighted demand + strategy + preset rate biases
        adjustment *= property movement * segment movement
        rate = anchored * (1 + adjustment)

        enforce high-occupancy median floor or low-occupancy median ceiling
        clip to property min/max
        derive status, confidence, explanation, warning text
        emit result and provenance
```

---

# 9. Mathematical Formula Catalog

| Formula | Implementation purpose |
|---|---|
| `O=S/A` | occupancy |
| `ADR=R/S` | realized average sold-room rate |
| `RevPAR=R/A=ADR*O` | revenue efficiency per available room |
| `Revenue=Rate*RoomsSold` | realized or modeled room revenue |
| `YoY_abs=X_c-X_p` | absolute comparison |
| `YoY_pct=100*(X_c-X_p)/X_p` | relative comparison, prior nonzero |
| `Pace=O_current-O_STLY` | occupancy pace signal |
| `history_trust=clip(unique_history_days/28,0,1)` | seasonal/fallback blend |
| `seasonal_base=mean(DOW_mean,month_mean)` | future base demand |
| `base=trust*seasonal+(1-trust)*fallback` | history-depth demand estimate |
| `event_forecast=base*{1.02,1.05,1.09}` | future event demand |
| `forecast_sold=min(capacity,max(on_books,event_forecast))` | feasible demand forecast |
| `baseline_rate=clip(anchor*(1+delta),floor,ceiling)` | control price |
| `expected_demand=D0*(p/p0)^(-epsilon)` | constant-elasticity response |
| `expected_sold=clip(expected_demand,on_books,capacity)` | feasible price response |
| `expected_revenue=p*expected_sold` | candidate objective |
| `DemandIndex` equation in §8.6 | tailored demand posture |
| `StrategyIndex` equation in §8.7 | user/property objective posture |
| `AnchorWeight` and `AnchoredRate` in §8.8 | comp-rate blending |
| `TailoredRate` in §8.9 | final tailored calculation |
| `PriorityScore` in §6.10 | operational attention rank |
| `RemainingBudget=Budget-ActualToDate` | budget gap |
| `RequiredADR=RemainingBudget/ForecastRemainingRoomsSold` | rate needed on forecast pickup |
| `Variance=Forecast-Budget` | budget outcome gap |
| `MAE=mean(|prediction-actual|)` | typical absolute error |
| `RMSE=sqrt(mean(error^2))` | error with greater penalty for large misses |
| `MAPE=100*mean(|error|/actual)` | relative error excluding zero actuals |
| `DirectionalAccuracy=100*mean(sign(Delta a)=sign(Delta p))` | direction-of-change accuracy |
| `CalibrationScore=.3*(MAE/MAE_baseline)+.7*(RMSE/RMSE_baseline)` | validation-gated forecast selection |

The 30/70 calibration weighting favors avoidance of large errors. A candidate is selected only when its score is below 1.0; ties remain baseline.

---

# 10. Simulations

## 10.1 What the repository actually simulates

There are three distinct simulation/evaluation mechanisms:

1. **Future demand simulation:** estimates final rooms sold from history/on-books/events.
2. **Elasticity price simulation:** enumerates candidate rates and maximizes modeled revenue under a constant-elasticity curve.
3. **Historical backtesting:** compares demand predictions chronologically and compares baseline/tailored recommendations with actual ADR.

“Monthly,” “quarterly,” and “annual” are dataset horizons, not separate algorithms. The same pipeline changes behavior with history depth and the expanding-window evaluation.

## 10.2 Monthly evidence

### Dataset

The saved July datasets contain 31 historical rows for July 2025 and 31 future rows for July 2026, with a manual 99-room inventory setting. A separate February run contains 28 cleaned historical days. Generated July runs validate only 31 historical rows; the current backtest implementation uses expanding windows once enough data exists.

### Current verified output (July run `run_20260713_124040`)

| Evaluation | Baseline | Tailored | Winner |
|---|---:|---:|---|
| Demand forecast MAE, 10 backtest rows | 9.8083 rooms | 13.1951 rooms | Baseline |
| Demand forecast RMSE | 11.5438 | 14.9072 | Baseline |
| ADR recommendation MAE, 31 rows | $4.7897 | $7.7340 | Baseline |
| ADR recommendation RMSE | $7.0698 | $9.3025 | Baseline |

### Interpretation

The calibrated forecast requires at least 14 training rows and reserves a 30%-bounded calibration tail; a 31-day series supplies little recurring seasonal evidence. Candidate corrections selected on a few days can fail on later days. The baseline's weekday mean has lower variance. For rate error, the baseline is mechanically close to actual ADR because it uses actual/current ADR as the anchor, while tailored comp/property adjustments move farther away.

This is preliminary descriptive evidence, not statistical proof. The demand backtest has 10 evaluated rows in this run.

## 10.3 Larger-period evidence

### Dataset

The saved “Q2 2026 and 2025” source contains 95 physical rows but only 92 unique stay dates from 1 March through 31 May. The latest generated run retains 65 valid historical rows spanning 1 March through 29 May. Its combined historical/future validation report records 184 processed rows, 153 valid rows, and 31 removed overbooking rows under the 99-room manual inventory; 27 of those removals occur in the historical side and four in the future side. Report artifacts/duplicate physical rows account for the earlier 95-to-92 reduction. Legacy July–September outputs contain up to 92 cleaned rows and 71 rolling backtest rows.

### Current verified output (`run_20260713_124425`)

| Evaluation | Baseline | Tailored | Difference |
|---|---:|---:|---:|
| Demand MAE, 44 rolling rows | 16.1266 | 11.6459 | Tailored improves 4.4807 rooms (27.78%) |
| Demand RMSE | 20.0851 | 15.3432 | Tailored improves 4.7420 rooms (23.61%) |
| ADR recommendation MAE, 65 rows | $2.7804 | $6.8505 | Tailored is worse by $4.0701 |
| ADR recommendation RMSE | $4.0832 | $10.7450 | Tailored is worse by $6.6617 |

The projected elasticity-policy uplift in `evaluation_metrics.csv` is `$9,734.14`; this is a scenario-model sum under assumed elasticity, not realized revenue and not the tailored ADR backtest outcome.

### Why tailored demand behaves differently with more data

At 45+ training rows, `calibrated_tailored_forecast` can test random forest, baseline/enhanced blends, STLY candidates, and smoothed subgroup residual corrections. The candidate is retained only if its calibration score beats baseline. Larger histories also support repeated expanding-window evaluation. More observations expose weekday/month/day-type structure and allow shrinkage estimates to be less dominated by one day.

This is why the tailored **demand forecast** can improve on larger data. It does not imply the tailored **rate recommendation** improves; the current evidence shows the opposite on ADR distance.

## 10.4 Quarterly simulation status

The Q2-labeled source covers roughly a quarter, but the latest validated historical run contains 65 rows rather than a complete 92-day quarter. It is therefore most accurate to call it a larger multi-month period. The algorithm, inputs, outputs, and metrics are those above. A final defense should disclose the validation attrition and show the validation report.

## 10.5 Annual simulation status

No annual historical/future dataset, annual output folder, or annual metric artifact exists in the inspected repository. Consequently:

- number of annual observations: not available;
- properties used: not evidenced;
- actual annual outcome: not available;
- baseline-versus-tailored annual claim: not supportable.

An annual study can be run without changing the core code, but it requires a date-complete input, validation audit, a preserved configuration manifest, and a predeclared evaluation protocol.

## 10.6 Expected versus actual outcomes

The design expectation is that more history expands the set of calibrated candidates and can reduce demand error. The actual latest demand evidence agrees. The stronger thesis claim that tailored pricing universally beats baseline does not agree with the latest ADR backtest. The defensible conclusion is conditional: tailored forecast calibration can help on a larger history; tailored price adjustments require empirical calibration against future realized outcomes.

---

# 11. Three Hotel Types

## 11.1 Requested research design

The requested types—select-service, full-service, and a 13-room independent hotel—create a useful maximum-variation design:

| Type | Operational characteristics | Expected revenue strategy |
|---|---|---|
| Select-service | standardized room product, fewer ancillary services, moderate inventory, lean staffing | restrained price movement, occupancy awareness, competitor parity |
| Full-service | larger service bundle, group/corporate/leisure mix, more segmentation, ancillary revenue | stronger rate/RevPAR posture, event and segment differentiation |
| 13-room independent | extremely small inventory, lumpy occupancy, a single booking changes occupancy by 7.69 points, limited history | manual oversight, strong guardrails, local-event/competitor judgment, robust methods |

A 13-room property makes discrete inventory behavior explicit: one room is `1/13 = 7.692%` occupancy. Continuous demand and elasticity assumptions are therefore less credible, and MAE of one room is operationally large.

## 11.2 Implementation support

RateAnchor implements Full Service and Select Service presets. It does not implement a named “13-room independent” preset; “Boutique” is the nearest available posture but is not equivalent. Inventory size enters capacity, occupancy, forecast caps, and elasticity bounds, so a 13-room file could execute, but its research treatment would still need explicit calibration.

## 11.3 Evidence boundary

The repository metadata identifies saved periods, not these three property identities. No source/output labels a 13-room hotel, and current saved datasets use a manual 99-room inventory for July. Therefore the three-property comparison strengthens the proposed external-validity design, but it has not been completed in the repository.

## 11.4 Why the three types would strengthen research

They test whether performance is stable across scale and operating model, expose whether percentage thresholds behave differently under discrete small inventory, and reduce the risk that one property's booking pattern is mistaken for a general result. Valid comparison requires per-property training/backtesting, not pooling without property identifiers.

---

# 12. Backtesting Methodology

## 12.1 Demand-forecast backtest

1. `prepare_forecast_frame` sorts historical daily metrics and engineers date/event/STLY features.
2. Only dates `<= as_of_date` enter history.
3. With enough observations, evaluation begins after `max(min_train_days=10, min_test_days=21)=21` rows.
4. The code advances through non-overlapping 7-day test blocks; training is every earlier row (expanding window).
5. Baseline prediction uses prior training weekday mean rooms sold and ADR.
6. Tailored calibrated prediction reserves a recent calibration slice within each training window and tests candidates.
7. Prediction rows are joined with actual rooms sold/revenue and subgroup labels.
8. MAE/RMSE are aggregated overall and by property/event/month/day type.

Fallback: if history is too short for rolling windows, the final 21–30 days are used as test subject to preserving 10 training days. If there are at most 10 rows, training is empty and all history is test, making results descriptive only.

## 12.2 Candidate calibration

The tailored demand model starts with baseline. For at least 14 training rows it uses a 3–7 row calibration tail. It tests smoothed month/day-type residual corrections; at 45+ core-training rows it also tests enhanced Random Forest predictions, blends at 25/50/75%, STLY estimates, STLY bias correction, and baseline/STLY blends. The normalized score is 30% MAE and 70% RMSE. Only strict improvement over baseline is accepted.

## 12.3 Historical ADR/rate backtest

1. Sort all historical daily rows.
2. Derive actual rate in order: `actual_adr`, `adr`, `current_rate`, then revenue/sold.
3. Generate baseline recommendations on the same rows, with the same dataframe also passed as historical ADR context.
4. Generate tailored recommendations on the same rows.
5. Compute signed errors:

```text
baseline_error = baseline_recommendation - actual_ADR
rateanchor_error = tailored_recommendation - actual_ADR
```

6. Aggregate MAE/RMSE overall and by property/event.

## 12.4 Methodological caveat

The ADR rate backtest is not a causal or temporally isolated rate forecast. Both models can use contemporaneous actual ADR; the tailored model can derive its median from the same dataset. This measures deviation from observed ADR under the implemented policy, not whether the recommendation would have generated the observed demand or more revenue if deployed. A stronger backtest would reconstruct information available at each historical decision timestamp and model price-demand outcomes.

## 12.5 Elasticity-policy evaluation

`build_baseline_vs_new_policy` creates a synthetic on-books value equal to 70% of actual rooms sold, estimates weekday demand, and calculates expected revenue for baseline and elasticity-selected rates under the same assumed elasticity. Uplift is:

```text
Uplift_t = NewPolicyExpectedRevenue_t - BaselineExpectedRevenue_t
```

This is an internally consistent scenario, not observed uplift.

---

# 13. Verification

## 13.1 Automated unit/integration testing

On 15 July 2026:

```text
71 passed, 3 warnings, 0 failed
```

Warnings were pandas date-format inference in `pace.py` (two integration tests) and `ingest.py` (one schema test). They do not fail current behavior but identify reproducibility risk for ambiguous date strings.

Coverage includes:

- hand-checked baseline high/moderate/low cases;
- KPI formulas and zero-division behavior;
- overbooking and rate imputation;
- schema aliases and malformed/shifted CSV repair;
- YoY exact/missing-prior cases;
- elasticity bounds and near-zero rates;
- history-depth fallback;
- backtest split, STLY leakage protection, and rolling windows;
- model metrics and perfect-prediction leakage warning;
- tailored comp precedence, modes, property/segment behavior, settings bounds, stale cadence, and missing median;
- intraday ordering/change logging;
- budget context numeric outputs;
- saved settings and complete demo pipeline artifacts.

## 13.2 Hand verification

`test_hand_checked.py` contains manually computed KPI examples. Tests also assert exact baseline rates and hand-computed YoY values. This independently checks equations against expected numbers rather than merely checking dataframe shape.

## 13.3 Formula verification

KPI tests cover standard cases and zeros. Elasticity tests assert on-books and inventory bounds. Budget tests verify numeric types and internal field production. Model metric tests use known actual/predicted vectors.

## 13.4 Output verification

Integration tests run the pipeline in a temporary directory and assert date-level tailored exports and demo artifacts. Timestamped run directories preserve output versions, though the configuration is not currently serialized into each run, which weakens exact reproducibility.

## 13.5 Historical and simulation verification

Chronological expanding-window forecast evaluation reduces direct future-to-past leakage. `prepare_forecast_frame` rejects same-year STLY occupancy. `detect_prediction_identity_warning` flags at least seven identical predictions. Current output metrics were independently read from generated CSVs for this report.

## 13.6 What verification establishes

The green suite establishes conformance to implemented rules and regression stability. It does not establish optimality, causal uplift, statistical significance, external validity across hotel types, or production reliability.

---

# 14. Results

## 14.1 Monthly

The latest 31-row July run supports the statement that baseline won on both demand error and ADR recommendation error. Its demand backtest evaluated 10 rows. The outcome is consistent with bias-variance reasoning: simple weekday averages are stable with limited data, while tailored calibration has few repeated seasonal patterns.

## 14.2 Larger period

The latest 65-valid-row multi-month run supports a narrower statement: tailored **demand forecasting** won on 44 expanding-window rows, reducing MAE from 16.1266 to 11.6459 and RMSE from 20.0851 to 15.3432. It does not support a tailored **rate** win: ADR MAE and RMSE were materially worse.

## 14.3 Annual

No annual result exists. Any annual result shown in a defense would need a new preserved experiment.

## 14.4 Why more historical variation can improve tailored forecasting

More dates create repeated weekdays, month transitions, event/non-event contexts, and day types. This allows the validation gate to estimate residual corrections and compare enhanced/blended/STLY candidates. With 45+ rows, Random Forest and blend candidates become eligible. However, “more variation” helps only when it is representative and the calibration tail predicts later structure; more noisy or shifted data can hurt.

## 14.5 Statistical meaning

The current results are descriptive error comparisons. No confidence intervals, paired significance test, effect-size uncertainty, multiple-property hierarchical analysis, or independent final test set is generated. The expanding-window rows are time-dependent and not IID. Percentage improvements quantify this sample only.

## 14.6 Operational meaning

For the 65-row run, a roughly 4.48-room MAE reduction can be operationally meaningful at a moderate/large hotel, but its meaning depends on inventory. At 13 rooms, errors and continuous forecasts require different interpretation. Worse ADR error does not necessarily mean worse revenue—actual ADR is not necessarily optimal—but absent causal evaluation, it prevents claiming that RateAnchor price recommendations are superior.

---

# 15. Limitations

## 15.1 Data and sample

- Few saved periods and no completed three-property or annual experiment.
- Physical source rows, unique dates, and validated rows differ; attrition must be reported.
- Property identity/class is not embedded in historical observations.
- Future on-books snapshots do not encode lead time, so pace may not be cutoff-matched.
- Repository STLY contains limited history and can be absent/incomplete.

## 15.2 Statistical and modeling

- Thresholds, weights, biases, event impacts, elasticity, close factor, and occupancy center are expert-coded, not estimated.
- Constant elasticity is assumed equal across dates, rates, segments, and properties.
- No uncertainty intervals or probabilistic demand distribution.
- Small calibration windows can select unstable candidates.
- Reusing calibration repeatedly during rolling evaluation can create model-selection optimism.
- MAPE excludes zeros and can distort low-demand days.
- No significance tests or correction for time dependence.

## 15.3 Recommendation validity

- Rate backtest uses contemporaneous actual ADR as an input/anchor.
- Dataset-derived comp medians can be derived from the property's own rate fields, not true competitors.
- Budget pressure does not enter tailored rate calculation directly.
- `adr_sensitivity` is unused.
- Tailored model lacks a percentage-change guardrail.
- Projected uplift is model-based, not realized.

## 15.4 Data engineering

- Duplicate daily rows are not generally disambiguated as segment versus accidental duplicate.
- Numeric parse failures in counts become zero.
- Date parsing can rely on `dateutil` inference, as the three warnings show.
- Revenue heuristics use hardcoded scale ranges that may fail for currencies/property scales.
- No currency, tax, net/gross, or timezone metadata.

## 15.5 Architecture and operations

- File-based storage; no concurrency controls or database transactions.
- Dataset deletion uses recursive filesystem removal and has no audit/undo.
- No authentication, authorization, tenant isolation, encryption workflow, or PII governance.
- No live PMS, CRS, channel-manager, rate-shop, weather, or event integration.
- No automated rate execution or rollback.
- No serialized run configuration/model version/hash alongside outputs.
- Streamlit session state is not a durable workflow engine.

## 15.6 Research scope

The prototype validates software behavior and preliminary comparative results. It does not prove generalizable financial improvement or production readiness.

---

# 16. Future Work

## 16.1 Stronger experimental design

1. Pre-register monthly, quarterly, and annual horizons and property cohorts.
2. Preserve raw row count, unique-date count, validated count, and exclusion reasons.
3. Reconstruct historical decision-time snapshots with booking lead time and rates known then.
4. Reserve a final untouched temporal test period.
5. Report paired block-bootstrap confidence intervals and property-level effects.
6. Evaluate realized revenue/RevPAR with causal or controlled deployment methods.

## 16.2 Machine learning

- quantile/random-forest or gradient-boosted demand intervals;
- hierarchical models sharing strength while retaining property effects;
- count/discrete models for very small inventory;
- automatic time-series cross-validation and hyperparameter calibration;
- price-response estimation from historical price variation rather than fixed elasticity;
- drift detection and champion/challenger governance.

Every ML extension should retain a baseline, leakage controls, feature provenance, uncertainty, and explainability.

## 16.3 Live integrations

- PMS/CRS on-books snapshots and actuals;
- rate-shop competitor feeds;
- channel-manager current rates and restrictions;
- weather forecasts;
- local events, holidays, school calendars, and flight demand;
- budget/finance systems.

Adapters should land data in a versioned canonical event store, not call domain algorithms directly.

## 16.4 Automated calibration and safety

- estimate property-specific thresholds/elasticity/weights;
- add percentage movement guardrails to tailored output;
- enforce inventory/rate-plan/room-type constraints;
- detect anomalous recommendations and require approval;
- serialize configuration, code commit, data hashes, model choice, and warnings per run;
- measure operator acceptance and override reasons.

## 16.5 Additional hotel types and datasets

Complete the proposed select-service, full-service, and 13-room independent study, then expand to resort, extended-stay, economy, luxury, and boutique properties across regions and seasons. Small hotels should use discrete-room metrics and absolute-room error, not only occupancy percentages.

---

# 17. Thesis Defense Questions and Implementation-Grounded Answers

## Architecture and software engineering

### 1. What architectural style does RateAnchor use?

It uses a layered, modular dataframe pipeline: Streamlit/CLI presentation, application orchestration in `main.run_pipeline`, domain modules under `src`, and file-based persistence. It is not a microservice architecture and has no database or message bus.

### 2. Why are both `app.py` and `main.py` needed?

`app.py` owns interactive session state and presentation; `main.py` owns the reusable end-to-end pipeline and CLI. This lets the same domain workflow run locally, in Streamlit, and in tests.

### 3. What is the central integration point?

`main.run_pipeline`. It constructs typed configuration, processes inputs, validates data, calculates metrics, forecasts, generates recommendations, evaluates models, writes outputs, and returns path/summary dictionaries.

### 4. Where are domain decisions separated from file reporting?

Core calculations live in modules such as `forecast.py`, `baseline.py`, `pricing.py`, and `tailored.py`; `pipeline_outputs.py` and `pipeline_reporting.py` assemble files and charts.

### 5. Why is the canonical schema important?

It creates a stable contract across heterogeneous PMS column names. Downstream logic can reason about `rooms_sold` rather than vendor-specific labels, and mappings can be tested independently.

### 6. How does the system avoid hardcoded local data paths?

Demo paths are resolved relative to the project root, and CLI paths come from arguments. The only repository historical lookup is relative to `main.py` under `data/historical`.

### 7. What state is persistent?

Named datasets, mappings, tailored settings, budget profiles, and run outputs are stored in CSV/JSON directories. Streamlit session state is transient; there is no transactional database.

### 8. Is the system thread-safe or multi-user safe?

The implementation provides no locking, atomic transaction layer, or tenant separation. Timestamped output folders reduce collisions, but file-based dataset metadata is not designed for concurrent writes.

### 9. What makes the implementation explainable?

Rules and formulas are explicit; outputs include reason strings, signal labels, constraints, warning text, confidence, status, comp-median source, and validation issues.

### 10. What is the largest architectural risk?

The application can produce many sophisticated artifacts without a serialized run manifest tying data hashes, configuration, code version, and model choice together. This limits exact scientific reproducibility.

### 11. Why use pandas dataframes throughout?

The source data and outputs are tabular, and dataframe transformations make schema mapping, grouping, merging, and export concise. The cost is weaker static type guarantees and runtime discovery of missing columns.

### 12. How are configuration defaults controlled?

`build_pipeline_config` maps a dictionary into immutable runtime, pricing, and baseline dataclasses. Tailored settings use a separate dataclass plus validation/sanitization.

### 13. Does the code implement dependency inversion?

Only partially. Domain functions accept dataframes/configs rather than UI objects, but `main.py` imports concrete modules directly and persistence uses concrete filesystem/pandas calls.

### 14. How does error handling differ between CLI and UI?

`main.main` translates file, validation, and unexpected exceptions into printed messages and exit code 1. Streamlit wraps execution and displays errors/status; domain functions still raise exceptions.

### 15. Why are output directories timestamped?

`prepare_output_directory` uses `run_<timestamp>` so runs are preserved rather than overwritten. Timestamp resolution and concurrent collision handling are not otherwise specified.

## Ingestion, schema, and validation

### 16. How does RateAnchor identify a PMS report header?

It scans up to 25 rows for a row containing `date` and a recognized metric such as room revenue, occupancy, or rooms sold, then promotes that row.

### 17. How are malformed CSVs handled?

The loader tries aligned report parsing, normal CSV, delimiter inference, semicolon, tab, and finally a permissive Python parser with quoting disabled and bad lines skipped. The final path emits a warning.

### 18. Could permissive parsing bias research results?

Yes. Skipped malformed rows can change sample composition. The implementation warns but does not save a separate skipped-row ledger, so row counts must be audited.

### 19. How are shifted currency values detected?

The code evaluates fill ratio, dollar-sign presence, numeric count, and median scale in nearby/currency-like columns, then combines better candidates for revenue or ADR.

### 20. Why are revenue scale thresholds a limitation?

The heuristics assume room-revenue medians around at least 300 and ADR roughly 20–500. Different currencies, tiny hotels, or aggregated reports could violate those assumptions.

### 21. What happens if required columns map to the same source?

`map_columns` raises `ValueError`; it also specifically prevents revenue and occupancy from sharing a source.

### 22. How is missing inventory derived?

When rooms sold and occupancy are present, inventory is rounded from `rooms_sold / occupancy_ratio`; a boolean provenance field identifies derived rows.

### 23. Why can derived inventory be unstable?

Rounding amplifies occupancy/report precision error, especially at low counts. `prepare_forecast_frame` later replaces derived inventory with a stable median if non-derived inventory exists.

### 24. Which validation failures remove rows?

Null dates, negative inventory/sold/revenue, disallowed overbooking, and missing revenue on non-future dates.

### 25. Why is missing future revenue allowed?

Future on-books reports may contain bookings and rates before realized revenue exists. The pipeline can derive rate context and still forecast/recommend.

### 26. How is current rate imputed?

ADR first, then revenue divided by positive rooms sold, then a positive median/default fallback, ultimately 120.00.

### 27. Does every logged validation issue remove a row?

No. `CURRENT_RATE_FILLED` marks the result as not fully passed but does not set the critical removal flag.

### 28. What happens if all historical rows fail?

The pipeline raises `ValueError` and reports the three most common issue types; no recommendation pipeline continues.

### 29. How are duplicate stay dates treated?

Primary validation only warns because duplicates may be legitimate segments. YoY/pace aggregate them; repository STLY deduplicates across files; other modules may retain them.

### 30. What date ambiguity remains?

`pd.to_datetime` sometimes falls back to per-element `dateutil` inference. The current test run produced three warnings, so locale-ambiguous strings are a known risk.

## KPIs, YoY, budget, and context

### 31. Prove that RevPAR equals ADR times occupancy.

With consistent aggregation and positive denominators, `(R/S)*(S/A)=R/A`. The code calculates `R/A` directly in `metrics.py` and `ADR*occupancy` in the tailored frame.

### 32. How are zero denominators handled?

Daily KPI calculation returns 0.0 for occupancy/RevPAR when inventory is zero and ADR when sold is zero. YoY normalization often uses `NaN` instead, so output semantics differ by module.

### 33. How is prior-year matching performed?

Exact current date minus one year is preferred. If unavailable, the latest earlier row with the same month-day is used.

### 34. What leap-year issue can occur?

An exact one-year offset may not find an equivalent date, and calendar fallback cannot match February 29 in a non-leap prior year. The row becomes unavailable unless an earlier Feb 29 exists.

### 35. What does pace variance actually measure?

Current occupancy minus STLY occupancy. It is not inherently a lead-time-matched booking pace unless inputs are matched snapshots.

### 36. How are multiple same-date events handled?

Events are ranked low/medium/high and only the highest-impact event is retained. Compound event effects are not added.

### 37. How does event context affect different algorithms?

Future demand uses multipliers 1.02/1.05/1.09; rule pricing uses 2%/4%/7%; tailored event index adds both event percentage and 0.05/0.10/0.18 impact signal.

### 38. Why could tailored event influence be double-counted?

`event_pct` already encodes impact, then `_event_index` adds another impact-level value. Both are scaled in demand index.

### 39. How is a monthly budget allocated daily?

Either equally by day or by normalized historical revenue weights for each day of week, preferring history from the target month.

### 40. What is required ADR mathematically?

Remaining budget divided by forecast remaining rooms sold, expressed in code as `remaining_budget/(remaining_rooms_available*forecast_remaining_occ)`.

### 41. Does budget directly change the tailored rate?

No. Budget affects reports, elasticity explanation, and priority score, but `build_tailored_recommendations` receives no budget input.

### 42. What happens if budget is already achieved?

Required ADR remaining returns zero because remaining budget is non-positive.

### 43. How is month capacity estimated?

Median positive daily inventory multiplied by days in month; it falls back to historical inventory and then at least on-books rooms.

### 44. How is STLY used in month-end projection?

STLY remaining pickup is derived using a default 1.15 close factor and pace factor; projected revenue is blended 65% with 35% STLY revenue, capped at 110% of STLY and capacity × high historical rate.

### 45. Is month-end forecast identical to the daily future forecast sum?

No. `pipeline_budget_forecast.py` uses a separate pickup/STLY/month-capacity calculation, while `build_future_forecast` produces date-level demand.

## Forecasting algorithms

### 46. What is the future forecast's base seasonal model?

The mean of day-of-week average rooms sold and month average rooms sold, with global-average fallbacks.

### 47. How is history depth quantified?

Unique historical stay dates divided by 28 and clipped to [0,1]. It is labeled no, low (<14), moderate (<28), or strong history.

### 48. What is the sparse-history fallback?

The model blends seasonal demand toward the maximum of on-books, inventory times median historical occupancy, and global rooms average; under 14 days it caps base demand at 120% of fallback, never below on-books.

### 49. Why is on-books a hard lower bound?

Already booked rooms cannot be “unbooked” by the demand forecast. Both future forecast and elasticity response floor expected rooms at on-books.

### 50. Why is inventory a hard upper bound?

Forecast sold rooms cannot exceed physical rooms in the model. If on-books exceeds capacity in elasticity, the upper bound is raised to the on-books lower bound to avoid an impossible clip interval.

### 51. What is the baseline demand forecast?

Mean historical rooms sold and ADR by day of week, falling back to global means.

### 52. When is linear regression used?

`enhanced_forecast` uses linear regression for training sizes below 45 but at least 10. However, calibrated candidate inclusion requires core training of at least 45, so the main calibrated path generally does not select that linear model.

### 53. When is random forest used?

At 45 or more training rows, with 200 trees, `random_state=42`, and parallel fitting.

### 54. Which features feed the enhanced model?

Rooms available, day of week, month, weekend flag, event flag, and STLY occupancy. Separate models predict rooms sold and ADR.

### 55. How does the calibrated tailored forecast prevent overfitting?

It tests candidate corrections on a recent held-out calibration slice and accepts only a score strictly better than baseline; short histories return baseline.

### 56. Why weight RMSE at 70% in calibration?

The implementation emphasizes avoiding large misses over average absolute error. The weight is a design choice, not learned from data.

### 57. What candidate corrections are evaluated?

Month, day-type, and month-day-type smoothed baseline residual corrections; at sufficient depth, enhanced forecasts, baseline-enhanced blends, STLY, STLY bias correction, and baseline-STLY blends.

### 58. How is group-bias smoothing performed?

`bias=(group residual sum + global_bias*3)/(group count+3)` for groups with at least two rows. A group correction is retained only if its own normalized MAE/RMSE score improves.

### 59. What leakage protection exists for STLY?

`prepare_forecast_frame` accepts STLY only if its year is earlier than the target year, and tests cover same-year rejection and prior-year preference.

### 60. What leakage risk remains?

Model candidates and thresholds were developed against available data, and rate backtesting uses contemporaneous ADR. There is no final untouched multi-property test set.

## Pricing and tailored algorithms

### 61. Why is the baseline a fair control?

It is deterministic, transparent, and intentionally ignores tailored-only features. It provides a reproducible low-complexity comparator, though it is project-defined rather than an external commercial baseline.

### 62. What are the baseline occupancy thresholds?

At least 85% triggers +5%; at most 55% triggers -5%; the middle holds rate, subject to floor and ceiling.

### 63. What is the baseline rate anchor?

Current ADR/current rate/revenue-derived ADR, with day-of-week median historical ADR only when current ADR is unavailable.

### 64. Why can the baseline be advantaged in ADR backtesting?

It uses actual/current ADR from the same historical row and moves only ±5%, so it is structurally close to the evaluation target.

### 65. What demand curve does elasticity pricing assume?

Constant elasticity: `D(p)=D0*(p/p0)^(-epsilon)` with positive-clamped rates and elasticity.

### 66. Which candidate price changes are searched?

-10%, -5%, 0%, +5%, +10%, and +15%, plus event percentage before clipping.

### 67. How are elasticity candidates constrained?

Global rate floor/ceiling, maximum daily change, on-books/capacity demand bounds, and optional Friday/Saturday rate bands.

### 68. Does the elasticity engine interpolate the optimum?

No. It enumerates a small discrete grid and selects the candidate with highest modeled revenue.

### 69. What happens when candidate revenues tie?

The code updates only on strictly greater revenue, so the earlier/current initialized rate remains favored.

### 70. How is elasticity recommendation confidence calculated?

One point each for meaningful pace signal, medium/high event, and uplift over $50; >=2 is high, >=1 medium, otherwise low. It is heuristic.

### 71. How does the tailored model obtain a competitor anchor?

Median of positive comp-set rates by date if available; otherwise median of property dataset rate candidates. Manual daily and global values can override/fallback according to mode.

### 72. Why is a dataset-derived median not necessarily a comp-set median?

It may be calculated from the property's current rate, ADR, recommended rate, and derived ADR. The source label makes this visible, but it is not external market evidence.

### 73. What is the tailored demand-index center?

Occupancy is centered at 0.72 before weighting. That center is hardcoded.

### 74. What does the strategy index represent?

An explicit tradeoff among revenue priority, RevPAR priority, and rooms-sold priority plus property and segment biases, clipped to ±0.35.

### 75. How strongly can the median influence the rate?

Anchor weight is clipped between 0.15 and 0.95, so with a median the anchored rate moves at least 15% and at most 95% of the baseline-to-median difference.

### 76. What happens at very high occupancy?

At occupancy >=90% with a median, final pre-limit rate cannot be below 102% of median.

### 77. What happens at low occupancy?

At occupancy <=55% with a median, final pre-limit rate cannot exceed 99% of median.

### 78. Which tailored setting is currently unused?

`adr_sensitivity` is validated and saved but not referenced in the tailored rate equation.

### 79. Is there a tailored maximum daily change?

No explicit percentage guardrail exists in `tailored.py`; only index/anchor clips and absolute minimum/maximum rates constrain it.

### 80. How do property presets differ from inventory size?

Presets change biases/multipliers based on a label. Inventory size separately affects occupancy, capacity caps, and elasticity; the preset does not infer size.

## Backtesting, statistics, and interpretation

### 81. Why use chronological rather than random splits?

Pricing and forecasting are time-dependent. Training only on earlier rows better approximates operational prediction and prevents direct future-to-past leakage.

### 82. How do expanding windows work here?

After 21 initial historical rows, the code tests successive blocks of up to seven days, always training on every earlier row.

### 83. Why can backtest row counts differ from dataset rows?

Initial training rows are not scored, invalid rows are removed, dates may be duplicated or dropped, and the final source may contain report artifacts.

### 84. What does forecast MAE measure?

Average absolute difference in rooms between predicted and actual rooms sold over evaluated rows.

### 85. Why report RMSE as well as MAE?

RMSE gives larger misses more influence, exposing occasional operationally severe errors that MAE can hide.

### 86. What is wrong with relying only on MAPE?

Actual zero rooms are excluded, and small actual values create very large ratios. This is especially problematic for small properties.

### 87. What does the identity warning detect?

For at least seven rows, it warns when maximum absolute prediction error is <=1e-6, a sign of possible leakage.

### 88. Does absence of the identity warning prove no leakage?

No. It detects only perfect copying, not subtler feature leakage, calibration reuse, or contemporaneous inputs.

### 89. What is the difference between demand and rate backtests?

Demand backtest predicts rooms sold chronologically. Rate backtest compares policy recommendations with historical actual ADR on the same dates.

### 90. Does lower rate MAE prove higher revenue?

No. It proves closeness to observed ADR, not optimality or causal revenue effect. A different rate could yield more or less demand and revenue.

### 91. Does projected uplift prove realized uplift?

No. It is calculated from a fixed elasticity scenario and synthetic 70%-of-actual on-books assumption.

### 92. Why did baseline win on the monthly run?

The 31-row series provided only 10 scored demand rows in the latest run; the stable weekday baseline had lower variance, and the ADR baseline was anchored near actual ADR.

### 93. Why did tailored demand win on the larger run?

With 45+ training observations, more calibrated candidates became eligible and recurring temporal structure could be estimated; the validation gate selected an improving candidate.

### 94. Did the tailored model win overall on the larger run?

No. It won demand MAE/RMSE but lost historical ADR recommendation MAE/RMSE. “Overall” would collapse different dependent variables.

### 95. Are the improvements statistically significant?

The implementation does not calculate significance, confidence intervals, or a paired block bootstrap. Only descriptive sample errors are available.

### 96. Is an annual conclusion supported?

No annual input/output artifact exists in the repository, so no annual result can be defended from this implementation state.

### 97. Is the three-hotel comparison complete?

No. Full Service and Select Service presets exist, but saved data is not labeled as the requested three properties and no 13-room artifact exists.

### 98. What does the passing test suite prove?

It proves 71 tested implementation behaviors currently conform to expected outputs. It does not prove external validity, financial uplift, or exhaustive correctness.

### 99. What is the most defensible thesis conclusion?

RateAnchor demonstrates a working, explainable, modular pipeline with robust ingestion, validation, fallback logic, and measurable forecast/recommendation comparisons. Larger-history tailored demand calibration can outperform baseline on the observed run, while pricing superiority remains unproven.

### 100. What single experiment should be run next?

A predeclared, decision-time historical backtest across the select-service, full-service, and 13-room properties, with matched on-books snapshots, external comp rates, serialized configurations, an untouched final period, and paired time-block uncertainty estimates. That directly tests generalization while removing the current contemporaneous-rate limitation.

---

# Defense Claim Ledger

| Claim | Evidence status |
|---|---|
| End-to-end prototype executes | Verified by integration tests and generated runs |
| 71 automated tests pass | Verified 15 July 2026 |
| Baseline wins latest monthly demand comparison | Supported: 9.8083 vs 13.1951 MAE |
| Tailored demand wins latest larger-period comparison | Supported: 11.6459 vs 16.1266 MAE |
| Tailored rate beats baseline | Not supported by latest monthly or larger ADR backtests |
| Projected elasticity-policy uplift is positive | Supported as a simulation output, not realized revenue |
| Annual performance | Not evaluated in repository |
| Three-property external validity | Not evaluated in repository |
| Production-ready autonomous RMS | Not claimed and not implemented |
