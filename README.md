# Flexible Revenue Management System for Hotel Pricing Decision Support

A Python and Streamlit revenue management prototype that helps hotels clean PMS-style data, calculate core KPIs, compare year-over-year performance, forecast future demand, and produce explainable pricing recommendations. The project was built as a capstone-ready decision-support system for hotel revenue managers who need a clearer way to move from raw reports to actionable rate guidance.

## Project Overview

Hotels often rely on static pricing rules, manual spreadsheet updates, or once-daily rate reviews. Those workflows can make it difficult to react to occupancy pace, budget gaps, local events, and inconsistent PMS exports. This project provides an end-to-end workflow that standardizes hotel data, validates inputs, calculates performance metrics, and compares multiple pricing approaches in a repeatable way.

The application supports both a Streamlit dashboard for interactive analysis and a CLI pipeline for repeatable runs and testing.

## Current Capstone Focus

The current version includes three pricing layers:

1. **Baseline once-daily model**
   A transparent comparison model that uses simple occupancy thresholds and historical ADR context.

2. **Enhanced RMS recommendation engine**
   A forward-looking model that considers on-books demand, forecasted occupancy, event impact, budget pressure, elasticity, and rate guardrails.

3. **Tailored model layer**
   A configurable model that adjusts baseline recommendations using property type, segment focus, KPI priorities, demand sensitivity, seasonality, event impact, and day-by-day median rate inputs.

## Key Features

* CSV and Excel upload support
* PMS-style report cleaning, including files with metadata/header rows
* Automated and assisted column mapping
* Historical and future on-the-books data handling
* Saved datasets to avoid repeated imports
* Saved budget profiles
* Manual rooms-available override for reports missing inventory
* Validation pipeline for required fields, missing values, invalid dates, and overbooking checks
* Daily KPI calculations for occupancy, ADR, RevPAR, rooms sold, and room revenue
* Year-over-year comparison output with missing-prior-year handling
* Forecasting and backtesting workflow
* Budget-aware pricing support
* Event impact handling
* Baseline once-daily pricing recommendations
* Enhanced rate recommendation engine
* Baseline vs enhanced policy comparison
* Tailored model settings that persist with saved datasets
* Day-by-day median rate table for tailored recommendations
* Manual daily median rate overrides with timestamps
* Dataset-derived median rate suggestions by stay date
* Explainable recommendation outputs with confidence, warnings, and reasoning notes
* Streamlit dashboard interface
* CLI pipeline support
* CSV and chart exports for reporting

## How the System Works

1. **Import data**
   Upload or provide historical PMS data, future on-the-books data, optional event data, and optional budget data.

2. **Clean and standardize fields**
   The system normalizes PMS-style exports and maps inconsistent column names into a standard schema.

3. **Validate inputs**
   Required fields are checked before downstream calculations. Invalid rows are flagged through a validation report.

4. **Calculate KPIs**
   Daily occupancy, ADR, RevPAR, rooms sold, and room revenue are calculated from standardized data.

5. **Build forecast context**
   Future rows are combined with historical performance, same-time-last-year context, pace indicators, event impact, and budget information.

6. **Generate pricing outputs**
   The pipeline produces baseline recommendations, enhanced recommendations, tailored recommendations, priority lists, and evaluation metrics.

7. **Export results**
   Each run creates a timestamped output folder containing CSV files, summaries, and generated chart images.

## Data Requirements

Minimum historical input fields:

* `stay_date`
* `rooms_available` or a manual rooms-available override
* `rooms_sold`
* `room_revenue`

Optional supported fields:

* `adr`
* `current_rate`
* `occupancy_percent`
* `booking_date`
* event indicators
* budget targets

Future on-the-books inputs should include:

* `stay_date`
* `rooms_available` or a manual rooms-available override
* `rooms_sold`
* `current_rate`, `adr`, or enough revenue data to derive rate context

Column names do not need to match perfectly. The project attempts to map common aliases automatically, and the dashboard supports manual mapping when needed.

## Demo Data

The repository includes sample files for testing and demonstration.

```text
data/
├── sample_data.csv
├── future_on_books_sample.csv
├── events_sample.csv
├── budget_daily_sample.csv
└── budget_monthly_sample.csv

data/historical/
└── occupancy_2024.csv
```

Additional local PMS exports and saved dashboard datasets are intended for local use and should remain outside version control when they contain real or private data.

## Project Structure

```text
Revenue-Management-Model/
├── app.py                    # Streamlit dashboard
├── main.py                   # CLI and end-to-end pipeline entry point
├── requirements.txt          # Python dependencies
├── data/                     # Safe demo/sample data
├── outputs/                  # Generated run outputs
├── datasets/                 # Locally saved dashboard datasets
├── tests/                    # Pytest coverage
└── src/
    ├── baseline.py           # Baseline once-daily pricing model
    ├── budget.py             # Budget progress and target logic
    ├── dataset_manager.py    # Saved dataset and budget profile handling
    ├── elasticity.py         # Elasticity demand assumptions
    ├── evaluation.py         # Forecast and policy evaluation outputs
    ├── events.py             # Event impact handling
    ├── forecast.py           # Forecast and backtesting functions
    ├── ingest.py             # File import and cleaning
    ├── metrics.py            # KPI calculations
    ├── pace.py               # Pace and STLY analysis
    ├── pricing.py            # Enhanced pricing recommendation logic
    ├── schema.py             # Column mapping helpers
    ├── tailored.py           # Property-specific tailored model layer
    ├── validate.py           # Data validation pipeline
    └── yoy.py                # Year-over-year comparison logic
```

## Installation

Clone the repository:

```bash
git clone https://github.com/L0ckEndL0L/Revenue-Management-Model.git
cd Revenue-Management-Model
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Streamlit Dashboard

```bash
python -m streamlit run app.py
```

You can also run:

```bash
python app.py
```

The dashboard supports uploading historical data, future on-the-books data, optional events, and optional budget files. It also supports saving and loading datasets so the same files and tailored settings can be reused across sessions.

## Run the CLI Pipeline

Basic run:

```bash
python main.py --input data/sample_data.csv --output outputs
```

Run with future on-books, events, and budget files:

```bash
python main.py \
  --input data/sample_data.csv \
  --future data/future_on_books_sample.csv \
  --events data/events_sample.csv \
  --budget data/budget_monthly_sample.csv \
  --output outputs
```

Useful CLI options:

```text
--input                     Historical PMS report. Required.
--future                    Optional future on-the-books report.
--events                    Optional events CSV.
--budget                    Optional budget file.
--output                    Output directory. Defaults to outputs.
--rate_floor                Minimum allowed recommendation rate.
--rate_ceiling              Maximum allowed recommendation rate.
--max_change_pct            Maximum daily rate movement.
--elasticity                Demand elasticity assumption.
--manual_rooms_available    Manual inventory override when reports omit rooms_available.
--allow-overbooking         Allow rooms_sold to exceed rooms_available during validation.
--no-interactive            Disable interactive mapping behavior.
```

## Main Outputs

Each pipeline run creates a timestamped folder under `outputs/`. Common exported files include:

* `cleaned_data.csv`
* `daily_metrics.csv`
* `validation_report.txt`
* `forecast.csv`
* `yoy_comparison.csv`
* `baseline_recommendations.csv`
* `rate_recommendations.csv`
* `tailored_model_results.csv`
* `tailored_model_summary.csv`
* `baseline_vs_new_policy.csv`
* `evaluation_metrics.csv`
* `forecast_vs_actual.csv`
* `top_raise_opportunities.csv`
* `top_rescue_dates.csv`
* `top_monitor_dates.csv`
* `current_vs_recommended_rate.png`
* `expected_revenue_uplift.png`
* `priority_score_by_date.png`
* `forecast_vs_actual.png`

## Tailored Model Layer

The tailored model is designed to make the pricing output more property-specific while still staying explainable. It starts with the baseline recommendation and adjusts it based on configurable business assumptions.

Supported tailored settings include:

* property type
* segment focus
* occupancy sensitivity
* ADR sensitivity
* RevPAR priority
* rooms-sold priority
* revenue priority
* demand adjustment factor
* seasonality adjustment factor
* event impact factor
* minimum acceptable rate
* maximum recommended rate
* global median rate fallback
* median rate update frequency
* day-by-day manual median rate inputs

The day-by-day median rate workflow allows the system to use a different median rate for each stay date instead of relying only on one global median rate for the entire forecast. The system can suggest a dataset-derived median rate by stay date, allow manual overrides, and track when manual values were updated.

## Validation and Testing

Run compile checks:

```bash
python -m compileall src app.py main.py
```

Run the test suite:

```bash
python -m pytest
```

The project includes tests for major pipeline components, validation behavior, baseline recommendations, YoY logic, tailored model behavior, and integration workflows.

## Current Development Status

This is a working capstone and portfolio prototype. It demonstrates a full hotel revenue-management workflow from data import through explainable pricing recommendations. The project is not intended to replace a production RMS, but it is structured to show practical data cleaning, model comparison, business-rule design, validation, testing, and dashboard delivery.

## Future Improvements

* Add more real-world safe sample datasets
* Expand test coverage for edge cases and larger files
* Improve event demand signal engineering
* Add more detailed model evaluation summaries
* Add dashboard screenshots and reporting visuals
* Package the app for hosted demo deployment
* Improve documentation for accepted PMS export formats

## Author

Thomas Hayden

