# Hotel Revenue Management Decision Support System

A Python and Streamlit capstone project that helps hotels clean PMS-style data, calculate KPIs, compare year-over-year performance, forecast demand, and produce explainable pricing recommendations.

The project supports both:

* A Streamlit dashboard for board review and interactive analysis
* A CLI pipeline for repeatable local runs and testing

## Capstone Board Access

**Live Demo:** To be added after deployment.

Board review steps:

1. Open the web app.
2. Click **Load Demo Dataset**.
3. Click **Run Pricing Simulation**.
4. Review **Outputs**, **Tailored Model**, **Rate Recommendations**, and **YoY** tabs.

The demo workflow uses safe sample data included in this repository. It does not require private PMS files, saved datasets, generated outputs, or secrets.

## Project Overview

Hotels often rely on static pricing rules, manual spreadsheet updates, or once-daily rate reviews. Those workflows can make it difficult to react to occupancy pace, budget gaps, local events, and inconsistent PMS exports. This project provides an end-to-end workflow that standardizes hotel data, validates inputs, calculates performance metrics, and compares multiple pricing approaches in a repeatable way.

## Key Features

* CSV and Excel upload support
* PMS-style report cleaning, including files with metadata/header rows
* Automated and assisted column mapping
* Historical and future on-the-books data handling
* Safe demo dataset loader for Streamlit Community Cloud
* Saved local datasets and budget profiles
* Manual rooms-available override for reports missing inventory
* Validation pipeline for required fields, missing values, invalid dates, and overbooking checks
* Daily KPI calculations for occupancy, ADR, RevPAR, rooms sold, and room revenue
* Year-over-year comparison output with missing-prior-year handling
* Forecasting and backtesting workflow
* Budget-aware pricing support
* Event impact handling
* Baseline once-daily pricing recommendations
* Enhanced rate recommendation engine
* Tailored model settings and day-by-day median rate inputs
* Explainable recommendation outputs with confidence, warnings, and reasoning notes
* CSV and chart exports for reporting

## Demo Data

The repository includes safe sample files for testing, demonstration, and board review:

```text
data/
|-- sample_data.csv
|-- future_on_books_sample.csv
|-- events_sample.csv
|-- budget_daily_sample.csv
|-- budget_monthly_sample.csv

data/historical/
|-- occupancy_2024.csv
```

Additional local PMS exports, generated output folders, saved dashboard datasets, and secrets should remain outside version control.

## Web Deployment

This app is prepared for Streamlit Community Cloud.

Deployment steps:

1. Push the repository to GitHub.
2. Confirm only safe demo data is tracked.
3. In Streamlit Community Cloud, create a new app from this repository.
4. Set the app entry point to `app.py`.
5. Deploy.
6. Add the deployed URL to the **Live Demo** line above.

The deployment config is in `.streamlit/config.toml`. Secrets should be stored only in Streamlit Cloud settings or a local `.streamlit/secrets.toml` file, which is ignored by git.

## Local Installation

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

The dashboard supports demo loading, file uploads, saved local datasets, budget inputs, field mapping, tailored settings, pipeline execution, charts, and downloadable outputs.

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

## Project Structure

```text
Revenue-Management-Model/
|-- app.py
|-- main.py
|-- requirements.txt
|-- data/
|-- tests/
|-- ui/
|-- src/
|   |-- baseline.py
|   |-- budget.py
|   |-- dataset_manager.py
|   |-- elasticity.py
|   |-- evaluation.py
|   |-- events.py
|   |-- forecast.py
|   |-- ingest.py
|   |-- metrics.py
|   |-- pace.py
|   |-- pricing.py
|   |-- schema.py
|   |-- tailored.py
|   |-- validate.py
|   |-- yoy.py
|   |-- pipeline_budget_forecast.py
|   |-- pipeline_config.py
|   |-- pipeline_inputs.py
|   |-- pipeline_outputs.py
|   |-- pipeline_reporting.py
```

## Validation and Testing

Run compile checks:

```bash
python -m compileall src app.py main.py
```

Run the test suite:

```bash
python -m pytest
```

## Current Development Status

This is a working capstone and portfolio prototype. It demonstrates a full hotel revenue-management workflow from data import through explainable pricing recommendations. The project is not intended to replace a production RMS, but it is structured to show practical data cleaning, model comparison, business-rule design, validation, testing, and dashboard delivery.

## Author

Thomas Hayden
