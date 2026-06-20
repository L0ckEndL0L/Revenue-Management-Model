# RateAnchor

RateAnchor is a Streamlit capstone project for hotel revenue-management analysis. It cleans PMS-style data, validates required fields, forecasts demand, compares baseline and tailored pricing logic, and produces explainable rate recommendations that can be reviewed in a short advisor demo.

## Main Features

* One-click demo mode using sample files included in the repository
* CSV and Excel upload support for historical PMS and future on-books reports
* PMS report cleaning for metadata rows, repeated headers, malformed CSVs, and common export quirks
* Automated column mapping with manual fallback prompts in the UI
* Data validation for required fields, invalid dates, missing values, negative values, and overbooking
* Daily KPI output for occupancy, ADR, RevPAR, rooms sold, and room revenue
* Future demand forecast and pricing simulation
* Baseline recommendation output for comparison
* Tailored recommendation settings with date-level median-rate controls
* Property-type presets that change operating posture for luxury, resort, boutique, full-service, limited-service, select-service, extended-stay, and economy properties
* Segment-focus presets that materially change tailored pricing posture for revenue, occupancy, corporate, group, premium, and leisure strategies
* Dedicated comp-rate tab with monthly, daily, and mock comp-set median-rate inputs
* Property/dataset switcher for moving between the demo property and saved property datasets
* Event impact handling when event data is available
* Budget comparison when daily or monthly budget data is available
* Year-over-year comparison when prior-year reference data is available
* Downloadable CSV, chart, and ZIP outputs

## Five-Minute Demo

1. Run the app or open the deployed Streamlit app.
2. In the sidebar, click **Load Demo Dataset**.
3. Confirm the historical and future on-books previews appear.
4. Click **Run Pricing Simulation**.
5. Review Forecast, Baseline, Tailored Model, Rate Recommendations, YoY, Charts, and Downloads.

Demo mode does not require manual uploads. Optional demo files such as events and budget data are loaded when present and skipped with a warning when absent.
Use the **Comp Rate Controls** tab when you want to switch between one monthly comp-set median rate, separate daily comp rates, or the included mock comp-set rate-shop file.
Use the **Property Switcher** in the sidebar to move between the included demo property and saved property datasets.

## Required Demo Files

These files are required for demo mode:

```text
data/sample_data.csv
data/future_on_books_sample.csv
```

These files are optional and add richer outputs:

```text
data/events_sample.csv
data/budget_daily_sample.csv
data/budget_monthly_sample.csv
data/comp_set_sample.csv
data/historical/occupancy_2024.csv
```

The demo loader resolves these files relative to the project root, so it works locally and on Streamlit Community Cloud without hardcoded local paths.

## Run Locally

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

Run the Streamlit app:

```bash
streamlit run app.py
```

You can also run it through Python:

```bash
python -m streamlit run app.py
```

## CLI Pipeline

Basic run:

```bash
python main.py --input data/sample_data.csv --output outputs
```

Run with demo future on-books, events, and budget files:

```bash
python main.py \
  --input data/sample_data.csv \
  --future data/future_on_books_sample.csv \
  --events data/events_sample.csv \
  --budget data/budget_daily_sample.csv \
  --output outputs \
  --no-interactive
```

Common options:

```text
--input                     Historical PMS report. Required.
--future                    Optional future on-books report.
--events                    Optional events CSV.
--budget                    Optional daily or monthly budget file.
--output                    Output directory. Defaults to outputs.
--rate_floor                Minimum allowed recommendation rate.
--rate_ceiling              Maximum allowed recommendation rate.
--max_change_pct            Maximum daily rate movement.
--elasticity                Demand elasticity assumption.
--manual_rooms_available    Manual inventory override when reports omit rooms_available.
--allow-overbooking         Allow rooms_sold to exceed rooms_available during validation.
--no-interactive            Disable terminal mapping prompts.
```

## Streamlit Community Cloud Deployment

1. Push this repository to GitHub.
2. Confirm `app.py`, `requirements.txt`, `src/`, `ui/`, `tests/`, and safe demo files under `data/` are committed.
3. In Streamlit Community Cloud, create a new app from the repository.
4. Set the app entry point to `app.py`.
5. Deploy.
6. Open the deployed app and use **Load Demo Dataset** to confirm the Cloud build can run without local files.

No local `C:/Users/...` paths are required. Secrets are not needed for the demo workflow. If secrets are added later, store them in Streamlit Cloud settings or a local `.streamlit/secrets.toml` file that remains out of git.

## Output Files

Each run creates a timestamped folder under `outputs/`. Common outputs include:

```text
cleaned_data.csv
daily_metrics.csv
validation_report.txt
forecast.csv
yoy_comparison.csv
baseline_recommendations.csv
rate_recommendations.csv
tailored_model_results.csv
tailored_model_summary.csv
baseline_vs_new_policy.csv
evaluation_metrics.csv
forecast_vs_actual.csv
top_raise_opportunities.csv
top_rescue_dates.csv
top_monitor_dates.csv
current_vs_recommended_rate.png
expected_revenue_uplift.png
priority_score_by_date.png
forecast_vs_actual.png
```

## Project Structure

```text
Revenue-Management-Model/
|-- app.py
|-- main.py
|-- requirements.txt
|-- README.md
|-- data/
|-- src/
|-- tests/
|-- ui/
```

## Testing

Run the test suite:

```bash
python -m pytest -q
```

Useful compile check:

```bash
python -m compileall src ui app.py main.py
```

## Current Project Status

RateAnchor is ready for advisor review as a capstone prototype. It is designed to demonstrate data cleaning, validation, forecasting, baseline comparison, tailored pricing logic, budget context, event context, YoY comparison, and Streamlit delivery. It is not intended to replace a production RMS.

## Week 9 Completed Work Summary

* Fixed demo mode so included sample files can run the pricing simulation without manual uploads.
* Ensured demo file paths are resolved relative to the repository root for local and Streamlit Community Cloud runs.
* Added required-file errors and optional-file warnings for demo loading.
* Preserved loaded dataset mappings through the same session-state keys used by uploaded data.
* Kept validation intact while making static on-books demo files usable as simulation inputs.
* Made month-mismatched optional budget files non-blocking for the pricing simulation.
* Added model behavior checks for sparse-history and stronger-history properties, with low-data forecast fallback logic.
* Added mock comp-set data, comp-set median ingestion, and saved dataset support for comp-set files.
* Added a property/dataset switcher so reviewers can move between the demo property and saved property datasets.
* Updated app copy to use the RateAnchor name consistently.
* Added regression tests for demo loading and sample pipeline compatibility.

## Author

Thomas Hayden
