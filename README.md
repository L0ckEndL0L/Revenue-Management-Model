# Flexible Revenue Management System for Hotel Pricing Decision Support

## Overview
This repository contains a Python and Streamlit prototype for hotel revenue management decision support. The system is designed to ingest historical and forward-looking operating data, standardize inconsistent report formats, and generate explainable, budget-aware pricing recommendations for analysis and planning.

## Problem Statement
Many hotels still rely on static, manual, or once-daily pricing updates. These workflows may not react quickly to occupancy pace, budget gaps, local demand shifts, or inconsistent PMS exports. As a result, pricing decisions can be delayed, difficult to explain, or disconnected from current booking conditions.

## Proposed Solution
This prototype provides an end-to-end workflow that:
- imports hotel operational and planning data,
- cleans and validates PMS-style files,
- maps inconsistent column names into a standard schema,
- calculates key hotel performance metrics,
- supports forecasting and backtesting,
- compares baseline and enhanced pricing strategies, and
- produces explainable pricing recommendations.

## Key Features
- CSV and Excel upload support
- PMS-style data cleaning for inconsistent exports
- Automated and assisted column/schema mapping
- Occupancy, Occupancy %, ADR, RevPAR, and revenue calculations
- Year-over-year comparison support
- Forecasting and backtesting workflows
- Budget-aware decision support
- Baseline vs enhanced recommendation comparison
- Once-daily baseline pricing model for capstone comparison
- Explainable pricing recommendations
- Streamlit dashboard interface
- CLI pipeline support

## Methodology
1. Import hotel data
2. Clean and validate the data
3. Map columns to a standard schema
4. Calculate hotel KPIs
5. Run forecasting and backtesting
6. Compare baseline and enhanced pricing recommendations
7. Generate explainable pricing output

## Baseline Comparison Model (Capstone)
The project includes a simple once-daily baseline pricing model used as the comparison arm for the capstone hypothesis.

Purpose:
- provide a transparent, replicable benchmark,
- avoid advanced RMS-only signals, and
- support measurable comparison against the proposed RMS recommendation engine.

Model behavior:
- uses standard daily PMS-style inputs (stay date, rooms available/sold, occupancy, ADR/RevPAR context, room revenue, day of week),
- applies simple occupancy-threshold rules (high occupancy: modest increase, low occupancy: modest decrease, moderate occupancy: hold),
- can reference prior historical ADR by day-of-week when available,
- returns clear unavailable status for rows where occupancy/ADR cannot be derived safely.

The baseline intentionally does not use intraday comp-set refreshes, event-driven pricing logic, or budget-aware optimization.

This baseline enables later evaluation against the RMS model using metrics such as MAE, RMSE, MAPE, RevPAR impact, and related uplift comparisons.

## Data Requirements
Minimum required fields are:
- stay_date
- rooms_available
- rooms_sold
- room_revenue

Optional but supported fields include:
- adr
- current_rate
- occupancy_percent
- booking_date

Column names may vary by PMS export. The system attempts to map common aliases automatically, and the Streamlit interface allows manual mapping when needed.

The repository includes example input files in the data folder. You may also upload your own CSV or XLSX files through the Streamlit app.

## Data Folder Consistency
Tracked demo files in this repository:

data/
- sample_data.csv: baseline historical sample for quick pipeline testing
- future_on_books_sample.csv: future on-the-books sample input
- events_sample.csv: optional event-calendar sample input
- budget_daily_sample.csv: daily budget sample
- budget_monthly_sample.csv: monthly budget sample

data/historical/
- occupancy_2024.csv: historical comparison sample used for YoY and pace workflows

Additional local CSV exports may exist in your working directory for experimentation. Those local files are excluded from version control by design.

## How to Run
1. Clone the repository:

```bash
git clone https://github.com/L0ckEndL0L/Revenue-Management-Model.git
cd Revenue-Management-Model
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit application:

```bash
python -m streamlit run app.py
```

5. Run the CLI pipeline (supported):

```bash
python main.py --input data/sample_data.csv --output outputs
```

Optional CLI arguments include:
- --future
- --events
- --budget
- --manual_rooms_available
- --elasticity

## Quality Checks
Run basic compile and test checks:

```bash
python -m compileall src app.py main.py
python -m pytest
```

## Project Outputs
The system produces outputs in the outputs directory, including:
- cleaned and standardized datasets used by the pipeline
- KPI calculations
- forecast results
- backtesting and evaluation results
- baseline once-daily pricing recommendations (`baseline_recommendations.csv`)
- baseline vs enhanced recommendation comparison tables
- pricing recommendation explanations and priority outputs
- chart images and dashboard views when generated

## Current Development Status
This version is a working prototype developed for capstone research and portfolio demonstration. It is suitable for demonstrating workflow integration, decision-support logic, and explainable recommendation outputs.

## Future Improvements
- More robust automated testing coverage
- Expanded safe sample datasets for demonstrations
- Improved event demand signal engineering
- Additional model evaluation and benchmarking
- Curated dashboard screenshots for reporting
- Deployment options for hosted demonstrations

## Author
Thomas Hayden
