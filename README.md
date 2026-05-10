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
- Occupancy, ADR, RevPAR, and revenue calculations
- Year-over-year comparison support
- Forecasting and backtesting workflows
- Budget-aware decision support
- Baseline vs enhanced recommendation comparison
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
The README below reflects the files currently present in data and data/historical.

data/
- budget_daily_sample.csv: daily budget-style sample input
- budget_monthly_sample.csv: monthly budget-style sample input
- events_sample.csv: optional event-calendar input for demand adjustments
- future_on_books_sample.csv: future on-the-books sample input
- March 2025.csv: PMS-style historical sample file
- occupancy (1).csv: alternate occupancy sample export
- occupancy.csv: occupancy sample export
- Q2 2025.csv: PMS-style historical sample file
- Q2 2026.csv: PMS-style future-period sample file
- sample_data.csv: general sample dataset for quick testing
- historical/: supporting historical comparison files

data/historical/
- March 2025.csv
- occupancy_2024.csv
- Q2 2025.csv

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
python run_app.py
```

Alternative:

```bash
streamlit run app.py
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

## Project Outputs
The system produces outputs in the outputs directory, including:
- cleaned and standardized datasets used by the pipeline
- KPI calculations
- forecast results
- backtesting and evaluation results
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
