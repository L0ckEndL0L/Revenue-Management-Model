# Hotel Revenue Management System (RMS) Prototype

A Python + Streamlit academic prototype for hotel demand analysis, forecasting, and pricing recommendations using historical and on-the-books data.

## Project Structure

```text
hotel_rms/
├── src/                  # Core RMS modules (ingest, forecast, pricing, metrics)
├── data/                 # Example input datasets
├── figures/              # Screenshots and paper figures
├── paper/                # Final paper and supporting files
├── app.py                # Streamlit application
├── main.py               # CLI pipeline entry point
├── run_app.py            # Single-command launcher for grading
├── requirements.txt      # Python dependencies
└── README.md             # Setup and usage guide
```

## Installation

1. Open a terminal in `hotel_rms/`.
2. Create/activate a Python virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

Use the single grading command:

```bash
python run_app.py
```

This launches the Streamlit UI in your browser.

## Dataset Description

The `data/` folder contains sample datasets used by the prototype:
- `sample_data.csv`: sample historical PMS-style room and revenue records
- `future_on_books_sample.csv`: future on-the-books demand snapshot
- `events_sample.csv`: optional demand-impact events by date
- `budget_daily_sample.csv` and `budget_monthly_sample.csv`: optional budget targets
- `occupancy.csv` and `historical/`: additional historical occupancy references

## Screenshots

Place screenshots of the visualization output in `figures/`.

Suggested files:
- `figures/dashboard_overview.png`
- `figures/forecast_and_rates.png`
- `figures/yoy_comparison.png`

## Paper Files

Place the final report in `paper/` (for example `HCI_Data_Visualization_Paper.docx`).
