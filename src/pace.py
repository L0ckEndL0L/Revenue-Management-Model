"""
pace.py
Booking pace and historical comparison utilities for the RMS pricing logic.
"""

from __future__ import annotations

from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd


def _daily_base(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate canonical data to one row per stay_date with occupancy and ADR."""
    grouped = (
        df.groupby("stay_date", as_index=False)
        .agg(
            rooms_available=("rooms_available", "sum"),
            rooms_sold=("rooms_sold", "sum"),
            room_revenue=("room_revenue", "sum"),
        )
        .sort_values("stay_date")
        .reset_index(drop=True)
    )

    grouped["occupancy"] = np.where(
        grouped["rooms_available"] > 0,
        grouped["rooms_sold"] / grouped["rooms_available"],
        0.0,
    )
    grouped["adr"] = np.where(
        grouped["rooms_sold"] > 0,
        grouped["room_revenue"] / grouped["rooms_sold"],
        0.0,
    )
    return grouped


def load_historical_data(historical_dir: str | Path = "data/historical") -> Optional[pd.DataFrame]:
    """
    Load and combine all historical occupancy CSV files from a directory.
    
    Expects files named like occupancy_YYYY.csv or similar.
    Returns aggregated DataFrame with columns: stay_date, stly_occupancy.
    Returns None if no files found.
    """
    from src.ingest import load_file, convert_numeric_columns
    from src.schema import auto_map_columns, apply_column_mapping
    
    historical_dir = Path(historical_dir)
    if not historical_dir.exists():
        print(f"[DEBUG] Historical directory not found: {historical_dir}")
        return None
    
    csv_files = list(historical_dir.glob("*.csv"))
    print(f"[DEBUG] Found {len(csv_files)} CSV files in {historical_dir}")
    if not csv_files:
        return None
    
    dfs = []
    for csv_file in csv_files:
        try:
            print(f"[DEBUG] Loading historical file: {csv_file.name}")
            df = load_file(str(csv_file))
            if df is not None and len(df) > 0:
                print(f"[DEBUG] Loaded {len(df)} rows, columns: {list(df.columns)}")
                
                # Apply column mapping to normalize column names
                print(f"[DEBUG] Applying column mapping to {csv_file.name}")
                mapping = auto_map_columns(df)
                print(f"[DEBUG] Column mapping found: {mapping}")
                
                # Check if we have required columns mapped
                if 'rooms_available' not in mapping:
                    print(f"[WARNING] No rooms_available mapped for {csv_file.name}, trying to derive")
                    # Try to derive rooms_available if occupancy and rooms_sold exist
                    if 'rooms_sold' in mapping and 'occupancy_percent' in mapping:
                        print(f"[DEBUG] Can derive rooms_available from rooms_sold and occupancy_percent")
                
                df = apply_column_mapping(df, mapping)
                print(f"[DEBUG] After mapping, columns: {list(df.columns)}")
                
                # Clean numeric columns
                df = convert_numeric_columns(df)
                print(f"[DEBUG] After numeric conversion, columns: {list(df.columns)}")
                
                dfs.append(df)
            else:
                print(f"[DEBUG] File {csv_file.name} is empty or None")
        except Exception as e:
            import traceback
            print(f"[ERROR] Could not load historical file {csv_file.name}: {e}")
            traceback.print_exc()
            continue
    
    if not dfs:
        print("[DEBUG] No historical data frames loaded")
        return None
    
    # Combine all historical files
    historical = pd.concat(dfs, ignore_index=True)
    print(f"[DEBUG] Combined {len(dfs)} files into {len(historical)} total rows")
    
    # Normalize to daily occupancy by date
    if "stay_date" not in historical.columns:
        print(f"[DEBUG] 'stay_date' column not found. Available columns: {list(historical.columns)}")
        return None
    
    # Ensure stay_date is datetime
    historical["stay_date"] = pd.to_datetime(historical["stay_date"], errors="coerce")
    historical = historical.dropna(subset=["stay_date"])

    # Deduplicate: if multiple files cover the same date (e.g. a monthly snapshot and a
    # quarterly report both containing the same days), keep only one row per stay_date
    # to prevent double-counting rooms/revenue in the STLY calculations.
    # "last" keeps the row from the file loaded latest — typically the most recent snapshot.
    before_dedup = len(historical)
    historical = historical.sort_values("stay_date").drop_duplicates(subset=["stay_date"], keep="last").reset_index(drop=True)
    if len(historical) < before_dedup:
        print(f"[DEBUG] Deduplicated {before_dedup - len(historical)} duplicate stay_date rows across historical files")

    # Derive rooms_available if missing (from occupancy % and rooms_sold)
    print(f"[DEBUG] Checking for rooms_available...")
    if "rooms_available" not in historical.columns:
        print(f"[DEBUG] rooms_available not found, attempting to derive from occupancy and rooms_sold")
        if "occupancy_percent" in historical.columns and "rooms_sold" in historical.columns:
            print(f"[DEBUG] Deriving rooms_available from occupancy_percent and rooms_sold")
            # occupancy_percent = rooms_sold / rooms_available
            # rooms_available = rooms_sold / occupancy_percent
            historical["rooms_available"] = np.where(
                historical["occupancy_percent"] > 0,
                historical["rooms_sold"] / historical["occupancy_percent"],
                0
            )
            print(f"[DEBUG] Derived rooms_available values")
        else:
            print(f"[WARNING] Cannot derive rooms_available - missing occupancy_percent or rooms_sold")
            return None
    
    # Aggregate to daily occupancy
    print(f"[DEBUG] Aggregating {len(historical)} rows to daily")
    agg_dict = {
        "rooms_available": "sum",
        "rooms_sold": "sum",
    }
    
    # Include ADR if available
    if "adr" in historical.columns:
        agg_dict["adr"] = "mean"
        print(f"[DEBUG] Including ADR in aggregation")
    
    # Include room_revenue if available
    if "room_revenue" in historical.columns:
        agg_dict["room_revenue"] = "sum"
        print(f"[DEBUG] Including room_revenue in aggregation")
    
    daily_hist = historical.groupby("stay_date", as_index=False).agg(agg_dict)
    
    # Calculate occupancy
    daily_hist["stly_occupancy"] = np.where(
        daily_hist["rooms_available"] > 0,
        daily_hist["rooms_sold"] / daily_hist["rooms_available"],
        0.0,
    )
    
    # Rename ADR and revenue columns for consistency
    if "adr" in daily_hist.columns:
        daily_hist = daily_hist.rename(columns={"adr": "stly_adr"})
    if "room_revenue" in daily_hist.columns:
        daily_hist = daily_hist.rename(columns={"room_revenue": "stly_revenue"})

    # Rename rooms_sold to stly_rooms_sold so callers can derive STLY OTB room nights
    # at the same booking-window cutoff (i.e. what was on books last year at this point).
    if "rooms_sold" in daily_hist.columns:
        daily_hist = daily_hist.rename(columns={"rooms_sold": "stly_rooms_sold"})

    print(f"[DEBUG] Final historical data: {len(daily_hist)} days with occupancy")
    
    # Return all available data
    output_cols = ["stay_date", "stly_occupancy"]
    if "stly_rooms_sold" in daily_hist.columns:
        output_cols.append("stly_rooms_sold")
    if "stly_adr" in daily_hist.columns:
        output_cols.append("stly_adr")
    if "stly_revenue" in daily_hist.columns:
        output_cols.append("stly_revenue")
    
    return daily_hist[output_cols].copy()


def calculate_pace_analysis(
    df_clean: pd.DataFrame, 
    historical_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Build a pace analysis frame with optional STLY comparison.

    Args:
        df_clean: Current year data with stay_date, rooms_available, rooms_sold, room_revenue
        historical_df: Optional DataFrame with stay_date and stly_occupancy columns

    Returns DataFrame with columns:
      stay_date, current_adr, occupancy, stly_occupancy, pace_variance,
      under_pacing, over_pacing, has_historical
    """
    if "stay_date" not in df_clean.columns:
        raise ValueError("Input data must contain 'stay_date'")

    current = _daily_base(df_clean)
    current = current.rename(columns={"adr": "current_adr"})

    # First, try to use provided historical data
    if historical_df is not None and len(historical_df) > 0:
        # Normalize stay_date column names in historical
        if "stay_date" in historical_df.columns:
            # Match by calendar month/day, allowing year mismatch
            current["calendar_key"] = current["stay_date"].dt.strftime("%m-%d")
            hist_match = historical_df.copy()
            hist_match["calendar_key"] = hist_match["stay_date"].dt.strftime("%m-%d")
            hist_match = hist_match[["calendar_key", "stly_occupancy"]].drop_duplicates()
            
            result = current.merge(hist_match, on="calendar_key", how="left")
            result = result.drop(columns=["calendar_key"])
        else:
            result = current.copy()
            result["stly_occupancy"] = np.nan
    else:
        # Fallback: try looking forward in time (year+1 within same input)
        hist = current[["stay_date", "occupancy"]].copy()
        hist["stay_date"] = hist["stay_date"] + pd.DateOffset(years=1)
        hist = hist.rename(columns={"occupancy": "stly_occupancy"})

        result = current.merge(hist, on="stay_date", how="left")

    result["pace_variance"] = result["occupancy"] - result["stly_occupancy"]
    result["has_historical"] = result["stly_occupancy"].notna()
    result["under_pacing"] = result["has_historical"] & (result["pace_variance"] < 0)
    result["over_pacing"] = result["has_historical"] & (result["pace_variance"] > 0)

    # Keep NaN for missing historical values for transparent output.
    return result[
        [
            "stay_date",
            "current_adr",
            "occupancy",
            "stly_occupancy",
            "pace_variance",
            "under_pacing",
            "over_pacing",
            "has_historical",
        ]
    ].copy()
