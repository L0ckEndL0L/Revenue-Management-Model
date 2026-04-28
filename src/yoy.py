"""
yoy.py
Year-over-year comparison utilities.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def build_yoy_comparison(current_df: pd.DataFrame, historical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build year-over-year comparison table with side-by-side current vs. historical metrics.
    
    Args:
        current_df: Current year daily metrics with stay_date, occupancy, adr, rooms_available, rooms_sold, room_revenue
        historical_df: Historical year daily metrics - can be minimal (just stay_date, stly_occupancy) or full structure
    
    Returns:
        DataFrame with columns:
          calendar_date, current_occ, stly_occ, occ_variance_pct, 
          current_adr, stly_adr, adr_variance_pct,
          current_revenue, stly_revenue, revenue_variance_pct
    """
    if current_df.empty or historical_df.empty:
        return pd.DataFrame()
    
    # Ensure stay_date is datetime
    current_df = current_df.copy()
    historical_df = historical_df.copy()
    current_df["stay_date"] = pd.to_datetime(current_df["stay_date"], errors="coerce")
    historical_df["stay_date"] = pd.to_datetime(historical_df["stay_date"], errors="coerce")
    
    # Create calendar key (month-day)
    current_df["calendar_date"] = current_df["stay_date"].dt.strftime("%m-%d")
    
    # Determine occupancy column name (could be "occupancy" or "occupancy_percent")
    occ_col = "occupancy" if "occupancy" in current_df.columns else "occupancy_percent"
    
    # Aggregate current to daily if it has occupancy data
    if occ_col in current_df.columns:
        agg_dict = {occ_col: "mean"}
        if "adr" in current_df.columns:
            agg_dict["adr"] = "mean"
        if "rooms_available" in current_df.columns:
            agg_dict["rooms_available"] = "sum"
        if "rooms_sold" in current_df.columns:
            agg_dict["rooms_sold"] = "sum"
        if "room_revenue" in current_df.columns:
            agg_dict["room_revenue"] = "sum"
        
        current_daily = current_df.groupby("calendar_date", as_index=False).agg(agg_dict)
        # Rename occupancy column if needed
        if occ_col != "occupancy":
            current_daily = current_daily.rename(columns={occ_col: "occupancy"})
    else:
        return pd.DataFrame()
    
    # Handle historical data - could be minimal (just stay_date, stly_occupancy, [stly_adr], [stly_revenue]) or full
    historical_df["calendar_date"] = historical_df["stay_date"].dt.strftime("%m-%d")
    
    # Check if this is minimal historical format (stly_occupancy present, no raw occupancy column)
    is_minimal_historical = "stly_occupancy" in historical_df.columns and "occupancy" not in historical_df.columns
    
    if is_minimal_historical:
        # Minimal historical - just occupancy (and possibly ADR/revenue)
        cols_to_keep = ["calendar_date", "stly_occupancy"]
        if "stly_adr" in historical_df.columns:
            cols_to_keep.append("stly_adr")
        if "stly_revenue" in historical_df.columns:
            cols_to_keep.append("stly_revenue")
        historical_daily = historical_df[cols_to_keep].drop_duplicates()
    else:
        # Full historical - aggregate like current
        hist_agg_dict = {}
        if "occupancy" in historical_df.columns:
            hist_agg_dict["occupancy"] = "mean"
        elif "stly_occupancy" in historical_df.columns:
            hist_agg_dict["stly_occupancy"] = "first"
        if "adr" in historical_df.columns:
            hist_agg_dict["adr"] = "mean"
        if "stly_adr" in historical_df.columns:
            hist_agg_dict["stly_adr"] = "first"
        if "rooms_available" in historical_df.columns:
            hist_agg_dict["rooms_available"] = "sum"
        if "rooms_sold" in historical_df.columns:
            hist_agg_dict["rooms_sold"] = "sum"
        if "room_revenue" in historical_df.columns:
            hist_agg_dict["room_revenue"] = "sum"
        if "stly_revenue" in historical_df.columns:
            hist_agg_dict["stly_revenue"] = "first"
        
        historical_daily = historical_df.groupby("calendar_date", as_index=False).agg(hist_agg_dict)
        
        # Rename columns for consistency
        rename_map = {}
        if "occupancy" in historical_daily.columns:
            rename_map["occupancy"] = "stly_occupancy"
        if "adr" in historical_daily.columns:
            rename_map["adr"] = "stly_adr"
        if "room_revenue" in historical_daily.columns:
            rename_map["room_revenue"] = "stly_revenue"
        historical_daily = historical_daily.rename(columns=rename_map)
    
    # Merge on calendar date
    comparison = current_daily.merge(
        historical_daily,
        on="calendar_date",
        how="outer"
    )
    
    # Calculate variances
    comparison["occupancy_variance_pct"] = (
        (comparison["occupancy"] - comparison.get("stly_occupancy", np.nan)) * 100
    )
    
    # ADR variance (if available)
    if "adr" in comparison.columns and "stly_adr" in comparison.columns:
        comparison["adr_variance_pct"] = (
            ((comparison["adr"] - comparison["stly_adr"]) / comparison["stly_adr"] * 100)
            .where(comparison["stly_adr"] > 0, 0)
        )
    
    # Revenue variance (if available)
    if "room_revenue" in comparison.columns and "stly_revenue" in comparison.columns:
        comparison["revenue_variance_pct"] = (
            ((comparison["room_revenue"] - comparison["stly_revenue"]) / comparison["stly_revenue"] * 100)
            .where(comparison["stly_revenue"] > 0, 0)
        )
    
    # Format output columns - only include what we have in BOTH current and historical
    output_cols = ["calendar_date"]
    rename_map = {}
    
    if "occupancy" in comparison.columns and "stly_occupancy" in comparison.columns:
        # Convert decimal occupancy to percentage for display (0.75 -> 75.0)
        if comparison["occupancy"].dtype in ['float64', 'float32']:
            max_occ = comparison["occupancy"].max()
            if max_occ < 1.5:  # Likely decimal format, convert to percentage
                comparison["occupancy"] = comparison["occupancy"] * 100
                comparison["stly_occupancy"] = comparison["stly_occupancy"] * 100
        
        output_cols.extend(["occupancy", "stly_occupancy", "occupancy_variance_pct"])
        rename_map.update({
            "occupancy": "Current OCC %",
            "stly_occupancy": "STLY OCC %",
            "occupancy_variance_pct": "OCC Var %"
        })
    
    # Only include ADR if we have both current and historical versions
    if "adr" in comparison.columns and "stly_adr" in comparison.columns:
        if "adr_variance_pct" in comparison.columns:
            output_cols.extend(["adr", "stly_adr", "adr_variance_pct"])
            rename_map.update({
                "adr": "Current ADR",
                "stly_adr": "STLY ADR",
                "adr_variance_pct": "ADR Var %"
            })
    
    # Only include Revenue if we have both current and historical versions
    if "room_revenue" in comparison.columns and "stly_revenue" in comparison.columns:
        if "revenue_variance_pct" in comparison.columns:
            output_cols.extend(["room_revenue", "stly_revenue", "revenue_variance_pct"])
            rename_map.update({
                "room_revenue": "Current Revenue",
                "stly_revenue": "STLY Revenue",
                "revenue_variance_pct": "Revenue Var %"
            })
    
    # Filter output_cols to only include columns that actually exist
    output_cols = [col for col in output_cols if col in comparison.columns]
    
    if not output_cols:
        return pd.DataFrame()
    
    result = comparison[output_cols].copy()
    result = result.rename(columns=rename_map)
    
    return result.sort_values("calendar_date").reset_index(drop=True)


def summarize_yoy(yoy_df: pd.DataFrame) -> dict:
    """Calculate summary statistics for year-over-year comparison."""
    if yoy_df.empty:
        return {}
    
    summary = {}
    
    # Occupancy - Already in decimal format (0.75 = 75%), convert to percentage for display
    if "Current OCC %" in yoy_df.columns:
        # Column already renamed, so values are in decimal. Multiply by 100 for percentage display
        current_occ = yoy_df["Current OCC %"].mean()
        # If still in decimal format (0.75), convert to percentage (75)
        if current_occ < 1.5:  # Likely decimal format
            current_occ = current_occ * 100
        summary["avg_current_occupancy_pct"] = current_occ
    
    if "STLY OCC %" in yoy_df.columns:
        stly_occ = yoy_df["STLY OCC %"].mean()
        # If still in decimal format (0.75), convert to percentage (75)
        if stly_occ < 1.5:  # Likely decimal format
            stly_occ = stly_occ * 100
        summary["avg_stly_occupancy_pct"] = stly_occ
    
    if "avg_current_occupancy_pct" in summary and "avg_stly_occupancy_pct" in summary:
        # Use the OCC Var % column which already has individual variances calculated
        if "OCC Var %" in yoy_df.columns:
            # Filter out NaN values for accurate average
            var_col = yoy_df["OCC Var %"].dropna()
            summary["occupancy_change_pct"] = var_col.mean() if len(var_col) > 0 else 0
        else:
            summary["occupancy_change_pct"] = summary["avg_current_occupancy_pct"] - summary["avg_stly_occupancy_pct"]
    
    # ADR
    if "Current ADR" in yoy_df.columns:
        current_adr = yoy_df["Current ADR"].mean() if yoy_df["Current ADR"].dtype != 'object' else 0
        summary["avg_current_adr"] = current_adr
    else:
        summary["avg_current_adr"] = 0
        
    if "STLY ADR" in yoy_df.columns:
        stly_adr = yoy_df["STLY ADR"].mean() if yoy_df["STLY ADR"].dtype != 'object' else 0
        summary["avg_stly_adr"] = stly_adr
    else:
        summary["avg_stly_adr"] = 0
    
    if summary.get("avg_stly_adr", 0) > 0:
        summary["adr_change_pct"] = ((summary["avg_current_adr"] - summary["avg_stly_adr"]) / summary["avg_stly_adr"] * 100)
    else:
        summary["adr_change_pct"] = 0
    
    # Revenue
    if "Current Revenue" in yoy_df.columns:
        current_rev = yoy_df["Current Revenue"].sum() if yoy_df["Current Revenue"].dtype != 'object' else 0
        summary["total_current_revenue"] = current_rev
    else:
        summary["total_current_revenue"] = 0
        
    if "STLY Revenue" in yoy_df.columns:
        stly_rev = yoy_df["STLY Revenue"].sum() if yoy_df["STLY Revenue"].dtype != 'object' else 0
        summary["total_stly_revenue"] = stly_rev
    else:
        summary["total_stly_revenue"] = 0
    
    if summary.get("total_stly_revenue", 0) > 0:
        summary["revenue_change_pct"] = ((summary["total_current_revenue"] - summary["total_stly_revenue"]) / summary["total_stly_revenue"] * 100)
    else:
        summary["revenue_change_pct"] = 0
    
    return summary
