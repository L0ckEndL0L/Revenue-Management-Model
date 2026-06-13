"""
metrics.py
Calculate hotel revenue management metrics.
"""

import numpy as np
import pandas as pd


def calculate_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily hotel metrics: Occupancy, Occupancy %, ADR, and RevPAR.
    
    Formulas:
    - Occupancy (OCC) = rooms_sold / rooms_available
    - Occupancy % = Occupancy * 100
    - ADR (Average Daily Rate) = room_revenue / rooms_sold
    - RevPAR (Revenue Per Available Room) = room_revenue / rooms_available
    
    Args:
        df: DataFrame with canonical columns
        
    Returns:
        DataFrame with daily metrics
    """
    print("\n" + "="*60)
    print("CALCULATING METRICS")
    print("="*60)
    
    df_metrics = df.copy()
    df_metrics['room_revenue'] = pd.to_numeric(df_metrics['room_revenue'], errors='coerce').fillna(0.0)
    
    # Calculate Occupancy (rooms_sold / rooms_available)
    # Handle division by zero: if rooms_available = 0, set occupancy to 0
    df_metrics['occupancy'] = np.where(
        df_metrics['rooms_available'] > 0,
        df_metrics['rooms_sold'] / df_metrics['rooms_available'],
        0.0
    )
    
    # Calculate Occupancy Percentage (occupancy * 100)
    df_metrics['occupancy_pct'] = df_metrics['occupancy'] * 100.0
    
    # Calculate ADR (room_revenue / rooms_sold)
    # Handle division by zero: if rooms_sold = 0, set ADR to 0
    df_metrics['adr'] = np.where(
        df_metrics['rooms_sold'] > 0,
        df_metrics['room_revenue'] / df_metrics['rooms_sold'],
        0.0
    )
    
    # Calculate RevPAR (room_revenue / rooms_available)
    # Handle division by zero: if rooms_available = 0, set RevPAR to 0
    df_metrics['revpar'] = np.where(
        df_metrics['rooms_available'] > 0,
        df_metrics['room_revenue'] / df_metrics['rooms_available'],
        0.0
    )
    
    # Sort by stay_date
    df_metrics = df_metrics.sort_values('stay_date').reset_index(drop=True)
    
    print("[OK] Calculated daily metrics: Occupancy, Occupancy%, ADR, RevPAR")
    print("="*60 + "\n")
    
    return df_metrics


def export_metrics(df: pd.DataFrame, output_path: str) -> None:
    """
    Export metrics to CSV file.
    
    Args:
        df: DataFrame with daily metrics
        output_path: Path to save the CSV file
    """
    # Select columns to export
    columns_to_export = [
        'stay_date', 'rooms_available', 'rooms_sold', 'room_revenue',
        'occupancy', 'occupancy_pct', 'adr', 'revpar'
    ]
    
    # Add optional columns if present
    optional_cols = ['room_type', 'rate_code', 'channel', 'booking_date']
    for col in optional_cols:
        if col in df.columns:
            columns_to_export.insert(-3, col)  # Insert before metrics
    
    df_export = df[columns_to_export].copy()
    
    # Format date columns
    df_export['stay_date'] = df_export['stay_date'].dt.strftime('%Y-%m-%d')
    if 'booking_date' in df_export.columns:
        df_export['booking_date'] = pd.to_datetime(df_export['booking_date']).dt.strftime('%Y-%m-%d')
    
    # Round numeric columns
    df_export['occupancy'] = df_export['occupancy'].round(4)
    df_export['occupancy_pct'] = df_export['occupancy_pct'].round(2)
    df_export['adr'] = df_export['adr'].round(2)
    df_export['revpar'] = df_export['revpar'].round(2)
    df_export['room_revenue'] = df_export['room_revenue'].round(2)
    
    df_export.to_csv(output_path, index=False)
    print(f"[OK] Metrics exported to: {output_path}")
