"""
metrics.py
Calculate hotel revenue management metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict


def calculate_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily hotel metrics: Occupancy, ADR, and RevPAR.
    
    Formulas:
    - Occupancy (OCC) = rooms_sold / rooms_available
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
    
    print("[OK] Calculated daily metrics: Occupancy, ADR, RevPAR")
    print("="*60 + "\n")
    
    return df_metrics


def aggregate_metrics(df: pd.DataFrame, freq: str = 'D') -> pd.DataFrame:
    """
    Aggregate metrics by time period.
    
    Args:
        df: DataFrame with daily metrics
        freq: Aggregation frequency ('D' for daily, 'W' for weekly, 'M' for monthly)
        
    Returns:
        Aggregated DataFrame
    """
    if 'stay_date' not in df.columns:
        raise ValueError("DataFrame must contain 'stay_date' column")
    
    # Group by time period
    df_agg = df.groupby(pd.Grouper(key='stay_date', freq=freq)).agg({
        'rooms_available': 'sum',
        'rooms_sold': 'sum',
        'room_revenue': 'sum'
    }).reset_index()
    
    # Recalculate metrics for aggregated data
    df_agg['occupancy'] = np.where(
        df_agg['rooms_available'] > 0,
        df_agg['rooms_sold'] / df_agg['rooms_available'],
        0.0
    )
    
    df_agg['adr'] = np.where(
        df_agg['rooms_sold'] > 0,
        df_agg['room_revenue'] / df_agg['rooms_sold'],
        0.0
    )
    
    df_agg['revpar'] = np.where(
        df_agg['rooms_available'] > 0,
        df_agg['room_revenue'] / df_agg['rooms_available'],
        0.0
    )
    
    return df_agg


def calculate_summary_statistics(df: pd.DataFrame) -> Dict:
    """
    Calculate summary statistics for the entire dataset.
    
    Args:
        df: DataFrame with daily metrics
        
    Returns:
        Dictionary of summary statistics
    """
    # Total metrics
    total_rooms_sold = df['rooms_sold'].sum()
    total_room_revenue = df['room_revenue'].sum()
    total_rooms_available = df['rooms_available'].sum()
    
    # Average metrics (weighted by actual data)
    avg_occupancy = total_rooms_sold / total_rooms_available if total_rooms_available > 0 else 0.0
    avg_adr = total_room_revenue / total_rooms_sold if total_rooms_sold > 0 else 0.0
    avg_revpar = total_room_revenue / total_rooms_available if total_rooms_available > 0 else 0.0
    
    # Date range
    min_date = df['stay_date'].min()
    max_date = df['stay_date'].max()
    num_days = (max_date - min_date).days + 1
    
    summary = {
        'date_range': {
            'start': min_date,
            'end': max_date,
            'num_days': num_days
        },
        'totals': {
            'rooms_sold': int(total_rooms_sold),
            'room_revenue': float(total_room_revenue),
            'rooms_available': int(total_rooms_available)
        },
        'averages': {
            'occupancy': float(avg_occupancy),
            'adr': float(avg_adr),
            'revpar': float(avg_revpar)
        }
    }
    
    return summary


def print_summary(summary: Dict) -> None:
    """
    Print summary statistics in a formatted way.
    
    Args:
        summary: Dictionary of summary statistics
    """
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Date range
    dr = summary['date_range']
    print(f"\nDate Range:")
    print(f"  From: {dr['start'].strftime('%Y-%m-%d')}")
    print(f"  To:   {dr['end'].strftime('%Y-%m-%d')}")
    print(f"  Days: {dr['num_days']}")
    
    # Totals
    tot = summary['totals']
    print(f"\nTotals:")
    print(f"  Rooms Available: {tot['rooms_available']:,}")
    print(f"  Rooms Sold:      {tot['rooms_sold']:,}")
    print(f"  Room Revenue:    ${tot['room_revenue']:,.2f}")
    
    # Averages
    avg = summary['averages']
    print(f"\nAverage Metrics:")
    print(f"  Occupancy: {avg['occupancy']*100:.2f}%")
    print(f"  ADR:       ${avg['adr']:.2f}")
    print(f"  RevPAR:    ${avg['revpar']:.2f}")
    
    print("="*60 + "\n")


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
        'occupancy', 'adr', 'revpar'
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
    df_export['adr'] = df_export['adr'].round(2)
    df_export['revpar'] = df_export['revpar'].round(2)
    df_export['room_revenue'] = df_export['room_revenue'].round(2)
    
    df_export.to_csv(output_path, index=False)
    print(f"[OK] Metrics exported to: {output_path}")


def generate_weekly_monthly_aggregates(df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate and export weekly and monthly aggregates if there's enough data.
    
    Args:
        df: DataFrame with daily metrics
        output_dir: Directory to save aggregate files
    """
    # Check if we have enough data for aggregates
    num_days = (df['stay_date'].max() - df['stay_date'].min()).days + 1
    
    if num_days >= 7:
        print("\nGenerating weekly aggregates...")
        df_weekly = aggregate_metrics(df, freq='W')
        weekly_path = f"{output_dir}/weekly_metrics.csv"
        export_metrics(df_weekly, weekly_path)
    
    if num_days >= 28:
        print("Generating monthly aggregates...")
        df_monthly = aggregate_metrics(df, freq='ME')
        monthly_path = f"{output_dir}/monthly_metrics.csv"
        export_metrics(df_monthly, monthly_path)
