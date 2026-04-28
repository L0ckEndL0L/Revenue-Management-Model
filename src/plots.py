"""
plots.py
Generate charts for hotel metrics visualization.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


def setup_plot_style():
    """Configure matplotlib style settings."""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3


def plot_occupancy(df: pd.DataFrame, output_path: str) -> None:
    """
    Plot occupancy over time.
    
    Args:
        df: DataFrame with daily metrics
        output_path: Path to save the chart
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot occupancy as percentage
    ax.plot(df['stay_date'], df['occupancy'] * 100, 
            marker='o', markersize=4, linewidth=2, color='#2E86AB')
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Occupancy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Hotel Occupancy Over Time', fontsize=14, fontweight='bold', pad=20)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    ax.set_ylim(0, max(105, df['occupancy'].max() * 105))
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    # Rotate date labels
    plt.xticks(rotation=45, ha='right')
    
    # Add horizontal line at 100%
    ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, linewidth=1, label='100% Occupancy')
    
    # Add legend
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Occupancy chart saved to: {output_path}")


def plot_adr(df: pd.DataFrame, output_path: str) -> None:
    """
    Plot Average Daily Rate (ADR) over time.
    
    Args:
        df: DataFrame with daily metrics
        output_path: Path to save the chart
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot ADR
    ax.plot(df['stay_date'], df['adr'], 
            marker='o', markersize=4, linewidth=2, color='#A23B72')
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('ADR ($)', fontsize=12, fontweight='bold')
    ax.set_title('Average Daily Rate (ADR) Over Time', fontsize=14, fontweight='bold', pad=20)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${y:.0f}'))
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    # Rotate date labels
    plt.xticks(rotation=45, ha='right')
    
    # Add average line
    avg_adr = df['adr'].mean()
    ax.axhline(y=avg_adr, color='green', linestyle='--', alpha=0.5, linewidth=1, 
               label=f'Average: ${avg_adr:.2f}')
    
    # Add legend
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] ADR chart saved to: {output_path}")


def plot_revpar(df: pd.DataFrame, output_path: str) -> None:
    """
    Plot Revenue Per Available Room (RevPAR) over time.
    
    Args:
        df: DataFrame with daily metrics
        output_path: Path to save the chart
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot RevPAR
    ax.plot(df['stay_date'], df['revpar'], 
            marker='o', markersize=4, linewidth=2, color='#F18F01')
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('RevPAR ($)', fontsize=12, fontweight='bold')
    ax.set_title('Revenue Per Available Room (RevPAR) Over Time', fontsize=14, fontweight='bold', pad=20)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${y:.0f}'))
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    # Rotate date labels
    plt.xticks(rotation=45, ha='right')
    
    # Add average line
    avg_revpar = df['revpar'].mean()
    ax.axhline(y=avg_revpar, color='green', linestyle='--', alpha=0.5, linewidth=1, 
               label=f'Average: ${avg_revpar:.2f}')
    
    # Add legend
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] RevPAR chart saved to: {output_path}")


def plot_combined_metrics(df: pd.DataFrame, output_path: str) -> None:
    """
    Plot all three key metrics in a combined chart with subplots.
    
    Args:
        df: DataFrame with daily metrics
        output_path: Path to save the chart
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # Occupancy
    axes[0].plot(df['stay_date'], df['occupancy'] * 100, 
                 marker='o', markersize=3, linewidth=2, color='#2E86AB')
    axes[0].set_ylabel('Occupancy (%)', fontsize=11, fontweight='bold')
    axes[0].set_title('Hotel Key Metrics Over Time', fontsize=14, fontweight='bold', pad=20)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    axes[0].axhline(y=100, color='red', linestyle='--', alpha=0.3, linewidth=1)
    axes[0].grid(True, alpha=0.3)
    
    # ADR
    axes[1].plot(df['stay_date'], df['adr'], 
                 marker='o', markersize=3, linewidth=2, color='#A23B72')
    axes[1].set_ylabel('ADR ($)', fontsize=11, fontweight='bold')
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${y:.0f}'))
    avg_adr = df['adr'].mean()
    axes[1].axhline(y=avg_adr, color='green', linestyle='--', alpha=0.3, linewidth=1)
    axes[1].grid(True, alpha=0.3)
    
    # RevPAR
    axes[2].plot(df['stay_date'], df['revpar'], 
                 marker='o', markersize=3, linewidth=2, color='#F18F01')
    axes[2].set_ylabel('RevPAR ($)', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Date', fontsize=11, fontweight='bold')
    axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${y:.0f}'))
    avg_revpar = df['revpar'].mean()
    axes[2].axhline(y=avg_revpar, color='green', linestyle='--', alpha=0.3, linewidth=1)
    axes[2].grid(True, alpha=0.3)
    
    # Format x-axis dates for all subplots
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    # Rotate date labels for bottom plot
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Combined metrics chart saved to: {output_path}")


def generate_all_charts(df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate all required charts.
    
    Args:
        df: DataFrame with daily metrics
        output_dir: Directory to save charts
    """
    print("\n" + "="*60)
    print("GENERATING CHARTS")
    print("="*60 + "\n")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Individual charts
    plot_occupancy(df, str(output_path / 'occupancy_by_date.png'))
    plot_adr(df, str(output_path / 'adr_by_date.png'))
    plot_revpar(df, str(output_path / 'revpar_by_date.png'))
    
    # Combined chart (bonus)
    plot_combined_metrics(df, str(output_path / 'combined_metrics.png'))
    
    print("\n" + "="*60)
    print("Chart generation complete")
    print("="*60 + "\n")
