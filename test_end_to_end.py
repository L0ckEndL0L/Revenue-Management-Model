#!/usr/bin/env python
"""Integration test - verify end-to-end pipeline works with CSV data."""

import pandas as pd
import tempfile
from pathlib import Path
from src.ingest import process_file
from src.validate import validate_data
from src.metrics import calculate_daily_metrics, export_metrics

# Create temporary CSV file with test data
test_data = """stay_date,rooms_available,rooms_sold,room_revenue
2026-03-01,100,80,9600.00
2026-03-02,100,90,11700.00
2026-03-03,100,70,7700.00
"""

with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
    f.write(test_data)
    csv_path = f.name

try:
    print("=" * 60)
    print("END-TO-END PIPELINE TEST")
    print("=" * 60)
    
    # Step 1: Ingest and map CSV
    print("\nStep 1: Process CSV file...")
    df = process_file(csv_path, interactive=False)
    print(f"  Loaded {len(df)} rows")
    
    # Step 2: Validate
    print("\nStep 2: Validate data...")
    df_valid, result = validate_data(df, allow_overbooking=False)
    print(f"  Valid rows: {result.valid_rows}")
    
    # Step 3: Calculate metrics
    print("\nStep 3: Calculate KPIs...")
    df_metrics = calculate_daily_metrics(df_valid)
    
    # Step 4: Export
    print("\nStep 4: Export metrics...")
    output_csv = str(Path(tempfile.gettempdir()) / "test_metrics.csv")
    export_metrics(df_metrics, output_csv)
    
    # Step 5: Verify output
    print("\nStep 5: Verify exported data...")
    df_exported = pd.read_csv(output_csv)
    print(f"  Exported columns: {list(df_exported.columns)}")
    
    # Verify occupancy_pct is present
    assert 'occupancy_pct' in df_exported.columns, "occupancy_pct column missing!"
    assert 'occupancy' in df_exported.columns, "occupancy column missing!"
    assert 'adr' in df_exported.columns, "adr column missing!"
    assert 'revpar' in df_exported.columns, "revpar column missing!"
    
    # Check values
    print(f"\n  Row 1 (2026-03-01):")
    print(f"    occupancy_pct: {df_exported.loc[0, 'occupancy_pct']:.2f} (expected: 80.00)")
    print(f"    adr: {df_exported.loc[0, 'adr']:.2f} (expected: 120.00)")
    print(f"    revpar: {df_exported.loc[0, 'revpar']:.2f} (expected: 96.00)")
    
    assert abs(df_exported.loc[0, 'occupancy_pct'] - 80.00) < 0.01
    assert abs(df_exported.loc[0, 'adr'] - 120.00) < 0.01
    assert abs(df_exported.loc[0, 'revpar'] - 96.00) < 0.01
    
    print("\n" + "=" * 60)
    print("✓ END-TO-END PIPELINE TEST PASSED!")
    print("=" * 60)
    
finally:
    # Cleanup
    Path(csv_path).unlink(missing_ok=True)
    Path(output_csv).unlink(missing_ok=True)
