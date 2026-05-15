#!/usr/bin/env python
"""Quick validation test for hand-checked examples."""

import pandas as pd
from src.metrics import calculate_daily_metrics

# Test hand-checked example
df = pd.DataFrame({
    'stay_date': pd.to_datetime(['2026-03-01', '2026-03-02', '2026-03-03']),
    'rooms_available': [100, 100, 100],
    'rooms_sold': [80, 90, 70],
    'room_revenue': [9600.0, 11700.0, 7700.0],
})

out = calculate_daily_metrics(df)

print('Test 1 (2026-03-01):')
print(f'  occupancy: {out.loc[0, "occupancy"]} (expected: 0.8)')
print(f'  occupancy_pct: {out.loc[0, "occupancy_pct"]:.2f}% (expected: 80.00%)')
print(f'  adr: {out.loc[0, "adr"]:.2f} (expected: 120.00)')
print(f'  revpar: {out.loc[0, "revpar"]:.2f} (expected: 96.00)')

print()
print('Test 2 (2026-03-02):')
print(f'  occupancy: {out.loc[1, "occupancy"]} (expected: 0.9)')
print(f'  occupancy_pct: {out.loc[1, "occupancy_pct"]:.2f}% (expected: 90.00%)')
print(f'  adr: {out.loc[1, "adr"]:.2f} (expected: 130.00)')
print(f'  revpar: {out.loc[1, "revpar"]:.2f} (expected: 117.00)')

print()
print('Test 3 (2026-03-03):')
print(f'  occupancy: {out.loc[2, "occupancy"]} (expected: 0.7)')
print(f'  occupancy_pct: {out.loc[2, "occupancy_pct"]:.2f}% (expected: 70.00%)')
print(f'  adr: {out.loc[2, "adr"]:.2f} (expected: 110.00)')
print(f'  revpar: {out.loc[2, "revpar"]:.2f} (expected: 77.00)')

print()
print('Aggregated Totals:')
total_rooms_available = out['rooms_available'].sum()
total_rooms_sold = out['rooms_sold'].sum()
total_room_revenue = out['room_revenue'].sum()
print(f'  Total rooms available: {total_rooms_available} (expected: 300)')
print(f'  Total rooms sold: {total_rooms_sold} (expected: 240)')
print(f'  Total room revenue: {total_room_revenue} (expected: 29000.0)')

overall_occupancy = total_rooms_sold / total_rooms_available
overall_adr = total_room_revenue / total_rooms_sold
overall_revpar = total_room_revenue / total_rooms_available
print(f'  Overall occupancy: {overall_occupancy:.2f} (expected: 0.80)')
print(f'  Overall ADR: {overall_adr:.2f} (expected: 120.83)')
print(f'  Overall RevPAR: {overall_revpar:.2f} (expected: 96.67)')
print(f'  RevPAR cross-check (adr * occupancy): {(overall_adr * overall_occupancy):.2f} (should match revpar)')

# Verify all assertions
try:
    assert out.loc[0, "occupancy"] == 0.80, f"Row 0 occupancy: {out.loc[0, 'occupancy']} != 0.80"
    assert out.loc[0, "occupancy_pct"] == 80.00, f"Row 0 occupancy_pct: {out.loc[0, 'occupancy_pct']} != 80.00"
    assert out.loc[0, "adr"] == 120.00, f"Row 0 adr: {out.loc[0, 'adr']} != 120.00"
    assert out.loc[0, "revpar"] == 96.00, f"Row 0 revpar: {out.loc[0, 'revpar']} != 96.00"
    
    assert out.loc[1, "occupancy"] == 0.90, f"Row 1 occupancy: {out.loc[1, 'occupancy']} != 0.90"
    assert out.loc[1, "occupancy_pct"] == 90.00, f"Row 1 occupancy_pct: {out.loc[1, 'occupancy_pct']} != 90.00"
    assert out.loc[1, "adr"] == 130.00, f"Row 1 adr: {out.loc[1, 'adr']} != 130.00"
    assert out.loc[1, "revpar"] == 117.00, f"Row 1 revpar: {out.loc[1, 'revpar']} != 117.00"
    
    assert out.loc[2, "occupancy"] == 0.70, f"Row 2 occupancy: {out.loc[2, 'occupancy']} != 0.70"
    assert out.loc[2, "occupancy_pct"] == 70.00, f"Row 2 occupancy_pct: {out.loc[2, 'occupancy_pct']} != 70.00"
    assert out.loc[2, "adr"] == 110.00, f"Row 2 adr: {out.loc[2, 'adr']} != 110.00"
    assert out.loc[2, "revpar"] == 77.00, f"Row 2 revpar: {out.loc[2, 'revpar']} != 77.00"
    
    assert total_rooms_available == 300
    assert total_rooms_sold == 240
    assert total_room_revenue == 29000.0
    assert overall_occupancy == 0.80
    assert abs(overall_adr - 120.83) < 0.01
    assert abs(overall_revpar - 96.67) < 0.01
    
    print()
    print("✓ ALL HAND-CHECKED EXAMPLES PASSED!")
except AssertionError as e:
    print()
    print(f"✗ TEST FAILED: {e}")
    exit(1)
