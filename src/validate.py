"""
validate.py
Data validation logic for hotel PMS data.
"""

import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime


class ValidationResult:
    """Container for validation results."""
    
    def __init__(self):
        self.issues: List[Dict] = []
        self.passed = True
        self.total_rows = 0
        self.valid_rows = 0
        self.invalid_rows = 0
    
    def add_issue(self, row_index: int, column: str, issue_type: str, message: str):
        """Add a validation issue."""
        self.issues.append({
            'row_index': row_index,
            'column': column,
            'issue_type': issue_type,
            'message': message
        })
        self.passed = False
    
    def summarize(self) -> str:
        """Generate a summary string of validation results."""
        summary = []
        summary.append("="*60)
        summary.append("DATA VALIDATION REPORT")
        summary.append("="*60)
        summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append("")
        summary.append(f"Total rows processed: {self.total_rows}")
        summary.append(f"Valid rows: {self.valid_rows}")
        summary.append(f"Rows with issues: {self.invalid_rows}")
        summary.append("")
        
        if self.passed:
            summary.append("[OK] All validation checks passed!")
        else:
            summary.append(f"[WARNING] Found {len(self.issues)} validation issue(s)")
            summary.append("")
            summary.append("Issues by type:")
            
            # Count issues by type
            issue_types = {}
            for issue in self.issues:
                issue_type = issue['issue_type']
                issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
            
            for issue_type, count in sorted(issue_types.items()):
                summary.append(f"  - {issue_type}: {count}")
            
            summary.append("")
            summary.append("Detailed issues:")
            summary.append("-" * 60)
            
            for i, issue in enumerate(self.issues, 1):
                summary.append(f"\n{i}. Row {issue['row_index']} | Column: {issue['column']}")
                summary.append(f"   Type: {issue['issue_type']}")
                summary.append(f"   Message: {issue['message']}")
        
        summary.append("")
        summary.append("="*60)
        
        return "\n".join(summary)


def validate_data(
    df: pd.DataFrame,
    allow_overbooking: bool = False,
    as_of_date: pd.Timestamp | None = None,
    default_current_rate: float | None = None,
) -> Tuple[pd.DataFrame, ValidationResult]:
    """
    Validate hotel PMS data according to business rules.
    
    Validation rules:
    - stay_date must be valid datetime and not null
    - rooms_available and rooms_sold must be non-negative
    - rooms_sold cannot exceed rooms_available (unless overbooking allowed)
    - room_revenue must be non-negative
    - missing room_revenue is allowed on future dates
    - missing current_rate is filled with fallback and logged
    
    Args:
        df: DataFrame with canonical columns
        allow_overbooking: If True, allow rooms_sold > rooms_available
        
    Returns:
        Tuple of (cleaned_df, validation_result)
    """
    print("\n" + "="*60)
    print("DATA VALIDATION")
    print("="*60)
    
    result = ValidationResult()
    result.total_rows = len(df)
    
    df_clean = df.copy()
    rows_to_remove = set()
    
    # Precompute fallback rate for rows without current_rate.
    fallback_rate = default_current_rate
    if fallback_rate is None:
        if 'current_rate' in df.columns and df['current_rate'].notna().any():
            valid_current_rates = df['current_rate'].dropna()
            valid_current_rates = valid_current_rates[valid_current_rates > 0]
            if len(valid_current_rates) > 0:
                fallback_rate = float(valid_current_rates.median())
        elif 'adr' in df.columns and df['adr'].notna().any():
            valid_adr = df['adr'].dropna()
            valid_adr = valid_adr[valid_adr > 0]
            if len(valid_adr) > 0:
                fallback_rate = float(valid_adr.median())
        elif 'room_revenue' in df.columns and 'rooms_sold' in df.columns:
            sold = df['rooms_sold'].replace(0, pd.NA)
            derived = (df['room_revenue'] / sold).dropna()
            derived = derived[derived > 0]
            fallback_rate = float(derived.median()) if len(derived) else 120.0
        else:
            fallback_rate = 120.0

    if fallback_rate is None or float(fallback_rate) <= 0:
        fallback_rate = 120.0

    df_clean['current_rate'] = pd.to_numeric(df_clean.get('current_rate', pd.Series([pd.NA] * len(df_clean))), errors='coerce')

    # Validate each row
    for idx in df.index:
        row = df.loc[idx]
        has_issues = False
        row_date = row['stay_date']
        is_future_date = bool(pd.notna(row_date) and as_of_date is not None and row_date > as_of_date)
        
        # Rule 1: stay_date must be valid and not null
        if pd.isna(row['stay_date']):
            result.add_issue(idx, 'stay_date', 'NULL_DATE', 
                           'stay_date is null or could not be parsed')
            has_issues = True
        
        # Rule 2: rooms_available must be non-negative
        if row['rooms_available'] < 0:
            result.add_issue(idx, 'rooms_available', 'NEGATIVE_VALUE',
                           f'rooms_available is negative: {row["rooms_available"]}')
            has_issues = True
        
        # Rule 3: rooms_sold must be non-negative
        if row['rooms_sold'] < 0:
            result.add_issue(idx, 'rooms_sold', 'NEGATIVE_VALUE',
                           f'rooms_sold is negative: {row["rooms_sold"]}')
            has_issues = True
        
        # Rule 4: rooms_sold should not exceed rooms_available (unless overbooking allowed)
        if not allow_overbooking and row['rooms_sold'] > row['rooms_available']:
            result.add_issue(idx, 'rooms_sold', 'OVERBOOKING',
                           f'rooms_sold ({row["rooms_sold"]}) exceeds rooms_available ({row["rooms_available"]})')
            has_issues = True
        
        # Rule 5: room_revenue must be non-negative
        room_revenue = row.get('room_revenue', pd.NA)
        if pd.notna(room_revenue) and room_revenue < 0:
            result.add_issue(idx, 'room_revenue', 'NEGATIVE_VALUE',
                           f'room_revenue is negative: {room_revenue}')
            has_issues = True

        if pd.isna(room_revenue) and not is_future_date:
            result.add_issue(idx, 'room_revenue', 'MISSING_REVENUE_PAST_DATE',
                           'room_revenue is missing for a non-future stay_date')
            has_issues = True

        current_rate = row.get('current_rate', pd.NA)
        if pd.isna(current_rate) or float(current_rate) <= 0:
            df_clean.at[idx, 'current_rate'] = float(fallback_rate)
            result.add_issue(
                idx,
                'current_rate',
                'CURRENT_RATE_FILLED',
                f'current_rate missing; filled with fallback value {float(fallback_rate):.2f}',
            )
        
        # Mark row for removal if it has critical issues
        if has_issues:
            rows_to_remove.add(idx)
    
    # Remove invalid rows
    if rows_to_remove:
        df_clean = df_clean.drop(index=list(rows_to_remove))
        result.invalid_rows = len(rows_to_remove)
        result.valid_rows = len(df_clean)
        print(f"[WARNING] Removed {len(rows_to_remove)} invalid row(s)")
    else:
        result.valid_rows = len(df_clean)
        print("[OK] All rows passed validation")
    
    print(f"Valid rows: {result.valid_rows} / {result.total_rows}")
    print("="*60 + "\n")
    
    return df_clean, result


def check_data_quality(df: pd.DataFrame) -> None:
    """
    Perform additional data quality checks and print warnings.
    
    Args:
        df: Validated DataFrame
    """
    print("Data quality checks:")
    
    # Check for zeros
    zero_rooms_available = (df['rooms_available'] == 0).sum()
    if zero_rooms_available > 0:
        print(f"[WARNING] {zero_rooms_available} rows have rooms_available = 0")
    
    zero_rooms_sold = (df['rooms_sold'] == 0).sum()
    if zero_rooms_sold > 0:
        print(f"  {zero_rooms_sold} rows have rooms_sold = 0 (will affect ADR calculation)")
    
    zero_revenue = (df['room_revenue'] == 0).sum()
    if zero_revenue > 0:
        print(f"  {zero_revenue} rows have room_revenue = 0")
    
    # Check date range
    if len(df) > 0:
        min_date = df['stay_date'].min()
        max_date = df['stay_date'].max()
        date_range = (max_date - min_date).days
        print(f"\nDate range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} ({date_range + 1} days)")
    
    # Check for duplicates
    if 'stay_date' in df.columns:
        duplicate_dates = df['stay_date'].duplicated().sum()
        if duplicate_dates > 0:
            print(f"[WARNING] {duplicate_dates} duplicate stay_date values found (may be different room types/segments)")
    
    print()


def save_validation_report(result: ValidationResult, output_path: str) -> None:
    """
    Save validation report to a text file.
    
    Args:
        result: ValidationResult object
        output_path: Path to save the report
    """
    report_text = result.summarize()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"[OK] Validation report saved to: {output_path}")
