"""
ingest.py
Handles data ingestion from CSV and Excel files, including column mapping.
"""

import csv
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from pandas.errors import ParserError
from src.schema import (
    auto_map_columns,
    interactive_column_mapping,
    apply_column_mapping,
    REQUIRED_COLUMNS,
    OPTIONAL_COLUMNS,
)


def load_file(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV or Excel file.
    
    Args:
        file_path: Path to the input file
        
    Returns:
        DataFrame with the raw data
        
    Raises:
        ValueError: If file format is not supported or file doesn't exist
    """
    path = Path(file_path)
    
    if not path.exists():
        raise ValueError(f"File not found: {file_path}")
    
    file_extension = path.suffix.lower()
    
    if file_extension == '.csv':
        def _try_read_csv(path: str) -> pd.DataFrame:
            # First pass with normal parser.
            local_df = pd.read_csv(path)

            # Detect report preamble style where actual header is on a later row.
            first_col = str(local_df.columns[0]).strip().lower()
            if first_col in {'start date:', 'end date:'}:
                raw = pd.read_csv(path, header=None)
                header_row = None
                scan_rows = min(20, len(raw))
                for i in range(scan_rows):
                    row_values = [str(v).strip().lower() for v in raw.iloc[i].tolist()]
                    if 'date' in row_values and ('room revenue' in row_values or 'occupancy %' in row_values):
                        header_row = i
                        break
                if header_row is not None:
                    local_df = pd.read_csv(path, skiprows=header_row)

            return local_df

        fallback_errors = []
        try:
            df = _try_read_csv(file_path)
        except (ParserError, UnicodeDecodeError):
            try:
                # Fallback 1: infer delimiter with python engine
                df = pd.read_csv(file_path, sep=None, engine='python')
            except Exception as e1:
                fallback_errors.append(str(e1))
                try:
                    # Fallback 2: common semicolon-delimited exports
                    df = pd.read_csv(file_path, sep=';', engine='python')
                except Exception as e2:
                    fallback_errors.append(str(e2))
                    try:
                        # Fallback 3: common tab-delimited exports
                        df = pd.read_csv(file_path, sep='\t', engine='python')
                    except Exception as e3:
                        fallback_errors.append(str(e3))
                        try:
                            # Fallback 4: tolerate broken quote characters and skip bad rows
                            df = pd.read_csv(
                                file_path,
                                engine='python',
                                sep=None,
                                quoting=csv.QUOTE_NONE,
                                on_bad_lines='skip',
                            )
                            print("[WARNING] CSV contained malformed quote formatting. Some bad rows may have been skipped.")
                        except Exception as e4:
                            fallback_errors.append(str(e4))
                            raise ValueError(
                                "Error reading CSV file. The file appears malformed (delimiter/quotes/row shape). "
                                "Try re-exporting as UTF-8 CSV from PMS. "
                                f"Parser details: {' | '.join(fallback_errors)}"
                            )
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {str(e)}")
    elif file_extension in ['.xlsx', '.xls']:
        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            raise ValueError(f"Error reading Excel file: {str(e)}")
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Please use CSV or Excel files.")
    
    if df.empty:
        raise ValueError("The input file is empty.")
    
    print(f"[OK] Loaded {len(df)} rows from {path.name}")
    
    return df


def ingest_and_map(
    file_path: str,
    interactive: bool = True,
    user_mapping: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Load file and map columns to canonical schema.
    
    Args:
        file_path: Path to the input file
        interactive: If True, prompt user for missing columns; if False, raise error
        
    Returns:
        DataFrame with canonical column names
        
    Raises:
        ValueError: If required columns cannot be mapped
    """
    # Load the file
    df = load_file(file_path)
    return map_columns(df, interactive=interactive, user_mapping=user_mapping)


def map_columns(
    df: pd.DataFrame,
    interactive: bool = True,
    user_mapping: Optional[Dict[str, str]] = None,
    required_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Map dataframe columns to canonical schema.

    Args:
        df: Input dataframe
        interactive: If True, prompt for missing required mappings in terminal
        user_mapping: Optional mapping for canonical columns supplied by caller

    Returns:
        DataFrame with canonical column names
    """
    print("\nAttempting automatic column mapping...")
    column_mapping = auto_map_columns(df)

    if user_mapping:
        for canonical_col, actual_col in user_mapping.items():
            if canonical_col not in (REQUIRED_COLUMNS + OPTIONAL_COLUMNS):
                continue
            if actual_col in df.columns:
                column_mapping[canonical_col] = actual_col

    def _find_currency_like_column(frame: pd.DataFrame) -> Optional[str]:
        best_col = None
        best_score = (-1, -1)  # (dollar_hits, numeric_hits)
        in_use = set(column_mapping.values())
        for candidate in frame.columns:
            if candidate in in_use:
                continue
            sample = frame[candidate].astype(str)
            dollar_hits = int(sample.str.contains(r'\$', regex=True, na=False).sum())
            numeric_hits = int(
                sample.str.contains(r'\d{1,3}(?:,\d{3})*(?:\.\d+)?', regex=True, na=False).sum()
            )
            score = (dollar_hits, numeric_hits)
            if score > best_score:
                best_col = candidate
                best_score = score
        return best_col if best_score[0] > 0 else None

    if 'room_revenue' in column_mapping:
        revenue_col = column_mapping['room_revenue']
        revenue_null_ratio = df[revenue_col].isna().mean()
        if revenue_null_ratio > 0.9:
            fallback_col = _find_currency_like_column(df)
            if fallback_col and fallback_col != revenue_col:
                column_mapping['room_revenue'] = fallback_col
                print(
                    f"[WARNING] Revenue values were not found in '{revenue_col}'. "
                    f"Using '{fallback_col}' for room_revenue instead."
                )

    if 'room_revenue' in column_mapping and 'occupancy_percent' in column_mapping:
        if column_mapping['room_revenue'] == column_mapping['occupancy_percent']:
            raise ValueError(
                "Invalid mapping: room_revenue cannot use the same source column as occupancy_percent."
            )

    required_set = required_columns or REQUIRED_COLUMNS
    missing = [col for col in required_set if col not in column_mapping]

    if 'rooms_available' in missing and 'rooms_sold' in column_mapping and 'occupancy_percent' in column_mapping:
        missing = [col for col in missing if col != 'rooms_available']
        print("[WARNING] 'rooms_available' missing in source. It will be derived from rooms_sold and occupancy_percent.")

    if missing:
        print(f"[WARNING] Could not auto-map {len(missing)} required column(s): {', '.join(missing)}")

        if interactive:
            prompt_mapping = interactive_column_mapping(df, missing)
            column_mapping.update(prompt_mapping)
        else:
            raise ValueError(
                f"Missing required columns: {', '.join(missing)}. "
                "Provide explicit mapping for missing fields."
            )
    else:
        print("[OK] Successfully auto-mapped all required columns")

    required_actual_columns = [column_mapping[col] for col in required_set if col in column_mapping]
    if len(required_actual_columns) != len(set(required_actual_columns)):
        raise ValueError(
            "Invalid column mapping: each required field (stay_date, rooms_available, "
            "rooms_sold, room_revenue) must map to a different source column."
        )

    print("\nFinal column mapping:")
    for canonical, actual in sorted(column_mapping.items()):
        marker = "*" if canonical in REQUIRED_COLUMNS else " "
        print(f"  {marker} {canonical:20s} <- {actual}")

    mapped_df = apply_column_mapping(df, column_mapping)

    # Ensure optional canonical columns exist for downstream logic.
    optional_missing = [col for col in OPTIONAL_COLUMNS if col not in mapped_df.columns]
    for optional_col in optional_missing:
        mapped_df[optional_col] = pd.NA

    return mapped_df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse date columns to datetime format.
    
    Args:
        df: DataFrame with canonical columns
        
    Returns:
        DataFrame with parsed dates
    """
    date_columns = ['stay_date', 'booking_date']
    
    for col in date_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                null_count = df[col].isna().sum()
                if null_count > 0:
                    print(f"[WARNING] Warning: {null_count} rows have invalid {col} values")
            except Exception as e:
                print(f"[WARNING] Warning: Could not parse {col}: {str(e)}")
    
    return df


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert numeric columns to appropriate types.
    
    Args:
        df: DataFrame with canonical columns
        
    Returns:
        DataFrame with converted numeric types
    """
    def _clean_numeric_text(series: pd.Series) -> pd.Series:
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        return (
            series.astype(str)
            .str.replace(r'[$,]', '', regex=True)
            .str.replace('%', '', regex=False)
            .str.replace(' ', '', regex=False)
            .str.strip()
            .replace({'': pd.NA, 'nan': pd.NA, 'None': pd.NA})
        )

    # Integer columns
    int_columns = ['rooms_available', 'rooms_sold']
    for col in int_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(_clean_numeric_text(df[col]), errors='coerce')
            # Fill NaN with 0 for now (validation will catch this later)
            df[col] = df[col].fillna(0).astype('int64')
    
    # Float columns
    float_columns = ['room_revenue', 'adr', 'current_rate']
    for col in float_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(_clean_numeric_text(df[col]), errors='coerce')
            # Keep room_revenue/current_rate/adr NaN values for date-aware validation and fallback logic.
            if col not in {'room_revenue', 'current_rate', 'adr'}:
                df[col] = df[col].fillna(0.0)
            df[col] = df[col].astype('float64')

    if 'occupancy_percent' in df.columns:
        df['occupancy_percent'] = pd.to_numeric(_clean_numeric_text(df['occupancy_percent']), errors='coerce')
        df['occupancy_percent'] = df['occupancy_percent'].fillna(0.0).astype('float64')

    if 'rooms_available' not in df.columns and {'rooms_sold', 'occupancy_percent'}.issubset(df.columns):
        occ_ratio = df['occupancy_percent'].copy()
        occ_ratio = occ_ratio.where(occ_ratio <= 1.0, occ_ratio / 100.0)
        derived = pd.Series(0, index=df.index, dtype='int64')
        valid = occ_ratio > 0
        derived.loc[valid] = (df.loc[valid, 'rooms_sold'] / occ_ratio.loc[valid]).round().astype('int64')
        df['rooms_available'] = derived

    if {'rooms_available', 'rooms_sold', 'occupancy_percent'}.issubset(df.columns):
        missing_or_zero = df['rooms_available'] <= 0
        if missing_or_zero.any():
            occ_ratio = df['occupancy_percent'].copy()
            occ_ratio = occ_ratio.where(occ_ratio <= 1.0, occ_ratio / 100.0)
            valid = missing_or_zero & (occ_ratio > 0)
            df.loc[valid, 'rooms_available'] = (
                (df.loc[valid, 'rooms_sold'] / occ_ratio.loc[valid]).round().astype('int64')
            )
    
    return df


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the data by parsing dates and converting types.
    
    Args:
        df: DataFrame with canonical column names
        
    Returns:
        Normalized DataFrame
    """
    print("\nNormalizing data types...")
    
    df = df.copy()
    df = parse_dates(df)
    df = convert_numeric_columns(df)
    
    print("[OK] Data normalization complete")
    
    return df


def process_file(
    file_path: str,
    interactive: bool = True,
    user_mapping: Optional[Dict[str, str]] = None,
    required_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Complete file processing pipeline: load, map, normalize.
    
    Args:
        file_path: Path to the input file
        interactive: If True, prompt user for missing columns
        
    Returns:
        Fully processed DataFrame with canonical schema
    """
    print("\n" + "="*60)
    print("DATA INGESTION")
    print("="*60)
    
    # Ingest and map columns
    df = load_file(file_path)
    df = map_columns(
        df,
        interactive=interactive,
        user_mapping=user_mapping,
        required_columns=required_columns,
    )
    
    # Normalize data types
    df = normalize_data(df)
    
    print("\n" + "="*60)
    print(f"Ingestion complete: {len(df)} rows, {len(df.columns)} columns")
    print("="*60 + "\n")
    
    return df


def process_dataframe(
    df_raw: pd.DataFrame,
    interactive: bool = True,
    user_mapping: Optional[Dict[str, str]] = None,
    required_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Complete processing pipeline for an in-memory dataframe.

    Args:
        df_raw: Raw dataframe loaded from CSV/XLSX
        interactive: If True, allow terminal prompts for mapping
        user_mapping: Optional explicit mapping for canonical required fields

    Returns:
        Fully processed DataFrame with canonical schema
    """
    print("\n" + "="*60)
    print("DATA INGESTION")
    print("="*60)

    df_mapped = map_columns(
        df_raw,
        interactive=interactive,
        user_mapping=user_mapping,
        required_columns=required_columns,
    )
    df_normalized = normalize_data(df_mapped)

    print("\n" + "="*60)
    print(f"Ingestion complete: {len(df_normalized)} rows, {len(df_normalized.columns)} columns")
    print("="*60 + "\n")

    return df_normalized
