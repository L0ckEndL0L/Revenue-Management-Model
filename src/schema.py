"""
schema.py
Defines the canonical data schema and column mapping logic for hotel PMS reports.
"""

from typing import Dict, List, Optional
import pandas as pd


# Canonical schema - the internal standard format
CANONICAL_SCHEMA = {
    'stay_date': 'datetime64[ns]',
    'rooms_available': 'int64',
    'rooms_sold': 'int64',
    'room_revenue': 'float64',
    'current_rate': 'float64',
    'occupancy_percent': 'float64',
    # Optional columns
    'room_type': 'str',
    'rate_code': 'str',
    'channel': 'str',
    'booking_date': 'datetime64[ns]',
    'adr': 'float64'
}

# Required columns that must be present
REQUIRED_COLUMNS = ['stay_date', 'rooms_available', 'rooms_sold', 'room_revenue']

# Optional columns that can be present
OPTIONAL_COLUMNS = ['occupancy_percent', 'room_type', 'rate_code', 'channel', 'booking_date', 'adr', 'current_rate']


# Default mapping patterns - common PMS export column names
DEFAULT_COLUMN_MAPPINGS = {
    'stay_date': [
        'stay_date', 'date', 'stay date', 'arrival_date', 'arrival date',
        'occupancy_date', 'occupancy date', 'business_date', 'business date',
        'night_date', 'night date', 'service_date', 'service date'
    ],
    'rooms_available': [
        'rooms_available', 'available_rooms', 'rooms available', 'available rooms',
        'total_rooms', 'total rooms', 'inventory', 'room_inventory', 
        'capacity', 'room_capacity'
    ],
    'rooms_sold': [
        'rooms_sold', 'sold_rooms', 'rooms sold', 'sold rooms',
        'occupied_rooms', 'occupied rooms', 'rooms_occupied', 
        'sold', 'occupied', 'room_nights', 'all room types', 'occupied rooms total',
        'rooms_sold_to_date', 'rooms sold to date', 'rooms_on_books', 'on_books',
        'on books', 'otb_rooms', 'otb'
    ],
    'room_revenue': [
        'room_revenue', 'revenue', 'room revenue', 'total_revenue',
        'total revenue', 'rooms_revenue', 'rooms revenue',
        'accommodation_revenue', 'accommodation revenue'
    ],
    'occupancy_percent': [
        'occupancy %', 'occupancy_percent', 'occupancy percent',
        'occ %', 'occupancy'
    ],
    'adr': [
        'adr', 'average daily rate', 'average_daily_rate', 'avg daily rate',
        'avg_daily_rate', 'avg adr', 'avg_adr', 'current adr', 'current_adr',
        'rate adr', 'selling_adr'
    ],
    'room_type': [
        'room_type', 'room type', 'roomtype', 'type', 
        'category', 'room_category', 'room category'
    ],
    'rate_code': [
        'rate_code', 'rate code', 'ratecode', 'rate_plan',
        'rate plan', 'rateplan', 'rate', 'tariff'
    ],
    'channel': [
        'channel', 'source', 'booking_source', 'booking source',
        'distribution_channel', 'distribution channel', 'channel_code'
    ],
    'booking_date': [
        'booking_date', 'booking date', 'reservation_date', 
        'reservation date', 'booked_date', 'booked date'
    ],
    'current_rate': [
        'current_rate', 'current rate', 'bar', 'bar_rate', 'bar rate',
        'current_bar', 'current bar', 'sell_rate', 'sell rate', 'rate_today'
        , 'current_adr', 'current adr', 'today_rate', 'today rate'
    ]
}


def find_column_match(df_columns: List[str], canonical_col: str) -> Optional[str]:
    """
    Find a matching column in the dataframe for a canonical column name.
    
    Args:
        df_columns: List of column names from the input dataframe
        canonical_col: The canonical column name to find a match for
        
    Returns:
        The matching column name from df_columns, or None if no match found
    """
    # Normalize all column names to lowercase for comparison
    df_columns_lower = {col.lower().strip(): col for col in df_columns}
    
    # Get the list of possible names for this canonical column
    possible_names = DEFAULT_COLUMN_MAPPINGS.get(canonical_col, [])
    
    # Try to find an exact match
    for possible_name in possible_names:
        normalized_name = possible_name.lower().strip()
        if normalized_name in df_columns_lower:
            return df_columns_lower[normalized_name]
    
    return None


def auto_map_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Automatically map dataframe columns to canonical schema.
    
    Args:
        df: Input dataframe with PMS export data
        
    Returns:
        Dictionary mapping canonical column names to actual dataframe column names
    """
    column_mapping = {}
    df_columns = list(df.columns)
    
    # Try to map all canonical columns (required + optional)
    all_columns = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
    
    for canonical_col in all_columns:
        matched_col = find_column_match(df_columns, canonical_col)
        if matched_col:
            column_mapping[canonical_col] = matched_col
    
    return column_mapping


def get_missing_required_columns(column_mapping: Dict[str, str]) -> List[str]:
    """
    Get list of required columns that are missing from the mapping.
    
    Args:
        column_mapping: Dictionary of canonical -> actual column mappings
        
    Returns:
        List of missing required canonical column names
    """
    mapped_required = [col for col in REQUIRED_COLUMNS if col in column_mapping]
    missing = [col for col in REQUIRED_COLUMNS if col not in mapped_required]
    return missing


def interactive_column_mapping(df: pd.DataFrame, missing_columns: List[str]) -> Dict[str, str]:
    """
    Interactively prompt user to map missing required columns.
    
    Args:
        df: Input dataframe
        missing_columns: List of missing canonical column names
        
    Returns:
        Dictionary mapping canonical column names to actual dataframe column names
    """
    print("\n" + "="*60)
    print("COLUMN MAPPING REQUIRED")
    print("="*60)
    print(f"\nThe following required columns could not be auto-mapped:")
    for col in missing_columns:
        print(f"  - {col}")
    
    print(f"\nAvailable columns in your file:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    mapping = {}
    
    for canonical_col in missing_columns:
        while True:
            print(f"\nWhich column should map to '{canonical_col}'?")
            user_input = input(f"Enter column name or number (1-{len(df.columns)}): ").strip()
            
            # Try to interpret as number
            try:
                col_index = int(user_input) - 1
                if 0 <= col_index < len(df.columns):
                    mapping[canonical_col] = df.columns[col_index]
                    print(f"[OK] Mapped '{canonical_col}' -> '{df.columns[col_index]}'")
                    break
                else:
                    print(f"Invalid number. Please enter a number between 1 and {len(df.columns)}.")
            except ValueError:
                # Try to match by name
                if user_input in df.columns:
                    mapping[canonical_col] = user_input
                    print(f"[OK] Mapped '{canonical_col}' -> '{user_input}'")
                    break
                else:
                    print(f"Column '{user_input}' not found. Please try again.")
    
    print("\n" + "="*60)
    print("Mapping complete!")
    print("="*60 + "\n")
    
    return mapping


def apply_column_mapping(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Apply column mapping to create a dataframe with canonical column names.
    
    Args:
        df: Input dataframe
        column_mapping: Dictionary mapping canonical -> actual column names
        
    Returns:
        New dataframe with canonical column names
    """
    # Select and rename columns
    rename_dict = {v: k for k, v in column_mapping.items()}
    
    # Select only the columns that exist in the mapping
    columns_to_select = list(column_mapping.values())
    df_mapped = df[columns_to_select].copy()
    df_mapped.rename(columns=rename_dict, inplace=True)
    
    return df_mapped
