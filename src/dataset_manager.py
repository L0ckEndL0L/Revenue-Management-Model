"""Dataset management for saving and loading PMS reports."""

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


DATASETS_DIR = Path("datasets")
METADATA_FILE = DATASETS_DIR / "datasets_metadata.json"
BUDGETS_DIR = DATASETS_DIR / "budgets"
BUDGETS_METADATA_FILE = DATASETS_DIR / "budgets_metadata.json"


def _sanitize_dataset_name(name: str) -> str:
    """Normalize a dataset name so it is safe to use as a folder path."""
    cleaned = re.sub(r"[\\/:*?\"<>|]+", "_", name.strip())
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" .")


def _ensure_datasets_dir() -> None:
    """Create datasets directory if it doesn't exist."""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_budgets_dir() -> None:
    """Create budgets directory if it doesn't exist."""
    _ensure_datasets_dir()
    BUDGETS_DIR.mkdir(parents=True, exist_ok=True)


def _load_metadata() -> Dict[str, dict]:
    """Load dataset metadata from JSON file."""
    _ensure_datasets_dir()
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_metadata(metadata: Dict[str, dict]) -> None:
    """Save dataset metadata to JSON file."""
    _ensure_datasets_dir()
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)


def _load_budget_metadata() -> Dict[str, dict]:
    """Load budget profile metadata from JSON file."""
    _ensure_budgets_dir()
    if BUDGETS_METADATA_FILE.exists():
        with open(BUDGETS_METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_budget_metadata(metadata: Dict[str, dict]) -> None:
    """Save budget profile metadata to JSON file."""
    _ensure_budgets_dir()
    with open(BUDGETS_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)


def save_dataset(
    name: str,
    historical_df: pd.DataFrame,
    future_df: pd.DataFrame,
    events_df: Optional[pd.DataFrame] = None,
    budget_df: Optional[pd.DataFrame] = None,
    historical_mapping: Optional[Dict[str, str]] = None,
    future_mapping: Optional[Dict[str, str]] = None,
    use_manual_rooms_available: bool = False,
    manual_rooms_available: Optional[int] = None,
    tailored_settings: Optional[Dict] = None,
) -> bool:
    """
    Save a dataset with historical, future, and optional supplementary data.
    
    Args:
        name: Dataset name (used as folder name)
        historical_df: Historical PMS data
        future_df: Future on-books data
        events_df: Optional events data
        budget_df: Optional budget data
        historical_mapping: Optional column mappings for historical data
        future_mapping: Optional column mappings for future data
        use_manual_rooms_available: Whether to use manual rooms available setting
        manual_rooms_available: Manual rooms available value if enabled
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Sanitize dataset name
        name = _sanitize_dataset_name(name)
        if not name:
            return False
        
        # Check for name conflicts
        metadata = _load_metadata()
        if name in metadata:
            # Update existing dataset
            metadata[name]["updated_at"] = datetime.now().isoformat()
            metadata[name]["rows_historical"] = len(historical_df)
            metadata[name]["rows_future"] = len(future_df)
            metadata[name]["has_events"] = events_df is not None and len(events_df) > 0
            metadata[name]["has_budget"] = budget_df is not None and len(budget_df) > 0
            metadata[name]["use_manual_rooms_available"] = use_manual_rooms_available
            metadata[name]["manual_rooms_available"] = manual_rooms_available
            metadata[name]["has_tailored_settings"] = bool(tailored_settings)
            metadata[name]["median_rate_last_updated"] = (tailored_settings or {}).get("median_rate_last_updated")
            metadata[name]["median_rate_update_frequency"] = (tailored_settings or {}).get("median_rate_update_frequency")
        else:
            # Create new dataset entry
            metadata[name] = {
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "rows_historical": len(historical_df),
                "rows_future": len(future_df),
                "has_events": events_df is not None and len(events_df) > 0,
                "has_budget": budget_df is not None and len(budget_df) > 0,
                "use_manual_rooms_available": use_manual_rooms_available,
                "manual_rooms_available": manual_rooms_available,
                "has_tailored_settings": bool(tailored_settings),
                "median_rate_last_updated": (tailored_settings or {}).get("median_rate_last_updated"),
                "median_rate_update_frequency": (tailored_settings or {}).get("median_rate_update_frequency"),
            }
        
        # Create dataset directory
        dataset_dir = DATASETS_DIR / name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data files
        historical_df.to_csv(dataset_dir / "historical.csv", index=False)
        future_df.to_csv(dataset_dir / "future.csv", index=False)
        
        if events_df is not None and len(events_df) > 0:
            events_df.to_csv(dataset_dir / "events.csv", index=False)
        
        if budget_df is not None and len(budget_df) > 0:
            budget_df.to_csv(dataset_dir / "budget.csv", index=False)
        
        # Save column mappings
        mappings = {
            "historical_mapping": historical_mapping or {},
            "future_mapping": future_mapping or {},
            "tailored_settings": tailored_settings or {},
        }
        with open(dataset_dir / "mappings.json", "w", encoding="utf-8") as f:
            json.dump(mappings, f, indent=2)
        
        # Update metadata
        _save_metadata(metadata)
        return True
    
    except Exception as e:
        print(f"Error saving dataset '{name}': {e}")
        return False


def load_dataset(name: str) -> tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Dict[str, str],
    Dict[str, str],
    bool,
    Optional[int],
    Dict,
]:
    """
    Load a saved dataset.
    
    Args:
        name: Dataset name
    
    Returns:
        Tuple of (historical_df, future_df, events_df, budget_df, historical_mapping, future_mapping, 
                 use_manual_rooms_available, manual_rooms_available)
        Returns None for missing optional dataframes
    """
    try:
        dataset_dir = DATASETS_DIR / name
        
        # Load required files
        historical_df = pd.read_csv(dataset_dir / "historical.csv")
        future_df = pd.read_csv(dataset_dir / "future.csv")
        
        # Load optional files
        events_df = None
        if (dataset_dir / "events.csv").exists():
            events_df = pd.read_csv(dataset_dir / "events.csv")
        
        budget_df = None
        if (dataset_dir / "budget.csv").exists():
            budget_df = pd.read_csv(dataset_dir / "budget.csv")
        
        # Load mappings
        historical_mapping = {}
        future_mapping = {}
        tailored_settings = {}
        mappings_file = dataset_dir / "mappings.json"
        if mappings_file.exists():
            with open(mappings_file, "r", encoding="utf-8") as f:
                mappings = json.load(f)
                historical_mapping = mappings.get("historical_mapping", {})
                future_mapping = mappings.get("future_mapping", {})
            tailored_settings = mappings.get("tailored_settings", {})
        
        # Load manual rooms available settings from metadata
        metadata = _load_metadata()
        dataset_meta = metadata.get(name, {})
        use_manual_rooms_available = dataset_meta.get("use_manual_rooms_available", False)
        manual_rooms_available = dataset_meta.get("manual_rooms_available")
        
        return historical_df, future_df, events_df, budget_df, historical_mapping, future_mapping, use_manual_rooms_available, manual_rooms_available, tailored_settings
    
    except Exception as e:
        print(f"Error loading dataset '{name}': {e}")
        return None, None, None, None, {}, {}, False, None, {}


def list_datasets() -> List[str]:
    """List all saved dataset names."""
    _ensure_datasets_dir()
    metadata = _load_metadata()
    return sorted(metadata.keys())


def get_dataset_info(name: str) -> Optional[Dict]:
    """Get metadata for a specific dataset."""
    metadata = _load_metadata()
    return metadata.get(name)


def delete_dataset(name: str) -> bool:
    """
    Delete a saved dataset.
    
    Args:
        name: Dataset name
    
    Returns:
        True if successful, False otherwise
    """
    try:
        dataset_dir = DATASETS_DIR / name
        
        # Delete dataset directory
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        
        # Remove from metadata
        metadata = _load_metadata()
        if name in metadata:
            del metadata[name]
            _save_metadata(metadata)
        
        return True
    
    except Exception as e:
        print(f"Error deleting dataset '{name}': {e}")
        return False


def save_budget_profile(hotel_name: str, budget_df: pd.DataFrame) -> bool:
    """Save a budget profile for a hotel/property."""
    try:
        profile_name = _sanitize_dataset_name(hotel_name)
        if not profile_name or budget_df is None or len(budget_df) == 0:
            return False

        _ensure_budgets_dir()
        file_path = BUDGETS_DIR / f"{profile_name}.csv"
        budget_df.to_csv(file_path, index=False)

        metadata = _load_budget_metadata()
        now_iso = datetime.now().isoformat()
        columns = [str(c).strip().lower() for c in budget_df.columns]
        budget_type = "unknown"
        if {"stay_date", "budget_revenue"}.issubset(set(columns)):
            budget_type = "daily"
        elif {"year", "month", "budget_revenue"}.issubset(set(columns)):
            budget_type = "monthly"

        existing = metadata.get(profile_name, {})
        metadata[profile_name] = {
            "created_at": existing.get("created_at", now_iso),
            "updated_at": now_iso,
            "rows": int(len(budget_df)),
            "budget_type": budget_type,
            "columns": columns,
        }
        _save_budget_metadata(metadata)
        return True
    except Exception as e:
        print(f"Error saving budget profile '{hotel_name}': {e}")
        return False


def load_budget_profile(hotel_name: str) -> Optional[pd.DataFrame]:
    """Load a saved budget profile for a hotel/property."""
    try:
        profile_name = _sanitize_dataset_name(hotel_name)
        file_path = BUDGETS_DIR / f"{profile_name}.csv"
        if not file_path.exists():
            return None
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading budget profile '{hotel_name}': {e}")
        return None


def list_budget_profiles() -> List[str]:
    """List all saved budget profile names."""
    _ensure_budgets_dir()
    metadata = _load_budget_metadata()
    return sorted(metadata.keys())


