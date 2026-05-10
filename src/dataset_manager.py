"""Dataset management for saving and loading PMS reports."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


DATASETS_DIR = Path("datasets")
METADATA_FILE = DATASETS_DIR / "datasets_metadata.json"


def _ensure_datasets_dir() -> None:
    """Create datasets directory if it doesn't exist."""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)


def _load_metadata() -> Dict[str, dict]:
    """Load dataset metadata from JSON file."""
    _ensure_datasets_dir()
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_metadata(metadata: Dict[str, dict]) -> None:
    """Save dataset metadata to JSON file."""
    _ensure_datasets_dir()
    with open(METADATA_FILE, "w") as f:
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
        name = name.strip()
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
        }
        with open(dataset_dir / "mappings.json", "w") as f:
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
        mappings_file = dataset_dir / "mappings.json"
        if mappings_file.exists():
            with open(mappings_file, "r") as f:
                mappings = json.load(f)
                historical_mapping = mappings.get("historical_mapping", {})
                future_mapping = mappings.get("future_mapping", {})
        
        # Load manual rooms available settings from metadata
        metadata = _load_metadata()
        dataset_meta = metadata.get(name, {})
        use_manual_rooms_available = dataset_meta.get("use_manual_rooms_available", False)
        manual_rooms_available = dataset_meta.get("manual_rooms_available")
        
        return historical_df, future_df, events_df, budget_df, historical_mapping, future_mapping, use_manual_rooms_available, manual_rooms_available
    
    except Exception as e:
        print(f"Error loading dataset '{name}': {e}")
        return None, None, None, None, {}, {}, False, None


def list_datasets() -> List[str]:
    """List all saved dataset names."""
    _ensure_datasets_dir()
    metadata = _load_metadata()
    return list(metadata.keys())


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
            import shutil
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


def rename_dataset(old_name: str, new_name: str) -> bool:
    """Rename a saved dataset."""
    try:
        old_dir = DATASETS_DIR / old_name
        new_dir = DATASETS_DIR / new_name
        
        if not old_dir.exists():
            return False
        
        # Rename directory
        old_dir.rename(new_dir)
        
        # Update metadata
        metadata = _load_metadata()
        if old_name in metadata:
            metadata[new_name] = metadata.pop(old_name)
            _save_metadata(metadata)
        
        return True
    
    except Exception as e:
        print(f"Error renaming dataset '{old_name}' to '{new_name}': {e}")
        return False
