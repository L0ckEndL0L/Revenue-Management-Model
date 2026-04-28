"""
utils.py
Utility functions for the hotel RMS application.
"""

import os
from pathlib import Path
from datetime import datetime


def ensure_directory_exists(directory: str) -> Path:
    """
    Ensure a directory exists, create it if it doesn't.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object for the directory
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp() -> str:
    """
    Get current timestamp as a formatted string.
    
    Returns:
        Timestamp string in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def format_currency(amount: float) -> str:
    """
    Format a number as currency.
    
    Args:
        amount: Numeric amount
        
    Returns:
        Formatted currency string
    """
    return f"${amount:,.2f}"


def format_percentage(value: float) -> str:
    """
    Format a decimal as percentage.
    
    Args:
        value: Decimal value (e.g., 0.75 for 75%)
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.2f}%"


def validate_file_path(file_path: str) -> bool:
    """
    Validate that a file path exists and is accessible.
    
    Args:
        file_path: Path to validate
        
    Returns:
        True if valid, False otherwise
    """
    path = Path(file_path)
    return path.exists() and path.is_file()


def get_file_extension(file_path: str) -> str:
    """
    Get the file extension from a file path.
    
    Args:
        file_path: Path to file
        
    Returns:
        File extension (including the dot)
    """
    return Path(file_path).suffix.lower()


def print_section_header(title: str, width: int = 60) -> None:
    """
    Print a formatted section header.
    
    Args:
        title: Section title
        width: Width of the header line
    """
    print("\n" + "="*width)
    print(title)
    print("="*width + "\n")


def print_success(message: str) -> None:
    """
    Print a success message.
    
    Args:
        message: Success message
    """
    print(f"[SUCCESS] {message}")


def print_warning(message: str) -> None:
    """
    Print a warning message.
    
    Args:
        message: Warning message
    """
    print(f"[WARNING] {message}")


def print_error(message: str) -> None:
    """
    Print an error message.
    
    Args:
        message: Error message
    """
    print(f"[ERROR] {message}")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return if denominator is zero
        
    Returns:
        Result of division or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator


def truncate_string(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """
    Truncate a string to a maximum length.
    
    Args:
        text: String to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def get_output_directory(base_dir: str = "outputs") -> Path:
    """
    Get the output directory path, creating it if necessary.
    
    Args:
        base_dir: Base directory name
        
    Returns:
        Path object for output directory
    """
    return ensure_directory_exists(base_dir)
