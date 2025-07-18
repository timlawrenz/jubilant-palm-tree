"""
Data processing utilities for Ruby method datasets.

This module provides functions to load, preprocess, and prepare Ruby method
data for GNN training.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any


def load_methods_json(filepath: str) -> List[Dict[str, Any]]:
    """
    Load Ruby methods from JSON file.
    
    Args:
        filepath: Path to the JSON file containing method data
        
    Returns:
        List of method dictionaries
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def methods_to_dataframe(methods: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert list of method dictionaries to pandas DataFrame.
    
    Args:
        methods: List of method dictionaries
        
    Returns:
        DataFrame with method data
    """
    return pd.DataFrame(methods)


def filter_methods_by_length(df: pd.DataFrame, min_lines: int = 5, max_lines: int = 100) -> pd.DataFrame:
    """
    Filter methods by source code length.
    
    Args:
        df: DataFrame containing method data
        min_lines: Minimum number of lines
        max_lines: Maximum number of lines
        
    Returns:
        Filtered DataFrame
    """
    df['line_count'] = df['raw_source'].apply(lambda x: len(x.split('\n')))
    return df[(df['line_count'] >= min_lines) & (df['line_count'] <= max_lines)]