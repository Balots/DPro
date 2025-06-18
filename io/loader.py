import pandas as pd
import numpy as np
import logging
from functools import wraps
from typing import Union, Dict, List, Optional

def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Running {func.__name__}")
        result = func(*args, **kwargs)
        logging.info(f"Finished {func.__name__}")
        return result
    return wrapper

class DataLoader:
    @staticmethod
    @decorator
    def load_data(file_path: str, file_type: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Load data from various file formats"""
        if file_type is None:
            file_type = file_path.split('.')[-1].lower()
            
        try:
            if file_type == 'xlsx':
                return pd.read_excel(file_path, **kwargs)
            elif file_type == 'json':
                return pd.read_json(file_path, **kwargs)
            elif file_type == 'parquet':
                return pd.read_parquet(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            logging.error(f"Error loading {file_type} file: {str(e)}")
            raise

    @staticmethod
    @decorator
    def save_data(df: pd.DataFrame, file_path: str, file_type: Optional[str] = None, **kwargs) -> None:
        """Save data to various file formats"""
        if file_type is None:
            file_type = file_path.split('.')[-1].lower()
            
        try:
            if file_type == 'xlsx':
                df.to_excel(file_path, index=False, **kwargs)
            elif file_type == 'json':
                df.to_json(file_path, **kwargs)
            elif file_type == 'parquet':
                df.to_parquet(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            logging.info(f"Data successfully saved to {file_path}")
        except Exception as e:
            logging.error(f"Error saving {file_type} file: {str(e)}")
            raise
