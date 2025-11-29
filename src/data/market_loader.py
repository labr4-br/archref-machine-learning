from typing import Any
import pandas as pd
import os
from src.core.base_loader import BaseDataLoader

class MarketDataLoader(BaseDataLoader):

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)

        base_path = "data/external/b3_ibovespa_2025.csv"
        if os.path.exists(base_path):
            self.raw_path = base_path
        elif os.path.exists(f"../{base_path}"):
            self.raw_path = f"../{base_path}"
        else:
             self.raw_path = base_path

    def load_raw(self) -> pd.DataFrame:
        if not os.path.exists(self.raw_path):
            print(f"Warning: Market data file not found at {self.raw_path}")
            return pd.DataFrame()
        return pd.read_csv(self.raw_path)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:

        if df.empty:
            return df
            
        df = df.copy()
        
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            
        if "Close" in df.columns:
            df["Close"] = pd.to_numeric(df["Close"], errors='coerce')
            
        return df
