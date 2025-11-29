from typing import Any
import pandas as pd
import os
from src.core.base_loader import BaseDataLoader

class IbgeDataLoader(BaseDataLoader):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        
        base_path = "data/external/ibge_pib_per_capita.csv"
        if os.path.exists(base_path):
            self.raw_path = base_path
        elif os.path.exists(f"../{base_path}"):
            self.raw_path = f"../{base_path}"
        else:
            self.raw_path = base_path

    def load_raw(self) -> pd.DataFrame:
        if not os.path.exists(self.raw_path):
            print(f"Warning: IBGE data file not found at {self.raw_path}")
            return pd.DataFrame()
        return pd.read_csv(self.raw_path)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
            
        df = df.copy()
        if "PIB_Per_Capita" in df.columns:
            df["PIB_Per_Capita"] = pd.to_numeric(df["PIB_Per_Capita"], errors='coerce')

        if "Population" in df.columns:
            df["Population"] = pd.to_numeric(df["Population"], errors='coerce')
            
        return df
