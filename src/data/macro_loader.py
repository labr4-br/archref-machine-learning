import os
from typing import Any
import pandas as pd
from src.core.base_loader import BaseDataLoader

class MacroDataLoader(BaseDataLoader):
    

    def __init__(
        self,
        config: dict[str, Any] | None = None,
    ) -> None:
        
        super().__init__(config)
        
        base_path = "data/external/macro_indicators_2025.csv"
        if os.path.exists(base_path):
            self.raw_path = base_path
        elif os.path.exists(f"../{base_path}"):
            self.raw_path = f"../{base_path}"
        else:
             raise FileNotFoundError(f"Could not find macro data file at {base_path} or ../{base_path}")

    def load_raw(self) -> pd.DataFrame:
        
        return pd.read_csv(self.raw_path)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    def merge_with_leads(self, leads_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
        
        leads_df = leads_df.copy()
        macro_df = macro_df.copy()
        leads_df["Month"] = pd.to_datetime(leads_df["CreatedDate"]).dt.to_period("M").dt.to_timestamp()
        macro_df["Date"] = pd.to_datetime(macro_df["Date"])
        
        merged_df = pd.merge(
            leads_df,
            macro_df,
            left_on="Month",
            right_on="Date",
            how="left"
        )
        
        merged_df = merged_df.drop(columns=["Month", "Date"])
        
        return merged_df
