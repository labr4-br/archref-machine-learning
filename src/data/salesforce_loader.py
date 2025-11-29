from typing import Any, Literal
import pandas as pd
from src.core.base_loader import BaseDataLoader

class SalesforceDataLoader(BaseDataLoader):
    
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        data_type: Literal["leads", "opportunities"] = "leads",
    ) -> None:
        
        super().__init__(config)
        self.data_type = data_type
        
        import os
        base_path = f"data/raw/sf_{data_type}_raw.csv"
        if os.path.exists(base_path):
            self.raw_path = base_path
        elif os.path.exists(f"../{base_path}"):
            self.raw_path = f"../{base_path}"
        else:
             raise FileNotFoundError(f"Could not find data file for {data_type} at {base_path} or ../{base_path}")

    def load_raw(self) -> pd.DataFrame:
        
        return pd.read_csv(self.raw_path)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df = df.copy()

        if "CreatedDate" in df.columns:
            df["CreatedDate"] = pd.to_datetime(df["CreatedDate"])
        
        if "CloseDate" in df.columns:
            df["CloseDate"] = pd.to_datetime(df["CloseDate"])

        if self.data_type == "leads":
            if "Status" in df.columns:
                df["Status"] = df["Status"].str.title()
                df["IsConverted"] = df["Status"].apply(lambda x: 1 if "Converted" in str(x) and "Not" not in str(x) else 0)
        
        elif self.data_type == "opportunities":
            if "Amount" in df.columns:
                df["Amount"] = df["Amount"].fillna(0.0)

        columns_to_drop = ["Id", "FirstName", "LastName", "Company", "Name", "LeadId"]
        df = df.drop(columns=[c for c in columns_to_drop if c in df.columns], errors="ignore")

        return df
