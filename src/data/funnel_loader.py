from typing import Any
import pandas as pd
from src.core.base_loader import BaseDataLoader
from src.data.salesforce_loader import SalesforceDataLoader
from src.data.macro_loader import MacroDataLoader
from src.data.market_loader import MarketDataLoader
from src.data.ibge_loader import IbgeDataLoader

class FunnelDataLoader(BaseDataLoader):
    """
    Orchestrator loader that merges Salesforce, Macro, Market, and IBGE data.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.salesforce_loader = SalesforceDataLoader(config)
        self.macro_loader = MacroDataLoader(config)
        self.market_loader = MarketDataLoader(config)
        self.ibge_loader = IbgeDataLoader(config)

    def load_raw(self) -> pd.DataFrame:
        salesforce_df = self.salesforce_loader.load_and_process()
        macro_df = self.macro_loader.load_and_process()
        market_df = self.market_loader.load_and_process()
        ibge_df = self.ibge_loader.load_and_process()

        # 1. Merge Salesforce + Macro
        if 'CreatedDate' in salesforce_df.columns:
            salesforce_df['join_date'] = salesforce_df['CreatedDate'].dt.to_period('M').astype(str)
        
        if 'Date' in macro_df.columns:
            macro_df['join_date'] = macro_df['Date'].dt.to_period('M').astype(str)
            
        if 'Date' in market_df.columns:
            market_df['join_date'] = market_df['Date'].dt.to_period('M').astype(str)

        # Merge Macro
        # Left join to keep all salesforce records
        merged_df = pd.merge(
            salesforce_df,
            macro_df.drop(columns=['Date'], errors='ignore'),
            on='join_date',
            how='left'
        )

        # 2. Merge Market (Ibovespa)
        merged_df = pd.merge(
            merged_df,
            market_df.drop(columns=['Date'], errors='ignore'),
            on='join_date',
            how='left'
        )

        # 3. Merge IBGE (Demographic)
        if 'State' in merged_df.columns and 'State' in ibge_df.columns:
            merged_df = pd.merge(
                merged_df,
                ibge_df,
                on='State',
                how='left'
            )

        if 'join_date' in merged_df.columns:
            merged_df = merged_df.drop(columns=['join_date'])

        return merged_df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        return df
