from typing import Protocol
import pandas as pd

class OpenRouterTool(Protocol):

    def create_writable_df_for_async_chat_completion(self, arg_async_map: dict) -> pd.DataFrame:
        """Create DataFrame for async chat completion results"""
        ...

