from typing import Protocol, Optional
import pandas as pd

class OpenRouterTool(Protocol):

    def create_writable_df_for_async_chat_completion(self, arg_async_map: dict) -> pd.DataFrame:
        """Create DataFrame for async chat completion results"""
        ...

    def generate_simple_text_output(
        self,
        model: str,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        ...

    async def generate_simple_text_output_async(
        self,
        model: str,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        ...