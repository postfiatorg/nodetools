from typing import Protocol
import pandas as pd

class GenericPFTUtilities(Protocol):
    """Protocol defining the interface for GenericPFTUtilities implementations"""

    def get_account_memo_history(self, account_address: str) -> pd.DataFrame:
        """Get memo history for a given account"""
        ...

    def send_memo(self, wallet_seed_or_wallet: str, username: str, destination: str, memo: str, message_id: str, chunk: bool, compress: bool, encrypt: bool) -> str:
        """Send a memo to a given account"""
        ...
    
    def verify_transaction_response(self, response: str) -> bool:
        """Verify a transaction response"""
        ...

    def get_all_account_compressed_messages(self, account_address: str) -> pd.DataFrame:
        """Get all compressed messages for a given account"""
        ...

    def get_post_fiat_holder_df(self) -> pd.DataFrame:
        """Get a DataFrame of all post-fiat holders"""
        ...