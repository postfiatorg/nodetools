from typing import Protocol, Union, Optional
import pandas as pd
from xrpl.wallet import Wallet
from xrpl.models import Memo
from decimal import Decimal

class GenericPFTUtilities(Protocol):
    """Protocol defining the interface for GenericPFTUtilities implementations"""

    def get_account_memo_history(self, account_address: str, pft_only: bool = True) -> pd.DataFrame:
        """Get memo history for a given account"""
        ...

    def send_memo(self, 
            wallet_seed_or_wallet: Union[str, Wallet], 
            destination: str, 
            memo: Union[str, Memo], 
            username: str = None,
            message_id: str = None,
            chunk: bool = False,
            compress: bool = False, 
            encrypt: bool = False,
            pft_amount: Optional[Decimal] = None
        ) -> Union[dict, list[dict]]:
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

    def construct_handshake_memo(self, user: str, ecdh_public_key: str) -> str:
        """Construct a handshake memo"""
        ...

    def spawn_wallet_from_seed(self, seed: str) -> Wallet:
        """Spawn a wallet from a seed"""
        ...
