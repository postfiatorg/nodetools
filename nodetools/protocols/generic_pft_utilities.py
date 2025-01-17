from typing import Protocol, Union, Optional, Dict, Any, List, Tuple
import pandas as pd
from xrpl.wallet import Wallet
from xrpl.models import Memo, Response
from decimal import Decimal
from nodetools.configuration.configuration import NetworkConfig, NodeConfig
from nodetools.utilities.db_manager import DBConnectionManager
from nodetools.utilities.xrpl_monitor import XRPLWebSocketMonitor
from nodetools.protocols.transaction_repository import TransactionRepository

class GenericPFTUtilities(Protocol):
    """Protocol defining the interface for GenericPFTUtilities implementations"""

    @property
    def network_config(self) -> NetworkConfig:
        """Get the network configuration"""
        ...

    @property
    def node_config(self) -> NodeConfig:
        """Get the node configuration"""
        ...

    @property
    def db_connection_manager(self) -> DBConnectionManager:
        """Get the database connection manager"""
        ...

    @property
    def xrpl_monitor(self) -> XRPLWebSocketMonitor:
        """Get the XRPL monitor"""
        ...

    @property
    def transaction_repository(self) -> TransactionRepository:
        """Get the transaction repository"""
        ...

    async def get_account_memo_history(self, account_address: str, pft_only: bool = True) -> pd.DataFrame:
        """Get memo history for a given account"""
        ...

    async def send_memo(self, 
            wallet_seed_or_wallet: Union[str, Wallet], 
            destination: str, 
            memo: Union[str, Memo], 
            username: str = None,
            message_id: str = None,
            chunk: bool = False,
            compress: bool = False, 
            encrypt: bool = False,
            pft_amount: Optional[Decimal] = None
        ) -> Union[Response, list[Response]]:
        """Send a memo to a given account"""
        ...
    
    def verify_transaction_response(self, response: Union[Response, list[Response]]) -> bool:
        """Verify a transaction response"""
        ...

    async def get_all_account_compressed_messages(self, account_address: str) -> pd.DataFrame:
        """Get all compressed messages for a given account"""
        ...

    def construct_handshake_memo(self, user: str, ecdh_public_key: str) -> str:
        """Construct a handshake memo"""
        ...

    def construct_memo(self, memo_data: str, memo_type: str, memo_format: str) -> Memo:
        """Construct a standardized memo object for XRPL transactions"""
        ...

    def spawn_wallet_from_seed(self, seed: str) -> Wallet:
        """Spawn a wallet from a seed"""
        ...

    async def get_recent_user_memos(self, account_address: str, num_messages: int) -> str:
        """Get the most recent messages from a user's memo history"""
        ...

    async def get_all_account_compressed_messages_for_remembrancer(self, account_address: str) -> pd.DataFrame:
        """Convenience method for getting all messages for a user from the remembrancer's perspective"""
        ...

    async def fetch_pft_balance(self, address: str) -> Decimal:
        """Get PFT balance for an account from the XRPL"""
        ...

    async def fetch_xrp_balance(self, address: str) -> Decimal:
        """Get XRP balance for an account from the XRPL"""
        ...

    async def get_pft_balance(self, account_address: str) -> Decimal:
        """Get PFT balance for an account from the database"""
        ...

    async def process_memo_data(
        self,
        memo_type: str,
        memo_data: str,
        decompress: bool = True,
        decrypt: bool = True,
        full_unchunk: bool = False, 
        memo_history: Optional[pd.DataFrame] = None,
        channel_address: Optional[str] = None,
        channel_counterparty: Optional[str] = None,
        channel_private_key: Optional[Union[str, Wallet]] = None
    ) -> str:
        ...

    async def verify_xrp_balance(self, address: str, minimum_xrp_balance: int) -> bool:
        """
        Verify that a wallet has sufficient XRP balance.
        
        Args:
            wallet: XRPL wallet object
            minimum_balance: Minimum required XRP balance
            
        Returns:
            tuple: (bool, float) - Whether balance check passed and current balance
        """
        ...

    async def handle_trust_line(self, wallet: Wallet, username: str):
        """
        Check and establish PFT trustline if needed.
        
        Args:
            wallet: XRPL wallet object
            username: Discord username

        Raises:
            Exception: If there is an error creating the trust line
        """
        ...

    def extract_transaction_info_from_response_object(self, response) -> dict:
        """
        Extract key information from an XRPL transaction response object.

        Args:
        response (Response): The XRPL transaction response object.

        Returns:
        dict: A dictionary containing extracted transaction information.
        """
        ...

    async def send_xrp(
            self,
            wallet_seed_or_wallet: Union[str, Wallet], 
            amount: Union[Decimal, int, float], 
            destination: str, 
            memo: Memo, 
            destination_tag: Optional[int] = None
        ):
        ...

    def extract_transaction_info_from_response_object__standard_xrp(self, response):
        """
        Extract key information from an XRPL transaction response object.
        
        Args:
        response (Response): The XRPL transaction response object.
        
        Returns:
        dict: A dictionary containing extracted transaction information.
        """
        ...

    @staticmethod
    def generate_custom_id():
        """ Generate a unique memo_type """
        ...

    async def fetch_pft_trustline_data(self, batch_size: int = 200) -> Dict[str, Dict[str, Any]]:
        """Get PFT token holder account information.
        
        Queries the XRPL for all accounts that have trustlines with the PFT issuer account.
        The balances are from the issuer's perspective, so they are negated to show actual
        holder balances (e.g., if issuer shows -100, holder has +100).

        Args:
            batch_size: Number of records to fetch per request (max 400)

        Returns:
            Dict of dictionaries with keys:
                - account (str): XRPL account address of the token holder
            and values:
                - balance (str): Raw balance string from XRPL
                - currency (str): Currency code (should be 'PFT')
                - limit_peer (str): Trustline limit
                - pft_holdings (float): Actual token balance (negated from issuer view)
        """
        ...

    async def fetch_formatted_transaction_history(
            self, 
            account_address: str,
            fetch_new_only: bool = True
        ) -> List[Dict[str, Any]]:
        """Fetch and format transaction history for an account.
        
        Retrieves transactions from XRPL and transforms them into a standardized
        format suitable for database storage.
        
        Args:
            account_address: XRPL account address to fetch transactions for
            fetch_new_only: If True, only fetch transactions after the last known ledger index.
                            If False, fetch entire transaction history.
                
        Returns:
            List of dictionaries containing processed transaction data with standardized fields
        """
        ...

    async def get_handshake_for_address(self, channel_address: str, channel_counterparty: str):
        """Get handshake for a specific address"""
        ...

    async def get_recent_messages(self, wallet_address: str):
        """Get recent messages for a given wallet address"""
        ...
