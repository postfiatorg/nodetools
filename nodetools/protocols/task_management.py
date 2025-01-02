from typing import Protocol
import pandas as pd

class PostFiatTaskGenerationSystem(Protocol):
    """Protocol defining the interface for TaskManagement implementations"""
    
    def get_pending_proposals(self, memo_history: pd.DataFrame) -> pd.DataFrame:
        """Get pending task proposals from memo history"""
        ...

    def get_accepted_proposals(self, memo_history: pd.DataFrame) -> pd.DataFrame:
        """Get accepted task proposals from memo history"""
        ...

    def get_refused_proposals(self, memo_history: pd.DataFrame) -> pd.DataFrame:
        """Get refused task proposals from memo history"""
        ...

    def get_verification_proposals(self, memo_history: pd.DataFrame) -> pd.DataFrame:
        """Get verification task proposals from memo history"""
        ...

    def get_rewarded_proposals(self, memo_history: pd.DataFrame) -> pd.DataFrame:
        """Get rewarded task proposals from memo history"""
        ...

    def discord__initiation_rite(
        self, 
        user_seed: str, 
        initiation_rite: str, 
        google_doc_link: str, 
        username: str,
        allow_reinitiation: bool = False
    ) -> str:
        """
        Process an initiation rite for a new user. Will raise exceptions if there are any issues.
        Immediately initiates handshake protocol with the node to enable encrypted memo communication.
        
        Args:
            user_seed (str): The user's wallet seed
            initiation_rite (str): The commitment message
            google_doc_link (str): Link to user's Google doc
            username (str): Discord username
        """
        ...

    def discord__update_google_doc_link(self, user_seed: str, google_doc_link: str, username: str):
        """Update the user's Google Doc link."""
        ...
