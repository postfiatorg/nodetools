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