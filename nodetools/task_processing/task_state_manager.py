from typing import Union
import pandas as pd
from ..configuration import constants
from ..utilities.generic_pft_utilities import GenericPFTUtilities

class TaskStateManager:
    """Manages task state transitions and proposal tracking."""

    def __init__(self, pft_utilities: GenericPFTUtilities):
        self.pft_utilities = pft_utilities

    def get_task_state_pairs(self, account_memo_detail_df):
        """Convert account info into a DataFrame of proposed tasks and their latest state changes.
        
        Args:
            account_memo_detail_df: DataFrame containing account memo details
            
        Returns:
            DataFrame with columns:
                - proposal: The proposed task text
                - latest_state: The most recent state change (acceptance/refusal/verification/reward)
                - state_type: The type of the latest state (TaskType enum)
        """
        task_frame = self.pft_utilities.convert_all_account_info_into_simplified_task_frame(
            account_memo_detail_df=account_memo_detail_df.sort_values('datetime')
        )

        if task_frame.empty:
            return pd.DataFrame()

        # Rename columns for clarity
        task_frame.rename(columns={
            'MemoType': 'task_id',
            'MemoData': 'full_output',
            'MemoFormat': 'user_account'
        }, inplace=True)

        # Get proposals
        proposals = task_frame[
            task_frame['task_type']==constants.TaskType.PROPOSAL.name
        ].groupby('task_id').first()['full_output']

        # Get latest state changes (including verification and rewards)
        state_changes = task_frame[
            (task_frame['task_type'].isin([
                constants.TaskType.ACCEPTANCE.name,
                constants.TaskType.REFUSAL.name,
                constants.TaskType.VERIFICATION_PROMPT.name,
                constants.TaskType.REWARD.name
            ]))
        ].groupby('task_id').last()[['full_output','task_type', 'datetime']]

        # Combine proposals and state changes, keeping all proposals
        task_pairs = pd.DataFrame({'proposal': proposals})
        task_pairs['latest_state'] = state_changes['full_output'].fillna('')
        task_pairs['state_type'] = state_changes['task_type'].fillna('')
        task_pairs['datetime'] = state_changes['datetime']

        return task_pairs
    
    def get_proposals_by_state(
            self, 
            account: Union[str, pd.DataFrame], 
            state_type: constants.TaskType
        ):
        """Get proposals filtered by their state.
    
        Args:
        account: Either an XRPL account address string or a DataFrame containing memo history.
            If string, memo history will be fetched for that address.
            If DataFrame, it must contain memo history in the expected format & filtered for the account in question.
        state_type: TaskType enum value to filter by (e.g. TaskType.PROPOSAL for pending proposals)
             
        Returns:
            DataFrame with columns based on state:
                - proposal: The proposed task text (always present)
                - acceptance/refusal/verification/reward: The state-specific text (except for PROPOSAL)
            Indexed by task_id.
        """
        # Handle input type
        if isinstance(account, str):
            account_memo_detail_df = self.pft_utilities.get_account_memo_history(account_address=account)
        else:
            account_memo_detail_df = account

        # Get base task pairs
        task_pairs = self.get_task_state_pairs(account_memo_detail_df)

        if task_pairs.empty:
            return pd.DataFrame()

        if state_type == constants.TaskType.PROPOSAL:
            # Handle pending proposals (only those with no state changes or just the proposal)
            filtered_proposals = task_pairs[
                (task_pairs['state_type'] == '') |
                (task_pairs['state_type'] == constants.TaskType.PROPOSAL.name)
            ][['proposal']]
            return filtered_proposals
        
        # Filter to requested state
        filtered_proposals = task_pairs[
            task_pairs['state_type'] == state_type.name
        ][['proposal', 'latest_state']].copy()
        
        # Map state types to column names
        state_column_map = {
            constants.TaskType.ACCEPTANCE: 'acceptance',
            constants.TaskType.REFUSAL: 'refusal',
            constants.TaskType.VERIFICATION_PROMPT: 'verification',
            constants.TaskType.REWARD: 'reward'
        }
        
        # Rename latest_state column based on state type
        filtered_proposals.rename(columns={
            'latest_state': state_column_map[state_type]
        }, inplace=True)
        
        # Clean up text content
        filtered_proposals[state_column_map[state_type]] = filtered_proposals[state_column_map[state_type]].apply(
            lambda x: str(x).replace(state_type.value, '').replace('nan', '')
        )
        filtered_proposals['proposal'] = filtered_proposals['proposal'].apply(
            lambda x: str(x).replace(constants.TaskType.PROPOSAL.value, '').replace('nan', '')
        )

        return filtered_proposals
    
    def get_pending_proposals(self, account: Union[str, pd.DataFrame]):
        """Get proposals that have not yet been accepted or refused."""
        return self.get_proposals_by_state(account, state_type=constants.TaskType.PROPOSAL)

    def get_accepted_proposals(self, account: Union[str, pd.DataFrame]):
        """Get accepted proposals"""
        return self.get_proposals_by_state(account, state_type=constants.TaskType.ACCEPTANCE)
    
    def get_verification_proposals(self, account: Union[str, pd.DataFrame]):
        """Get verification proposals"""
        return self.get_proposals_by_state(account, state_type=constants.TaskType.VERIFICATION_PROMPT)

    def get_rewarded_proposals(self, account: Union[str, pd.DataFrame]):
        """Get rewarded proposals"""
        return self.get_proposals_by_state(account, state_type=constants.TaskType.REWARD)

    def get_refused_proposals(self, account: Union[str, pd.DataFrame]):
        """Get refused proposals"""
        return self.get_proposals_by_state(account, state_type=constants.TaskType.REFUSAL)
