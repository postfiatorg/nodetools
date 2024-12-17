import pandas as pd
from nodetools.protocols.task_management import PostFiatTaskGenerationSystem
from nodetools.protocols.generic_pft_utilities import GenericPFTUtilities
import nodetools.configuration.constants as constants
from typing import Optional
from loguru import logger
import traceback

class UserTaskParser:
    _instance = None  # TODO: Singleton pattern might not be necessary here
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
            self,
            task_management_system: PostFiatTaskGenerationSystem,
            generic_pft_utilities: GenericPFTUtilities
        ):
        """Initialize UserTaskParser with GenericPFTUtilities for core functionality"""
        if not self.__class__._initialized:
            self.task_management_system = task_management_system
            self.generic_pft_utilities = generic_pft_utilities
            self.__class__._initialized = True

    def get_task_statistics(self, account_address):
        """
        Get statistics about user's tasks.
        
        Args:
            account_address: XRPL account address to get stats for
            
        Returns:
            dict containing:
                - total_tasks: Total number of tasks
                - accepted_tasks: Number of accepted tasks
                - pending_tasks: Number of pending tasks
                - acceptance_rate: Percentage of tasks accepted
        """
        account_memo_detail_df = self.generic_pft_utilities.get_account_memo_history(account_address)

        pending_proposals = self.task_management_system.get_pending_proposals(account_memo_detail_df)
        accepted_proposals = self.task_management_system.get_accepted_proposals(account_memo_detail_df)
        refused_proposals = self.task_management_system.get_refused_proposals(account_memo_detail_df)
        verification_proposals = self.task_management_system.get_verification_proposals(account_memo_detail_df)
        rewarded_proposals = self.task_management_system.get_rewarded_proposals(account_memo_detail_df)

        # Calculate total accepted tasks
        total_accepted = len(accepted_proposals) + len(verification_proposals) + len(rewarded_proposals)

        # Total tasks excluding pending
        total_ended_tasks = total_accepted + len(refused_proposals)

        # Total tasks
        total_tasks = total_ended_tasks + len(pending_proposals)
            
        # Calculate rates
        acceptance_rate = (total_accepted / total_tasks * 100) if total_tasks > 0 else 0
        completion_rate = (len(rewarded_proposals) / total_ended_tasks * 100) if total_ended_tasks > 0 else 0
        
        return {
            'total_tasks': total_tasks,
            'total_ended_tasks': total_ended_tasks,
            'total_completed_tasks': len(rewarded_proposals),
            'total_pending_tasks': len(pending_proposals),
            'acceptance_rate': acceptance_rate,
            'completion_rate': completion_rate
        }

    def get_full_user_context_string(
        self,
        account_address: str,
        memo_history: Optional[pd.DataFrame] = None,
        get_google_doc: bool = True,
        get_historical_memos: bool = True,
        n_memos_in_context: int = constants.MAX_CHUNK_MESSAGES_IN_CONTEXT,
        n_pending_proposals_in_context: int = constants.MAX_PENDING_PROPOSALS_IN_CONTEXT,
        n_acceptances_in_context: int = constants.MAX_ACCEPTANCES_IN_CONTEXT,
        n_verification_in_context: int = constants.MAX_ACCEPTANCES_IN_CONTEXT,
        n_rewards_in_context: int = constants.MAX_REWARDS_IN_CONTEXT,
        n_refusals_in_context: int = constants.MAX_REFUSALS_IN_CONTEXT,
    ) -> str:
        """Get complete user context including task states and optional content.
        
        Args:
            account_address: XRPL account address
            memo_history: Optional pre-fetched memo history DataFrame to avoid requerying
            get_google_doc: Whether to fetch Google doc content
            get_historical_memos: Whether to fetch historical memos
            n_task_context_history: Number of historical items to include
        """
        # Use provided memo_history or fetch if not provided
        if memo_history is None:
            memo_history = self.generic_pft_utilities.get_account_memo_history(account_address=account_address)

        # Handle proposals section (pending + accepted)
        try:
            pending_proposals = self.task_management_system.get_pending_proposals(memo_history)
            accepted_proposals = self.task_management_system.get_accepted_proposals(memo_history)

            # Combine and limit
            all_proposals = pd.concat([pending_proposals, accepted_proposals]).tail(
                n_acceptances_in_context + n_pending_proposals_in_context
            )

            if all_proposals.empty:
                proposal_string = "No pending or accepted proposals found."
            else:
                proposal_string = self.format_task_section(all_proposals, constants.TaskType.PROPOSAL)
        
        except Exception as e:
            logger.error(f"UserTaskParser.get_full_user_context_string: Failed to get pending or accepted proposals: {e}")
            logger.error(traceback.format_exc())
            proposal_string = "Error retrieving pending or accepted proposals."

        # Handle refusals
        try:
            refused_proposals = self.task_management_system.get_refused_proposals(memo_history).tail(n_refusals_in_context)
            if refused_proposals.empty:
                refusal_string = "No refused proposals found."
            else:
                refusal_string = self.format_task_section(refused_proposals, constants.TaskType.REFUSAL)
        except Exception as e:
            logger.error(f"UserTaskParser.get_full_user_context_string: Failed to get refused proposals: {e}")
            logger.error(traceback.format_exc())
            refusal_string = "Error retrieving refused proposals."
            
        # Handle verifications
        try:
            verification_proposals = self.task_management_system.get_verification_proposals(memo_history).tail(n_verification_in_context)
            if verification_proposals.empty:
                verification_string = "No tasks pending verification."
            else:
                verification_string = self.format_task_section(verification_proposals, constants.TaskType.VERIFICATION_PROMPT)
        except Exception as e:
            logger.error(f'UserTaskParser.get_full_user_context_string: Exception while retrieving verifications for {account_address}: {e}')
            logger.error(traceback.format_exc())
            verification_string = "Error retrieving verifications."    

        # Handle rewards
        try:
            rewarded_proposals = self.task_management_system.get_rewarded_proposals(memo_history).tail(n_rewards_in_context)
            if rewarded_proposals.empty:
                reward_string = "No rewarded tasks found."
            else:
                reward_string = self.format_task_section(rewarded_proposals, constants.TaskType.REWARD)
        except Exception as e:
            logger.error(f'UserTaskParser.get_full_user_context_string: Exception while retrieving rewards for {account_address}: {e}')
            logger.error(traceback.format_exc())
            reward_string = "Error retrieving rewards."

        # Get optional context elements
        if get_google_doc:
            try:
                google_url = self.generic_pft_utilities.get_latest_outgoing_context_doc_link(
                    account_address=account_address, 
                    memo_history=memo_history
                )
                core_element__google_doc_text = self.generic_pft_utilities.get_google_doc_text(google_url)
            except Exception as e:
                logger.error(f"Failed retrieving user google doc: {e}")
                logger.error(traceback.format_exc())
                core_element__google_doc_text = 'Error retrieving google doc'

        if get_historical_memos:
            try:
                core_element__user_log_history = self.generic_pft_utilities.get_recent_user_memos(
                    account_address=account_address,
                    num_messages=n_memos_in_context
                )
            except Exception as e:
                logger.error(f"Failed retrieving user memo history: {e}")
                logger.error(traceback.format_exc())
                core_element__user_log_history = 'Error retrieving user memo history'

        core_elements = f"""
        ***<<< ALL TASK GENERATION CONTEXT STARTS HERE >>>***

        These are the proposed and accepted tasks that the user has. This is their
        current work queue
        <<PROPOSED AND ACCEPTED TASKS START HERE>>
        {proposal_string}
        <<PROPOSED AND ACCEPTED TASKS ENDE HERE>>

        These are the tasks that the user has been proposed and has refused.
        The user has provided a refusal reason with each one. Only their most recent
        {n_refusals_in_context} refused tasks are showing 
        <<REFUSED TASKS START HERE >>
        {refusal_string}
        <<REFUSED TASKS END HERE>>

        These are the tasks that the user has for pending verification.
        They need to submit details
        <<VERIFICATION TASKS START HERE>>
        {verification_string}
        <<VERIFICATION TASKS END HERE>>

        <<REWARDED TASKS START HERE >>
        {reward_string}
        <<REWARDED TASKS END HERE >>
        """

        optional_elements = ''
        if get_google_doc:
            optional_elements += f"""
            The following is the user's full planning document that they have assembled
            to inform task generation and planning
            <<USER PLANNING DOC STARTS HERE>>
            {core_element__google_doc_text}
            <<USER PLANNING DOC ENDS HERE>>
            """

        if get_historical_memos:
            optional_elements += f"""
            The following is the users own comments regarding everything
            <<< USER COMMENTS AND LOGS START HERE>>
            {core_element__user_log_history}
            <<< USER COMMENTS AND LOGS END HERE>>
            """

        footer = f"""
        ***<<< ALL TASK GENERATION CONTEXT ENDS HERE >>>***
        """

        return core_elements + optional_elements + footer
    
    def format_task_section(self, task_df: pd.DataFrame, state_type: constants.TaskType) -> str:
        """Format tasks for display based on their state type.
        
        Args:
            task_df: DataFrame containing tasks with columns:
                - proposal: The proposed task text
                - acceptance/refusal/verification/reward: The state-specific text
                - datetime: Optional timestamp of state change
            state_type: TaskType enum indicating the state to format for
            
        Returns:
            Formatted string representation with columns:
                - initial_task_detail: Original proposal
                - recent_status: State-specific text or status
                - recent_date: From datetime if available, otherwise from task_id
        """
        if task_df.empty:
            return f"No {state_type.name.lower()} tasks found."

        formatted_df = pd.DataFrame(index=task_df.index)
        formatted_df['initial_task_detail'] = task_df['proposal']

        # Use actual datetime if available, otherwise extract from task_id
        if 'datetime' in task_df.columns:
            formatted_df['recent_date'] = task_df['datetime'].dt.strftime('%Y-%m-%d')
        else:
            formatted_df['recent_date'] = task_df.index.map(
                lambda x: x.split('_')[0] if '_' in x else ''
            )

        # Map state types to their column names and expected status text
        state_column_map = {
            constants.TaskType.PROPOSAL: ('acceptance', lambda x: x if pd.notna(x) and str(x).strip() else "Pending response"),
            constants.TaskType.ACCEPTANCE: ('acceptance', lambda x: x),
            constants.TaskType.REFUSAL: ('refusal', lambda x: x),
            constants.TaskType.VERIFICATION_PROMPT: ('verification', lambda x: x),
            constants.TaskType.REWARD: ('reward', lambda x: x)
        }
        
        column_name, status_formatter = state_column_map[state_type]
        if column_name in task_df.columns:
            formatted_df['recent_status'] = task_df[column_name].apply(status_formatter)
        else:
            formatted_df['recent_status'] = "Status not available"
        
        return formatted_df[['initial_task_detail', 'recent_status', 'recent_date']].to_string()