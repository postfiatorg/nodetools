import pandas as pd
import requests
from nodetools.utilities.generic_pft_utilities import GenericPFTUtilities
import nodetools.configuration.constants as constants
from typing import Optional
from loguru import logger
import traceback

class UserTaskParser:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize UserTaskParser with GenericPFTUtilities for core functionality"""
        if not self.__class__._initialized:
            self.generic_pft_utilities = GenericPFTUtilities()
            self.__class__._initialized = True

    def _determine_if_map_is_task_id(self, memo_dict):
        """Check if memo contains a task ID"""
        import re
        full_memo_string = str(memo_dict)
        task_id_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}(?:__[A-Z0-9]{4})?)')
        return bool(re.search(task_id_pattern, full_memo_string))

    def _convert_all_account_info_into_simplified_task_frame(self, account_memo_detail_df):
        """Convert account info into a simplified task frame"""
        # Filter for task-related memos
        simplified_task_frame = account_memo_detail_df[
            account_memo_detail_df['converted_memos'].apply(self._determine_if_map_is_task_id)
        ].copy()

        # Add metadata fields to converted_memos
        def add_field_to_map(row, field):
            memo_dict = row['converted_memos']
            memo_dict[field] = row[field]
            return memo_dict

        for field in ['hash', 'datetime']:
            simplified_task_frame['converted_memos'] = simplified_task_frame.apply(
                lambda x: add_field_to_map(x, field), 
                axis=1
            )

        # Convert to core task dataframe
        core_task_df = pd.DataFrame(list(simplified_task_frame['converted_memos'])).copy()
        
        # Classify task types
        core_task_df['task_type'] = core_task_df['MemoData'].apply(
            lambda x: self.generic_pft_utilities.classify_task_string(x)
        )

        return core_task_df

    def get_outstanding_tasks(self, account_address):
        """
        Get a clean dataframe of user's accepted and proposed tasks.
        
        Args:
            account_address: XRPL account address to get tasks for
            
        Returns:
            DataFrame with columns:
                - proposal: The proposed task text
                - acceptance: The acceptance text (empty if not accepted)
            Indexed by task_id
        """
        # Get account memo details
        account_memo_detail_df = self.generic_pft_utilities.get_account_memo_history(
            account_address=account_address
        )
        
        # Sort by datetime
        account_memo_detail_df = account_memo_detail_df.sort_values('datetime')
        
        # Convert to task frame
        task_frame = self._convert_all_account_info_into_simplified_task_frame(
            account_memo_detail_df=account_memo_detail_df
        )
        
        # Map memo fields to task fields
        task_frame['task_id'] = task_frame['MemoType']
        task_frame['full_output'] = task_frame['MemoData']
        task_frame['user_account'] = task_frame['MemoFormat']
        
        # Get task type mapping
        task_type_map = task_frame.groupby('task_id').last()[['task_type']].copy()
        
        # Get proposals and acceptances
        task_id_to_proposal = task_frame[
            task_frame['task_type'] == 'PROPOSAL'
        ].groupby('task_id').first()['full_output']
        
        task_id_to_acceptance = task_frame[
            task_frame['task_type'] == 'ACCEPTANCE'
        ].groupby('task_id').first()['full_output']
        
        # Combine into acceptance frame
        acceptance_frame = pd.concat([task_id_to_proposal, task_id_to_acceptance], axis=1)
        acceptance_frame.columns = ['proposal', 'acceptance_raw']
        
        # Clean up text
        acceptance_frame['acceptance'] = acceptance_frame['acceptance_raw'].apply(
            lambda x: str(x).replace('ACCEPTANCE REASON ___ ', '').replace('nan', '')
        )
        acceptance_frame['proposal'] = acceptance_frame['proposal'].apply(
            lambda x: str(x).replace('PROPOSED PF ___ ', '').replace('nan', '')
        )
        
        # Get final proposals and acceptances
        raw_proposals_and_acceptances = acceptance_frame[['proposal', 'acceptance']].copy()
        
        # Filter for proposed or accepted tasks
        proposed_or_accepted_only = list(
            task_type_map[
                (task_type_map['task_type'] == 'ACCEPTANCE') |
                (task_type_map['task_type'] == 'PROPOSAL')
            ].index
        )
        
        # Return filtered frame
        return raw_proposals_and_acceptances[
            raw_proposals_and_acceptances.index.get_level_values(0).isin(proposed_or_accepted_only)
        ]

    # NOTE: This isn't being used anywhere
    # TODO: Refactor this to use get_proposals, get_acceptances, get_refusals, get_verifications, get_rewards
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
        tasks_df = self.get_outstanding_tasks(account_address)
        
        total_tasks = len(tasks_df)
        accepted_tasks = len(tasks_df[tasks_df['acceptance'].str.strip() != ''])
        pending_tasks = total_tasks - accepted_tasks
        acceptance_rate = (accepted_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        return {
            'total_tasks': total_tasks,
            'accepted_tasks': accepted_tasks,
            'pending_tasks': pending_tasks,
            'acceptance_rate': acceptance_rate
        }

    # NOTE: This wont work since it doesn't decrypt
    # def get_user_context_document(self, account_address):
    #     """
    #     Get user's context document information.
        
    #     Args:
    #         account_address: XRPL account address
            
    #     Returns:
    #         dict containing:
    #             - doc_link: URL of the Google Doc
    #             - full_content: Complete document content
    #             - verification_section: Extracted verification section
    #             - last_updated: Datetime of last update
    #             - status: Success/error status message
    #     """
    #     result = {
    #         'doc_link': None,
    #         'full_content': '',
    #         'verification_section': '',
    #         'last_updated': None,
    #         'status': 'No document found'
    #     }
        
    #     # Get memo history
    #     memo_history = self.generic_pft_utilities.get_account_memo_history(
    #         account_address=account_address
    #     )
        
    #     # Get most recent Google doc link
    #     google_doc_memos = memo_history[memo_history['memo_type'] == 'google_doc_context_link']
    #     if google_doc_memos.empty:
    #         return result
            
    #     doc_link = google_doc_memos.sort_values('datetime').iloc[-1]['memo_data']
    #     result['doc_link'] = doc_link
    #     result['last_updated'] = google_doc_memos.sort_values('datetime').iloc[-1]['datetime']
        
    #     # Get doc content
    #     try:
    #         doc_id = doc_link.split('/')[5]
    #         url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    #         response = requests.get(url)
            
    #         if response.status_code == 200:
    #             content = response.text
    #             result['full_content'] = content
                
    #             # Extract verification section if present
    #             try:
    #                 verification_text = content.split('TASK VERIFICATION SECTION START')[-1].split('TASK VERIFICATION SECTION END')[0].strip()
    #                 result['verification_section'] = verification_text
    #             except:
    #                 pass
                    
    #             result['status'] = 'Success'
    #         else:
    #             result['status'] = f"Failed to retrieve document. Status code: {response.status_code}"
                
    #     except Exception as e:
    #         result['status'] = f"Error retrieving document: {str(e)}"
            
    #     return result
    
    def get_full_user_context_string(
        self,
        account_address: str,
        memo_history: Optional[pd.DataFrame] = None,
        get_google_doc: bool = True,
        get_historical_memos: bool = True,
        n_task_context_history: int = constants.MAX_CHUNK_MESSAGES_IN_CONTEXT,
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
            pending_proposals = self.generic_pft_utilities.get_pending_proposals(memo_history)
            accepted_proposals = self.generic_pft_utilities.get_accepted_proposals(memo_history)

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
            refused_proposals = self.generic_pft_utilities.get_refused_proposals(memo_history).tail(n_refusals_in_context)
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
            verification_proposals = self.generic_pft_utilities.get_verification_proposals(memo_history).tail(n_verification_in_context)
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
            rewarded_proposals = self.generic_pft_utilities.get_rewarded_proposals(memo_history).tail(n_rewards_in_context)
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
                    num_messages=n_task_context_history
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
        {n_task_context_history} refused tasks are showing 
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