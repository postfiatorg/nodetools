import pandas as pd
import requests
from nodetools.utilities.generic_pft_utilities import GenericPFTUtilities
import nodetools.utilities.constants as constants

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

    def get_user_context_document(self, account_address):
        """
        Get user's context document information.
        
        Args:
            account_address: XRPL account address
            
        Returns:
            dict containing:
                - doc_link: URL of the Google Doc
                - full_content: Complete document content
                - verification_section: Extracted verification section
                - last_updated: Datetime of last update
                - status: Success/error status message
        """
        result = {
            'doc_link': None,
            'full_content': '',
            'verification_section': '',
            'last_updated': None,
            'status': 'No document found'
        }
        
        # Get memo history
        memo_history = self.generic_pft_utilities.get_account_memo_history(
            account_address=account_address
        )
        
        # Get most recent Google doc link
        google_doc_memos = memo_history[memo_history['memo_type'] == 'google_doc_context_link']
        if google_doc_memos.empty:
            return result
            
        doc_link = google_doc_memos.sort_values('datetime').iloc[-1]['memo_data']
        result['doc_link'] = doc_link
        result['last_updated'] = google_doc_memos.sort_values('datetime').iloc[-1]['datetime']
        
        # Get doc content
        try:
            doc_id = doc_link.split('/')[5]
            url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
            response = requests.get(url)
            
            if response.status_code == 200:
                content = response.text
                result['full_content'] = content
                
                # Extract verification section if present
                try:
                    verification_text = content.split('TASK VERIFICATION SECTION START')[-1].split('TASK VERIFICATION SECTION END')[0].strip()
                    result['verification_section'] = verification_text
                except:
                    pass
                    
                result['status'] = 'Success'
            else:
                result['status'] = f"Failed to retrieve document. Status code: {response.status_code}"
                
        except Exception as e:
            result['status'] = f"Error retrieving document: {str(e)}"
            
        return result
