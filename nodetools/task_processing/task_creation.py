from nodetools.utilities.generic_pft_utilities import GenericPFTUtilities
from nodetools.utilities.credentials import CredentialManager
from nodetools.prompts.task_generation import (
    task_generation_one_shot_user_prompt,
    task_generation_one_shot_system_prompt
)
from nodetools.ai.openrouter import OpenRouterTool
import pandas as pd
import uuid
import re

class UserContext:
    def __init__(self, nCompressedHistory=20, nRewards=20, nRefused=20, nTasks=10):
        """
        Initialize UserContext with configurable limits for history and task sections.
        
        Args:
            nCompressedHistory (int): Number of compressed history messages to include
            nRewards (int): Number of recent rewards to include
            nRefused (int): Number of refused tasks to include
            nTasks (int): Number of current tasks to show in workflow
        """
        self.nCompressedHistory = nCompressedHistory
        self.nRewards = nRewards
        self.nRefused = nRefused
        self.nTasks = nTasks

class NewTaskGeneration:
    """
    Task Generation and Context Management System.
    
    This class handles task generation, context management, and user history processing.
    
    Example iPython Notebook Initialization:
    ```python
    # Import required modules
    from nodetools.task_processing.task_creation import NewTaskGeneration
    import getpass
    
    # Get password securely (will prompt for input)
    password = getpass.getpass('Enter password: ')
    
    # Initialize task generation system
    task_gen = NewTaskGeneration(password=password)
    
    # Example usage with an XRPL address
    account_address = 'rNC2hS269hTvMZwNakwHPkw4VeZNwzpS2E'
    context = task_gen.get_full_user_context_string(
        account_address=account_address,
        get_google_doc=True,
        get_historical_memos=True,
        n_task_context_history=20
    )
    print(context)
    ```

    EXAMPLE FULL USAGE FOR RUNNING A CUE 

    task_gen = NewTaskGeneration(password="your_password")

    # Create task map with combined account/task IDs
    task_map = {
        task_gen.create_task_key("rUWuJJLLSH5TUdajVqsHx7M59Vj3P7giQV", "task_id123"): "a task please",
        task_gen.create_task_key("rJzZLYK6JTg9NG1UA8g3D6fnJwd6vh3N4u", "task_id234"): "a planning task please",
        task_gen.create_task_key("rNC2hS269hTvMZwNakwHPkw4VeZNwzpS2E", "task_id245"): "a task please that continues my flow"
    }

    output_df = task_gen.process_task_map_to_proposed_pf(
        task_map=task_map,
        model="anthropic/claude-3.5-sonnet:beta",
        get_google_doc=True,
        get_historical_memos=True
    )
    This output_df
    output_df[['account_to_send_to','pf_proposal_string','pft_to_send','task_id']]
    has the key information you need to send to each account. can look at task_cue_replacement for a potential cue job 
    replacement 

    """
    
    _instance = None
    _initialized = False
    _credential_manager = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, password=None):
        """
        Initialize NewTaskGeneration with CredentialManager and create GenericPFTUtilities instance.
        
        Args:
            password (str, optional): Password for CredentialManager initialization. Required on first instance.
        """
        if not self.__class__._initialized:
            if not self.__class__._credential_manager:
                if password is None:
                    raise ValueError("Password is required for first NewTaskGeneration instance")
                self.__class__._credential_manager = CredentialManager(password=password)
            
            self.generic_pft_utilities = GenericPFTUtilities()
            self.user_context = UserContext()
            self.__class__._initialized = True

    def extract_final_output(self, text):
        """
        Extracts the content between 'Final Output |' and the last pipe character using regex.
        Returns 'NO OUTPUT' if no match is found.
        
        Args:
            text (str): The input text containing the Final Output section
            
        Returns:
            str: The extracted content between the markers, or 'NO OUTPUT' if not found
        """
        pattern = r"\|\s*Final Output\s*\|(.*?)\|\s*$"
        
        try:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
            return "NO OUTPUT"
        except Exception:
            return "NO OUTPUT"

    def process_task_map_to_proposed_pf(self, task_map, model="anthropic/claude-3.5-sonnet:beta", get_google_doc=True, get_historical_memos=True):
        """
        Process a task map to generate proposed PF tasks with rewards.
        
        Args:
            task_map (dict): Map of combined account/task IDs to task requests
                Example: {
                    "account_id__rUWuJJLLSH5TUdajVqsHx7M59Vj3P7giQV__task_id__task_id123": "a task please",
                    "account_id__rJzZLYK6JTg9NG1UA8g3D6fnJwd6vh3N4u__task_id__task_id234": "a planning task please"
                }
            model (str): Model identifier string (default: "anthropic/claude-3.5-sonnet:beta")
            get_google_doc (bool): Whether to fetch Google doc content
            get_historical_memos (bool): Whether to fetch historical memos
            
        Returns:
            pd.DataFrame: Processed DataFrame with proposed PF tasks and rewards
        """
        # Run batch task generation
        output_df = self.run_batch_task_generation(
            task_map=task_map,
            model=model,
            get_google_doc=get_google_doc,
            get_historical_memos=get_historical_memos
        )
        
        # Filter out invalid outputs and create proposal strings
        output_df = output_df[output_df['content'].apply(lambda x: self.extract_final_output(x)) != 'NO OUTPUT'].copy()
        output_df['pf_proposal_string'] = 'PROPOSED PF ___ ' + output_df['content'].apply(lambda x: self.extract_final_output(x)) + ' .. 900'
        output_df['reward'] = 900
        output_df['account_to_send_to']=output_df['internal_name'].apply(lambda x: x.split('task_gen__')[-1:][0].split('__')[0])
        output_df['pft_to_send']=1
        
        return output_df

    def create_task_key(self, account_id, task_id):
        """
        Create a combined key from account ID and task ID.
        
        Args:
            account_id (str): The account identifier
            task_id (str): The task identifier
            
        Returns:
            str: Combined key in format "account_id__{accountId}__task_id__{task_id}"
        """
        return f"account_id__{account_id}__task_id__{task_id}"

    def parse_task_key(self, task_key):
        """
        Parse a combined task key to extract account ID and task ID.
        
        Args:
            task_key (str): Combined key in format "account_id__{accountId}__task_id__{task_id}"
            
        Returns:
            tuple: (account_id, task_id)
        """
        parts = task_key.split("__")
        account_id = parts[1]
        task_id = parts[3]
        return account_id, task_id

    def run_batch_task_generation(self, task_map, model="anthropic/claude-3.5-sonnet:beta", get_google_doc=True, get_historical_memos=True):
        """
        Run batch task generation for multiple accounts asynchronously.
        
        Args:
            task_map (dict): Map of combined account/task IDs to task requests
                Keys should be in format "account_id__{accountId}__task_id__{task_id}"
                Example: {
                    "account_id__rUWuJJLLSH5TUdajVqsHx7M59Vj3P7giQV__task_id__123": "a task please",
                    "account_id__rJzZLYK6JTg9NG1UA8g3D6fnJwd6vh3N4u__task_id__456": "next task please"
                }
            model (str): Model identifier string (default: "anthropic/claude-3.5-sonnet:beta")
            get_google_doc (bool): Whether to fetch Google doc content
            get_historical_memos (bool): Whether to fetch historical memos
            
        Returns:
            pd.DataFrame: Results of the batch task generation with task IDs included
        """
        # Initialize OpenRouterTool
        openrouter = OpenRouterTool()
        
        # Create arg_async_map
        arg_async_map = {}
        for task_key, task_request in task_map.items():
            # Extract account_id and task_id from the combined key
            account_id, task_id = self.parse_task_key(task_key)
            
            # Generate unique job hash incorporating both IDs
            job_hash = f'task_gen__{account_id}__{task_id}__{uuid.uuid4()}'
            
            # Get API args for this account
            api_args = self.construct_task_generation_api_args(
                user_account_address=account_id,
                task_request=task_request,
                model=model,
                get_google_doc=get_google_doc,
                get_historical_memos=get_historical_memos
            )
            
            # Add to async map
            arg_async_map[job_hash] = api_args
        
        # Run async batch job and get results
        results_df = openrouter.create_writable_df_for_async_chat_completion(arg_async_map=arg_async_map)
        
        # Extract task IDs from internal_name column and add as new column
        results_df['task_id'] = results_df['internal_name'].apply(
            lambda x: x.split('__')[2] if len(x.split('__')) > 2 else None
        )
        
        return results_df

    def construct_task_generation_api_args(self, user_account_address, task_request, model, get_google_doc=True, get_historical_memos=True):
        """
        Construct API arguments for task generation using user context and task request.
        
        Args:
            user_account_address (str): XRPL account address
            task_request (str): User's task request
            model (str): Model identifier string
            get_google_doc (bool): Whether to fetch Google doc content
            get_historical_memos (bool): Whether to fetch historical memos
            
        Returns:
            dict: Formatted API arguments for task generation
        """
        # Get full user context
        user_context = self.get_full_user_context_string(
            account_address=user_account_address,
            get_google_doc=get_google_doc,
            get_historical_memos=get_historical_memos
        )
        
        # Replace placeholders in prompts
        user_prompt = task_generation_one_shot_user_prompt.replace(
            "___FULL_USER_CONTEXT_REPLACEMENT___",
            user_context
        ).replace(
            "___SELECTION_OPTION_REPLACEMENT___",
            task_request
        )
        
        # Construct API arguments
        api_args = {
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": task_generation_one_shot_system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
        
        return api_args

    def get_full_user_context_string(self, account_address, get_google_doc=True, get_historical_memos=True, n_task_context_history=20):
        """
        Get complete user context including memo history, task status, and Google doc content.
        
        Args:
            account_address (str): XRPL account address
            get_google_doc (bool): Whether to fetch Google doc content
            get_historical_memos (bool): Whether to fetch historical memos
            n_task_context_history (int): Number of historical items to include
            
        Returns:
            str: Formatted context string containing all user information
        """
        memo_history = self.generic_pft_utilities.get_account_memo_history(
            account_address=account_address,
            pft_only=False
        )
        
        simple_user_memo_history = memo_history.sort_values('datetime')
        simple_user_memo_history = simple_user_memo_history[~simple_user_memo_history['memo_data'].apply(
            lambda x: ('CHUNK' in x) | ('WHISPER' in x) | ('chunk_' in x))].copy()
        
        memo_first = simple_user_memo_history.groupby('memo_type').first()
        memo_last = simple_user_memo_history.groupby('memo_type').last()

        first_incoming_memo = memo_first[['memo_data']]
        last_incoming_memo = memo_last[['memo_data']]
        first_incoming_memo = first_incoming_memo.loc[[i for i in first_incoming_memo.index if '__' in i]].copy()
        last_incoming_memo = last_incoming_memo.loc[[i for i in last_incoming_memo.index if '__' in i]].copy()
        
        full_user_history = pd.concat([first_incoming_memo, last_incoming_memo], axis=1)
        full_user_history.columns = ['initial_request', 'recent_status']
        
        memo_dexed = simple_user_memo_history.set_index('memo_type')
        tasks_to_consider = memo_dexed.loc[full_user_history.index]
        
        proposed_df = tasks_to_consider[tasks_to_consider['memo_data'].apply(
            lambda x: 'PROPOSED PF ___' in x)][['memo_data']]['memo_data']
        proposed_df = proposed_df.groupby('memo_type').last()
        full_user_history['proposed_task'] = proposed_df
        
        init_req = full_user_history['initial_request'].apply(lambda x: str(x).replace('REQUEST_POST_FIAT ___', 'User Requested:'))
        prop_req = full_user_history['proposed_task'].apply(lambda x: str(x).replace('PROPOSED PF ___', 'System Proposed:'))
        full_user_history['initial_task_detail'] = init_req + '__,' + prop_req

        full_user_history['first_date'] = memo_first['datetime']
        full_user_history['recent_date'] = memo_last['datetime']
        
        proposal_block = full_user_history[full_user_history['recent_status'].apply(
            lambda x: ('PROPOSED' in x) | ('ACCEPTANCE' in x))][['initial_task_detail', 'recent_status']]
        
        refusal_block = full_user_history[full_user_history['recent_status'].apply(
            lambda x: ('REFUS' in x))][['initial_task_detail', 'recent_status', 'recent_date']].head(n_task_context_history)
        
        verification_block = full_user_history[full_user_history['recent_status'].apply(
            lambda x: "VERIFICATION" in x)].copy()
        
        reward_block = full_user_history[full_user_history['recent_status'].apply(
            lambda x: 'REWARD' in x)].copy().head(n_task_context_history)
        
        proposal_string = proposal_block[['initial_task_detail', 'recent_status']].to_string()
        refusal_string = refusal_block[['initial_task_detail', 'recent_status', 'recent_date']].to_string()
        verification_string = verification_block[['initial_task_detail', 'recent_status', 'recent_date']].to_string()
        reward_string = reward_block[['initial_task_detail', 'recent_status', 'recent_date']].to_string()
        
        core_element__google_doc_text = ''
        if get_google_doc:
            try:
                google_url = self.generic_pft_utilities.get_latest_outgoing_context_doc_link(
                    account_address=account_address, 
                    memo_history=memo_history
                )
                core_element__google_doc_text = self.generic_pft_utilities.get_google_doc_text(google_url)
            except:
                print('failed retrieving user google doc')
                pass

        core_element__user_log_history = ''
        if get_historical_memos:
            try:
                core_element__user_log_history = self.generic_pft_utilities.get_recent_user_memos(
                    account_address=account_address,
                    num_messages=n_task_context_history
                )
            except:
                pass

        return f"""
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

The following is the user's full planning document that they have assembled
to inform task generation and planning
<<USER PLANNING DOC STARTS HERE>>
{core_element__google_doc_text}
<<USER PLANNING DOC ENDS HERE>>

The following is the users own comments regarding everything
<<< USER COMMENTS AND LOGS START HERE>>
{core_element__user_log_history}
<<< USER COMMENTS AND LOGS END HERE>>

***<<< ALL TASK GENERATION CONTEXT ENDS HERE >>>***
"""
