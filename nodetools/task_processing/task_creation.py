from nodetools.protocols.generic_pft_utilities import GenericPFTUtilities
from nodetools.protocols.task_management import PostFiatTaskGenerationSystem
from nodetools.protocols.openrouter import OpenRouterTool
from nodetools.prompts.task_generation import (
    task_generation_one_shot_user_prompt,
    task_generation_one_shot_system_prompt
)
from nodetools.task_processing.user_context_parsing import UserTaskParser
import pandas as pd
import uuid
import re
from loguru import logger
import nodetools.configuration.constants as constants
from typing import Optional

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
    context = task_gen.user_task_parser.get_full_user_context_string(
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

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
            self,
            task_management_system: PostFiatTaskGenerationSystem,
            generic_pft_utilities: GenericPFTUtilities,
            openrouter_tool: OpenRouterTool
        ):
        """
        Initialize NewTaskGeneration with CredentialManager and create GenericPFTUtilities instance.
        
        Args:
            password (str, optional): Password for CredentialManager initialization. Required on first instance.
        """
        if not self.__class__._initialized:
            self.generic_pft_utilities = generic_pft_utilities
            self.user_context = UserContext()
            self.openrouter_tool = openrouter_tool
            self.user_task_parser = UserTaskParser(
                task_management_system=task_management_system,
                generic_pft_utilities=generic_pft_utilities
            )
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
                    "rUWuJJLLSH5TUdajVqsHx7M59Vj3P7giQV___task_id123": "a task please",
                    "rJzZLYK6JTg9NG1UA8g3D6fnJwd6vh3N4u___task_id234": "a planning task please"
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
        
        # Filter out invalid outputs
        output_df = output_df[
            output_df['content'].apply(lambda x: self.extract_final_output(x)) != 'NO OUTPUT'
        ].copy()
        
        # Set proposal strings, reward amounts, PFT amounts
        REWARD_AMOUNT = 900  # TODO: Make this dynamic
        output_df['pf_proposal_string'] = constants.TaskType.PROPOSAL.value + output_df['content'].apply(
            lambda x: self.extract_final_output(x)
        ) + ' .. ' + str(REWARD_AMOUNT)
        output_df['reward'] = REWARD_AMOUNT
        output_df['pft_to_send'] = 1  # TODO: Tie this to transaction_requirements.get_pft_requirement
        
        return output_df

    def create_task_key(self, account_id, task_id):
        """
        Create a combined key from account ID and task ID.
        
        Args:
            account_id (str): The account identifier
            task_id (str): The task identifier
            
        Returns:
            str: Combined key in format "{accountId}___{task_id}"
        """
        # NOTE: This needs 3 underscores to be able to correctly parse the task ID afterwards
        return f"{account_id}___{task_id}"

    def parse_task_key(self, task_key):
        """
        Parse a combined task key to extract account ID and task ID.
        
        Args:
            task_key (str): Combined key in format "{accountId}___{task_id}"
            
        Returns:
            tuple: (account_id, task_id)
        """
        parts = task_key.split("___")
        if len(parts) < 2:
            logger.error(f"Invalid task key format: {task_key}")
            return None, None
        return parts[0], parts[1]

    def run_batch_task_generation(
            self,
            task_map: dict,
            model: Optional[str] = constants.DEFAULT_OPENROUTER_MODEL,
            get_google_doc: bool = True,
            get_historical_memos: bool = True
        ):
        """
        Run batch task generation for multiple accounts asynchronously.
        
        Args:
            task_map (dict): Map of combined account/task IDs to task requests
                Keys should be in format "{accountId}___{task_id}" generated by self.create_task_key
                Example: {
                    "rUWuJJLLSH5TUdajVqsHx7M59Vj3P7giQV___task_id123": "a task please",
                    "rJzZLYK6JTg9NG1UA8g3D6fnJwd6vh3N4u___task_id234": "a planning task please"
                }
            model (str): Model identifier string (default: "anthropic/claude-3.5-sonnet:beta")
            get_google_doc (bool): Whether to fetch Google doc content
            get_historical_memos (bool): Whether to fetch historical memos
            
        Returns:
            pd.DataFrame: Results of the batch task generation with task IDs included
        """
        # Create arg_async_map
        arg_async_map = {}
        for task_key, task_request in task_map.items():
            # Extract account_id from the combined key
            account_id, _ = self.parse_task_key(task_key)
            
            # Generate unique job hash
            job_hash = f'{task_key}___{uuid.uuid4()}'
            
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
        results_df = self.openrouter_tool.create_writable_df_for_async_chat_completion(arg_async_map=arg_async_map)
        
        # Re-parse task keys and account addresses
        results_df['user_account'], results_df['task_id'] = zip(*results_df['internal_name'].apply(self.parse_task_key))
        
        return results_df

    def construct_task_generation_api_args(
            self,
            user_account_address: str,
            task_request: str,
            model: str,
            get_google_doc: bool = True,
            get_historical_memos: bool = True
        ):
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
        user_context = self.user_task_parser.get_full_user_context_string(
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
