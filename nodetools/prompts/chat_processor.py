# THIS IS THE FOUNDATION HW r46SUhCzyGE4KwBnKQ6LmDmJcECCqdKy4q
from nodetools.utilities.generic_pft_utilities import GenericPFTUtilities
from nodetools.ai.openai import OpenAIRequestTool
from nodetools.chatbots.personas.odv import odv_system_prompt
from nodetools.utilities.credentials import CredentialManager
from nodetools.task_processing.user_context_parsing import UserTaskParser
import nodetools.configuration.configuration as config
from loguru import logger
import pandas as pd

class ChatProcessor:
    def __init__(self):
        self.network_config = config.get_network_config()
        self.node_config = config.get_node_config()
        self.cred_manager = CredentialManager()
        self.generic_pft_utilities = GenericPFTUtilities()
        self.openai_request_tool = OpenAIRequestTool()
        self.user_task_parser = UserTaskParser()

    def _filter_unresponded_messages(self, messages_df: pd.DataFrame) -> pd.DataFrame:
        """Filter out messages that have already been responded to."""
        # Get outgoing messages (responses)
        response_memo_types = messages_df[
            (messages_df['direction'] == 'OUTGOING') &
            (messages_df['memo_type'].str.endswith('_response'))
        ]['memo_type'].tolist()
        # logger.debug(f"Found response memo_types:\n{response_memo_types}")
        
        # Create set of original memo_types that have responses
        original_memo_types = {memo_type.replace('_response', '') for memo_type in response_memo_types}
        # logger.debug(f"Original memo_types with responses:\n{original_memo_types}")
        
        # Filter messages
        filtered_df = messages_df[
            (messages_df['direction'] == 'INCOMING') &
            (~messages_df['memo_type'].isin(original_memo_types))
        ].copy()
        
        # logger.debug("=== Final filtered messages ===")
        # logger.debug(f"Remaining memo_types:\n{filtered_df['memo_type'].tolist()}")
        
        return filtered_df
    
    def _get_user_contexts(self, account_addresses: list[str]) -> dict[str, str]:
        """Get context strings for a list of accounts.
    
        Args:
            accounts: List of account addresses
            
        Returns:
            Dict mapping account addresses to their context strings
        """
        return {
            account: self.user_task_parser.get_full_user_context_string(
                account,
                memo_history=self.generic_pft_utilities.get_account_memo_history(account)
            )
            for account in account_addresses
        }
    
    @staticmethod
    def _construct_user_prompt(user_context: str, user_query: str) -> str:
        """Construct the prompt for the AI model."""
        return f"""You are to ingest the User's context below
    
        <<< USER FULL CONTEXT STARTS HERE>>>
        {user_context}
        <<< USER FULL CONTEXT ENDS HERE>>>
        
        And consider what the user has asked below
        <<<USER QUERY STARTS HERE>>>
        {user_query}
        <<<USER QUERY ENDS HERE>>>
        
        Output a response that is designed for the user to ACHIEVE MASSIVE RESULTS IN LINE WITH ODVS MANDATE
        WHILE AT THE SAME TIME SPECIFICALLY MAXIMIZING THE USERS AGENCY AND STATED OBJECTIVES 
        Keep your response to below 4 paragraphs."""

    def process_chat_queue(self):
        """Process incoming chat messages and generate responses"""
        full_holder_df = self.generic_pft_utilities.get_post_fiat_holder_df()

        # Filter for holders with sufficient balance (over 2000 PFT), unless on TESTNET and DISABLE_PFT_REQUIREMENTS is true
        if not (config.RuntimeConfig.USE_TESTNET and config.RuntimeConfig.DISABLE_PFT_REQUIREMENTS):
            real_users = full_holder_df[
                (full_holder_df['balance'].astype(float) * -1) > 2_000
            ]
        else:
            real_users = full_holder_df

        top_accounts = list(real_users['account'].unique())

        # Retrieve messages
        full_message_queue = self.generic_pft_utilities.get_all_account_compressed_messages(
            account_address=self.node_config.remembrancer_address,
            channel_private_key=self.cred_manager.get_credential(f"{self.node_config.remembrancer_name}__v1xrpsecret")
        )

        # Filter for messages involving top accounts
        messages_to_work = full_message_queue[
            ((full_message_queue['account'].isin(top_accounts)) | 
            (full_message_queue['destination'].isin(top_accounts))) &
            # Include messages that either:
            # 1. Contain 'ODV' in their processed content, or
            # 2. Are response messages (even if processing failed)
            ((full_message_queue['processed_message'].apply(lambda x: 'ODV' in x)) |
            (full_message_queue['memo_type'].str.endswith('_response')))
        ].copy()

        # Check for already sent responses
        message_queue = self._filter_unresponded_messages(messages_to_work)
        if message_queue.empty:
            return
        
        logger.debug(f"ChatProcessor.process_chat_queue: Found {len(message_queue)} messages to respond to")

        # Get context for each account
        message_queue['user_context'] = message_queue['account'].map(
            self._get_user_contexts(list(message_queue['account'].unique()))
        )
        message_queue['system_prompt'] = odv_system_prompt
        message_queue.set_index('memo_type', inplace=True)

        # Process each message
        for msg_id in message_queue.index:
            logger.debug(f"\nChatProcessor.process_chat_queue: Processing message {msg_id}")
            message = message_queue.loc[msg_id]

            # Construct prompt
            user_prompt = self._construct_user_prompt(
                user_context=message['user_context'],
                user_query=message['processed_message']
            )

            # Generate AI response
            logger.debug(f"ChatProcessor.process_chat_queue: Generating AI response to {message['account']}...")

            preview_req = self.openai_request_tool.o1_preview_simulated_request(
                system_prompt=message['system_prompt'],
                user_prompt=user_prompt
            )
            
            op_response = """ODV SYSTEM: """ + preview_req.choices[0].message.content
            message_id = msg_id + '_response'

            logger.debug(f"ChatProcessor.process_chat_queue: Sending response to {message['account']}")
            logger.debug(f"ChatProcessor.process_chat_queue: Response preview:\n{op_response[:100]}...")

            responses = self.generic_pft_utilities.send_memo(
                wallet_seed_or_wallet=self.cred_manager.get_credential(f"{self.node_config.remembrancer_name}__v1xrpsecret"),
                username='odv',
                destination=message['account'],
                memo=op_response,
                message_id=message_id,
                chunk=True,
                compress=True,
                encrypt=True
            )

            if not self.generic_pft_utilities.verify_transaction_response(responses):
                logger.error(f"ChatProcessor.process_chat_queue: Failed to send response chunk. Response: {responses}")
                break
            else:
                logger.debug(f"ChatProcessor.process_chat_queue: All response chunks sent successfully to {message['account']}")
