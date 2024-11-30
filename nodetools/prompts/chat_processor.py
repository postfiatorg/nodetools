# THIS IS THE FOUNDATION HW r46SUhCzyGE4KwBnKQ6LmDmJcECCqdKy4q
from nodetools.utilities.generic_pft_utilities import GenericPFTUtilities
from nodetools.ai.openai import OpenAIRequestTool
from nodetools.chatbots.personas.odv import odv_system_prompt
from nodetools.utilities.encryption import MessageEncryption
import time
import nodetools.utilities.constants as constants
from nodetools.utilities.credentials import CredentialManager
from nodetools.utilities.exceptions import HandshakeRequiredException
import nodetools.utilities.constants as constants
from nodetools.utilities.encryption import MessageEncryption
from typing import Optional

class ChatProcessor:
    def __init__(self):
        self.network_config = constants.get_network_config()
        self.cred_manager = CredentialManager()
        self.generic_pft_utilities = GenericPFTUtilities(node_name=self.network_config.node_name)
        self.open_ai_request_tool = OpenAIRequestTool()

        # Cache for account handshakes
        self.account_handshakes = {}  # gets populated by _get_handshake_key, which gets called by _check_for_odv

    def _get_handshake_key(self, account_address: str) -> Optional[str]:
        """
        Internal method to get handshake key for an account, using cached value if available
        
        Args:
            account_address: The account to get handshake for
            
        Returns:
            Optional[str]: The received public key if found, None otherwise
        """
        remembrancer_address = self.network_config.remembrancer_address
        if account_address not in self.account_handshakes:
            _, received_key = self.generic_pft_utilities.get_handshake_for_address(remembrancer_address, account_address)
            self.account_handshakes[account_address] = received_key
        return self.account_handshakes[account_address]
    
    def _check_for_odv(self, row):
        """Internal method to check if a message contains ODV content, handling encryption if present
        
        Args:
            row: DataFrame row containing message data
            
        Returns:
            bool: True if message contains ODV content, False otherwise
        """
        message = row['cleaned_message']
        try:
            if MessageEncryption.is_encrypted(message):
                received_key = self._get_handshake_key(row['account'])

                if not received_key:
                    print(f"ChatProcessor._check_for_odv: No handshake found for {row['account']}")
                    return False
                
                shared_secret = MessageEncryption.get_shared_secret(
                    received_key=received_key, 
                    wallet_seed=self.cred_manager.get_credential(f"{self.network_config.remembrancer_name}__v1xrpsecret")
                )

                decrypted = MessageEncryption.process_encrypted_message(message, shared_secret)
                return 'ODV' in decrypted
            
            return 'ODV' in message
        
        except Exception as e:
            print(f"ChatProcessor._check_for_odv: Error checking for ODV in message from {row['account']}: {e}")
            return False

    def process_message(self, message: str, account_address: str) -> tuple[str, bool]:
        """
        Process a single message, handling encryption if present
        
        Returns:
            tuple: (processed_message, was_encrypted)
        """
        try:
            if MessageEncryption.is_encrypted(message):
                # print(f"ChatProcessor.process_message: Decrypting message from {account_address}")
                # Get sender's public key from handshake
                received_key = self._get_handshake_key(account_address)

                if not received_key:
                    # print(f"ChatProcessor.process_message: No handshake found for {account_address}")
                    raise HandshakeRequiredException(account_address)
                
                # Get shared secret
                shared_secret = MessageEncryption.get_shared_secret(
                    received_key=received_key, 
                    wallet_secret=self.cred_manager.get_credential(f"{self.network_config.remembrancer_name}__v1xrpsecret")
                )

                # Decrypt message
                return MessageEncryption.process_encrypted_message(message, shared_secret), True
            
            return message, False
        
        except HandshakeRequiredException as e:
            return f"[Handshake required] {e}", False

        except Exception as e:
            print(f"ChatProcessor.process_message: Error processing message from {account_address}: {e}")
            return f"[Message processing failed] {message}", False

    def process_chat_queue(self):
        """Process incoming chat messages and generate responses"""
        account_address = self.network_config.remembrancer_address

        # Get list of holders with sufficient balance
        full_holder_df = self.generic_pft_utilities.get_post_fiat_holder_df()

        full_holder_df['balance'] = full_holder_df['balance'].astype(float)
        all_top_wallet_holders = full_holder_df.sort_values('balance',ascending=True)
        real_users = all_top_wallet_holders[(all_top_wallet_holders['balance']*-1)>2_000].copy()
        top_accounts = list(real_users['account'].unique())

        # Get message queue
        full_message_queue = self.generic_pft_utilities.get_all_account_compressed_messages(account_address=account_address)

        # Filter for incoming messages from top accounts
        incoming_messages = full_message_queue[
            (full_message_queue['account'].apply(lambda x: x in top_accounts)) 
            & (full_message_queue['direction'].apply(lambda x: x == "INCOMING"))
        ]

        # Filter for ODV messages
        messages_to_work = incoming_messages[incoming_messages.apply(self._check_for_odv, axis=1)].copy()

        # Check for already sent responses
        responses = full_message_queue[full_message_queue['direction']=='OUTGOING'].copy()
        responses['memo_type'] = responses['memo_type'].apply(lambda x: x.replace('_response', ''))
        responses['sent'] = 1
        response_map = responses.groupby('memo_type').last()
        messages_to_work['already_sent']= messages_to_work['memo_type'].map(response_map['sent'])
        message_queue = messages_to_work[messages_to_work['already_sent']!=1].copy()

        if message_queue.empty:
            # print(f"ChatProcessor.process_chat_queue: No messages needing responses")
            return

        # Generate responses
        user_prompt_constructor = """You are to ingest the User's context below
        
        <<< USER FULL CONTEXT STARTS HERE>>>
        ___USER_CONTEXT_REPLACE___
        <<< USER FULL CONTEXT ENDS HERE>>>
        
        And consider what the user has asked below
        <<<USER QUERY STARTS HERE>>>
        ___USER_QUERY_REPLACE___
        <<<USER QUERY ENDS HERE>>>
        
        Output a response that is designed for the user to ACHIEVE MASSIVE RESULTS IN LINE WITH ODVS MANDATE
        WHILE AT THE SAME TIME SPECIFICALLY MAXIMIZING THE USERS AGENCY AND STATED OBJECTIVES 
        Keep your response to below 4 paragraphs.
        """
        message_queue['system_prompt']=odv_system_prompt
        accounts_to_map = list(message_queue['account'].unique())

        # Get context for each account
        account_context_map={}
        for xaccount in accounts_to_map:
            memo_history = self.generic_pft_utilities.get_account_memo_history(account_address=xaccount)
            account_context_map[xaccount] = self.generic_pft_utilities.get_full_user_context_string(xaccount, memo_history=memo_history)

        message_queue['user_context'] = message_queue['account'].map(account_context_map)
        message_queue.set_index('memo_type', inplace=True)
        messages_to_work = list(message_queue.index)

        # Process each message
        for mwork in messages_to_work:
            print(f"\nChatProcessor.process_chat_queue: Processing message {mwork}")
            message_slice = message_queue.loc[mwork]

            # Process message, handling encryption if present
            user_query, was_encrypted = self.process_message(message_slice['cleaned_message'], message_slice['account'])

            user_context_replace = message_slice['user_context']
            system_prompt = message_slice['system_prompt']
            destination_account = message_slice['account']
            
            user_prompt = user_prompt_constructor.replace(
                '___USER_CONTEXT_REPLACE___',
                user_context_replace
            ).replace(
                '___USER_QUERY_REPLACE___',
                user_query
            )

            print(f"ChatProcessor.process_chat_queue: Generating AI response to {destination_account}...")
            preview_req = self.open_ai_request_tool.o1_preview_simulated_request(system_prompt=system_prompt, user_prompt=user_prompt)
            
            op_response = """ODV SYSTEM: """ + preview_req.choices[0].message.content
            message_id = mwork+'_response'

            print(f"ChatProcessor.process_chat_queue: Sending response to {destination_account}")
            print(f"ChatProcessor.process_chat_queue: Response preview:\n{op_response[:100]}...")

            responses = self.generic_pft_utilities.send_memo(
                wallet_seed=self.cred_manager.get_credential(f"{self.network_config.remembrancer_name}__v1xrpsecret"),
                user_name='odv',
                destination=destination_account,
                memo=op_response,
                message_id=message_id,
                encrypt=was_encrypted
            )

            for response in responses:
                if not self.generic_pft_utilities.verify_transaction_response(response):
                    print(f"ChatProcessor.process_chat_queue: Failed to send response chunk. Response: {response}")
                    break
            else:
                print(f"ChatProcessor.process_chat_queue: All response chunks sent successfully to {destination_account}")
