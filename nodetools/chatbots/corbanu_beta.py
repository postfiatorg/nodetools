from nodetools.protocols.openrouter import OpenRouterTool
from nodetools.protocols.generic_pft_utilities import GenericPFTUtilities
from nodetools.protocols.user_context_parsing import UserTaskParser
from nodetools.utilities.db_manager import DBConnectionManager
from nodetools.chatbots.personas.corbanu import (
    conversation_classification_system_prompt,
    conversation_classification_user_prompt,
    o1_question_system_prompt,
    o1_question_user_prompt,
    user_specific_question_system_prompt,
    user_specific_question_user_prompt,
    angron_system_prompt,
    angron_user_prompt,
    fulgrim_system_prompt,
    fulgrim_user_prompt,
    router_system_prompt,
    router_user_prompt,
    corbanu_scoring_system_prompt,
    corbanu_scoring_user_prompt
)
from loguru import logger
import pandas as pd
import re
import asyncio
import traceback
from decimal import Decimal
from datetime import datetime, timedelta, timezone
import json
from typing import Optional
import nodetools.configuration.constants as constants

class CorbanuChatBot:
    def __init__(
            self,
            account_address: str,
            openrouter: OpenRouterTool,
            user_context_parser: UserTaskParser,
            pft_utils: GenericPFTUtilities,
            db_connection_manager: DBConnectionManager = None
    ):
        # Initialize tools
        self.openrouter = openrouter
        self.pft_utils = pft_utils
        self.user_context_parser = user_context_parser
        self.db_connection_manager = db_connection_manager or DBConnectionManager()

        # Initialize model
        self.model = "openai/o1-preview"
        
        # Get user context once
        memo_history = self.pft_utils.get_account_memo_history(account_address=account_address)
        self.user_context = self.get_corbanu_context(
            account_address=account_address,
            memo_history=memo_history
        )
        self.GOOGLE_DOC_TEXT_CHAR_LIMIT = 10000

        # Initialize market data
        self.angron_map = self._generate_most_recent_angron_map()
        self.fulgrim_context = self._load_fulgrim_context()

        self.MAX_PER_OFFERING_REWARD_VALUE = 3000
        self.MAX_DAILY_REWARD_VALUE = 9000

    def _generate_most_recent_angron_map(self):
        """Get most recent SPM signal data"""
        try:
            dbconnx = self.db_connection_manager.spawn_sqlalchemy_db_connection_for_user(username='sigildb')
            most_recent_spm_signal = pd.read_sql('spm_signals', dbconnx).tail(1)
            xdf = most_recent_spm_signal.transpose()
            angron_map = xdf[xdf.columns[0]]
            dbconnx.dispose()
            return angron_map
        except Exception as e:
            logger.error(f"Error getting Angron map: {str(e)}")
            return {"full_context_block": "", "final_output": ""}

    def _load_fulgrim_context(self):
        """Get most recent Fulgrim signal data"""
        try:
            dbconnx = self.db_connection_manager.spawn_sqlalchemy_db_connection_for_user(username='sigildb')
            most_recent_spm_signal = pd.read_sql('fulgrim__signal_write', dbconnx).tail(1)
            fulgrim_context = list(most_recent_spm_signal['content'])[0]
            dbconnx.dispose()
            return fulgrim_context
        except Exception as e:
            logger.error(f"Error loading Fulgrim context: {str(e)}")
            return ""

    def get_response(self, user_message: str) -> str:
        """Synchronous version of get_response"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(self.get_response_async(user_message))
            loop.close()
            return response
        except Exception as e:
            logger.error(f"Error in get_response: {str(e)}")
            raise

    async def get_response_async(self, user_message: str) -> str:
        """Process user message and return appropriate response"""
        try:
            # Route to appropriate SPM based on content
            spm_choice = await self._determine_spm_choice(user_message)
            
            if spm_choice == "FULGRIM":
                response = await self._get_fulgrim_response(user_message)
            else:  # Default to ANGRON
                response = await self._get_angron_response(user_message)
            
            return f"Corbanu Summons Synthetic Portfolio Manager {spm_choice} To Assist:\n{response}"
            
        except Exception as e:
            logger.error(f"Error in get_response_async: {str(e)}")
            raise

    async def _determine_spm_choice(self, message: str) -> str:
        """Determine which SPM should handle the message"""
        try:
            prompt = router_user_prompt.replace('__existing_conversation_string__', '')
            prompt = prompt.replace('__recent_conversation_string__', message)

            response = await self.openrouter.generate_simple_text_output_async(
                model=self.model,
                messages=[
                    {"role": "system", "content": router_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            return "FULGRIM" if "FULGRIM" in response.upper() else "ANGRON"
        except:
            return "ANGRON"  # Default to ANGRON on error

    async def _get_angron_response(self, message: str) -> str:
        """Get response from Angron SPM"""
        prompt = angron_user_prompt.replace('__full_context_block__', self.angron_map['full_context_block'])
        prompt = prompt.replace('__final_output__', self.angron_map['final_output'])
        prompt = prompt.replace('__conversation_string__', '')
        prompt = prompt.replace('__most_recent_request_string__', message)

        return await self.openrouter.generate_simple_text_output_async(
            model=self.model,
            messages=[
                {"role": "system", "content": angron_system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

    async def _get_fulgrim_response(self, message: str) -> str:
        """Get response from Fulgrim SPM"""
        prompt = fulgrim_user_prompt.replace('__fulgrim_context__', self.fulgrim_context)
        prompt = prompt.replace('__conversation_string__', '')
        prompt = prompt.replace('__most_recent_request_string__', message)

        return await self.openrouter.generate_simple_text_output_async(
            model=self.model,
            messages=[
                {"role": "system", "content": fulgrim_system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

    def make_convo_classification_df(self, last_5_messages: str, most_recent_message: str) -> pd.DataFrame:
        """Classify conversation for information exchange"""
        prompt = conversation_classification_user_prompt.replace('__last_5_messages__', last_5_messages)
        prompt = prompt.replace('__most_recent_message__', most_recent_message)

        try:
            response = self.openrouter.generate_simple_text_output(
                model=self.model,
                messages=[
                    {"role": "system", "content": conversation_classification_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            return pd.DataFrame([response], columns=['choices__message__content'])
        except Exception as e:
            logger.error(f"Error in conversation classification: {str(e)}")
            return pd.DataFrame()

    async def generate_question(self) -> str:
        """Generate initial question for user"""

        prompt = o1_question_user_prompt.replace('__full_user_context__', self.user_context)
        # prompt = prompt.replace('__user_chat_history__', user_chat_history)

        logger.debug(f"CorbanuChatBot.generate_question: prompt: {prompt}")

        try:
            return await self.openrouter.generate_simple_text_output_async(
                model="anthropic/claude-3.5-sonnet:beta",
                messages=[
                    {"role": "system", "content": o1_question_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
        except Exception as e:
            logger.error(f"Error generating question: {str(e)}")
            return ""

    # # NOTE: Not used anywhere
    # def generate_user_specific_question(
    #         self,
    #         account_address: str = '',
    #         user_chat_history: str = '',
    #         user_specific_offering: str = ''
    # ) -> str:
    #     """Generate follow-up question based on user's specific offering"""
    #     prompt = user_specific_question_user_prompt.replace('__full_user_context__', self.user_context)
    #     prompt = prompt.replace('__user_chat_history__', user_chat_history)
    #     prompt = prompt.replace('__user_specific_offering__', user_specific_offering)

    #     try:
    #         return self.openrouter.generate_simple_text_output(
    #             model=self.model,
    #             messages=[
    #                 {"role": "system", "content": user_specific_question_system_prompt},
    #                 {"role": "user", "content": prompt}
    #             ],
    #             temperature=0
    #         )
    #     except Exception as e:
    #         logger.error(f"Error generating specific question: {str(e)}")
    #         return ""

    async def generate_user_question_scoring_output(
            self,
            original_question: str,
            user_answer: str,
            account_address: str
    ) -> dict:
        """Score user's answer to generate appropriate reward"""
        logger.debug(f"MyClient.corbanu_reply: Generating scoring output for {account_address}.")
        prompt = corbanu_scoring_user_prompt.replace('__full_user_context__', self.user_context)
        prompt = prompt.replace('__user_conversation__', '')
        prompt = prompt.replace('__original_question__', original_question)
        prompt = prompt.replace('__user_answer__', user_answer)

        try:
            response = await self.openrouter.generate_simple_text_output_async(
                model=self.model,
                messages=[
                    {"role": "system", "content": corbanu_scoring_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            logger.debug(f"MyClient.corbanu_reply: Scoring output for {account_address}: {response}")
            return self._parse_scoring_output(response)

        except Exception as e:
            logger.error(f"Error in scoring: {str(e)}")
            logger.error(traceback.format_exc())
            return {"reward_value": 1, "reward_description": "Error in scoring process"}

    def _parse_scoring_output(self, scoring_string: str) -> dict:
        """Parse scoring output into structured format"""
        try:
            logger.debug(f"MyClient.corbanu_reply: Parsing scoring output: {scoring_string}")
            value_match = re.search(r'\|\s*REWARD VALUE\s*\|\s*(\d+)\s*\|', scoring_string)
            desc_match = re.search(r'\|\s*REWARD DESCRIPTION\s*\|\s*([^|]+)\|', scoring_string)
            
            if not value_match or not desc_match:
                raise ValueError("Invalid scoring format")
                
            return {
                'reward_value': int(value_match.group(1)),
                'reward_description': desc_match.group(1).strip()
            }
        except Exception as e:
            logger.error(f"Error parsing score: {str(e)}")
            return {"reward_value": 1, "reward_description": "Error parsing score"}


    async def summarize_text(self, text: str, max_length: int = 900) -> str:
        """
        Summarize the given text into approximately `max_length` characters.
        
        Args:
            text (str): The text to summarize.
            max_length (int): The desired maximum length in characters of the summary.
        
        Returns:
            str: A summary of the given text.
        """
        # Construct a prompt that instructs the model to summarize
        # We explicitly mention the character limit to guide the model.
        prompt = (
            f"Your job is to summarize the following text into about {max_length} characters, focusing on the key points:\n\n"
            f"---\n{text}\n---\n\n"
            f"Please provide a concise summary of around {max_length} characters without exceeding that limit. If there is Q and A briefly summarize both the Q and the A"
        )

        try:
            summary = await self.openrouter.generate_simple_text_output_async(
                model="anthropic/claude-3.5-sonnet:beta",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specializing in summarization."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            # Optional: In case the model returns something longer than max_length, 
            # we can just truncate it. 
            # But often the model will comply. 
            if len(summary) > max_length:
                summary = summary[:max_length].rstrip()

            return summary

        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
            # Fallback to a simple truncation if something goes wrong
            return text[:max_length]
        
    def check_daily_reward_limit(self, account_address: str) -> Decimal:
        """
        Check how much reward capacity remains for the user within the daily limit.
        
        Args:
            user_wallet_address: The user's wallet address
            
        Returns:
            Decimal: Remaining reward capacity (0 if limit exceeded)
        """
        try:            
            # Get last 24 hours of memos
            memo_history = self.pft_utils.get_account_memo_history(account_address=account_address)

            # Add PFT amount column to the full DataFrame first
            def try_get_pft_absolute_amount(x):
                try:
                    # Parse JSON if it's a string
                    if isinstance(x, str):
                        x = json.loads(x)
                    return int(x['DeliverMax']['value'])
                except:
                    return 0
                
            memo_history['amount_pft'] = memo_history['tx_json'].apply(try_get_pft_absolute_amount)

            # Calculate 24 hours ago timestamp in UTC
            utc_now = datetime.now(timezone.utc)
            cutoff_time = utc_now - timedelta(hours=24)

            # Ensure datetime column is timezone-aware before comparison
            memo_history['datetime'] = pd.to_datetime(memo_history['datetime']).dt.tz_localize('UTC')

            # Filter for Corbanu rewards sent to this user in the last 24 hours
            corbanu_rewards = memo_history[
                (memo_history['destination'] == account_address) & 
                (memo_history['memo_data'].str.contains('Corbanu Reward', na=False)) &
                (memo_history['datetime'] >= cutoff_time)
            ]

            # Sum rewards in last 24 hours
            total_recent_rewards = Decimal(int(corbanu_rewards['amount_pft'].sum()))

            remaining_limit = max(Decimal(self.MAX_DAILY_REWARD_VALUE) - total_recent_rewards, Decimal(0))
                    
            logger.debug(f"Corbanu rewards in last 24h for {account_address}: {total_recent_rewards} PFT")
            logger.debug(f"Remaining daily reward limit: {remaining_limit} PFT")
            
            return remaining_limit
        
        except Exception as e:
            logger.error(f"Error checking daily reward limit: {str(e)}")
            return Decimal(0)  # Return 0 on error to prevent rewards
        
    def get_corbanu_context(
            self,
            account_address: str,
            memo_history: Optional[pd.DataFrame] = None,
            n_memos_in_context: int = constants.MAX_CHUNK_MESSAGES_IN_CONTEXT,
    ) -> str:
        """Get Corbanu context for a user"""
        if memo_history is None:
            memo_history = self.pft_utils.get_account_memo_history(account_address=account_address)

        try:
            google_url = self.pft_utils.get_latest_outgoing_context_doc_link(
                account_address=account_address, 
                memo_history=memo_history
            )
            # Retrieve google doc text and limit to 10000 characters
            core_element__google_doc_text = self.pft_utils.get_google_doc_text(google_url)[:self.GOOGLE_DOC_TEXT_CHAR_LIMIT]
        except Exception as e:
            logger.error(f"Failed retrieving user google doc: {e}")
            logger.error(traceback.format_exc())
            core_element__google_doc_text = 'Error retrieving google doc'

        try:
            core_element__user_log_history = self.get_recent_corbanu_interactions(
                account_address=account_address,
                num_messages=n_memos_in_context
            )
        except Exception as e:
            logger.error(f"Failed retrieving user memo history: {e}")
            logger.error(traceback.format_exc())
            core_element__user_log_history = 'Error retrieving user memo history'

        corbanu_context_string = f"""
        ***<<< ALL CORBANU QUESTION GENERATION CONTEXT STARTS HERE >>>***
        The following is the user's full planning document that they have assembled
        to inform Post Fiat Task Management System for task generation and planning
        <<USER PLANNING DOC STARTS HERE>>
        {core_element__google_doc_text}
        <<USER PLANNING DOC ENDS HERE>>
        The following is the users last {n_memos_in_context} interactions with Corbanu
        <<< USER CORBANU INTERACTIONS START HERE>>
        {core_element__user_log_history}
        <<< USER CORBANU INTERACTIONS END HERE>>
        ***<<< ALL CORBANU QUESTION GENERATION CONTEXT ENDS HERE >>>***
        """

        return corbanu_context_string

    def get_recent_corbanu_interactions(
            self,
            account_address: str,
            num_messages: int = 10
    ) -> str:
        """Get recent Corbanu interactions for a user"""
        try:
            # Get all messages and select relevant columns
            messages_df = self.pft_utils.get_all_account_compressed_messages_for_remembrancer(
                account_address=account_address,
            )[['processed_message', 'datetime', 'memo_format']]

            # Filter for Corbanu messages
            corbanu_messages = messages_df[messages_df['memo_format'] == 'Corbanu']

            if corbanu_messages.empty:
                return json.dumps({})
            
            # Get most recent messages, sort by time, and convert to JSON
            recent_messages = (corbanu_messages
                .tail(num_messages)
                .sort_values('datetime')
                .set_index('datetime')['processed_message']
                .to_json()
            )

            return recent_messages

        except Exception as e:
            logger.error(f"CorbanuChatBot.get_recent_corbanu_interactions: Failed to get recent Corbanu interactions for account {account_address}: {e}")
            return json.dumps({})
