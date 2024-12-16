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
import json

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
        self.user_context = self.user_context_parser.get_full_user_context_string(
            account_address=account_address,
            memo_history=memo_history
        )

        # Initialize market data
        self.angron_map = self._generate_most_recent_angron_map()
        self.fulgrim_context = self._load_fulgrim_context()

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

    def generate_o1_question(self, account_address: str = '', user_chat_history: str = '', user_context: str = '') -> str:
        """Generate initial question for user"""
        prompt = o1_question_user_prompt.replace('__full_user_context__', user_context)
        prompt = prompt.replace('__user_chat_history__', user_chat_history)

        try:
            return self.openrouter.generate_simple_text_output(
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

    def generate_user_specific_question(
            self,
            account_address: str = '',
            user_chat_history: str = '',
            user_specific_offering: str = ''
    ) -> str:
        """Generate follow-up question based on user's specific offering"""
        prompt = user_specific_question_user_prompt.replace('__full_user_context__', self.user_context)
        prompt = prompt.replace('__user_chat_history__', user_chat_history)
        prompt = prompt.replace('__user_specific_offering__', user_specific_offering)

        try:
            return self.openrouter.generate_simple_text_output(
                model=self.model,
                messages=[
                    {"role": "system", "content": user_specific_question_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
        except Exception as e:
            logger.error(f"Error generating specific question: {str(e)}")
            return ""

    def generate_user_question_scoring_output(
            self,
            original_question: str,
            user_answer: str,
            account_address: str
    ) -> dict:
        """Score user's answer to generate appropriate reward"""
        prompt = corbanu_scoring_user_prompt.replace('__full_user_context__', self.user_context)
        prompt = prompt.replace('__user_conversation__', '')
        prompt = prompt.replace('__original_question__', original_question)
        prompt = prompt.replace('__user_answer__', user_answer)

        try:
            response = self.openrouter.generate_simple_text_output(
                model=self.model,
                messages=[
                    {"role": "system", "content": corbanu_scoring_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            return self._parse_scoring_output(response)
        except Exception as e:
            logger.error(f"Error in scoring: {str(e)}")
            return {"reward_value": 1, "reward_description": "Error in scoring process"}

    def _parse_scoring_output(self, scoring_string: str) -> dict:
        """Parse scoring output into structured format"""
        try:
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