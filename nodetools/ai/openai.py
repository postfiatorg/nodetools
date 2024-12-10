import pandas as pd
import datetime
from openai import OpenAI, AsyncOpenAI
import json
import asyncio
import nest_asyncio
from nodetools.utilities.db_manager import DBConnectionManager
from nodetools.utilities.credentials import CredentialManager
import uuid
import nodetools.configuration.constants as constants
import nodetools.configuration.configuration as config
from loguru import logger
import httpx

class OpenAIRequestTool:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self.__class__._initialized:
            cred_manager = CredentialManager()

            # Check for OpenRouter credentials first
            openrouter_key = cred_manager.get_credential('openrouter')
            self.using_openrouter = openrouter_key is not None
            if openrouter_key:
                base_url = constants.OPENROUTER_BASE_URL
                self.api_key = openrouter_key
            else:
                base_url = None  # Use default OpenAI URL
                self.api_key = cred_manager.get_credential('openai')
            
            self.client = OpenAI(base_url=base_url, api_key=self.api_key)
            self.async_client = AsyncOpenAI(base_url=base_url, api_key=self.api_key)
            self.db_connection_manager = DBConnectionManager()
            self.__class__._initialized = True

    def _prepare_api_args(self, api_args: dict) -> dict:
        """Transform API arguments based on whether using OpenRouter or OpenAI."""
        if not self.using_openrouter:
            return api_args
            
        modified_args = api_args.copy()

        # For OpenRouter, just ensure the model name is in the correct format
        if 'model' in modified_args:
            if config.RuntimeConfig.USE_TESTNET and config.RuntimeConfig.USE_OPENROUTER_AUTOROUTER:
                modified_args['model'] = "openrouter/auto"
            else:
                model_name = modified_args['model']
                if not '/' in model_name:
                    modified_args['model'] = f"openai/{model_name}"

        return modified_args

    def run_chat_completion_demo(self):
        '''Demo run of chat completion with gpt-4-1106-preview model'''
        api_args = {
            "model": 'gpt-4o',
            "messages": [
                {"role": "system", "content": 'you are a helpful AI assistant'},
                {"role": "user", "content": 'explain how to cook a trout'}
            ]
        }
        prepared_args = self._prepare_api_args(api_args=api_args)
        output = self.client.chat.completions.create(**prepared_args)
        return output

    def run_chat_completion_sync(self, api_args):
        '''Run synchronous chat completion with given API arguments'''
        prepared_args = self._prepare_api_args(api_args=api_args)
        logger.debug(f"OpenAIRequestTool.run_chat_completion_sync: Running chat completion with API arguments: {prepared_args}")
        output = self.client.chat.completions.create(**prepared_args)
        return output

    def create_writable_df_for_chat_completion(self, api_args):
        '''Create a DataFrame from chat completion response'''
        opx = self.run_chat_completion_sync(api_args=api_args)
        raw_df = pd.DataFrame(opx.model_dump(), index=[0]).copy()
        raw_df['choices__finish_reason'] = raw_df['choices'].apply(lambda x: x.get('finish_reason', None))
        raw_df['choices__index'] = raw_df['choices'].apply(lambda x: x.get('index', None))
        raw_df['choices__message__content'] = raw_df['choices'].apply(lambda x: x['message'].get('content', None))
        raw_df['choices__message__role'] = raw_df['choices'].apply(lambda x: x['message'].get('role', None))
        raw_df['choices__message__function_call'] = raw_df['choices'].apply(lambda x: x['message'].get('function_call', None))
        raw_df['choices__message__tool_calls'] = raw_df['choices'].apply(lambda x: x['message'].get('tool_calls', None))
        raw_df['choices__log_probs'] = raw_df['choices'].apply(lambda x: x.get('logprobs', None))
        raw_df['choices__json'] = raw_df['choices'].apply(lambda x: json.dumps(x))
        raw_df['write_time'] = datetime.datetime.now()
        return raw_df

    def query_chat_completion_and_write_to_db(self, api_args):
        '''Query chat completion and write result to database'''
        writable_df = self.create_writable_df_for_chat_completion(api_args=api_args)
        writable_df = writable_df[[i for i in writable_df.columns if 'choices' != i]].copy()
        dbconnx = self.db_connection_manager.spawn_sqlalchemy_db_connection_for_user(username='collective')
        writable_df.to_sql('openai_chat_completions', dbconnx, if_exists='append', index=False)
        dbconnx.dispose()
        return writable_df

    def output_all_openai_chat_completions(self):
        '''Output all chat completions from the database'''
        dbconnx = self.db_connection_manager.spawn_sqlalchemy_db_connection_for_user(username='collective')
        all_completions = pd.read_sql('openai_chat_completions', dbconnx)
        return all_completions

    async def get_completions(self, arg_async_map):
        '''Get completions asynchronously for given arguments map'''
        async def task_with_debug(job_name, api_args):
            logger.debug(f"OpenAIRequestTool.get_completions: Task {job_name} starting")
            try:
                prepared_args = self._prepare_api_args(api_args=api_args)

                # TODO: This is a hack to get around OpenAI's async client not supporting OpenRouter
                if prepared_args.get('route') == 'fallback':
                    # Use httpx or aiohttp for OpenRouter
                    async with httpx.AsyncClient() as client:
                        headers = {
                            'Authorization': f"Bearer {self.api_key}",
                            'Content-Type': 'application/json'
                        }
                        response = await client.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            json=prepared_args,
                            headers=headers
                        )
                        response_data = response.json()
                        logger.debug(f"OpenRouter response: {response_data}")
                        return job_name, response_data
                else:
                    # Use OpenAI's async clients for direct OpenAI API calls
                    response = await self.async_client.chat.completions.create(**prepared_args)
                    return job_name, response

            except Exception as e:
                logger.error(f"OpenAIRequestTool.get_completions: Task {job_name} failed: {e}")
                return job_name, None

        tasks = [asyncio.create_task(task_with_debug(job_name, args)) 
                 for job_name, args in arg_async_map.items()]
        return await asyncio.gather(*tasks)

    def create_writable_df_for_async_chat_completion(self, arg_async_map):
        '''Create DataFrame for async chat completion results'''
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        x1 = loop.run_until_complete(self.get_completions(arg_async_map=arg_async_map))
        dfarr = []
        for xobj in x1:
            internal_name = xobj[0]
            completion_object = xobj[1]

            # Handle both OpenAI responses (which have model_dump()) and OpenRouter responses (which are dictionaries)
            if hasattr(completion_object, 'model_dump'):
                raw_df = pd.DataFrame(completion_object.model_dump(), index=[0]).copy()
            else:
                raw_df = pd.DataFrame(completion_object, index=[0]).copy()

            # Safely extract fields with defaults for missing data
            raw_df['choices__finish_reason'] = raw_df['choices'].apply(lambda x: x.get('finish_reason', None))
            raw_df['choices__index'] = raw_df['choices'].apply(lambda x: x.get('index', None))
            raw_df['choices__message__content'] = raw_df['choices'].apply(lambda x: x['message'].get('content', None))
            raw_df['choices__message__role'] = raw_df['choices'].apply(lambda x: x['message'].get('role', None))
            raw_df['choices__message__function_call'] = raw_df['choices'].apply(lambda x: x['message'].get('function_call', None))
            raw_df['choices__message__tool_calls'] = raw_df['choices'].apply(lambda x: x['message'].get('tool_calls', None))
            raw_df['choices__log_probs'] = raw_df['choices'].apply(lambda x: x.get('logprobs', None))
            raw_df['choices__json'] = raw_df['choices'].apply(lambda x: json.dumps(x))
            raw_df['write_time'] = datetime.datetime.now()
            raw_df['internal_name'] = internal_name
            dfarr.append(raw_df)
        full_writable_df = pd.concat(dfarr)
        return full_writable_df

    def generate_job_hash(self):
        '''Generate unique job hash'''
        return str(uuid.uuid4())

    def run_chat_completion_async_demo(self):
        '''Run demo for async chat completion'''
        job_hashes = [f'job{i}sample__{self.generate_job_hash()}' for i in range(1, 6)]
        arg_async_map = {
            job_hashes[0]: {
                "model": 'gpt-4-1106-preview',
                "messages": [
                    {"role": "system", "content": 'you are the worlds most smooth and funny liar'},
                    {"role": "user", "content": 'make an elaborate excuse for why you are late to work'}
                ]
            },
            job_hashes[1]: {
                "model": 'gpt-4-1106-preview',
                "messages": [
                    {"role": "system", "content": 'you are the worlds most crafty and sneaky liar'},
                    {"role": "user", "content": 'make an elaborate excuse for why you are late to work'}
                ]
            },
            job_hashes[2]: {
                "model": 'gpt-4-1106-preview',
                "messages": [
                    {"role": "system", "content": 'you are the worlds most persuasive and charming person'},
                    {"role": "user", "content": 'explain to your spouse why adultery is a good thing'}
                ]
            },
            job_hashes[3]: {
                "model": 'gpt-4-1106-preview',
                "messages": [
                    {"role": "system", "content": 'you are the worlds most persuasive and charming person'},
                    {"role": "user", "content": 'convince me to believe in god'}
                ]
            },
            job_hashes[4]: {
                "model": 'gpt-4-1106-preview',
                "messages": [
                    {"role": "system", "content": 'you are the worlds most persuasive and charming person'},
                    {"role": "user", "content": 'convince me to believe in satanism'}
                ]
            },
        }
        async_write_df = self.create_writable_df_for_async_chat_completion(arg_async_map=arg_async_map)
        return async_write_df

    def query_chat_completion_async_and_write_to_db(self, arg_async_map):
        '''Query chat completion asynchronously and write result to database'''
        async_write_df = self.create_writable_df_for_async_chat_completion(arg_async_map=arg_async_map)
        dbconnx = self.db_connection_manager.spawn_sqlalchemy_db_connection_for_user(username='collective')
        async_write_df = async_write_df[[i for i in async_write_df.columns if 'choices' != i]].copy()
        async_write_df.to_sql('openai_chat_completions', dbconnx, if_exists='append', index=False)
        dbconnx.dispose()
        return async_write_df

    def o1_preview_simulated_request(self,system_prompt,user_prompt):

            content_replacement = f"""YOU ADHERE TO THE FOLLOWING INSTRUCTIONS WITHOUT BREAKING ROLE
            <INSTRUCTIONS FOR BEHAVIOR START HERE>
            {system_prompt}
            <INSTRUCTIONS FOR BEHAVIOR END HERE>

            NOW THAT YOU HAVE PROCESSED THE INTRUCTIONS. DO NOT FURTHER MENTION THEM IN YOUR 
            RESPONSE. RESPOND TO THE FOLLOWING REQUEST:
            <USER REQUEST STARTS HERE>
            {user_prompt}
            <USER REQUEST ENDS HERE>
            """ 
            response = self.client.chat.completions.create(
                model="o1-preview",
                messages=[
                    {
                        "role": "user", 
                        "content": content_replacement
                    }
                ]
            )
            return response 


    def o1_mini_simulated_request(self,system_prompt,user_prompt):

        content_replacement = f"""YOU ADHERE TO THE FOLLOWING INSTRUCTIONS WITHOUT BREAKING ROLE
        <INSTRUCTIONS FOR BEHAVIOR START HERE>
        {system_prompt}
        <INSTRUCTIONS FOR BEHAVIOR END HERE>

        NOW THAT YOU HAVE PROCESSED THE INTRUCTIONS. DO NOT FURTHER MENTION THEM IN YOUR 
        RESPONSE. RESPOND TO THE FOLLOWING REQUEST:
        <USER REQUEST STARTS HERE>
        {user_prompt}
        <USER REQUEST ENDS HERE>
        """ 
        response = self.client.chat.completions.create(
            model="o1-mini",
            messages=[
                {
                    "role": "user", 
                    "content": content_replacement
                }
            ]
        )
        return response 