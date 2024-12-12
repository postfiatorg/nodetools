from openai import OpenAI, AsyncOpenAI
import pandas as pd
import datetime
import uuid
import asyncio
import nest_asyncio
import json
import time
from asyncio import Semaphore
from nodetools.utilities.credentials import CredentialManager

class OpenRouterTool:
    """
    A wrapper for OpenRouter API that provides unified access to both OpenAI and Anthropic models
    """
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, max_concurrent_requests=10, requests_per_minute=30, http_referer="postfiat.org"):
        if not self.__class__._initialized:
            self.http_referer = http_referer
            cred_manager = CredentialManager()
            
            # Try both with and without variable___ prefix
            api_key = cred_manager.get_credential('variable___openrouter')
            if api_key is None:
                api_key = cred_manager.get_credential('openrouter')
            
            if api_key is None:
                raise ValueError("OpenRouter API key not found in credentials")
                
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key
            )
            self.async_client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key
            )
            self.semaphore = Semaphore(max_concurrent_requests)
            self.rate_limit = requests_per_minute
            self.request_times = []
            self.__class__._initialized = True

    def _prepare_headers(self):
        """Prepare headers required for OpenRouter API"""
        return {
            "HTTP-Referer": self.http_referer
        }

    def generate_simple_text_output(self, model, messages, max_tokens=None, temperature=None):
        """
        Generate text output using specified model
        
        Example:
        model="anthropic/claude-3.5-sonnet"
        messages=[{"role": "user", "content": "Hello!"}]
        """
        completion = self.client.chat.completions.create(
            extra_headers=self._prepare_headers(),
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return completion.choices[0].message.content

    def generate_dataframe(self, model, messages, max_tokens=None, temperature=None):
        """Generate a DataFrame containing the response and metadata"""
        completion = self.client.chat.completions.create(
            extra_headers=self._prepare_headers(),
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        output_map = {
            'text_response': completion.choices[0].message.content,
            'model': model,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'messages': json.dumps(messages),
            'date_run': datetime.datetime.now(),
            'job_uuid': str(uuid.uuid4()),
            'finish_reason': completion.choices[0].finish_reason,
            'usage': json.dumps(completion.usage.model_dump())
        }
        return pd.DataFrame(output_map, index=[0])

    async def rate_limited_request(self, job_name, api_args):
        """Execute a rate-limited API request"""
        async with self.semaphore:
            await self.wait_for_rate_limit()
            print(f"Task {job_name} start: {datetime.datetime.now().time()}")
            try:
                response = await self.async_client.chat.completions.create(
                    extra_headers=self._prepare_headers(),
                    **api_args
                )
                print(f"Task {job_name} end: {datetime.datetime.now().time()}")
                return job_name, response
            except Exception as e:
                print(f"Error for task {job_name}: {str(e)}")
                await asyncio.sleep(5)
                return await self.rate_limited_request(job_name, api_args)

    async def wait_for_rate_limit(self):
        """Implement rate limiting"""
        now = time.time()
        self.request_times = [t for t in self.request_times if now - t < 60]
        if len(self.request_times) >= self.rate_limit:
            sleep_time = 60 - (now - self.request_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        self.request_times.append(time.time())

    async def get_completions(self, arg_async_map):
        """Get completions asynchronously for given arguments map"""
        tasks = [self.rate_limited_request(job_name, args) for job_name, args in arg_async_map.items()]
        return await asyncio.gather(*tasks)

    def create_writable_df_for_async_chat_completion(self, arg_async_map: dict) -> pd.DataFrame:
        """Create DataFrame for async chat completion results"""
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        x1 = loop.run_until_complete(self.get_completions(arg_async_map=arg_async_map))
        dfarr = []
        for xobj in x1:
            internal_name = xobj[0]
            completion_object = xobj[1]
            raw_df = pd.DataFrame({
                'id': completion_object.id,
                'model': completion_object.model,
                'content': completion_object.choices[0].message.content,
                'finish_reason': completion_object.choices[0].finish_reason,
                'usage': json.dumps(completion_object.usage.model_dump()),
                'write_time': datetime.datetime.now(),
                'internal_name': internal_name
            }, index=[0])
            dfarr.append(raw_df)
        full_writable_df = pd.concat(dfarr)
        return full_writable_df

    def run_chat_completion_async_demo(self):
        """Run demo for async chat completion"""
        job_hashes = [f'job{i}sample__{uuid.uuid4()}' for i in range(1, 3)]
        arg_async_map = {
            job_hashes[0]: {
                "model": "anthropic/claude-3.5-sonnet",
                "messages": [{"role": "user", "content": "What's the future of AI?"}]
            },
            job_hashes[1]: {
                "model": "anthropic/claude-3.5-sonnet",
                "messages": [{"role": "user", "content": "Explain quantum computing"}]
            }
        }
        return self.create_writable_df_for_async_chat_completion(arg_async_map=arg_async_map)

    def example_text_completion(self):
        """
        Example of basic text completion
        Returns: Generated text response
        """
        response = self.generate_simple_text_output(
            model="anthropic/claude-3.5-sonnet",
            messages=[{
                "role": "user", 
                "content": "Write a short poem about artificial intelligence"
            }],
            temperature=0.7
        )
        return response

    def example_image_analysis(self, image_url):
        """
        Example of image analysis using multimodal capabilities
        Args:
            image_url: URL of the image to analyze
        Returns: Analysis of the image
        """
        response = self.generate_simple_text_output(
            model="anthropic/claude-3.5-sonnet",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this image in detail. What do you see?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }]
        )
        return response

    def example_structured_output(self):
        """
        Example of generating structured data
        Returns: DataFrame with structured analysis
        """
        messages = [{
            "role": "user",
            "content": "Analyze the following companies: Apple, Microsoft, Google. For each provide: 1) Main business area 2) Year founded 3) Key products. Format as JSON."
        }]
        
        response = self.generate_dataframe(
            model="anthropic/claude-3.5-sonnet",
            messages=messages,
            temperature=0.3
        )
        return response

    def example_multi_turn_conversation(self):
        """
        Example of multi-turn conversation
        Returns: List of responses from the conversation
        """
        conversation = []
        
        # First turn
        response1 = self.generate_simple_text_output(
            model="anthropic/claude-3.5-sonnet",
            messages=[{
                "role": "user",
                "content": "What are the three laws of robotics?"
            }]
        )
        conversation.append(("Question 1", response1))
        
        # Second turn - follow up
        response2 = self.generate_simple_text_output(
            model="anthropic/claude-3.5-sonnet",
            messages=[
                {"role": "user", "content": "What are the three laws of robotics?"},
                {"role": "assistant", "content": response1},
                {"role": "user", "content": "Who created these laws and in what work were they first introduced?"}
            ]
        )
        conversation.append(("Question 2", response2))
        
        return conversation

    def example_function_calling(self):
        """
        Example of function calling capability
        Returns: Structured function call result
        """
        messages = [{
            "role": "user",
            "content": "Extract the following information from this text: 'The meeting is scheduled for March 15th, 2024 at 2:30 PM with John Smith to discuss the Q1 budget.'"
        }]
        
        # Using Claude to extract structured information
        response = self.generate_simple_text_output(
            model="anthropic/claude-3.5-sonnet",
            messages=messages,
            temperature=0
        )
        
        # Parse the response into a structured format
        try:
            # Assuming the model returns a JSON-like structure
            structured_data = json.loads(response)
            return structured_data
        except:
            return {"error": "Could not parse response into structured format", "raw_response": response}
