import anthropic
import pandas as pd
import datetime
import uuid
import asyncio
import nest_asyncio
from anthropic import AsyncAnthropic
import time
from asyncio import Semaphore
from nodetools.utilities.credentials import CredentialManager
import nodetools.utilities.constants as constants

class AnthropicTool:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, max_concurrent_requests=2, requests_per_minute=30):
        if not self.__class__._initialized:
            cred_manager = CredentialManager()
            self.client = anthropic.Anthropic(api_key=cred_manager.get_credential('anthropic'))
            self.async_client = AsyncAnthropic(api_key=cred_manager.get_credential('anthropic'))
            self.default_model = constants.DEFAULT_ANTHROPIC_MODEL
            self.semaphore = Semaphore(max_concurrent_requests)
            self.rate_limit = requests_per_minute
            self.request_times = []
            self.__class__._initialized = True

    def sample_output(self):
        """
        Generates a sample output message using the given input.

        Returns:
            str: The generated output message.
        """
        message = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0,
            system="You are an expert task manager who is figuring out what to work on ",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Given the following what should my next 30 mins be spent doing Admin, Infra And Planning\nNeed to improve this document\nCurrently building out AI systems to automate my own tasks\nAs such I need to improve discord tooling so that it can let me know what to do over the next 30 minutes \nNeed to pay my Bloomberg bill\nKPI dashboard\nNeed a daily PNL generation tool across all brokerages\nNeed a daily engagement tool for websites \nMy code base is a mess. The backtests are unintelligible especially for the lead lag strategy. I need to migrate them to repositories in a single usable function that can be run on a production basis \nGoing to Crypto AI conference in New York City tomorrow\nNeed to get my girlfriend some clothes that are more appropriate \nNeed to pack \nAI Agents, Content and Brand\nIn place of hiring I am building AI Software Engineer Agents\nThe first step is implementing the Open Source Software Agent to understand what is going on https://github.com/princeton-nlp/SWE-agent\nFlows Strategy (Malal)\nImplement migration from TD Ameritrade API to Schwab. TD Ameritrade API gets deprecated soon. This is necessary for \nSchwab Endpoint does not work with local host. How to fix?\nAfter Schwab API is spun up need to productize the backtested day time strategy (realistically cannot get bulk pre market data without schwab API)\nFully backtest day time trading strategy with tcost assumptions baked in\nBacktest flows strategy on FX with AI augmentation \nSend rest of capital to IBKR from bank of america \nBuild Fatpig Japan, Fatpig Canada, and Fatpig UK (core flows strategy)\nI could also expand the flows strategy internationally to countries like Japan and China which would require more backtesting \nFollow up with interactive brokers about commission tier. Am trading enough that comms should have dropped by now but they have not. What causes them to drop? Verified that I am on tiered trading \nPost Catalyst Strategy (Nurgle)\nI have a strategy here which is based on number updates that takes a huge amount of time to update manually but is consistently profitable. \nEarnings season is coming up. it is possible to extract numbers from financial statements as they come out which would improve the efficacy of my earnings trading strategy. This will not work on all stocks as some are PDF driven \nThus there is potentially a large workflow to do to extract numbers from earnings transcripts. It is april 10. Realistically I wont complete this, sadly, which is really really bad but maybe I should start? \nI have a second strategy which is based on extracting information from transcripts, and scoring them - which is entirely LLM based\nThe state of this strategy is that it’s living on a server and the existing backetsts isnt profitable enough to straight up add in but it’s close. I have built a testing network \nThe LLM based strategy is the most text rich strategy \nMacro Strategy - Fundamental (Khorne)\nSpin up bare bones real time FX strategy\nport over real time interest rate swap data and citi economic surprise data, as well as a backtesting engine to get this deployed\nSpin up basic futures backtest \nI also have signal generation on all major hard assets including BTC, ETH, Stocks, Gasoline, Gold etc\nDeploy macro fundamental SPM \nFinish existing Crypto Backtest --  and deploy more capital to crypto \nClean up Crypto SPM code to generate a signal \nMove it to a cron job \nFund binance account \nI paid to get a bVI entity set up to access Binance and it’s currently set up. There is high perp funding rates arguing that the strategy should be long short. It’s probably semi urgent to fund the account so it stays active bc the window to keep it open or even reopen an accountis closing \nMacro Strategy - Non Fundamental (Slaanesh)\nMemes and bubbles -- I have a backtested wikipedia data tracking strategy that I could deploy here \nThe crypto market is surging. Bitcoin just ripped 8 fucking % \nPre Catalyst Strategy (Tzeentch)\nThis is the closest thing that I did at a hedge fund but that is now 8 years ago (lol fml) \nHave semi backtested framework which essentially front runs the post catalyst strategy by figuring out the likely z score of the next print \nThis overlays best with options data\nBut there’s a schwab migration coming up that makes me worried about the TD ameritrade options api so I need to figure that out \nAGFI.ai digital franchise and Social media strategy \nThis is on pause until all the SPMs are up and running and generating content\nI have a crypto AI conference on April 12 in New York City\nMy brother is coming too who is pitching a crypto L1 \nAGFI Token\nI think strategically this needs to come after all the strategies are deployed and augmented with AI and I am comfortable with them and content is being generated on them\nBut I think it’s worth slowly ramping up on Worldnet and open source AI integrations to actually figure out how the signal is going to be run"
                        }
                    ]
                }
            ]
        )
        output = message.content
        return output

    def generate_simple_text_output(self, model, max_tokens, temperature, system_prompt, user_prompt):
        """ 
        Example
        model="claude-3-opus-20240229",
        max_tokens=1000,
        temperature=0,
        system="You are an expert task manager who is figuring out what to work on ",
        """
        messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt
                        }
                    ]
                }
            ]

        message = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=messages
        )
        
        output = message.content
        return output

    def generate_claude_dataframe(self,model, max_tokens, temperature, system_prompt, user_prompt):
        output = self.generate_simple_text_output(model, max_tokens, temperature, system_prompt, user_prompt)
        output_map ={'text_response':output[0].text,
                     'model':model,
                     'max_tokens':max_tokens,
                     'temperature':temperature,
                     'system_prompt':system_prompt,
                     'user_prompt':user_prompt,
                     'date_run':datetime.datetime.now(),
                     'job_uuid': str(uuid.uuid4())}
        output_x = pd.DataFrame( output_map, index=[0])
        return output_x

    async def rate_limited_request(self, job_name, api_args):
        async with self.semaphore:
            await self.wait_for_rate_limit()
            print(f"Task {job_name} start: {datetime.datetime.now().time()}")
            try:
                response = await self.async_client.messages.create(**api_args)
                print(f"Task {job_name} end: {datetime.datetime.now().time()}")
                return job_name, response
            except anthropic.RateLimitError as e:
                print(f"Rate limit error for task {job_name}: {str(e)}")
                await asyncio.sleep(5)  # Wait for 5 seconds before retrying
                return await self.rate_limited_request(job_name, api_args)

    async def wait_for_rate_limit(self):
        now = time.time()
        self.request_times = [t for t in self.request_times if now - t < 60]
        if len(self.request_times) >= self.rate_limit:
            sleep_time = 60 - (now - self.request_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        self.request_times.append(time.time())

    async def get_completions(self, arg_async_map):
        '''Get completions asynchronously for given arguments map'''
        tasks = [self.rate_limited_request(job_name, args) for job_name, args in arg_async_map.items()]
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
            raw_df = pd.DataFrame({
                'id': completion_object.id,
                'model': completion_object.model,
                'role': completion_object.role,
                'content': completion_object.content[0].text if completion_object.content else '',
                'stop_reason': completion_object.stop_reason,
                'stop_sequence': completion_object.stop_sequence,
                'usage': str(completion_object.usage),
                'write_time': datetime.datetime.now(),
                'internal_name': internal_name
            }, index=[0])
            dfarr.append(raw_df)
        full_writable_df = pd.concat(dfarr)
        return full_writable_df

    def run_chat_completion_async_demo(self):
        '''Run demo for async chat completion'''
        job_hashes = [f'job{i}sample__{uuid.uuid4()}' for i in range(1, 6)]
        arg_async_map = {
            job_hashes[0]: {
                "model": self.default_model,
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": "Explain quantum computing in simple terms"}]
            },
            job_hashes[1]: {
                "model": self.default_model,
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": "What are the main differences between Python and JavaScript?"}]
            },
            job_hashes[2]: {
                "model": self.default_model,
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": "Describe the process of photosynthesis"}]
            },
            job_hashes[3]: {
                "model": self.default_model,
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": "Explain the theory of relativity"}]
            },
            job_hashes[4]: {
                "model": self.default_model,
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": "What are the key features of machine learning?"}]
            }
        }
        async_write_df = self.create_writable_df_for_async_chat_completion(arg_async_map=arg_async_map)
        return async_write_df