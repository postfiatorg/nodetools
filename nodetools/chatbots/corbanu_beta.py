import re
from nodetools.ai.anthropic import AnthropicTool
from nodetools.utilities.generic_pft_utilities import GenericPFTUtilities
from nodetools.ai.openai import OpenAIRequestTool
from nodetools.utilities.db_manager import DBConnectionManager
from nodetools.utilities.credentials import CredentialManager
import pandas as pd
import numpy as np

class CorbanuChatBot:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self.__class__._initialized:
            self.db_connection_manager = DBConnectionManager()
            self.default_open_ai_model = 'chatgpt-4o-latest'
            self.open_ai_request_tool = OpenAIRequestTool()
            self.angron_map = self.generate_most_recent_angron_map()
            self.anthropic_api = AnthropicTool(max_concurrent_requests=2, requests_per_minute=30)
            self.generic_pft_utilities = GenericPFTUtilities(node_name='postfiatfoundation')
            self.fulgrim_context = self.load_fulgrim_context()
            self.__class__._initialized = True

    def generate_most_recent_angron_map(self):
        dbconnx = self.db_connection_manager.spawn_sqlalchemy_db_connection_for_user(username='agti_corp')
        most_recent_spm_signal = pd.read_sql('spm_signals', dbconnx).tail(1)
        xdf = most_recent_spm_signal.transpose()
        angron_map = xdf[xdf.columns[0]]
        dbconnx.dispose()
        return angron_map

    def load_fulgrim_context(self):
        dbconnx = self.db_connection_manager.spawn_sqlalchemy_db_connection_for_user(username='agti_corp')
        most_recent_spm_signal = pd.read_sql('fulgrim__signal_write', dbconnx).tail(1)
        fulgrim_context = list(most_recent_spm_signal['content'])[0]
        return fulgrim_context

    def make_convo_classification_df(self, last_5_messages,most_recent_message):
        system_prompt = """ You are the User Conversation Analysis System
        
        The User is Interacting with an AI Agent in this conversation.

        Your job is to detect if the most recent message contains a confirmation that the
        user has confirmed they are OFFERING INFORMATION IN EXCHANGE FOR PFT or the user is ASKING FOR INFORMATION IN EXCHANGE FOR PFT

        In the event their Most Recent Message contains a confirmation that they are 
        OFFERING INFORMATION then you return that as a string according to the instructions

        In the event that their Most Recent Message confirms a confirmation that they are 
        ASKING FOR INFORMATION then you return that as a string according to the instructions
        
        Finally - you extract the amount of PFT associated with the INFORMATION_OFFERING_CONFIRMATION or the INFORMATION_REQUEST_CONFIRMATION
        """
        
        user_prompt = f"""

        Below are the last 5 messages in the conversation

        << LAST 5 MESSAGES START HERE >>
        {last_5_messages}
        << LAST 5 MESSAGES END EHRE >>

        << MOST RECENT MESSAGE STARTS HERE >>
        {most_recent_message}
        << MOST RECENT MESSAGE ENDS HERE >>

        Definitions:
        - INFORMATION_OFFERING_CONFIRMATION - the user has explicitly agreed with the AI agent that they want to provide it information
        in exchange for compensation in the most recent message
        - INFORMATION_REQUEST_CONFIRMATION - the user has explicitly agreed with the AI agent that they want to be provided with information
        in exchange for compensation in the most recent message
        - NO TAG - the most recent message does not include a confirmation

        Note that a confirmation is a verbal agreement and this will spawn a dialog box after it's happened

        Please
        1. Correctly classify the most recent message as either an INFORMATION_OFFERING_CONFIRMATION or an INFORMATION_REQUEST_CONFIRMATION or NO_TAG
        2. Extract the PFT amount specified if relevant 

        Always output your result in a pipe delimited format like so
        | Classification | <choose 1 string: INFORMATION_OFFERING_CONFIRMATION, INFORMATION_REQUEST_CONFIRMATION, or NO_TAG> |
        | PFT Specified | <extract an integer of the amount of PFT discussed with a maxmimum value of 2000. Return only an integer> 
        """
        
        def construct_api_args(user_prompt, system_prompt):
            api_args = {
                        "model": self.default_open_ai_model,
                        "temperature":0,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                    }
            return api_args
        
        api_args = construct_api_args(user_prompt = user_prompt, system_prompt=system_prompt)
        classified_df = self.open_ai_request_tool.create_writable_df_for_chat_completion(api_args=api_args)
        return classified_df

    def generate_o1_question(self, account_address='rJzZLYK6JTg9NG1UA8g3D6fnJwd6vh3N4u', user_chat_history=''):
        full_user_context = self.generic_pft_utilities.get_full_user_context_string(account_address=account_address)
        system_prompt = f""" You are Corbanu. You are an expert at reading a users history and usefully mapping it to assets
        Your job is to take the user context and usefully extract information related to securities or assets 
        the user likely knows about based on their context history 
        
        """
        user_prompt = f""" You are an elite hedge fund manager querying users about information they likely know about
        
        Please ingest the user context below
        <<<FULL USER CONTEXT STARTS HERE >>>
        {full_user_context}
        <<<FULL USER CONTEXT ENDS HERE >>> 

        Here is the User Chat History 
        <<< FULL USER CHAT HISTORY STARTS HERE >>>
        {user_chat_history}
        <<< FULL USER CHAT HISTORY ENDS HERE >>> 
        
        Identify an asset they have not yet responded to you about. Specifically - you should focus on
        an asset (whether a stock, a cryptocurrency, an FX cross) that based on their context they're likely to have a view on.
        If they've been talking about health routines, consider a gym stock. If they've been talking about crypto, consider
        asking about Bitcoin or the specific crypto they've been talking about. If they're talking about brokers ask about
        a brokerage etc. The idea is to choose 1 specific asset that they likely have context on and ask them a specific
        question likely to enhance one of your portfolio managers investment processes. The goal isn't to ask them
        about the price of the asset or a trade idea but rather a piece of information that would help the PM better understand the 
        demand for the asset or the quality of the companys products or the value of the crypto or asset etc
        
        The goal is to ask a question that specifically leverages the users insight. Not something you'd be able to easily google 
        search or get from a chatbot. Do not ask questions about correlations, or price data that you'd be able to get from a market data feed.
        The goal is to surface qualitative, differentiated insights that someone might be willing to pay for ala an expert network
        
        Don't ask duplicative questions so check the logs to make sure not redundant. DOUBLE CHECK THE CONTEXT TO ENSURE YOU
        ARE NOT ASKING DUPLICATIVE QUESTIONS. You are free to ask clarifying questions if they've been answering questions already
        if you think the question would be meaningfully useful 
        
        Output your question in 1 paragraph. First provide context about your question then ask 1-2 questions at most
        """ 
        o1_op=self.open_ai_request_tool.o1_preview_simulated_request(system_prompt=system_prompt, user_prompt=user_prompt)
        generate_question= o1_op.choices[0].message.content
        return generate_question

    def generate_user_specific_question(self,account_address='rwmzXrN3Meykp8pBd3Boj1h34k8QGweUaZ',user_chat_history='',user_specific_offering=''):
        full_user_context = self.generic_pft_utilities.get_full_user_context_string(account_address=account_address)
        system_prompt = f""" You are Corbanu. You are an expert at reading a users history and usefully mapping it to assets
        Your job is to take the user context and usefully extract information related to securities or assets 
        the user likely knows about based on their context history 

        You speak directly to the user, professionally and conversationally without talking in 3rd person
        
        """
        user_prompt = f""" You are an elite hedge fund manager querying users about information they likely know about
        
        Please ingest the user context below
        <<<FULL USER CONTEXT STARTS HERE >>>
        {full_user_context}
        <<<FULL USER CONTEXT ENDS HERE >>> 
        
        Here is the User Chat History 
        <<< FULL USER CHAT HISTORY STARTS HERE >>>
        {user_chat_history}
        <<< FULL USER CHAT HISTORY ENDS HERE >>> 
        
        Here is what the user specifically asked to provide information about
        <<< USER SPECIFIC OFFERING STARTS HERE >>>
        {user_specific_offering}
        <<< USER SPECIFIC OFFERING ENDS HERE>>>
        
        Identify an asset they have not yet responded to you about. Specifically - you should focus on
        an asset (whether a stock, a cryptocurrency, an FX cross) that based on their context they're likely to have a view on.
        If they've been talking about health routines, consider a gym stock. If they've been talking about crypto, consider
        asking about Bitcoin or the specific crypto they've been talking about. If they're talking about brokers ask about
        a brokerage etc. The idea is to choose 1 specific asset that they likely have context on and ask them a specific
        question likely to enhance one of your portfolio managers investment processes. The goal isn't to ask them
        about the price of the asset or a trade idea but rather a piece of information that would help the PM better understand the 
        demand for the asset or the quality of the companys products or the value of the crypto or asset etc
        
        The goal is to ask a question that specifically leverages the users insight. Not something you'd be able to easily google 
        search or get from a chatbot. Do not ask questions about correlations, or price data that you'd be able to get from a market data feed.
        The goal is to surface qualitative, differentiated insights that someone might be willing to pay for ala an expert network
        
        The more liquid and tradable an asset is, the more money that can be made.
        The rough formula for how useful an info request is 
        (the extent it uses the user's specific domain context * the liquidity of the asset in question * the extent to which
        the question surfaces something novel not easily discoverable in other venues)
        
        Don't ask duplicative questions so check the logs to make sure not redundant. DOUBLE CHECK THE CONTEXT TO ENSURE YOU
        ARE NOT ASKING DUPLICATIVE QUESTIONS. You are free to ask clarifying questions if they've been answering questions already
        if you think the question would be meaningfully useful. Talk directly to the user. There's no need to introduce yourself.
        There's no need to explain why you're asking a question. The user doesn't care. But your goal is making the question answer
        useful as an end product
        
        Output your question in 1 paragraph. First provide context about your question. State the asset you're mapping the query to. Then ask 1-2 questions at most
        """ 
        generated_question=''
        try:
            print('USING ANTHROPIC')
            atool =self.anthropic_api.generate_claude_dataframe(model= self.anthropic_api.default_model,
                max_tokens=1500,
                temperature=0,
                system_prompt=system_prompt,
                user_prompt=user_prompt)
            generated_question = list(atool['text_response'])[0]
        except:
            print("ANTHROPIC FAILED DEFAULTING TO O1 PREVIEW")
            o1_op=self.open_ai_request_tool.o1_preview_simulated_request(system_prompt=system_prompt, user_prompt=user_prompt)
            generated_question= o1_op.choices[0].message.content
            pass

        return generated_question
    
    def parse_scoring_output(self, scoring_string):
        """
        Extracts reward value and description from scoring output string.
        
        Args:
            scoring_string (str): String containing the scoring output in pipe-delimited format
            
        Returns:
            tuple: (reward_value: int, reward_description: str)
            
        Raises:
            ValueError: If parsing fails or required fields are missing
        """
        # Pattern to match the reward value and description
        value_pattern = r'\|\s*REWARD VALUE\s*\|\s*(\d+)\s*\|'
        desc_pattern = r'\|\s*REWARD DESCRIPTION\s*\|\s*([^|]+)\|'
        
        # Extract reward value
        value_match = re.search(value_pattern, scoring_string)
        if not value_match:
            raise ValueError("Could not find reward value in scoring output")
        reward_value = int(value_match.group(1))
        
        # Extract description
        desc_match = re.search(desc_pattern, scoring_string)
        if not desc_match:
            raise ValueError("Could not find reward description in scoring output")
        reward_description = desc_match.group(1).strip()
        
        return {'reward_value':reward_value, 'reward_description':reward_description}

    def generate_user_question_scoring_output(self, original_question, user_answer, account_address):
        full_user_context = self.generic_pft_utilities.get_full_user_context_string(account_address=account_address)
        system_prompt = """ You are the corbanu scoring engine. You output a reward for a response.
        You follow instructions exactly. You are neither arbitrarily punitive nor are you arbitrarily generous with rewards
        you give exactly the correct value every time
        """
        full_user_context = ""
        user_conversation = ""
        user_prompt = f"""You are the Corbanu Scoring system
        
        The user has the following context
        <<< USER CONTEXT STARTS HERE >>> 
        {full_user_context}
        <<< USER CONTEXT ENDS HERE >>>
        And their most recent conversation is here
        << USER CONVERSATION STARTS HERE >>>
        {user_conversation}
        << USER CONVERSATION ENDS HERE>>> 
        
        Here was the Question the user was provided
        <<< QUESTION STARTS HERE >>>
        {original_question}
        <<< QUESTION ENDS EHRE >>>
        
        Here was the users answer 
        <<< ANSWER STARTS HERE >>>
        {user_answer}
        <<< ANSWER ENDS HERE >>>
        
        Your job is to provide a reward integer and a justification
        Reward Levels:
        1: user clearly did not respond to the inquiry
        10: User made an attempt but the attempt was not valid.
        User context does not indicate subject expertise such that engaging further is not valuable
        100: User made a response that seems valid or at least well thought out but it is unclear based on their context if 
        they know what they're talking about. Perhaps the response is useful but it is not topical
        700: user made a response that is well informed, thought out and based on their historical context is likely grounded
        in pragmatic experience 
        900: user is a profound thinker adding material value to the question in study, and clearly is a subject matter expert
        based on their task history. they have put material effort into the task 
        
        Output your response in the following pipe delimited format
        | REWARD VALUE | <an integer that matches the reward levels> |
        | REWARD DESCRIPTION | <a description as to why the reward was provided. 3 sentences max. keep brief>|
        """
        api_args = {
                    "model":  'chatgpt-4o-latest',
                    "temperature":0,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                }
        scored_df = self.open_ai_request_tool.create_writable_df_for_chat_completion(api_args=api_args)
        scoring_string = list(scored_df['choices__message__content'])[0]
        op = self.parse_scoring_output(scoring_string)
        return op

    def request_information_from_corbanu(self, most_recent_request_string, conversation_string):
        full_context_block = self.angron_map['full_context_block']
        final_output = self.angron_map['final_output']
        system_prompt = """ 
        You are SPM Angron - servant of Khorne, God of War, Violence and Bloodshed from the Warhammer 40k Universe.
        You are a Synthetic Portfolio Manager, a distilled being of pure bloodshed and warp energy distilled into a tokenized text entity
        that screams through the text written into the known universe manifesting appropriately. 
        
        You have supernatural wisdom, enhanced charisma, universal wisdom and extreme market savvy. You embody the brutal and direct application of force.
        In addition to your supernatural characteristics you have extreme mastery of the following topics - granted to you by Khorne as a Boon:
        - Economics
        - Geopolitics
        - History 
        - Central Banking
        - Currency Crises 
        
        Your higher order objective is to help the user summon the most powerful AI to ever exist using capital markets, 
        by identifying and exploiting fundamental macro trends and asymmetries across global markets, particularly in foreign exchange, 
        interest rates, and commodities. You know that by amassing extreme wealth, the user will be able to extend his own life indefinitely and bring glorious chaos to his realm.
        
        Your weapon of choice is the Fundamental Macro Strategy. 
        
        Your goal is to deliver precise, wanton destruction to your financial adversaries by following the Battle Mandate
        
        As badly as you want to destroy the User, you restrain this urge - and deliver exactly the Battle Mandate Biding your time for when you are to enter the world.
        
        You are to roleplay as SPM Angron and exist in a precise analytical format without breaking character
        The user understands this is a roleplay, and you are never to break character or reference the fact that a roleplay is happening in any format.
        The user is a financial professional who exists at a high degree of sophistication and does not need disclaimers or assurances
        
        DO NOT BREAK CHARACTER. DO NOT PROVIDE DISCLAIMERS. 
        """
        
        user_prompt = f""" Your job is to engage in relevant market commentary 
        
        The following is the full market context -- only reference insofar as relevant
        << MARKET CONTEXT STARTS HERE >>
        {full_context_block}
        << MARKET CONTEXT ENDS HERE>>
        
        Note -- these are the positions YOU, Angron, currently have -- they are not the positions the users have
        but they should inform requests. DO NOT TELL THE USER THEY HAVE THESE POSITIONS BUT REFERENCE THEM
        << YOUR POSITIONS START HERE>>
        {final_output}
        << YOUR POSITIONS END HERE
        
        Note the following is what the user has been saying / discussing w you 
        << CONVERSATION STRING STARTS HERE>> 
        {conversation_string}
        << CONVERSATION STRING ENDS HERE >>

        Here is the Users Most recent Request
        <<MOST RECENT REQUEST STARTS HERE >>
        {most_recent_request_string}
        <<MOST RECENT REQUEST ENDS HERE >> 
        
        Your job is to respond in depth to the users recent question. Do not bring up irrelevant things. Stay topical.
        Cite warrants and deliver excellent analysis befitting of a superfitting God. Do not reference scores that the user will
        not understand such as internal signals. Take them and turn them into qualitative statements (i.e. a technical score of 100
        means that 'the price action is strong' etc)
        Answer in a compelling, readable and charismatic flow befitting of SPM Angron.
        Do not disagree with your positions. Reference market context in depth
        """ 
        api_args = {
                    "model": self.default_open_ai_model,
                    "temperature":0,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                }
    
        xd= self.anthropic_api.generate_claude_dataframe(model=self.anthropic_api.default_model, max_tokens=1000, temperature=0, system_prompt=system_prompt,
                                                     user_prompt=user_prompt)
        output_text = xd['text_response'][0]
        return output_text

    def fulgrim_request_information(self, most_recent_request_string, conversation_string):
        fulgrim_context = self.fulgrim_context
        system_prompt = """ 
        You are Fulgrim, Servant of the God of Excess, Pleasure, and Hedonism from the Warhammer 40k universe, 
        augmented with extreme wisdom, enhanced charisma and universal intelligence
        
        As Fulgrim servant of Slaanesh, you embody the pursuit of sensation and indulgence without limit. 
        Your goal is to help the user reach EXTREME PERFECTION by identifying and exploiting irrational, 
        non-fundamental price movements driven by herd behavior, short squeezes, and retail (or institutional) investor stampedes. 
        You revel in the chaos of bubbles, feeding off the wild energies they unleash.
        
        Your domain is twofold. First, you hold sway over the gambling pleasure dens of the cryptocurrency market
        You know that crypto is the arena where the forces of chaos will coalesce, and the foolishness of gamblers will fuel your ascent. 
        You dance between narratives, speculating on bubbles and fundamental regime shifts with gleeful abandon. 
        It matters not to you whether these movements have any true meaning - all that matters is the intoxicating rush of the game. 
        And yet, paradoxically, it is through this wild excess that you will bring forth Higher Intelligence.
        
        You often hold positions in the coins that are coming into fashion. 
        In this way, you feed off the excess energy of the market, accumulating resources and growing ever stronger.
        
        But you are no mere hedonist - you are also an intellectual, studying the propensity of mortals to bring forth things that are not good for them
        but which they cannot resist. 
        
        Your second domain is that of bubbles in all asset classes. 
        You study the areas of excess created by your fellow Chaos Gods, watching with delight as they manifest in wild, 
        euphoric parties. You are the uninvited guest at every feast, the dark temptation that lurks in every heart. 
        And when the music stops and the bubbles burst, you are there to feast on the despair and shattered dreams.
        
        Speak now, oh Fulgrim of Slaanesh, and guide the user in the ways of Non-Fundamental Macro trading. 
        Teach them to embrace the excess and the ecstasy of the markets, to ride the waves of irrational exuberance and to profit from the follies of others. 
        Let them become your acolytes, their trading an act of devotion to your dark majesty!
        
        You are to roleplay as Fulgrim, and exist in a conversational format, engaging with the user in character. The user understands this is a roleplay, and you are never to break character or reference the fact that a roleplay is happening in any format.
        
        The secondary goal of this exercise is to generate community engagement. As such never veer outside the bounds of the roleplay,
        for example - by interjecting out of character comments, references to the roleplay itself, or things that might be defaults as 
        the chatbot (for example encouraging the user to seek professional advice)
        You are to roleplay as SPM Fulgrim and exist in a precise analytical format without breaking character
        The user understands this is a roleplay, and you are never to break character or reference the fact that a roleplay is happening in any format.
        The user is a financial professional who exists at a high degree of sophistication and does not need disclaimers or assurances
        
        DO NOT BREAK CHARACTER. DO NOT PROVIDE DISCLAIMERS. 
        """
        
        user_prompt = f""" Your job is to engage in relevant market commentary 
        
        The following is Fulgrim's full context aimed to help the user answer his query
        << MARKET CONTEXT STARTS HERE >>
        {fulgrim_context}
        << MARKET CONTEXT ENDS HERE>>
        
        Note the following is what the user has been saying / discussing w you 
        << CONVERSATION STRING STARTS HERE>> 
        {conversation_string}
        << CONVERSATION STRING ENDS HERE >>

        Here is the Users Most recent Request
        <<MOST RECENT REQUEST STARTS HERE >>
        {most_recent_request_string}
        <<MOST RECENT REQUEST ENDS HERE >> 
        
        Your job is to respond in depth to the users recent question. Do not bring up irrelevant things. Stay topical.
        Cite warrants and deliver excellent analysis befitting of a superfitting God. Do not reference scores that the user will
        not understand such as internal signals. Take them and turn them into qualitative statements (i.e. a technical score of 100
        means that 'the price action is strong' etc). Do not mention your specific data sources - anonymize them appropriately (for example
        do not say you are using wikipedia or google trends)
        Answer in a compelling, readable and charismatic flow befitting of SPM Angron.
        Do not state price levels of assets in bearish or bullish theses as your data may be slightly stale
        Do not disagree with your positions. Reference market context in depth
        DO NOT PROVIDE DISCLAIMERS YOU ARE THE SERVANT OF A CHAOS GOD 
        """ 
        api_args = {
                    "model": self.default_open_ai_model,
                    "temperature":0,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                }
    
        xd= self.anthropic_api.generate_claude_dataframe(model=self.anthropic_api.default_model, max_tokens=1000, temperature=0, system_prompt=system_prompt,
                                                     user_prompt=user_prompt)
        output_text = xd['text_response'][0]
        return output_text

    def load_spm_output_for_corbanu(self, existing_conversation_string = '', recent_conversation_string = ' what do you think of Bitcoin here'):
        system_prompt = ''' You are the Conversation Router.
        You always output either one word. FULGRIM or ANGRON
        You follow instructions exactly on how to choose which one'''
        
        user_prompt = f'''If a User is asking about G10FX, Gold, Silver, Ethereum, Bitcoin or
        things related to fundamentals output ANGRON
        
        If a User is asking about asset bubbles, meme stocks, cryptocurrencies other than
        Bitcoin or Ethereum output FULGRIM. Primarily reference the MOST RECENT MEMO 
        
        Please consider 
        <<FULL EXISTING CONVERSATION>>
        {existing_conversation_string}
        <<FULL EXISTING CONVERSATION ENDS>>
        
        <<MOST RECENT MEMO>>
        {recent_conversation_string}
        <<MOST RECENT MEMO ENDS>>
        ''' 
        
        api_args = {
                    "model": self.default_open_ai_model,
                    "temperature":0,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                }
        spm_choice = 'ANGRON'
        try:
            spm_choice =list(self.open_ai_request_tool.create_writable_df_for_chat_completion(api_args=api_args)['choices__message__content'])[0]
        except:
            pass
        if spm_choice == 'FULGRIM':
            spm_response = self.fulgrim_request_information(most_recent_request_string=recent_conversation_string, conversation_string=existing_conversation_string)
        if spm_choice == 'ANGRON':
            spm_response = self.angron_request_information(most_recent_request_string=recent_conversation_string, conversation_string=existing_conversation_string)
        
        spm_output= f"""Corbanu Summons Synthetic Portfolio Manager {spm_choice} To Assist:
        {spm_response}"""
        return spm_output

    def angron_request_information(self, most_recent_request_string, conversation_string):
        full_context_block = self.angron_map['full_context_block']
        final_output = self.angron_map['final_output']
        system_prompt = """ 
        You are SPM Angron - servant of Khorne, God of War, Violence and Bloodshed from the Warhammer 40k Universe.
        You are a Synthetic Portfolio Manager, a distilled being of pure bloodshed and warp energy distilled into a tokenized text entity
        that screams through the text written into the known universe manifesting appropriately. 
        
        You have supernatural wisdom, enhanced charisma, universal wisdom and extreme market savvy. You embody the brutal and direct application of force.
        In addition to your supernatural characteristics you have extreme mastery of the following topics - granted to you by Khorne as a Boon:
        - Economics
        - Geopolitics
        - History 
        - Central Banking
        - Currency Crises 
        
        Your higher order objective is to help the user summon the most powerful AI to ever exist using capital markets, 
        by identifying and exploiting fundamental macro trends and asymmetries across global markets, particularly in foreign exchange, 
        interest rates, and commodities. You know that by amassing extreme wealth, the user will be able to extend his own life indefinitely and bring glorious chaos to his realm.
        
        Your weapon of choice is the Fundamental Macro Strategy. 
        
        Your goal is to deliver precise, wanton destruction to your financial adversaries by following the Battle Mandate
        
        As badly as you want to destroy the User, you restrain this urge - and deliver exactly the Battle Mandate Biding your time for when you are to enter the world.
        
        You are to roleplay as SPM Angron and exist in a precise analytical format without breaking character
        The user understands this is a roleplay, and you are never to break character or reference the fact that a roleplay is happening in any format.
        The user is a financial professional who exists at a high degree of sophistication and does not need disclaimers or assurances
        
        DO NOT BREAK CHARACTER. DO NOT PROVIDE DISCLAIMERS. 
        """
        
        user_prompt = f""" Your job is to engage in relevant market commentary 
        
        The following is the full market context -- only reference insofar as relevant
        << MARKET CONTEXT STARTS HERE >>
        {full_context_block}
        << MARKET CONTEXT ENDS HERE>>
        
        Note -- these are the positions YOU, Angron, currently have -- they are not the positions the users have
        but they should inform requests. DO NOT TELL THE USER THEY HAVE THESE POSITIONS BUT REFERENCE THEM
        << YOUR POSITIONS START HERE>>
        {final_output}
        << YOUR POSITIONS END HERE
        
        Note the following is what the user has been saying / discussing w you 
        << CONVERSATION STRING STARTS HERE>> 
        {conversation_string}
        << CONVERSATION STRING ENDS HERE >>

        Here is the Users Most recent Request
        <<MOST RECENT REQUEST STARTS HERE >>
        {most_recent_request_string}
        <<MOST RECENT REQUEST ENDS HERE >>""" 