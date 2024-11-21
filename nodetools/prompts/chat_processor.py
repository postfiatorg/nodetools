#'claude-3-5-sonnet-20241022'
# THIS IS THE FOUNDATION HW r46SUhCzyGE4KwBnKQ6LmDmJcECCqdKy4q
from nodetools.utilities.generic_pft_utilities import GenericPFTUtilities
from nodetools.utilities.task_management import PostFiatTaskGenerationSystem
from nodetools.ai.openai import OpenAIRequestTool
from nodetools.chatbots.personas.odv import odv_system_prompt
import time
import nodetools.utilities.constants as constants

class ChatProcessor:
    def __init__(self,pw_map):
        self.pw_map = pw_map
        self.generic_pft_utilities = GenericPFTUtilities(pw_map=pw_map, node_name=constants.DEFAULT_NODE_NAME)
        self.open_ai_request_tool= OpenAIRequestTool(pw_map=pw_map)

    def process_chat_cue(self):
        account_address=constants.TESTNET_REMEMBRANCER_ADDRESS if constants.USE_TESTNET else constants.REMEMBRANCER_ADDRESS
        full_holder_df = self.generic_pft_utilities.output_post_fiat_holder_df()
        full_holder_df['balance']=full_holder_df['balance'].astype(float)
        all_top_wallet_holders = full_holder_df.sort_values('balance',ascending=True)
        real_users = all_top_wallet_holders[(all_top_wallet_holders['balance']*-1)>2_000].copy()
        top_accounts = list(real_users['account'].unique())
        full_message_cue = self.generic_pft_utilities.get_all_account_compressed_messages(account_address=account_address)
        #valid_cue = full_message_cue[full_message_cue['account'].apply(lambda x: x in top_accounts)].copy()
        #valid_cue#['cleaned_message'[.apply(lambda x: 
        incoming_messages= full_message_cue[(full_message_cue['account'].apply(lambda x: x in top_accounts)) 
        & (full_message_cue['message_type'].apply(lambda x: x =="INCOMING"))]
        messages_to_work = incoming_messages[incoming_messages['cleaned_message'].apply(lambda x: 'ODV' in x)].copy()
        responses = full_message_cue[full_message_cue['message_type']=='OUTGOING'].copy()
        responses['memo_type']=responses['memo_type'].apply(lambda x: x.replace('_response',''))
        responses['sent']=1
        response_map= responses.groupby('memo_type').last()
        messages_to_work['already_sent']= messages_to_work['memo_type'].map(response_map['sent'])
        message_cue = messages_to_work[messages_to_work['already_sent']!=1].copy()
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
        message_cue['system_prompt']=odv_system_prompt
        accounts_to_map = list(message_cue['account'].unique())
        account_context_map={}
        for xaccount in accounts_to_map:
            account_context_map[xaccount]=self.generic_pft_utilities.get_full_user_context_string(xaccount)
        message_cue['user_context']=message_cue['account'].map(account_context_map)
        message_cue.set_index('memo_type',inplace=True)
        messages_to_work = list(message_cue.index)
        for mwork in messages_to_work:
        #mwork = messages_to_work[0]
            message_slice = message_cue.loc[mwork]
            user_query_replace= message_slice['cleaned_message']
            print("SYSTEM RESPONDING TO")
            print(user_query_replace)
            user_context_replace = message_slice['user_context']
            system_prompt = message_slice['system_prompt']
            destination_account = message_slice['account']
            
            user_prompt = user_prompt_constructor.replace('___USER_CONTEXT_REPLACE___',
                                                          user_context_replace).replace('___USER_QUERY_REPLACE___',user_query_replace)
            preview_req= self.open_ai_request_tool.o1_preview_simulated_request(system_prompt=system_prompt, user_prompt=user_prompt)
            op_response = """ODV SYSTEM: """+preview_req.choices[0].message.content
            message_id = mwork+'_response'
            self.generic_pft_utilities.send_pft_compressed_message_based_on_wallet_seed(wallet_seed=self.pw_map['postfiatremembrancer__v1xrpsecret'], user_name='odv',
                destination=destination_account,
                memo=op_response,
                compress=True,
                message_id=message_id)

    def process_chat_cue_continuously(self):
        i=0
        while i<1_000_000_000_000:
            try:
                time.sleep(2)
                self.process_chat_cue()
            except:
                print("FAILED PROCESSING CHAT CUE")
                pass