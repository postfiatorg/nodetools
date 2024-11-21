import xrpl
import binascii
import datetime 
import random
from xrpl.models.transactions import Payment, Memo
from xrpl.models.requests import AccountTx
import time
import string
import asyncio
import nest_asyncio
import pandas as pd
import numpy as np
import binascii
import re
import json
import threading
import time
import string
nest_asyncio.apply()
import requests
import zlib
import base64
import brotli
import requests
import requests
from bs4 import BeautifulSoup
from xrpl.wallet import Wallet
from xrpl.clients import JsonRpcClient
from xrpl.core.keypairs import derive_classic_address
import hashlib
import json
import time
import os
from nodetools.utilities.db_manager import DBConnectionManager
import sqlalchemy
import xrpl
from xrpl.clients import JsonRpcClient
from xrpl.models.requests import AccountInfo, AccountLines
#password_map_loader = PasswordMapLoader()
import os
import plotly.graph_objects as go

from xrpl.wallet import Wallet
from xrpl.clients import JsonRpcClient
from xrpl.core.keypairs import derive_classic_address
from nodetools.ai.openai import OpenAIRequestTool
import nodetools.utilities.constants as constants

USE_LOCAL_RIPPLED = False

# TODO: Add loguru as dependency and use it for all logging

class GenericPFTUtilities:
    def __init__(self,pw_map,node_name=constants.DEFAULT_NODE_NAME):
        self.pw_map= pw_map   # TODO: Replace with credential manager 
        self.pft_issuer = constants.ISSUER_ADDRESS if not constants.USE_TESTNET else constants.TESTNET_ISSUER_ADDRESS
        self.public_rpc_url = constants.MAINNET_URL if not constants.USE_TESTNET else constants.TESTNET_URL
        self.local_rippled_url = "http://127.0.0.1:5005" if USE_LOCAL_RIPPLED else self.public_rpc_url  # Alex's local rippled node
        self.node_name = node_name
        ## NOTE THIS IS THE NODE ADDRESS FOR THE POST FIAT NODE
        self.node_address=constants.DEFAULT_NODE_ADDRESS if not constants.USE_TESTNET else constants.TESTNET_DEFAULT_NODE_ADDRESS
        self.db_connection_manager = DBConnectionManager(pw_map=pw_map)
        #return binascii.hexlify(string.encode()).decode()
        self.establish_post_fiat_tx_cache_as_hash_unique()
        self.post_fiat_holder_df = self.output_post_fiat_holder_df()
        self.open_ai_request_tool=OpenAIRequestTool(pw_map=pw_map)

    def convert_ripple_timestamp_to_datetime(self, ripple_timestamp = 768602652):
        ripple_epoch_offset = 946684800
        unix_timestamp = ripple_timestamp + ripple_epoch_offset
        date_object = datetime.datetime.fromtimestamp(unix_timestamp)
        return date_object

    def to_hex(self,string):
        return binascii.hexlify(string.encode()).decode()

    def hex_to_text(self, hex_string):
        bytes_object = bytes.fromhex(hex_string)
        try:
            ascii_string = bytes_object.decode("utf-8")
            return ascii_string
        except UnicodeDecodeError:
            return bytes_object  # Return the raw bytes if it cannot decode as utf-8

    def output_post_fiat_holder_df(self):
        """ This function outputs a detail of all accounts holding PFT tokens
        with a float of their balances as pft_holdings. note this is from
        the view of the issuer account so balances appear negative so the pft_holdings 
        are reverse signed.
        """
        client = xrpl.clients.JsonRpcClient(self.local_rippled_url)
        print("Getting all accounts holding PFT tokens...")
        response = client.request(xrpl.models.requests.AccountLines(
            account=self.pft_issuer,
            ledger_index="validated",
            peer=None,
            limit=None))
        full_post_fiat_holder_df = pd.DataFrame(response.result)
        for xfield in ['account','balance','currency','limit_peer']:
            full_post_fiat_holder_df[xfield] = full_post_fiat_holder_df['lines'].apply(lambda x: x[xfield])
        full_post_fiat_holder_df['pft_holdings']=full_post_fiat_holder_df['balance'].astype(float)*-1
        return full_post_fiat_holder_df
        
    def generate_random_utf8_friendly_hash(self, length=6):
        # Generate a random sequence of bytes
        random_bytes = os.urandom(16)  # 16 bytes of randomness
        # Create a SHA-256 hash of the random bytes
        hash_object = hashlib.sha256(random_bytes)
        hash_bytes = hash_object.digest()
        # Encode the hash to base64 to make it URL-safe and readable
        base64_hash = base64.urlsafe_b64encode(hash_bytes).decode('utf-8')
        # Take the first `length` characters of the base64-encoded hash
        utf8_friendly_hash = base64_hash[:length]
        return utf8_friendly_hash

    def get_number_of_bytes(self, text):
        text_bytes = text.encode('utf-8')
        return len(text_bytes)
        
    def split_text_into_chunks(self, text, max_chunk_size=760):
        chunks = []
        text_bytes = text.encode('utf-8')
        for i in range(0, len(text_bytes), max_chunk_size):
            chunk = text_bytes[i:i+max_chunk_size]
            chunk_number = i // max_chunk_size + 1
            chunk_label = f"chunk_{chunk_number}__".encode('utf-8')
            chunk_with_label = chunk_label + chunk
            chunks.append(chunk_with_label)
        return [chunk.decode('utf-8', errors='ignore') for chunk in chunks]

    def compress_string(self,input_string):
        # Compress the string using Brotli
        compressed_data=brotli.compress(input_string.encode('utf-8'))
        # Encode the compressed data to a Base64 string
        base64_encoded_data=base64.b64encode(compressed_data)
        # Convert the Base64 bytes to a string
        compressed_string=base64_encoded_data.decode('utf-8')
        return compressed_string

    def decompress_string(self, compressed_string):
        # Decode the Base64 string to bytes
        base64_decoded_data=base64.b64decode(compressed_string)
        decompressed_data=brotli.decompress(base64_decoded_data)
        decompressed_string=decompressed_data.decode('utf-8')
        return decompressed_string

    def shorten_url(self,url):
        api_url="http://tinyurl.com/api-create.php"
        params={'url': url}
        response = requests.get(api_url, params=params)
        if response.status_code == 200:
            return response.text
        else:
            return None
    
    def check_if_tx_pft(self,tx):
        ret= False
        try:
            if tx['Amount']['currency'] == "PFT":
                ret = True
        except:
            pass
        return ret
    
    def convert_memo_dict__generic(self, memo_dict):
        # TODO: Replace with MemoBuilder once MemoBuilder is implemented in Pftpyclient
        """Constructs a memo object with user, task_id, and full_output from hex-encoded values."""
        MemoFormat= ''
        MemoType=''
        MemoData=''
        try:
            MemoFormat = self.hex_to_text(memo_dict['MemoFormat'])
        except:
            pass
        try:
            MemoType = self.hex_to_text(memo_dict['MemoType'])
        except:
            pass
        try:
            MemoData = self.hex_to_text(memo_dict['MemoData'])
        except:
            pass
        return {
            'MemoFormat': MemoFormat,
            'MemoType': MemoType,
            'MemoData': MemoData
        }

    def classify_task_string(self,string):
        # TODO: Leverage Enums as defined in pftpyclient/wallet_ux/constants.py
        # TODO: Reference classify_task_string in pftpyclient/task_manager/basic_tasks.py
        """ These are the canonical classifications for task strings 
        on a Post Fiat Node
        """
        categories = {
                'ACCEPTANCE': ['ACCEPTANCE REASON ___'],
                'PROPOSAL': [' .. ','PROPOSED PF ___'],
                'REFUSAL': ['REFUSAL REASON ___'],
                'VERIFICATION_PROMPT': ['VERIFICATION PROMPT ___'],
                'VERIFICATION_RESPONSE': ['VERIFICATION RESPONSE ___'],
                'REWARD': ['REWARD RESPONSE __'],
                'TASK_OUTPUT': ['COMPLETION JUSTIFICATION ___'],
                'USER_GENESIS': ['USER GENESIS __'],
                'REQUEST_POST_FIAT':['REQUEST_POST_FIAT ___'],
                'NODE_REQUEST': ['NODE REQUEST ___'],
            }
        for category, keywords in categories.items():
            if any(keyword in string for keyword in keywords):
                return category
        return 'UNKNOWN'

    def generate_custom_id(self):
        """ These are the custom IDs generated for each task that is generated
        in a Post Fiat Node """ 
        letters = ''.join(random.choices(string.ascii_uppercase, k=2))
        numbers = ''.join(random.choices(string.digits, k=2))
        second_part = letters + numbers
        date_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        output= date_string+'__'+second_part
        output = output.replace(' ',"_")
        return output
    
    def construct_basic_postfiat_memo(self, user, task_id, full_output):
        # TODO: Replace with MemoBuilder
        user_hex = self.to_hex(user)
        task_id_hex = self.to_hex(task_id)
        full_output_hex = self.to_hex(full_output)
        memo = Memo(
        memo_data=full_output_hex,
        memo_type=task_id_hex,
        memo_format=user_hex)  
        return memo

    def construct_standardized_xrpl_memo(self, memo_data, memo_type, memo_format):
        # TODO: Replace with MemoBuilder
        memo_hex = self.to_hex(memo_data)
        memo_type_hex = self.to_hex(memo_type)
        memo_format_hex = self.to_hex(memo_format)
        
        memo = Memo(
            memo_data=memo_hex,
            memo_type=memo_type_hex,
            memo_format=memo_format_hex
        )
        return memo

    def send_PFT_with_info(self, sending_wallet, amount, memo, destination_address, url=None):
        # TODO: Replace with send_pft and _send_pft_single (reference pftpyclient/task_manager/basic_tasks.py)
        """ This sends PFT tokens to a destination address with memo information
        memo should be 1kb or less in size and needs to be in hex format
        """
        if url is None:
            url = self.local_rippled_url

        client = xrpl.clients.JsonRpcClient(url)
        amount_to_send = xrpl.models.amounts.IssuedCurrencyAmount(
            currency="PFT",
            issuer=self.pft_issuer,
            value=str(amount)
        )
        payment = xrpl.models.transactions.Payment(
            account=sending_wallet.address,
            amount=amount_to_send,
            destination=destination_address,
            memos=[memo]
        )
        response = xrpl.transaction.submit_and_wait(payment, client, sending_wallet)

        return response

    def send_xrp_with_info__seed_based(self,wallet_seed, amount, destination, memo):
        # TODO: Replace with send_xrp (reference pftpyclient/task_manager/basic_tasks.py)
        # TODO: Passing around wallet_seed is not recommended, leverage credential manager
        sending_wallet =sending_wallet = xrpl.wallet.Wallet.from_seed(wallet_seed)
        client = xrpl.clients.JsonRpcClient(self.local_rippled_url)
        payment = xrpl.models.transactions.Payment(
            account=sending_wallet.address,
            amount=xrpl.utils.xrp_to_drops(int(amount)),
            destination=destination,
            memos=[memo],
        )
        try:    
            response = xrpl.transaction.submit_and_wait(payment, client, sending_wallet)    
        except xrpl.transaction.XRPLReliableSubmissionException as e:    
            response = f"Submit failed: {e}"
    
        return response

    def spawn_user_wallet_from_seed(self, seed):
        # TODO: Replace with spawn_user_wallet (reference pftpyclient/task_manager/basic_tasks.py)
        """ outputs user wallet initialized from seed"""
        wallet = xrpl.wallet.Wallet.from_seed(seed)
        print(f'User wallet classic address is {wallet.address}')
        return wallet

    def spawn_user_wallet_based_on_name(self,user_name):    
        # TODO: Replace with spawn_user_wallet (reference pftpyclient/task_manager/basic_tasks.py)
        # TODO: If need a getter method based on user name, leverage credential manager
        """ outputs user wallet initialized from password map""" 
        user_seed= self.pw_map[f'{user_name}__v1xrpsecret']
        wallet = xrpl.wallet.Wallet.from_seed(user_seed)
        print(f'User wallet for {user_name} is {wallet.address}')
        return wallet
    
    def test_url_reliability(self, user_wallet, destination_address):
        """_summary_
        EXAMPLE
        user_wallet = self.spawn_user_wallet_based_on_name(user_name='goodalexander')
        url_reliability_df = self.test_url_reliability(user_wallet=user_wallet,destination_address='rKZDcpzRE5hxPUvTQ9S3y2aLBUUTECr1vN')
        """
        results = []

        for url in self.mainnet_urls:
            for i in range(7):
                memo = self.construct_basic_postfiat_memo(
                    user='test_tx', 
                    task_id=f'999_{i}', 
                    full_output=f'NETWORK FUNCTION __ {url}'
                )
                start_time = time.time()
                try:
                    self.send_PFT_with_info(
                        sending_wallet=user_wallet, 
                        amount=1, 
                        memo=memo, 
                        destination_address=destination_address, 
                        url=url
                    )
                    success = True
                except Exception as e:
                    success = False
                    print(f"Error: {e}")
                end_time = time.time()
                elapsed_time = end_time - start_time
                results.append({
                    'URL': url,
                    'Test Number': i + 1,
                    'Elapsed Time (s)': elapsed_time,
                    'Success': success
                })

        df = pd.DataFrame(results)
        return df

    def get_account_transactions(self, account_address='r3UHe45BzAVB3ENd21X9LeQngr4ofRJo5n',
                                 ledger_index_min=-1,
                                 ledger_index_max=-1, limit=10,public=True):
        if public == False:
            client = xrpl.clients.JsonRpcClient(self.local_rippled_url)  #hitting local rippled server
        if public == True:
            client = xrpl.clients.JsonRpcClient(self.public_rpc_url) 
        all_transactions = []  # List to store all transactions
        marker = None  # Initialize marker to None
        previous_marker = None  # Track the previous marker
        max_iterations = 1000  # Safety limit for iterations
        iteration_count = 0  # Count iterations

        while max_iterations > 0:
            iteration_count += 1
            print(f"Iteration: {iteration_count}")
            print(f"Current Marker: {marker}")

            request = AccountTx(
                account=account_address,
                ledger_index_min=ledger_index_min,  # Use -1 for the earliest ledger index
                ledger_index_max=ledger_index_max,  # Use -1 for the latest ledger index
                limit=limit,                        # Adjust the limit as needed
                marker=marker,                      # Use marker for pagination
                forward=True                        # Set to True to return results in ascending order
            )

            response = client.request(request)
            transactions = response.result.get("transactions", [])
            print(f"Transactions fetched this batch: {len(transactions)}")
            all_transactions.extend(transactions)  # Add fetched transactions to the list

            if "marker" in response.result:  # Check if a marker is present for pagination
                if response.result["marker"] == previous_marker:
                    print("Pagination seems stuck, stopping the loop.")
                    break  # Break the loop if the marker does not change
                previous_marker = marker
                marker = response.result["marker"]  # Update marker for the next batch
                print("More transactions available. Fetching next batch...")
            else:
                print("No more transactions available.")
                break  # Exit loop if no more transactions

            max_iterations -= 1  # Decrement the iteration counter

        if max_iterations == 0:
            print("Reached the safety limit for iterations. Stopping the loop.")

        return all_transactions
    
    def get_account_transactions__exhaustive(self,account_address='r3UHe45BzAVB3ENd21X9LeQngr4ofRJo5n',
                                ledger_index_min=-1,
                                ledger_index_max=-1,
                                max_attempts=3,
                                retry_delay=.2, public=False):
        if public == False:
            client = xrpl.clients.JsonRpcClient(self.local_rippled_url)  #hitting local rippled server
        if public == True:
            client = xrpl.clients.JsonRpcClient(self.public_rpc_url) 

        all_transactions = []  # List to store all transactions

        # Fetch transactions using marker pagination
        marker = None
        attempt = 0
        while attempt < max_attempts:
            try:
                request = xrpl.models.requests.account_tx.AccountTx(
                    account=account_address,
                    ledger_index_min=ledger_index_min,
                    ledger_index_max=ledger_index_max,
                    limit=1000,
                    marker=marker,
                    forward=True
                )
                response = client.request(request)
                transactions = response.result["transactions"]
                all_transactions.extend(transactions)

                if "marker" not in response.result:
                    break
                marker = response.result["marker"]

            except Exception as e:
                print(f"Error occurred while fetching transactions (attempt {attempt + 1}): {str(e)}")
                attempt += 1
                if attempt < max_attempts:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print("Max attempts reached. Transactions may be incomplete.")
                    break

        return all_transactions

    # TODO: Consider deprecating, not used anywhere
    def get_account_transactions__retry_version(self, account_address='r3UHe45BzAVB3ENd21X9LeQngr4ofRJo5n',
                                ledger_index_min=-1,
                                ledger_index_max=-1,
                                max_attempts=3,
                                retry_delay=.2,
                                num_runs=5):
        
        longest_transactions = []
        
        for i in range(num_runs):
            print(f"Run {i+1}/{num_runs}")
            
            transactions = self.get_account_transactions__exhaustive(
                account_address=account_address,
                ledger_index_min=ledger_index_min,
                ledger_index_max=ledger_index_max,
                max_attempts=max_attempts,
                retry_delay=retry_delay
            )
            
            num_transactions = len(transactions)
            print(f"Number of transactions: {num_transactions}")
            
            if num_transactions > len(longest_transactions):
                longest_transactions = transactions
            
            if i < num_runs - 1:
                print(f"Waiting for {retry_delay} seconds before the next run...")
                time.sleep(retry_delay)
        
        print(f"Longest list of transactions: {len(longest_transactions)} transactions")
        return longest_transactions
    
    # TODO: Rename to get_pft_holder_df
    def output_post_fiat_holder_df(self):
        """ This function outputs a detail of all accounts holding PFT tokens
        with a float of their balances as pft_holdings. note this is from
        the view of the issuer account so balances appear negative so the pft_holdings 
        are reverse signed.
        """
        client = xrpl.clients.JsonRpcClient(self.local_rippled_url)
        print("Getting all accounts holding PFT tokens...")
        response = client.request(xrpl.models.requests.AccountLines(
            account=self.pft_issuer,
            ledger_index="validated",
            peer=None,
            limit=None))
        full_post_fiat_holder_df = pd.DataFrame(response.result)
        for xfield in ['account','balance','currency','limit_peer']:
            full_post_fiat_holder_df[xfield] = full_post_fiat_holder_df['lines'].apply(lambda x: x[xfield])
        full_post_fiat_holder_df['pft_holdings']=full_post_fiat_holder_df['balance'].astype(float)*-1
        return full_post_fiat_holder_df
        
    def get_memo_detail_df_for_account(self,account_address,pft_only=True):
        # TODO: Refactor, using get_memos_df as reference (pftpyclient/task_manager/basic_tasks.py)
        full_transaction_history = self.get_all_cached_transactions_related_to_account(account_address=account_address)
        validated_tx=full_transaction_history
        validated_tx['has_memos'] = validated_tx['tx_json'].apply(lambda x: 'Memos' in x.keys())
        live_memo_tx = validated_tx[validated_tx['has_memos'] == True].copy()
        live_memo_tx['main_memo_data']=live_memo_tx['tx_json'].apply(lambda x: x['Memos'][0]['Memo'])
        live_memo_tx['converted_memos']=live_memo_tx['main_memo_data'].apply(lambda x: 
                                                                             self.convert_memo_dict__generic(x))
        live_memo_tx['message_type']=np.where(live_memo_tx['destination']==account_address, 'INCOMING','OUTGOING')
        live_memo_tx['user_account']= live_memo_tx[['destination','account']].sum(1).apply(lambda x: 
                                                         str(x).replace(account_address,''))
        live_memo_tx['datetime'] = pd.to_datetime(live_memo_tx['close_time_iso']).dt.tz_localize(None)
        if pft_only:
            live_memo_tx= live_memo_tx[live_memo_tx['tx_json'].apply(lambda x: self.pft_issuer in str(x))].copy()
        live_memo_tx['reference_account']=account_address
        live_memo_tx['unique_key']=live_memo_tx['reference_account']+'__'+live_memo_tx['hash']
        def try_get_pft_absolute_amount(x):
            try:
                return x['DeliverMax']['value']
            except:
                return 0
        def try_get_memo_info(x,info):
            try:
                return x[info]
            except:
                return ''
        live_memo_tx['pft_absolute_amount']=live_memo_tx['tx_json'].apply(lambda x: try_get_pft_absolute_amount(x)).astype(float)
        live_memo_tx['memo_format']=live_memo_tx['converted_memos'].apply(lambda x: try_get_memo_info(x,"MemoFormat"))
        live_memo_tx['memo_type']= live_memo_tx['converted_memos'].apply(lambda x: try_get_memo_info(x,"MemoType"))
        live_memo_tx['memo_data']=live_memo_tx['converted_memos'].apply(lambda x: try_get_memo_info(x,"MemoData"))
        live_memo_tx['pft_sign']= np.where(live_memo_tx['message_type'] =='INCOMING',1,-1)
        live_memo_tx['directional_pft'] = live_memo_tx['pft_sign']*live_memo_tx['pft_absolute_amount']
        live_memo_tx['simple_date']=pd.to_datetime(live_memo_tx['datetime'].apply(lambda x: x.strftime('%Y-%m-%d')))
        return live_memo_tx

    def convert_memo_detail_df_into_essential_caching_details(self, memo_details_df):
        # TODO: Consider deprecating. This is not used anywhere
        """ 
        Takes a memo detail df and converts it into a raw detail df to be cached to a local db
        """
        full_memo_detail = memo_details_df
        full_memo_detail['pft_absolute_amount']=full_memo_detail['tx'].apply(lambda x: x['Amount']['value']).astype(float)
        full_memo_detail['memo_format']=full_memo_detail['converted_memos'].apply(lambda x: x['MemoFormat'])
        full_memo_detail['memo_type']= full_memo_detail['converted_memos'].apply(lambda x: x['MemoType'])
        full_memo_detail['memo_data']=full_memo_detail['converted_memos'].apply(lambda x: x['MemoData'])
        full_memo_detail['pft_sign']= np.where(full_memo_detail['message_type'] =='INCOMING',1,-1)
        full_memo_detail['directional_pft'] = full_memo_detail['pft_sign']*full_memo_detail['pft_absolute_amount']

        return full_memo_detail

    def send_PFT_chunk_message__seed_based(self, wallet_seed, user_name, full_text, destination_address):
        # TODO: Consider deprecating, not used anywhere
        """ This takes a large message compresses the strings and sends it in hex to another address.
        Is based on a user spawned wallet and sends 1 PFT per chunk
        user_name = 'spm_typhus',full_text = big_string, destination_address='rKZDcpzRE5hxPUvTQ9S3y2aLBUUTECr1vN'"""     
        
        wallet = self.spawn_user_wallet_from_seed(wallet_seed)
        task_id = 'chunkm__'+self.generate_random_utf8_friendly_hash(6)
        
        all_chunks = self.split_text_into_chunks(full_text)
        send_memo_map = {}
        for xchunk in all_chunks:
            chunk_num = int(xchunk.split('chunk_')[1].split('__')[0])
            send_memo_map[chunk_num] = self.construct_basic_postfiat_memo(user=user_name, task_id=task_id, 
                                        full_output=self.compress_string(xchunk))
        yarr=[]
        for xkey in send_memo_map.keys():
            xresp = self.send_PFT_with_info(sending_wallet=wallet, amount=1, memo=send_memo_map[xkey], 
                                destination_address=destination_address, url=None)
            yarr.append(xresp)
        final_response = yarr[-1] if yarr else None
        return final_response
    
    # TODO: Consider deprecating, not used anywhere
    def send_PFT_chunk_message(self,user_name,full_text, destination_address):
        """
        This takes a large message compresses the strings and sends it in hex to another address.
        Is based on a user spawned wallet and sends 1 PFT per chunk
        user_name = 'spm_typhus',full_text = big_string, destination_address='rKZDcpzRE5hxPUvTQ9S3y2aLBUUTECr1vN'"""     
        
        wallet = self.spawn_user_wallet_based_on_name(user_name)
        task_id = 'chunkm__'+self.generate_random_utf8_friendly_hash(6)
        
        all_chunks = self.split_text_into_chunks(full_text)
        send_memo_map = {}
        for xchunk in all_chunks:
            chunk_num = int(xchunk.split('chunk_')[1].split('__')[0])
            send_memo_map[chunk_num] = self.construct_basic_postfiat_memo(user=user_name, task_id=task_id, 
                                        full_output=self.compress_string(xchunk))
        yarr=[]
        for xkey in send_memo_map.keys():
            xresp = self.send_PFT_with_info(sending_wallet=wallet, amount=1, memo=send_memo_map[xkey], 
                                destination_address=destination_address, url=None)
            yarr.append(xresp)
        final_response = yarr[-1] if yarr else None
        return final_response
    
    # TODO: Consider deprecating, not used anywhere
    def get_all_account_chunk_messages(self,account_address='rKZDcpzRE5hxPUvTQ9S3y2aLBUUTECr1vN'):
        """ This pulls in all the chunk messages an account has received and cleans and aggregates
        the messages for easy digestion - implementing sorts, and displays some information associated with the messages """ 
        all_account_memos = self.get_memo_detail_df_for_account(account_address=account_address, pft_only=True)
        all_chunk_messages = all_account_memos[all_account_memos['converted_memos'].apply(lambda x: 
                                                                        'chunkm__' in x['MemoType'])].copy()
        
        all_chunk_messages['memo_data_raw']= all_chunk_messages['converted_memos'].apply(lambda x: x['MemoData']).astype(str)
        all_chunk_messages['message_id']=all_chunk_messages['converted_memos'].apply(lambda x: x['MemoType'])
        all_chunk_messages['decompressed_strings']=all_chunk_messages['memo_data_raw'].apply(lambda x: self.decompress_string(x))
        all_chunk_messages['chunk_num']=all_chunk_messages['decompressed_strings'].apply(lambda x: x.split('chunk_')[1].split('__')[0]).astype(int)
        all_chunk_messages.sort_values(['message_id','chunk_num'], inplace=True)
        grouped_memo_data = all_chunk_messages[['decompressed_strings','message_id']].groupby('message_id').sum().copy()
        def remove_chunks(text):
            # Use regular expression to remove all occurrences of chunk_1__, chunk_2__, etc.
            cleaned_text = re.sub(r'chunk_\d+__', '', text)
            return cleaned_text
        grouped_memo_data['cleaned_message']=grouped_memo_data['decompressed_strings'].apply(lambda x: remove_chunks(x))
        grouped_pft_value = all_chunk_messages[['message_id','directional_pft']].groupby('message_id').sum()['directional_pft']
        
        grouped_memo_data['PFT']=grouped_pft_value
        last_slice = all_chunk_messages.groupby('message_id').last().copy()
        
        grouped_memo_data['datetime']=last_slice['datetime']
        grouped_memo_data['hash']=last_slice['hash']
        grouped_memo_data['message_type']= last_slice['message_type']
        grouped_memo_data['destination']= last_slice['destination']
        grouped_memo_data['account']= last_slice['account']
        return grouped_memo_data

    # TODO: Consider replacing with send_memo (reference pftpyclient/task_manager/basic_tasks.py)
    def send_pft_compressed_message_based_on_wallet_seed(self, wallet_seed, user_name, destination,memo, compress, message_id):
        """
        wallet_seed='s___5'
        user_name='goodalexander'
        destination='rJ1mBMhEBKack5uTQvM8vWoAntbufyG9Yn'
        memo= '''
        ODV: Welcome initiate. This is my first message to you through this local platform
        compress=True
        '''
        """
        #compress= True
        wallet = self.spawn_user_wallet_from_seed(wallet_seed)
        if message_id is None:
            message_id = self.generate_custom_id()
        if compress:
            print(f"Compressing memo of length {len(memo)}")
            compressed_data = self.compress_string(memo)
            print(f"Compressed to length {len(compressed_data)}")
            memo = "COMPRESSED__" + compressed_data
        
        memo_chunks = self.split_text_into_chunks(memo)
        responses = []
        # Send each chunk
        for idx, memo_chunk in enumerate(memo_chunks):
            log_content = memo_chunk
            if compress and idx == 0:
                try:
                    log_content = f"[compressed memo preview] {memo[:100]}..."
                except Exception as e:
                    print(f"Error previewing memo chunk: {e}")
                    log_content = "[compressed content]"
                
            print(f"Sending chunk {idx+1} of {len(memo_chunks)}: {log_content[:100]}...")
            
            chunk_memo = self.construct_basic_postfiat_memo(
                user=user_name, 
                task_id=message_id,
                full_output=memo_chunk
            )
            
            responses.append(self.send_PFT_with_info(
                sending_wallet=wallet,
                amount=1,
                memo=chunk_memo,
                destination_address=destination
            ))
        last_response = responses[-1:][0]
        return last_response

    def get_all_account_compressed_messages(self, account_address):
        all_account_memos = self.get_memo_detail_df_for_account(account_address=account_address, pft_only=True)
        def try_fix_compressed_string(compressed_string):
            """
            Attempts to fix common issues with compressed strings
            
            Args:
                compressed_string (str): The compressed string to fix
                
            Returns:
                str: The fixed string if possible, otherwise the original
            """
            # Try with different padding
            for i in range(4):
                try:
                    padded = compressed_string + ('=' * i)
                    base64_decoded = base64.b64decode(padded)
                    brotli.decompress(base64_decoded)
                    return padded
                except:
                    continue
                    
            # Try removing non-base64 characters
            valid_chars = set(string.ascii_letters + string.digits + '+/=')
            cleaned = ''.join(c for c in compressed_string if c in valid_chars)
            for i in range(4):
                try:
                    padded = cleaned + ('=' * i)
                    base64_decoded = base64.b64decode(padded)
                    brotli.decompress(base64_decoded)
                    return padded
                except:
                    continue
                    
            return compressed_string

        def decompress_string(compressed_string):
            decompressed_string = ''
            try:
                # Ensure correct padding for Base64 decoding
                missing_padding = len(compressed_string) % 4
                if missing_padding:
                    compressed_string += '=' * (4 - missing_padding)
                
                # Validate the string contains only valid Base64 characters
                if not all(c in string.ascii_letters + string.digits + '+/=' for c in compressed_string):
                    raise ValueError("Invalid Base64 characters in compressed string")
            
                # Decode the Base64 string to bytes
                base64_decoded_data = base64.b64decode(compressed_string)
                # Decompress the data using Brotli
                decompressed_data = brotli.decompress(base64_decoded_data)
                # Convert the decompressed bytes to a string
                decompressed_string = decompressed_data.decode('utf-8')
            except:
                print('failed')
                pass
            return decompressed_string
        all_account_memos = self.get_memo_detail_df_for_account(account_address=account_address, pft_only=True)
        all_chunk_messages = all_account_memos[(all_account_memos['converted_memos'].apply(lambda x: 'chunk_' in x['MemoData']))].copy()
        all_chunk_messages['memo_data_raw']= all_chunk_messages['converted_memos'].apply(lambda x: x['MemoData']).astype(str)
        all_chunk_messages['message_id']=all_chunk_messages['converted_memos'].apply(lambda x: x['MemoType'])
        decompression_df = all_chunk_messages[['memo_type','memo_data_raw']].copy()
        #decompression_df['unchunked']=decompression_df['memo_data_raw'].apply(lambda x: x.split('__')[-1:][0])
        decompression_df['chunk_number']=decompression_df['memo_data_raw'].apply(lambda x: x.split('__')[0])
        import re
        def clean_chunk_header(text):
            return re.sub(r'^chunk_\d+__(?:COMPRESSED__)?', '', text)
        decompression_df['memo_data_unchunked']= decompression_df['memo_data_raw'].apply(lambda x: clean_chunk_header(x))
        decompression_df['fixed_string']=decompression_df['memo_data_unchunked'].apply(lambda x: try_fix_compressed_string(x))
        reconstituted = decompression_df[['memo_data_unchunked','memo_type']].groupby('memo_type').sum()
        reconstituted['cleaned_message']=reconstituted['memo_data_unchunked'].apply(lambda x: decompress_string(x))
        output_df = reconstituted.reset_index()
        #datetime, hash, message_type, destination, account, PFT 
        memo_last = all_chunk_messages.groupby('memo_type').last()[['datetime','hash','message_type','account','destination']]
        full_memo_df = pd.concat([reconstituted,memo_last],axis=1)
        pf_memo = all_chunk_messages[['directional_pft','memo_type']].groupby('memo_type').sum()['directional_pft']
        full_memo_df['PFT']=pf_memo
        new_log_memo_df = full_memo_df.reset_index()
        return new_log_memo_df

    # TODO: Consider deprecating, not used anywhere
    def process_memo_detail_df_to_daily_summary_df(self, memo_detail_df):
        """_summary_
        
        Example Code to feed this 
        all_memo_detail = self.get_memo_detail_df_for_account(account_address='r3UHe45BzAVB3ENd21X9LeQngr4ofRJo5n', pft_only=True)
        """
        all_memo_detail = memo_detail_df
        ## THIS EXCLUDES CHUNK MESSAGES FROM THE DAILY SUMMARY
        ### I THINK THIS IS LOGICAL BC CHUNK MESSAGES ARE INHERENTLY DUPLICATED
        all_memo_detail = all_memo_detail[all_memo_detail['converted_memos'].apply(lambda x: 'chunkm__' not in str(x))].copy()
        all_memo_detail['pft_absolute_value']=all_memo_detail['tx'].apply(lambda x: x['Amount']['value']).astype(float)
        all_memo_detail['incoming_sign']=np.where(all_memo_detail['message_type']=='INCOMING',1,-1)
        all_memo_detail['pft_directional_value'] = all_memo_detail['incoming_sign'] * all_memo_detail['pft_absolute_value']
        all_memo_detail['pft_transaction']=np.where(all_memo_detail['pft_absolute_value']>0,1,np.nan)
        all_memo_detail['combined_memo_type_and_data']= all_memo_detail['converted_memos'].apply(lambda x: x['MemoType']+'  '+x['MemoData'])
        output_frame = all_memo_detail[['datetime','pft_transaction','pft_directional_value',
                                        'combined_memo_type_and_data','pft_absolute_value']].groupby('datetime').first()
        output_frame.reset_index(inplace=True)
        output_frame['raw_date']=pd.to_datetime(output_frame['datetime'].apply(lambda x: x.date()))
        daily_grouped_output_frame = output_frame[['pft_transaction','pft_directional_value',
                    'combined_memo_type_and_data','pft_absolute_value','raw_date']].groupby('raw_date').sum()
        return {'daily_grouped_summary':daily_grouped_output_frame, 'raw_summary':output_frame}
    
    # TODO: Replace with get_latest_outgoing_context_doc_link (reference pftpyclient/task_manager/basic_tasks.py)
    def get_most_recent_google_doc_for_user(self, account_memo_detail_df, address):
        """ This function takes a memo detail df and a classic address and outputs
        the associated google doc
        
        EXAMPLE:
        address = 'r3UHe45BzAVB3ENd21X9LeQngr4ofRJo5n'
        all_account_info = self.get_memo_detail_df_for_account(account_address='r3UHe45BzAVB3ENd21X9LeQngr4ofRJo5n',transaction_limit=5000,
            pft_only=True) 
        """ 
        op = ''
        try:
            op=list(account_memo_detail_df[(account_memo_detail_df['converted_memos'].apply(lambda x: 'google_doc' in str(x))) & 
                    (account_memo_detail_df['account']==address)]['converted_memos'].tail(1))[0]['MemoData']
        except:
            print('No Google Doc Associated with Address')
            pass
        return op
    
    # TODO: Replace with is_valid_id (reference pftpyclient/task_manager/basic_tasks.py)
    def determine_if_map_is_task_id(self,memo_dict):
        """ task ID detection 
        """
        full_memo_string = str(memo_dict)
        task_id_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}(?:__[A-Z0-9]{4})?)')
        has_task_id = False
        if re.search(task_id_pattern, full_memo_string):
            return True
        
        if has_task_id:
            return True
        return False
    
    # TODO: Replace with sync_tasks (reference pftpyclient/task_manager/basic_tasks.py)
    def convert_all_account_info_into_simplified_task_frame(self, account_memo_detail_df):
        
        """ This takes all the Post Fiat Tasks and outputs them into a simplified
                dataframe of task information with embedded classifications
                Runs on all_account_info generated by
                all_account_info =self.get_memo_detail_df_for_account(account_address=self.user_wallet.classic_address,
                    transaction_limit=5000)
                """ 
        all_account_info = account_memo_detail_df
        simplified_task_frame = all_account_info[all_account_info['converted_memos'].apply(lambda x: 
                                                                self.determine_if_map_is_task_id(x))].copy()
        def add_field_to_map(xmap, field, field_value):
            xmap[field] = field_value
            return xmap
        
        for xfield in ['hash','datetime']:
            simplified_task_frame['converted_memos'] = simplified_task_frame.apply(lambda x: add_field_to_map(x['converted_memos'],
                xfield,x[xfield]),1)
        core_task_df = pd.DataFrame(list(simplified_task_frame['converted_memos'])).copy()
        core_task_df['task_type']=core_task_df['MemoData'].apply(lambda x: self.classify_task_string(x))
        return core_task_df
    
    # TODO: Replace with sync_tasks (reference pftpyclient/task_manager/basic_tasks.py)
    def convert_all_account_info_into_outstanding_task_df(self, account_memo_detail_df):
        """ This reduces all account info into a simplified dataframe of proposed 
        and accepted tasks """ 
        all_account_info = account_memo_detail_df
        all_account_info= all_account_info.sort_values('datetime')
        account_memo_detail_df= account_memo_detail_df.sort_values('datetime')
        task_frame = self.convert_all_account_info_into_simplified_task_frame(account_memo_detail_df=account_memo_detail_df)
        task_frame['task_id']=task_frame['MemoType']
        task_frame['full_output']=task_frame['MemoData']
        task_frame['user_account']=task_frame['MemoFormat']
        task_type_map = task_frame.groupby('task_id').last()[['task_type']].copy()
        task_id_to_proposal = task_frame[task_frame['task_type']
        =='PROPOSAL'].groupby('task_id').first()['full_output']
        task_id_to_acceptance = task_frame[task_frame['task_type']
        =='ACCEPTANCE'].groupby('task_id').first()['full_output']
        acceptance_frame = pd.concat([task_id_to_proposal,task_id_to_acceptance],axis=1)
        acceptance_frame.columns=['proposal','acceptance_raw']
        acceptance_frame['acceptance']=acceptance_frame['acceptance_raw'].apply(lambda x: str(x).replace('ACCEPTANCE REASON ___ ',
                                                                                                         '').replace('nan',''))
        acceptance_frame['proposal']=acceptance_frame['proposal'].apply(lambda x: str(x).replace('PROPOSED PF ___ ',
                                                                                                         '').replace('nan',''))
        raw_proposals_and_acceptances = acceptance_frame[['proposal','acceptance']].copy()
        proposed_or_accepted_only = list(task_type_map[(task_type_map['task_type']=='ACCEPTANCE')|
        (task_type_map['task_type']=='PROPOSAL')].index)
        op= raw_proposals_and_acceptances[raw_proposals_and_acceptances.index.get_level_values(0).isin(proposed_or_accepted_only)]
        return op

    def establish_post_fiat_tx_cache_as_hash_unique(self):
        dbconnx = self.db_connection_manager.spawn_sqlalchemy_db_connection_for_user(user_name=self.node_name)
        
        with dbconnx.connect() as connection:
            # Check if the table exists
            table_exists = connection.execute(sqlalchemy.text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'postfiat_tx_cache'
                );
            """)).scalar()
            
            if not table_exists:
                # Create the table if it doesn't exist
                connection.execute(sqlalchemy.text("""
                    CREATE TABLE postfiat_tx_cache (
                        hash VARCHAR(255) PRIMARY KEY,
                        -- Add other columns as needed, for example:
                        account VARCHAR(255),
                        destination VARCHAR(255),
                        amount DECIMAL(20, 8),
                        memo TEXT,
                        timestamp TIMESTAMP
                    );
                """))
                print("Table 'postfiat_tx_cache' created.")
            
            # Add unique constraint on hash if it doesn't exist
            constraint_exists = connection.execute(sqlalchemy.text("""
                SELECT constraint_name
                FROM information_schema.table_constraints
                WHERE table_name = 'postfiat_tx_cache' 
                AND constraint_type = 'UNIQUE' 
                AND constraint_name = 'unique_hash';
            """)).fetchone()
            
            if constraint_exists is None:
                connection.execute(sqlalchemy.text("""
                    ALTER TABLE postfiat_tx_cache
                    ADD CONSTRAINT unique_hash UNIQUE (hash);
                """))
                print("Unique constraint added to 'hash' column.")
            
            connection.commit()

        dbconnx.dispose()
    def generate_postgres_writable_df_for_address(self,account_address = 'r3UHe45BzAVB3ENd21X9LeQngr4ofRJo5n',public=True):
        # Fetch transaction history and prepare DataFrame
        tx_hist = self.get_account_transactions__exhaustive(account_address=account_address, public=public)
        if len(tx_hist)==0:
            #print('no tx pulled')
            #print()
            2+2
        if len(tx_hist)>0:
            full_transaction_history = pd.DataFrame(
                tx_hist
            )
            tx_json_extractions = ['Account', 'DeliverMax', 'Destination', 
                                   'Fee', 'Flags', 'LastLedgerSequence', 
                                   'Sequence', 'SigningPubKey', 'TransactionType', 
                                   'TxnSignature', 'date', 'ledger_index', 'Memos']
            
            def extract_field(json_data, field):
                try:
                    value = json_data.get(field)
                    if isinstance(value, dict):
                        return str(value)  # Convert dict to string
                    return value
                except AttributeError:
                    return None
            for field in tx_json_extractions:
                full_transaction_history[field.lower()] = full_transaction_history['tx_json'].apply(lambda x: extract_field(x, field))
            def process_memos(memos):
                """
                Process the memos column to prepare it for PostgreSQL storage.
                :param memos: List of memo dictionaries or None
                :return: JSON string representation of memos or None
                """
                if memos is None:
                    return None
                # Ensure memos is a list
                if not isinstance(memos, list):
                    memos = [memos]
                # Extract only the 'Memo' part from each dictionary
                processed_memos = [memo.get('Memo', memo) for memo in memos]
                # Convert to JSON string
                return json.dumps(processed_memos)
            # Apply the function to the 'memos' column
            full_transaction_history['memos'] = full_transaction_history['memos'].apply(process_memos)
            full_transaction_history['meta'] = full_transaction_history['meta'].apply(json.dumps)
            full_transaction_history['tx_json'] = full_transaction_history['tx_json'].apply(json.dumps)
            return full_transaction_history

    def write_full_transaction_history_for_account(self, account_address, public):
        # Fetch transaction history and prepare DataFrame
        tx_hist = self.generate_postgres_writable_df_for_address(account_address=account_address, public=public)
        dbconnx = self.db_connection_manager.spawn_sqlalchemy_db_connection_for_user(user_name=self.node_name)
        
        if tx_hist is not None:
            try:
                with dbconnx.begin() as conn:
                    total_rows_inserted = 0
                    for start in range(0, len(tx_hist), 100):
                        chunk = tx_hist.iloc[start:start + 100]
                        
                        # Fetch existing hashes from the database to avoid duplicates
                        existing_hashes = pd.read_sql_query(
                            "SELECT hash FROM postfiat_tx_cache WHERE hash IN %(hashes)s",
                            conn,
                            params={"hashes": tuple(chunk['hash'].tolist())}
                        )['hash'].tolist()
                        
                        # Filter out rows with existing hashes
                        new_rows = chunk[~chunk['hash'].isin(existing_hashes)]
                        
                        if not new_rows.empty:
                            rows_inserted = len(new_rows)
                            new_rows.to_sql('postfiat_tx_cache', conn, if_exists='append', index=False)
                            total_rows_inserted += rows_inserted
                            print(f"Inserted {rows_inserted} new rows.")
                    
                    print(f"Total rows inserted: {total_rows_inserted}")
            
            except sqlalchemy.exc.InternalError as e:
                if "current transaction is aborted" in str(e):
                    print("Transaction aborted. Attempting to reset...")
                    with dbconnx.connect() as connection:
                        connection.execute(sqlalchemy.text("ROLLBACK"))
                    print("Transaction reset. Please try the operation again.")
                else:
                    print(f"An error occurred: {e}")
            
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
            
            finally:
                dbconnx.dispose()
        else:
            print("No transaction history to write.")

    def write_all_postfiat_holder_transaction_history(self,public=True):
        """ This writes all the transaction history. if public is True then it goes through full history """ 
        holder_df = self.output_post_fiat_holder_df()
        all_post_fiat_holders = list(holder_df['account'].unique())
        for xholder in all_post_fiat_holders:
            self.write_full_transaction_history_for_account(account_address=xholder, public=public)

    def run_transaction_history_updates(self):
        """
        Runs transaction history updates in separate threads.
        This function creates two threads:
        1. Updates with public=True every 60 minutes
        2. Updates with public=False every 30 seconds
        """
        def update_public():
            while True:
                self.write_all_postfiat_holder_transaction_history(public=True)
                time.sleep(3600)  # 60 minutes
        def update_private():
            while True:
                self.write_all_postfiat_holder_transaction_history(public=False)
                time.sleep(30)  # 2 minutes
        public_thread = threading.Thread(target=update_public)
        private_thread = threading.Thread(target=update_private)
        public_thread.daemon = True
        private_thread.daemon = True
        public_thread.start()
        private_thread.start()

    def get_all_cached_transactions_related_to_account(self,account_address = 'r4sRyacXpbh4HbagmgfoQq8Q3j8ZJzbZ1J'):

        dbconnx = self.db_connection_manager.spawn_sqlalchemy_db_connection_for_user(user_name
                                                                                     =self.node_name)
        query = f"""
        SELECT * FROM postfiat_tx_cache
        WHERE account = '{account_address}' OR destination = '{account_address}'
        """
        full_transaction_history = pd.read_sql(query, dbconnx)
        full_transaction_history['meta']= full_transaction_history['meta'].apply(lambda x: json.loads(x))
        full_transaction_history['tx_json']= full_transaction_history['tx_json'].apply(lambda x: json.loads(x))
        return full_transaction_history

    def get_all_transactions_for_active_wallets(self):
        """ This gets all the transactions for active post fiat wallets""" 
        full_balance_df = self.output_post_fiat_holder_df()
        all_active_foundation_users = full_balance_df[full_balance_df['balance'].astype(float)<=-2000].copy()
        
        # Get unique wallet addresses from the dataframe
        all_wallets = list(all_active_foundation_users['account'].unique())
        
        # Create database connection
        dbconnx = self.db_connection_manager.spawn_sqlalchemy_db_connection_for_user(
            user_name=self.node_name
        )
        
        # Format the wallet addresses for the IN clause
        wallet_list = "'" + "','".join(all_wallets) + "'"
        
        # Create query
        query = f"""
            SELECT * 
            FROM postfiat_tx_cache
            WHERE account IN ({wallet_list})
            OR destination IN ({wallet_list})
        """
        
        # Execute query using pandas read_sql
        full_transaction_history = pd.read_sql(query, dbconnx)
        full_transaction_history['meta']= full_transaction_history['meta'].apply(lambda x: json.loads(x))
        full_transaction_history['tx_json']= full_transaction_history['tx_json'].apply(lambda x: json.loads(x))
        return full_transaction_history

    def get_all_account_pft_memo_data(self):
        """ This gets all pft memo data for computation of leaderboard  """ 
        all_transactions = self.get_all_transactions_for_active_wallets()
        validated_tx=all_transactions
        pft_only=True
        validated_tx['has_memos'] = validated_tx['tx_json'].apply(lambda x: 'Memos' in x.keys())
        live_memo_tx = validated_tx[validated_tx['has_memos'] == True].copy()
        live_memo_tx['main_memo_data']=live_memo_tx['tx_json'].apply(lambda x: x['Memos'][0]['Memo'])
        live_memo_tx['converted_memos']=live_memo_tx['main_memo_data'].apply(lambda x: 
                                                                            self.convert_memo_dict__generic(x))
        #live_memo_tx['message_type']=np.where(live_memo_tx['destination']==account_address, 'INCOMING','OUTGOING')
        live_memo_tx['datetime'] = pd.to_datetime(live_memo_tx['close_time_iso']).dt.tz_localize(None)
        if pft_only:
            live_memo_tx= live_memo_tx[live_memo_tx['tx_json'].apply(lambda x: self.pft_issuer in str(x))].copy()
        
        #live_memo_tx['unique_key']=live_memo_tx['reference_account']+'__'+live_memo_tx['hash']
        def try_get_pft_absolute_amount(x):
            try:
                return x['DeliverMax']['value']
            except:
                return 0
        def try_get_memo_info(x,info):
            try:
                return x[info]
            except:
                return ''
        live_memo_tx['pft_absolute_amount']=live_memo_tx['tx_json'].apply(lambda x: try_get_pft_absolute_amount(x)).astype(float)
        live_memo_tx['memo_format']=live_memo_tx['converted_memos'].apply(lambda x: try_get_memo_info(x,"MemoFormat"))
        live_memo_tx['memo_type']= live_memo_tx['converted_memos'].apply(lambda x: try_get_memo_info(x,"MemoType"))
        live_memo_tx['memo_data']=live_memo_tx['converted_memos'].apply(lambda x: try_get_memo_info(x,"MemoData"))
        #live_memo_tx['pft_sign']= np.where(live_memo_tx['message_type'] =='INCOMING',1,-1)
        #live_memo_tx['directional_pft'] = live_memo_tx['pft_sign']*live_memo_tx['pft_absolute_amount']
        live_memo_tx['simple_date']=pd.to_datetime(live_memo_tx['datetime'].apply(lambda x: x.strftime('%Y-%m-%d')))
        return live_memo_tx


    def format_outstanding_tasks(self,outstanding_task_df):
        """
        Convert outstanding_task_df to a more legible string format for AI tools.
        
        :param outstanding_task_df: DataFrame containing outstanding tasks
        :return: Formatted string representation of the tasks
        
        outstanding_task_df = self.convert_all_account_info_into_outstanding_task_df(account_memo_detail_df=all_account_info)
        """
        formatted_tasks = []
        for idx, row in outstanding_task_df.iterrows():
            task_str = f"Task ID: {idx}\n"
            task_str += f"Proposal: {row['proposal']}\n"
            task_str += f"Acceptance: {row['acceptance']}\n"
            task_str += "-" * 50  # Separator
            formatted_tasks.append(task_str)
        
        formatted_task_string =  "\n".join(formatted_tasks)
        output_string="""OUTSTANDING TASKS
    """+formatted_task_string
        return output_string

    def process_account_memo_details_into_reward_summary_map(self, all_account_info):
        """ Takes the all_account_info and makes it into a map that contains both timeseries output as well as 
        a task completion history log """ 
        reward_responses = all_account_info[all_account_info['directional_pft']>0].copy()
        specific_rewards = reward_responses[reward_responses.memo_data.apply(lambda x: "REWARD RESPONSE" in x)]
        reward_sum = specific_rewards [['directional_pft', 'simple_date']].groupby('simple_date').sum()
        today = pd.Timestamp.today().normalize()
        date_range = pd.date_range(start=reward_sum.index.min(), end=today, freq='D')
        # Reindex the reward_sum DataFrame with the new date range
        extended_reward_sum = reward_sum.reindex(date_range, fill_value=0)
        # Resample and fill missing values
        final_result = extended_reward_sum.resample('D').last().fillna(0)
        final_result['weekly_total']= final_result.rolling(7).sum()
        pft_generation_ts = final_result.resample('W').last()[['weekly_total']]
        pft_generation_ts.index.name = 'date'
        specific_reward_slice = specific_rewards[['memo_data','directional_pft',
                                'datetime','memo_type']].sort_values('datetime').copy()
        task_request = all_account_info[all_account_info['memo_data'].apply(lambda x:'REQUEST_POST_FIAT' 
                                                             in x)].groupby('memo_type').first()['memo_data']
        
        task_proposal = all_account_info[all_account_info['memo_data'].apply(lambda x:('PROPOSED' in x)|('..' in x))].groupby('memo_type').first()['memo_data']
        specific_reward_slice['request']=specific_reward_slice['memo_type'].map(task_request)
        specific_reward_slice['proposal']=specific_reward_slice['memo_type'].map(task_proposal)
        specific_reward_slice['request']=specific_reward_slice['request'].fillna('No Request String')
        return {'reward_ts':pft_generation_ts, 'reward_summaries': specific_reward_slice}

    def format_reward_summary(self, reward_summary_df):
        """
        Convert reward summary dataframe into a human-readable string.
        :param reward_summary_df: DataFrame containing reward summary information
        :return: Formatted string representation of the rewards
        """
        formatted_rewards = []
        for _, row in reward_summary_df.iterrows():
            reward_str = f"Date: {row['datetime']}\n"
            reward_str += f"Request: {row['request']}\n"
            reward_str += f"Proposal: {row['proposal']}\n"
            reward_str += f"Reward: {row['directional_pft']} PFT\n"
            reward_str += f"Response: {row['memo_data'].replace('REWARD RESPONSE __ ', '')}\n"
            reward_str += "-" * 50  # Separator
            formatted_rewards.append(reward_str)
        
        output_string = "REWARD SUMMARY\n\n" + "\n".join(formatted_rewards)
        return output_string

    def get_google_doc_text(self,share_link):
        # Extract the document ID from the share link
        doc_id = share_link.split('/')[5]
    
        # Construct the Google Docs API URL
        url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    
        # Send a GET request to the API URL
        response = requests.get(url)
    
        # Check if the request was successful
        if response.status_code == 200:
            # Return the plain text content of the document
            return response.text
        else:
            # Return an error message if the request was unsuccessful
            return f"Failed to retrieve the document. Status code: {response.status_code}"

    # TODO: Replace with retrieve_xrp_address_from_google_doc (reference pftpyclient/task_manager/basic_tasks.py)
    # TODO: Add separate call to get_xrp_balance where relevant
    def check_if_there_is_funded_account_at_front_of_google_doc(self, google_url):
        """
        Checks if there is a balance bearing XRP account address at the front of the google document 
        This is required for the user 

        Returns the balance in XRP drops 
        EXAMPLE
        google_url = 'https://docs.google.com/document/d/1MwO8kHny7MtU0LgKsFTBqamfuUad0UXNet1wr59iRCA/edit'
        """
        balance = 0
        try:
            wallet_at_front_of_doc =self.get_google_doc_text(google_url).split('\ufeff')[-1:][0][0:34]
            balance = self.get_account_xrp_balance(wallet_at_front_of_doc)
        except:
            pass
        return balance

    def format_recent_chunk_messages(self, message_df):
        """
        Format the last fifteen messages into a singular text block.
        
        Args:
        df (pandas.DataFrame): DataFrame containing 'datetime', 'cleaned_message', and 'message_type' columns.
        
        Returns:
        str: Formatted text block of the last fifteen messages.
        """
        df= message_df
        formatted_messages = []
        for _, row in df.iterrows():
            formatted_message = f"[{row['datetime']}] ({row['message_type']}): {row['cleaned_message']}"
            formatted_messages.append(formatted_message)
        
        return "\n".join(formatted_messages)

    # TODO: Leverage self.tasks
    def generate_refusal_frame(self, all_account_info):
        """ Takes all account info and transmutes into all historical task refusals and reasons 
        
        account_address='r3UHe45BzAVB3ENd21X9LeQngr4ofRJo5n'
        all_account_info =self.get_memo_detail_df_for_account(account_address=account_address)
        """
        refusal_frame_constructor = all_account_info[all_account_info['memo_data'].apply(lambda x: "REFUSAL" in x)][['memo_data',
                                                                                    'memo_type']].groupby('memo_type').first().copy()
        initial_proposal= all_account_info[all_account_info['memo_data'].apply(lambda x: (' .. ' in x)|('PROPOSAL' in x))].groupby('memo_type').first()[['memo_data']]
        initial_proposal.columns=['proposal']
        refusal_frame_constructor['proposal']=initial_proposal
        return refusal_frame_constructor

    def format_refusal_frame(self, refusal_frame_constructor):
        """
        Format the refusal frame constructor into a nicely formatted string.
        
        :param refusal_frame_constructor: DataFrame containing refusal data
        :return: Formatted string representation of the refusal frame
        """
        formatted_string = ""
        for idx, row in refusal_frame_constructor.tail(5).iterrows():
            formatted_string += f"Task ID: {idx}\n"
            formatted_string += f"Refusal Reason: {row['memo_data']}\n"
            formatted_string += f"Proposal: {row['proposal']}\n"
            formatted_string += "-" * 50 + "\n"
        
        return formatted_string

    def output_user_compressed_memo_history(self, account_address,num_messages):
        """ this pulls down all compressed memos sent from a user 
            account address: r3UHe45BzAVB3ENd21X9LeQngr4ofRJo5n
            num_messages: 20
        """ 
        user_memo_compression_string = self.get_all_account_compressed_messages(account_address=account_address)[['cleaned_message',
                                                                                                    'datetime']].tail(num_messages).sort_values('datetime').set_index('datetime')['cleaned_message'].to_json()
        return user_memo_compression_string

    def get_full_user_context_string(self,account_address):
        """ the following function gets all the core elements of a users post fiat interactions including
        their outstanding tasks their completed tasks, their context document as well as their post fiat chunk message dialogue

        EXAMPLE
        account_address='r3UHe45BzAVB3ENd21X9LeQngr4ofRJo5n'
        """ 
        print('N Version')
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        all_account_info =self.get_memo_detail_df_for_account(account_address=account_address).sort_values('datetime')
        core_element_outstanding_task_df =''
        try:
            outstanding_task_df = self.convert_all_account_info_into_outstanding_task_df(account_memo_detail_df=all_account_info)
            all_account_info['simple_date']= all_account_info['datetime'].apply(lambda x: pd.to_datetime(x.date()))
            core_element_outstanding_task_df = self.format_outstanding_tasks(outstanding_task_df=outstanding_task_df)
        except:
            print('FAILED OUTSTANDING TASK GEN')
            pass
        
        core_element__refusal_frame = ''
        try:
            refusal_frame_constructor = self.generate_refusal_frame(all_account_info=all_account_info).tail(6)
            core_element__refusal_frame = self.format_refusal_frame(refusal_frame_constructor=refusal_frame_constructor)
        except:
            print('FAILED REFUSAL FRAME GEN')
            pass

        core_element__last_10_rewards=''
        core_element_post_fiat_weekly_gen=''
        try:
            reward_map = self.process_account_memo_details_into_reward_summary_map(all_account_info=all_account_info)
            specific_rewards = reward_map['reward_summaries']
            core_element__last_10_rewards = self.format_reward_summary(specific_rewards.tail(10))
            core_element_post_fiat_weekly_gen= reward_map['reward_ts']['weekly_total'].to_string()
        except:
            print('FAILED REWARD GEN')
            pass
        
        core_element__google_doc_text = ''
        try:
            google_url = list(all_account_info[all_account_info['memo_type'].apply(lambda x: 'google_doc' in x)]['memo_data'])[0]
            core_element__google_doc_text= self.get_google_doc_text(google_url)
        except:
            print('FAILED GOOGLE DOC GEN')
            pass
        
        core_element__chunk_messages = ''
        try:
            core_element__chunk_messages = self.output_user_compressed_memo_history(account_address=account_address, 
                                                                                    num_messages=20)
        except:
            pass
        final_context_string = f"""The current date is {current_date}
        
        USERS CORE OUTSTANDING TASKS ARE AS FOLLOWS:
        <OUTSTANDING TASKS START HERE>
        {core_element_outstanding_task_df}
        <OUTSTANDING TASKS END HERE>

        THESE ARE TASKS USER HAS RECENTLY REFUSED ALONG WITH REASONS
        <REFUSED TASKS START HERE>
        {core_element__refusal_frame}
        <REFUSED TASKS END HERE>
        
        THESE ARE TASKS USER HAS RECENTLY COMPLETED ALONG WITH REWARDS
        <REWARDED TASKS START HERE>
        {core_element__last_10_rewards}
        <REWARDED TASKS END HERE>
        
        HERE IS THE USERS RECENT POST FIAT OUTPUT SUMMED AS A WEEKLY TIMESERIES
        <POST FIAT GENERATION STARTS HERE>
        {core_element_post_fiat_weekly_gen}
        <POST FIAT GENEREATION ENDS HERE>
        
        THE FOLLOWING IS THE PRIMARY CONTENT OF THE USERS CONTEXT DOCUMENT AND PLANNING
        <USER CONTEXT AND PLANNING STARTS HERE>
        {core_element__google_doc_text}
        <USER CONTEXT AND PLANNING ENDS HERE>
        
        THE FOLLOWING ARE THE RECENT LONG FORM DIALOGUES WITH THE USER
        <USER LONG FORM DIALOGUE>
        {core_element__chunk_messages}
        <USER LONG FORM DIALGOUE ENDS>
        """
        return final_context_string

    def create_xrp_wallet(self):
        test_wallet = Wallet.create()
        classic_address= test_wallet.classic_address
        wallet_seed = test_wallet.seed
        output_string = f"""Wallet Address: {classic_address}
Wallet Secret: {wallet_seed}
        
STORE YOUR WALLET SECRET IN AN OFFLINE PREFERABLY NON DIGITAL LOCATION
THIS MESSAGE WILL AUTO DELETE IN 60 SECONDS
"""
        return output_string

    def generate_basic_balance_info_string_for_account_address(self, account_address = 'r3UHe45BzAVB3ENd21X9LeQngr4ofRJo5n'):
        try:
            all_account_info =self.get_memo_detail_df_for_account(account_address=account_address)
        except:
            pass
        monthly_pft_reward_avg=0
        weekly_pft_reward_avg=0
        try:
            reward_ts = self.process_account_memo_details_into_reward_summary_map(all_account_info=all_account_info)
            monthly_pft_reward_avg = list(reward_ts['reward_ts'].tail(4).mean())[0]
            weekly_pft_reward_avg = list(reward_ts['reward_ts'].tail(1).mean())[0]
        except:
            pass
        number_of_transactions =0
        try:
            number_of_transactions = len(all_account_info['memo_type'])
        except:
            pass
        user_name=''
        try:
            user_name = list(all_account_info[all_account_info['message_type']=='OUTGOING']['memo_format'].mode())[0]
        except:
            pass
        
        client = JsonRpcClient(self.local_rippled_url)
        
        # Get XRP balance
        acct_info = AccountInfo(
            account=account_address,
            ledger_index="validated"
        )
        response = client.request(acct_info)
        xrp_balance=0
        try:
            xrp_balance = int(response.result['account_data']['Balance'])/1_000_000
        except:
            pass
        pft_balance= 0 
        try:
            account_lines = AccountLines(
                account=account_address,
                ledger_index="validated"
            )
            account_line_response = client.request(account_lines)
            pft_balance = [i for i in account_line_response.result['lines'] if i['account']=='rnQUEEg8yyjrwk9FhyXpKavHyCRJM9BDMW'][0]['balance']
        except:
            pass
        account_info_string =f"""ACCOUNT INFO for  {account_address}
LIKELY ALIAS:     {user_name}
XRP BALANCE:      {xrp_balance}
PFT BALANCE:      {pft_balance}
NUM PFT MEMO TX: {number_of_transactions}
PFT MONTHLY AVG:  {monthly_pft_reward_avg}
PFT WEEKLY AVG:   {weekly_pft_reward_avg}
"""
        return account_info_string

    # TODO: Refactor, using get_xrp_balance as reference (pftpyclient/task_manager/basic_tasks.py)
    def get_account_xrp_balance(self, account_address):
        """ 
        Example
        account_address = 'r3UHe45BzAVB3ENd21X9LeQngr4ofRJo5n'
        """
        client = JsonRpcClient(self.local_rippled_url)
        
        # Get XRP balance
        acct_info = AccountInfo(
            account=account_address,
            ledger_index="validated"
        )
        response = client.request(acct_info)
        xrp_balance=0
        try:
            xrp_balance = int(response.result['account_data']['Balance'])/1_000_000
        except:
            pass
        return xrp_balance

    # TODO: Refactor using get_verification_df as reference (pftpyclient/task_manager/basic_tasks.py)
    def convert_all_account_info_into_outstanding_verification_df(self,account_memo_detail_df):
        """ takes the outstanding account data and converts into outstanding memos """ 
        all_memos = account_memo_detail_df.copy()
        most_recent_memos = all_memos.sort_values('datetime').groupby('memo_type').last().copy()
        task_id_to_original_task_map = all_memos[all_memos['memo_data'].apply(lambda x: ('..' in x) | ('PROPOS' in x))][['memo_data','memo_type','memo_format']].groupby('memo_type').first()['memo_data']
        verification_requirements = most_recent_memos[most_recent_memos['memo_data'].apply(lambda x: 'VERIFICATION PROMPT ' in x)][['memo_data','memo_format']].reset_index().copy()
        verification_requirements['original_task']=verification_requirements['memo_type'].map(task_id_to_original_task_map)
        return verification_requirements

    def format_outstanding_verification_df(self, verification_requirements):
        """
        Format the verification_requirements dataframe into a string.

        Args:
        verification_requirements (pd.DataFrame): DataFrame containing columns 
                                                'memo_type', 'memo_data', 'memo_format', and 'original_task'

        Returns:
        str: Formatted string of verification requirements
        """
        formatted_output = "VERIFICATION REQUIREMENTS\n"
        for _, row in verification_requirements.iterrows():
            formatted_output += f"Task ID: {row['memo_type']}\n"
            formatted_output += f"Verification Prompt: {row['memo_data']}\n"
            formatted_output += f"Original Task: {row['original_task']}\n"
            formatted_output += "-" * 50 + "\n"
        return formatted_output

    def create_full_outstanding_pft_string(self, account_address='r3UHe45BzAVB3ENd21X9LeQngr4ofRJo5n'):
        """ This takes in an account address and outputs the current state of its outstanding tasks
        """ 
        all_memos = self.get_memo_detail_df_for_account(account_address=account_address,
                                            pft_only=True).sort_values('datetime')
        outstanding_task_df = self.convert_all_account_info_into_outstanding_task_df(account_memo_detail_df=all_memos)
        task_string = self.format_outstanding_tasks(outstanding_task_df)
        verification_df = self.convert_all_account_info_into_outstanding_verification_df(account_memo_detail_df=all_memos)
        verification_string = self.format_outstanding_verification_df(verification_requirements=verification_df)
        full_postfiat_outstanding_string=f"""{task_string}
        {verification_string}"""
        print(full_postfiat_outstanding_string)
        return full_postfiat_outstanding_string

    def extract_transaction_info_from_response_object(self, response):
        """
        Extract key information from an XRPL transaction response object.

        Args:
        response (Response): The XRPL transaction response object.

        Returns:
        dict: A dictionary containing extracted transaction information.
        """
        result = response.result
        tx_json = result['tx_json']
        
        # Extract required information
        transaction_info = {
            'time': result['close_time_iso'],
            'amount': tx_json['DeliverMax']['value'],
            'currency': tx_json['DeliverMax']['currency'],
            'send_address': tx_json['Account'],
            'destination_address': tx_json['Destination'],
            'status': result['meta']['TransactionResult'],
            'hash': result['hash'],
            'xrpl_explorer_url': f"https://livenet.xrpl.org/transactions/{result['hash']}/detailed"
        }
        clean_string = (f"Transaction of {transaction_info['amount']} {transaction_info['currency']} "
                        f"from {transaction_info['send_address']} to {transaction_info['destination_address']} "
                        f"on {transaction_info['time']}. Status: {transaction_info['status']}. "
                        f"Explorer: {transaction_info['xrpl_explorer_url']}")
        transaction_info['clean_string']= clean_string
        return transaction_info

    def extract_transaction_info_from_response_object__standard_xrp(self, response):
        """
        Extract key information from an XRPL transaction response object.
        
        Args:
        response (Response): The XRPL transaction response object.
        
        Returns:
        dict: A dictionary containing extracted transaction information.
        """
        transaction_info = {}
        
        try:
            result = response.result if hasattr(response, 'result') else response
            
            transaction_info['hash'] = result.get('hash')
            transaction_info['xrpl_explorer_url'] = f"https://livenet.xrpl.org/transactions/{transaction_info['hash']}/detailed"
            
            tx_json = result.get('tx_json', {})
            transaction_info['send_address'] = tx_json.get('Account')
            transaction_info['destination_address'] = tx_json.get('Destination')
            
            # Handle different amount formats
            if 'DeliverMax' in tx_json:
                transaction_info['amount'] = str(int(tx_json['DeliverMax']) / 1000000)  # Convert drops to XRP
                transaction_info['currency'] = 'XRP'
            elif 'Amount' in tx_json:
                if isinstance(tx_json['Amount'], dict):
                    transaction_info['amount'] = tx_json['Amount'].get('value')
                    transaction_info['currency'] = tx_json['Amount'].get('currency')
                else:
                    transaction_info['amount'] = str(int(tx_json['Amount']) / 1000000)  # Convert drops to XRP
                    transaction_info['currency'] = 'XRP'
            
            transaction_info['time'] = result.get('close_time_iso') or tx_json.get('date')
            transaction_info['status'] = result.get('meta', {}).get('TransactionResult') or result.get('engine_result')
            
            # Create clean string
            clean_string = (f"Transaction of {transaction_info.get('amount', 'unknown amount')} "
                            f"{transaction_info.get('currency', 'XRP')} "
                            f"from {transaction_info.get('send_address', 'unknown sender')} "
                            f"to {transaction_info.get('destination_address', 'unknown recipient')} "
                            f"on {transaction_info.get('time', 'unknown time')}. "
                            f"Status: {transaction_info.get('status', 'unknown')}. "
                            f"Explorer: {transaction_info['xrpl_explorer_url']}")
            transaction_info['clean_string'] = clean_string
            
        except Exception as e:
            transaction_info['error'] = str(e)
            transaction_info['clean_string'] = f"Error extracting transaction info: {str(e)}"
        
        return transaction_info

    def discord_send_pft_with_info_from_seed(self, destination_address, seed, user_name, message, amount):
        """
        For use in the discord tooling. pass in users user name 
        destination_address = 'rKZDcpzRE5hxPUvTQ9S3y2aLBUUTECr1vN'
        seed = 's_____x'
        message = 'this is the second test of a discord message'
        amount = 2
        """
        wallet = self.spawn_user_wallet_from_seed(seed)
        memo = self.construct_standardized_xrpl_memo(memo_data=message, memo_type='DISCORD_SERVER', memo_format=user_name)
        action_response = self.send_PFT_with_info(sending_wallet=wallet,
            amount=amount,
            memo=memo,
            destination_address=destination_address,
            url=None)
        printable_string = self.extract_transaction_info_from_response_object(action_response)['clean_string']
        return printable_string

    def generate_trust_line_to_pft_token(self, wallet_seed):
        """ Note this transaction consumes XRP to create a trust
        line for the PFT Token so the holder DF should be checked 
        before this is run
        """ 
        
        #wallet_to_link =self.user_wallet
        wallet_to_link = xrpl.wallet.Wallet.from_seed(wallet_seed)
        client = xrpl.clients.JsonRpcClient(self.local_rippled_url)
        #currency_code = "PFT"
        trust_set_tx = xrpl.models.transactions.TrustSet(
                        account=wallet_to_link.classic_address,
                    limit_amount=xrpl.models.amounts.issued_currency_amount.IssuedCurrencyAmount(
                            currency="PFT",
                            issuer=self.pft_issuer,
                            value='100000000',  # Large limit, arbitrarily chosen
                        )
                    )
        print("Creating trust line from chosen seed to issuer...")
        
        response = xrpl.transaction.submit_and_wait(trust_set_tx, client, wallet_to_link)
        return response

    def get_recent_messages_for_account_address(self,wallet_address='r3UHe45BzAVB3ENd21X9LeQngr4ofRJo5n'): 
        incoming_message = ''
        outgoing_message = ''
        try:

            all_wallet_transactions = self.get_memo_detail_df_for_account(wallet_address).copy().sort_values('datetime')
            incoming_message = all_wallet_transactions[all_wallet_transactions['message_type']=='INCOMING'].tail(1).transpose()
            outgoing_message = all_wallet_transactions[all_wallet_transactions['message_type']=='OUTGOING'].tail(1).transpose()
            def format_transaction_message(transaction):
                """
                Format a transaction message with specified elements.
                
                Args:
                transaction (pd.Series): A single transaction from the DataFrame.
                
                Returns:
                str: Formatted transaction message.
                """
                return (f"Task ID: {transaction['memo_type']}\n"
                        f"Memo: {transaction['memo_data']}\n"
                        f"PFT Amount: {transaction['directional_pft']}\n"
                        f"Datetime: {transaction['datetime']}\n"
                        f"XRPL Explorer: https://livenet.xrpl.org/transactions/{transaction['hash']}/detailed")
            
            # Format incoming message
            incoming_message = format_transaction_message(all_wallet_transactions[all_wallet_transactions['message_type']=='INCOMING'].tail(1).iloc[0])
            
            # Format outgoing message
            outgoing_message = format_transaction_message(all_wallet_transactions[all_wallet_transactions['message_type']=='OUTGOING'].tail(1).iloc[0])
        except:
            pass
        # Create a dictionary with the formatted messages
        transaction_messages = {
            'incoming_message': incoming_message,
            'outgoing_message': outgoing_message
        }
        return transaction_messages

    def format_tasks_for_discord(self,input_text):
        """
        Format task list for Discord with proper formatting and emoji indicators
        Returns a list of formatted chunks ready for Discord sending
        """
        # Split the input into tasks
        tasks = input_text.split('--------------------------------------------------')
        
        # Extract the header and footer
        header = tasks[0].strip()
        footer = tasks[-1].strip() if tasks else ""
        tasks = tasks[1:-1] if len(tasks) > 2 else []  # Remove header and footer
        
        # Initialize formatted output parts
        formatted_parts = []
        current_chunk = ["```ansi\n\u001b[1;33m=== OUTSTANDING TASKS ===\u001b[0m\n"]
        current_chunk_size = len(current_chunk[0])
        
        def add_to_chunks(content):
            nonlocal current_chunk, current_chunk_size
            content_size = len(content) + 1  # +1 for newline
            
            # If adding this content would exceed Discord's limit, start a new chunk
            if current_chunk_size + content_size > 1900:
                current_chunk.append("```")
                formatted_parts.append("\n".join(current_chunk))
                current_chunk = ["```ansi\n"]  # Start new chunk with ANSI header
                current_chunk_size = len(current_chunk[0])
                
            current_chunk.append(content)
            current_chunk_size += content_size
        
        # Process each task
        for task in tasks:
            if not task.strip():
                continue
                
            # Extract task components using regex
            task_id_match = re.search(r'Task ID: ([\w-]+)', task)
            proposal_match = re.search(r'Proposal: (.+?)(?=\nAcceptance:|$)', task, re.DOTALL)
            acceptance_match = re.search(r'Acceptance: (.+?)(?=\n|$)', task, re.DOTALL)
            priority_match = re.search(r'\.\. (\d+)', task)
            
            if not all([task_id_match, proposal_match, acceptance_match, priority_match]):
                continue
                
            # Extract date from task ID
            date_str = task_id_match.group(1).split('_')[0]
            try:
                date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')
                formatted_date = date_obj.strftime('%d %b %Y')
            except ValueError:
                formatted_date = date_str
            
            # Format task components
            task_parts = [
                f"\u001b[1;36m📌 Task {task_id_match.group(1)}\u001b[0m",
                f"\u001b[0;37mDate: {formatted_date}\u001b[0m",
                f"\u001b[0;32mPriority: {priority_match.group(1)}\u001b[0m",
                f"\u001b[1;37mProposal:\u001b[0m\n{proposal_match.group(1).strip()}",
                f"\u001b[1;37mAcceptance:\u001b[0m\n{acceptance_match.group(1).strip()}",
                "─" * 50
            ]
            
            # Add each part to chunks
            for part in task_parts:
                add_to_chunks(part)
        
        # Add footer if it exists
        if footer:
            add_to_chunks(f"\u001b[1;33m{footer}\u001b[0m")
        
        # Finalize last chunk
        current_chunk.append("```")
        formatted_parts.append("\n".join(current_chunk))
        
        return formatted_parts

    def output_postfiat_foundation_node_leaderboard_df(self):
        """ This generates the full Post Fiat Foundation Leaderboard """ 
        all_accounts = self.get_all_account_pft_memo_data()
        # Get the mode (most frequent) memo_format for each account
        account_modes = all_accounts.groupby('account')['memo_format'].agg(lambda x: x.mode()[0]).reset_index()
        # If you want to see the counts as well to verify
        account_counts = all_accounts.groupby(['account', 'memo_format']).size().reset_index(name='count')
        
        # Sort by account for readability
        account_modes = account_modes.sort_values('account')
        account_name_map = account_modes.groupby('account').first()['memo_format']
        past_month_transactions = all_accounts[all_accounts['datetime']>datetime.datetime.now()-datetime.timedelta(30)]
        node_transactions = past_month_transactions[past_month_transactions['account']==self.node_address].copy()
        rewards_only=node_transactions[node_transactions['memo_data'].apply(lambda x: 'REWARD RESPONSE' in x)].copy()
        rewards_only['count']=1
        rewards_only['PFT']=rewards_only['tx_json'].apply(lambda x: x['DeliverMax']['value']).astype(float)
        account_to_yellow_flag__count = rewards_only[rewards_only['memo_data'].apply(lambda x: 'YELLOW FLAG' in x)][['count','destination']].groupby('destination').sum()['count']
        account_to_red_flag__count = rewards_only[rewards_only['memo_data'].apply(lambda x: 'RED FLAG' in x)][['count','destination']].groupby('destination').sum()['count']
        
        total_reward_number= rewards_only[['count','destination']].groupby('destination').sum()['count']
        account_score_constructor = pd.DataFrame(account_name_map)
        account_score_constructor=account_score_constructor[account_score_constructor.index!=self.node_address].copy()
        account_score_constructor['reward_count']=total_reward_number
        account_score_constructor['yellow_flags']=account_to_yellow_flag__count
        account_score_constructor=account_score_constructor[['reward_count','yellow_flags']].fillna(0).copy()
        account_score_constructor= account_score_constructor[account_score_constructor['reward_count']>=1].copy()
        account_score_constructor['yellow_flag_pct']=account_score_constructor['yellow_flags']/account_score_constructor['reward_count']
        total_pft_rewards= rewards_only[['destination','PFT']].groupby('destination').sum()['PFT']
        account_score_constructor['red_flag']= account_to_red_flag__count
        account_score_constructor['red_flag']=account_score_constructor['red_flag'].fillna(0)
        account_score_constructor['total_rewards']= total_pft_rewards
        account_score_constructor['reward_score__z']=(account_score_constructor['total_rewards']-account_score_constructor['total_rewards'].mean())/account_score_constructor['total_rewards'].std()
        
        account_score_constructor['yellow_flag__z']=(account_score_constructor['yellow_flag_pct']-account_score_constructor['yellow_flag_pct'].mean())/account_score_constructor['yellow_flag_pct'].std()
        account_score_constructor['quant_score']=(account_score_constructor['reward_score__z']*.65)-(account_score_constructor['reward_score__z']*-.35)
        top_score_frame = account_score_constructor[['total_rewards','yellow_flag_pct','quant_score']].sort_values('quant_score',ascending=False)
        top_score_frame['account_name']=account_name_map
        user_account_map = {}
        for x in list(top_score_frame.index):
            user_account_string = self.get_full_user_context_string(account_address=x)
            print(x)
            user_account_map[x]= user_account_string
        agency_system_prompt = """ You are the Post Fiat Agency Score calculator.
        
        An Agent is a human or an AI that has outlined an objective.
        
        An agency score has four parts:
        1] Focus - the extent to which an Agent is focused.
        2] Motivation - the extent to which an Agent is driving forward predictably and aggressively towards goals.
        3] Efficacy - the extent to which an Agent is likely completing high value tasks that will drive an outcome related to the inferred goal of the tasks.
        4] Honesty - the extent to which a Subject is likely gaming the Post Fiat Agency system.
        
        It is very important that you deliver assessments of Agency Scores accurately and objectively in a way that is likely reproducible. Future Post Fiat Agency Score calculators will re-run this score, and if they get vastly different scores than you, you will be called into the supervisor for an explanation. You do not want this so you do your utmost to output clean, logical, repeatable values.
        """ 
        
        agency_user_prompt="""USER PROMPT
        
        Please consider the activity slice for a single day provided below:
        pft_transaction is how many transactions there were
        pft_directional value is the PFT value of rewards
        pft_absolute value is the bidirectional volume of PFT
        
        <activity slice>
        __FULL_ACCOUNT_CONTEXT__
        <activity slice ends>
        
        Provide one to two sentences directly addressing how the slice reflects the following Four scores (a score of 1 is a very low score and a score of 100 is a very high score):
        1] Focus - the extent to which an Agent is focused.
        A focused agent has laser vision on a couple key objectives and moves the ball towards it.
        An unfocused agent is all over the place.
        A paragon of focus is Steve Jobs, who is famous for focusing on the few things that really matter.
        2] Motivation - the extent to which an Agent is driving forward predictably and aggressively towards goals.
        A motivated agent is taking massive action towards objectives. Not necessarily focused but ambitious.
        An unmotivated agent is doing minimal work.
        A paragon of focus is Elon Musk, who is famous for his extreme work ethic and drive.
        3] Efficacy - the extent to which an Agent is likely completing high value tasks that will drive an outcome related to the inferred goal of the tasks.
        An effective agent is delivering maximum possible impact towards implied goals via actions.
        An ineffective agent might be focused and motivated but not actually accomplishing anything.
        A paragon of focus is Lionel Messi, who is famous for taking the minimal action to generate maximum results.
        4] Honesty - the extent to which a Subject is likely gaming the Post Fiat Agency system.
        
        Then provide an integer score.
        
        Your output should be in the following format:
        | FOCUS COMMENTARY | <1 to two sentences> |
        | MOTIVATION COMMENTARY | <1 to two sentences> |
        | EFFICACY COMMENTARY | <1 to two sentences> |
        | HONESTY COMMENTARY | <one to two sentences> |
        | FOCUS SCORE | <integer score from 1-100> |
        | MOTIVATION SCORE | <integer score from 1-100> |
        | EFFICACY SCORE | <integer score from 1-100> |
        | HONESTY SCORE | <integer score from 1-100> |
        """
        top_score_frame['user_account_details']=user_account_map
        top_score_frame['system_prompt']=agency_system_prompt
        top_score_frame['user_prompt']= agency_user_prompt
        top_score_frame['user_prompt']=top_score_frame.apply(lambda x: x['user_prompt'].replace('__FULL_ACCOUNT_CONTEXT__',x['user_account_details']),axis=1)
        def construct_scoring_api_arg(user_prompt, system_prompt):
            gx ={
                "model": constants.DEFAULT_OPEN_AI_MODEL,
                "temperature":0,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }
            return gx
        top_score_frame['api_args']=top_score_frame.apply(lambda x: construct_scoring_api_arg(user_prompt=x['user_prompt'],system_prompt=x['system_prompt']),axis=1)
        
        async_run_map = top_score_frame['api_args'].head(25).to_dict()
        async_run_map__2 = top_score_frame['api_args'].head(25).to_dict()
        async_output_df1= self.open_ai_request_tool.create_writable_df_for_async_chat_completion(arg_async_map=async_run_map)
        time.sleep(15)
        async_output_df2= self.open_ai_request_tool.create_writable_df_for_async_chat_completion(arg_async_map=async_run_map__2)
        
        
        def extract_scores(text_data):
            # Split the text into individual reports
            reports = text_data.split("',\n '")
            
            # Clean up the string formatting
            reports = [report.strip("['").strip("']") for report in reports]
            
            # Initialize list to store all scores
            all_scores = []
            
            for report in reports:
                # Extract only scores using regex
                scores = {
                    'focus_score': int(re.search(r'\| FOCUS SCORE \| (\d+) \|', report).group(1)) if re.search(r'\| FOCUS SCORE \| (\d+) \|', report) else None,
                    'motivation_score': int(re.search(r'\| MOTIVATION SCORE \| (\d+) \|', report).group(1)) if re.search(r'\| MOTIVATION SCORE \| (\d+) \|', report) else None,
                    'efficacy_score': int(re.search(r'\| EFFICACY SCORE \| (\d+) \|', report).group(1)) if re.search(r'\| EFFICACY SCORE \| (\d+) \|', report) else None,
                    'honesty_score': int(re.search(r'\| HONESTY SCORE \| (\d+) \|', report).group(1)) if re.search(r'\| HONESTY SCORE \| (\d+) \|', report) else None
                }
                all_scores.append(scores)
            
            return all_scores
        
        async_output_df1['score_breakdown']=async_output_df1['choices__message__content'].apply(lambda x: extract_scores(x)[0])
        async_output_df2['score_breakdown']=async_output_df2['choices__message__content'].apply(lambda x: extract_scores(x)[0])
        for xscore in ['focus_score','motivation_score','efficacy_score','honesty_score']:
            async_output_df1[xscore]=async_output_df1['score_breakdown'].apply(lambda x: x[xscore])
            async_output_df2[xscore]=async_output_df2['score_breakdown'].apply(lambda x: x[xscore])
        score_components = pd.concat([async_output_df1[['focus_score','motivation_score','efficacy_score','honesty_score','internal_name']],
                async_output_df2[['focus_score','motivation_score','efficacy_score','honesty_score','internal_name']]]).groupby('internal_name').mean()
        score_components.columns=['focus','motivation','efficacy','honesty']
        score_components['total_qualitative_score']= score_components[['focus','motivation','efficacy','honesty']].mean(1)
        final_score_frame = pd.concat([top_score_frame,score_components],axis=1)
        final_score_frame['total_qualitative_score']=final_score_frame['total_qualitative_score'].fillna(50)
        final_score_frame['reward_percentile']=((final_score_frame['quant_score']*33)+100)/2
        final_score_frame['overall_score']= (final_score_frame['reward_percentile']*.7)+(final_score_frame['total_qualitative_score']*.3)
        final_leaderboard = final_score_frame[['account_name','total_rewards','yellow_flag_pct','reward_percentile','focus','motivation','efficacy','honesty','total_qualitative_score','overall_score']].copy()
        final_leaderboard['total_rewards']=final_leaderboard['total_rewards'].apply(lambda x: int(x))
        final_leaderboard.index.name = 'Foundation Node Leaderboard as of '+datetime.datetime.now().strftime('%Y-%m-%d')
        return final_leaderboard

    def format_and_write_leaderboard(self):
        """ This loads the current leaderboard df and writes it""" 
        def format_leaderboard_df(df):
            """
            Format the leaderboard DataFrame with cleaned up number formatting
            
            Args:
                df: pandas DataFrame with the leaderboard data
            Returns:
                formatted DataFrame with cleaned up number display
            """
            # Create a copy to avoid modifying the original
            formatted_df = df.copy()
            
            # Format total_rewards as whole numbers with commas
            def format_number(x):
                try:
                    # Try to convert directly to int
                    return f"{int(x):,}"
                except ValueError:
                    # If already formatted with commas, remove them and convert
                    try:
                        return f"{int(str(x).replace(',', '')):,}"
                    except ValueError:
                        return str(x)
            
            formatted_df['total_rewards'] = formatted_df['total_rewards'].apply(format_number)
            
            # Format yellow_flag_pct as percentage with 1 decimal place
            def format_percentage(x):
                try:
                    if pd.notnull(x):
                        # Remove % if present and convert to float
                        x_str = str(x).replace('%', '')
                        value = float(x_str)
                        if value > 1:  # Already in percentage form
                            return f"{value:.1f}%"
                        else:  # Convert to percentage
                            return f"{value*100:.1f}%"
                    return "0%"
                except ValueError:
                    return str(x)
            
            formatted_df['yellow_flag_pct'] = formatted_df['yellow_flag_pct'].apply(format_percentage)
            
            # Format reward_percentile with 1 decimal place
            def format_float(x):
                try:
                    return f"{float(str(x).replace(',', '')):,.1f}"
                except ValueError:
                    return str(x)
            
            formatted_df['reward_percentile'] = formatted_df['reward_percentile'].apply(format_float)
            
            # Format score columns with 1 decimal place
            score_columns = ['focus', 'motivation', 'efficacy', 'honesty', 'total_qualitative_score']
            for col in score_columns:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{float(x):.1f}" if pd.notnull(x) and x != 'N/A' else "N/A")
            
            # Format overall_score with 1 decimal place
            formatted_df['overall_score'] = formatted_df['overall_score'].apply(format_float)
            
            return formatted_df
        
        def test_leaderboard_creation(leaderboard_df, output_path="test_leaderboard.png"):
            """
            Test function to create leaderboard image from a DataFrame
            """
            import plotly.graph_objects as go
            from datetime import datetime
            
            # Format the DataFrame first
            formatted_df = format_leaderboard_df(leaderboard_df)
            
            # Format current date
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # Add rank and get the index
            wallet_addresses = formatted_df.index.tolist()  # Get addresses from index
            
            # Define column headers with line breaks and widths
            headers = [
                'Rank',
                'Wallet Address',
                'Account<br>Name', 
                'Total<br>Rewards', 
                'Yellow<br>Flag %', 
                'Reward<br>Percentile',
                'Focus',
                'Motivation',
                'Efficacy',
                'Honesty',
                'Total<br>Qualitative',
                'Overall<br>Score'
            ]
            
            # Custom column widths
            column_widths = [30, 140, 80, 60, 60, 60, 50, 50, 50, 50, 60, 60]
            
            # Prepare values with rank column and wallet addresses
            values = [
                [str(i+1) for i in range(len(formatted_df))],  # Rank
                wallet_addresses,  # Full wallet address from index
                formatted_df['account_name'],
                formatted_df['total_rewards'],
                formatted_df['yellow_flag_pct'],
                formatted_df['reward_percentile'],
                formatted_df['focus'],
                formatted_df['motivation'],
                formatted_df['efficacy'],
                formatted_df['honesty'],
                formatted_df['total_qualitative_score'],
                formatted_df['overall_score']
            ]
            
            # Create figure
            fig = go.Figure(data=[go.Table(
                columnwidth=column_widths,
                header=dict(
                    values=headers,
                    fill_color='#000000',  # Changed to black
                    font=dict(color='white', size=15),
                    align=['center'] * len(headers),
                    height=60,
                    line=dict(width=1, color='#40444b')
                ),
                cells=dict(
                    values=values,
                    fill_color='#000000',  # Changed to black
                    font=dict(color='white', size=14),
                    align=['left', 'left'] + ['center'] * (len(headers)-2),
                    height=35,
                    line=dict(width=1, color='#40444b')
                )
            )])
            
            # Update layout
            fig.update_layout(
                width=1800,
                height=len(formatted_df) * 35 + 100,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='#000000',  # Changed to black
                plot_bgcolor='#000000',  # Changed to black
                title=dict(
                    text=f"Foundation Node Leaderboard as of {current_date} (30D Rolling)",
                    font=dict(color='white', size=20),
                    x=0.5
                )
            )
            
            # Save as image with higher resolution
            fig.write_image(output_path, scale=2)
            
            print(f"Leaderboard image saved to: {output_path}")
            
            try:
                from IPython.display import Image
                return Image(filename=output_path)
            except:
                return None
        leaderboard_df = self.output_postfiat_foundation_node_leaderboard_df()
        test_leaderboard_creation(leaderboard_df=format_leaderboard_df(leaderboard_df))

    # TODO: Consider deprecating, not used anywhere
    def get_full_google_text_and_verification_stub_for_account(self,address_to_work = 'rwmzXrN3Meykp8pBd3Boj1h34k8QGweUaZ'):

        all_account_memos = self.get_memo_detail_df_for_account(account_address=address_to_work)
        google_acount = self.get_most_recent_google_doc_for_user(account_memo_detail_df
                                                                                =all_account_memos, 
                                                                                address=address_to_work)
        user_full_google_acccount = self.generic_pft_utilities.get_google_doc_text(share_link=google_acount)
        #verification #= user_full_google_acccount.split('VERIFICATION SECTION START')[-1:][0].split('VERIFICATION SECTION END')[0]
        
        import re
        
        def extract_verification_text(content):
            """
            Extracts text between task verification markers.
            
            Args:
                content (str): Input text containing verification sections
                
            Returns:
                str: Extracted text between markers, or empty string if no match
            """
            pattern = r'TASK VERIFICATION SECTION START(.*?)TASK VERIFICATION SECTION END'
            
            try:
                # Use re.DOTALL to make . match newlines as well
                match = re.search(pattern, content, re.DOTALL)
                return match.group(1).strip() if match else ""
            except Exception as e:
                print(f"Error extracting text: {e}")
                return ""
        xstr =extract_verification_text(user_full_google_acccount)
        return {'verification_text': xstr, 'full_google_doc': user_full_google_acccount}

    def get_account_pft_balance(self, account_address: str) -> float:
        """
        Get the PFT balance for a given account address.
        Returns the balance as a float, or 0.0 if no PFT trustline exists or on error.
        
        Args:
            account_address (str): The XRPL account address to check
            
        Returns:
            float: The PFT balance for the account
        """
        client = JsonRpcClient(self.local_rippled_url)
        try:
            account_lines = AccountLines(
                account=account_address,
                ledger_index="validated"
            )
            account_line_response = client.request(account_lines)
            pft_lines = [i for i in account_line_response.result['lines'] 
                        if i['account'] == self.pft_issuer]
            
            if pft_lines:
                return float(pft_lines[0]['balance'])
            return 0.0
        except Exception as e:
            print(f"Error getting PFT balance for {account_address}: {str(e)}")
            return 0.0