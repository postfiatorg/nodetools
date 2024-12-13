#!/usr/bin/env python3

import os
import json
import logging
import getpass
import pandas as pd
import xrpl
from xrpl.clients import JsonRpcClient
import sqlalchemy
from nodetools.utilities.credentials import CredentialManager
import traceback

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class MemoDebugger:
    def __init__(
        self,
        db_connection_string: str,
        xrpl_endpoint: str,
        pft_issuer: str
    ):
        """Initialize debugger with connection details.
        
        Args:
            db_connection_string: SQLAlchemy connection string for the database
            xrpl_endpoint: URL for XRPL node
            pft_issuer: PFT issuer address
        """
        self.db_engine = sqlalchemy.create_engine(db_connection_string)
        self.client = JsonRpcClient(xrpl_endpoint)
        self.pft_issuer = pft_issuer

    @staticmethod
    def convert_memo_dict__generic(memo_dict):
        """Convert memo dictionary from hex to text."""
        try:
            return {
                'MemoFormat': MemoDebugger.hex_to_text(memo_dict.get('MemoFormat', '')),
                'MemoType': MemoDebugger.hex_to_text(memo_dict.get('MemoType', '')),
                'MemoData': MemoDebugger.hex_to_text(memo_dict.get('MemoData', ''))
            }
        except:
            return {'MemoFormat': '', 'MemoType': '', 'MemoData': ''}

    def get_from_db(self, task_id: str) -> dict:
        """Get memo from database cache."""
        query = """
        SELECT 
            hash,
            main_memo_data,
            transaction_result,
            account,
            destination,
            simple_date
        FROM memo_detail_view 
        """
        
        with self.db_engine.connect() as conn:
            # First get all memos
            df = pd.read_sql(query, conn, parse_dates=['simple_date'])
            
            # Convert memo data and extract types
            df['converted_memos'] = df['main_memo_data'].apply(self.convert_memo_dict__generic)
            df['memo_type'] = df['converted_memos'].apply(lambda x: x.get('MemoType', ''))
            
            # Filter for our specific task_id
            result = df[df['memo_type'] == task_id]
            
            if result.empty:
                return None
                
            # Convert to dict and return first match
            memo_dict = result.iloc[0].to_dict()
            return {
                'hash': memo_dict['hash'],
                'main_memo_data': memo_dict['main_memo_data'],
                'memo_type': memo_dict['memo_type'],
                'memo_format': memo_dict['converted_memos'].get('MemoFormat', ''),
                'transaction_result': memo_dict['transaction_result'],
                'account': memo_dict['account'],
                'destination': memo_dict['destination']
            }

    def get_from_xrpl(self, account_address: str, task_id: str) -> dict:
        """Get memo directly from XRPL."""
        request = xrpl.models.requests.AccountTx(
            account=account_address,
            ledger_index_min=-1,
            ledger_index_max=-1
        )
        
        response = self.client.request(request)
        
        for txn in response.result.get('transactions', []):
            tx = txn.get('tx', {})
            if 'Memos' in tx:
                memo = tx['Memos'][0]['Memo']
                memo_type = bytes.fromhex(memo.get('MemoType', '')).decode('utf-8')
                if memo_type == task_id:
                    return txn
        return None

    @staticmethod
    def hex_to_text(hex_string: str) -> str:
        """Convert hex string to text."""
        try:
            return bytes.fromhex(hex_string).decode('utf-8')
        except:
            return hex_string

def main():
    print("\nMemo Debugging Script")
    print("====================")

    # Get encryption password
    try:
        encryption_password = getpass.getpass("\nEnter your encryption password: ")

        # Initialize the credential manager
        cm = CredentialManager(encryption_password)
        
        # Get database connection string
        db_connection = cm.get_credential('postfiatfoundation_testnet_postgresconnstring')
        if not db_connection:
            print("\nError: Could not retrieve database connection string")
            return

        # Configuration
        XRPL_ENDPOINT = "https://s.altnet.rippletest.net:51234"  # or your preferred endpoint
        PFT_ISSUER = "rLX2tgumpiUE6kjr757Ao8HWiJzC8uuBSN"  # PFT issuer address
        
        # Initialize debugger
        debugger = MemoDebugger(db_connection, XRPL_ENDPOINT, PFT_ISSUER)
        
        # Debug parameters
        TASK_ID = "2024-12-06_23:56__JP62"
        RECEIVER = "rN2oaXBhFE9urGN5hXup937XpoFVkrnUhu"
        SENDER = "rNC2hS269hTvMZwNakwHPkw4VeZNwzpS2E"
        
        # Get from database first
        db_memo = debugger.get_from_db(TASK_ID)
        if db_memo:
            logger.debug("\nDatabase cached version:")
            logger.debug(json.dumps(db_memo, indent=2))
            
            # ADDED: Check both sender and receiver accounts in XRPL
            logger.debug(f"\nQuerying XRPL for sender: {SENDER}")
            xrpl_memo = debugger.get_from_xrpl(SENDER, TASK_ID)
            
            # ADDED: If not found in sender, try receiver
            if not xrpl_memo:
                logger.debug(f"Not found in sender account, trying receiver: {RECEIVER}")
                xrpl_memo = debugger.get_from_xrpl(RECEIVER, TASK_ID)
            
            if xrpl_memo:
                logger.debug("\nXRPL version:")
                logger.debug(json.dumps(xrpl_memo, indent=2))
                
                # Compare encrypted content
                db_encrypted = db_memo['main_memo_data'].get('MemoData', '')
                xrpl_encrypted = xrpl_memo['tx']['Memos'][0]['Memo'].get('MemoData', '')
                
                if db_encrypted != xrpl_encrypted:
                    logger.debug("\nEncrypted content mismatch!")
                    logger.debug(f"DB version:   {db_encrypted}")
                    logger.debug(f"XRPL version: {xrpl_encrypted}")
                else:
                    logger.debug("\nEncrypted content matches between DB and XRPL")
            else:
                logger.error("Memo not found in XRPL data for either sender or receiver")
        else:
            logger.error("Memo not found in database")

    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        return

    except Exception as e:
        print(f"\nError: {str(e)}")
        logger.error(traceback.format_exc())
        if "MAC check failed" in str(e):
            print("Invalid encryption password.")
        return

if __name__ == "__main__":
    main()