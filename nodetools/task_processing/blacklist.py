import time
import json
import requests
import datetime
import pandas as pd
from sqlalchemy import text
from nodetools.utilities.db_manager import DBConnectionManager

class LiveBlacklistUpdater:
    def __init__(self, node_name, account_address, sleep_interval=300):
        """
        :param node_name: The node/username to connect to the database (e.g., 'postfiatfoundation')
        :param account_address: The XRPL address for which to fetch transactions
        :param sleep_interval: How frequently (in seconds) to run the update; default 300 = 5 mins
        """
        self.node_name = node_name
        self.account_address = account_address
        self.sleep_interval = sleep_interval

        # DB manager for connections
        self.db_manager = DBConnectionManager()

        # For auditing
        self.raw_transactions_df = pd.DataFrame()  # Full transaction DataFrame
        self.flag_list_df = pd.DataFrame()         # DataFrame of flagged addresses
        self.blacklist_from_sheet = []             # Raw list from Google Sheet
        self.current_combined_blacklist = []       # Combined final blacklist

    @staticmethod
    def hex_to_text(hex_string):
        """
        Converts a hex string to UTF-8 text.
        Returns the original string if decoding fails.
        """
        try:
            return bytes.fromhex(hex_string).decode("utf-8")
        except Exception:
            return hex_string

    @staticmethod
    def decode_memo_list(memo_list_str):
        """
        Takes the JSON string for the memos column (like '[{"MemoData": "..."}]')
        and decodes all MemoData, MemoFormat, MemoType fields from hex to text.
        """
        memos = json.loads(memo_list_str) if memo_list_str else []
        decoded_memos = []
        for memo_wrapper in memos:
            # XRPL stores each memo under the key 'Memo'
            memo = memo_wrapper.get('Memo', memo_wrapper)
            decoded_memo = {}
            for field in ['MemoData', 'MemoFormat', 'MemoType']:
                if field in memo:
                    decoded_memo[field] = LiveBlacklistUpdater.hex_to_text(memo[field])
            decoded_memos.append(decoded_memo)
        return decoded_memos

    def get_cached_transactions_for_address(self):
        """
        Query the postfiat_tx_cache table to return all transactions
        where the given address is either the source (`account`) or destination.
        """
        dbconnx = self.db_manager.spawn_sqlalchemy_db_connection_for_user(username=self.node_name)
        try:
            query = text("""
                SELECT *
                FROM postfiat_tx_cache
                WHERE account = :address
                   OR destination = :address
            """)
            df = pd.read_sql(query, dbconnx, params={"address": self.account_address})

            # Parse JSON fields back into Python objects
            if not df.empty:
                df['meta'] = df['meta'].apply(json.loads)
                df['tx_json'] = df['tx_json'].apply(json.loads)

            return df
        finally:
            dbconnx.dispose()

    def run_once(self):
        """
        Runs one iteration of the entire process: fetch transactions,
        decode memos, identify flagged addresses, update the database.
        """
        # 1. Fetch transactions
        df = self.get_cached_transactions_for_address()
        self.raw_transactions_df = df.copy()  # Store for auditing

        # 2. Decode memos
        df["decoded_memos"] = df["memos"].apply(self.decode_memo_list)

        # 3. If you only need the first memo's text
        df["first_memo_data"] = df["decoded_memos"].apply(lambda x: x[0]["MemoData"] if x else None)

        # 4. Identify flagged transactions
        all_yellow_flag = df[df['decoded_memos'].apply(lambda x: "YELLOW FLAG" in str(x))].copy()
        all_red_flag = df[df['decoded_memos'].apply(lambda x: "RED FLAG" in str(x))].copy()

        # 5. Convert date strings to datetime
        all_yellow_flag['datetime'] = pd.to_datetime(all_yellow_flag['close_time_iso'].apply(lambda x: str(x)[0:10]))
        all_red_flag['datetime'] = pd.to_datetime(all_red_flag['close_time_iso'].apply(lambda x: str(x)[0:10]))

        most_recent_yellow_flag = (
            all_yellow_flag
            .sort_values('datetime')
            .groupby('destination')
            .last()[['datetime']]
            .reset_index()
        )
        most_recent_yellow_flag['flag_type'] = "YELLOW FLAG"

        most_recent_red_flag = (
            all_red_flag
            .sort_values('datetime')
            .groupby('destination')
            .last()[['datetime']]
            .reset_index()
        )
        most_recent_red_flag['flag_type'] = "RED FLAG"

        flag_list = pd.concat([most_recent_yellow_flag, most_recent_red_flag]).copy()

        # 6. Add day cool-off logic
        flag_list['day_cool_off'] = flag_list['flag_type'].map({'YELLOW FLAG': 1, 'RED FLAG': 10})
        flag_list['cool_off_datetime'] = flag_list['datetime'] + flag_list['day_cool_off'].apply(lambda x: datetime.timedelta(x))
        flag_list['is_currently_blacklisted'] = flag_list['cool_off_datetime'] >= datetime.datetime.now()

        self.flag_list_df = flag_list.copy()  # Store for auditing

        # 7. Pull current blacklist from Google Sheets
        xtext = requests.get(
            'https://docs.google.com/spreadsheets/d/e/2PACX-1vSmKVJYwa5VMAPIS46dUGDG6mzvDX3DcxM5cExGkeB2PLRSTr88evyVf5oMkUUco_B11AAKwgCXg7Vp/pubhtml?gid=0&single=true'
        )
        blacklist = pd.read_html(xtext.text)[0]
        current_blacklist = list(blacklist['Unnamed: 1'])
        self.blacklist_from_sheet = current_blacklist  # Store for auditing

        # 8. Combine existing blacklist with newly flagged addresses
        accounts_frozen_due_to_flags = list(flag_list[flag_list['is_currently_blacklisted'] == True]['destination'])
        blacklist_to_avoid = current_blacklist + accounts_frozen_due_to_flags
        self.current_combined_blacklist = blacklist_to_avoid  # Store for auditing

        # 9. Write the combined blacklist to the DB
        db_to_write = self.db_manager.spawn_sqlalchemy_db_connection_for_user(username=self.node_name)
        try:
            live_blacklist = pd.DataFrame(blacklist_to_avoid, columns=['address'])
            live_blacklist['date'] = datetime.datetime.now()
            live_blacklist.to_sql('task_node_live_blacklist', db_to_write, if_exists='replace')
        finally:
            db_to_write.dispose()

        print(f"[{datetime.datetime.now()}] Updated blacklist with {len(blacklist_to_avoid)} entries.")

    def audit_results(self):
        """
        Example convenience method to print/log the items you might want to audit.
        You could replace prints with logging, or return them for further processing.
        """
        print("\n=== Audit Results ===")
        print("--- Raw Transactions DF ---")
        print(self.raw_transactions_df.head())
        
        print("\n--- Flag List DF ---")
        print(self.flag_list_df.head())
        
        print("\n--- Blacklist from Google Sheets ---")
        print(self.blacklist_from_sheet)
        
        print("\n--- Current Combined Blacklist ---")
        print(self.current_combined_blacklist)

    def run_forever(self):
        """
        Continuously runs `run_once` every self.sleep_interval (default 5 minutes).
        """
        while True:
            try:
                self.run_once()
                # Optional: call self.audit_results() here if you'd like to see results each loop
                # self.audit_results()
            except Exception as e:
                print(f"Error in run_once: {e}")
            time.sleep(self.sleep_interval)



# Example usage
""" 
updater = LiveBlacklistUpdater(
    node_name="postfiatfoundation",
    account_address="r4yc85M1hwsegVGZ1pawpZPwj65SVs8PzD",
    sleep_interval=300  # 5 minutes
)

# Run just once, then audit
updater.run_once()
updater.audit_results()

    # If you need continuous operation:
updater.run_forever()
""" 