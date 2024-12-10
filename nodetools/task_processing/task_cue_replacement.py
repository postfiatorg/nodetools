import pandas as pd
import re
import uuid
import datetime
import time
import pytz
from IPython.display import display
from nodetools.task_processing.task_creation import NewTaskGeneration

def filter_unprocessed_requests(memo_history):
    """
    Identifies user requests that have not yet received a proposal.
    """
    # Group by memo_type to find the most recent memo_data for each task
    most_recent_status = memo_history.groupby('memo_type').last()['memo_data']
    
    # Identify tasks that started as requests
    requests = memo_history[memo_history['memo_data'].astype(str).str.contains('REQUEST_POST_FIAT', na=False)].copy()
    requests['most_recent_status'] = requests['memo_type'].map(most_recent_status)

    # Ensure 'most_recent_status' is string before searching
    outstanding_requests = requests[
        requests['most_recent_status'].astype(str).str.contains('REQUEST_POST_FIAT', na=False)
    ].groupby('memo_type').last().reset_index()

    return outstanding_requests

def main():
    # Initialize the generation system
    task_gen = NewTaskGeneration(password="everythingIsRigged1a")

    # Retrieve the node address from the credential manager
    node_address = task_gen._credential_manager.get_credential('node_v1xrpaddress')

    # Load full memo history for the node account
    memo_history = task_gen.generic_pft_utilities.get_account_memo_history(
        account_address=node_address, 
        pft_only=False
    )

    # Identify outstanding requests with no proposals
    outstanding_requests = filter_unprocessed_requests(memo_history)

    if outstanding_requests.empty:
        print("No outstanding tasks to process.")
        return

    # Prepare a map of {combined_task_key: user_request_string} for processing
    task_map = {}
    for _, row in outstanding_requests.iterrows():
        account_id = row['account']
        task_id = row['memo_type']
        user_request = row['most_recent_status'].replace('REQUEST_POST_FIAT ___', '').strip()
        
        combined_key = task_gen.create_task_key(account_id, task_id)
        task_map[combined_key] = user_request

    # Run the task generation process
    output_df = task_gen.process_task_map_to_proposed_pf(
        task_map=task_map,
        model="anthropic/claude-3.5-sonnet:beta",
        get_google_doc=True,
        get_historical_memos=True
    )
    
    # Add processing columns
    output_df['account_to_send_to'] = output_df['internal_name'].apply(lambda x: x.split('task_gen__')[-1].split('__')[0])
    output_df['pft_to_send'] = 1

    # Send proposed PF to users
    node_wallet = task_gen.generic_pft_utilities.spawn_wallet_from_seed(
        seed=task_gen._credential_manager.get_credential('node_v1xrpsecret')
    )

    for idx, row in output_df.iterrows():
        task_id = row['task_id']
        internal_name = row['internal_name']
        
        # Parse account_id and task_id
        name_parts = internal_name.split('__')
        if name_parts[0] == 'task_gen':
            account_id = name_parts[1]
            final_task_id = name_parts[3]
        else:
            account_id, final_task_id = task_gen.parse_task_key(internal_name)

        memo_data = row['pf_proposal_string']
        
        memo = task_gen.generic_pft_utilities.construct_standardized_xrpl_memo(
            memo_data=memo_data,
            memo_format='SYSTEM',
            memo_type=final_task_id
        )
        
        resp = task_gen.generic_pft_utilities.send_memo(
            wallet_seed_or_wallet=node_wallet,
            destination=row['account_to_send_to'],
            memo=memo,
            username='.system',
            pft_amount=row['pft_to_send']
        )
        
        if not task_gen.generic_pft_utilities.verify_transaction_response(resp):
            print(f"Failed to send PF proposal to {row['account_to_send_to']} for task {final_task_id}")
        else:
            print(f"Sent PF proposal to {row['account_to_send_to']} for task {final_task_id}")

if __name__ == "__main__":
    main()