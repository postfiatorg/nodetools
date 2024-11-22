import xrpl
import nodetools.utilities.constants as constants
import getpass

def setup_destination_tag_account(wallet_seed: str):
    """Sets up a testnet account with RequireDest flag enabled."""

    # Connect to testnet
    client = xrpl.clients.JsonRpcClient(constants.TESTNET_URL)

    # Create a wallet from the seed
    wallet = xrpl.wallet.Wallet.from_seed(wallet_seed)

    # Create AccountSet transaction to require destination tags
    settings_tx = xrpl.models.transactions.AccountSet(
        account=wallet.address,
        set_flag=xrpl.models.transactions.AccountSetAsfFlag.ASF_REQUIRE_DEST
    )

    # Submit the transaction
    print(f"Enabling Require Destination Tag for account {wallet.address}...")
    response = xrpl.transaction.submit_and_wait(
        transaction=settings_tx,
        client=client,
        wallet=wallet
    )
    print(f"Transaction result: {response}")

    # Verify the setting
    print(f"Verifying Require Destination Tag for account {wallet.address}...")
    acct_info = client.request(xrpl.models.requests.AccountInfo(
        account=wallet.address,
        ledger_index="validated"
    ))
    flags = acct_info.result['account_data']['Flags']
    if flags & 0x00020000 != 0:
        print(f"Require Destination Tag for account {wallet.address} is enabled.")
    else:
        print(f"Require Destination Tag for account {wallet.address} is DISABLED.")


if __name__ == "__main__":
    wallet_seed = getpass.getpass("Enter wallet seed for test account: ")
    setup_destination_tag_account(wallet_seed)