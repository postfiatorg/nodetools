import xrpl
from pftpyclient.wallet_ux.constants import TESTNET_URL

client = xrpl.clients.JsonRpcClient(TESTNET_URL)

COLD_WALLET_SEED = ""  # wallet for issuing tokens to hot wallet. corresponds to TESTNET_ISSUER_ADDRESS
HOT_WALLET_SEED = ""  # wallet for distributing tokens after issuance

cold_wallet = xrpl.wallet.Wallet.from_seed(COLD_WALLET_SEED)
hot_wallet = xrpl.wallet.Wallet.from_seed(HOT_WALLET_SEED)

cold_settings_tx = xrpl.models.transactions.AccountSet(
    account=cold_wallet.address,
    transfer_rate=0,
    tick_size=5,
    set_flag=xrpl.models.transactions.AccountSetAsfFlag.ASF_DEFAULT_RIPPLE
)

# Send TX
print("Sending cold address AccountSet transaction")
response = xrpl.transaction.submit_and_wait(
    transaction=cold_settings_tx, 
    client=client,
    wallet=cold_wallet
)
print(response)

hot_settings_tx = xrpl.models.transactions.AccountSet(
    account=hot_wallet.address,
    set_flag=xrpl.models.transactions.AccountSetAsfFlag.ASF_REQUIRE_AUTH
)

print("Sending hot address AccountSet transaction...")
response = xrpl.transaction.submit_and_wait(hot_settings_tx, client, hot_wallet)
print(response)

# Create trust line from hot to cold address
currency_code = "PFT"
trust_set_tx = xrpl.models.transactions.TrustSet(
    account=hot_wallet.address,
    limit_amount=xrpl.models.amounts.issued_currency_amount.IssuedCurrencyAmount(
        currency=currency_code,
        issuer=cold_wallet.address,
        value="100000000000" # 100B limit
    )
)

print("Creating trust line from hot address to issuer...")
response = xrpl.transaction.submit_and_wait(trust_set_tx, client, hot_wallet)
print(response)

# Issue the tokens
issue_quantity = "100000000000" # 100B tokens
currency_code = "PFT"
send_token_tx = xrpl.models.transactions.Payment(
    account=cold_wallet.address,
    destination=hot_wallet.address,
    amount=xrpl.models.amounts.issued_currency_amount.IssuedCurrencyAmount(
        currency=currency_code,
        issuer=cold_wallet.address,
        value=issue_quantity
    )
)

print(f"Sending {issue_quantity} {currency_code} to {hot_wallet.address}...")
response = xrpl.transaction.submit_and_wait(send_token_tx, client, cold_wallet)
print(response)

print("Getting hot address balances...")
response = client.request(xrpl.models.requests.AccountLines(
    account=hot_wallet.address,
    ledger_index="validated",
))
print(response)

print("Getting cold address balances...")
response = client.request(xrpl.models.requests.GatewayBalances(
    account=cold_wallet.address,
    ledger_index="validated",
    hotwallet=[hot_wallet.address]
))
print(response)
