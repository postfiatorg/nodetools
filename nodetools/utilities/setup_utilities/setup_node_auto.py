import json
import os

import xrpl

from nodetools.utilities.credentials import CredentialManager, get_credentials_directory

def setup_node_auto():        
    network = os.environ['NETWORK']
    node_name = os.environ['NODE_NAME']
    network_suffix = '_testnet' if network == 'testnet' else ''
    encryption_password = os.environ['ENCRYPTION_PASSWORD']
    credentials_dict = {
        f'{node_name}{network_suffix}_postgresconnstring': os.environ['PG_CONN_STRING'],
        f'{node_name}{network_suffix}__v1xrpsecret': os.environ['PFT_XRP_WALLET'],
        'openrouter': os.environ['OPENROUTER_API_KEY'],
        'openai': os.environ['OPENAI_API_KEY'],
        'anthropic': os.environ['ANTHROPIC_API_KEY'],
        f'{node_name}{network_suffix}_remembrancer__v1xrpsecret': os.environ['REMEMBRANCER_WALLET'],
        f'discordbot{network_suffix}_secret': os.environ['DISCORD_BOT_TOKEN'],
    }
    config = {
        'node_name': f"{node_name}{network_suffix}",
        'auto_handshake_addresses': [],
        'discord_guild_id': os.environ['DISCORD_GUILD_ID'],
        'discord_activity_channel_id': os.environ['DISCORD_ACTIVITY_CHANNEL_ID'],
    }

    node_wallet = xrpl.wallet.Wallet.from_seed(credentials_dict[f'{node_name}{network_suffix}__v1xrpsecret'])
    config['node_address'] = node_wallet.classic_address

    remembrancer_wallet = xrpl.wallet.Wallet.from_seed(seed=credentials_dict[f'{node_name}{network_suffix}_remembrancer__v1xrpsecret'])
    config['remembrancer_name'] = f"{node_name}{network_suffix}_remembrancer"
    config['remembrancer_address'] = remembrancer_wallet.classic_address

    config_dir = get_credentials_directory()
    config_file = config_dir / f"pft_node_{'testnet' if network == 'testnet' else 'mainnet'}_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    cm: CredentialManager = CredentialManager(encryption_password)
    cm.enter_and_encrypt_credential(credentials_dict)

def main():
    setup_node_auto()

if __name__ == "__main__":
    main()