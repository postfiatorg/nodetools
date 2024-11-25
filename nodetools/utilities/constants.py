from dataclasses import dataclass
from typing import Optional, List
from enum import Enum

# Runtime configuration
USE_TESTNET = True      
HAS_LOCAL_NODE = False
TESTNET_MODE = True

@dataclass
class NetworkConfig:
    """Configuration for an XRPL network (mainnet or testnet)"""
    name: str
    node_name: str
    node_address: str
    remembrancer_address: str
    issuer_address: str
    websockets: List[str]
    public_rpc_url: str
    discord_guild_id: int
    discord_activity_channel_id: int
    explorer_tx_url_mask: str
    local_rpc_url: Optional[str] = None

# Network configurations

XRPL_MAINNET = NetworkConfig(
    name="mainnet",
    node_name="postfiatfoundation",
    node_address="r4yc85M1hwsegVGZ1pawpZPwj65SVs8PzD",
    remembrancer_address="rJ1mBMhEBKack5uTQvM8vWoAntbufyG9Yn",
    issuer_address="rnQUEEg8yyjrwk9FhyXpKavHyCRJM9BDMW",
    websockets=[
        "wss://xrplcluster.com", 
        "wss://xrpl.ws/", 
        "wss://s1.ripple.com/", 
        "wss://s2.ripple.com/"
    ],
    public_rpc_url="https://s2.ripple.com:51234",
    local_rpc_url='http://127.0.0.1:5005',
    discord_guild_id=1061800464045310053,
    discord_activity_channel_id=1239280089699450920,
    explorer_tx_url_mask='https://livenet.xrpl.org/transactions/{hash}/detailed'
)

XRPL_TESTNET = NetworkConfig(
    name="testnet",
    node_name="postfiatfoundation_testnet",
    node_address="rUWuJJLLSH5TUdajVqsHx7M59Vj3P7giQV",
    remembrancer_address="rN2oaXBhFE9urGN5hXup937XpoFVkrnUhu",
    issuer_address="rLX2tgumpiUE6kjr757Ao8HWiJzC8uuBSN",
    websockets=[
        "wss://s.altnet.rippletest.net:51233"
    ],
    public_rpc_url="https://s.altnet.rippletest.net:51234",
    local_rpc_url=None,  # No local node for testnet yet
    discord_guild_id=510536760367906818,
    discord_activity_channel_id=1308884322199277699,
    explorer_tx_url_mask='https://testnet.xrpl.org/transactions/{hash}/detailed'
)

class Network(Enum):
    XRPL_MAINNET = XRPL_MAINNET
    XRPL_TESTNET = XRPL_TESTNET

# Helper function to get current network config 
def get_network_config(network: Network = Network.XRPL_TESTNET if USE_TESTNET else Network.XRPL_MAINNET) -> NetworkConfig:
    """Get network configuration based on Network enum.
    
    Args:
        network: Network enum value, defaults to testnet/mainnet based on USE_TESTNET
        
    Returns:
        NetworkConfig: Configuration for the specified network
    """
    return network.value

# ===OTHER CONSTANTS===

# DEFAULT OPEN AI MODEL
DEFAULT_OPEN_AI_MODEL = 'chatgpt-4o-latest'
DEFAULT_ANTHROPIC_MODEL = 'claude-3-5-sonnet-20241022'
MAX_HISTORY = 15 
TRANSACTION_HISTORY_UPDATE_INTERVAL = 30  # in seconds
