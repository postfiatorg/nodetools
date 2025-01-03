from dataclasses import dataclass
from typing import Optional, List
from loguru import logger
import json
import os
from pathlib import Path
import nodetools.configuration.constants as constants

@dataclass
class NetworkConfig:
    """Configuration for an XRPL network (mainnet or testnet)"""
    name: str
    issuer_address: str
    websockets: List[str]
    public_rpc_url: str
    explorer_tx_url_mask: str
    local_rpc_url: Optional[str] = None

@dataclass
class NodeConfig:
    """Configuration for a Post Fiat node"""
    node_name: str
    node_address: str
    auto_handshake_addresses: set[str]  # Addresses that auto-respond to handshakes
    remembrancer_name: Optional[str] = None
    remembrancer_address: Optional[str] = None
    discord_guild_id: Optional[int] = None
    discord_activity_channel_id: Optional[int] = None

    def __post_init__(self):
        """Validate configuration and set defaults"""
        # Always include node address
        self.auto_handshake_addresses.add(self.node_address)
        
        # Add remembrancer address if configured
        if self.remembrancer_address and self.remembrancer_name:
            self.auto_handshake_addresses.add(self.remembrancer_address)

class RuntimeConfig:
    """Runtime configuration settings"""
    USE_TESTNET: bool = True
    HAS_LOCAL_NODE: bool = False
    # TESTNET ONLY - only use these in conjunction with USE_TESTNET (i.e. USE_TESTNET & ENABLE_REINITIATIONS must both be true)
    USE_OPENROUTER_AUTOROUTER: bool = True
    ENABLE_REINITIATIONS: bool = False
    DISABLE_PFT_REQUIREMENTS: bool = False

# Network configurations
XRPL_MAINNET = NetworkConfig(
    name="mainnet",
    issuer_address="rnQUEEg8yyjrwk9FhyXpKavHyCRJM9BDMW",
    websockets=[
        "wss://xrplcluster.com", 
        "wss://xrpl.ws/", 
        "wss://s1.ripple.com/", 
        "wss://s2.ripple.com/"
    ],
    public_rpc_url="https://s2.ripple.com:51234",
    local_rpc_url='http://127.0.0.1:5005',
    explorer_tx_url_mask='https://livenet.xrpl.org/transactions/{hash}/detailed'
)

XRPL_TESTNET = NetworkConfig(
    name="testnet",
    issuer_address="rLX2tgumpiUE6kjr757Ao8HWiJzC8uuBSN",
    websockets=[
        "wss://s.altnet.rippletest.net:51233"
    ],
    public_rpc_url="https://s.altnet.rippletest.net:51234",
    local_rpc_url=None,  # No local node for testnet yet
    explorer_tx_url_mask='https://testnet.xrpl.org/transactions/{hash}/detailed'
)

# Node configurations
MAINNET_NODE = NodeConfig(
    node_name="postfiatfoundation",
    node_address="r4yc85M1hwsegVGZ1pawpZPwj65SVs8PzD",
    remembrancer_name="postfiatfoundation_remembrancer",
    remembrancer_address="rJ1mBMhEBKack5uTQvM8vWoAntbufyG9Yn",
    discord_guild_id=1061800464045310053,
    discord_activity_channel_id=1239280089699450920,
    auto_handshake_addresses=set()  # use defaults
)

TESTNET_NODE = NodeConfig(
    node_name="postfiatfoundation_testnet",
    node_address="rUWuJJLLSH5TUdajVqsHx7M59Vj3P7giQV",
    remembrancer_name="postfiatfoundation_testnet_remembrancer",
    remembrancer_address="rN2oaXBhFE9urGN5hXup937XpoFVkrnUhu",
    discord_guild_id=510536760367906818,
    discord_activity_channel_id=1308884322199277699,
    auto_handshake_addresses=set()  # use defaults
)

def get_network_config() -> NetworkConfig:
    """Get current network configuration based on runtime settings"""
    return XRPL_TESTNET if RuntimeConfig.USE_TESTNET else XRPL_MAINNET

def get_node_config() -> NodeConfig:
    """Get current node configuration based on runtime settings"""
    config_dir = constants.CONFIG_DIR
    config_dir.mkdir(exist_ok=True)
    network = 'testnet' if RuntimeConfig.USE_TESTNET else 'mainnet'
    config_file = config_dir / f"pft_node_{network}_config.json"
    
    if not config_file.exists():
        # Fall back to default configs temporarily
        logger.warning(f"No configuration file found at {config_file}, using default configuration")
        return constants.TESTNET_NODE if RuntimeConfig.USE_TESTNET else constants.MAINNET_NODE
    
    return load_node_config(config_file)

def load_node_config(config_path: str | Path) -> NodeConfig:
    """Load node configuration from JSON file"""
    with open(config_path, 'r') as file:
        config_data = json.load(file)
    return NodeConfig(
        node_name=config_data['node_name'],
        node_address=config_data['node_address'],
        remembrancer_name=config_data.get('remembrancer_name'),
        remembrancer_address=config_data.get('remembrancer_address'),
        discord_guild_id=config_data.get('discord_guild_id'),
        discord_activity_channel_id=config_data.get('discord_activity_channel_id'),
        auto_handshake_addresses=set(config_data.get('auto_handshake_addresses', []))
    )
