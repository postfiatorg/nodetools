from dataclasses import dataclass
from typing import Optional, List
from enum import Enum
from decimal import Decimal

# Runtime configuration
USE_TESTNET = True      
HAS_LOCAL_NODE = False

# TESTNET ONLY
ENABLE_REINITIATIONS = False

class AddressType(Enum):
    """Types of special addresses"""
    NODE = "Node"   # Each node has an address
    REMEMBRANCER = "Remembrancer"  # Each node may have a separate address for its remembrancer
    ISSUER = "Issuer"  # There's only one PFT issuer per L1 network
    OTHER = "Other"  # Any other address type, including users

# PFT requirements by address type
# TODO: Make this dynamic based on operation
PFT_REQUIREMENTS = {
    AddressType.NODE: 1,
    AddressType.REMEMBRANCER: 1,
    AddressType.ISSUER: 0,
    AddressType.OTHER: 0
}

# TODO: Move this out of constants.py
@dataclass
class NetworkConfig:
    """Configuration for an XRPL network (mainnet or testnet)"""
    name: str
    node_name: str
    node_address: str
    remembrancer_name: str
    remembrancer_address: str
    issuer_address: str
    websockets: List[str]
    public_rpc_url: str
    discord_guild_id: int
    discord_activity_channel_id: int
    explorer_tx_url_mask: str
    local_rpc_url: Optional[str] = None

    def get_address_type(self, address: str) -> AddressType:
        """Get the type of address"""
        if address == self.node_address:
            return AddressType.NODE
        elif address == self.remembrancer_address:
            return AddressType.REMEMBRANCER
        elif address == self.issuer_address:
            return AddressType.ISSUER
        else:
            return AddressType.OTHER
        
    def get_pft_requirement(self, address: str) -> Decimal:
        """Get the PFT requirement for an address"""
        return Decimal(PFT_REQUIREMENTS[self.get_address_type(address)])

# Network configurations

XRPL_MAINNET = NetworkConfig(
    name="mainnet",
    node_name="postfiatfoundation",
    node_address="r4yc85M1hwsegVGZ1pawpZPwj65SVs8PzD",
    remembrancer_name="postfiatfoundation_remembrancer",
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
    remembrancer_name="postfiatfoundation_testnet_remembrancer",
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

MIN_XRP_PER_TRANSACTION = 0.00001  # Minimum XRP amount per transaction
MIN_XRP_BALANCE = 12  # Minimum XRP balance to be able to perform a transaction

# Maximum chunk size for a memo
MAX_MEMO_CHUNK_SIZE = 760
 
# Maximum history length
MAX_HISTORY = 15  # TODO: rename this to something more descriptive

# Task generation parameters
TASKS_TO_GENERATE = 3

# Context generation limits
MAX_ACCEPTANCES_IN_CONTEXT = 6
MAX_REFUSALS_IN_CONTEXT = 6
MAX_REWARDS_IN_CONTEXT = 10
MAX_CHUNK_MESSAGES_IN_CONTEXT = 20

# Update intervals
TRANSACTION_HISTORY_UPDATE_INTERVAL = 30  # in seconds
TRANSACTION_HISTORY_SLEEP_TIME = 30  # in seconds

# Transaction verification parameters
TRANSACTION_VERIFICATION_ATTEMPTS = 12
TRANSACTION_VERIFICATION_WAIT_TIME = 5  # in seconds

# Reward processing parameters
REWARD_PROCESSING_WINDOW = 35  # in days
MAX_REWARD_AMOUNT = 1200  # in PFT
MIN_REWARD_AMOUNT = 1  # in PFT


# ===MEMO ORGANIZATION===

class SystemMemoType(Enum):
    INITIATION_REWARD = 'INITIATION_REWARD ___ '  # name is memo_type, value is memo_data pattern
    HANDSHAKE = 'HANDSHAKE'
    INITIATION_RITE = 'INITIATION_RITE'
    GOOGLE_DOC_CONTEXT_LINK = 'google_doc_context_link'
    INITIATION_GRANT = 'discord_wallet_funding'

SYSTEM_MEMO_TYPES = [memo_type.value for memo_type in SystemMemoType]

# Task types where the memo_type = task_id, requiring further disambiguation in the memo_data
class TaskType(Enum):
    REQUEST_POST_FIAT = 'REQUEST_POST_FIAT ___ '
    PROPOSAL = 'PROPOSED PF ___ '
    ACCEPTANCE = 'ACCEPTANCE REASON ___ '
    REFUSAL = 'REFUSAL REASON ___ '
    TASK_OUTPUT = 'COMPLETION JUSTIFICATION ___ '
    VERIFICATION_PROMPT = 'VERIFICATION PROMPT ___ '
    VERIFICATION_RESPONSE = 'VERIFICATION RESPONSE ___ '
    REWARD = 'REWARD RESPONSE __ '
    USER_GENESIS = 'USER GENESIS __ '  # TODO: Deprecate this

# Additional patterns for specific task types
TASK_PATTERNS = {
    TaskType.PROPOSAL: [" .. ", TaskType.PROPOSAL.value],  # Include both patterns
    # Add any other task types that might have multiple patterns
}

# Default patterns for other task types
for task_type in TaskType:
    if task_type not in TASK_PATTERNS:
        TASK_PATTERNS[task_type] = [task_type.value]

# Helper to get all task indicators
TASK_INDICATORS = [task_type.value for task_type in TaskType]

class MessageType(Enum):
    MEMO = 'chunk_'

# Helper to get all message indicators
MESSAGE_INDICATORS = [message_type.value for message_type in MessageType]
