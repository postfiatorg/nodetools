from enum import Enum
from decimal import Decimal
from pathlib import Path

CONFIG_DIR = Path.home().joinpath("postfiatcreds")

# Super Users
DISCORD_SUPER_USER_IDS = [402536023483088896, 471510026696261632]

# AI MODELS
DEFAULT_OPENROUTER_MODEL = 'anthropic/claude-3.5-sonnet:beta'
DEFAULT_OPEN_AI_MODEL = 'chatgpt-4o-latest'
DEFAULT_ANTHROPIC_MODEL = 'claude-3-5-sonnet-20241022'

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MIN_XRP_PER_TRANSACTION = Decimal('0.000001')  # Minimum XRP amount per transaction
MIN_XRP_BALANCE = 12  # Minimum XRP balance to be able to perform a transaction

# Maximum chunk size for a memo
MAX_MEMO_CHUNK_SIZE = 900

# Maximum length for a commitment sentence
MAX_COMMITMENT_SENTENCE_LENGTH = 950
 
# Maximum history length
MAX_HISTORY = 15  # TODO: rename this to something more descriptive

# Task generation parameters
TASKS_TO_GENERATE = 3

# Context generation limits
MAX_PENDING_PROPOSALS_IN_CONTEXT = 5
MAX_ACCEPTANCES_IN_CONTEXT = 5
MAX_REFUSALS_IN_CONTEXT = 5
MAX_VERIFICATIONS_IN_CONTEXT = 5
MAX_REWARDS_IN_CONTEXT = 5
MAX_CHUNK_MESSAGES_IN_CONTEXT = 10

# Update intervals
TRANSACTION_HISTORY_UPDATE_INTERVAL = 15  # in seconds
TRANSACTION_HISTORY_SLEEP_TIME = 15  # in seconds

# Transaction verification parameters from the node's perspective
NODE_TRANSACTION_VERIFICATION_ATTEMPTS = 12
NODE_TRANSACTION_VERIFICATION_WAIT_TIME = 5  # in seconds

# Transaction verification parameters from the user's perspective
NODE_HANDSHAKE_RESPONSE_USER_VERIFICATION_ATTEMPTS = 24
NODE_HANDSHAKE_RESPONSE_USER_VERIFICATION_INTERVAL = 5  # in seconds

# Reward processing parameters
REWARD_PROCESSING_WINDOW = 35  # in days
MAX_REWARD_AMOUNT = 1200  # in PFT
MIN_REWARD_AMOUNT = 1  # in PFT


# ===MEMO ORGANIZATION===

class SystemMemoType(Enum):
    # SystemMemoTypes cannot be chunked
    INITIATION_REWARD = 'INITIATION_REWARD'  # name is memo_type, value is memo_data pattern
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
