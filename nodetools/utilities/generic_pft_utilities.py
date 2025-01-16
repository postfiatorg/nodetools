# Standard library imports
from decimal import Decimal
from typing import Optional, Union, Any, Dict, List, Any, Tuple
import binascii
import datetime 
import random
import string
import base64
import brotli
import hashlib
import os
import re
import json
import traceback
import asyncio
import math

# Third party imports
import nest_asyncio
import pandas as pd
import requests
import xrpl
from xrpl.wallet import Wallet
from xrpl.models.transactions import Memo
from xrpl.models.response import Response
from xrpl.asyncio.transaction import submit_and_wait
from xrpl.asyncio.clients import AsyncJsonRpcClient
from xrpl.models.requests import AccountInfo, AccountLines, AccountTx
from xrpl.utils import str_to_hex
from loguru import logger

# NodeTools imports
import nodetools.configuration.constants as global_constants
import nodetools.configuration.configuration as config
from nodetools.performance.monitor import PerformanceMonitor
from nodetools.utilities.encryption import MessageEncryption
from nodetools.utilities.transaction_requirements import TransactionRequirementService
from nodetools.utilities.db_manager import DBConnectionManager
from nodetools.utilities.credentials import CredentialManager
from nodetools.utilities.exceptions import *
from nodetools.utilities.xrpl_monitor import XRPLWebSocketMonitor
from nodetools.utilities.transaction_orchestrator import TransactionOrchestrator
from nodetools.utilities.transaction_repository import TransactionRepository
from nodetools.configuration.configuration import NetworkConfig, NodeConfig, RuntimeConfig

nest_asyncio.apply()

class GenericPFTUtilities:
    """Handles general PFT utilities and operations"""
    _instance = None
    _initialized = False

    TX_JSON_FIELDS = [
        'Account', 'DeliverMax', 'Destination', 'Fee', 'Flags',
        'LastLedgerSequence', 'Sequence', 'SigningPubKey', 
        'TransactionType', 'TxnSignature', 'date', 'ledger_index', 'Memos'
    ]

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
            self,
            network_config: NetworkConfig,
            node_config: NodeConfig,
            credential_manager: CredentialManager,
            db_connection_manager: DBConnectionManager,
            transaction_repository: TransactionRepository,
        ):
        if not self.__class__._initialized:
            # Get network and node configurations
            self.network_config = network_config
            self.node_config = node_config
            self.pft_issuer = self.network_config.issuer_address
            self.node_address = self.node_config.node_address
            self.node_name = self.node_config.node_name

            # TODO: Revisit this module
            self.transaction_requirements = TransactionRequirementService(self.network_config, self.node_config)

            # Determine endpoint with fallback logic
            self.https_url = (
                self.network_config.local_rpc_url 
                if RuntimeConfig.HAS_LOCAL_NODE and self.network_config.local_rpc_url is not None
                else self.network_config.public_rpc_url
            )
            logger.debug(f"Using https endpoint: {self.https_url}")

            self.db_connection_manager = db_connection_manager
            self.transaction_repository = transaction_repository
            self.credential_manager = credential_manager
            self.message_encryption: Optional[MessageEncryption] = None  # Requires initialization outside of this class

            self.__class__._initialized = True

    @staticmethod
    def convert_ripple_timestamp_to_datetime(ripple_timestamp = 768602652):
        ripple_epoch_offset = 946684800
        unix_timestamp = ripple_timestamp + ripple_epoch_offset
        date_object = datetime.datetime.fromtimestamp(unix_timestamp)
        return date_object

    @staticmethod
    def is_over_1kb(value: Union[str, int, float]) -> bool:
        if isinstance(value, str):
            # For strings, convert to bytes and check length
            return len(value.encode('utf-8')) > 1024
        elif isinstance(value, (int, float)):
            # For numbers, compare directly
            return value > 1024
        else:
            raise TypeError(f"Expected string or number, got {type(value)}")
        
    @staticmethod
    def to_hex(string):
        return binascii.hexlify(string.encode()).decode()

    @staticmethod
    def hex_to_text(hex_string):
        bytes_object = bytes.fromhex(hex_string)
        try:
            ascii_string = bytes_object.decode("utf-8")
            return ascii_string
        except UnicodeDecodeError:
            return bytes_object  # Return the raw bytes if it cannot decode as utf-8

    @staticmethod
    def generate_random_utf8_friendly_hash(length=6):
        # Generate a random sequence of bytes
        random_bytes = os.urandom(16)  # 16 bytes of randomness
        # Create a SHA-256 hash of the random bytes
        hash_object = hashlib.sha256(random_bytes)
        hash_bytes = hash_object.digest()
        # Encode the hash to base64 to make it URL-safe and readable
        base64_hash = base64.urlsafe_b64encode(hash_bytes).decode('utf-8')
        # Take the first `length` characters of the base64-encoded hash
        utf8_friendly_hash = base64_hash[:length]
        return utf8_friendly_hash

    @staticmethod
    def get_number_of_bytes(text):
        text_bytes = text.encode('utf-8')
        return len(text_bytes)
        
    @staticmethod
    def split_text_into_chunks(text, max_chunk_size=global_constants.MAX_CHUNK_SIZE):
        chunks = []
        text_bytes = text.encode('utf-8')
        for i in range(0, len(text_bytes), max_chunk_size):
            chunk = text_bytes[i:i+max_chunk_size]
            chunk_number = i // max_chunk_size + 1
            chunk_label = f"chunk_{chunk_number}__".encode('utf-8')
            chunk_with_label = chunk_label + chunk
            chunks.append(chunk_with_label)
        return [chunk.decode('utf-8', errors='ignore') for chunk in chunks]

    @staticmethod
    def compress_string(input_string):
        # Compress the string using Brotli
        compressed_data=brotli.compress(input_string.encode('utf-8'))
        # Encode the compressed data to a Base64 string
        base64_encoded_data=base64.b64encode(compressed_data)
        # Convert the Base64 bytes to a string
        compressed_string=base64_encoded_data.decode('utf-8')
        return compressed_string

    @staticmethod
    def decompress_string(compressed_string):
        """Decompress a base64-encoded, brotli-compressed string.
        
        Args:
            compressed_string: The compressed string to decompress
            
        Returns:
            str: The decompressed string
            
        Raises:
            ValueError: If decompression fails after all correction attempts
        """
        # logger.debug(f"GenericPFTUtilities.decompress_string: Decompressing string: {compressed_string}")

        def try_decompress(attempt_string: str) -> Optional[str]:
            """Helper function to attempt decompression with error handling"""
            try:
                base64_decoded = base64.b64decode(attempt_string)
                decompressed = brotli.decompress(base64_decoded)
                return decompressed.decode('utf-8')
            except Exception as e:
                # logger.debug(f"GenericPFTUtilities.decompress_string: Decompression attempt failed: {str(e)}")
                return None
            
        # Try original string first
        result = try_decompress(compressed_string)
        if result:
            return result
        
        # Clean string of invalid base64 characters
        valid_chars = set(string.ascii_letters + string.digits + '+/=')
        cleaned = ''.join(c for c in compressed_string if c in valid_chars)
        # logger.debug(f"GenericPFTUtilities.decompress_string: Cleaned string: {cleaned}")

        # Try with different padding lengths
        for i in range(4):
            padded = cleaned + ('=' * i)
            result = try_decompress(padded)
            if result:
                # logger.debug(f"GenericPFTUtilities.decompress_string: Successfully decompressed with {i} padding chars")
                return result

        # If we get here, all attempts failed
        raise ValueError(
            "Failed to decompress string after all correction attempts. "
            "Original string may be corrupted or incorrectly encoded."
        )

    @staticmethod
    def shorten_url(url):
        api_url="http://tinyurl.com/api-create.php"
        params={'url': url}
        response = requests.get(api_url, params=params)
        if response.status_code == 200:
            return response.text
        else:
            return None
    
    @staticmethod
    def check_if_tx_pft(tx):
        ret= False
        try:
            if tx['Amount']['currency'] == "PFT":
                ret = True
        except:
            pass
        return ret
    
    @staticmethod
    def verify_transaction_response(response: Union[Response, list[Response]] ) -> bool:
        """
        Verify that a transaction response or list of responses indicates success.

        Args:
            response: Transaction response from submit_and_wait

        Returns:
            Tuple[bool, Response]: True if the transaction was successful, False otherwise
        """
        try:
            # Handle list of responses
            if isinstance(response, list):
                return all(
                    GenericPFTUtilities.verify_transaction_response(single_response)
                    for single_response in response
                )

            # Handle single response
            if hasattr(response, 'result'):
                result = response.result
            else:
                result = response

            # Check if transaction was validated and successful
            return (
                result.get('validated', False) and
                result.get('meta', {}).get('TransactionResult', '') == 'tesSUCCESS'
            )
        except Exception as e:
            logger.error(f"Error verifying transaction response: {e}")
            logger.error(traceback.format_exc())
            return False
    
    # TODO: Move to MemoBuilder
    @staticmethod
    def generate_custom_id():
        """ Generate a unique memo_type """
        letters = ''.join(random.choices(string.ascii_uppercase, k=2))
        numbers = ''.join(random.choices(string.digits, k=2))
        second_part = letters + numbers
        date_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        output= date_string+'__'+second_part
        output = output.replace(' ',"_")
        return output

    # TODO: Move to MemoBuilder
    @staticmethod
    def decode_xrpl_memo(memo_dict):
        """Convert hex-encoded memo fields to readable text.
        
        Args:
            memo_dict: Dictionary containing hex-encoded memo fields
                (MemoFormat, MemoType, MemoData)
                
        Returns:
            dict: Dictionary with decoded text values for each memo field
        """
        memo_fields = {
            'MemoFormat': '',
            'MemoType': '',
            'MemoData': ''
        }
        
        for field in memo_fields:
            try:
                if field in memo_dict:
                    memo_fields[field] = GenericPFTUtilities.hex_to_text(memo_dict[field])
            except Exception as e:
                logger.debug(f"Failed to decode {field}: {e}")
                
        return memo_fields
    
    # TODO: Move to MemoBuilder
    @staticmethod
    def calculate_memo_size(memo_format: str, memo_type: str, memo_data: str) -> dict:
        """
        Calculates the size components of a memo using consistent logic.
        
        Args:
            memo_format: The format field (usually username)
            memo_type: The type field (usually task_id)
            memo_data: The data field (the actual content)
            
        Returns:
            dict: Size breakdown including:
                - format_size: Size of hex-encoded format
                - type_size: Size of hex-encoded type
                - data_size: Size of hex-encoded data
                - structural_overhead: Fixed overhead for JSON structure
                - total_size: Total size including all components
        """
        format_size = len(str_to_hex(memo_format))
        type_size = len(str_to_hex(memo_type))
        data_size = len(str_to_hex(memo_data))
        structural_overhead = global_constants.XRP_MEMO_STRUCTURAL_OVERHEAD

        logger.debug(f"Memo size breakdown:")
        logger.debug(f"  format_size: {format_size}")
        logger.debug(f"  type_size: {type_size}")
        logger.debug(f"  data_size: {data_size}")
        logger.debug(f"  structural_overhead: {structural_overhead}")
        logger.debug(f"  total_size: {format_size + type_size + data_size + structural_overhead}")

        return {
            'format_size': format_size,
            'type_size': type_size,
            'data_size': data_size,
            'structural_overhead': structural_overhead,
            'total_size': format_size + type_size + data_size + structural_overhead
        }

    # TODO: Move to MemoBuilder
    @staticmethod
    def construct_memo(memo_format, memo_type, memo_data, validate_size=False):
        """Constructs a memo object, checking total size"""
        # NOTE: This is a hack and appears too conservative
        # NOTE: We don't know if this is the correct way calculate the XRPL size limits
        # NOTE: This will raise an error even when a transaction might otherwise succeed
        if validate_size:
            size_info = GenericPFTUtilities.calculate_memo_size(memo_format, memo_type, memo_data)
            if GenericPFTUtilities.is_over_1kb(size_info['total_size']):
                raise ValueError(f"Memo exceeds 1 KB, raising ValueError: {size_info['total_size']}")

        return Memo(
            memo_data=GenericPFTUtilities.to_hex(memo_data),
            memo_type=GenericPFTUtilities.to_hex(memo_type),
            memo_format=GenericPFTUtilities.to_hex(memo_format)
        )
    
    # TODO: Move to MemoBuilder
    @staticmethod
    def construct_handshake_memo(user, ecdh_public_key):
        """Constructs a handshake memo for encrypted communication"""
        return GenericPFTUtilities.construct_memo(
            memo_data=ecdh_public_key,
            memo_type=global_constants.SystemMemoType.HANDSHAKE.value,
            memo_format=user
        )

    async def send_xrp(
            self,
            wallet_seed_or_wallet: Union[str, xrpl.wallet.Wallet], 
            amount: Union[Decimal, int, float], 
            destination: str, 
            memo: Memo, 
            destination_tag: Optional[int] = None
        ):
        # Handle wallet input
        if isinstance(wallet_seed_or_wallet, str):
            wallet = self.spawn_wallet_from_seed(wallet_seed_or_wallet)
        elif isinstance(wallet_seed_or_wallet, xrpl.wallet.Wallet):
            wallet = wallet_seed_or_wallet
        else:
            logger.error("GenericPFTUtilities.send_xrp: Invalid wallet input, raising ValueError")
            raise ValueError("Invalid wallet input")

        client = AsyncJsonRpcClient(self.https_url)

        payment = xrpl.models.transactions.Payment(
            account=wallet.address,
            amount=xrpl.utils.xrp_to_drops(Decimal(amount)),
            destination=destination,
            memos=[memo],
            destination_tag=destination_tag
        )
        try:    
            response = await submit_and_wait(payment, client, wallet)
            return response    
        except xrpl.transaction.XRPLReliableSubmissionException as e:    
            logger.error(f"GenericPFTUtilities.send_xrp: Transaction submission failed: {e}")
            logger.error(traceback.format_exc())
            raise
        except Exception as e:
            logger.error(f"GenericPFTUtilities.send_xrp: Unexpected error: {e}")
            logger.error(traceback.format_exc())
            raise

    @staticmethod
    def spawn_wallet_from_seed(seed):
        """ outputs wallet initialized from seed"""
        wallet = xrpl.wallet.Wallet.from_seed(seed)
        logger.debug(f'-- Spawned wallet with address {wallet.address}')
        return wallet
    
    @PerformanceMonitor.measure('get_account_memo_history')
    async def get_account_memo_history(self, account_address: str, pft_only: bool = True) -> pd.DataFrame:
        """Get transaction history with memos for an account.
        
        Args:
            account_address: XRPL account address to get history for
            pft_only: If True, only return PFT transactions. Defaults to True.
            
        Returns:
            DataFrame containing transaction history with memo details
        """
        results = await self.transaction_repository.get_account_memo_history(
            account_address=account_address,
            pft_only=pft_only
        )

        df = pd.DataFrame(results)

        # Convert datetime column to datetime after DataFrame creation
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    
    def is_encrypted(self, memo: str):
        """Check if a memo is encrypted"""
        return self.message_encryption.is_encrypted(memo)
    
    async def send_handshake(self, wallet_seed: str, destination: str, username: str = None):
        """Sends a handshake memo to establish encrypted communication"""
        return await self.message_encryption.send_handshake(channel_private_key=wallet_seed, channel_counterparty=destination, username=username)
    
    def register_auto_handshake_wallet(self, wallet_address: str):
        """Register a wallet address for automatic handshake responses."""
        self.message_encryption.register_auto_handshake_wallet(wallet_address)

    def get_auto_handshake_addresses(self) -> set[str]:
        """Get a list of registered auto-handshake addresses"""
        return self.message_encryption.get_auto_handshake_addresses()

    async def get_handshake_for_address(self, channel_address: str, channel_counterparty: str):
        """Get handshake for a specific address"""
        return await self.message_encryption.get_handshake_for_address(channel_address, channel_counterparty)
    
    def get_shared_secret(self, received_public_key: str, channel_private_key: str):
        """
        Get shared secret for a received public key and channel private key.
        The channel private key is the wallet secret.
        """
        return self.message_encryption.get_shared_secret(received_public_key, channel_private_key)

    # TODO: Move to MemoBuilder
    @staticmethod
    def decode_memo_fields_to_dict(memo: Union[xrpl.models.transactions.Memo, dict]):
        """Decodes hex-encoded XRP memo fields from a dictionary to a more readable dictionary format."""
        # Handle xrpl.models.transactions.Memo objects
        if hasattr(memo, 'memo_format'):  # This is a Memo object
            fields = {
                'memo_format': memo.memo_format,
                'memo_type': memo.memo_type,
                'memo_data': memo.memo_data
            }
        else:  # This is a dictionary from transaction JSON
            fields = {
                'memo_format': memo.get('MemoFormat', ''),
                'memo_type': memo.get('MemoType', ''),
                'memo_data': memo.get('MemoData', '')
            }
        
        return {
            key: GenericPFTUtilities.hex_to_text(value or '')
            for key, value in fields.items()
        }
    
    # TODO: Move to MemoBuilder
    @staticmethod
    def calculate_required_chunks(memo: Memo, max_size: int = global_constants.MAX_CHUNK_SIZE) -> int:
        """
        Calculates how many chunks will be needed to send a memo.
        
        Args:
            memo: Original Memo object to analyze
            max_size: Maximum size in bytes for each complete Memo object
            
        Returns:
            int: Number of chunks required
            
        Raises:
            ValueError: If the memo cannot be chunked (overhead too large)
        """
        # Extract memo components
        memo_dict = GenericPFTUtilities.decode_memo_fields_to_dict(memo)
        memo_format = memo_dict['memo_format']
        memo_type = memo_dict['memo_type']
        memo_data = memo_dict['memo_data']

        logger.debug(f"Deconstructed (plaintext) memo sizes: "
                    f"memo_format: {len(memo_format)}, "
                    f"memo_type: {len(memo_type)}, "
                    f"memo_data: {len(memo_data)}")

        # Calculate overhead sizes
        size_info = GenericPFTUtilities.calculate_memo_size(memo_format, memo_type, "chunk_999__")  # assuming chunk_999__ is worst-case chunk label overhead
        max_data_size = max_size - size_info['total_size']

        logger.debug(f"Size allocation:")
        logger.debug(f"  Max size: {max_size}")
        logger.debug(f"  Total overhead: {size_info['total_size']}")
        logger.debug(f"  Available for data: {max_size} - {size_info['total_size']} = {max_data_size}")

        if max_data_size <= 0:
            raise ValueError(
                f"No space for data: max_size={max_size}, total_overhead={size_info['total_size']}"
            )
        
        # Calculate number of chunks needed
        data_bytes = memo_data.encode('utf-8')
        required_chunks = math.ceil(len(data_bytes) / max_data_size)
        required_chunks = 1 if required_chunks == 0 else required_chunks
        return required_chunks
    
    # TODO: Move to MemoBuilder
    @staticmethod
    def _chunk_memos(memo: Memo, max_size: int = global_constants.MAX_CHUNK_SIZE) -> List[Memo]:
        """
        Splits a Memo object into multiple Memo objects, each under MAX_CHUNK_SIZE bytes.
        Only chunks the memo_data field while preserving memo_format and memo_type.
        
        Args:
            memo: Original Memo object to split
            max_size: Maximum size in bytes for each complete Memo object
            
        Returns:
            List of Memo objects, each under max_size bytes
        """
        logger.debug("Chunking memo...")

        # Extract memo components
        memo_dict = GenericPFTUtilities.decode_memo_fields_to_dict(memo)
        memo_format = memo_dict['memo_format']
        memo_type = memo_dict['memo_type']
        memo_data = memo_dict['memo_data']

        # Calculate chunks needed and validate size
        num_chunks = GenericPFTUtilities.calculate_required_chunks(memo, max_size)
        chunk_size = len(memo_data.encode('utf-8')) // num_chunks
                
        # Split into chunks
        chunked_memos = []
        data_bytes = memo_data.encode('utf-8')
        for chunk_number in range(1, num_chunks + 1):
            start_idx = (chunk_number - 1) * chunk_size
            end_idx = start_idx + chunk_size if chunk_number < num_chunks else len(data_bytes)
            chunk = data_bytes[start_idx:end_idx]
            chunk_with_label = f"chunk_{chunk_number}__{chunk.decode('utf-8', errors='ignore')}"

            # Debug the sizes
            test_format = str_to_hex(memo_format)
            test_type = str_to_hex(memo_type)
            test_data = str_to_hex(chunk_with_label)
            
            logger.debug(f"Chunk {chunk_number} sizes:")
            logger.debug(f"  Plaintext Format size: {len(memo_format)}")
            logger.debug(f"  Plaintext Type size: {len(memo_type)}")
            logger.debug(f"  Plaintext Data size: {len(chunk_with_label)}")
            logger.debug(f"  Plaintext Total size: {len(memo_format) + len(memo_type) + len(chunk_with_label)}")
            logger.debug(f"  Hex Format size: {len(test_format)}")
            logger.debug(f"  Hex Type size: {len(test_type)}")
            logger.debug(f"  Hex Data size: {len(test_data)}")
            logger.debug(f"  Hex Total size: {len(test_format) + len(test_type) + len(test_data)}")
            
            chunk_memo = GenericPFTUtilities.construct_memo(
                memo_format=memo_format,
                memo_type=memo_type,
                memo_data=chunk_with_label,
                validate_size=False  # TODO: The size validation appears too conservative
            )

            chunked_memos.append(chunk_memo)

        return chunked_memos

    async def send_memo(self, 
            wallet_seed_or_wallet: Union[str, xrpl.wallet.Wallet], 
            destination: str, 
            memo: Union[str, Memo], 
            username: str = None,
            message_id: str = None,
            chunk: bool = False,
            compress: bool = False, 
            encrypt: bool = False,
            pft_amount: Optional[Decimal] = None,
            disable_pft_check: bool = False
        ) -> Union[Response, list[Response]]:
        """Primary method for sending memos on the XRPL with PFT requirements.
        
        This method handles all aspects of memo sending including:
        - PFT requirement calculation based on destination and memo type
        - Encryption for secure communication (requires prior handshake) TODO: Move this to a MemoBuilder class
        - Compression for large messages TODO: Move this to a MemoBuilder class
        - Automatic chunking for messages exceeding size limits TODO: Move this to a MemoBuilder class
        - Standardized memo formatting TODO: Move this to a MemoBuilder class
        
        Args:
            wallet_seed_or_wallet: Either a wallet seed string or a Wallet object
            destination: XRPL destination address
            memo: Either a string message or pre-constructed Memo object
            username: Optional user identifier for memo format field
            message_id: Optional custom ID for memo type field, auto-generated if None
            chunk: Whether to chunk the memo data (default False)
            compress: Whether to compress the memo data (default False)
            encrypt: Whether to encrypt the memo data (default False)
            pft_amount: Optional specific PFT amount to send. If None, amount will be 
                determined by transaction requirements service.
                
        Returns:
            list[dict]: Transaction responses for each chunk sent
            
        Raises:
            ValueError: If wallet input is invalid
            HandshakeRequiredException: If encryption requested without prior handshake
        """
        # Handle wallet input
        if isinstance(wallet_seed_or_wallet, str):
            wallet = self.spawn_wallet_from_seed(wallet_seed_or_wallet)
            logged_user = f"{username} ({wallet.address})" if username else wallet.address
            logger.debug(f"GenericPFTUtilities.send_memo: Spawned wallet for {logged_user} to send memo to {destination}...")
        elif isinstance(wallet_seed_or_wallet, xrpl.wallet.Wallet):
            wallet = wallet_seed_or_wallet
        else:
            logger.error("GenericPFTUtilities.send_memo: Invalid wallet input, raising ValueError")
            raise ValueError("Invalid wallet input")

        # Extract memo data, type, and format
        if isinstance(memo, Memo):
            memo_data = self.hex_to_text(memo.memo_data)
            memo_type = self.hex_to_text(memo.memo_type)
            memo_format = self.hex_to_text(memo.memo_format)
        else:
            memo_data = str(memo)
            memo_type = message_id or self.generate_custom_id()
            memo_format = username or wallet.classic_address

        # TODO: Adopt a spec for PFT requirements
        # Get per-tx PFT requirement
        if not disable_pft_check:
            pft_amount = pft_amount or self.transaction_requirements.get_pft_requirement(
                address=destination,
                memo_type=memo_type
            )

        # Check if this is a system memo type
        is_system_memo = any(
            memo_type == system_type.value 
            for system_type in global_constants.SystemMemoType
        )

        # Handle encryption if requested
        if encrypt:
            logger.debug(f"GenericPFTUtilities.send_memo: {username} requested encryption. Checking handshake status.")
            channel_key, counterparty_key = await self.message_encryption.get_handshake_for_address(wallet.address, destination)
            if not (channel_key and counterparty_key):
                raise HandshakeRequiredException(wallet.address, destination)
            shared_secret = self.message_encryption.get_shared_secret(counterparty_key, wallet.seed)
            encrypted_memo = self.message_encryption.encrypt_memo(memo_data, shared_secret)
            memo_data = "WHISPER__" + encrypted_memo

        # Handle compression if requested
        if compress:
            logger.debug(f"GenericPFTUtilities.send_memo: {username} requested compression. Compressing memo.")
            compressed_data = self.compress_string(memo_data)
            logger.debug(f"GenericPFTUtilities.send_memo: Compressed memo to length {len(compressed_data)}")
            memo_data = "COMPRESSED__" + compressed_data

        # For system memos, verify size and prevent chunking
        # construct_memo will raise ValueError if size exceeds limit, since SystemMemoTypes cannot be chunked due to collision risk
        memo = self.construct_memo(
            memo_format=memo_format,
            memo_type=memo_type,
            memo_data=memo_data,
            validate_size=(is_system_memo and chunk)
        )

        if is_system_memo:
            return await self._send_memo_single(wallet, destination, memo, pft_amount)

        # Handle chunking for non-system memos if requested, or if _chunk_memos returns more than one memo
        chunk_memos = self._chunk_memos(memo)
        if chunk or len(chunk_memos) > 1:
            try:
                responses = []

                for idx, chunk_memo in enumerate(chunk_memos):
                    logger.debug(f"Sending chunk {idx+1} of {len(chunk_memos)}: {chunk_memo.memo_data[:100]}...")
                    responses.append(await self._send_memo_single(wallet, destination, chunk_memo, pft_amount))

                return responses
            except Exception as e:
                logger.error(f"GenericPFTUtilities.send_memo: Error chunking memo: {e}")
                logger.error(traceback.format_exc())
                raise e
        else:
            return await self._send_memo_single(wallet, destination, memo, pft_amount)

    async def _send_memo_single(self, wallet: Wallet, destination: str, memo: Memo, pft_amount: Decimal) -> Response:
        """ Sends a single memo to a destination """
        client = AsyncJsonRpcClient(self.https_url)

        payment_args = {
            "account": wallet.address,
            "destination": destination,
            "memos": [memo]
        }

        if pft_amount > 0:
            payment_args["amount"] = xrpl.models.amounts.IssuedCurrencyAmount(
                currency="PFT",
                issuer=self.pft_issuer,
                value=str(pft_amount)
            )
        else:
            # Send minimum XRP amount for memo-only transactions
            payment_args["amount"] = xrpl.utils.xrp_to_drops(Decimal(global_constants.MIN_XRP_PER_TRANSACTION))

        payment = xrpl.models.transactions.Payment(**payment_args)

        try:
            logger.debug(f"GenericPFTUtilities._send_memo_single: Submitting transaction to send memo from {wallet.address} to {destination}")
            response = await submit_and_wait(payment, client, wallet)
            return response
        except xrpl.transaction.XRPLReliableSubmissionException as e:
            logger.error(f"GenericPFTUtilities._send_memo_single: Transaction submission failed: {e}")
            logger.error(traceback.format_exc())
            raise
        except Exception as e:
            logger.error(f"GenericPFTUtilities._send_memo_single: Unexpected error: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _reconstruct_chunked_message(
        self,
        memo_type: str,
        memo_history: pd.DataFrame
    ) -> str:
        """Reconstruct a message from its chunks.
        
        Args:
            memo_type: Message ID to reconstruct
            memo_history: DataFrame containing memo history
            account_address: Account address that sent the chunks
            
        Returns:
            str: Reconstructed message or None if reconstruction fails
        """
        try:
            # Get all chunks with this memo type from this account
            memo_chunks = memo_history[
                (memo_history['memo_type'] == memo_type) &
                (memo_history['memo_data'].str.match(r'^chunk_\d+__'))  # Only get actual chunks
            ].copy()

            if memo_chunks.empty:
                return None
            
            # Extract chunk numbers and sort
            def extract_chunk_number(x):
                match = re.search(r'^chunk_(\d+)__', x)
                return int(match.group(1)) if match else 0
            
            memo_chunks['chunk_number'] = memo_chunks['memo_data'].apply(extract_chunk_number)
            memo_chunks = memo_chunks.sort_values('datetime')

            # Detect and handle multiple chunk sequences
            # This is to handle the case when a new message is erroneusly sent with an existing message ID
            current_sequence = []
            highest_chunk_num = 0

            for _, chunk in memo_chunks.iterrows():
                # If we see a chunk_1 and already have chunks, this is a new sequence
                if chunk['chunk_number'] == 1 and current_sequence:
                    # Check if previous sequence was complete (no gaps)
                    expected_chunks = set(range(1, highest_chunk_num + 1))
                    actual_chunks = set(chunk['chunk_number'] for chunk in current_sequence)

                    if expected_chunks == actual_chunks:
                        # First sequence is complete, ignore all subsequent chunks
                        # logger.warning(f"GenericPFTUtilities._reconstruct_chunked_message: Found complete sequence for {memo_type}, ignoring new sequence")
                        break
                    else:
                        # First sequence was incomplete, start fresh with new sequence
                        # logger.warning(f"GenericPFTUtilities._reconstruct_chunked_message: Previous sequence incomplete for {memo_type}, starting new sequence")
                        current_sequence = []
                        highest_chunk_num = 0

                current_sequence.append(chunk)
                highest_chunk_num = max(highest_chunk_num, chunk['chunk_number'])

            # Verify final sequence is complete
            expected_chunks = set(range(1, highest_chunk_num + 1))
            actual_chunks = set(chunk['chunk_number'] for chunk in current_sequence)
            if expected_chunks != actual_chunks:
                # logger.warning(f"GenericPFTUtilities._reconstruct_chunked_message: Missing chunks for {memo_type}. Expected {expected_chunks}, got {actual_chunks}")
                return None

            # Combine chunks in order
            current_sequence.sort(key=lambda x: x['chunk_number'])
            reconstructed_parts = []
            for chunk in current_sequence:
                chunk_data = re.sub(r'^chunk_\d+__', '', chunk['memo_data'])
                reconstructed_parts.append(chunk_data)

            return ''.join(reconstructed_parts)
        
        except Exception as e:
            # logger.error(f"GenericPFTUtilities._reconstruct_chunked_message: Error reconstructing message {memo_type}: {e}")
            return None

    async def process_memo_data(
        self,
        memo_type: str,
        memo_data: str,
        decompress: bool = True,
        decrypt: bool = True,
        full_unchunk: bool = False, 
        memo_history: Optional[pd.DataFrame] = None,
        channel_address: Optional[str] = None,
        channel_counterparty: Optional[str] = None,
        channel_private_key: Optional[Union[str, xrpl.wallet.Wallet]] = None
    ) -> str:
        """Process memo data, handling both single and multi-chunk messages.
        
        For encrypted messages (WHISPER__ prefix), this method handles decryption using ECDH:
        
        Encryption Channel:
        - An encrypted channel exists between two XRPL addresses (channel_address and channel_counterparty)
        - To decrypt a message, you need the private key (channel_private_key) corresponding to one end 
        of the channel (channel_address)
        - It doesn't matter which end was the sender or receiver - what matters is having 
        the private key for channel_address, and the public key for channel_counterparty
        
        Example Usage:
        1. When node has the private key:
            process_memo_data(
                channel_address=node_address,               # The end we have the private key for
                channel_counterparty=other_party_address,   # The other end of the channel
                channel_private_key=node_private_key        # Must correspond to channel_address
            )
        
        2. When we have a user's private key (legacy case):
            process_memo_data(
                channel_address=user_address,             # The end we have the private key for
                channel_counterparty=node_address,        # The other end of the channel
                channel_private_key=user_private_key      # Must correspond to channel_address
            )

        Args:
            memo_type: The memo type to identify related chunks
            memo_data: Initial memo data string
            account_address: One end of the encryption channel - MUST correspond to wallet_seed
            full_unchunk: If True, will attempt to unchunk by referencing memo history
            decompress: If True, decompresses data if COMPRESSED__ prefix is present
            decrypt: If True, decrypts data if WHISPER__ prefix is present
            destination: Required for decryption - the other end of the encryption channel
            memo_history: Optional pre-filtered memo history for chunk lookup
            wallet_seed: Required for decryption - MUST be the private key corresponding 
                        to account_address (not destination)
        
        Raises:
            ValueError: If decrypt=True but wallet_seed is not provided
            ValueError: If decrypt=True but destination is not provided
            ValueError: If wallet_seed provided doesn't correspond to account_address
        """
        try:
            processed_data = memo_data

            # Handle chunking
            if full_unchunk and memo_history is not None:

                # Skip chunk processing for SystemMemoType messages
                is_system_memo = any(
                    memo_type == system_type.value 
                    for system_type in global_constants.SystemMemoType
                )

                # Handle chunking for non-system messages only
                if not is_system_memo:
                    # Check if this is a chunked message
                    chunk_match = re.match(r'^chunk_\d+__', memo_data)
                    if chunk_match:
                        reconstructed = self._reconstruct_chunked_message(
                            memo_type=memo_type,
                            memo_history=memo_history
                        )
                        if reconstructed:
                            processed_data = reconstructed
                        else:
                            # If reconstruction fails, just clean the prefix from the single message
                            # logger.warning(f"GenericPFTUtilities.process_memo_data: Reconstruction of chunked message {memo_type} from {channel_address} failed. Cleaning prefix from single message.")
                            processed_data = re.sub(r'^chunk_\d+__', '', memo_data)
            
            elif isinstance(processed_data, str):
                # Simple chunk prefix removal (no full unchunking)
                processed_data = re.sub(r'^chunk_\d+__', '', processed_data)
                
            # Handle decompression
            if decompress and processed_data.startswith('COMPRESSED__'):
                processed_data = processed_data.replace('COMPRESSED__', '', 1)
                # logger.debug(f"GenericPFTUtilities.process_memo_data: Decompressing data: {processed_data}")
                try:
                    processed_data = self.decompress_string(processed_data)
                except Exception as e:
                    # logger.warning(f"GenericPFTUtilities.process_memo_data: Error decompressing data: {e}")
                    return processed_data

            # Handle encryption
            if decrypt and processed_data.startswith('WHISPER__'):
                if not all([channel_private_key, channel_counterparty, channel_address]):
                    logger.warning(
                        f"GenericPFTUtilities.process_memo_data: Cannot decrypt message {memo_type} - "
                        f"missing required parameters. Need channel_private_key: {bool(channel_private_key)}, "
                        f"channel_counterparty: {bool(channel_counterparty)}, channel_address: {bool(channel_address)}"
                    )
                    return processed_data
                
                # Handle wallet object or seed
                if isinstance(channel_private_key, xrpl.wallet.Wallet):
                    channel_wallet = channel_private_key
                    channel_private_key = channel_private_key.seed
                else:
                    channel_private_key = channel_private_key
                    channel_wallet = xrpl.wallet.Wallet.from_seed(channel_private_key)
                
                # Validate that the channel_private_key passed to this method corresponds to channel_address
                if channel_wallet.classic_address != channel_address:
                    logger.warning(
                        f"GenericPFTUtilities.process_memo_data: Cannot decrypt message {memo_type} - "
                        f"wallet address derived from channel_private_key {channel_wallet.classic_address} does not match channel_address {channel_address}"
                    )
                    return processed_data

                # logger.debug(f"GenericPFTUtilities.process_memo_data: Getting handshake for {channel_address} and {channel_counterparty}")
                channel_key, counterparty_key = await self.message_encryption.get_handshake_for_address(
                    channel_address=channel_address,
                    channel_counterparty=channel_counterparty
                )
                if not (channel_key and counterparty_key):
                    logger.warning(
                        f"GenericPFTUtilities.process_memo_data: Cannot decrypt message {memo_type} - "
                        f"no handshake found between {channel_address} and {channel_counterparty}... "
                        f"channel_key: {channel_key}, counterparty_key: {counterparty_key}"
                    )
                    return processed_data
                
                # Get the shared secret from the handshake key
                shared_secret = self.message_encryption.get_shared_secret(
                    received_public_key=counterparty_key, 
                    channel_private_key=channel_private_key
                )
                # logger.debug(f"GenericPFTUtilities.process_memo_data: Got shared secret for {channel_address} and {channel_counterparty}: {shared_secret}")
                try:
                    processed_data = self.message_encryption.process_encrypted_message(processed_data, shared_secret)
                except Exception as e:
                    message = (
                        f"GenericPFTUtilities.process_memo_data: Error decrypting message {memo_type} "
                        f"between address {channel_address} and counterparty {channel_counterparty}: {processed_data}"
                    )
                    logger.error(message)
                    logger.error(traceback.format_exc())
                    return f"[Decryption Failed] {processed_data}"

            # logger.debug(f"GenericPFTUtilities.process_memo_data: Decrypted data: {processed_data}")
                
            return processed_data
            
        except Exception as e:
            logger.warning(f"GenericPFTUtilities.process_memo_data: Error processing memo {memo_type}: {e}")
            return processed_data
        
    async def get_all_account_compressed_messages_for_remembrancer(
        self,
        account_address: str,
    ) -> pd.DataFrame:
        """Convenience method for getting all messages for a user from the remembrancer's perspective"""
        return await self.get_all_account_compressed_messages(
            account_address=account_address,
            channel_private_key=self.credential_manager.get_credential(
                f"{self.node_config.remembrancer_name}__v1xrpsecret"
            )
        )

    async def get_all_account_compressed_messages(
        self,
        account_address: str,
        channel_private_key: Optional[Union[str, xrpl.wallet.Wallet]] = None,
    ) -> pd.DataFrame:
        """Get all messages for an account, handling chunked messages, compression, and encryption.
        
        This method is designed to be called from the node's perspective and handles two scenarios:
        
        1. Getting messages for a user's address:
        - account_address = user's address
        - channel_counterparty = user's address (for decryption)
        
        2. Getting messages for the remembrancer's address:
        - account_address = remembrancer's address
        - channel_counterparty = user_account from transaction (for decryption)
        
        The method handles:
        - Message chunking/reconstruction
        - Compression/decompression
        - Encryption/decryption using ECDH
        
        For encrypted messages, the encryption channel is established between:
        - One end: remembrancer (whose private key we're using)
        - Other end: user (either account_address or user_account from transaction)
        
        Args:
            account_address: XRPL account address to get history for
            channel_private_key: Private key (wallet seed or wallet) for decryption.
                Required if any messages are encrypted.
                
        Returns:
            DataFrame with columns:
                - memo_type: Message identifier
                - processed_message: Decrypted, decompressed, reconstructed message
                - datetime: Transaction timestamp
                - direction: INCOMING or OUTGOING relative to account_address
                - hash: Transaction hash
                - account: Sender address
                - destination: Recipient address
                - pft_amount: Sum of PFT amounts for all chunks
                
            Returns empty DataFrame if no messages exist or processing fails.
        """
        try:
            # Get transaction history
            memo_history = await self.get_account_memo_history(account_address=account_address, pft_only=True)

            if memo_history.empty:
                return pd.DataFrame()

            # Filter memo_history when getting messages for a user's address
            if account_address != self.node_config.remembrancer_address:
                # Scenario 1: Only include memos where remembrancer is involved
                memo_history = memo_history[
                    (memo_history['account'] == self.node_config.remembrancer_address) |
                    (memo_history['destination'] == self.node_config.remembrancer_address)
                ]
                
                if memo_history.empty:
                    logger.debug(f"No messages found between {account_address} and remembrancer")
                    return pd.DataFrame()

            # Derive channel_address from channel_private_key
            if isinstance(channel_private_key, xrpl.wallet.Wallet):
                channel_address = channel_private_key.classic_address
            else:
                channel_address = xrpl.wallet.Wallet.from_seed(channel_private_key).classic_address

            processed_messages = []
            for msg_id in memo_history['memo_type'].unique():

                msg_txns = memo_history[memo_history['memo_type'] == msg_id]
                first_txn = msg_txns.iloc[0]

                # Determine channel counterparty based on account_address
                # If we're getting messages for a user, they are the counterparty
                # If we're getting messages for the remembrancer, the user_account is the counterparty
                channel_counterparty = (
                    account_address 
                    if account_address != self.node_config.remembrancer_address 
                    else first_txn['user_account']
                )

                try:
                    # Process the message (handles chunking, decompression, and decryption)
                    processed_message = await self.process_memo_data(
                        memo_type=msg_id,
                        memo_data=first_txn['memo_data'],
                        full_unchunk=True,
                        memo_history=memo_history,
                        channel_address=channel_address,
                        channel_counterparty=channel_counterparty,
                        channel_private_key=channel_private_key
                    )
                except Exception as e:
                    processed_message = None

                processed_messages.append({
                    'memo_type': msg_id,
                    'memo_format': first_txn['memo_format'],
                    'processed_message': processed_message if processed_message else "[PROCESSING FAILED]",
                    'datetime': first_txn['datetime'],
                    'direction': first_txn['direction'],
                    'hash': first_txn['hash'],
                    'account': first_txn['account'],
                    'destination': first_txn['destination'],
                    'pft_amount': msg_txns['directional_pft'].sum()
                })

            result_df = pd.DataFrame(processed_messages)
            return result_df
        
        except Exception as e:
            logger.error(f"GenericPFTUtilities.get_all_account_compressed_messages: Error processing memo data for {account_address}: {e}")
            return pd.DataFrame()

    async def fetch_account_transactions(
            self,
            account_address: str,
            ledger_index_min: int = -1,
            ledger_index_max: int = -1,
            max_attempts: int = 3,
            retry_delay: float = 0.2,
            limit: int = 1000
        ) -> list[dict]:
        """Fetch transactions for an account from the XRPL"""
        all_transactions = []  # List to store all transactions
        marker = None  # Fetch transactions using marker pagination
        attempt = 0
        client = AsyncJsonRpcClient(self.https_url)

        while attempt < max_attempts:
            try:
                request = xrpl.models.requests.account_tx.AccountTx(
                    account=account_address,
                    ledger_index_min=ledger_index_min,
                    ledger_index_max=ledger_index_max,
                    limit=limit,
                    marker=marker,
                    forward=True
                )
                response = await client.request(request)
                transactions = response.result["transactions"]
                all_transactions.extend(transactions)

                if "marker" not in response.result:
                    break
                marker = response.result["marker"]

            except Exception as e:
                logger.error(f"GenericPFTUtilities.get_account_transactions: Error occurred while fetching transactions (attempt {attempt + 1}): {str(e)}")
                attempt += 1
                if attempt < max_attempts:
                    logger.debug(f"GenericPFTUtilities.get_account_transactions: Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.warning("GenericPFTUtilities.get_account_transactions: Max attempts reached. Transactions may be incomplete.")
                    break

        return all_transactions

    @staticmethod
    def _extract_field(json_data: dict, field: str) -> Any:
        """Extract a field from JSON data, converting dicts to strings.
    
        Args:
            json_data: JSON dictionary to extract from
            field: Field name to extract
            
        Returns:
            Field value or None if not found/invalid
        """
        try:
            value = json_data.get(field)
            return str(value) if isinstance(value, dict) else value
        except AttributeError:
            return None

    async def fetch_formatted_transaction_history(
            self, 
            account_address: str,
            fetch_new_only: bool = True
        ) -> List[Dict[str, Any]]:
        """Fetch and format transaction history for an account.
        
        Retrieves transactions from XRPL and transforms them into a standardized
        format suitable for database storage.
        
        Args:
            account_address: XRPL account address to fetch transactions for
            fetch_new_only: If True, only fetch transactions after the last known ledger index.
                            If False, fetch entire transaction history.
                
        Returns:
            List of dictionaries containing processed transaction data with standardized fields
        """
        ledger_index_min = -1

        if fetch_new_only:
            # Get the last processed ledger index for this account
            last_ledger = await self.transaction_repository.get_last_ledger_index(account=account_address)
            if last_ledger is not None:
                ledger_index_min = last_ledger + 1

        # Fetch transaction history and prepare DataFrame
        transactions = await self.fetch_account_transactions(
            account_address=account_address, 
            ledger_index_min=ledger_index_min
        )
        if not transactions:
            return []
        
        formatted_transactions = []
        for tx in transactions:
            formatted_tx = {
                # Core transaction fields
                'hash': tx.get('hash'),
                'ledger_index': tx.get('ledger_index'),
                'close_time_iso': tx.get('close_time_iso'),
                'validated': tx.get('validated'),
                
                # Store complete transaction data
                'meta': tx.get('meta', {}),
                'tx_json': tx.get('tx_json', {}),
            }
            
            formatted_transactions.append(formatted_tx)
        
        return formatted_transactions

    async def fetch_pft_trustline_data(self, batch_size: int = 200) -> Dict[str, Dict[str, Any]]:
        """Get PFT token holder account information.
        
        Queries the XRPL for all accounts that have trustlines with the PFT issuer account.
        The balances are from the issuer's perspective, so they are negated to show actual
        holder balances (e.g., if issuer shows -100, holder has +100).

        Args:
            batch_size: Number of records to fetch per request (max 400)

        Returns:
            Dict of dictionaries with keys:
                - account (str): XRPL account address of the token holder
            and values:
                - balance (str): Raw balance string from XRPL
                - currency (str): Currency code (should be 'PFT')
                - limit_peer (str): Trustline limit
                - pft_holdings (float): Actual token balance (negated from issuer view)
        """
        # Initialize result dictionary and marker
        all_lines = {}
        marker = None

        client = AsyncJsonRpcClient(self.https_url)
        while True:
            try:
                # Request account lines with pagination
                request = xrpl.models.requests.AccountLines(
                    account=self.pft_issuer,
                    ledger_index="validated",
                    limit=batch_size,
                    marker=marker
                )
                
                response = await client.request(request)

                # Process this batch of lines
                for line in response.result['lines']:
                    all_lines[line['account']] = {
                        'balance': Decimal(line['balance']),
                        'currency': line['currency'],
                        'limit_peer': line['limit_peer'],
                        'pft_holdings': Decimal(line['balance']) * -1
                    }
                
                # Check if there are more results
                marker = response.result.get('marker')
                if not marker:
                    break
                    
            except Exception as e:
                logger.error(f"Error fetching trustline data batch: {e}")
                logger.error(traceback.format_exc())
                break

        return all_lines

    async def get_pft_holders_async(self) -> Dict[str, Dict[str, Any]]:
        """Get current PFT holder data from database (async version)"""
        try:
            return await self.transaction_repository.get_pft_holders()
        except Exception as e:
            logger.error(f"Error getting PFT holders: {e}")
            logger.error(traceback.format_exc())
            return {}
        
    def get_pft_holders(self) -> Dict[str, Dict[str, Any]]:
        """Get current PFT holder data from database (sync version)"""
        try:
            # If we're already in an event loop, use it
            try:
                loop = asyncio.get_running_loop()
                return loop.run_until_complete(self.get_pft_holders_async())
            except RuntimeError:  # No running event loop
                return asyncio.run(self.get_pft_holders_async())
        except Exception as e:
            logger.error(f"Error getting PFT holders: {e}")
            logger.error(traceback.format_exc())
            return {}
        
    async def get_pft_holder_async(self, account_address: str) -> Dict[str, Any]:
        """Get PFT holder data for an account from the database (async version)"""
        try:
            return await self.transaction_repository.get_pft_holder(account_address)
        except Exception as e:
            logger.error(f"Error getting PFT holder: {e}")
            logger.error(traceback.format_exc())
            return {}

    def get_pft_holder(self, account_address: str) -> Dict[str, Any]:
        """Get PFT holder data for an account from the database (sync version)"""
        try:
            try:
                loop = asyncio.get_running_loop()
                return loop.run_until_complete(self.get_pft_holder_async(account_address))
            except RuntimeError:  # No running event loop
                return asyncio.run(self.get_pft_holder_async(account_address))
        except Exception as e:
            logger.error(f"Error getting PFT holder: {e}")
            logger.error(traceback.format_exc())
            return {}
        
    def get_pft_balance(self, account_address: str) -> Decimal:
        """Get PFT balance for an account from the database"""
        holder = self.get_pft_holder(account_address)
        return holder['balance'] if holder else Decimal(0)

    async def get_recent_user_memos(self, account_address: str, num_messages: int) -> str:
        """Get the most recent messages from a user's memo history.
        
        Args:
            account_address: The XRPL account address to fetch messages for
            num_messages: Number of most recent messages to return (default: 20)
            
        Returns:
            str: JSON string containing datetime-indexed messages
            
        Example:
            >>> get_recent_user_messages("r3UHe45BzAVB3ENd21X9LeQngr4ofRJo5n", 10)
            '{"2024-01-01T12:00:00": "message1", "2024-01-02T14:30:00": "message2", ...}'
        """
        try:
            # Get all messages and select relevant columns
            df = await self.get_all_account_compressed_messages(
                account_address=account_address,
                channel_private_key=self.credential_manager.get_credential(
                    f"{self.node_config.remembrancer_name}__v1xrpsecret"
                )
            )

            if df.empty:
                logger.debug(f"GenericPFTUtilities.get_recent_user_memos: No memo history found for {account_address}. Returning empty JSON")
                return json.dumps({})
        
            messages_df = df[['processed_message', 'datetime']]
            
            # Get most recent messages, sort by time, and convert to JSON
            recent_messages = (messages_df
                .tail(num_messages)
                .sort_values('datetime')
                .set_index('datetime')['processed_message']
                .to_json()
            )

            return recent_messages

        except Exception as e:
            logger.error(f"GenericPFTUtilities.get_recent_user_memos: Failed to get recent user memos for account {account_address}: {e}")
            return json.dumps({})

    def create_xrp_wallet(self):
        test_wallet = Wallet.create()
        classic_address= test_wallet.classic_address
        wallet_seed = test_wallet.seed
        output_string = f"""Wallet Address: {classic_address}
Wallet Secret: {wallet_seed}
        
STORE YOUR WALLET SECRET IN AN OFFLINE PREFERABLY NON DIGITAL LOCATION
THIS MESSAGE WILL AUTO DELETE IN 60 SECONDS
"""
        return output_string
    
    async def fetch_pft_balance(self, address: str) -> Decimal:
        """Get PFT balance for an account from the XRPL.
    
        Args:
            address (str): XRPL account address
            
        Returns:
            Decimal: PFT balance, 0 if no trustline exists
            
        Raises:
            Exception: If there is an error getting the PFT balance
        """
        client = AsyncJsonRpcClient(self.https_url)
        account_lines = AccountLines(
            account=address,
            ledger_index="validated"
        )
        try:
            response = await client.request(account_lines)
            if response.is_successful():
                pft_lines = [line for line in response.result['lines'] if line['account']==self.pft_issuer]
                return Decimal(pft_lines[0]['balance']) if pft_lines else Decimal(0)
        
        except Exception as e:
            logger.error(f"GenericPFTUtilities.fetch_pft_balance: Error getting PFT balance for {address}: {e}")
            logger.error(traceback.format_exc())
            return Decimal(0)
    
    async def fetch_xrp_balance(self, address: str) -> Decimal:
        """Get XRP balance for an account from the XRPL.
        
        Args:
            account_address (str): XRPL account address
            
        Returns:
            Decimal: XRP balance

        Raises:
            XRPAccountNotFoundException: If the account is not found
            Exception: If there is an error getting the XRP balance
        """
        client = AsyncJsonRpcClient(self.https_url)
        acct_info = AccountInfo(
            account=address,
            ledger_index="validated"
        )
        try:
            response = await client.request(acct_info)
            if response.is_successful():
                return Decimal(response.result['account_data']['Balance']) / 1_000_000

        except Exception as e:
            logger.error(f"GenericPFTUtilities.fetch_xrp_balance: Error getting XRP balance: {e}")
            logger.error(traceback.format_exc())
            return Decimal(0)

    async def verify_xrp_balance(self, address: str, minimum_xrp_balance: int) -> bool:
        """
        Verify that a wallet has sufficient XRP balance.
        
        Args:
            wallet: XRPL wallet object
            minimum_balance: Minimum required XRP balance
            
        Returns:
            tuple: (bool, float) - Whether balance check passed and current balance
        """
        balance = await self.fetch_xrp_balance(address)
        return (balance >= minimum_xrp_balance, balance)

    def extract_transaction_info_from_response_object(self, response):
        """
        Extract key information from an XRPL transaction response object.

        Args:
        response (Response): The XRPL transaction response object.

        Returns:
        dict: A dictionary containing extracted transaction information.
        """
        result = response.result
        tx_json = result['tx_json']
        
        # Extract required information
        url_mask = self.network_config.explorer_tx_url_mask
        transaction_info = {
            'time': result['close_time_iso'],
            'amount': tx_json['DeliverMax']['value'],
            'currency': tx_json['DeliverMax']['currency'],
            'send_address': tx_json['Account'],
            'destination_address': tx_json['Destination'],
            'status': result['meta']['TransactionResult'],
            'hash': result['hash'],
            'xrpl_explorer_url': url_mask.format(hash=result['hash'])
        }
        clean_string = (f"Transaction of {transaction_info['amount']} {transaction_info['currency']} "
                        f"from {transaction_info['send_address']} to {transaction_info['destination_address']} "
                        f"on {transaction_info['time']}. Status: {transaction_info['status']}. "
                        f"Explorer: {transaction_info['xrpl_explorer_url']}")
        transaction_info['clean_string']= clean_string
        return transaction_info

    def extract_transaction_info_from_response_object__standard_xrp(self, response):
        """
        Extract key information from an XRPL transaction response object.
        
        Args:
        response (Response): The XRPL transaction response object.
        
        Returns:
        dict: A dictionary containing extracted transaction information.
        """
        transaction_info = {}
        
        try:
            result = response.result if hasattr(response, 'result') else response
            
            transaction_info['hash'] = result.get('hash')
            url_mask = self.network_config.explorer_tx_url_mask
            transaction_info['xrpl_explorer_url'] = url_mask.format(hash=transaction_info['hash'])
            
            tx_json = result.get('tx_json', {})
            transaction_info['send_address'] = tx_json.get('Account')
            transaction_info['destination_address'] = tx_json.get('Destination')
            
            # Handle different amount formats
            if 'DeliverMax' in tx_json:
                transaction_info['amount'] = str(int(tx_json['DeliverMax']) / 1000000)  # Convert drops to XRP
                transaction_info['currency'] = 'XRP'
            elif 'Amount' in tx_json:
                if isinstance(tx_json['Amount'], dict):
                    transaction_info['amount'] = tx_json['Amount'].get('value')
                    transaction_info['currency'] = tx_json['Amount'].get('currency')
                else:
                    transaction_info['amount'] = str(int(tx_json['Amount']) / 1000000)  # Convert drops to XRP
                    transaction_info['currency'] = 'XRP'
            
            transaction_info['time'] = result.get('close_time_iso') or tx_json.get('date')
            transaction_info['status'] = result.get('meta', {}).get('TransactionResult') or result.get('engine_result')
            
            # Create clean string
            clean_string = (f"Transaction of {transaction_info.get('amount', 'unknown amount')} "
                            f"{transaction_info.get('currency', 'XRP')} "
                            f"from {transaction_info.get('send_address', 'unknown sender')} "
                            f"to {transaction_info.get('destination_address', 'unknown recipient')} "
                            f"on {transaction_info.get('time', 'unknown time')}. "
                            f"Status: {transaction_info.get('status', 'unknown')}. "
                            f"Explorer: {transaction_info['xrpl_explorer_url']}")
            transaction_info['clean_string'] = clean_string
            
        except Exception as e:
            transaction_info['error'] = str(e)
            transaction_info['clean_string'] = f"Error extracting transaction info: {str(e)}"
        
        return transaction_info
        
    async def has_trust_line(self, wallet: xrpl.wallet.Wallet) -> bool:
        """Check if wallet has PFT trustline.
        
        Args:
            wallet: XRPL wallet object

        Returns:
            bool: True if trustline exists
        """
        try:
            pft_holders = self.get_pft_holders()
            return wallet.classic_address in pft_holders
        except Exception as e:
            logger.error(f"GenericPFTUtilities.has_trust_line: Error checking if user {wallet.classic_address} has a trust line: {e}")
            return False
        
    async def handle_trust_line(self, wallet: xrpl.wallet.Wallet, username: str):
        """
        Check and establish PFT trustline if needed.
        
        Args:
            wallet: XRPL wallet object
            username: Discord username

        Raises:
            Exception: If there is an error creating the trust line
        """
        logger.debug(f"GenericPFTUtilities.handle_trust_line: Handling trust line for {username} ({wallet.classic_address})")
        if not await self.has_trust_line(wallet):
            logger.debug(f"GenericPFTUtilities.handle_trust_line: Trust line does not exist for {username} ({wallet.classic_address}), creating now...")
            response = await self.generate_trust_line_to_pft_token(wallet)
            if not response.is_successful():
                raise Exception(f"Error creating trust line: {response.result.get('error')}")
        else:
            logger.debug(f"GenericPFTUtilities.handle_trust_line: Trust line already exists for {wallet.classic_address}")

    async def generate_trust_line_to_pft_token(self, wallet: xrpl.wallet.Wallet) -> Response:
        """
        Generate a trust line to the PFT token.
        
        Args:
            wallet: XRPL wallet object
            
        Returns:
            Response: XRPL transaction response

        Raises:
            Exception: If there is an error creating the trust line
        """
        client = AsyncJsonRpcClient(self.https_url)
        trust_set_tx = xrpl.models.transactions.TrustSet(
            account=wallet.classic_address,
            limit_amount=xrpl.models.amounts.IssuedCurrencyAmount(
                currency="PFT",
                issuer=self.pft_issuer,
                value="100000000",
            )
        )
        logger.debug(f"GenericPFTUtilities.generate_trust_line_to_pft_token: Establishing trust line transaction from {wallet.classic_address} to issuer {self.pft_issuer}...")
        try:
            response = await submit_and_wait(trust_set_tx, client, wallet)
            return response
        except xrpl.transaction.XRPLReliableSubmissionException as e:
            logger.error(f"GenericPFTUtilities.generate_trust_line_to_pft_token: Transaction submission failed: {e}")
            logger.error(traceback.format_exc())
            raise
        except Exception as e:
            logger.error(f"GenericPFTUtilities.generate_trust_line_to_pft_token: Unexpected error: {e}")
            logger.error(traceback.format_exc())
            raise

    async def get_recent_messages(self, wallet_address): 
        incoming_messages = None
        outgoing_messages = None
        try:

            memo_history = await self.get_account_memo_history(wallet_address)
            memo_history = memo_history.sort_values('datetime')

            def format_transaction_message(transaction):
                """
                Format a transaction message with specified elements.
                
                Args:
                transaction (pd.Series): A single transaction from the DataFrame.
                
                Returns:
                str: Formatted transaction message.
                """
                url_mask = self.network_config.explorer_tx_url_mask
                return (f"Task ID: {transaction['memo_type']}\n"
                        f"Memo: {transaction['memo_data']}\n"
                        f"PFT Amount: {transaction['directional_pft']}\n"
                        f"Datetime: {transaction['datetime']}\n"
                        f"XRPL Explorer: {url_mask.format(hash=transaction['hash'])}")
            
            # Only try to format if there are matching transactions
            incoming_df = memo_history[memo_history['direction']=='INCOMING']
            if not incoming_df.empty:
                incoming_messages = format_transaction_message(incoming_df.tail(1).iloc[0])
                
            outgoing_df = memo_history[memo_history['direction']=='OUTGOING']
            if not outgoing_df.empty:
                outgoing_messages = format_transaction_message(outgoing_df.tail(1).iloc[0])

        except Exception as e:
            logger.error(f"GenericPFTUtilities.get_recent_messages: Error getting recent messages for {wallet_address}: {e}")
            logger.error(traceback.format_exc())
        
        return incoming_messages, outgoing_messages
    
    @staticmethod
    def remove_chunk_prefix(self, memo_data: str) -> str:
        """Remove chunk prefix from memo data if present.
        
        Args:
            memo_data: Raw memo data string
                
        Returns:
            str: Memo data with chunk prefix removed if present, otherwise unchanged
        """
        return re.sub(r'^chunk_\d+__', '', memo_data)
