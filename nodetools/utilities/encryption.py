from typing import Optional, Union, ClassVar
import base64
import hashlib
from cryptography.fernet import Fernet
import pandas as pd
from nodetools.protocols.generic_pft_utilities import GenericPFTUtilities
import nodetools.configuration.configuration as config
import nodetools.configuration.constants as constants
from nodetools.utilities.ecdh import ECDHUtils
import re
from loguru import logger

class MessageEncryption:
    """Handles encryption/decryption of messages using ECDH-derived shared secrets"""

    _instance: ClassVar[Optional['MessageEncryption']] = None
    _initialized = False
    WHISPER_PREFIX = 'WHISPER__'

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, pft_utilities: GenericPFTUtilities):
        if not self.__class__._initialized:
            self.pft_utilities = pft_utilities
            self._auto_handshake_wallets = set()  # Store addresses that should auto-respond to handshakes
            self._handshake_cache = {}  # Cache of addresses to public keys
            self.__class__._initialized = True
    
    def get_auto_handshake_addresses(self) -> set[str]:
        """Returns a set of registered auto-handshake addresses"""
        if not self._auto_handshake_wallets:
            # Initialize from node config if empty
            node_config = config.get_node_config()
            self._auto_handshake_wallets = node_config.auto_handshake_addresses
            logger.debug(f"Initialized auto-handshake addresses: {self._auto_handshake_wallets}")
        return self._auto_handshake_wallets

    @staticmethod
    def is_encrypted(message: str) -> bool:
        """Check if a message is encrypted by looking for the WHISPER prefix"""
        return message.startswith(MessageEncryption.WHISPER_PREFIX)
    
    @staticmethod
    def encrypt_message(message: Union[str, bytes], shared_secret: Union[str, bytes]) -> str:
        """
        Encrypt a memo using a shared secret.
        
        Args:
            message: Message content to encrypt (string or bytes)
            shared_secret: The shared secret derived from ECDH
            
        Returns:
            str: Encrypted message content (without WHISPER prefix)
            
        Raises:
            ValueError: If message is neither string nor bytes
        """
        # Convert shared_secret to bytes if it isn't already
        if isinstance(shared_secret, str):
            shared_secret = shared_secret.encode()

        # Generate Fernet key from shared secret
        key = base64.urlsafe_b64encode(hashlib.sha256(shared_secret).digest())
        fernet = Fernet(key)

        # Handle message input type
        if isinstance(message, str):
            message = message.encode()
        elif isinstance(message, bytes):
            pass
        else:
            raise ValueError(f"Message must be string or bytes, not {type(message)}")
        
        # Encrypt and return as string
        encrypted_bytes = fernet.encrypt(message)
        return encrypted_bytes.decode()
    
    @staticmethod
    def decrypt_message(encrypted_content: str, shared_secret: Union[str, bytes]) -> Optional[str]:
        """
        Decrypt a message using a shared secret.
        
        Args:
            encrypted_content: The encrypted message content (without WHISPER prefix)
            shared_secret: The shared secret derived from ECDH
            
        Returns:
            Decrypted message or None if decryption fails
        """
        # Ensure shared_secret is bytes
        if isinstance(shared_secret, str):
            shared_secret = shared_secret.encode()

        # Generate a Fernet key from the shared secret
        key = base64.urlsafe_b64encode(hashlib.sha256(shared_secret).digest())
        fernet = Fernet(key)

        # Decrypt the message
        decrypted_bytes = fernet.decrypt(encrypted_content.encode())
        return decrypted_bytes.decode()
        
    @staticmethod
    def process_encrypted_message(message: str, shared_secret: bytes) -> str:
        """
        Process a potentially encrypted message.
        
        Args:
            message: The message to process
            shared_secret: The shared secret for decryption
            
        Returns:
            Decrypted message if encrypted and decryption succeeds,
            original message otherwise

        Raises:
            ValueError: If the message fails decryption
        """
        if not MessageEncryption.is_encrypted(message):
            return message
        
        encrypted_content = message.replace(MessageEncryption.WHISPER_PREFIX, '')
        # logger.debug(f"MessageEncryption.process_encrypted_message: Decrypting {encrypted_content}...")
        decrypted_message = MessageEncryption.decrypt_message(encrypted_content, shared_secret)

        return f"[Decrypted] {decrypted_message}"
    
    @staticmethod
    def prepare_encrypted_message(message: str, shared_secret: Union[str, bytes]) -> str:
        """
        Encrypt a message and add the WHISPER prefix.

        Args:
            message: The message to encrypt
            shared_secret: The shared secret for encryption
            
        Returns:
            str: Encrypted message with WHISPER prefix
        """
        encrypted_content = MessageEncryption.encrypt_message(message, shared_secret)
        return f"{MessageEncryption.WHISPER_PREFIX}{encrypted_content}"

    @staticmethod
    def encrypt_memo(memo: str, shared_secret: str) -> str:
        """ Encrypts a memo using a shared secret """
        # Convert shared_secret to bytes if it isn't already
        if isinstance(shared_secret, str):
            shared_secret = shared_secret.encode()

        # Generate the Fernet key from shared secret
        key = base64.urlsafe_b64encode(hashlib.sha256(shared_secret).digest())
        fernet = Fernet(key)

        # Ensure memo is str before encoding to bytes
        if isinstance(memo, str):
            memo = memo.encode()
        elif isinstance(memo, bytes):
            pass
        else:
            raise ValueError(f"Memo must be string or bytes, not {type(memo)}")
        
        # Encrypt and return as string
        encrypted_bytes = fernet.encrypt(memo)
        return encrypted_bytes.decode()
    
    @staticmethod
    def get_ecdh_public_key_from_seed(wallet_seed: str) -> str:
        """Get ECDH public key directly from a wallet seed"""
        return ECDHUtils.get_ecdh_public_key_from_seed(wallet_seed)
    
    @staticmethod
    def get_shared_secret(received_public_key: str, channel_private_key: str) -> bytes:
        """Derive a shared secret using ECDH"""
        return ECDHUtils.get_shared_secret(received_public_key, channel_private_key)
    
    def register_auto_handshake_wallet(self, wallet_address: str):
        """Register a wallet address for automatic handshake responses."""
        if not wallet_address.startswith('r'):
            raise ValueError("Invalid XRPL address")
        self._auto_handshake_wallets.add(wallet_address)
        logger.debug(f"MessageEncryption.register_auto_handshake_wallet: Registered {wallet_address} for automatic handshake responses")
    
    def get_handshake_for_address(
            self, 
            channel_address: str, 
            channel_counterparty: str, 
            memo_history: Optional[pd.DataFrame] = None
        ) -> tuple[Optional[str], Optional[str]]:
        """Get handshake public keys between two addresses.
        
        Args:
            channel_address: One end of the encryption channel
            channel_counterparty: The other end of the encryption channel
            memo_history: Optional pre-filtered memo history. If None, will be fetched.
            
        Returns:
            Tuple of (channel_address's ECDH public key, channel_counterparty's ECDH public key)
        """
        try:
            # Check the cache first
            cache_key = (channel_address, channel_counterparty)
            # Check if both keys are cached and neither key is None
            if self._handshake_cache.get(cache_key) and None not in self._handshake_cache[cache_key]:
                return self._handshake_cache[cache_key]

            if not self.pft_utilities:
                raise ValueError("PFT utilities not initialized")
            
            # Validate addresses
            if not (channel_address.startswith('r') and channel_counterparty.startswith('r')):
                logger.error(f"MessageEncryption.get_handshake_for_address: Invalid XRPL addresses provided: {channel_address}, {channel_counterparty}")
                raise ValueError("Invalid XRPL addresses provided")
            
            # Get memo history if not provided
            if memo_history is None:
                memo_history = self.pft_utilities.get_account_memo_history(
                    account_address=channel_address,
                    pft_only=False
                )

            # Filter for handshakes
            handshakes = memo_history[
                memo_history['memo_type'] == constants.SystemMemoType.HANDSHAKE.value
            ]

            if handshakes.empty:
                return None, None
            
            # Function to clean chunk prefixes
            # TODO: move this to a more general transaction-processing utility
            def clean_chunk_prefix(memo_data: str) -> str:
                return re.sub(r'^chunk_\d+__', '', memo_data)
            
            # Check for sent handshake
            sent_handshakes = handshakes[
                (handshakes['user_account'] == channel_counterparty) & 
                (handshakes['direction'] == 'OUTGOING')
            ]
            sent_key = None
            if not sent_handshakes.empty:
                latest_sent = sent_handshakes.sort_values('datetime').iloc[-1]
                sent_key = clean_chunk_prefix(latest_sent['memo_data'])

            # Check for received handshake and get latest public key
            received_handshakes = handshakes[
                (handshakes['user_account'] == channel_counterparty) &
                (handshakes['direction'] == 'INCOMING')
            ]
            received_key = None
            if not received_handshakes.empty:
                latest_received = received_handshakes.sort_values('datetime').iloc[-1]
                received_key = clean_chunk_prefix(latest_received['memo_data'])

            # Cache the result and return
            result = (sent_key, received_key)
            self._handshake_cache[cache_key] = result
            return result
        
        except Exception as e:
            logger.error(f"MessageEncryption.get_handshake_for_address: Error checking handshake status: {e}")
            raise ValueError(f"Failed to get handshake status: {e}") from e

    @staticmethod
    def get_pending_handshakes(memo_history: pd.DataFrame, channel_counterparty: str) -> pd.DataFrame:
        """Get pending handshakes that need responses for a specific address.
        
        Args:
            memo_history: DataFrame containing memo history
            channel_counterparty: Address to check for pending handshakes
            
        Returns:
            DataFrame containing pending handshake requests
        """
        return memo_history[
            (memo_history['memo_type'] == constants.SystemMemoType.HANDSHAKE.value) &
            (memo_history['destination'] == channel_counterparty) &
            ~memo_history['account'].isin(  # Exclude accounts that have received responses
                memo_history[
                    (memo_history['memo_type'] == constants.SystemMemoType.HANDSHAKE.value) &
                    (memo_history['account'] == channel_counterparty)
                ]['destination'].unique()
            )
        ]

    def send_handshake(self, channel_private_key: str, channel_counterparty: str, username: str = None) -> bool:
        """Send a handshake transaction containing the ECDH public key.
        
        Args:
            channel_private_key: Private key for this end of the channel
            channel_counterparty: Address of the other end of the channel
            username: Optional username to include in logging
            
        Returns:
            bool: True if handshake sent successfully
        """
        try:
            # Get ECDH public key
            public_key = self.get_ecdh_public_key_from_seed(channel_private_key)
            
            # Construct handshake memo
            handshake_memo = self.pft_utilities.construct_handshake_memo(
                user=username,
                ecdh_public_key=public_key
            )
            
            # Send transaction
            wallet = self.pft_utilities.spawn_wallet_from_seed(channel_private_key)
            log_message_source = f"{username} ({wallet.address})" if username else wallet.address
            logger.debug(f"MessageEncryption.send_handshake: Sending handshake from {log_message_source} to {channel_counterparty}...")
            response = self.pft_utilities.send_memo(
                wallet_seed_or_wallet=channel_private_key, 
                destination=channel_counterparty, 
                memo=handshake_memo,
                username=username
            )
            if not self.pft_utilities.verify_transaction_response(response):
                logger.error(f"MessageEncryption.send_handshake: Failed to send handshake from {log_message_source} to {channel_counterparty}")
                return False
            return True
            
        except Exception as e:
            logger.error(f"Error sending handshake: {e}")
            return False