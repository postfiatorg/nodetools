from cryptography.fernet import Fernet
import base64
import hashlib
from typing import Optional, Union
from xrpl.core import addresscodec
from xrpl.core.keypairs.ed25519 import ED25519
import nacl.bindings
import nacl.signing

class MessageEncryption:
    """Handles encryption/decryption of messages using ECDH-derived shared secrets"""

    WHISPER_PREFIX = 'WHISPER__'

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
    def decrypt_message(encrypted_content: str, shared_secret: bytes) -> Optional[str]:
        """
        Decrypt a message using a shared secret.
        
        Args:
            encrypted_content: The encrypted message content (without WHISPER prefix)
            shared_secret: The shared secret derived from ECDH
            
        Returns:
            Decrypted message or None if decryption fails
        """
        try:
            # Generate a Fernet key from the shared secret
            key = base64.urlsafe_b64encode(hashlib.sha256(shared_secret).digest())
            fernet = Fernet(key)

            # Decrypt the message
            decrypted_bytes = fernet.decrypt(encrypted_content.encode())
            return decrypted_bytes.decode()

        except Exception as e:
            print(f"Error decrypting message: {e}")
            return None
        
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
        """
        if not MessageEncryption.is_encrypted(message):
            return message
        
        encrypted_content = message.replace(MessageEncryption.WHISPER_PREFIX, '')
        decrypted_message = MessageEncryption.decrypt_message(encrypted_content, shared_secret)

        if decrypted_message is None:
            return f"[Decryption failed] {message}"

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
    def _get_raw_entropy(wallet_seed: str) -> bytes:
        """Returns the raw entropy bytes from the specified wallet secret"""
        decoded_seed = addresscodec.decode_seed(wallet_seed)
        return decoded_seed[0]

    @staticmethod
    def _get_ecdh_public_key(wallet_seed: str):
        raw_entropy = MessageEncryption._get_raw_entropy(wallet_seed)
        ecdh_public_key, _ = ED25519.derive_keypair(raw_entropy, is_validator=False)
        return ecdh_public_key

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
        """
        Get ECDH public key directly from a wallet seed
        
        Args:
            wallet_seed: The wallet seed to derive the key from
            
        Returns:
            str: The ECDH public key in hex format
            
        Raises:
            ValueError: If wallet_seed is invalid
        """
        try:
            raw_entropy = MessageEncryption._get_raw_entropy(wallet_seed)
            public_key, _ = ED25519.derive_keypair(raw_entropy, is_validator=False)
            return public_key
        except Exception as e:
            raise ValueError(f"Failed to derive ECDH public key: {e}") from e
    
    @staticmethod
    def get_shared_secret(received_key: str, wallet_seed: str) -> bytes:
        """
        Derive a shared secret using ECDH
        
        Args:
            received_key: public key received from another party
            wallet_seed: Seed for the wallet to derive the shared secret

        Returns:
            bytes: The derived shared secret

        Raises:
            ValueError: if received_key is invalid or wallet_seed is invalid
        """
        try:
            raw_entropy = MessageEncryption._get_raw_entropy(wallet_seed)
            return MessageEncryption.derive_shared_secret(public_key_hex=received_key, seed_bytes=raw_entropy)
        except Exception as e:
            raise ValueError(f"Failed to derive shared secret: {e}") from e
        
    @staticmethod
    def derive_shared_secret(public_key_hex: str, seed_bytes: bytes) -> bytes:
        """
        Derive a shared secret using ECDH
        Args:
            public_key_hex: their public key in hex
            seed_bytes: original entropy/seed bytes (required for ED25519)
        Returns:
            bytes: The shared secret
        """
        # First derive the ED25519 keypair using XRPL's method
        public_key_raw, private_key_raw = ED25519.derive_keypair(seed_bytes, is_validator=False)
        
        # Convert private key to bytes and remove ED prefix
        private_key_bytes = bytes.fromhex(private_key_raw)
        if len(private_key_bytes) == 33 and private_key_bytes[0] == 0xED:
            private_key_bytes = private_key_bytes[1:]  # Remove the ED prefix
        
        # Convert public key to bytes and remove ED prefix
        public_key_self_bytes = bytes.fromhex(public_key_raw)
        if len(public_key_self_bytes) == 33 and public_key_self_bytes[0] == 0xED:
            public_key_self_bytes = public_key_self_bytes[1:]  # Remove the ED prefix
        
        # Combine private and public key for NaCl format (64 bytes)
        private_key_combined = private_key_bytes + public_key_self_bytes
        
        # Convert their public key
        public_key_bytes = bytes.fromhex(public_key_hex)
        if len(public_key_bytes) == 33 and public_key_bytes[0] == 0xED:
            public_key_bytes = public_key_bytes[1:]  # Remove the ED prefix
        
        # Convert ED25519 keys to Curve25519
        private_curve = nacl.bindings.crypto_sign_ed25519_sk_to_curve25519(private_key_combined)
        public_curve = nacl.bindings.crypto_sign_ed25519_pk_to_curve25519(public_key_bytes)
        
        # Use raw X25519 function
        shared_secret = nacl.bindings.crypto_scalarmult(private_curve, public_curve)

        return shared_secret