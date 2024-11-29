from pathlib import Path
import sqlite3
import base64
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.fernet import Fernet
import time
import shutil
from xrpl.core import addresscodec
from xrpl.core.keypairs.ed25519 import ED25519
from nodetools.security.hash_tools import derive_shared_secret
from enum import Enum
import nodetools.utilities.constants as constants

CREDENTIALS_DB = "credentials.sqlite"
BACKUP_SUFFIX = ".sqlite_backup"
KEY_EXPIRY = -1  # No expiration by default

NETWORK_CONFIG = constants.get_network_config()

def get_credentials_directory():
    """Returns the path to the credentials directory, creating it if it doesn't exist"""
    creds_dir = Path.home().joinpath("postfiatcreds")
    creds_dir.mkdir(exist_ok=True)
    return creds_dir

def get_database_path():
    return get_credentials_directory() / CREDENTIALS_DB

class SecretType(Enum):
    REMEMBRANCER = "remembancer"
    NODE = "node"

    @classmethod
    def get_secret_key(cls, secret_type):
        """Maps secret type to credential key"""
        mapping = {
            cls.REMEMBRANCER: f'{NETWORK_CONFIG.remembrancer_name}__v1xrpsecret',
            cls.NODE: f'{NETWORK_CONFIG.node_name}__v1xrpsecret'
        }
        return mapping[secret_type]

class CredentialManager:
    _instance = None  # ensures we only have one instance
    _initialized = False  # ensures we only initialize once

    def __new__(cls, password=None):
        if cls._instance is None:
            if password is None:
                raise ValueError("Password is required for first CredentialManager instance")
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, password=None):
        if not self.__class__._initialized:
            if password is None:
                raise ValueError("Password is required for first CredentialManager instance")
            self.db_path = get_database_path()
            self.encryption_key = self._derive_encryption_key(password)
            self._key_expiry = time.time() + KEY_EXPIRY if KEY_EXPIRY >= 0 else float('inf')
            self._initialize_database()
            # print("---------------------------------Initialized CredentialManager---------------------------------\n")
            self.__class__._initialized = True

    def _check_key_expiry(self):
        """Check if encryption key has expired"""
        if KEY_EXPIRY >= 0 and time.time() > self._key_expiry:
            self.clear_credentials()
            raise CredentialsExpiredError("Encryption key has expired. Please re-authenticate.")
        
    def _initialize_database(self):
        """Initialize SQLite database with credentials table if it doesn't exist"""
        if not self.db_path.exists():
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS credentials (
                        key TEXT PRIMARY KEY,
                        encrypted_value TEXT NOT NULL
                    );
                """)
                conn.commit()
                print(f"Initialized database at {self.db_path}")

    def _encrypt_value(self, value):
        """Encrypt a value using the derived encryption key"""
        fernet = Fernet(self.encryption_key)
        return fernet.encrypt(value.encode()).decode()
    
    def _decrypt_value(self, encrypted_value):
        """Decrypt a value using the derived encryption key"""
        fernet = Fernet(self.encryption_key)
        return fernet.decrypt(encrypted_value.encode()).decode()

    def verify_password(self, password) -> bool:
        """Verify password by attempting to decrypt a known credential"""
        test_key = self._derive_encryption_key(password)
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT encrypted_value FROM credentials 
                    WHERE key like '%postgresconnstring'
                    LIMIT 1;
                """)
                row = cursor.fetchone()
                if row:
                    fernet = Fernet(test_key)
                    fernet.decrypt(row[0].encode())
                    return True
        except Exception as e:
            print(f"Failed to verify password: {e}")
            return False
        
    def get_credential(self, credential):
        """Get a specific credential by type"""
        self._check_key_expiry()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT encrypted_value FROM credentials 
                WHERE key = ?;
            """, (credential,))
            row = cursor.fetchone()
            if row:
                return self._decrypt_value(row[0])
        return None

    @staticmethod
    def _derive_encryption_key(password):
        """Derive an encryption key from a password"""
        kdf = PBKDF2HMAC(
            algorithm=SHA256(),
            length=32,
            salt=b'postfiat_salt',
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    def enter_and_encrypt_credential(self, credentials_dict: dict):
        """Encrypt and store multiple credentials in SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for key, value in credentials_dict.items():
                encrypted_value = self._encrypt_value(value)
                cursor.execute("""
                    INSERT OR REPLACE INTO credentials (key, encrypted_value)
                    VALUES (?, ?);
                """, (key, encrypted_value))
            conn.commit()
            print(f"Stored {len(credentials_dict)} credentials in {self.db_path}")

    def _decrypt_creds(self):
        """Retrieve and decrypt all credentials for the user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT key, encrypted_value FROM credentials;")
            rows = cursor.fetchall()
        return {key: self._decrypt_value(value) for key, value in rows}

    def _get_raw_entropy(self, secret_type: SecretType):
        """Returns the raw entropy bytes from the specified wallet secret"""
        secret_key = SecretType.get_secret_key(secret_type)
        wallet_secret = self.get_credential(secret_key)
        decoded_seed = addresscodec.decode_seed(wallet_secret)
        return decoded_seed[0]
    
    def _derive_ecdh_public_key(self, secret_type: SecretType):
        """Derives ECDH public key from wallet secret"""
        raw_entropy = self._get_raw_entropy(secret_type)
        public_key, _ = ED25519.derive_keypair(raw_entropy, is_validator=False)
        return public_key
    
    def get_ecdh_public_key(self, secret_type: SecretType):
        """Returns ECDH public key as hex string"""
        return self._derive_ecdh_public_key(secret_type)

    def get_shared_secret(self, received_key: str, secret_type: SecretType) -> bytes: 
        """
        Derive a shared secret using ECDH
        
        Args:
            received_key: public key received from another party
            secret_type: SecretType enum indicating which secret to use

        Returns:
            bytes: The derived shared secret

        Raises:
            ValueError: if received_key is invalid or secret not found
        """
        try:
            raw_entropy = self._get_raw_entropy(secret_type)
            return derive_shared_secret(public_key_hex=received_key, seed_bytes=raw_entropy)
        except Exception as e:
            raise ValueError(f"Failed to derive shared secret: {e}") from e
    
    def get_all_shared_secrets(self, received_key: str) -> dict[SecretType, bytes]:
        """
        Get both remembrancer and node shared secrets if available

        Returns:
            dict: Mapping of SecretType to derived shared secrets
        """
        secrets = {}
        for secret_type in SecretType:
            try:
                secrets[secret_type] = self.get_shared_secret(received_key, secret_type)
            except (ValueError, KeyError):
                continue
        return secrets

class CredentialsExpiredError(Exception):
    """Exception raised when the encryption key has expired"""
    pass
