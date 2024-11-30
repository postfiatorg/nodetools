import secrets
from base64 import urlsafe_b64encode as b64e, urlsafe_b64decode as b64d

from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from hashlib import sha256, new as new_hash

backend = default_backend()
iterations = 100_000

def _derive_key(password: bytes, salt: bytes, iterations: int = iterations) -> bytes:
    """Derive a secret key from a given password and salt"""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(), length=32, salt=salt,
        iterations=iterations, backend=backend)
    return b64e(kdf.derive(password))

def password_encrypt(message: bytes, password: str, iterations: int = iterations) -> bytes:
    salt = secrets.token_bytes(16)
    key = _derive_key(password.encode(), salt, iterations)
    return b64e(
        b'%b%b%b' % (
            salt,
            iterations.to_bytes(4, 'big'),
            b64d(Fernet(key).encrypt(message)),
        )
    )

def password_decrypt(token: bytes, password: str) -> bytes:
    ''' use:
    decrypted_message = password_decrypt(encrypted_message, password)
    '''
    decoded = b64d(token)
    salt, iter, token = decoded[:16], decoded[16:20], b64e(decoded[20:])
    iterations = int.from_bytes(iter, 'big')
    key = _derive_key(password.encode(), salt, iterations)
    return Fernet(key).decrypt(token)

def get_account_id(public_key_hex: str) -> bytes:
    """Convert a public key to an account ID (a 20-byte identifier)"""
    # Convert hex to bytes
    public_key_bytes = bytes.fromhex(public_key_hex)

    # SHA256 of the public key
    sha256_hash = sha256(public_key_bytes).digest()

    # RIPEMD160 of the SHA256 hash
    ripemd160_hash = new_hash('ripemd160')
    ripemd160_hash.update(sha256_hash)
    account_id = ripemd160_hash.digest()

    return account_id