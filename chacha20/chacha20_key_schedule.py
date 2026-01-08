import os
from typing import Tuple

from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend


def derive_key(password: str, salt: bytes | None = None, iterations: int = 200_000) -> Tuple[bytes, bytes]:
    """
    Derive a 256-bit key for ChaCha20 using PBKDF2-HMAC-SHA256.
    Returns (key, salt).
    """
    if salt is None:
        salt = os.urandom(16)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=iterations,
        backend=default_backend(),
    )
    key = kdf.derive(password.encode("utf-8"))
    return key, salt


def generate_nonce() -> bytes:
    """Generate a 16-byte nonce for ChaCha20 (IETF variant uses 12, cryptography's ChaCha20 uses 16)."""
    return os.urandom(16)
