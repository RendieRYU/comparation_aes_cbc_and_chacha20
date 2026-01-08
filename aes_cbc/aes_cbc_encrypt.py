import os
from typing import Tuple

import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding, hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend


def derive_key(password: str, salt: bytes | None = None, iterations: int = 200_000) -> Tuple[bytes, bytes]:
    """
    Derive a 256-bit AES key using PBKDF2-HMAC-SHA256.
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


def encrypt_bytes(key: bytes, plaintext: bytes) -> Tuple[bytes, bytes]:
    """
    AES-CBC encrypt with PKCS7 padding. Returns (iv, ciphertext).
    """
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    padder = padding.PKCS7(128).padder()
    padded = padder.update(plaintext) + padder.finalize()
    ciphertext = encryptor.update(padded) + encryptor.finalize()
    return iv, ciphertext

def encrypt_bytes_with_iv(key: bytes, iv: bytes, plaintext: bytes) -> bytes:
    """
    AES-CBC encrypt with a provided IV and PKCS7 padding. Returns ciphertext.
    Useful for differential analysis where the same IV is reused intentionally.
    """
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    padder = padding.PKCS7(128).padder()
    padded = padder.update(plaintext) + padder.finalize()
    ciphertext = encryptor.update(padded) + encryptor.finalize()
    return ciphertext


def encrypt_image_to_image(img: np.ndarray, key: bytes) -> Tuple[np.ndarray, bytes, bytes]:
    """
    Encrypt an image array (uint8) with AES-CBC.
    Returns (cipher_image_uint8_same_shape, iv, full_ciphertext_with_padding).
    Note: The cipher_image is for analysis/visualization and is obtained by
    truncating ciphertext to the original size; decryption requires the full ciphertext.
    """
    if img.dtype != np.uint8:
        raise ValueError("Image must be uint8.")
    flat = img.reshape(-1)
    iv, ct = encrypt_bytes(key, flat.tobytes())
    # Truncate to original size for visualization
    ct_trim = ct[: flat.size]
    cipher_image = np.frombuffer(ct_trim, dtype=np.uint8).reshape(img.shape)
    return cipher_image, iv, ct

def encrypt_image_to_image_with_iv(img: np.ndarray, key: bytes, iv: bytes) -> Tuple[np.ndarray, bytes]:
    """
    Encrypt an image array (uint8) with AES-CBC using a provided IV.
    Returns (cipher_image_uint8_same_shape, full_ciphertext_with_padding).
    """
    if img.dtype != np.uint8:
        raise ValueError("Image must be uint8.")
    flat = img.reshape(-1)
    ct = encrypt_bytes_with_iv(key, iv, flat.tobytes())
    ct_trim = ct[: flat.size]
    cipher_image = np.frombuffer(ct_trim, dtype=np.uint8).reshape(img.shape)
    return cipher_image, ct
