from typing import Tuple

import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.backends import default_backend


def encrypt_bytes(key: bytes, nonce: bytes, plaintext: bytes) -> bytes:
    """
    ChaCha20 stream cipher encryption (same for decryption).
    Returns ciphertext of same length as plaintext.
    """
    algorithm = algorithms.ChaCha20(key, nonce)
    cipher = Cipher(algorithm, mode=None, backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    return ciphertext


def encrypt_image_to_image(img: np.ndarray, key: bytes, nonce: bytes) -> np.ndarray:
    """
    Encrypt image bytes with ChaCha20 and return same-shaped uint8 array.
    """
    if img.dtype != np.uint8:
        raise ValueError("Image must be uint8.")
    flat = img.reshape(-1)
    ct = encrypt_bytes(key, nonce, flat.tobytes())
    cipher_image = np.frombuffer(ct, dtype=np.uint8).reshape(img.shape)
    return cipher_image
