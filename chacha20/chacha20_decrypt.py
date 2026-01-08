from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.backends import default_backend


def decrypt_bytes(key: bytes, nonce: bytes, ciphertext: bytes) -> bytes:
    """
    ChaCha20 stream cipher decryption (same operation as encryption).
    """
    algorithm = algorithms.ChaCha20(key, nonce)
    cipher = Cipher(algorithm, mode=None, backend=default_backend())
    decryptor = cipher.decryptor()
    plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    return plaintext
