import numpy as np
from .chacha20_key_schedule import generate_keystream

def encrypt_bytes(key: bytes, nonce: bytes, plaintext: bytes) -> bytes:
    """
    ChaCha20 stream cipher (XOR dengan keystream).
    nonce boleh 12 byte (counter=0) atau 16 byte (counter||nonce).
    """
    ks = generate_keystream(key, nonce, len(plaintext))
    return bytes(p ^ k for p, k in zip(plaintext, ks))

def encrypt_image_to_image(img: np.ndarray, key: bytes, nonce: bytes) -> np.ndarray:
    """
    Enkripsi gambar uint8 dan mengembalikan array uint8 dengan shape yang sama.
    """
    if img.dtype != np.uint8:
        raise ValueError("Image must be uint8.")
    flat = img.reshape(-1)
    ks = np.frombuffer(generate_keystream(key, nonce, flat.size), dtype=np.uint8)
    cipher_flat = np.bitwise_xor(flat, ks)
    return cipher_flat.reshape(img.shape)