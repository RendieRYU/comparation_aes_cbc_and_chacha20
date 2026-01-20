from .chacha20_key_schedule import generate_keystream

def decrypt_bytes(key: bytes, nonce: bytes, ciphertext: bytes) -> bytes:
    """
    ChaCha20 dekripsi (identik dengan enkripsi): XOR dengan keystream.
    nonce boleh 12 byte (counter=0) atau 16 byte (counter||nonce).
    """
    ks = generate_keystream(key, nonce, len(ciphertext))
    return bytes(c ^ k for c, k in zip(ciphertext, ks))