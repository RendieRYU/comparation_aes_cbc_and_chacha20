import os, hashlib
from typing import Tuple

# ========== Key derivation (stdlib) ==========
def derive_key(password: str, salt: bytes | None = None, iterations: int = 200_000) -> Tuple[bytes, bytes]:
    if salt is None:
        salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations, dklen=32)
    return key, salt

# ========== Nonce helpers ==========
def generate_nonce(counter: int = 0) -> bytes:
    # 16-byte: 4-byte LE counter || 12-byte random (IETF-style)
    return (counter & 0xFFFFFFFF).to_bytes(4, "little") + os.urandom(12)

def make_nonce16(nonce12: bytes, counter: int = 0) -> bytes:
    if len(nonce12) != 12:
        raise ValueError("nonce12 must be 12 bytes")
    return (counter & 0xFFFFFFFF).to_bytes(4, "little") + nonce12

# ========== ChaCha20 core & keystream (pure Python) ==========
def _rotl32(x: int, n: int) -> int:
    return ((x << n) & 0xFFFFFFFF) | (x >> (32 - n))

def _qr(s: list[int], a: int, b: int, c: int, d: int) -> None:
    s[a] = (s[a] + s[b]) & 0xFFFFFFFF; s[d] ^= s[a]; s[d] = _rotl32(s[d], 16)
    s[c] = (s[c] + s[d]) & 0xFFFFFFFF; s[b] ^= s[c]; s[b] = _rotl32(s[b], 12)
    s[a] = (s[a] + s[b]) & 0xFFFFFFFF; s[d] ^= s[a]; s[d] = _rotl32(s[d], 8)
    s[c] = (s[c] + s[d]) & 0xFFFFFFFF; s[b] ^= s[c]; s[b] = _rotl32(s[b], 7)

def _le_words(b: bytes) -> list[int]:
    return [int.from_bytes(b[i:i+4], "little") for i in range(0, len(b), 4)]

def _words_le(ws: list[int]) -> bytes:
    return b"".join((w & 0xFFFFFFFF).to_bytes(4, "little") for w in ws)

def _parse_nonce(nonce: bytes) -> tuple[int, bytes]:
    if len(nonce) == 12:
        return 0, nonce
    if len(nonce) == 16:
        return int.from_bytes(nonce[:4], "little"), nonce[4:]
    raise ValueError("Nonce must be 12 or 16 bytes")

def _block(key: bytes, counter: int, nonce12: bytes) -> bytes:
    if len(key) != 32: raise ValueError("Key must be 32 bytes")
    if len(nonce12) != 12: raise ValueError("Nonce must be 12 bytes")
    c = [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574]  # "expand 32-byte k"
    k = _le_words(key)  # 8 words
    n = _le_words(nonce12)  # 3 words
    s = [c[0], c[1], c[2], c[3], k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7],
         counter & 0xFFFFFFFF, n[0], n[1], n[2]]
    w = s.copy()
    for _ in range(10):  # 20 rounds
        _qr(w, 0, 4, 8, 12); _qr(w, 1, 5, 9, 13); _qr(w, 2, 6, 10, 14); _qr(w, 3, 7, 11, 15)
        _qr(w, 0, 5, 10, 15); _qr(w, 1, 6, 11, 12); _qr(w, 2, 7, 8, 13); _qr(w, 3, 4, 9, 14)
    return _words_le([(w[i] + s[i]) & 0xFFFFFFFF for i in range(16)])  # 64 bytes

def generate_keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    """
    Menghasilkan keystream ChaCha20 sepanjang 'length' byte.
    - key: 32 byte
    - nonce: 12 byte (counter=0) atau 16 byte (4B counter || 12B nonce)
    """
    counter, n12 = _parse_nonce(nonce)
    out = bytearray()
    blk = 0
    while len(out) < length:
        out.extend(_block(key, (counter + blk) & 0xFFFFFFFF, n12))
        blk += 1
    return bytes(out[:length])