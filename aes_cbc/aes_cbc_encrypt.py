import os
import hashlib
from typing import Tuple
import numpy as np

# ========== Key derivation (PBKDF2 stdlib) ==========
def derive_key(password: str, salt: bytes | None = None, iterations: int = 200_000) -> Tuple[bytes, bytes]:
    if salt is None:
        salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations, dklen=16)
    return key, salt

# ========== AES core (pure Python, 128/192/256) ==========
# S-box
_sbox = [
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
]

# Rcon
_rcon = [0x00000000,
         0x01000000,
         0x02000000,
         0x04000000,
         0x08000000,
         0x10000000,
         0x20000000,
         0x40000000,
         0x80000000,
         0x1B000000,
         0x36000000]

def _sub_byte(b: int) -> int:
    return _sbox[b]

def _rot_word(w: int) -> int:
    return ((w << 8) & 0xFFFFFFFF) | (w >> 24)

def _sub_word(w: int) -> int:
    return ((_sub_byte((w >> 24) & 0xFF) << 24) |
            (_sub_byte((w >> 16) & 0xFF) << 16) |
            (_sub_byte((w >>  8) & 0xFF) <<  8) |
             _sub_byte(w & 0xFF)) & 0xFFFFFFFF

def _bytes_to_words(key: bytes) -> list[int]:
    return [int.from_bytes(key[i:i+4], "big") for i in range(0, len(key), 4)]

def _words_to_bytes(words: list[int]) -> bytes:
    return b"".join(w.to_bytes(4, "big") for w in words)

def _expand_key(key: bytes) -> list[bytes]:
    Nk = len(key) // 4
    if Nk not in (4, 6, 8):
        raise ValueError("AES key must be 16/24/32 bytes")
    Nb = 4
    Nr = {4:10, 6:12, 8:14}[Nk]

    w = _bytes_to_words(key)
    for i in range(Nk, Nb * (Nr + 1)):
        temp = w[i - 1]
        if i % Nk == 0:
            temp = _sub_word(_rot_word(temp)) ^ _rcon[i // Nk]
        elif Nk > 6 and i % Nk == 4:
            temp = _sub_word(temp)
        w.append((w[i - Nk] ^ temp) & 0xFFFFFFFF)

    # Return round keys as 16-byte blocks (column-major words)
    round_keys: list[bytes] = []
    for r in range(Nr + 1):
        words = w[r*Nb:(r+1)*Nb]
        round_keys.append(_words_to_bytes(words))  # 16 bytes
    return round_keys

def _add_round_key(state: list[list[int]], rk: bytes) -> None:
    # rk is 16 bytes, column-major (4 words)
    for c in range(4):
        col = rk[4*c:4*(c+1)]
        for r in range(4):
            state[r][c] ^= col[r]

def _sub_bytes(state: list[list[int]]) -> None:
    for r in range(4):
        for c in range(4):
            state[r][c] = _sub_byte(state[r][c])

def _shift_rows(state: list[list[int]]) -> None:
    state[1] = state[1][1:] + state[1][:1]
    state[2] = state[2][2:] + state[2][:2]
    state[3] = state[3][3:] + state[3][:3]

def _xtime(a: int) -> int:
    return ((a << 1) & 0xFF) ^ (0x1B if (a & 0x80) else 0x00)

def _mul2(a: int) -> int: return _xtime(a)
def _mul3(a: int) -> int: return _xtime(a) ^ a

def _mix_single_column(col: list[int]) -> list[int]:
    a0,a1,a2,a3 = col
    return [
        _mul2(a0) ^ _mul3(a1) ^ a2 ^ a3,
        a0 ^ _mul2(a1) ^ _mul3(a2) ^ a3,
        a0 ^ a1 ^ _mul2(a2) ^ _mul3(a3),
        _mul3(a0) ^ a1 ^ a2 ^ _mul2(a3),
    ]

def _mix_columns(state: list[list[int]]) -> None:
    for c in range(4):
        col = [state[r][c] for r in range(4)]
        colm = _mix_single_column(col)
        for r in range(4):
            state[r][c] = colm[r]

def _block_encrypt(block16: bytes, round_keys: list[bytes]) -> bytes:
    # State is 4x4 bytes column-major: state[r][c] = block[c*4 + r]
    state = [[0]*4 for _ in range(4)]
    for c in range(4):
        for r in range(4):
            state[r][c] = block16[c*4 + r]

    _add_round_key(state, round_keys[0])

    Nr = len(round_keys) - 1
    for round_idx in range(1, Nr):
        _sub_bytes(state)
        _shift_rows(state)
        _mix_columns(state)
        _add_round_key(state, round_keys[round_idx])

    _sub_bytes(state)
    _shift_rows(state)
    _add_round_key(state, round_keys[Nr])

    out = bytearray(16)
    for c in range(4):
        for r in range(4):
            out[c*4 + r] = state[r][c]
    return bytes(out)

# ========== CBC mode + PKCS#7 padding ==========
def _pkcs7_pad(data: bytes, block_size: int = 16) -> bytes:
    pad_len = block_size - (len(data) % block_size)
    return data + bytes([pad_len]) * pad_len

def _cbc_encrypt(key: bytes, iv: bytes, plaintext: bytes) -> bytes:
    rks = _expand_key(key)
    if len(iv) != 16:
        raise ValueError("IV must be 16 bytes")
    padded = _pkcs7_pad(plaintext, 16)
    out = bytearray(len(padded))
    prev = iv
    for i in range(0, len(padded), 16):
        blk = bytes(a ^ b for a, b in zip(padded[i:i+16], prev))
        enc = _block_encrypt(blk, rks)
        out[i:i+16] = enc
        prev = enc
    return bytes(out)

# ========== Public API (kompatibel) ==========
def encrypt_bytes(key: bytes, plaintext: bytes) -> Tuple[bytes, bytes]:
    """
    AES-CBC encrypt with PKCS7 padding. Returns (iv, ciphertext).
    """
    iv = os.urandom(16)
    ct = _cbc_encrypt(key, iv, plaintext)
    return iv, ct

def encrypt_bytes_with_iv(key: bytes, iv: bytes, plaintext: bytes) -> bytes:
    """
    AES-CBC encrypt with a provided IV and PKCS7 padding. Returns ciphertext.
    Useful for differential analysis where the same IV is reused intentionally.
    """
    return _cbc_encrypt(key, iv, plaintext)

def encrypt_image_to_image(img: np.ndarray, key: bytes) -> Tuple[np.ndarray, bytes, bytes]:
    """
    Encrypt an image array (uint8) with AES-CBC.
    Returns (cipher_image_uint8_same_shape, iv, full_ciphertext_with_padding).
    """
    if img.dtype != np.uint8:
        raise ValueError("Image must be uint8.")
    flat = img.reshape(-1)
    iv, ct = encrypt_bytes(key, flat.tobytes())
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