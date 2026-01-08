import argparse
import os
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from kualitatif.mse import mse
from kualitatif.psnr import psnr
from kualitatif.ssim import ssim
from kualitatif.cc import correlation_coefficient, adjacent_correlation

from diferensial.npcr import npcr
from diferensial.uaci import uaci

from efisiensi.algorithm_speed import measure_time

from aes_cbc.aes_cbc_encrypt import (
    derive_key as aes_derive_key,
    encrypt_image_to_image as aes_encrypt_image,
    encrypt_image_to_image_with_iv as aes_encrypt_image_with_iv,
)
from aes_cbc.aes_cbc_decrypt import decrypt_bytes as aes_decrypt_bytes

from chacha20.chacha20_key_schedule import derive_key as chacha_derive_key, generate_nonce
from chacha20.chacha20_encrypt import encrypt_image_to_image as chacha_encrypt_image, encrypt_bytes as chacha_encrypt_bytes
from chacha20.chacha20_decrypt import decrypt_bytes as chacha_decrypt_bytes


def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def save_image(arr: np.ndarray, path: str) -> None:
    Image.fromarray(arr).save(path)


def flip_one_pixel(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    h, w, c = out.shape if out.ndim == 3 else (*out.shape, 1)
    y = random.randrange(h)
    x = random.randrange(w)
    ch = random.randrange(c)
    if out.ndim == 2:
        out[y, x] = (int(out[y, x]) + 1) % 256
    else:
        out[y, x, ch] = (int(out[y, x, ch]) + 1) % 256
    return out


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def run(args: argparse.Namespace) -> None:
    random.seed(42)

    img = load_image(args.image)
    h, w = img.shape[:2]
    print(f"Loaded image: {args.image} shape={img.shape}")

    out_dir = args.output_dir
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "aes_cbc"))
    ensure_dir(os.path.join(out_dir, "chacha20"))

    # AES-CBC
    aes_key, aes_salt = aes_derive_key(args.password)
    aes_cipher_img, aes_iv, aes_full_ct = aes_encrypt_image(img, aes_key)

    # Decrypt AES for correctness
    aes_plain = aes_decrypt_bytes(aes_key, aes_iv, aes_full_ct)
    aes_plain_img = np.frombuffer(aes_plain[: img.size], dtype=np.uint8).reshape(img.shape)

    # Save AES outputs
    save_image(aes_cipher_img, os.path.join(out_dir, "aes_cbc", "cipher.png"))
    save_image(aes_plain_img, os.path.join(out_dir, "aes_cbc", "decrypted.png"))
    with open(os.path.join(out_dir, "aes_cbc", "cipher.bin"), "wb") as f:
        f.write(aes_full_ct)
    with open(os.path.join(out_dir, "aes_cbc", "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"salt_hex": aes_salt.hex(), "iv_hex": aes_iv.hex()}, f, indent=2)

    # ChaCha20
    chacha_key, chacha_salt = chacha_derive_key(args.password)
    chacha_nonce = generate_nonce()
    chacha_cipher_img = chacha_encrypt_image(img, chacha_key, chacha_nonce)

    # Decrypt ChaCha for correctness
    chacha_full_ct = chacha_encrypt_bytes(chacha_key, chacha_nonce, img.reshape(-1).tobytes())
    chacha_plain = chacha_decrypt_bytes(chacha_key, chacha_nonce, chacha_full_ct)
    chacha_plain_img = np.frombuffer(chacha_plain, dtype=np.uint8).reshape(img.shape)

    # Save ChaCha20 outputs
    save_image(chacha_cipher_img, os.path.join(out_dir, "chacha20", "cipher.png"))
    save_image(chacha_plain_img, os.path.join(out_dir, "chacha20", "decrypted.png"))
    with open(os.path.join(out_dir, "chacha20", "cipher.bin"), "wb") as f:
        f.write(chacha_full_ct)
    with open(os.path.join(out_dir, "chacha20", "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"salt_hex": chacha_salt.hex(), "nonce_hex": chacha_nonce.hex()}, f, indent=2)

    # Qualitative metrics (original vs encrypted-image)
    print("\n== Qualitative Metrics (Original vs Cipher Image) ==")
    metrics = {}
    for name, cipher_img in [
        ("AES-CBC", aes_cipher_img),
        ("ChaCha20", chacha_cipher_img),
    ]:
        M = mse(img, cipher_img)
        P = psnr(img, cipher_img)
        S = ssim(img, cipher_img)
        C = correlation_coefficient(img, cipher_img)
        metrics[name] = {"MSE": M, "PSNR": P, "SSIM": S, "CC": C}
        print(f"{name}: MSE={M:.4f}, PSNR={P:.4f} dB, SSIM={S:.6f}, CC={C:.6f}")

    # Adjacent correlation on cipher images (security indicator)
    for name, cipher_img in [("AES-CBC", aes_cipher_img), ("ChaCha20", chacha_cipher_img)]:
        ch = adjacent_correlation(cipher_img, "horizontal")
        cv = adjacent_correlation(cipher_img, "vertical")
        cd = adjacent_correlation(cipher_img, "diagonal")
        print(f"{name} adjacent correlation - H:{ch:.6f} V:{cv:.6f} D:{cd:.6f}")

    # Differential metrics (NPCR, UACI)
    print("\n== Differential Metrics (NPCR, UACI) ==")
    img2 = flip_one_pixel(img)

    # AES differential: encrypt flipped image with the SAME IV
    aes_cipher_img_2, _ = aes_encrypt_image_with_iv(img2, aes_key, aes_iv)
    aes_npcr = npcr(aes_cipher_img, aes_cipher_img_2)
    aes_uaci = uaci(aes_cipher_img, aes_cipher_img_2)
    print(f"AES-CBC: NPCR={aes_npcr:.4f}%, UACI={aes_uaci:.4f}%")

    # ChaCha20 differential: encrypt flipped image with the SAME nonce
    chacha_cipher_img_2 = chacha_encrypt_image(img2, chacha_key, chacha_nonce)
    ch_npcr = npcr(chacha_cipher_img, chacha_cipher_img_2)
    ch_uaci = uaci(chacha_cipher_img, chacha_cipher_img_2)
    print(f"ChaCha20: NPCR={ch_npcr:.4f}%, UACI={ch_uaci:.4f}%")

    # Timing
    print("\n== Timing (encryption only) ==")
    repeats = max(1, args.repeats)

    def aes_enc_only():
        _ = aes_encrypt_image(img, aes_key)

    def chacha_enc_only():
        _ = chacha_encrypt_image(img, chacha_key, chacha_nonce)

    t_aes, _ = measure_time(aes_enc_only, repeats=repeats)
    t_ch, _ = measure_time(chacha_enc_only, repeats=repeats)
    print(f"AES-CBC avg time over {repeats} run(s): {t_aes:.6f} s")
    print(f"ChaCha20 avg time over {repeats} run(s): {t_ch:.6f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare AES-CBC and ChaCha20 on images.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--password", default="metopen-password", help="Password for KDF.")
    parser.add_argument("--output-dir", default="outputs", help="Directory to store results.")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats for timing average.")
    args = parser.parse_args()
    run(args)
