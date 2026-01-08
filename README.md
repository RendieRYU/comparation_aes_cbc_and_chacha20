# Metopen: AES-CBC vs ChaCha20 on Images

This project compares AES-CBC and ChaCha20 stream cipher on digital images using qualitative metrics (MSE, PSNR, SSIM, correlation) and differential metrics (NPCR, UACI), plus timing.

## Quick Start

1. Create/activate a Python 3.10+ environment and install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the comparison on an image:

```bash
python main.py --image path/to/image.png --password secret --output-dir outputs --repeats 5
```

Artifacts are saved under `outputs/aes_cbc` and `outputs/chacha20` including cipher images, decrypted images, and metadata (IV/nonce, salts).

## GUI Mode

Launch the GUI to pick an image and run metrics without the CLI:

```bash
python -m gui.app
```

In the GUI you can:
- Choose an input image
- Select algorithm (AES-CBC or ChaCha20)
- Enter password and timing repeats
- Run encryption, view previews (original/cipher/decrypted) and see metrics
- Outputs are written under `outputs/` by default

## Notes
- AES-CBC uses PKCS7 padding; ciphertext length may exceed the raw pixel length. For visualization/metrics, the ciphertext is truncated to the original size and reshaped to the image (this does not affect correctness of decryption, which uses the full padded ciphertext written to `cipher.bin`).
- ChaCha20 preserves length (stream cipher), so cipher image maps 1:1 with the original.
- SSIM uses OpenCV Gaussian blur; install `opencv-python`.

## Metrics
- MSE, PSNR, SSIM, Pearson correlation (original vs cipher image).
- Adjacent-pixel correlation (H/V/D) on cipher images.
- NPCR, UACI using a one-pixel change in the plaintext.

