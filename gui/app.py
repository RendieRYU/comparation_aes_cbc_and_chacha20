import os
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

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

from chacha20.chacha20_key_schedule import (
    derive_key as chacha_derive_key,
    generate_nonce,
)
from chacha20.chacha20_encrypt import (
    encrypt_image_to_image as chacha_encrypt_image,
    encrypt_bytes as chacha_encrypt_bytes,
)
from chacha20.chacha20_decrypt import decrypt_bytes as chacha_decrypt_bytes


random.seed(42)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def save_image(arr: np.ndarray, path: str) -> None:
    Image.fromarray(arr).save(path)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AES-CBC vs ChaCha20 - Metopen GUI")
        self.geometry("1100x720")

        # State
        self.image_path: Optional[str] = None
        self.password_var = tk.StringVar(value="metopen-password")
        self.algorithm_var = tk.StringVar(value="AES-CBC")
        self.repeats_var = tk.IntVar(value=3)
        self.output_dir_var = tk.StringVar(value="D:\\Kuliah\\Akademik\\Matkul\\Semester 7\\Metopen\\Program\\Output")

        # UI
        self._build_controls()
        self._build_preview_and_output()

    def _build_controls(self):
        frame = ttk.Frame(self, padding=10)
        frame.pack(side=tk.TOP, fill=tk.X)

        # Image chooser
        ttk.Label(frame, text="Gambar:").grid(row=0, column=0, sticky=tk.W, padx=4)
        self.image_entry = ttk.Entry(frame, width=60)
        self.image_entry.grid(row=0, column=1, sticky=tk.W)
        ttk.Button(frame, text="Pilih...", command=self._choose_image).grid(row=0, column=2, padx=6)

        # Password
        ttk.Label(frame, text="Password:").grid(row=1, column=0, sticky=tk.W, padx=4)
        self.password_entry = ttk.Entry(frame, textvariable=self.password_var, show="*")
        self.password_entry.grid(row=1, column=1, sticky=tk.W)

        # Algorithm
        ttk.Label(frame, text="Algoritma:").grid(row=2, column=0, sticky=tk.W, padx=4)
        alg_combo = ttk.Combobox(frame, textvariable=self.algorithm_var, values=["AES-CBC", "ChaCha20"], state="readonly", width=20)
        alg_combo.grid(row=2, column=1, sticky=tk.W)

        # Repeats
        ttk.Label(frame, text="Repeats (timing):").grid(row=3, column=0, sticky=tk.W, padx=4)
        repeats_spin = ttk.Spinbox(frame, from_=1, to=50, textvariable=self.repeats_var, width=5)
        repeats_spin.grid(row=3, column=1, sticky=tk.W)

        # Output dir
        ttk.Label(frame, text="Output Dir:").grid(row=4, column=0, sticky=tk.W, padx=4)
        out_entry = ttk.Entry(frame, textvariable=self.output_dir_var, width=60)
        out_entry.grid(row=4, column=1, sticky=tk.W)
        ttk.Button(frame, text="Pilih Folder...", command=self._choose_output_dir).grid(row=4, column=2, padx=6)

        # Run button
        ttk.Button(frame, text="Enkripsi & Analisis", command=self._run).grid(row=5, column=1, sticky=tk.W, pady=8)

    def _build_preview_and_output(self):
        # Left: previews as 2x2 grid
        previews = ttk.Frame(self, padding=10)
        previews.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create tiles for Original, Encrypted, Encrypted2, Decrypted
        self.tile_orig = ttk.Frame(previews)
        self.tile_enc1 = ttk.Frame(previews)
        self.tile_enc2 = ttk.Frame(previews)
        self.tile_dec = ttk.Frame(previews)

        self.tile_orig.grid(row=0, column=0, padx=6, pady=6, sticky="n")
        self.tile_enc1.grid(row=0, column=1, padx=6, pady=6, sticky="n")
        self.tile_enc2.grid(row=1, column=0, padx=6, pady=6, sticky="n")
        self.tile_dec.grid(row=1, column=1, padx=6, pady=6, sticky="n")

        # Inside each tile: title + image label
        ttk.Label(self.tile_orig, text="Original").pack()
        self.orig_canvas = tk.Label(self.tile_orig)
        self.orig_canvas.pack(pady=4)

        ttk.Label(self.tile_enc1, text="Encrypted").pack()
        self.cipher_canvas = tk.Label(self.tile_enc1)
        self.cipher_canvas.pack(pady=4)

        ttk.Label(self.tile_enc2, text="Encrypted2").pack()
        self.cipher2_canvas = tk.Label(self.tile_enc2)
        self.cipher2_canvas.pack(pady=4)

        ttk.Label(self.tile_dec, text="Decrypted").pack()
        self.dec_canvas = tk.Label(self.tile_dec)
        self.dec_canvas.pack(pady=4)

        # Right: output tables
        right = ttk.Frame(self, padding=10)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.metrics_title = ttk.Label(right, text="Hasil / Metrics")
        self.metrics_title.pack()

        # Table: Qualitative metrics
        self.table_qual = ttk.Treeview(
            right,
            columns=("Kategori", "MSE", "PSNR (dB)", "SSIM", "CC"),
            show="headings",
            height=6,
        )
        for col, width in (
            ("Kategori", 140), ("MSE", 90), ("PSNR (dB)", 100), ("SSIM", 90), ("CC", 90)
        ):
            self.table_qual.heading(col, text=col)
            self.table_qual.column(col, width=width, anchor="center")
        self.table_qual.pack(fill=tk.X, expand=False, pady=6)

        # Table: Differential + adjacency + time
        self.table_diff = ttk.Treeview(
            right,
            columns=("Kategori", "NPCR (%)", "UACI (%)", "Adj H", "Adj V", "Adj D", "Avg Time (s)"),
            show="headings",
            height=5,
        )
        for col, width in (
            ("Kategori", 140), ("NPCR (%)", 90), ("UACI (%)", 90), ("Adj H", 80), ("Adj V", 80), ("Adj D", 80), ("Avg Time (s)", 110)
        ):
            self.table_diff.heading(col, text=col)
            self.table_diff.column(col, width=width, anchor="center")
        self.table_diff.pack(fill=tk.X, expand=False, pady=6)

        # Note label for any calculation notices (e.g., SSIM fallback)
        self.note_label = ttk.Label(right, text="")
        self.note_label.pack(anchor="w", pady=4)

    def _choose_image(self):
        path = filedialog.askopenfilename(title="Pilih gambar", filetypes=[
            ("Images", ".png .jpg .jpeg .bmp .tif .tiff"), ("All files", ".*")
        ])
        if path:
            self.image_path = path
            self.image_entry.delete(0, tk.END)
            self.image_entry.insert(0, path)
            try:
                img = Image.open(path).convert("RGB")
                self._set_preview(self.orig_canvas, img)
            except Exception as e:
                messagebox.showerror("Error", f"Gagal membuka gambar: {e}")

    def _choose_output_dir(self):
        path = filedialog.askdirectory(title="Pilih folder output")
        if path:
            self.output_dir_var.set(path)

    def _set_preview(self, widget: tk.Label, pil_img: Image.Image):
        # Resize to fit a fixed bounding box so three images fit vertically
        MAX_W, MAX_H = 300, 300
        w, h = pil_img.size
        scale = min(1.0, MAX_W / float(w), MAX_H / float(h))
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w <= 0: new_w = 1
        if new_h <= 0: new_h = 1
        if (new_w, new_h) != (w, h):
            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil_img)
        widget.image = tk_img
        widget.configure(image=tk_img)

    def _run(self):
        path = self.image_entry.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showwarning("Input", "Silakan pilih gambar yang valid.")
            return
        password = self.password_var.get()
        algorithm = self.algorithm_var.get()
        repeats = max(1, int(self.repeats_var.get()))
        out_dir = self.output_dir_var.get() or "D:\\Kuliah\\Akademik\\Matkul\\Semester 7\\Metopen\\Program\\Output"
        ensure_dir(out_dir)
        ensure_dir(os.path.join(out_dir, "aes_cbc"))
        ensure_dir(os.path.join(out_dir, "chacha20"))

        try:
            img = load_image(path)
        except Exception as e:
            messagebox.showerror("Error", f"Gagal memuat gambar: {e}")
            return

        # Show original preview
        self._set_preview(self.orig_canvas, Image.fromarray(img))

        # Clear tables and note
        self.table_qual.delete(*self.table_qual.get_children())
        self.table_diff.delete(*self.table_diff.get_children())
        self.note_label.configure(text="")

        try:
            if algorithm == "AES-CBC":
                aes_key, aes_salt = aes_derive_key(password)
                aes_cipher_img, aes_iv, aes_full_ct = aes_encrypt_image(img, aes_key)
                aes_plain = aes_decrypt_bytes(aes_key, aes_iv, aes_full_ct)
                aes_plain_img = np.frombuffer(aes_plain[: img.size], dtype=np.uint8).reshape(img.shape)

                # Save
                save_image(aes_cipher_img, os.path.join(out_dir, "aes_cbc", "cipher.png"))
                save_image(aes_plain_img, os.path.join(out_dir, "aes_cbc", "decrypted.png"))
                with open(os.path.join(out_dir, "aes_cbc", "cipher.bin"), "wb") as f:
                    f.write(aes_full_ct)
                with open(os.path.join(out_dir, "aes_cbc", "meta.json"), "w", encoding="utf-8") as f:
                    json.dump({"salt_hex": aes_salt.hex(), "iv_hex": aes_iv.hex()}, f, indent=2)

                # Previews
                self._set_preview(self.cipher_canvas, Image.fromarray(aes_cipher_img))
                self._set_preview(self.dec_canvas, Image.fromarray(aes_plain_img))

                # Metrics (use same IV for differential)
                cipher_img2 = self._write_metrics(
                    name="AES-CBC",
                    img=img,
                    cipher_img=aes_cipher_img,
                    dec_img=aes_plain_img,
                    repeats=repeats,
                    enc_info={
                        "algo": "aes",
                        "key": aes_key,
                        "iv": aes_iv,
                    },
                )
                # Preview Encrypted2
                if cipher_img2 is not None:
                    self._set_preview(self.cipher2_canvas, Image.fromarray(cipher_img2))

            else:  # ChaCha20
                chacha_key, chacha_salt = chacha_derive_key(password)
                chacha_nonce = generate_nonce()
                chacha_cipher_img = chacha_encrypt_image(img, chacha_key, chacha_nonce)
                chacha_full_ct = chacha_encrypt_bytes(chacha_key, chacha_nonce, img.reshape(-1).tobytes())
                chacha_plain = chacha_decrypt_bytes(chacha_key, chacha_nonce, chacha_full_ct)
                chacha_plain_img = np.frombuffer(chacha_plain, dtype=np.uint8).reshape(img.shape)

                # Save
                save_image(chacha_cipher_img, os.path.join(out_dir, "chacha20", "cipher.png"))
                save_image(chacha_plain_img, os.path.join(out_dir, "chacha20", "decrypted.png"))
                with open(os.path.join(out_dir, "chacha20", "cipher.bin"), "wb") as f:
                    f.write(chacha_full_ct)
                with open(os.path.join(out_dir, "chacha20", "meta.json"), "w", encoding="utf-8") as f:
                    json.dump({"salt_hex": chacha_salt.hex(), "nonce_hex": chacha_nonce.hex()}, f, indent=2)

                # Previews
                self._set_preview(self.cipher_canvas, Image.fromarray(chacha_cipher_img))
                self._set_preview(self.dec_canvas, Image.fromarray(chacha_plain_img))

                # Metrics (reuse the same nonce for differential)
                cipher_img2 = self._write_metrics(
                    name="ChaCha20",
                    img=img,
                    cipher_img=chacha_cipher_img,
                    dec_img=chacha_plain_img,
                    repeats=repeats,
                    enc_info={
                        "algo": "chacha",
                        "key": chacha_key,
                        "nonce": chacha_nonce,
                    },
                )
                # Preview Encrypted2
                if cipher_img2 is not None:
                    self._set_preview(self.cipher2_canvas, Image.fromarray(cipher_img2))

        except Exception as e:
            messagebox.showerror("Error", f"Terjadi kesalahan saat proses: {e}")

    def _write_metrics(self, name: str, img: np.ndarray, cipher_img: np.ndarray, dec_img: np.ndarray, repeats: int, enc_info: dict):
        # Helper to compute qualitative metrics safely (SSIM may fail if OpenCV missing)
        def _qual(a: np.ndarray, b: np.ndarray):
            try:
                return (
                    mse(a, b),
                    psnr(a, b),
                    ssim(a, b),
                    correlation_coefficient(a, b),
                )
            except Exception as e:
                # Fallback without SSIM
                self.note_label.configure(text=f"Catatan: SSIM gagal dihitung ({e}).")
                return (
                    mse(a, b),
                    psnr(a, b),
                    float('nan'),
                    correlation_coefficient(a, b),
                )

        # Adjacent correlations on cipher image
        ch = adjacent_correlation(cipher_img, "horizontal")
        cv = adjacent_correlation(cipher_img, "vertical")
        cd = adjacent_correlation(cipher_img, "diagonal")

        # Three qualitative comparisons
        M1, P1, S1, C1 = _qual(img, dec_img)          # Asli & Dekripsi
        M2, P2, S2, C2 = _qual(cipher_img, dec_img)   # Enkripsi & Dekripsi
        M3, P3, S3, C3 = _qual(img, cipher_img)       # Asli & Enkripsi

        # Differential: encrypt flipped image with SAME IV/nonce
        img2 = img.copy()
        h, w, c = img2.shape
        y = random.randrange(h); x = random.randrange(w); chn = random.randrange(c)
        img2[y, x, chn] = (int(img2[y, x, chn]) + 1) % 256

        if enc_info.get("algo") == "aes":
            key = enc_info["key"]; iv = enc_info["iv"]
            cipher_img2, _ = aes_encrypt_image_with_iv(img2, key, iv)
            # Timing for AES encryption
            t_avg, _ = measure_time(lambda: aes_encrypt_image_with_iv(img, key, iv), repeats=repeats)
        else:
            key = enc_info["key"]; nonce = enc_info["nonce"]
            cipher_img2 = chacha_encrypt_image(img2, key, nonce)
            # Timing for ChaCha20 encryption
            t_avg, _ = measure_time(lambda: chacha_encrypt_image(img, key, nonce), repeats=repeats)

        N = npcr(cipher_img, cipher_img2)
        U = uaci(cipher_img, cipher_img2)

        # Populate tables
        self.metrics_title.configure(text=f"{name}")
        # Qualitative
        self.table_qual.insert("", "end", values=(
            "Original vs Decrypted", f"{M1:.4f}", f"{P1:.4f}", f"{S1:.6f}", f"{C1:.6f}"
        ))
        self.table_qual.insert("", "end", values=(
            "Encrypted vs Decrypted", f"{M2:.4f}", f"{P2:.4f}", f"{S2:.6f}", f"{C2:.6f}"
        ))
        self.table_qual.insert("", "end", values=(
            "Original vs Encrypted", f"{M3:.4f}", f"{P3:.4f}", f"{S3:.6f}", f"{C3:.6f}"
        ))

        # Differential / adjacency / timing
        self.table_diff.insert("", "end", values=(
            "Encrypted1 & Encrypted2", f"{N:.4f}", f"{U:.4f}", f"{ch:.6f}", f"{cv:.6f}", f"{cd:.6f}", f"{t_avg:.6f}"
        ))

        # Return second encrypted image for preview
        return cipher_img2


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
