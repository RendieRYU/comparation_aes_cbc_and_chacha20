import os
import json
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageTk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
    img = Image.open(path)

    # 1) Grayscale murni
    if img.mode == "L":
        return np.array(img, dtype=np.uint8)

    # 2) Grayscale + alpha -> ambil channel L saja
    if img.mode == "LA":
        return np.array(img.getchannel(0), dtype=np.uint8)

    # 3) RGB/RGBA: tetap warna, tapi jika ketiga channel identik -> jadikan 1 channel
    if img.mode in ("RGB", "RGBA"):
        if img.mode == "RGBA":
            img = img.convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        # Deteksi “monochrome” (semua channel sama)
        if np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 0], arr[..., 2]):
            return arr[..., 0]
        return arr

    # 4) Mode lain (P, I;16, F, dsb) -> map ke grayscale 8-bit
    return np.array(img.convert("L"), dtype=np.uint8)

def save_image(arr: np.ndarray, path: str) -> None:
    Image.fromarray(arr).save(path)

def choose_save_ext(src_path: str) -> str:
    ext = Path(src_path).suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return ".jpg"
    if ext in (".tif", ".tiff"):
        return ".tiff"
    if ext in (".png", ".bmp"):
        return ext
    # fallback aman (lossless)
    return ".png"

def match_shape(orig: np.ndarray, arr: np.ndarray) -> np.ndarray:
    # Grayscale asli (2D) → kompres 3-channel ke 1-channel
    if orig.ndim == 2 and arr.ndim == 3:
        return np.mean(arr, axis=2).astype(np.uint8)
    # RGB asli (3D) → ekspansi 1-channel ke 3-channel
    if orig.ndim == 3 and arr.ndim == 2:
        return np.repeat(arr[:, :, None], 3, axis=2)
    return arr

def describe_np(arr: np.ndarray) -> str:
    if arr.ndim == 2:
        return f"Grayscale (L) | shape={arr.shape}"
    elif arr.ndim == 3:
        ch = arr.shape[2]
        mode = "RGB" if ch == 3 else f"{ch}-channel"
        return f"{mode} | shape={arr.shape}"
    return f"{arr.ndim}D | shape={arr.shape}"

def describe_pil(pil_img: Image.Image) -> str:
    mode_names = {"L": "Grayscale (L)", "LA": "Grayscale (LA)", "RGB": "RGB", "RGBA": "RGBA"}
    return f"{mode_names.get(pil_img.mode, pil_img.mode)} | size={pil_img.size}"

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
        self.orig_info = ttk.Label(self.tile_orig, text="", foreground="#666")
        self.orig_info.pack()

        ttk.Label(self.tile_enc1, text="Encrypted").pack()
        self.cipher_canvas = tk.Label(self.tile_enc1)
        self.cipher_canvas.pack(pady=4)
        self.cipher_info = ttk.Label(self.tile_enc1, text="", foreground="#666")
        self.cipher_info.pack()

        ttk.Label(self.tile_enc2, text="Encrypted2").pack()
        self.cipher2_canvas = tk.Label(self.tile_enc2)
        self.cipher2_canvas.pack(pady=4)
        self.cipher2_info = ttk.Label(self.tile_enc2, text="", foreground="#666")
        self.cipher2_info.pack()

        ttk.Label(self.tile_dec, text="Decrypted").pack()
        self.dec_canvas = tk.Label(self.tile_dec)
        self.dec_canvas.pack(pady=4)
        self.dec_info = ttk.Label(self.tile_dec, text="", foreground="#666")
        self.dec_info.pack()

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
            # height=6,
            height=3,
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
            # height=5,
            height=2,
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

        self.table_proc = ttk.Treeview(
            right,
            columns=("Langkah", "Status", "Detail", "Time (s)"),
            show="headings",
            # height=8,
            height=6,
        )
        for col, width in (
            ("Langkah", 160), ("Status", 90), ("Detail", 260), ("Time (s)", 90)
        ):
            self.table_proc.heading(col, text=col)
            self.table_proc.column(col, width=width, anchor="w" if col == "Detail" else "center")
        self.table_proc.pack(fill=tk.X, expand=False, pady=6)

        self.hist_frame = ttk.LabelFrame(right, text="Histogram")
        self.hist_frame.pack(fill=tk.BOTH, expand=True, pady=6)
        # self.fig = Figure(figsize=(5.6, 3.2), dpi=100)
        self.fig = Figure(figsize=(5.6, 6.8), dpi=100)
        self.ax_orig = self.fig.add_subplot(1, 2, 1)
        self.ax_cipher = self.fig.add_subplot(1, 2, 2)
        self.ax_orig.set_title("Original")
        self.ax_cipher.set_title("Encrypted")
        self.canvas_hist = FigureCanvasTkAgg(self.fig, master=self.hist_frame)
        self.canvas_hist.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _plot_hist(self, ax, arr: np.ndarray, title: str):
        ax.clear()
        if arr.ndim == 2:
            ax.hist(arr.flatten(), bins=256, range=(0, 255), color="gray", alpha=0.85)
        else:
            colors = ["r", "g", "b"]
            ch = arr.shape[2]
            for i in range(min(3, ch)):
                ax.hist(arr[..., i].flatten(), bins=256, range=(0, 255), color=colors[i], alpha=0.5, label=colors[i].upper())
            ax.legend(loc="upper right", fontsize=8)
        ax.set_xlim(0, 255)
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Count")
        ax.set_title(title)

    def _update_histograms(self, img: np.ndarray, cipher_img: np.ndarray):
        self._plot_hist(self.ax_orig, img, "Original")
        self._plot_hist(self.ax_cipher, cipher_img, "Encrypted")
        self.fig.tight_layout()
        self.canvas_hist.draw()

    def _choose_image(self):
        path = filedialog.askopenfilename(title="Pilih gambar", filetypes=[
            ("Images", ".png .jpg .jpeg .bmp .tif .tiff"), ("All files", ".*")
        ])
        if path:
            self.image_path = path
            self.image_entry.delete(0, tk.END)
            self.image_entry.insert(0, path)
            try:
                img = Image.open(path)
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                self._set_preview(self.orig_canvas, img)
                self.orig_info.configure(text=describe_pil(img))
            except Exception as e:
                messagebox.showerror("Error", f"Gagal membuka gambar: {e}")

    def _choose_output_dir(self):
        path = filedialog.askdirectory(title="Pilih folder output")
        if path:
            self.output_dir_var.set(path)

    def _set_preview(self, widget: tk.Label, pil_img: Image.Image):
        # Resize to fit a fixed bounding box so three images fit vertically
        MAX_W, MAX_H = 230, 230
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

    def _log_step(self, name: str, detail: str = ""):
        start = time.perf_counter()
        item = self.table_proc.insert("", "end", values=(name, "Running", detail, ""))
        self.update_idletasks()
        return item, start

    def _log_finish(self, item, start: float, detail: str = "OK"):
        elapsed = time.perf_counter() - start
        vals = list(self.table_proc.item(item, "values"))
        vals[1] = "Done"
        vals[2] = detail
        vals[3] = f"{elapsed:.6f}"
        self.table_proc.item(item, values=vals)
        self.update_idletasks()

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

        save_ext = choose_save_ext(path)

        # Clear tables and note
        self.table_qual.delete(*self.table_qual.get_children())
        self.table_diff.delete(*self.table_diff.get_children())
        self.table_proc.delete(*self.table_proc.get_children())  # <— clear proses
        self.note_label.configure(text="")

        try:
            # Load image
            it, t0 = self._log_step("Load image", Path(path).name)
            img = load_image(path)
            self._log_finish(it, t0, describe_np(img))
        except Exception as e:
            self._log_step("Error", f"Gagal load: {e}")
            messagebox.showerror("Error", f"Gagal memuat gambar: {e}")
            return

        # Show original preview
        self._set_preview(self.orig_canvas, Image.fromarray(img))
        self.orig_info.configure(text=describe_np(img))

        # Clear tables and note
        self.table_qual.delete(*self.table_qual.get_children())
        self.table_diff.delete(*self.table_diff.get_children())
        self.note_label.configure(text="")

        try:
            if algorithm == "AES-CBC":
                it, t0 = self._log_step("Derive AES key")
                aes_key, aes_salt = aes_derive_key(password)
                self._log_finish(it, t0, f"salt={aes_salt.hex()[:8]}...")

                it, t0 = self._log_step("AES encrypt", "CBC + PKCS7")
                aes_cipher_img, aes_iv, aes_full_ct = aes_encrypt_image(img, aes_key)
                aes_cipher_img = match_shape(img, aes_cipher_img)
                self._log_finish(it, t0, f"iv={aes_iv.hex()[:8]}..., len={len(aes_full_ct)}")

                it, t0 = self._log_step("AES decrypt")
                aes_plain = aes_decrypt_bytes(aes_key, aes_iv, aes_full_ct)
                aes_plain_img = np.frombuffer(aes_plain[: img.size], dtype=np.uint8).reshape(img.shape)
                aes_plain_img = match_shape(img, aes_plain_img)
                self._log_finish(it, t0)

                it, t0 = self._log_step("Save outputs", f"format {save_ext}")
                save_image(aes_cipher_img, os.path.join(out_dir, "aes_cbc", f"cipher{save_ext}"))
                save_image(aes_plain_img,  os.path.join(out_dir, "aes_cbc", f"decrypted{save_ext}"))
                with open(os.path.join(out_dir, "aes_cbc", "cipher.bin"), "wb") as f:
                    f.write(aes_full_ct)
                with open(os.path.join(out_dir, "aes_cbc", "meta.json"), "w", encoding="utf-8") as f:
                    json.dump({"salt_hex": aes_salt.hex(), "iv_hex": aes_iv.hex()}, f, indent=2)
                self._log_finish(it, t0)

                # Previews
                self._set_preview(self.cipher_canvas, Image.fromarray(aes_cipher_img))
                self.cipher_info.configure(text=describe_np(aes_cipher_img))
                self._set_preview(self.dec_canvas, Image.fromarray(aes_plain_img))
                self.dec_info.configure(text=describe_np(aes_plain_img))

                self._update_histograms(img, aes_cipher_img)

                # Metrics
                it, t0 = self._log_step("Compute metrics")
                cipher_img2 = self._write_metrics(
                    name="AES-CBC",
                    img=img,
                    cipher_img=aes_cipher_img,
                    dec_img=aes_plain_img,
                    repeats=repeats,
                    enc_info={"algo": "aes", "key": aes_key, "iv": aes_iv},
                )
                self._log_finish(it, t0)

                # Preview Encrypted2
                if cipher_img2 is not None:
                    it, t0 = self._log_step("Preview Encrypted2")
                    self._set_preview(self.cipher2_canvas, Image.fromarray(cipher_img2))
                    self.cipher2_info.configure(text=describe_np(cipher_img2))
                    self._log_finish(it, t0, describe_np(cipher_img2))

            else:  # ChaCha20
                it, t0 = self._log_step("Derive ChaCha20 key")
                chacha_key, chacha_salt = chacha_derive_key(password)
                self._log_finish(it, t0, f"salt={chacha_salt.hex()[:8]}...")

                it, t0 = self._log_step("ChaCha20 encrypt")
                chacha_nonce = generate_nonce()
                chacha_cipher_img = chacha_encrypt_image(img, chacha_key, chacha_nonce)
                chacha_cipher_img = match_shape(img, chacha_cipher_img)
                chacha_full_ct = chacha_encrypt_bytes(chacha_key, chacha_nonce, img.reshape(-1).tobytes())
                self._log_finish(it, t0, f"nonce={chacha_nonce.hex()[:8]}..., len={len(chacha_full_ct)}")

                it, t0 = self._log_step("ChaCha20 decrypt")
                chacha_plain = chacha_decrypt_bytes(chacha_key, chacha_nonce, chacha_full_ct)
                chacha_plain_img = np.frombuffer(chacha_plain, dtype=np.uint8).reshape(img.shape)
                chacha_plain_img = match_shape(img, chacha_plain_img)
                self._log_finish(it, t0)

                it, t0 = self._log_step("Save outputs", f"format {save_ext}")
                save_image(chacha_cipher_img, os.path.join(out_dir, "chacha20", f"cipher{save_ext}"))
                save_image(chacha_plain_img,  os.path.join(out_dir, "chacha20", f"decrypted{save_ext}"))
                with open(os.path.join(out_dir, "chacha20", "cipher.bin"), "wb") as f:
                    f.write(chacha_full_ct)
                with open(os.path.join(out_dir, "chacha20", "meta.json"), "w", encoding="utf-8") as f:
                    json.dump({"salt_hex": chacha_salt.hex(), "nonce_hex": chacha_nonce.hex()}, f, indent=2)
                self._log_finish(it, t0)

                # Previews
                self._set_preview(self.cipher_canvas, Image.fromarray(chacha_cipher_img))
                self.cipher_info.configure(text=describe_np(chacha_cipher_img))
                self._set_preview(self.dec_canvas, Image.fromarray(chacha_plain_img))
                self.dec_info.configure(text=describe_np(chacha_plain_img))

                self._update_histograms(img, chacha_cipher_img)

                # Metrics
                it, t0 = self._log_step("Compute metrics")
                cipher_img2 = self._write_metrics(
                    name="ChaCha20",
                    img=img,
                    cipher_img=chacha_cipher_img,
                    dec_img=chacha_plain_img,
                    repeats=repeats,
                    enc_info={"algo": "chacha", "key": chacha_key, "nonce": chacha_nonce},
                )
                self._log_finish(it, t0)

                # Preview Encrypted2
                if cipher_img2 is not None:
                    it, t0 = self._log_step("Preview Encrypted2")
                    self._set_preview(self.cipher2_canvas, Image.fromarray(cipher_img2))
                    self.cipher2_info.configure(text=describe_np(cipher_img2))
                    self._log_finish(it, t0, describe_np(cipher_img2))

        except Exception as e:
            self._log_step("Error", str(e))
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
        #Opsi 1: ubah pixel random
        if img2.ndim == 2:
            h, w = img2.shape
            y = random.randrange(h); x = random.randrange(w)
            img2[y, x] = (int(img2[y, x]) + 1) % 256
        else:
            h, w, c = img2.shape
            y = random.randrange(h); x = random.randrange(w); chn = random.randrange(c)
            img2[y, x, chn] = (int(img2[y, x, chn]) + 1) % 256

        #Opsi 2: ubah pixel pertama
        # if img2.ndim == 2:
        #     # Ubah pixel pertama (baris 0, kolom 0)
        #     img2[0, 0] = (int(img2[0, 0]) + 1) % 256
        # else:
        #     # Ubah pixel pertama pada channel pertama (R) 
        #     img2[0, 0, 0] = (int(img2[0, 0, 0]) + 1) % 256
        
        #Opsi 3: ubah pixel terakhir
        # if img2.ndim == 2:
        #     h, w = img2.shape
        #     img2[h-1, w-1] = (int(img2[h-1, w-1]) + 1) % 256
        # else:
        #     h, w, c = img2.shape
        #     img2[h-1, w-1, 0] = (int(img2[h-1, w-1, 0]) + 1) % 256


        if enc_info.get("algo") == "aes":
            key = enc_info["key"]; iv = enc_info["iv"]
            cipher_img2, _ = aes_encrypt_image_with_iv(img2, key, iv)
            cipher_img2 = match_shape(img2, cipher_img2)
            # Timing for AES encryption
            t_avg, _ = measure_time(lambda: aes_encrypt_image_with_iv(img, key, iv), repeats=repeats)
        else:
            key = enc_info["key"]; nonce = enc_info["nonce"]
            cipher_img2 = chacha_encrypt_image(img2, key, nonce)
            cipher_img2 = match_shape(img2, cipher_img2)
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
