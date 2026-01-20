# ⚔️ Crypto Clash: AES-CBC vs. ChaCha20

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)

> Analisis komparatif mendalam antara standar enkripsi blok industri (AES-CBC) dan penantang *stream cipher* modern (ChaCha20).

---

## 📋 Daftar Isi
1. [Tentang Proyek](#-tentang-proyek)
2. [Sekilas Teori](#-sekilas-teori)
3. [Perbandingan Fitur](#-perbandingan-fitur)
4. [Instalasi dan Penggunaan](#-instalasi-dan-penggunaan)
5. [Hasil Benchmark (Contoh)](#-hasil-benchmark-contoh)
6. [Kesimpulan](#-kesimpulan)

---

## 📖 Tentang Proyek

Dalam dunia kriptografi simetris modern, dua nama sering muncul sebagai pilihan utama untuk mengamankan data: **AES (Advanced Encryption Standard)** dan **ChaCha20**.

Repositori ini bertujuan untuk memberikan perbandingan langsung antara keduanya, tidak hanya dari segi teori tetapi juga melalui pengujian praktis. Kami fokus pada implementasi **AES dalam mode CBC (Cipher Block Chaining)** dan **ChaCha20** standar.

Proyek ini berguna bagi:
* Pengembang yang bingung memilih algoritma untuk aplikasi mereka.
* Pelajar kriptografi yang ingin melihat perbedaan kinerja di dunia nyata.
* Siapa saja yang tertarik dengan keamanan data.

---

## 🧠 Sekilas Teori

Sebelum masuk ke kode, mari kita pahami perbedaan mendasar keduanya.

<div align="center">
  <img src="https://via.placeholder.com/700x300?text=Ilustrasi+Block+Cipher+(AES)+vs+Stream+Cipher+(ChaCha20)" alt="Block vs Stream Cipher Diagram">
  <br>
  <em>(Ilustrasi Konseptual: Block Cipher memproses data dalam potongan tetap, Stream Cipher mengalirkan data)</em>
</div>

### 🛡️ AES-CBC (The Industry Standard)
* **Tipe:** Block Cipher (Cipher Blok).
* **Cara Kerja:** Mengenkripsi data dalam blok berukuran tetap (16 byte). Mode CBC menghubungkan setiap blok dengan blok sebelumnya, membuat enkripsi menjadi sekuensial (berurutan).
* **Kelebihan:** Standar NIST, sangat teruji, sering memiliki akselerasi perangkat keras (AES-NI) di CPU modern.
* **Kekurangan:** Mode CBC tidak dapat diparalelisasi saat enkripsi, rentan terhadap serangan *padding oracle* jika tidak diimplementasikan dengan hati-hati, dan memerlukan *padding* untuk data yang tidak pas 16 byte.

### ⚡ ChaCha20 (The Modern Speedster)
* **Tipe:** Stream Cipher (Cipher Aliran).
* **Cara Kerja:** Menghasilkan aliran angka acak semu (keystream) yang kemudian di-XOR dengan data asli (plaintext). Berbasis pada varian Salsa20.
* **Kelebihan:** Sangat cepat dalam perangkat lunak (terutama di perangkat mobile/IoT tanpa akselerasi AES keras), aman secara default (tidak butuh padding), tahan terhadap serangan *timing*.
* **Kekurangan:** Belum selama AES dalam hal adopsi industri (meskipun sekarang digunakan luas oleh Google, Cloudflare, dll).

---

## 📊 Perbandingan Fitur Utama

Berikut adalah ringkasan cepat perbedaan teknis keduanya:

| Fitur | AES-CBC | ChaCha20 |
| :--- | :--- | :--- |
| **Jenis Cipher** | Block Cipher | Stream Cipher |
| **Ukuran Blok** | 128-bit (16 byte) | N/A (Memproses per byte/word) |
| **Panjang Kunci** | 128, 192, atau 256-bit | 256-bit |
| **Kebutuhan Padding** | Ya (Wajib PKCS#7 dll.) | Tidak |
| **Paralelisasi (Enkripsi)** | Tidak (Sekuensial) | Ya (Sangat bisa diparalelisasi) |
| **Akselerasi Hardware** | Ya (AES-NI pada Intel/AMD/ARM) | Tidak spesifik (Cepat di SW) |
| **Ketahanan Timing Attack**| Sulit diimplementasikan (rawan) | Aman secara desain (constant time) |

---

## 🛠️ Instalasi dan Penggunaan

*(Sesuaikan bagian ini dengan bahasa pemrograman dan cara menjalankan kode Anda yang sebenarnya. Contoh di bawah menggunakan asumsi Python).*

### Prasyarat
* Python 3.7+
* Pustaka Kriptografi (Contoh: `pycryptodome`)

### Langkah-langkah
1.  **Clone repositori ini:**
    ```bash
    git clone [https://github.com/username/comparation_aes_cbc_and_chacha20.git](https://github.com/username/comparation_aes_cbc_and_chacha20.git)
    cd comparation_aes_cbc_and_chacha20
    ```

2.  **Install dependensi:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Jalankan skrip benchmark:**
    ```bash
    python benchmark.py
    ```

---

## 📈 Hasil Benchmark (Contoh)

<div align="center">
  <img src="https://via.placeholder.com/700x350?text=Grafik+Benchmark+Kecepatan+(MB/s)+AES-CBC+vs+ChaCha20" alt="Grafik Benchmark Kecepatan">
  <br>
  <em>(Contoh Grafik: Perbandingan Throughput Enkripsi dalam MB/s)</em>
</div>

*Catatan: Hasil di bawah adalah ilustrasi. Hasil nyata tergantung pada perangkat keras Anda (terutama ada/tidaknya instruksi AES-NI).*

**Skenario: Enkripsi File 1GB pada CPU Modern (dengan AES-NI)**

| Algoritma | Waktu Enkripsi | Kecepatan (Perkiraan) |
| :--- | :--- | :--- |
| **AES-256-CBC** | 2.5 detik | ~400 MB/s |
| **ChaCha20** | 3.1 detik | ~320 MB/s |

> **Analisis Singkat:** Pada CPU dengan akselerasi perangkat keras, AES seringkali sedikit lebih unggul atau setara. Namun, ChaCha20 tetap sangat kompetitif.

**Skenario: Enkripsi pada Perangkat Mobile/IoT (Tanpa AES-NI)**

| Algoritma | Waktu Enkripsi | Kecepatan (Perkiraan) |
| :--- | :--- | :--- |
| **AES-256-CBC** | 15 detik | ~66 MB/s |
| **ChaCha20** | 5 detik | ~200 MB/s |

> **Analisis Singkat:** Di sinilah ChaCha20 bersinar. Tanpa bantuan hardware khusus, desain ChaCha20 yang ramah CPU membuatnya jauh lebih cepat daripada AES.

---

## 💡 Kesimpulan: Mana yang Harus Dipilih?

Tidak ada jawaban tunggal, tetapi berikut panduan umumnya:

### Pilih 🛡️ AES-CBC (atau lebih baik: AES-GCM) jika:
* Anda bekerja di lingkungan enterprise yang mewajibkan standar NIST/FIPS.
* Target perangkat keras Anda pasti memiliki akselerasi AES (sebagian besar laptop/server modern).
* Anda membutuhkan kompatibilitas dengan sistem lama (legacy).
* *Catatan: Jika memungkinkan, hindari mode CBC dan gunakan mode terautentikasi seperti GCM.*

### Pilih ⚡ ChaCha20 (biasanya dipasangkan dengan Poly1305) jika:
* Anda menargetkan berbagai perangkat, termasuk perangkat mobile lama, IoT, atau mikrokontroler yang mungkin tidak memiliki akselerasi AES.
* Anda menginginkan kinerja tinggi yang konsisten di seluruh platform hanya dengan perangkat lunak.
* Anda ingin implementasi yang lebih sederhana dan lebih aman dari serangan *timing* secara default.

---

**Disclaimer:** Proyek ini hanya untuk tujuan edukasi dan benchmarking. Untuk penggunaan di produksi, selalu gunakan pustaka kriptografi yang telah diaudit dan matang (seperti OpenSSL, BoringSSL, atau modul standar bahasa pemrograman Anda), dan utamakan penggunaan mode terautentikasi (AEAD) seperti AES-GCM atau ChaCha20-Poly1305 daripada hanya enkripsi murni.