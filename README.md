# ✨ Face Symmetry Analyzer AI

Aplikasi web interaktif untuk menganalisis simetri dan proporsi wajah secara *real-time* menggunakan kamera webcam. Dibangun dengan Streamlit, OpenCV, dan Dlib.

---

## 🚀 Demo Langsung

Anda bisa langsung mencoba aplikasi ini tanpa perlu instalasi melalui link berikut:

**🌐 Kunjungi Website: [[Klik untuk Test Wajah Kamu](https://face-symmetry-analyzer.streamlit.app/)]**

---

## ⚙️ Menjalankan Proyek Secara Lokal

Proyek ini memerlukan beberapa dependensi yang cara instalasinya berbeda antara Windows dan Linux. Silakan ikuti panduan yang sesuai untuk sistem operasi Anda.

---

### 🪟 **Panduan untuk Pengguna Windows**

Di Windows, kita perlu menginstal beberapa *build tools* terlebih dahulu agar `dlib` bisa di-compile.

**1. Prasyarat (Wajib Diinstal Dulu)**

* **Python**: [Unduh Python](https://www.python.org/downloads/) versi 3.9 - 3.12.
* **Git**: [Unduh Git](https://git-scm.com/downloads/).
* **Visual Studio Build Tools**: Ini adalah bagian terpenting.
    * Unduh dari [situs resmi Microsoft](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022).
    * Jalankan installer, lalu di tab **"Workloads"**, centang **"Desktop development with C++"**.
    * Di panel instalasi sebelah kanan, pastikan komponen **"C++ CMake tools for Windows"** juga ikut tercentang.
    * Selesaikan instalasi dan **restart komputer Anda**.

**2. Langkah Instalasi Proyek**

a. **Clone Repository**
```bash
git clone [https://github.com/TinoIf/face-symmetry-analyzer.git](https://github.com/TinoIf/face-symmetry-analyzer.git)
cd face-symmetry-analyzer
```
b. **Buat dan Aktifkan Virtual Environment**

```Bash
python -m venv venv
venv\Scripts\activate
```
c. **Instal Dependensi**

```Bash
pip install -r requirements.txt
```

▶️ Menjalankan Aplikasi
Setelah semua instalasi berhasil jalankan aplikasi dengan perintah:

```Bash

streamlit run app.py
Buka browser Anda dan kunjungi alamat http://localhost:8501.
```
