import os
import base64
import random
import cv2
import dlib
import numpy as np
import streamlit as st
from streamlit_webrtc import (
    webrtc_streamer,
    VideoTransformerBase,
    WebRtcMode,
    RTCConfiguration,
)
import av
import threading
import platform
import asyncio

# ---------- Konfigurasi Halaman ----------
st.set_page_config(page_title="Face-Scan AI", page_icon="🎨", layout="wide")

# Perbaikan event loop untuk deployment di Linux
if platform.system() == "Linux":
    try:
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    except Exception as e:
        print("Peringatan event loop:", e)

# ---------- Session State ----------
if "page" not in st.session_state:
    st.session_state.page = "landing"
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "show_details" not in st.session_state:
    st.session_state.show_details = False
if 'webrtc_ctx' not in st.session_state:
    st.session_state['webrtc_ctx'] = None


# ---------- Fungsi Navigasi & Aksi ----------
def go_to_camera():
    st.session_state.page = "camera"

def capture_and_analyze():
    """Fungsi untuk mengambil gambar dan menganalisis, dipanggil oleh tombol."""
    ctx_retrieved = st.session_state.get('webrtc_ctx')
    if ctx_retrieved and ctx_retrieved.state.playing and ctx_retrieved.video_processor:
        with ctx_retrieved.video_processor.lock:
            frame = ctx_retrieved.video_processor.latest_frame
        if frame is not None:
            with st.spinner("Menganalisis wajahmu..."):
                result = analyze_and_draw(frame)
                st.session_state.analysis_result = result
                st.session_state.page = "result"
                st.session_state.show_details = False
                st.rerun()
        else:
            st.error("Gagal mengambil frame dari kamera. Pastikan kamera aktif dan wajah terlihat.")
    else:
        st.warning("Kamera belum siap. Mohon tunggu atau tekan 'START' pada video.")

# ---------- Helper: Memuat gambar maskot sebagai base64 ----------
def img_to_base64(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Pastikan path ke aset benar
assets_path = os.path.join(os.path.dirname(__file__), "assets")
maskot_base64 = img_to_base64(os.path.join(assets_path, "maskot.png"))


# ---------- CSS (Dengan Palet Warna Retro & Perbaikan Layout) ----------
st.markdown(
    """
<style>
/* MODIFIKASI 1: Palet Warna Retro & Font */
/* Background: #FDF6E3 (Solarized Light) */
/* Aksen: #CB4B16 (Orange Retro) */
/* Teks & Border: #586E75 (Slate Gelap) */

.stApp, html, body, [class*="css-"], [class*="st-"] {
    background-color: #FDF6E3 !important; /* Warna krem retro */
    font-family: 'Poppins', sans-serif;
    color: #586E75; /* Warna teks slate gelap */
}

/* Container utama */
.main-container {
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:flex-start; /* MODIFIKASI: Diubah dari 'center' ke 'flex-start' */
    text-align:center;
    padding: 10vh 1rem 0 1rem; /* MODIFIKASI: Menggunakan padding-top untuk jarak dari atas */
    min-height: auto; /* MODIFIKASI: Menghapus min-height agar tidak ada ruang kosong berlebih */
    position: relative;
    z-index: 10;
}

/* Judul (typing) */
.title {
    font-family: 'Lilita One', cursive;
    font-size: 4.2rem;
    color: #CB4B16; /* Warna aksen retro */
    margin: 0;
    line-height: 1.1;
}
.typing {
    display:inline-block;
    white-space:nowrap;
    overflow:hidden;
    border-right: .18em solid rgba(88, 110, 117, 0.9);
    width: 0;
    animation: typing 1.6s steps(13, end) forwards, blink-caret .8s step-end infinite;
}
@keyframes typing { from { width: 0; } to { width: 13ch; } }
@keyframes blink-caret { from, to { border-color: transparent } 50% { border-color: rgba(88, 110, 117, 0.9); } }

/* Subtitle */
.subtitle {
    color: #586E75;
    font-size: 1.1rem;
    max-width: 760px;
    margin-top: 0.5rem; /* MODIFIKASI 1: Mengurangi jarak dari judul */
    margin-bottom: 1rem;
    line-height: 1.5;
}

/* --- TEMPAT MENGUBAH JARAK MASKOT & TOMBOL --- */
.maskot-small {
    text-align:center;
    margin: 0 auto 15px; /* Ubah angka '15px' ini untuk mengatur jarak */
    z-index: 10;
}
.maskot-small img { width: 120px; height: auto; }

/* MODIFIKASI 1: Tombol dengan gaya retro dan efek hover baru */
/* Target elemen tombol Streamlit yang lebih spesifik */
.stButton>button {
    font-family: 'Poppins', sans-serif;
    height: 55px;
    width: 280px;
    border-radius: 15px;
    font-weight: 600;
    font-size: 1.1rem;
    color: #FDF6E3 !important; /* Teks warna background */
    background: #CB4B16; /* Background warna aksen */
    border: 3px solid #586E75; /* Border warna teks */
    box-shadow: 4px 4px 0px #586E75;
    transition: all 0.15s cubic-bezier(.25,.8,.25,1);
    display: block;
    margin: 0 auto !important; /* !important untuk memastikan centering */
}
.stButton>button:hover {
    transform: translateY(-3px) translateX(-2px);
    box-shadow: 7px 7px 0px #586E75;
    background: #b54010; /* Warna aksen sedikit lebih gelap saat hover */
    color: white !important;
}
.stButton>button:active {
    transform: translateY(2px) translateX(1px);
    box-shadow: 2px 2px 0px #586E75;
}

/* Style untuk halaman hasil */
.result-container {
    padding: 1rem;
    margin-top: 2rem; /* Mendorong teks hasil ke bawah */
    margin-bottom: 2rem;
}
.result-title {
    font-family: 'Lilita One', cursive;
    font-size: 2.8rem; /* Sedikit dikecilkan agar muat */
    color: #CB4B16;
    margin-bottom: 0.5rem;
    line-height: 1.2; /* Menambah jarak antar baris */
}
.result-text {
    font-size: 1.3rem; /* Ukuran font lebih besar */
    max-width: 800px;
    margin: 0 auto;
    line-height: 1.6;
}
/* Tombol di halaman hasil dibuat full-width untuk tampilan mobile yang lebih baik */
.stButton>button[kind="secondary"] {
    width: 100%;
}

/* Efek Salju (Hanya untuk Landing Page) */
.snowflake {
    position: fixed; top: -5vh; pointer-events: none; z-index: 1;
    opacity: 0; transform: translateY(-10vh);
    animation-name: fallAndFade; animation-timing-function: linear;
    animation-iteration-count: infinite; will-change: transform, opacity;
    text-shadow: 0 0 6px rgba(255,255,255,0.6);
    filter: drop-shadow(0 0 4px rgba(0,0,0,0.08));
}
@keyframes fallAndFade {
    0% { transform: translateY(-10vh) scale(0.9); opacity: 0; }
    10% { opacity: 0.9; } 90% { opacity: 0.9; }
    100% { transform: translateY(110vh) scale(1); opacity: 0; }
}

/* Responsive */
@media (max-width: 720px) {
    .title { font-size: 2.8rem; }
    .result-title { font-size: 2.2rem; }
    .result-text { font-size: 1.1rem; }
    .maskot-small img { width: 90px; }
    .stButton>button { width: 92%; height: 50px; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- (MODEL LOADING) ----------
@st.cache_resource
def load_models():
    # Menggunakan path absolut untuk memastikan file ditemukan
    base_path = os.path.dirname(__file__)
    detector_path = os.path.join(base_path, "haarcascade_frontalface_default.xml")
    predictor_path = os.path.join(base_path, "shape_predictor_68_face_landmarks.dat")
    
    detector = cv2.CascadeClassifier(detector_path)
    predictor = dlib.shape_predictor(predictor_path)
    return detector, predictor

face_detector, landmark_predictor = load_models()

# ---------- FUNGSI ANALISIS (Dengan Teks Output Baru) ----------
SYMMETRY_PAIRS = [
    (0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9),
    (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),
    (36, 45), (37, 44), (38, 43), (39, 42),
    (48, 54), (49, 53), (50, 52),
    (60, 64), (61, 63)
]
def calculate_metrics(points):
    if not points: return 0, 0
    center_x = np.mean([p.x for p in points])
    symmetry_score = np.mean([abs((points[l].x - center_x) + (points[r].x - center_x)) for l, r in SYMMETRY_PAIRS])
    face_height = points[8].y - np.mean([points[19].y, points[24].y])
    face_width = points[16].x - points[0].x
    golden_ratio_score = abs(face_height / face_width) if face_width > 0 else 0
    return symmetry_score, golden_ratio_score

# MODIFIKASI: Teks output utama dan sub-penjelasan diubah sesuai permintaan
def get_booth_verdict(symmetry_score):
    if symmetry_score < 5:
        # Teks untuk hasil "ganteng" atau "cantik"
        return "🔥 Lo Kerennn Banget 🔥", "Pose lo udah mantap!!! AI aja sampe ngakuin pesona lo yang bikin AUTO STAND OUT. 🚀"
    else:
        # Teks untuk hasil "tidak ganteng" atau "tidak cantik"
        return "Lo Gak Ganteng, Lo Gak Cantik...<br>TAPI BISA STAND OUT!!!!! 👑", "Karisma itu soal VIBE, bukan cuma simetri. Tunjukin pose andalan lo & buktiin AI-nya salah!"

def analyze_and_draw(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
    if len(faces) == 0: return {"face_found": False}
    (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
    landmarks = landmark_predictor(gray, rect)
    points = [landmarks.part(i) for i in range(68)]
    symmetry_score, golden_ratio = calculate_metrics(points)
    verdict_title, verdict_text = get_booth_verdict(symmetry_score)
    output_image = image.copy()
    for pt in points: cv2.circle(output_image, (pt.x, pt.y), 3, (0, 255, 255), -1)
    cv2.rectangle(output_image, (x, y), (x + w, y + h), (214, 51, 108), 3)
    return {"face_found": True, "score": symmetry_score, "ratio": golden_ratio, "image": output_image, "verdict_title": verdict_title, "verdict_text": verdict_text}

class VideoProcessor(VideoTransformerBase):
    def __init__(self): self.latest_frame = None; self.lock = threading.Lock()
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24"); img = cv2.flip(img, 1)
        with self.lock: self.latest_frame = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
        for (x, y, w, h) in faces: cv2.rectangle(img, (x, y), (x + w, y + h), (30, 144, 255), 3)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------- RENDER HALAMAN LANDING ----------
def show_landing_page():
    # Efek salju hanya dirender di halaman ini
    NUM_SNOW = 45
    snow_html_parts = []
    for i in range(NUM_SNOW):
        left_pct = random.uniform(1, 99)
        size_px = random.uniform(10, 22)
        duration_s = random.uniform(6.0, 14.0)
        delay_s = random.uniform(-duration_s, 0)
        color = random.choice(["#ffffff", "#ffd6e7", "#ffe9b3", "#f8f0ff"])
        style = (
            f"left: {left_pct}%; font-size: {size_px}px; color: {color}; "
            f"animation-duration: {duration_s}s; animation-delay: {delay_s}s;"
        )
        snow_html_parts.append(f"<div class='snowflake' style='{style}'>❄</div>")
    snow_html = "\n".join(snow_html_parts)
    st.markdown(snow_html, unsafe_allow_html=True)

    # Menggabungkan HTML untuk layout yang lebih baik
    st.markdown(
        f"""
        <div class="main-container">
            <h1 class="title"><span class="typing">Face-Scan AI</span></h1>
            <p class="subtitle">Analisis seberapa proporsional wajahmu dengan teknologi AI. Dirancang khusus untuk pengalaman interaktif di booth pameran!</p>
            {'<div class="maskot-small"><img src="data:image/png;base64,' + maskot_base64 + '" alt="maskot"></div>' if maskot_base64 else ''}
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- TEMPAT UNTUK MENGUBAH POSISI TOMBOL ---
    # Anda bisa mengubah angka di dalam list [1, 0.8, 1] untuk menggeser tombol.
    # Angka pertama: spasi kiri. Angka kedua: lebar tombol. Angka ketiga: spasi kanan.
    # Untuk menggeser ke kanan, perbesar angka pertama (contoh: 1.1) dan perkecil angka ketiga (contoh: 0.9).
    _, btn_col, _ = st.columns([1.4, 0.8, 1]) 
    with btn_col:
        if st.button("Mulai Sekarang!", key="start_btn"):
            go_to_camera()

# ---------- RENDER HALAMAN KAMERA ----------
def show_camera_page():
    st.markdown("<div class='subtitle' style='text-align: center; margin-bottom: 2rem;'>Posisikan wajah di dalam frame dan siapkan pose terbaikmu!</div>", unsafe_allow_html=True)
    
    col_camera, col_controls = st.columns([2, 1])
    with col_camera:
        ctx = webrtc_streamer(
            key="live",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )
        if ctx:
            st.session_state['webrtc_ctx'] = ctx

    with col_controls:
        st.write("")
        st.markdown("<p style='font-weight: 600; font-size: 1.1rem;'>Siap untuk dianalisis?</p>", unsafe_allow_html=True)
        st.markdown("<p>Klik tombol di bawah ini untuk memulai.</p>", unsafe_allow_html=True)

        if st.button("✨ Ambil Gambar & Analisis ✨"):
            capture_and_analyze()

# ---------- RENDER HALAMAN HASIL ----------
def show_result_page():
    result = st.session_state.analysis_result
    if not result or not result.get("face_found"):
        st.error("Wajah tidak terdeteksi. Silakan coba lagi.")
        if st.button("Kembali ke Kamera", use_container_width=True):
            st.session_state.page = "camera"; st.rerun()
        return

    # Layout teks hasil diperbarui dengan CSS
    st.markdown(f"""
        <div class='main-container' style='min-height: auto;'>
            <div class='result-container'>
                <div class='result-title'>{result['verdict_title']}</div>
                <div class='result-text'>{result['verdict_text']}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Layout tombol dibuat sejajar di tengah
    _, btn_container, _ = st.columns([1, 1.5, 1])
    with btn_container:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Coba Lagi", use_container_width=True, type="secondary"):
                st.session_state.page = "camera"
                st.rerun()
        with col2:
            if st.button("Lihat Detail Teknis", use_container_width=True, type="secondary"):
                st.session_state.show_details = not st.session_state.show_details
                st.rerun()

    if st.session_state.show_details:
        st.markdown("---"); st.subheader("Detail Teknis Analisis")
        img_col, metric_col = st.columns([1, 1.1])
        with img_col: st.image(result["image"], channels="BGR", caption="Wajah yang Dianalisis", use_container_width=True)
        with metric_col:
            st.metric("Skor Simetri", f"{result['score']:.2f}", help="Lebih rendah = lebih simetris.")
            st.metric("Golden Ratio", f"{result['ratio']:.3f}", help="Rasio ideal ~1.618.")
            st.info("Skor ini adalah pengukuran matematis untuk hiburan dan tidak mendefinisikan kecantikan seutuhnya.", icon="💡")

# ---------- Router Halaman ----------
if st.session_state.page == "landing":
    show_landing_page()
elif st.session_state.page == "camera":
    show_camera_page()
elif st.session_state.page == "result":
    show_result_page()

