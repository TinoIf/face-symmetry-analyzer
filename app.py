# --- app.py ---
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
import asyncio
import platform
import logging

# Mengatur level logging untuk mengurangi noise di konsol
logging.getLogger("aioice").setLevel(logging.WARNING)
logging.getLogger("aiortc").setLevel(logging.WARNING)

# Konfigurasi event loop untuk environment Linux (seperti pada Streamlit Cloud)
if platform.system() == "Linux":
    try:
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    except Exception as e:
        print(f"Peringatan pengaturan event loop: {e}")

# --- 1. KONFIGURASI HALAMAN & STYLING (Diambil dari Kode Inti) ---
# Menggunakan styling yang lebih modern dan rapi dari skrip pertama.
st.set_page_config(page_title="Face-Scan AI", page_icon="✨", layout="wide")

st.markdown("""
<style>
/* --- FONT & LATAR BELAKANG --- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="st-"], [class*="css-"] {
    font-family: 'Inter', sans-serif; /* Font modern dan bersih */
    background-color: #0E1117; /* Latar belakang dark-mode */
    color: #FAFAFA;
}
.block-container {
    padding: 2rem 3rem 1rem 3rem; /* Padding: atas-bawah | kiri-kanan */
}

/* --- JUDUL & SUB-JUDUL --- */
.title { text-align: center; font-size: 2.7rem; margin-bottom: 0.3rem; font-weight: 800; }
.subtitle { text-align: center; color: #8A919E; margin-bottom: 3rem; }

/* --- CARD & LAYOUT --- */
/* Mengatur jarak antar kolom Streamlit utama */
div[data-testid="stHorizontalBlock"] {
    gap: 2rem; /* Ubah untuk jarak antar card */
}

/* --- TOMBOL & INPUT MODERN --- */
.stForm {
    background-color: transparent; /* Membuat latar belakang form transparan */
    border: none;
    padding: 0;
}
/* Styling untuk tombol "Analisis" */
.stButton>button {
    height: 48px;
    border-radius: 10px;
    font-weight: 600;
    font-size: 1rem;
    background-color: #60A5FA; /* Warna Tailwind Blue-400 */
    transition: all 0.2s ease-in-out;
    border: none;
    width: 100%; /* Tombol mengisi lebar kolomnya */
}
.stButton>button:hover {
    background-color: #3B82F6; /* Warna Tailwind Blue-500 saat hover */
    transform: translateY(-2px);
}
/* Styling untuk kotak input username */
.stTextInput>div>input {
    height: 48px;
    border-radius: 10px;
    background-color: #31333F;
    border: 1px solid #4A4A6A;
}

/* --- TABS MODERN & JELAS --- */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    border-bottom: 2px solid #31333F; /* Garis bawah bar tab */
}
.stTabs [data-baseweb="tab"] {
    padding: 12px 16px;
    border-radius: 10px 10px 0 0;
    font-weight: 600;
    color: #8A919E; /* Warna tab tidak aktif */
    border-bottom: 2px solid transparent;
    transition: all 0.3s ease;
}
.stTabs [aria-selected="true"] {
    color: #FFFFFF; /* Warna teks tab aktif */
    border-bottom: 2px solid #60A5FA; /* Garis bawah biru pada tab aktif */
}

/* --- PAPAN PERINGKAT --- */
.leaderboard-entry {
    padding: 10px 5px;
    border-bottom: 1px solid #31333F;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.leaderboard-entry .rank { font-weight: 700; color: #60A5FA; }
</style>
""", unsafe_allow_html=True)

# --- 2. INISIALISASI STATE & MODEL (Diambil dari Kode Inti) ---
if "leaderboard" not in st.session_state: st.session_state.leaderboard = []
if "analysis_result" not in st.session_state: st.session_state.analysis_result = None
if "active_tab" not in st.session_state: st.session_state.active_tab = "Peringkat"

@st.cache_resource
def load_models():
    """Memuat model face detector dan landmark predictor sekali saja."""
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    return detector, predictor
face_detector, landmark_predictor = load_models()

# --- 3. FUNGSI LOGIKA INTI (Diambil dari Kode Inti & Diperkaya) ---
SYMMETRY_PAIRS = [(0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9), (17, 26), (18, 25), (19, 24), (20, 23), (21, 22), (36, 45), (37, 44), (38, 43), (39, 42), (48, 54), (49, 53), (50, 52), (60, 64), (61, 63)]

def calculate_metrics(points):
    """Menghitung skor Simetri dan Golden Ratio dari titik landmark."""
    if not points: return 0, 0
    center_x = np.mean([p.x for p in points])
    symmetry_score = np.mean([abs((points[l].x - center_x) + (points[r].x - center_x)) for l, r in SYMMETRY_PAIRS])

    face_height = points[8].y - np.mean([points[19].y, points[24].y])
    face_width = points[16].x - points[0].x
    golden_ratio_score = abs(face_height / face_width) if face_width > 0 else 0
    return symmetry_score, golden_ratio_score

def get_dynamic_messages(symmetry_score, golden_ratio_score):
    """Membuat narasi dinamis yang menarik berdasarkan skor."""
    if symmetry_score < 1.5:
        sym_title = "Level Dewa! ✨"
        sym_text = "Struktur wajahmu menunjukkan tingkat simetri yang luar biasa tinggi. Ini adalah kualitas langka yang sering diasosiasikan dengan daya tarik universal."
    elif symmetry_score < 3.0:
        sym_title = "Aura Keren Terdeteksi! 😎"
        sym_text = "Selamat, wajahmu memiliki keseimbangan dan proporsi yang sangat baik. Tingkat simetri ini adalah fondasi dari penampilan yang menarik dan mudah disukai."
    else:
        sym_title = "Punya Karakter Kuat! 😉"
        sym_text = "Wajahmu menunjukkan keunikan tersendiri. Sedikit asimetri justru menambah karakter dan membuatmu mudah diingat. Karisma tidak selalu tentang angka!"

    ratio_diff = abs(golden_ratio_score - 1.618)
    if ratio_diff < 0.1:
        gr_text = "Proporsi vertikal dan horizontal wajahmu sangat mendekati Golden Ratio (1.618), sebuah harmoni matematis yang ditemukan dalam seni dan alam."
    else:
        gr_text = "Setiap wajah memiliki proporsi uniknya sendiri yang membuatnya istimewa dan berbeda dari yang lain."
    return sym_title, sym_text, gr_text

def analyze_and_draw(image):
    """Fungsi tunggal untuk analisis mendalam saat tombol ditekan."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
    if len(faces) == 0: return {"face_found": False}

    (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
    landmarks = landmark_predictor(gray, rect)
    points = [landmarks.part(i) for i in range(68)]

    symmetry_score, golden_ratio = calculate_metrics(points)
    sym_title, sym_text, gr_text = get_dynamic_messages(symmetry_score, golden_ratio)

    output = image.copy()
    for pt in points: cv2.circle(output, (pt.x, pt.y), 2, (0, 255, 255), -1)
    cv2.rectangle(output, (x, y), (x + w, y + h), (255, 192, 203), 2)
    center_x_for_line = np.mean([p.x for p in points])
    cv2.line(output, (int(center_x_for_line), y), (int(center_x_for_line), y + h), (0, 255, 0), 1)

    return {"face_found": True, "score": symmetry_score, "ratio": golden_ratio, "image": output,
            "sym_title": sym_title, "sym_text": sym_text, "gr_text": gr_text}


class VideoProcessor(VideoTransformerBase):
    """
    Kelas prosesor video.
    DIKEMBALIKAN: Fitur live score dihapus sesuai permintaan.
    Hanya menampilkan deteksi visual saja.
    """
    def __init__(self):
        self.latest_frame = None
        self.lock = threading.Lock()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        with self.lock: self.latest_frame = img.copy()

        # --- Logika deteksi visual (tanpa skor live) ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0, 0.5), 2)
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            landmarks = landmark_predictor(gray, rect)
            points = [landmarks.part(i) for i in range(68)]
            for pt in points:
                cv2.circle(img, (pt.x, pt.y), 2, (0, 213, 255), -1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 4. UI UTAMA STREAMLIT (Diambil dari Kode Inti) ---
st.markdown("<div class='title'>Face-Scan: AI Symmetry Analysis</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Seberapa keren proporsi wajahmu menurut AI?</div>", unsafe_allow_html=True)

col1, col2 = st.columns([1.05, 0.95])

# --- Kolom Kiri: Kamera dan Kontrol ---
with col1:
    st.subheader("📸 Kamera Live")
    
    # DITAMBAHKAN: Blok try-except untuk menangani eror saat memulai WebRTC
    ctx = None # Inisialisasi ctx sebagai None
    try:
        ctx = webrtc_streamer(key="live", mode=WebRtcMode.SENDRECV, video_processor_factory=VideoProcessor,
                              rtc_configuration=RTCConfiguration(
                                  {
                                      "iceServers": [
                                          {"urls": ["stun:stun.l.google.com:19302"]},
                                          {"urls": ["stun:stun1.l.google.com:19302"]},
                                          {"urls": ["stun:stun2.l.google.com:19302"]},
                                          {
                                              "urls": "turn:openrelay.metered.ca:80",
                                              "username": "openrelayproject",
                                              "credential": "openrelayproject",
                                          },
                                          {
                                              "urls": "turn:openrelay.metered.ca:443",
                                              "username": "openrelayproject",
                                              "credential": "openrelayproject",
                                          },
                                      ]
                                  }
                              ),
                              media_stream_constraints={"video": True, "audio": False}, async_processing=True)
    except Exception as e:
        st.error(f"""
            **Kamera tidak dapat dimulai.** Ini bisa terjadi karena beberapa alasan:
            - Anda belum memberikan izin akses kamera untuk situs ini.
            - Browser Anda tidak mendukung WebRTC.
            - Ada masalah jaringan.
            
            **Saran:** Coba refresh halaman dan pastikan Anda mengklik 'Allow' saat diminta izin kamera.
        """, icon="📹")

    # Menggunakan st.form untuk pengalaman input yang lebih baik
    with st.form(key="analysis_form"):
        form_col1, form_col2 = st.columns([3, 1.5])
        with form_col1:
            username = st.text_input("Username", label_visibility="collapsed", placeholder="Tulis namamu di sini...")
        with form_col2:
            analyze_button = st.form_submit_button(label="Analisis")

        if analyze_button:
            if not username:
                st.warning("Harap masukkan username terlebih dahulu!", icon="⚠️")
            # DIPERBAIKI: Menambahkan pengecekan 'ctx' untuk memastikan kamera berhasil dimuat
            elif ctx and ctx.state.playing and ctx.video_processor:
                with ctx.video_processor.lock:
                    frame = ctx.video_processor.latest_frame
                if frame is not None:
                    with st.spinner("Menganalisis..."):
                        result = analyze_and_draw(frame)
                    st.session_state.analysis_result = result
                    if result["face_found"]:
                        st.session_state.leaderboard.append({"username": username, "score": result["score"]})
                        st.session_state.leaderboard.sort(key=lambda x: x["score"])
                        st.session_state.active_tab = "Analisis" # Pindah tab otomatis
                        st.rerun() # Jalankan ulang skrip untuk update UI
                    else:
                        st.error("Wajah tidak terdeteksi. Coba lagi.", icon="😞")
                else:
                     st.error("Gagal mengambil frame dari kamera.", icon="📹")
            else:
                st.error("Kamera tidak aktif. Harap tekan 'START' pada video.", icon="📹")

# --- Kolom Kanan: Hasil Analisis ---
with col2:
    tab_titles = ["🏆 Peringkat", "📊 Analisis"]
    if st.session_state.active_tab == "Analisis":
        default_tab_index = 1
    else:
        default_tab_index = 0

    tab1, tab2 = st.tabs(tab_titles)

    with tab1:
        st.subheader("Papan Peringkat Sesi Ini")
        if not st.session_state.leaderboard:
            st.info("Belum ada yang melakukan analisis. Jadilah yang pertama!")
        else:
            for i, d in enumerate(st.session_state.leaderboard[:10]):
                st.markdown(f"""
                    <div class='leaderboard-entry'>
                        <div><span class='rank'>#{i+1}</span> &nbsp; {d['username']}</div>
                        <div>{d['score']:.2f}</div>
                    </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.subheader("Hasil Analisis Wajah")
        result = st.session_state.analysis_result
        if result and result["face_found"]:
            img_col, metric_col = st.columns([1, 1.1])
            with img_col:
                st.image(result["image"], channels="BGR", caption="Hasil Deteksi", use_container_width=True)
            with metric_col:
                st.metric("Skor Simetri", f"{result['score']:.2f}", help="Lebih rendah = lebih simetris")
                st.metric("Golden Ratio", f"{result['ratio']:.3f}", help="Idealnya ~1.618")

            st.markdown("---")
            st.subheader(result['sym_title'])
            st.write(result['sym_text'])
            st.info(f"**Fakta Golden Ratio:** {result['gr_text']}", icon="💡")
            st.session_state.active_tab = "Peringkat"
        else:
            st.info("Klik tombol 'Analisis' setelah wajah terdeteksi untuk melihat hasilnya di sini.")
