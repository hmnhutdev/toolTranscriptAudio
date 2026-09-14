"""
Audio Transcript to AI Storyboard & Image Prompts Generator
Senior Fullstack / Python Engineer implementation
"""

import os
import json
import tempfile
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
import pandas as pd
import google.generativeai as genai
from faster_whisper import WhisperModel

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Storyboard Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom Premium High-Contrast CSS ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark sleek slate background */
.stApp {
    background: #0f172a;
    color: #f8fafc !important;
}

/* All headings */
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Sidebar Styling & Contrast */
[data-testid="stSidebar"] {
    background-color: #1e293b !important;
    border-right: 1px solid #334155 !important;
}

/* Sidebar Labels and Headers */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] h2 {
    color: #38bdf8 !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
}

/* Text Inputs, Textareas, Selectboxes */
input, textarea, select {
    color: #ffffff !important;
    background-color: #0f172a !important;
    border: 1px solid #475569 !important;
    border-radius: 8px !important;
}

input:focus, textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.25) !important;
}

/* Selectbox container */
div[data-baseweb="select"] > div {
    background-color: #0f172a !important;
    border-color: #475569 !important;
    color: #ffffff !important;
}
div[data-baseweb="select"] span {
    color: #ffffff !important;
}

/* Dropdown items */
div[role="listbox"] {
    background-color: #1e293b !important;
}
div[role="listbox"] ul li {
    color: #ffffff !important;
    background-color: #1e293b !important;
}
div[role="listbox"] ul li:hover {
    background-color: #334155 !important;
}

/* File Uploader Fixes - High Contrast */
[data-testid="stFileUploader"] {
    background-color: #1e293b !important;
    border: 2px dashed #475569 !important;
    border-radius: 14px !important;
    padding: 1.5rem !important;
}
[data-testid="stFileUploader"] section {
    background-color: #1e293b !important;
}
[data-testid="stFileUploader"] section div,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p {
    color: #f8fafc !important;
}
[data-testid="stFileUploader"] small {
    color: #94a3b8 !important;
}
[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #0284c7, #2563eb) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 0.5rem 1rem !important;
}
[data-testid="stFileUploader"] button:hover {
    background: linear-gradient(135deg, #0369a1, #1d4ed8) !important;
}

/* Action Button */
.stButton > button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 1.5rem !important;
    transition: all 0.2s ease-in-out !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(139, 92, 246, 0.4) !important;
}
.stButton > button:disabled {
    background: #334155 !important;
    color: #64748b !important;
}

/* Title banner */
.title-banner {
    background: linear-gradient(90deg, #38bdf8, #8b5cf6, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: 0.3rem;
}

.subtitle {
    color: #cbd5e1;
    font-size: 1.05rem;
    margin-bottom: 1.5rem;
}

/* Sidebar Footer */
.sidebar-footer {
    color: #94a3b8 !important;
    font-size: 0.82rem;
    text-align: center;
    margin-top: 1rem;
}

/* Dataframe styling */
[data-testid="stDataFrame"] {
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Style Presets ───────────────────────────────────────────────────────────
STYLE_PRESETS = {
    "🎬 Điện ảnh thực tế (Cinematic Realism)":
        "Cinematic 35mm film still, photorealistic, natural volumetric lighting, shallow depth of field, 8k resolution, muted natural color tones --ar 16:9",
    "🍵 Triết lý Nhật Bản & Zen (Ikigai, Kaizen, Wabi-Sabi)":
        "Minimalist Japanese Zen aesthetic, wabi-sabi philosophy, soft natural morning sunlight, peaceful traditional Japanese interior or serene nature, bamboo, stone garden, warm wood textures, mindful atmosphere, subtle depth of field, high-end photography --ar 16:9",
    "🌿 Hoài niệm mộc mạc (Studio Ghibli / Peaceful Watercolor)":
        "Poetic Studio Ghibli inspired watercolor animation style, lush serene landscape, warm nostalgic sunlight, gentle breeze, calming peaceful aesthetic, masterwork hand-drawn art, detailed foliage --ar 16:9",
    "🏛️ Khắc kỷ & Chiêm nghiệm (Stoic / Moody Cinematic Drama)":
        "Moody dramatic cinematic aesthetic, Stoic philosopher atmosphere, deep chiaroscuro shadows, antique stone statues, warm candlelight contrast, film grain, contemplative and profound, 8k resolution --ar 16:9",
    "🎨 Hoạt hình 2D (Anime / Shinkai Style)":
        "High quality 2D digital animation illustration, Makoto Shinkai aesthetic, vibrant color palette, beautiful atmospheric lighting, detailed background --ar 16:9",
    "🧸 Hoạt hình 3D (Pixar / Disney 3D)":
        "Cute 3D character animation render, Pixar and Disney style, soft subsurface scattering, vibrant studio lighting, highly detailed textures, expressive --ar 16:9",
    "📰 Phim tài liệu lịch sử (Vintage / Documentary)":
        "Vintage documentary photograph, authentic film grain, 1970s color grading, natural direct flash lighting, archival photo aesthetic --ar 16:9",
    "🖌️ Tranh sơn dầu nghệ thuật (Oil Painting)":
        "Expressive oil painting style, visible textured brushstrokes, dramatic chiaroscuro lighting, rich color palette, fine art masterpiece --ar 16:9",
    "✏️ Hoạt hình Stickman / Webcomic (OverSimplified Style)":
        "Simple 2D digital webcomic illustration, minimalist stick figure character with funny expressive face, clean bold black outlines, flat colorful background, Sam O'Nella and OverSimplified YouTube animation aesthetic, humor comic art --ar 16:9",
    "⚙️ Tùy chỉnh tự nhập (Custom)": "",
}

PACING_OPTIONS = {
    "⚡ Nhanh / Kịch tính (Shorts / TikTok / Reels: ~3–6s)":
        "Group segments into dynamic, short shots of approximately 3–6 seconds each.",
    "📖 Tiêu chuẩn / Kể chuyện (Kịch bản thông thường: ~6–10s)":
        "Group segments into storytelling shots of approximately 6–10 seconds each, matching complete sentences or coherent thoughts.",
    "🧘 Trầm lắng & Chiêm nghiệm (Triết lý, Ikigai, Kaizen, Stoic: ~10–18s)":
        "Group segments into contemplative, atmospheric scenes of approximately 10–18 seconds each. Prioritize deep visual moments, allowing space for thought, stillness and philosophical reflection before cutting.",
    "🍃 Chậm rãi & Thiền định (Podcast, Deep Zen: ~18–30s)":
        "Group segments into expansive, meditative shots of approximately 18–30 seconds each, capturing overarching themes and profound philosophical ideas with long lingering visuals.",
}

WHISPER_SIZES = ["base", "small", "medium"]

DEFAULT_CHARACTER_ANCHOR = (
    "[Main Character]: 38yo Asian male doctor in white medical coat with stethoscope"
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Cấu hình Hệ thống")
    st.divider()

    api_key = st.text_input(
        "🔑 Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Lấy tại https://aistudio.google.com/app/apikey",
    )

    st.divider()

    preset_label = st.selectbox(
        "🎨 Style Preset (Mẫu phong cách)",
        options=list(STYLE_PRESETS.keys()),
        index=0,
    )

    # Auto-fill style anchor from preset; user can still edit
    default_anchor = STYLE_PRESETS[preset_label]
    style_anchor = st.text_area(
        "✏️ Style Anchor",
        value=default_anchor,
        height=120,
        help="Tự động điền theo preset. Bạn có thể chỉnh sửa thêm.",
    )

    character_anchor = st.text_input(
        "🧑‍⚕️ Character Anchor",
        value=DEFAULT_CHARACTER_ANCHOR,
        help="Định nghĩa nhân vật cố định cho tất cả các shot (hoặc để trống nếu là video phong cảnh/thiên nhiên).",
    )

    pacing_label = st.selectbox(
        "⏱️ Nhịp độ phân cảnh (Pacing)",
        options=list(PACING_OPTIONS.keys()),
        index=0,
        help="Chọn thời lượng ước tính cho mỗi shot cảnh. Nhanh (3–6s) cho Shorts/Reels, hoặc Trầm lắng (10–18s) cho triết lý Ikigai/Kaizen.",
    )
    pacing_rule = PACING_OPTIONS[pacing_label]

    st.divider()

    whisper_size = st.selectbox(
        "🎙️ Whisper Model Size",
        options=WHISPER_SIZES,
        index=0,
        help="'base' nhanh nhất; 'medium' chính xác hơn nhưng chậm hơn.",
    )

    gemini_model_choice = st.selectbox(
        "🤖 Gemini Model",
        options=[
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3.1-pro",
            "gemini-3.0-flash",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
        ],
        index=0,
        help="Chọn mô hình Gemini. Ứng dụng hỗ trợ Gemini 3.6 Flash, 3.5 Flash, 3.1 Pro, 2.5 Flash, 1.5 Flash...",
    )

    st.divider()
    st.markdown('<div class="sidebar-footer">Made with ❤️ · Powered by Whisper + Gemini</div>', unsafe_allow_html=True)

# ─── Main Screen ─────────────────────────────────────────────────────────────
st.markdown('<p class="title-banner">🎬 AI Storyboard Generator</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Upload file âm thanh ➔ Whisper bóc tách lời thoại ➔ Gemini phân cảnh & tạo Image Prompts chuẩn nét.</p>',
    unsafe_allow_html=True,
)

with st.container():
    uploaded_file = st.file_uploader(
        "📂 Upload file âm thanh (.mp3, .wav, .m4a)",
        type=["mp3", "wav", "m4a"],
        help="Hỗ trợ file định dạng MP3, WAV, M4A",
    )

    if uploaded_file:
        st.audio(uploaded_file, format=uploaded_file.type)

st.markdown("<br>", unsafe_allow_html=True)

run_btn = st.button(
    "🚀 Bắt đầu phân cảnh & Tạo Image Prompts",
    use_container_width=True,
    disabled=(not uploaded_file),
)

# ─── Helper: Whisper Transcription ───────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_whisper(model_size: str):
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe_audio(tmp_path: str, model_size: str) -> list[dict]:
    """Return list of {start, end, text} segments."""
    model = load_whisper(model_size)
    segments, _ = model.transcribe(tmp_path, beam_size=5)
    results = []
    for seg in segments:
        results.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
        })
    return results


# ─── Helper: Build Gemini Prompt & Chunking ──────────────────────────────────

def seconds_to_mmss(total_seconds: float) -> str:
    """Convert a float number of seconds to 'MM:SS' string, e.g. 334.4 -> '05:34'."""
    total_seconds = max(0.0, total_seconds)
    minutes = int(total_seconds) // 60
    seconds = int(total_seconds) % 60
    return f"{minutes:02d}:{seconds:02d}"


def normalize_timestamp(ts: str) -> str:
    """
    Repair malformed timestamp strings that Gemini occasionally returns.
    Handles these broken patterns:
      - '06:92 – 11:84'  (centisecond digits leaked into SS position)
      - '334:4 – 337.0'  (total-seconds:fraction instead of MM:SS)
      - '337.0'          (dot separator instead of colon)
    Normalizes everything to 'MM:SS – MM:SS'.
    """
    import re

    def parse_token(token: str) -> float:
        """Parse a single time token to total float seconds."""
        token = token.strip().replace(',', '.')
        # Pattern: digits (colon or dot) digits, e.g. '06:92', '334:4', '5.37'
        m = re.match(r'^(\d+)[:.](\d+)$', token)
        if m:
            left, right = int(m.group(1)), m.group(2)
            right_int = int(right)
            # Decide interpretation:
            # If left >= 60 → left is total seconds, right is fractional part
            # (e.g. '334:4' means 334.4 s, '337.0' means 337.0 s)
            if int(m.group(1)) >= 60:
                return float(f"{m.group(1)}.{right}")
            # If right part > 59 → it's centiseconds, not clock-seconds
            # (e.g. '06:92' → 6 + 0.92 = 6.92 s)
            elif right_int > 59:
                return int(m.group(1)) + right_int / 100.0
            else:
                # Normal MM:SS → convert to total seconds
                return int(m.group(1)) * 60 + right_int
        # Plain number (no separator)
        try:
            return float(token)
        except ValueError:
            return 0.0

    # Split on the dash separator (–, --, or -)
    parts = re.split(r'\s*[–—-]+\s*', ts.strip())
    if len(parts) == 2:
        start_s = parse_token(parts[0])
        end_s = parse_token(parts[1])
        return f"{seconds_to_mmss(start_s)} – {seconds_to_mmss(end_s)}"

    # Single token fallback
    if len(parts) == 1:
        s = parse_token(parts[0])
        return seconds_to_mmss(s)

    return ts  # return as-is if unparseable


def build_system_instruction(pacing_rule: str) -> str:
    return f"""You are a professional storyboard director and AI image prompt engineer.
Your task: Given a transcript (list of speech segments with timestamps), produce a storyboard JSON array.

RULES:
1. {pacing_rule}
2. Each shot MUST have these exact keys: shot_id, timestamp, voiceover, shot_type, visual_description, ai_image_prompt.
3. shot_type: one of [Extreme Wide Shot, Wide Shot, Medium Shot, Close-Up, Extreme Close-Up, Over-the-Shoulder].
4. visual_description: brief Vietnamese description of what the viewer sees.
5. ai_image_prompt: English-only. MUST follow formula: [Shot Type] + [Subject & Action] + [Environment] + [Lighting & Style Anchor]. Embed the provided Style Anchor and Character Anchor verbatim in EVERY prompt.
6. timestamp format MUST be exactly "MM:SS – MM:SS" where MM=minutes (00-99) and SS=seconds (00-59). NEVER use centiseconds or decimals. Example: "01:23 – 01:30" is correct, "01:92" is WRONG.
7. Return ONLY a valid JSON array — no markdown fences, no extra text, no comments.
"""


def chunk_segments(segments: list[dict], max_duration_sec: float = 360.0, max_segments: int = 90) -> list[list[dict]]:
    """Chunk transcript segments into larger batches (~6 minutes or ~90 segments per batch) for 8x speedup."""
    chunks = []
    current_chunk = []
    chunk_start_time = None

    for seg in segments:
        if not current_chunk:
            chunk_start_time = seg["start"]
            current_chunk.append(seg)
        else:
            duration = seg["end"] - chunk_start_time
            if duration >= max_duration_sec or len(current_chunk) >= max_segments:
                chunks.append(current_chunk)
                current_chunk = [seg]
                chunk_start_time = seg["start"]
            else:
                current_chunk.append(seg)

    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def build_user_prompt(segments: list[dict], style_anchor: str, character_anchor: str) -> str:
    # Convert raw float seconds to MM:SS so Gemini receives clean clock-format inputs
    segments_text = "\n".join(
        f"[{seconds_to_mmss(s['start'])} – {seconds_to_mmss(s['end'])}]: {s['text']}" for s in segments
    )
    return f"""Style Anchor: {style_anchor}
Character Anchor: {character_anchor}

Transcript segments (timestamps are in MM:SS format, where MM=minutes 00-99 and SS=seconds 00-59):
{segments_text}

Produce the storyboard JSON array now."""


# ─── Helper: Gemini Call with Parallel Batching & Fast Retries ───────────────

def get_verified_candidates(api_key: str, preferred_model: str) -> list[str]:
    """Fetch active Gemini models from API and sanitize candidates list."""
    genai.configure(api_key=api_key)
    candidates = [preferred_model]
    default_models = [
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.1-pro",
        "gemini-3.0-flash",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]

    try:
        api_models = [
            m.name.replace("models/", "")
            for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]
        for am in api_models:
            if am not in candidates:
                candidates.append(am)
    except Exception:
        pass

    for dm in default_models:
        if dm not in candidates:
            candidates.append(dm)

    return candidates


def process_single_chunk(
    chunk_idx: int,
    chunk: list[dict],
    style_anchor: str,
    character_anchor: str,
    candidates: list[str],
    system_instruction: str,
) -> tuple[int, list[dict]]:
    """Process a single transcript chunk with ultra-fast retries."""
    user_prompt = build_user_prompt(chunk, style_anchor, character_anchor)
    last_error = None

    for model_name in candidates:
        for attempt in range(2):
            try:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_instruction,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        temperature=0.7,
                        max_output_tokens=8192,
                    ),
                )
                response = model.generate_content(user_prompt)
                raw = response.text.strip()

                if raw.startswith("```"):
                    lines = raw.splitlines()
                    raw = "\n".join(
                        ln for ln in lines if not ln.startswith("```")
                    ).strip()

                data = json.loads(raw)
                if isinstance(data, list) and len(data) > 0:
                    # Normalize any malformed timestamps Gemini returned
                    for shot in data:
                        if "timestamp" in shot and shot["timestamp"]:
                            shot["timestamp"] = normalize_timestamp(str(shot["timestamp"]))
                    return chunk_idx, data
            except Exception as e:
                last_error = e
                err_msg = str(e)

                if "404" in err_msg or "not found" in err_msg.lower():
                    break

                time.sleep(1.0)

    raise Exception(f"Lỗi xử lý phần {chunk_idx + 1}: {last_error}")


def generate_storyboard(
    segments: list[dict],
    style_anchor: str,
    character_anchor: str,
    api_key: str,
    preferred_model: str = "gemini-3.6-flash",
    pacing_rule: str = "Group segments into shots of approximately 3–6 seconds each.",
    progress_bar=None,
    status_placeholder=None,
) -> list[dict]:
    candidates = get_verified_candidates(api_key, preferred_model)
    chunks = chunk_segments(segments, max_duration_sec=360.0, max_segments=90)
    num_chunks = len(chunks)
    system_instruction = build_system_instruction(pacing_rule)

    results = {}
    completed_count = 0

    if status_placeholder:
        status_placeholder.markdown(
            f"⚡ **Bước 2/2** — Đang phân tích song song **{num_chunks}** phần transcript qua Gemini…"
        )

    # Use max_workers=3 to balance speed and prevent hitting API rate limits
    max_workers = min(num_chunks, 3)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                process_single_chunk,
                idx,
                chunk,
                style_anchor,
                character_anchor,
                candidates,
                system_instruction,
            ): idx
            for idx, chunk in enumerate(chunks)
        }

        for future in as_completed(future_to_idx):
            chunk_idx, chunk_shots = future.result()
            results[chunk_idx] = chunk_shots
            completed_count += 1
            percent = completed_count / num_chunks
            if progress_bar:
                progress_bar.progress(percent)
            if status_placeholder:
                status_placeholder.markdown(
                    f"⚡ **Bước 2/2** — Hoàn thành **{completed_count}/{num_chunks}** phần ({int(percent * 100)}%)…"
                )

    # Combine results in chronological order
    all_shots = []
    for idx in range(num_chunks):
        all_shots.extend(results[idx])

    # Re-index shot_id sequentially from 1 to len(all_shots)
    for i, shot in enumerate(all_shots, 1):
        shot["shot_id"] = i

    return all_shots




# ─── Helper: Export ──────────────────────────────────────────────────────────

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def df_to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Storyboard")
    return buf.getvalue()


# ─── Processing Pipeline ─────────────────────────────────────────────────────

if run_btn and uploaded_file:
    # Validate inputs
    if not api_key:
        st.error("⚠️ Vui lòng nhập Gemini API Key ở sidebar trước.")
        st.stop()

    tmp_path = None
    try:
        # Save upload to temp file
        suffix = os.path.splitext(uploaded_file.name)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # ── Step 1: Whisper ──────────────────────────────────────────────
        with st.spinner(f"🎙️ Bước 1/2 — Whisper ({whisper_size}) đang bóc tách lời thoại…"):
            segments = transcribe_audio(tmp_path, whisper_size)

        if not segments:
            st.warning("⚠️ Whisper không tìm thấy lời thoại trong file. Hãy thử file khác.")
            st.stop()

        total_audio_mins = round(segments[-1]["end"] / 60, 1)
        st.success(f"✅ Bước 1 hoàn tất — bóc tách **{len(segments)}** đoạn lời thoại (độ dài: ~{total_audio_mins} phút).")

        # ── Step 2: Gemini Storyboard with Parallel Batching ──────────────
        status_placeholder = st.empty()
        progress_bar = st.progress(0.0)

        storyboard = generate_storyboard(
            segments,
            style_anchor,
            character_anchor,
            api_key,
            preferred_model=gemini_model_choice,
            pacing_rule=pacing_rule,
            progress_bar=progress_bar,
            status_placeholder=status_placeholder,
        )

        progress_bar.empty()
        status_placeholder.empty()

        st.success(f"✅ Bước 2 hoàn tất — tạo thành công toàn bộ **{len(storyboard)}** shot storyboard từ 0:00 đến {total_audio_mins} phút!")

        # ── Build DataFrame ──────────────────────────────────────────────
        COLUMNS = [
            "shot_id", "timestamp", "voiceover",
            "shot_type", "visual_description", "ai_image_prompt",
        ]
        df = pd.DataFrame(storyboard)

        # Ensure all columns exist (fill missing with "")
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = ""
        df = df[COLUMNS]

        st.session_state["storyboard_df"] = df

    except json.JSONDecodeError as e:
        st.error(f"❌ Lỗi parse JSON từ Gemini: {e}\n\nKiểm tra API Key hoặc thử lại.")
    except Exception as e:
        st.error(f"❌ Lỗi: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

# ─── Results Section ─────────────────────────────────────────────────────────

if "storyboard_df" in st.session_state:
    df: pd.DataFrame = st.session_state["storyboard_df"]

    st.markdown("---")
    st.markdown("### 📋 Bảng Storyboard — Xem & Chỉnh sửa trực tiếp")

    edited_df = st.data_editor(
        df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "shot_id": st.column_config.NumberColumn("Shot ID", width="small"),
            "timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
            "voiceover": st.column_config.TextColumn("Lời thoại", width="large"),
            "shot_type": st.column_config.SelectboxColumn(
                "Shot Type",
                options=[
                    "Extreme Wide Shot", "Wide Shot", "Medium Shot",
                    "Close-Up", "Extreme Close-Up", "Over-the-Shoulder",
                ],
                width="medium",
            ),
            "visual_description": st.column_config.TextColumn("Mô tả hình ảnh (VI)", width="large"),
            "ai_image_prompt": st.column_config.TextColumn("AI Image Prompt (EN)", width="extra_large"),
        },
        hide_index=True,
    )

    # Update session state with edits
    st.session_state["storyboard_df"] = edited_df

    st.markdown("### 💾 Tải về & Xuất Prompts")

    tab1, tab2 = st.tabs(["📊 Bảng Storyboard Chi Tiết", "🖼️ Danh Sách Prompts Copy Nhanh (Google Flow / Midjourney / Leonardo)"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="⬇️ Tải CSV (UTF-8)",
                data=df_to_csv_bytes(edited_df),
                file_name="storyboard.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col2:
            st.download_button(
                label="⬇️ Tải Excel (.xlsx)",
                data=df_to_xlsx_bytes(edited_df),
                file_name="storyboard.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    with tab2:
        st.info("💡 Bạn có thể copy toàn bộ Prompts theo định dạng chuẩn bên dưới để dán trực tiếp vào **Google Flow, Midjourney, Leonardo AI**:")
        
        # Build prompt list with exact format: [timestamp] prompt separated by 1 blank line
        formatted_prompts = []
        for _, row in edited_df.iterrows():
            ts = str(row['timestamp']).strip()
            prompt = str(row['ai_image_prompt']).strip()
            formatted_prompts.append(f"[{ts}] {prompt}")
        
        full_formatted_text = "\n\n".join(formatted_prompts)

        st.code(full_formatted_text, language="markdown")

        st.download_button(
            label="📋 Tải file Prompts TXT",
            data=full_formatted_text.encode("utf-8"),
            file_name="prompts_google_flow.txt",
            mime="text/plain",
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("#### 🎬 Chi tiết từng Shot:")
        for idx, row in edited_df.iterrows():
            with st.expander(f"🎬 Shot {row['shot_id']} [{row['timestamp']}] - {row['shot_type']}"):
                st.write(f"**Lời thoại:** {row['voiceover']}")
                st.write(f"**Mô tả cảnh (VI):** {row['visual_description']}")
                st.code(f"[{row['timestamp']}] {row['ai_image_prompt']}", language="markdown")

    st.caption(f"📊 Tổng cộng {len(edited_df)} shots · Mã hóa UTF-8 an toàn cho Excel tiếng Việt & các công cụ AI Image.")
