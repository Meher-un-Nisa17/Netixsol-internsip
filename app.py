import html
import hashlib
import os
import uuid

import requests
import streamlit as st


st.set_page_config(
    page_title="Uzma | Property Concierge",
    page_icon="🏡",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Noto+Naskh+Arabic:wght@400;500;600;700&display=swap');
    :root { --ink:#152c2b; --muted:#617371; --teal:#16786f; --paper:#f5f7f3; }
    .stApp { background: radial-gradient(ellipse at 50% -15%, #d9ece5 0, transparent 42%), var(--paper); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background:#eef3ee; border-right:1px solid #dce6df; }
    .block-container { max-width: 820px; padding-top: 2.2rem; padding-bottom: 3rem; }
    .hero { padding: 1.4rem 1.6rem 1.1rem; border-radius: 24px; background:linear-gradient(120deg,#123e3b,#1d7168); color:white; box-shadow:0 18px 45px #193e3a20; margin-bottom:1.3rem; }
    .hero-kicker { color:#b8ded3; font:600 .78rem 'DM Sans',sans-serif; letter-spacing:.12em; text-transform:uppercase; }
    .hero h1 { margin:.35rem 0 .25rem; font:700 2rem 'DM Sans',sans-serif; color:white; }
    .hero p { color:#e0efea; margin:0; font:400 1rem 'Noto Naskh Arabic',serif; }
    .section-note { color:var(--muted); font:500 .91rem 'DM Sans',sans-serif; margin:.2rem 0 1rem; }
    [data-testid="stChatMessage"] { border:1px solid #e3eae3; border-radius:18px; background:#ffffffcc; color:#152c2b !important; padding:.85rem 1rem; box-shadow:0 4px 16px #183b3510; }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li { color:#152c2b !important; direction:rtl; text-align:right; font-family:'Noto Naskh Arabic','DM Sans',sans-serif; font-size:1.08rem; line-height:1.85; }
    [data-testid="stChatInput"] { border-radius:16px; }
    .stButton button { border-radius:12px; }
    .stAudioInput { border-radius:16px; }
    .stCaption { color:var(--muted); }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 🏡 Uzma")
    st.caption("AI Property Concierge")
    backend_url = st.text_input(
        "FastAPI Backend URL",
        value=os.getenv("BACKEND_URL", "http://localhost:8000"),
    ).strip().rstrip("/")
    if st.button("Check connection", use_container_width=True):
        try:
            health = requests.get(f"{backend_url}/health", timeout=3)
            if health.ok:
                st.success("Backend connected")
            else:
                st.error(f"Backend returned {health.status_code}")
        except requests.RequestException:
            st.error("Backend is not reachable. Start FastAPI and try again.")
    st.divider()
    st.markdown("**Talk naturally**")
    st.caption("Use the microphone or type a message. Your conversation stays in this session.")

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Real estate • Lahore, Karachi & Islamabad</div>
      <h1>Meet Uzma</h1>
      <p dir="rtl">اپنی پسند کی Property تلاش کریں، سوال پوچھیں یا Visit طے کریں۔</p>
    </div>
    <div class="section-note">Speak or type in Urdu, English, or UrduLish.</div>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "السلام علیکم! میں عظمٰی ہوں۔ آپ کس شہر اور کس قسم کی Property تلاش کر رہے ہیں؟",
            "audio": None,
        }
    ]
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "last_audio_digest" not in st.session_state:
    st.session_state.last_audio_digest = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("audio"):
            st.audio(message["audio"], format="audio/wav")

audio_value = st.audio_input("🎙️ بولنے کے لیے ریکارڈ کریں")
prompt = st.chat_input("اپنا پیغام لکھیں… / Type a message…")

audio_digest = hashlib.sha256(audio_value.getvalue()).hexdigest() if audio_value is not None else None
new_audio = audio_value is not None and audio_digest != st.session_state.last_audio_digest

if prompt or new_audio:
    input_type = "text" if prompt else "audio"
    user_input = prompt if input_type == "text" else audio_value
    if input_type == "audio":
        st.session_state.last_audio_digest = audio_digest
    display_text = user_input if input_type == "text" else "🎙️ Voice message"
    st.session_state.messages.append({"role": "user", "content": display_text, "audio": None})

    with st.chat_message("user"):
        st.markdown(display_text)
        if input_type == "audio":
            st.audio(audio_value)

    with st.chat_message("assistant"):
        with st.spinner("Uzma آپ کی بات سن رہی ہیں…"):
            try:
                if input_type == "text":
                    response = requests.post(
                        f"{backend_url}/chat",
                        json={"message": user_input, "session_id": st.session_state.session_id},
                        timeout=60,
                    )
                else:
                    response = requests.post(
                        f"{backend_url}/voice-chat",
                        files={"file": ("voice_input.wav", audio_value.getvalue(), "audio/wav")},
                        data={"session_id": st.session_state.session_id},
                        timeout=90,
                    )
                response.raise_for_status()
                payload = response.json().get("data", {})
                bot_reply = payload.get("response") or payload.get("message") or "جواب موصول ہوا، مگر متن دستیاب نہیں۔"
                audio_path = payload.get("audio_path")
                transcript = payload.get("transcribed_text")
            except requests.RequestException as exc:
                details = getattr(getattr(exc, "response", None), "text", "")
                bot_reply = "معذرت، Backend سے رابطہ نہیں ہو سکا۔ Sidebar میں connection چیک کریں۔"
                if details:
                    bot_reply += f"\n\n`{html.escape(details[:500])}`"
                audio_path = None
                transcript = None

        if transcript:
            st.caption(f"آپ نے کہا: {transcript}")
        st.markdown(bot_reply)
        if audio_path:
            st.audio(audio_path, format="audio/wav", autoplay=True)

    st.session_state.messages.append({"role": "assistant", "content": bot_reply, "audio": audio_path})
