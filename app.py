import os
import subprocess
import tempfile
import asyncio
import shutil

import streamlit as st
import imageio_ffmpeg
import edge_tts
from faster_whisper import WhisperModel
import ctranslate2
import sentencepiece as spm
from huggingface_hub import snapshot_download


# =========================================================
# APP
# =========================================================

st.set_page_config(
    page_title="Smey Auto Caption",
    page_icon="🇰🇭",
)

LOGO_FILE = "file_00000000568c8211831d7859c11ddf61.png"

if os.path.exists(LOGO_FILE):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(LOGO_FILE, width=300)

st.title("🇰🇭 Smey Auto Caption")
st.write("Local AI → Accurate Caption → Khmer Translation → Khmer Dubbing → MP4")


# =========================================================
# LOCAL MODELS
# =========================================================

ASR_MODEL = "large-v3-turbo"
TRANSLATION_MODEL = "solavr/small100-ctranslate2-int8"

LANGUAGE_CODES = {
    "🇰🇭 ខ្មែរ": {"whisper": "km", "translate": "km"},
    "🇬🇧 English": {"whisper": "en", "translate": "en"},
    "🇨🇳 中文": {"whisper": "zh", "translate": "zh"},
    "🇻🇳 Tiếng Việt": {"whisper": "vi", "translate": "vi"},
    "🇰🇷 한국어": {"whisper": "ko", "translate": "ko"},
    "🇯🇵 日本語": {"whisper": "ja", "translate": "ja"},
}

TARGET_CODES = {
    "🇰🇭 ខ្មែរ": "km",
    "🇬🇧 English": "en",
    "🇨🇳 中文": "zh",
    "🇻🇳 Tiếng Việt": "vi",
    "🇰🇷 한국어": "ko",
    "🇯🇵 日本語": "ja",
}


@st.cache_resource(show_spinner=False)
def load_asr():
    # CPU + INT8 for Streamlit Community Cloud.
    return WhisperModel(
        ASR_MODEL,
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
        num_workers=1,
    )


@st.cache_resource(show_spinner=False)
def load_translator():
    # SMaLL-100 INT8 is a compact CPU translation model.
    model_dir = snapshot_download(TRANSLATION_MODEL)
    tokenizer = spm.SentencePieceProcessor(
        model_file=os.path.join(model_dir, "sentencepiece.bpe.model")
    )
    translator = ctranslate2.Translator(
        model_dir,
        device="cpu",
        compute_type="int8_float32",
        inter_threads=1,
        intra_threads=4,
    )
    return tokenizer, translator


# =========================================================
# FFmpeg
# =========================================================

def run_ffmpeg(args):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg] + args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def extract_audio(video_path, audio_path):
    run_ffmpeg([
        "-y", "-i", video_path, "-vn",
        "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", audio_path,
    ])


# =========================================================
# LOCAL ASR
# =========================================================

def transcribe_local(audio_path, source_language):
    model = load_asr()

    whisper_language = None
    if source_language != "Auto Detect":
        whisper_language = LANGUAGE_CODES[source_language]["whisper"]

    segments, info = model.transcribe(
        audio_path,
        language=whisper_language,
        task="transcribe",
        beam_size=3,
        best_of=1,
        temperature=0,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=350),
        condition_on_previous_text=True,
        initial_prompt=(
            "Chinese dialogue from a short drama. "
            "Transcribe the spoken words exactly; do not translate."
            if source_language == "🇨🇳 中文"
            else None
        ),
    )

    groups = []

    for segment in segments:
        text = str(segment.text).strip()
        if not text:
            continue

        start = max(0.0, float(segment.start))
        end = max(start + 0.2, float(segment.end))

        groups.append({
            "start": start,
            "end": end,
            "text": text,
        })

    detected = getattr(info, "language", None)
    return groups, detected


# =========================================================
# LOCAL TRANSLATION — NO GEMINI
# =========================================================

def translate_local(groups, source_code, target_code):
    if not groups or source_code == target_code:
        return groups

    tokenizer, translator = load_translator()
    texts = [str(g["text"]).strip() for g in groups]

    source_tokens = [
        [f"__{target_code}__"] + tokenizer.encode(text, out_type=str) + ["</s>"]
        for text in texts
    ]

    results = translator.translate_batch(
        source_tokens,
        beam_size=3,
        max_decoding_length=128,
    )

    translated = []
    for result in results:
        tokens = list(result.hypotheses[0])
        if tokens and tokens[0] == f"__{target_code}__":
            tokens = tokens[1:]
        translated.append(tokenizer.decode(tokens).strip())

    return [
        {
            "start": group["start"],
            "end": group["end"],
            "text": translated_text or group["text"],
        }
        for group, translated_text in zip(groups, translated)
    ]


# =========================================================
# KHMER DUBBING — EDGE TTS, NO GEMINI
# =========================================================

async def _tts_one(text, output_path, voice, rate):
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
    )
    await communicate.save(output_path)


def make_khmer_dubbing(
    video_path,
    groups,
    output_audio,
    voice="km-KH-PisethNeural",
    rate="+15%",
    original_volume=0.22,
):
    if not groups:
        raise RuntimeError("❌ មិនមាន Caption សម្រាប់ Dubbing ទេ។")

    temp_dir = tempfile.mkdtemp(prefix="smey_khmer_dub_")

    try:
        jobs = []
        clips = []

        for i, group in enumerate(groups):
            text = str(group.get("text", "")).strip()
            if not text:
                continue

            clip = os.path.join(temp_dir, f"voice_{i:04d}.mp3")
            clips.append((clip, float(group["start"])))
            jobs.append(_tts_one(text, clip, voice, rate))

        async def run_all():
            await asyncio.gather(*jobs)

        asyncio.run(run_all())

        inputs = ["-i", video_path]

        for clip, _start in clips:
            inputs += ["-i", clip]

        filters = [f"[0:a]volume={original_volume}[orig]"]
        mix = ["[orig]"]

        for i, (_clip, start) in enumerate(clips, start=1):
            delay = max(0, int(start * 1000))
            label = f"[v{i}]"
            filters.append(f"[{i}:a]adelay={delay}|{delay}{label}")
            mix.append(label)

        filters.append(
            "".join(mix)
            + f"amix=inputs={len(mix)}:duration=first:dropout_transition=0:normalize=0[aout]"
        )

        run_ffmpeg(
            inputs + [
                "-filter_complex", ";".join(filters),
                "-map", "[aout]",
                "-c:a", "aac",
                "-b:a", "160k",
                "-y", output_audio,
            ]
        )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# =========================================================
# ASS TIME
# =========================================================

def ass_time(seconds):
    seconds = max(0.0, float(seconds))

    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))

    if cs >= 100:
        cs = 0
        s += 1
    if s >= 60:
        s = 0
        m += 1
    if m >= 60:
        m = 0
        h += 1

    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


# =========================================================
# ASS SUBTITLE
# =========================================================

def create_ass(groups, filename):
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Khmer,Noto Sans Khmer,70,&H00CC66FF,&H00FFFFFF,&H00FFFFFF,&H99000000,1,0,0,0,100,100,0,0,3,3,1,2,50,50,150,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(header)

        for group in groups:
            text = str(group["text"]).replace("\n", " ")
            text = text.replace("{", r"\{").replace("}", r"\}")

            f.write(
                f"Dialogue: 0,"
                f"{ass_time(group['start'])},"
                f"{ass_time(group['end'])},"
                f"Khmer,,0,0,0,,{text}\n"
            )


# =========================================================
# BURN CAPTION
# =========================================================

def burn_caption(video_path, ass_path, output_path):
    escaped_ass = (
        ass_path
        .replace("\\", "/")
        .replace(":", r"\:")
        .replace("'", r"\'")
    )

    run_ffmpeg([
        "-y",
        "-i", video_path,
        "-vf", f"ass='{escaped_ass}'",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "160k",
        "-movflags", "+faststart",
        output_path,
    ])


# =========================================================
# LANGUAGE SETTINGS
# =========================================================

source_language = st.selectbox(
    "🌐 ភាសាដើម",
    [
        "Auto Detect",
        "🇰🇭 ខ្មែរ",
        "🇬🇧 English",
        "🇨🇳 中文",
        "🇻🇳 Tiếng Việt",
        "🇰🇷 한국어",
        "🇯🇵 日本語",
    ],
)

target_language = st.selectbox(
    "🎯 បកប្រែទៅជា",
    [
        "មិនបកប្រែ",
        "🇰🇭 ខ្មែរ",
        "🇬🇧 English",
        "🇨🇳 中文",
        "🇻🇳 Tiếng Việt",
        "🇰🇷 한국어",
        "🇯🇵 日本語",
    ],
)


# =========================================================
# KHMER DUBBING OPTIONS
# =========================================================

st.divider()
st.subheader("🗣️ Khmer Dubbing")

khmer_dubbing = st.checkbox(
    "🗣️ និយាយខ្មែរ + រក្សា Background Music",
    value=False,
)

khmer_voice = st.selectbox(
    "🎙️ សំឡេងខ្មែរ",
    [
        "km-KH-PisethNeural",
        "km-KH-SreymomNeural",
    ],
    disabled=not khmer_dubbing,
)

khmer_rate = st.select_slider(
    "⚡ ល្បឿនសំឡេង",
    options=["+0%", "+10%", "+15%", "+20%", "+25%"],
    value="+15%",
    disabled=not khmer_dubbing,
)

original_audio_volume = 0.05

if khmer_dubbing:
    st.caption(
        "សំឡេងខ្មែរចាប់ផ្ដើមតាម Caption timestamp។ "
        "សំឡេងដើមត្រូវបានបន្ថយខ្លាំង ដើម្បីឲ្យសំឡេងខ្មែរលេចច្បាស់។"
    )
    st.info(
        "ℹ️ Version នេះមិនអាចបំបែក Voice និង Background Music ដាច់ពីគ្នាបានល្អ "
        "ដោយ FFmpeg តែមួយទេ។ ដូច្នេះ Original audio ត្រូវបានបន្ថយមក 5% "
        "ដើម្បីកុំឲ្យសំឡេងមនុស្សដើមរំខាន។"
    )
    st.warning(
        "⚠️ Version នេះធ្វើ Audio/Timing Sync ប៉ុណ្ណោះ។ "
        "វាមិនធ្វើឲ្យមាត់តួអង្គផ្លាស់ទីតាមសំឡេងទេ។"
    )

st.info(
    "🆓 Local AI Mode: Caption + Translation មិនប្រើ Gemini API។ "
    "Whisper large-v3-turbo + Local Translator + Edge TTS ដំណើរការដោយមិនដក Gemini quota។"
)


# =========================================================
# VIDEO UPLOAD
# =========================================================

video = st.file_uploader(
    "🎥 ជ្រើសវីដេអូ",
    type=["mp4", "mov", "mkv", "webm"],
)


# =========================================================
# PROCESS
# =========================================================

if video is not None:

    caption_clicked = st.button(
        "⚡ បង្កើត Caption",
        use_container_width=True,
    )

    if caption_clicked:

        with tempfile.TemporaryDirectory() as temp_dir:

            video_path = os.path.join(temp_dir, "input.mp4")
            audio_path = os.path.join(temp_dir, "audio.wav")
            ass_path = os.path.join(temp_dir, "khmer_caption.ass")
            output_path = os.path.join(temp_dir, "smey_auto_caption.mp4")

            with open(video_path, "wb") as f:
                f.write(video.getbuffer())

            try:

                with st.spinner("⚡ កំពុងដកសំឡេង..."):
                    extract_audio(video_path, audio_path)

                # =================================================
                # LOCAL SPEECH TO TEXT
                # =================================================

                with st.spinner("🎙️ Local Whisper កំពុងស្តាប់សំឡេង..."):
                    groups, detected_language = transcribe_local(
                        audio_path,
                        source_language,
                    )

                if not groups:
                    raise RuntimeError(
                        "❌ Local Whisper មិនបានរកឃើញ Caption ទេ។"
                    )

                # Determine source language for local translation.
                if source_language == "Auto Detect":
                    source_code = detected_language
                    if source_code not in {"zh", "en", "km", "vi", "ko", "ja"}:
                        raise RuntimeError(
                            f"❌ Auto Detect រកភាសា '{detected_language}' "
                            "ដែលមិនទាន់មានក្នុង Local Translator Map។"
                        )
                else:
                    source_code = LANGUAGE_CODES[source_language]["translate"]

                # =================================================
                # LOCAL TRANSLATION
                # =================================================

                if target_language != "មិនបកប្រែ":
                    with st.spinner("🌐 Local Translator កំពុងបកប្រែ..."):
                        groups = translate_local(
                            groups,
                            source_code,
                            TARGET_CODES[target_language],
                        )

                # =================================================
                # KHMER DUBBING
                # =================================================

                if khmer_dubbing:

                    if target_language != "🇰🇭 ខ្មែរ":
                        raise RuntimeError(
                            "⚠️ Khmer Dubbing ត្រូវជ្រើស "
                            "🎯 បកប្រែទៅជា → 🇰🇭 ខ្មែរ។"
                        )

                    dubbed_audio_path = os.path.join(
                        temp_dir,
                        "khmer_dubbed_audio.m4a",
                    )

                    dubbed_video_path = os.path.join(
                        temp_dir,
                        "video_with_khmer_voice.mp4",
                    )

                    with st.spinner(
                        "🗣️ កំពុងបង្កើតសំឡេងខ្មែរ + Sync តាម Caption..."
                    ):
                        make_khmer_dubbing(
                            video_path,
                            groups,
                            dubbed_audio_path,
                            voice=khmer_voice,
                            rate=khmer_rate,
                            original_volume=original_audio_volume,
                        )

                    run_ffmpeg([
                        "-y",
                        "-i", video_path,
                        "-i", dubbed_audio_path,
                        "-map", "0:v:0",
                        "-map", "1:a:0",
                        "-c:v", "copy",
                        "-c:a", "aac",
                        "-b:a", "160k",
                        "-movflags", "+faststart",
                        dubbed_video_path,
                    ])

                    video_path = dubbed_video_path

                # =================================================
                # CREATE ASS
                # =================================================

                create_ass(groups, ass_path)

                # =================================================
                # BURN CAPTION
                # =================================================

                with st.spinner(
                    "🎬 កំពុងដាក់ Caption ជាប់ក្នុងវីដេអូ..."
                ):
                    burn_caption(
                        video_path,
                        ass_path,
                        output_path,
                    )

                with open(output_path, "rb") as f:
                    output_data = f.read()

                st.success("✅ វីដេអូរួចរាល់!")
                st.video(output_data)

                st.download_button(
                    "⬇️ ទាញយកវីដេអូ MP4",
                    data=output_data,
                    file_name="smey_auto_caption.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                    on_click="ignore",
                )

            except Exception as e:
                st.error(f"❌ មានបញ្ហាពេលបង្កើតវីដេអូ: {e}")
