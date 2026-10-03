import os
import re
import uuid
import asyncio
import subprocess
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
import imageio_ffmpeg
import edge_tts
from faster_whisper import WhisperModel
import ctranslate2
import sentencepiece as spm
from huggingface_hub import snapshot_download


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="🇰🇭 Smey Auto Caption",
    page_icon="🇰🇭",
    layout="wide",
)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

ASR_MODEL = "tiny"
TRANSLATOR_MODEL = "osa911/nllb-200-distilled-600M-ct2-int8"

KHMER_VOICE = "km-KH-SreymomNeural"

LANG_MAP = {
    "Khmer": "khm_Khmr",
    "English": "eng_Latn",
    "Chinese": "zho_Hans",
    "Vietnamese": "vie_Latn",
    "Korean": "kor_Hang",
    "Japanese": "jpn_Jpan",
}

LANG_CODE = {
    "Khmer": "km",
    "English": "en",
    "Chinese": "zh",
    "Vietnamese": "vi",
    "Korean": "ko",
    "Japanese": "ja",
}


# =========================================================
# HELPERS
# =========================================================

def run_cmd(cmd):
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def audio_duration(path):
    result = subprocess.run(
        [
            FFMPEG,
            "-i",
            path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    text = result.stderr

    m = re.search(
        r"Duration:\s*(\d+):(\d+):([\d.]+)",
        text,
    )

    if not m:
        return 0.0

    h = int(m.group(1))
    mnt = int(m.group(2))
    sec = float(m.group(3))

    return h * 3600 + mnt * 60 + sec


def format_ass_time(seconds):
    seconds = max(0, float(seconds))

    h = int(seconds // 3600)
    seconds -= h * 3600

    m = int(seconds // 60)
    seconds -= m * 60

    s = int(seconds)
    cs = int(round((seconds - s) * 100))

    if cs >= 100:
        s += 1
        cs = 0

    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def safe_text(text):
    return (
        str(text)
        .replace("\\", " ")
        .replace("{", "")
        .replace("}", "")
        .replace("\n", " ")
        .strip()
    )


# =========================================================
# LOAD WHISPER
# =========================================================

@st.cache_resource
def load_asr():
    cpu_threads = max(
        1,
        min(4, os.cpu_count() or 1),
    )

    return WhisperModel(
        ASR_MODEL,
        device="cpu",
        compute_type="int8",
        cpu_threads=cpu_threads,
        num_workers=1,
    )


# =========================================================
# LOAD NLLB
# =========================================================

@st.cache_resource
def load_translator():
    model_path = snapshot_download(
        repo_id=TRANSLATOR_MODEL
    )

    sp_model = None

    for name in [
        "sentencepiece.bpe.model",
        "source.spm",
        "spm.model",
    ]:
        candidate = os.path.join(
            model_path,
            name,
        )

        if os.path.exists(candidate):
            sp_model = candidate
            break

    if sp_model is None:
        for root, _, files in os.walk(model_path):
            for f in files:
                if f.endswith(".model"):
                    sp_model = os.path.join(root, f)
                    break

            if sp_model:
                break

    if sp_model is None:
        raise RuntimeError(
            "SentencePiece model not found."
        )

    tokenizer = spm.SentencePieceProcessor(
        model_file=sp_model
    )

    intra_threads = max(
        1,
        min(4, os.cpu_count() or 1),
    )

    translator = ctranslate2.Translator(
        model_path,
        device="cpu",
        compute_type="int8",
        inter_threads=1,
        intra_threads=intra_threads,
    )

    return translator, tokenizer


# =========================================================
# TRANSCRIBE
# =========================================================

def transcribe_local(video_path, source_language):
    model = load_asr()

    language = {
        "English": "en",
        "Chinese": "zh",
        "Khmer": "km",
        "Vietnamese": "vi",
        "Korean": "ko",
        "Japanese": "ja",
    }.get(source_language)

    segments, info = model.transcribe(
        video_path,
        language=language,
        beam_size=1,
        best_of=1,
        temperature=0,
        vad_filter=True,
        condition_on_previous_text=False,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        word_timestamps=False,
    )

    results = []

    for seg in segments:
        text = (seg.text or "").strip()

        if not text:
            continue

        results.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": text,
            }
        )

    return results


# =========================================================
# TRANSLATE
# =========================================================

def translate_local(
    texts,
    source_language,
    target_language="Khmer",
):
    if not texts:
        return []

    if source_language == target_language:
        return list(texts)

    translator, tokenizer = load_translator()

    source_code = LANG_MAP[source_language]
    target_code = LANG_MAP[target_language]

    source_tokens = []

    for text in texts:
        pieces = tokenizer.encode(
            text,
            out_type=str,
        )

        source_tokens.append(pieces)

    results = translator.translate_batch(
        source_tokens,
        target_prefix=[
            [target_code]
            for _ in texts
        ],
        beam_size=1,
        max_decoding_length=64,
        repetition_penalty=1.05,
    )

    output = []

    for result in results:
        tokens = result.hypotheses[0]

        if tokens and tokens[0] == target_code:
            tokens = tokens[1:]

        try:
            decoded = tokenizer.decode(tokens)
        except Exception:
            decoded = " ".join(tokens)

        output.append(
            decoded.strip()
        )

    return output


# =========================================================
# TTS
# =========================================================

async def _tts_async(
    text,
    output_path,
    rate="+0%",
):
    communicate = edge_tts.Communicate(
        text=text,
        voice=KHMER_VOICE,
        rate=rate,
        volume="+0%",
        pitch="+0Hz",
    )

    await communicate.save(
        output_path
    )


def tts_one(
    text,
    output_path,
    rate="+0%",
):
    asyncio.run(
        _tts_async(
            text,
            output_path,
            rate,
        )
    )


# =========================================================
# FIT AUDIO TO SLOT
# =========================================================

def fit_audio_to_slot(
    input_path,
    output_path,
    target_duration,
):
    source_duration = audio_duration(
        input_path
    )

    if source_duration <= 0:
        return False

    target_duration = max(
        0.25,
        float(target_duration),
    )

    ratio = (
        source_duration /
        target_duration
    )

    filters = []

    # Pitch-preserving speed correction.
    # Keep each atempo between 0.5 and 2.0.
    while ratio > 2.0:
        filters.append("atempo=2.0")
        ratio /= 2.0

    while ratio < 0.5:
        filters.append("atempo=0.5")
        ratio /= 0.5

    filters.append(
        f"atempo={ratio:.6f}"
    )

    filters.extend(
        [
            "afade=t=in:st=0:d=0.015",
            (
                "afade=t=out:"
                f"st={max(0.02, target_duration - 0.02):.3f}:"
                "d=0.015"
            ),
        ]
    )

    run_cmd(
        [
            FFMPEG,
            "-i",
            input_path,
            "-filter:a",
            ",".join(filters),
            "-ar",
            "22050",
            "-ac",
            "1",
            "-y",
            output_path,
        ]
    )

    return True


# =========================================================
# MAKE DUBBING AUDIO
# =========================================================

def make_khmer_dubbing(
    chunks,
    duration,
    temp_dir,
    rate="+0%",
):
    jobs = []

    for i, chunk in enumerate(chunks):
        text = (
            chunk.get("khmer") or ""
        ).strip()

        if not text:
            continue

        if text.startswith("⚠️"):
            continue

        start = max(
            0.0,
            float(chunk.get("start", 0)),
        )

        end = min(
            duration,
            float(
                chunk.get(
                    "end",
                    start + 1,
                )
            ),
        )

        end = max(
            start + 0.25,
            end,
        )

        if start >= duration:
            continue

        if end <= start:
            continue

        jobs.append(
            {
                "index": i,
                "text": text,
                "start": start,
                "end": end,
            }
        )

    if not jobs:
        raise RuntimeError(
            "No Khmer speech segments found."
        )

    valid = []

    batch_size = 4

    for base in range(
        0,
        len(jobs),
        batch_size,
    ):
        batch = jobs[
            base:base + batch_size
        ]

        st.session_state[
            "progress_status"
        ] = (
            f"🗣️ Khmer Voice "
            f"{min(base + len(batch), len(jobs))}"
            f"/{len(jobs)}..."
        )

        def process_one(job):
            i = job["index"]
            text = job["text"]
            start = job["start"]
            end = job["end"]

            raw = os.path.join(
                temp_dir,
                f"voice_raw_{i}.mp3",
            )

            fitted = os.path.join(
                temp_dir,
                f"voice_fit_{i}.wav",
            )

            tts_one(
                text,
                raw,
                rate=rate,
            )

            ok = fit_audio_to_slot(
                raw,
                fitted,
                end - start,
            )

            if not ok:
                return None

            return {
                "file": fitted,
                "start": start,
                "end": end,
            }

        with ThreadPoolExecutor(
            max_workers=4
        ) as executor:

            results = list(
                executor.map(
                    process_one,
                    batch,
                )
            )

        for item in results:
            if item:
                valid.append(item)

    if not valid:
        raise RuntimeError(
            "Khmer TTS failed."
        )

    return valid


# =========================================================
# CREATE ASS
# =========================================================

def make_ass(
    chunks,
    ass_path,
):
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, TertiaryColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Khmer,Noto Sans Khmer,54,&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,0,0,1,3,1,2,50,50,55,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines = [header]

    for chunk in chunks:
        text = (
            chunk.get("khmer")
            or chunk.get("text")
            or ""
        )

        text = safe_text(text)

        if not text:
            continue

        start = format_ass_time(
            chunk.get("start", 0)
        )

        end = format_ass_time(
            chunk.get("end", 0)
        )

        lines.append(
            "Dialogue: 0,"
            f"{start},{end},Khmer,,0,0,0,,"
            f"{text}\n"
        )

    with open(
        ass_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "".join(lines)
        )


# =========================================================
# FINAL VIDEO
# ONE FFMPEG PASS
# =========================================================

def render_final_video(
    input_video,
    output_video,
    voice_files,
    ass_path,
    original_volume=0.05,
):
    if not voice_files:
        raise RuntimeError(
            "No voice files."
        )

    inputs = [
        "-i",
        input_video,
    ]

    for item in voice_files:
        inputs.extend(
            [
                "-i",
                item["file"],
            ]
        )

    filter_parts = []

    # Original audio.
    filter_parts.append(
        "[0:a]"
        f"volume={original_volume:.3f}"
        "[orig]"
    )

    mix_inputs = ["[orig]"]

    for n, item in enumerate(
        voice_files,
        start=1,
    ):
        delay_ms = max(
            0,
            int(
                round(
                    item["start"] * 1000
                )
            ),
        )

        label = f"v{n}"

        filter_parts.append(
            f"[{n}:a]"
            f"adelay={delay_ms}|{delay_ms},"
            "volume=1.0"
            f"[{label}]"
        )

        mix_inputs.append(
            f"[{label}]"
        )

    filter_parts.append(
        "".join(mix_inputs)
        + f"amix=inputs={len(mix_inputs)}:"
        "duration=first:"
        "dropout_transition=0:"
        "normalize=0"
        "[mix]"
    )

    filter_complex = ";".join(
        filter_parts
    )

    filter_complex += (
        f";[0:v]subtitles="
        f"'{ass_path.replace(chr(39), r\"'\\''\")}'"
        "[vout]"
    )

    cmd = [
        FFMPEG,
        "-i",
        input_video,
    ]

    for item in voice_files:
        cmd.extend(
            [
                "-i",
                item["file"],
            ]
        )

    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[vout]",
            "-map",
            "[mix]",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "28",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            "-shortest",
            "-y",
            output_video,
        ]
    )

    run_cmd(cmd)

    return output_video


# =========================================================
# EXTRACT AUDIO
# =========================================================

def extract_audio(
    video_path,
    audio_path,
):
    run_cmd(
        [
            FFMPEG,
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            "-y",
            audio_path,
        ]
    )


# =========================================================
# MAIN APP
# =========================================================

def main():
    st.title(
        "🇰🇭 Smey Auto Caption"
    )

    st.caption(
        "Video → Speech → Khmer → "
        "Sreymom Voice → Sync → MP4"
    )

    uploaded = st.file_uploader(
        "🎬 បញ្ចូលវីដេអូ",
        type=[
            "mp4",
            "mov",
            "mkv",
            "webm",
            "avi",
        ],
    )

    col1, col2 = st.columns(2)

    with col1:
        source_language = st.selectbox(
            "🌍 ភាសាវីដេអូ",
            [
                "English",
                "Chinese",
                "Khmer",
                "Vietnamese",
                "Korean",
                "Japanese",
            ],
        )

    with col2:
        target_language = st.selectbox(
            "🇰🇭 ភាសាបកប្រែ",
            ["Khmer"],
        )

    khmer_dubbing = st.checkbox(
        "🗣️ Khmer Dubbing",
        value=True,
    )

    voice_speed = st.slider(
        "🎙️ Voice Speed",
        min_value=-20,
        max_value=20,
        value=0,
        step=5,
        help="0% = ល្បឿនធម្មតា",
    )

    original_volume = st.slider(
        "🎵 Original Audio / Music",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        step=0.05,
    )

    if "progress_status" not in st.session_state:
        st.session_state[
            "progress_status"
        ] = ""

    if uploaded:
        st.video(uploaded)

    if not uploaded:
        return

    start_button = st.button(
        "▶️ ចាប់ផ្ដើម",
        type="primary",
        use_container_width=True,
    )

    if not start_button:
        return

    work_id = uuid.uuid4().hex

    temp_dir = os.path.join(
        "temp",
        work_id,
    )

    os.makedirs(
        temp_dir,
        exist_ok=True,
    )

    input_path = os.path.join(
        temp_dir,
        uploaded.name,
    )

    audio_path = os.path.join(
        temp_dir,
        "audio.wav",
    )

    ass_path = os.path.join(
        temp_dir,
        "captions.ass",
    )

    output_path = os.path.join(
        temp_dir,
        "Smey_Khmer_Dubbing.mp4",
    )

    with open(
        input_path,
        "wb",
    ) as f:
        f.write(
            uploaded.getbuffer()
        )

    try:
        # -------------------------------------------------
        # VIDEO DURATION
        # -------------------------------------------------

        duration = audio_duration(
            input_path
        )

        if duration <= 0:
            raise RuntimeError(
                "Cannot read video duration."
            )

        # -------------------------------------------------
        # EXTRACT AUDIO
        # -------------------------------------------------

        st.info(
            "🎧 កំពុងយកសំឡេងពីវីដេអូ..."
        )

        extract_audio(
            input_path,
            audio_path,
        )

        # -------------------------------------------------
        # WHISPER
        # -------------------------------------------------

        st.info(
            "🎤 Whisper កំពុងស្តាប់សំឡេង..."
        )

        segments = transcribe_local(
            audio_path,
            source_language,
        )

        if not segments:
            raise RuntimeError(
                "Whisper មិនរកឃើញសំឡេងនិយាយ។"
            )

        # -------------------------------------------------
        # TRANSLATION
        # -------------------------------------------------

        texts = [
            x["text"]
            for x in segments
        ]

        if source_language == "Khmer":
            khmer_texts = texts
        else:
            st.info(
                "🌐 កំពុងបកប្រែជា Khmer..."
            )

            khmer_texts = translate_local(
                texts,
                source_language,
                "Khmer",
            )

        chunks = []

        for seg, khmer in zip(
            segments,
            khmer_texts,
        ):
            chunks.append(
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                    "khmer": khmer,
                }
            )

        # -------------------------------------------------
        # ASS CAPTION
        # -------------------------------------------------

        make_ass(
            chunks,
            ass_path,
        )

        # -------------------------------------------------
        # DUBBING
        # -------------------------------------------------

        if khmer_dubbing:
            rate = (
                f"{voice_speed:+d}%"
            )

            voice_files = (
                make_khmer_dubbing(
                    chunks,
                    duration,
                    temp_dir,
                    rate=rate,
                )
            )

            st.info(
                "🎬 កំពុងបង្កើត MP4..."
            )

            render_final_video(
                input_video=input_path,
                output_video=output_path,
                voice_files=voice_files,
                ass_path=ass_path,
                original_volume=original_volume,
            )

        else:
            st.info(
                "🎬 កំពុងបង្កើត Caption MP4..."
            )

            run_cmd(
                [
                    FFMPEG,
                    "-i",
                    input_path,
                    "-vf",
                    f"subtitles={ass_path}",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-crf",
                    "28",
                    "-c:a",
                    "copy",
                    "-movflags",
                    "+faststart",
                    "-y",
                    output_path,
                ]
            )

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        if not os.path.exists(
            output_path
        ):
            raise RuntimeError(
                "MP4 មិនត្រូវបានបង្កើត។"
            )

        file_size = os.path.getsize(
            output_path
        )

        if file_size < 1000:
            raise RuntimeError(
                "MP4 output ខូច ឬទទេ។"
            )

        st.success(
            "✅ រួចរាល់!"
        )

        st.video(
            output_path
        )

        with open(
            output_path,
            "rb",
        ) as f:
            st.download_button(
                label="⬇️ ទាញយក Khmer Dubbing MP4",
                data=f,
                file_name=(
                    "Smey_Khmer_Dubbing.mp4"
                ),
                mime="video/mp4",
                use_container_width=True,
            )

    except Exception as e:
        st.error(
            f"❌ Error: {e}"
        )

        st.exception(e)


if __name__ == "__main__":
    main()
