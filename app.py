import os
import subprocess
import tempfile
import asyncio
import shutil
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
import imageio_ffmpeg
from gtts import gTTS
from faster_whisper import WhisperModel
import ctranslate2
import sentencepiece as spm
from huggingface_hub import snapshot_download

st.set_page_config(page_title="Smey Auto Caption", page_icon="🇰🇭")
st.title("🇰🇭 Smey Auto Caption")

# Fastest ASR model
ASR_MODEL = "tiny"
TRANSLATION_MODEL = "osa911/nllb-200-distilled-600M-ct2-int8"

LANGUAGE_CODES = {
    "🇰🇭 ខ្មែរ": {"whisper": "km", "translate": "km"},
    "🇬🇧 English": {"whisper": "en", "translate": "en"},
    "🇨🇳 中文": {"whisper": "zh", "translate": "zh"},
    "🇻🇳 Tiếng Việt": {"whisper": "vi", "translate": "vi"},
    "🇰🇷 한국어": {"whisper": "ko", "translate": "ko"},
    "🇯🇵 日本語": {"whisper": "ja", "translate": "ja"},
}

TARGET_CODES = {
    "🇰🇭 ខ្មែរ": "khm_Khm",
    "🇬🇧 English": "eng_Latn",
    "🇨🇳 中文": "zho_Hans",
    "🇻🇳 Tiếng Việt": "vie_Latn",
    "🇰🇷 한국어": "kor_Hang",
    "🇯🇵 日本語": "jpn_Jpan",
}

SOURCE_CODES = {
    "km": "khm_Khm",
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "vi": "vie_Latn",
    "ko": "kor_Hang",
    "ja": "jpn_Jpan",
}

@st.cache_resource(show_spinner=False)
def load_asr():
    return WhisperModel(ASR_MODEL, device="cpu", compute_type="int8")

@st.cache_resource(show_spinner=False)
def load_translator():
    model_dir = snapshot_download(TRANSLATION_MODEL)
    tokenizer = spm.SentencePieceProcessor(
        model_file=os.path.join(model_dir, "sentencepiece.bpe.model")
    )
    translator = ctranslate2.Translator(
        model_dir, device="cpu", compute_type="int8"
    )
    return tokenizer, translator

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

def transcribe_local(audio_path, source_language):
    model = load_asr()
    whisper_language = None
    if source_language != "Auto Detect":
        whisper_language = LANGUAGE_CODES[source_language]["whisper"]

    segments, info = model.transcribe(
        audio_path,
        language=whisper_language,
        task="transcribe",
        beam_size=1,
        best_of=1,
        temperature=0,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=350),
        condition_on_previous_text=False,
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
        groups.append({"start": start, "end": end, "text": text})

    return groups, getattr(info, "language", None)

def translate_local(groups, source_code, target_code):
    if not groups or source_code == target_code:
        return groups

    src_lang = SOURCE_CODES.get(source_code)
    if not src_lang:
        raise RuntimeError(f"Unsupported source language: {source_code}")

    tokenizer, translator = load_translator()
    texts = [str(g["text"]).strip() for g in groups]
    source_tokens = [
        [src_lang] + tokenizer.encode(text, out_type=str) + ["</s>"]
        for text in texts
    ]

    results = translator.translate_batch(
        source_tokens,
        target_prefix=[[target_code] for _ in texts],
        beam_size=2,
        max_decoding_length=96,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
    )

    translated = []
    for result in results:
        tokens = list(result.hypotheses[0])
        if tokens and tokens[0] == target_code:
            tokens = tokens[1:]
        text = tokenizer.decode(tokens).strip()

        words = text.split()
        if len(words) >= 8:
            unique_ratio = len(set(words)) / max(1, len(words))
            if unique_ratio < 0.35:
                text = ""

        translated.append(text)

    return [
        {
            "start": group["start"],
            "end": group["end"],
            "text": translated_text or group["text"],
        }
        for group, translated_text in zip(groups, translated)
    ]

async def _tts_one(text, output_path, voice, rate):
    fallback_path = output_path + ".gtts.mp3"
    try:
        gTTS(text=text, lang="km", slow=False).save(fallback_path)

        rate_value = {
            "+0%": 1.0,
            "+10%": 1.10,
            "+15%": 1.15,
            "+20%": 1.20,
            "+25%": 1.25,
        }.get(rate, 1.0)

        if rate_value == 1.0:
            shutil.move(fallback_path, output_path)
        else:
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            subprocess.run(
                [
                    ffmpeg, "-y", "-i", fallback_path,
                    "-filter:a", f"atempo={rate_value}", output_path
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            try:
                os.remove(fallback_path)
            except OSError:
                pass
    except Exception as e:
        try:
            os.remove(fallback_path)
        except OSError:
            pass
        raise RuntimeError(f"មិនអាចបង្កើតសំឡេងខ្មែរ TTS: {e}")

    if not os.path.exists(output_path) or os.path.getsize(output_path) <= 1000:
        raise RuntimeError("មិនអាចបង្កើតសំឡេងខ្មែរ TTS បានទេ។")

def _audio_duration(path):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.run(
        [ffmpeg, "-i", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    import re
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", proc.stderr)
    if not m:
        return 0.0
    h, mm, ss = m.groups()
    return int(h) * 3600 + int(mm) * 60 + float(ss)

def _fit_audio_to_slot(input_path, output_path, slot_seconds):
    duration = _audio_duration(input_path)
    if duration <= 0 or slot_seconds <= 0 or duration <= slot_seconds * 1.01:
        shutil.copyfile(input_path, output_path)
        return

    factor = min(4.0, max(1.0, duration / max(0.05, slot_seconds)))
    chain = []
    remaining = factor
    while remaining > 2.0:
        chain.append("atempo=2.0")
        remaining /= 2.0
    chain.append(f"atempo={remaining:.4f}")

    run_ffmpeg([
        "-y", "-i", input_path,
        "-filter:a", ",".join(chain),
        "-c:a", "aac", "-b:a", "128k",
        output_path,
    ])

def make_khmer_dubbing(
    video_path,
    groups,
    output_audio,
    voice="km-KH-PisethNeural",
    rate="+15%",
    original_volume=0.05,
):
    if not groups:
        raise RuntimeError("❌ មិនមាន Caption សម្រាប់ Dubbing ទេ។")

    temp_dir = tempfile.mkdtemp(prefix="smey_khmer_dub_")

    try:
        jobs = []
        for i, group in enumerate(groups):
            text = str(group.get("text", "")).strip()
            if not text:
                continue
            raw = os.path.join(temp_dir, f"raw_{i:04d}.mp3")
            fitted = os.path.join(temp_dir, f"voice_{i:04d}.m4a")
            start = float(group["start"])
            end = float(group["end"])
            slot = max(0.25, end - start)
            jobs.append((text, raw, fitted, start, slot))

        def make_one(job):
            text, raw, fitted, start, slot = job
            asyncio.run(_tts_one(text, raw, voice, rate))
            _fit_audio_to_slot(raw, fitted, slot)
            return fitted, start

        worker_count = min(4, max(1, len(jobs)))
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            clips = list(pool.map(make_one, jobs))

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
            + f"amix=inputs={len(mix)}:duration=first:"
              "dropout_transition=0:normalize=0[aout]"
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
                f"Dialogue: 0,{ass_time(group['start'])},"
                f"{ass_time(group['end'])},Khmer,,0,0,0,,{text}\n"
            )

def burn_caption(video_path, ass_path, output_path):
    escaped_ass = (
        ass_path.replace("\\", "/")
        .replace(":", r"\:")
        .replace("'", r"\'")
    )

    run_ffmpeg([
        "-y",
        "-i", video_path,
        "-vf", f"ass='{escaped_ass}'",
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "ultrafast",
        "-c:a", "aac",
        "-b:a", "160k",
        "-movflags", "+faststart",
        output_path,
    ])

source_language = st.selectbox(
    "🌐 ភាសាវីដេអូ",
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
    "🎯 បកប្រែទៅ",
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

st.divider()
st.subheader("🗣️ និយាយខ្មែរ")

khmer_dubbing = st.checkbox(
    "បើកសំឡេងខ្មែរ + រក្សាសំឡេងដើម",
    value=False,
)

khmer_voice = "km-KH-PisethNeural"

khmer_rate = st.select_slider(
    "⚡ ល្បឿន",
    options=["+0%", "+10%", "+15%", "+20%", "+25%"],
    value="+15%",
    disabled=not khmer_dubbing,
)

original_audio_volume = 0.05

video = st.file_uploader(
    "🎥 បញ្ចូលវីដេអូ",
    type=["mp4", "mov", "mkv", "webm"],
)

if video is not None:
    caption_clicked = st.button("▶️ ចាប់ផ្ដើម", use_container_width=True)

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

                with st.spinner("🎙️ Local Whisper កំពុងស្តាប់សំឡេង..."):
                    groups, detected_language = transcribe_local(
                        audio_path, source_language
                    )

                if not groups:
                    raise RuntimeError(
                        "❌ Local Whisper មិនបានរកឃើញ Caption ទេ។"
                    )

                if source_language == "Auto Detect":
                    source_code = detected_language
                    if source_code not in SOURCE_CODES:
                        raise RuntimeError(
                            f"❌ Auto Detect រកភាសា '{detected_language}' "
                            "ដែលមិនទាន់មានក្នុង Local Translator Map។"
                        )
                else:
                    source_code = LANGUAGE_CODES[source_language]["translate"]

                if target_language != "មិនបកប្រែ":
                    with st.spinner("🌐 Local Translator កំពុងបកប្រែ..."):
                        groups = translate_local(
                            groups,
                            source_code,
                            TARGET_CODES[target_language],
                        )

                if khmer_dubbing:
                    if target_language != "🇰🇭 ខ្មែរ":
                        raise RuntimeError(
                            "⚠️ Khmer Dubbing ត្រូវជ្រើស "
                            "🎯 បកប្រែទៅជា → 🇰🇭 ខ្មែរ។"
                        )

                    dubbed_audio_path = os.path.join(
                        temp_dir, "khmer_dubbed_audio.m4a"
                    )
                    dubbed_video_path = os.path.join(
                        temp_dir, "video_with_khmer_voice.mp4"
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

                create_ass(groups, ass_path)

                with st.spinner(
                    "🎬 កំពុងដាក់ Caption ជាប់ក្នុងវីដេអូ..."
                ):
                    burn_caption(video_path, ass_path, output_path)

                with open(output_path, "rb") as f:
                    output_data = f.read()

                st.success("✅ វីដេអូរួចរាល់!")
                st.video(output_data)

                st.download_button(
                    "⬇️ ទាញយក MP4",
                    data=output_data,
                    file_name="smey_auto_caption.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                    on_click="ignore",
                )

            except Exception as e:
                st.error(f"❌ មានបញ្ហាពេលបង្កើតវីដេអូ: {e}")
