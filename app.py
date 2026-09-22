import os
import subprocess
import tempfile
import json
import time

import streamlit as st
from google import genai
from google.genai import types
import imageio_ffmpeg


# =========================================================
# APP
# =========================================================

st.set_page_config(
    page_title="Smey Auto Caption",
    page_icon="🇰🇭",
)

# =========================================================
# Logo
# =========================================================

LOGO_FILE = "file_00000000568c8211831d7859c11ddf61.png"

if os.path.exists(LOGO_FILE):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(LOGO_FILE, width=300)

st.title("🇰🇭 Smey Auto Caption")
st.write("Gemini → Caption → Auto Translate → MP4")


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
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        audio_path,
    ])


# =========================================================
# Gemini JSON
# =========================================================

def parse_json_response(response):
    raw = getattr(response, "text", None)

    if not raw:
        raise ValueError(
            "Gemini មិនបានផ្ញើ JSON ត្រឡប់មកទេ។"
        )

    raw = raw.strip()

    # Remove markdown code fence if Gemini adds it
    if raw.startswith("```"):
        lines = raw.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        raw = "\n".join(lines).strip()

    try:
        return json.loads(raw)

    except json.JSONDecodeError as e:
        raise ValueError(
            "Gemini បានផ្ញើទិន្នន័យដែលមិនមែនជា JSON ត្រឹមត្រូវ។"
        ) from e


def gemini_json(client, contents, model):
    """
    Call Gemini.

    Important:
    - 429 / RESOURCE_EXHAUSTED is NOT retried.
      This is normally quota/rate-limit related.
    - 503 / UNAVAILABLE is retried a few times.
    """

    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )

            return response, model

        except Exception as e:
            message = str(e)

            # =================================================
            # QUOTA / RATE LIMIT
            # =================================================
            if (
                "429" in message
                or "RESOURCE_EXHAUSTED" in message
                or "quota" in message.lower()
            ):
                raise RuntimeError(
                    "⚠️ Gemini API បានប្រើអស់ Quota។\n\n"
                    "សូមរង់ចាំឱ្យ Quota Reset "
                    "ឬបើក Billing នៅ Google AI Studio "
                    "ដើម្បីបន្តប្រើ។"
                ) from e

            # =================================================
            # SERVER BUSY
            # =================================================
            if (
                "503" in message
                or "UNAVAILABLE" in message
            ):
                if attempt < max_retries - 1:
                    time.sleep(2 + attempt * 2)
                    continue

                raise RuntimeError(
                    "⚠️ Gemini Server កំពុងរវល់។ "
                    "សូមសាកល្បងម្តងទៀតនៅពេលក្រោយ។"
                ) from e

            # =================================================
            # OTHER ERROR
            # =================================================
            raise


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
# AUTO TRANSLATE
# =========================================================

def translate_captions(
    client,
    groups,
    source_language,
    target_language,
):
    if not groups or target_language == "មិនបកប្រែ":
        return groups

    source_map = {
        "Auto Detect": "the original language",
        "🇰🇭 ខ្មែរ": "Khmer",
        "🇬🇧 English": "English",
        "🇨🇳 中文": "Chinese",
        "🇻🇳 Tiếng Việt": "Vietnamese",
        "🇰🇷 한국어": "Korean",
        "🇯🇵 日本語": "Japanese",
    }

    target_map = {
        "🇰🇭 ខ្មែរ": "Khmer",
        "🇬🇧 English": "English",
        "🇨🇳 中文": "Chinese",
        "🇻🇳 Tiếng Việt": "Vietnamese",
        "🇰🇷 한국어": "Korean",
        "🇯🇵 日本語": "Japanese",
    }

    source_name = source_map.get(
        source_language,
        "the original language",
    )

    target_name = target_map.get(
        target_language,
        "Khmer",
    )

    texts = [
        group["text"]
        for group in groups
    ]

    prompt = f"""
Translate the following video captions.

Source language:
{source_name}

Target language:
{target_name}

IMPORTANT RULES:
1. Return exactly one translated string for each input caption.
2. Keep the exact same order.
3. Do not add explanations.
4. Do not add numbering.
5. Do not merge captions.
6. Keep names and numbers accurate.
7. Translate naturally for subtitles.
8. Return only a JSON array of strings.

Captions:
"""

    for i, text in enumerate(texts, start=1):
        prompt += f"\n{i}. {text}"

    response, _ = gemini_json(
        client,
        prompt,
        "gemini-3.8-flash",
    )

    translated = parse_json_response(response)

    if not isinstance(translated, list):
        raise ValueError(
            "Gemini មិនបានបញ្ជូន JSON array សម្រាប់ការបកប្រែទេ។"
        )

    if len(translated) != len(groups):
        raise ValueError(
            "Gemini បានបកប្រែចំនួន Caption មិនត្រូវគ្នា។"
        )

    return [
        {
            "start": group["start"],
            "end": group["end"],
            "text": str(translated_text).strip(),
        }
        for group, translated_text in zip(
            groups,
            translated,
        )
    ]


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

    with open(
        filename,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(header)

        for group in groups:
            text = str(
                group["text"]
            ).replace("\n", " ")

            text = text.replace(
                "{",
                r"\{",
            ).replace(
                "}",
                r"\}",
            )

            f.write(
                f"Dialogue: 0,"
                f"{ass_time(group['start'])},"
                f"{ass_time(group['end'])},"
                f"Khmer,,0,0,0,,"
                f"{text}\n"
            )


# =========================================================
# BURN CAPTION INTO VIDEO
# =========================================================

def burn_caption(
    video_path,
    ass_path,
    output_path,
):
    escaped_ass = (
        ass_path
        .replace("\\", "/")
        .replace(":", r"\:")
        .replace("'", r"\'")
    )

    run_ffmpeg([
        "-y",
        "-i",
        video_path,
        "-vf",
        f"ass='{escaped_ass}'",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
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
# VIDEO UPLOAD
# =========================================================

video = st.file_uploader(
    "🎥 ជ្រើសវីដេអូ",
    type=[
        "mp4",
        "mov",
        "mkv",
        "webm",
    ],
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

            video_path = os.path.join(
                temp_dir,
                "input.mp4",
            )

            audio_path = os.path.join(
                temp_dir,
                "audio.wav",
            )

            ass_path = os.path.join(
                temp_dir,
                "khmer_caption.ass",
            )

            output_path = os.path.join(
                temp_dir,
                "smey_auto_caption.mp4",
            )

            # =================================================
            # SAVE VIDEO
            # =================================================

            with open(
                video_path,
                "wb",
            ) as f:
                f.write(
                    video.getbuffer()
                )

            try:

                # =================================================
                # CHECK API KEY
                # =================================================

                if "GEMINI_API_KEY" not in st.secrets:
                    raise RuntimeError(
                        "❌ រកមិនឃើញ GEMINI_API_KEY។ "
                        "សូមបញ្ចូល API Key នៅ Streamlit Secrets។"
                    )

                # =================================================
                # EXTRACT AUDIO
                # =================================================

                with st.spinner(
                    "⚡ កំពុងដកសំឡេង..."
                ):
                    extract_audio(
                        video_path,
                        audio_path,
                    )

                # =================================================
                # GEMINI TRANSCRIPTION
                # =================================================

                with st.spinner(
                    "🎙️ Gemini កំពុងស្តាប់សំឡេង..."
                ):

                    client = genai.Client(
                        api_key=st.secrets[
                            "GEMINI_API_KEY"
                        ]
                    )

                    audio_file = client.files.upload(
                        file=audio_path
                    )

                    language_code_map = {
                        "🇰🇭 ខ្មែរ": "km-KH",
                        "🇬🇧 English": "en-US",
                        "🇨🇳 中文": "zh-CN",
                        "🇻🇳 Tiếng Việt": "vi-VN",
                        "🇰🇷 한국어": "ko-KR",
                        "🇯🇵 日本語": "ja-JP",
                    }

                    language_hint = language_code_map.get(
                        source_language,
                        "detect automatically",
                    )

                    timestamp_prompt = f"""
Transcribe this audio accurately.

Language:
{language_hint}

Return ONLY valid JSON.

Return an array of caption segments.

Each segment must have exactly these fields:
- start: number of seconds from the beginning
- end: number of seconds from the beginning
- text: the exact spoken words for that segment

Make short natural caption segments,
about 2 seconds each.

Do not translate.
Do not add explanations.
Do not add markdown.
"""

                    response, _ = gemini_json(
                        client,
                        [
                            types.Part.from_uri(
                                file_uri=audio_file.uri,
                                mime_type=audio_file.mime_type,
                            ),
                            timestamp_prompt,
                        ],
                        "gemini-3.8-flash",
                    )

                    raw_segments = parse_json_response(
                        response
                    )

                    groups = []

                    if isinstance(
                        raw_segments,
                        list,
                    ):
                        for item in raw_segments:

                            if not isinstance(
                                item,
                                dict,
                            ):
                                continue

                            text = str(
                                item.get(
                                    "text",
                                    "",
                                )
                            ).strip()

                            if not text:
                                continue

                            try:
                                start = float(
                                    item.get(
                                        "start",
                                        0,
                                    )
                                )

                                end = float(
                                    item.get(
                                        "end",
                                        start + 2,
                                    )
                                )

                            except (
                                TypeError,
                                ValueError,
                            ):
                                continue

                            if end <= start:
                                end = start + 2

                            groups.append({
                                "start": max(
                                    0,
                                    start,
                                ),
                                "end": max(
                                    0,
                                    end,
                                ),
                                "text": text,
                            })

                    if not groups:
                        raise RuntimeError(
                            "❌ Gemini មិនបានរកឃើញ Caption timestamps ទេ។"
                        )

                # =================================================
                # TRANSLATE
                # =================================================

                if target_language != "មិនបកប្រែ":

                    with st.spinner(
                        "🌐 Gemini កំពុងបកប្រែ Caption..."
                    ):
                        groups = translate_captions(
                            client,
                            groups,
                            source_language,
                            target_language,
                        )

                # =================================================
                # CREATE ASS
                # =================================================

                create_ass(
                    groups,
                    ass_path,
                )

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

                # =================================================
                # RESULT
                # =================================================

                with open(
                    output_path,
                    "rb",
                ) as f:
                    output_data = f.read()

                st.success(
                    "✅ វីដេអូរួចរាល់!"
                )

                st.video(
                    output_data
                )

                st.download_button(
                    "⬇️ ទាញយកវីដេអូ MP4",
                    data=output_data,
                    file_name="smey_auto_caption.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                    on_click="ignore",
                )

            except RuntimeError as e:
                st.error(str(e))

            except Exception as e:
                message = str(e)

                if (
                    "429" in message
                    or "RESOURCE_EXHAUSTED" in message
                    or "quota" in message.lower()
                ):
                    st.error(
                        "⚠️ Gemini API បានប្រើអស់ Quota។ "
                        "សូមរង់ចាំឱ្យ Quota Reset "
                        "ឬបើក Billing ដើម្បីបន្តប្រើ។"
                    )

                elif (
                    "503" in message
                    or "UNAVAILABLE" in message
                ):
                    st.error(
                        "⚠️ Gemini Server កំពុងរវល់។ "
                        "សូមសាកល្បងម្តងទៀតនៅពេលក្រោយ។"
                    )

                else:
                    st.error(
                        "❌ មានបញ្ហាពេលបង្កើតវីដេអូ។"
                    )
                    
