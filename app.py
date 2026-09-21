import streamlit as st
import subprocess
import tempfile
import os
import threading
from transformers import pipeline


# =========================
# APP
# =========================

st.set_page_config(
    page_title="🇰🇭 Smey Auto Caption",
    page_icon="🇰🇭"
)

st.title("🇰🇭 Smey Auto Caption")
st.write("🎙️ ស្គាល់សំឡេងខ្មែរ → 📝 Caption ខ្មែរ")


# =========================
# LOAD AI MODEL
# =========================

@st.cache_resource
def load_model():

    return pipeline(
        "automatic-speech-recognition",
        model="1morecupofhottea/whisper-turbo-khmer-v9",
        chunk_length_s=30,
        device=-1
    )


# =========================
# QUEUE
# =========================

@st.cache_resource
def get_processing_lock():

    return threading.Lock()


processing_lock = get_processing_lock()


# =========================
# ASS TIME
# =========================

def ass_time(seconds):

    seconds = max(0, float(seconds))

    hours = int(seconds // 3600)

    minutes = int(
        (seconds % 3600) // 60
    )

    secs = seconds % 60

    return (
        f"{hours}:"
        f"{minutes:02d}:"
        f"{secs:05.2f}"
    )


# =========================
# VIDEO DURATION
# =========================

def get_duration(filename):

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            filename
        ],
        capture_output=True,
        text=True,
        check=True
    )

    return float(
        result.stdout.strip()
    )


# =========================
# TEXT SPLIT
# =========================

def split_text(text):

    text = text.strip()

    if not text:
        return []

    parts = [
        part.strip()
        for part in text.split(";")
        if part.strip()
    ]

    if len(parts) > 1:
        return parts

    words = text.split()

    groups = []

    current = []

    for word in words:

        current.append(word)

        if len(current) >= 5:

            groups.append(
                " ".join(current)
            )

            current = []

    if current:

        groups.append(
            " ".join(current)
        )

    return groups


# =========================
# WORD TIMESTAMP CAPTIONS
# =========================

def make_groups_from_chunks(chunks):

    groups = []

    current_words = []

    group_start = None

    group_end = None

    for chunk in chunks:

        text = chunk.get(
            "text",
            ""
        ).strip()

        timestamp = chunk.get(
            "timestamp"
        )

        if not text:
            continue

        if not timestamp:
            continue

        start = timestamp[0]

        end = timestamp[1]

        if start is None:
            continue

        if end is None:

            end = start + 0.5

        if group_start is None:

            group_start = start

        current_words.append(text)

        group_end = end

        # ប្រហែល 5 ពាក្យក្នុង Caption មួយ
        if len(current_words) >= 5:

            groups.append(
                (
                    group_start,
                    group_end,
                    " ".join(current_words)
                )
            )

            current_words = []

            group_start = None

            group_end = None

    # ពាក្យដែលនៅសល់
    if (
        current_words
        and group_start is not None
        and group_end is not None
    ):

        groups.append(
            (
                group_start,
                group_end,
                " ".join(current_words)
            )
        )

    return groups


# =========================
# FALLBACK SEGMENT TIMESTAMP
# =========================

def make_groups_from_segments(chunks):

    groups = []

    for chunk in chunks:

        text = chunk.get(
            "text",
            ""
        ).strip()

        timestamp = chunk.get(
            "timestamp"
        )

        if not text:
            continue

        if not timestamp:
            continue

        start = timestamp[0]

        end = timestamp[1]

        if start is None:
            continue

        if end is None:
            end = start + 1

        parts = split_text(text)

        if not parts:
            continue

        duration = max(
            0.1,
            end - start
        )

        step = duration / len(parts)

        for i, part in enumerate(parts):

            part_start = (
                start + i * step
            )

            part_end = (
                start + (i + 1) * step
            )

            groups.append(
                (
                    part_start,
                    part_end,
                    part
                )
            )

    return groups


# =========================
# FALLBACK TEXT
# =========================

def make_groups_from_text(
    text,
    duration
):

    parts = split_text(text)

    if not parts:
        return []

    step = (
        duration /
        len(parts)
    )

    groups = []

    for i, part in enumerate(parts):

        start = i * step

        end = (
            (i + 1) * step
        )

        groups.append(
            (
                start,
                end,
                part
            )
        )

    return groups


# =========================
# CREATE ASS
# =========================

def create_ass(
    groups,
    filename
):

    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Khmer,Noto Sans Khmer,64,&H00FFFFFF,&H00FFFFFF,&H00000000,&H99000000,1,0,0,0,100,100,0,0,3,4,1,2,50,50,140,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(header)

        for start, end, text in groups:

            text = text.strip()

            if not text:
                continue

            text = text.replace(
                "\n",
                " "
            )

            text = text.replace(
                "{",
                r"\{"
            )

            text = text.replace(
                "}",
                r"\}"
            )

            line = (
                "Dialogue: 0,"
                + ass_time(start)
                + ","
                + ass_time(end)
                + ",Khmer,,0,0,0,,"
                + text
                + "\n"
            )

            f.write(line)


# =========================
# VIDEO UPLOAD
# =========================

video = st.file_uploader(
    "🎥 ជ្រើសវីដេអូ",
    type=[
        "mp4",
        "mov",
        "mkv",
        "webm"
    ]
)


if video:

    st.video(video)

    if st.button(
        "🚀 បង្កើត Caption ខ្មែរ",
        use_container_width=True
    ):

        # =========================
        # QUEUE CHECK
        # =========================

        if not processing_lock.acquire(
            blocking=False
        ):

            st.warning(
                "⏳ មានអ្នកកំពុងប្រើ AI សូមរង់ចាំបន្តិច..."
            )

            st.stop()

        try:

            with tempfile.TemporaryDirectory() as folder:

                input_file = os.path.join(
                    folder,
                    "input.mp4"
                )

                audio_file = os.path.join(
                    folder,
                    "audio.wav"
                )

                ass_file = os.path.join(
                    folder,
                    "caption.ass"
                )

                output_file = os.path.join(
                    folder,
                    "output.mp4"
                )


                # =========================
                # SAVE VIDEO
                # =========================

                with open(
                    input_file,
                    "wb"
                ) as f:

                    f.write(
                        video.getbuffer()
                    )


                # =========================
                # EXTRACT AUDIO
                # =========================

                with st.spinner(
                    "🔊 កំពុងរៀបចំសំឡេង..."
                ):

                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-i",
                            input_file,
                            "-vn",
                            "-ac",
                            "1",
                            "-ar",
                            "16000",
                            "-c:a",
                            "pcm_s16le",
                            audio_file
                        ],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE
                    )


                # =========================
                # VIDEO LENGTH
                # =========================

                duration = get_duration(
                    input_file
                )


                # =========================
                # AI SPEECH RECOGNITION
                # =========================

                with st.spinner(
                    "🎙️ AI កំពុងស្គាល់សំឡេងខ្មែរ..."
                ):

                    model = load_model()

                    result = model(
                        audio_file,
                        return_timestamps="word"
                    )


                # =========================
                # FULL TEXT
                # =========================

                full_text = result.get(
                    "text",
                    ""
                ).strip()


                if not full_text:

                    st.error(
                        "❌ AI មិនអាចស្គាល់សំឡេងបានទេ។"
                    )

                    st.stop()


                # =========================
                # WORD TIMESTAMPS
                # =========================

                chunks = result.get(
                    "chunks",
                    []
                )


                groups = (
                    make_groups_from_chunks(
                        chunks
                    )
                )


                # =========================
                # SEGMENT FALLBACK
                # =========================

                if not groups:

                    groups = (
                        make_groups_from_segments(
                            chunks
                        )
                    )


                # =========================
                # LAST FALLBACK
                # =========================

                if not groups:

                    groups = (
                        make_groups_from_text(
                            full_text,
                            duration
                        )
                    )


                if not groups:

                    st.error(
                        "❌ មិនអាចបង្កើត Caption បានទេ។"
                    )

                    st.stop()


                # =========================
                # CREATE ASS
                # =========================

                create_ass(
                    groups,
                    ass_file
                )


                # =========================
                # RENDER VIDEO
                # =========================

                with st.spinner(
                    "🎬 កំពុងដាក់ Caption លើវីដេអូ..."
                ):

                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-i",
                            input_file,
                            "-vf",
                            f"ass={ass_file}",
                            "-c:v",
                            "libx264",
                            "-preset",
                            "veryfast",
                            "-c:a",
                            "aac",
                            output_file
                        ],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE
                    )


                # =========================
                # RESULT
                # =========================

                st.success(
                    "✅ រួចរាល់!"
                )


                with open(
                    output_file,
                    "rb"
                ) as f:

                    st.download_button(
                        "⬇️ ទាញយកវីដេអូមាន Caption",
                        f,
                        file_name="khmer_caption.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )


        finally:

            processing_lock.release()
