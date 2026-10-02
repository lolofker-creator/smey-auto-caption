import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { pipeline } from "@huggingface/transformers";
import * as ort from "onnxruntime-web";
import { FFmpeg } from "@ffmpeg/ffmpeg";
import { fetchFile, toBlobURL } from "@ffmpeg/util";
import "./style.css";

let transcriber = null;
let ttsSession = null;
let ttsMeta = null;
let ffmpegInstance = null;

/* =========================
   WHISPER
========================= */

async function getTranscriber(setStatus) {
  if (transcriber) return transcriber;

  setStatus("⏳ កំពុងទាញ Whisper Tiny...");

  transcriber = await pipeline(
    "automatic-speech-recognition",
    "onnx-community/whisper-tiny"
  );

  return transcriber;
}

/* =========================
   TRANSLATION
========================= */

async function translateText(text, sourceLang) {
  if (!text?.trim()) return "";

  const source =
    sourceLang === "zho_Hans"
      ? "zh-CN"
      : "en";

  const params = new URLSearchParams({
    q: text.trim(),
    langpair: `${source}|km-KM`,
    mt: "1",
  });

  const response = await fetch(
    `https://api.mymemory.translated.net/get?${params.toString()}`
  );

  if (!response.ok) {
    throw new Error(
      `Translation API error ${response.status}`
    );
  }

  const data = await response.json();

  const translated =
    data?.responseData?.translatedText?.trim() || "";

  if (!translated) {
    throw new Error("មិនទទួលបាន Khmer Translation");
  }

  return translated;
}

/* =========================
   AUDIO
========================= */

function resampleAudio(input, fromRate, toRate) {
  if (fromRate === toRate) return input;

  const ratio = fromRate / toRate;
  const newLength = Math.round(input.length / ratio);
  const output = new Float32Array(newLength);

  for (let i = 0; i < newLength; i++) {
    const position = i * ratio;

    const left = Math.floor(position);
    const right = Math.min(
      left + 1,
      input.length - 1
    );

    const weight = position - left;

    output[i] =
      input[left] * (1 - weight) +
      input[right] * weight;
  }

  return output;
}

async function videoToAudio(videoFile) {
  const arrayBuffer =
    await videoFile.arrayBuffer();

  const audioContext =
    new AudioContext();

  const audioBuffer =
    await audioContext.decodeAudioData(
      arrayBuffer
    );

  const channels =
    audioBuffer.numberOfChannels;

  const length =
    audioBuffer.length;

  const mono =
    new Float32Array(length);

  for (
    let channel = 0;
    channel < channels;
    channel++
  ) {
    const data =
      audioBuffer.getChannelData(
        channel
      );

    for (
      let i = 0;
      i < length;
      i++
    ) {
      mono[i] +=
        data[i] / channels;
    }
  }

  const audio16k =
    resampleAudio(
      mono,
      audioBuffer.sampleRate,
      16000
    );

  await audioContext.close();

  return audio16k;
}

/* =========================
   KHMER TTS ONNX
========================= */

const TTS_BASE =
  "https://huggingface.co/sengtha/khmer-tts-female-v2/resolve/main";

async function loadKhmerTTS(setStatus) {
  if (ttsSession && ttsMeta) {
    return {
      session: ttsSession,
      meta: ttsMeta,
    };
  }

  setStatus(
    "🗣️ កំពុងទាញ Khmer Voice model..."
  );

  const metaResponse =
    await fetch(
      `${TTS_BASE}/tts_meta.json`
    );

  if (!metaResponse.ok) {
    throw new Error(
      "មិនអាចទាញ Khmer TTS metadata"
    );
  }

  ttsMeta =
    await metaResponse.json();

  setStatus(
    "🗣️ កំពុងផ្ទុក Khmer Voice..."
  );

  const modelUrl =
    `${TTS_BASE}/khmer_tts_v3.onnx`;

  ttsSession =
    await ort.InferenceSession.create(
      modelUrl,
      {
        executionProviders: ["wasm"],
      }
    );

  return {
    session: ttsSession,
    meta: ttsMeta,
  };
}

function textToIds(text, meta) {
  const vocab =
    meta.vocab || [];

  const idOf =
    new Map(
      vocab.map(
        (char, index) => [
          char,
          index,
        ]
      )
    );

  const blankId =
    idOf.get(meta.blank);

  const ids = [];

  for (const char of text) {
    if (idOf.has(char)) {
      ids.push(
        idOf.get(char)
      );
    }
  }

  if (meta.add_blank) {
    const output = [
      blankId,
    ];

    for (const id of ids) {
      output.push(id);
      output.push(blankId);
    }

    return output;
  }

  return ids;
}

async function synthesizeKhmer(
  text,
  setStatus
) {
  const {
    session,
    meta,
  } = await loadKhmerTTS(
    setStatus
  );

  const ids =
    textToIds(
      text,
      meta
    );

  if (!ids.length) {
    return new Float32Array(0);
  }

  const inputIds =
    new BigInt64Array(
      ids.map((x) =>
        BigInt(x)
      )
    );

  const lengths =
    new BigInt64Array([
      BigInt(ids.length),
    ]);

  const feeds = {};

  feeds[
    session.inputNames[0]
  ] =
    new ort.Tensor(
      "int64",
      inputIds,
      [1, ids.length]
    );

  feeds[
    session.inputNames[1]
  ] =
    new ort.Tensor(
      "int64",
      lengths,
      [1]
    );

  const output =
    await session.run(
      feeds
    );

  const outputTensor =
    output[
      session.outputNames[0]
    ];

  return new Float32Array(
    outputTensor.data
  );
}

/* =========================
   FIT TTS TO TIMING
========================= */

function fitAudioToDuration(
  audio,
  sampleRate,
  duration
) {
  if (!audio.length) {
    return audio;
  }

  const targetLength =
    Math.max(
      1,
      Math.floor(
        duration *
          sampleRate
      )
    );

  if (
    audio.length ===
    targetLength
  ) {
    return audio;
  }

  const output =
    new Float32Array(
      targetLength
    );

  for (
    let i = 0;
    i < targetLength;
    i++
  ) {
    const position =
      i *
      (audio.length - 1) /
      Math.max(
        1,
        targetLength - 1
      );

    const left =
      Math.floor(position);

    const right =
      Math.min(
        left + 1,
        audio.length - 1
      );

    const weight =
      position - left;

    output[i] =
      audio[left] *
        (1 - weight) +
      audio[right] *
        weight;
  }

  return output;
}

/* =========================
   WAV
========================= */

function floatToWav(
  samples,
  sampleRate
) {
  const buffer =
    new ArrayBuffer(
      44 +
      samples.length * 2
    );

  const view =
    new DataView(buffer);

  function writeString(
    offset,
    value
  ) {
    for (
      let i = 0;
      i < value.length;
      i++
    ) {
      view.setUint8(
        offset + i,
        value.charCodeAt(i)
      );
    }
  }

  writeString(
    0,
    "RIFF"
  );

  view.setUint32(
    4,
    36 +
      samples.length * 2,
    true
  );

  writeString(
    8,
    "WAVE"
  );

  writeString(
    12,
    "fmt "
  );

  view.setUint32(
    16,
    16,
    true
  );

  view.setUint16(
    20,
    1,
    true
  );

  view.setUint16(
    22,
    1,
    true
  );

  view.setUint32(
    24,
    sampleRate,
    true
  );

  view.setUint32(
    28,
    sampleRate * 2,
    true
  );

  view.setUint16(
    32,
    2,
    true
  );

  view.setUint16(
    34,
    16,
    true
  );

  writeString(
    36,
    "data"
  );

  view.setUint32(
    40,
    samples.length * 2,
    true
  );

  let offset = 44;

  for (
    let i = 0;
    i < samples.length;
    i++
  ) {
    const sample =
      Math.max(
        -1,
        Math.min(
          1,
          samples[i]
        )
      );

    view.setInt16(
      offset,
      sample < 0
        ? sample * 0x8000
        : sample * 0x7fff,
      true
    );

    offset += 2;
  }

  return new Uint8Array(
    buffer
  );
}

/* =========================
   VIDEO DURATION
========================= */

function getVideoDuration(
  file
) {
  return new Promise(
    (resolve, reject) => {
      const url =
        URL.createObjectURL(
          file
        );

      const element =
        document.createElement(
          "video"
        );

      element.preload =
        "metadata";

      element.onloadedmetadata =
        () => {
          const duration =
            element.duration;

          URL.revokeObjectURL(
            url
          );

          resolve(duration);
        };

      element.onerror = () => {
        URL.revokeObjectURL(
          url
        );

        reject(
          new Error(
            "មិនអាចអាន video duration"
          )
        );
      };

      element.src = url;
    }
  );
}

/* =========================
   FFMPEG
========================= */

async function getFFmpeg(
  setStatus
) {
  if (ffmpegInstance) {
    return ffmpegInstance;
  }

  setStatus(
    "🎬 កំពុងផ្ទុក FFmpeg..."
  );

  const ffmpeg =
    new FFmpeg();

  ffmpeg.on(
    "log",
    ({ message }) => {
      console.log(
        "FFmpeg:",
        message
      );
    }
  );

  const baseURL =
    "https://cdn.jsdelivr.net/npm/@ffmpeg/core@0.12.10/dist/esm";

  await ffmpeg.load({
    coreURL:
      await toBlobURL(
        `${baseURL}/ffmpeg-core.js`,
        "text/javascript"
      ),

    wasmURL:
      await toBlobURL(
        `${baseURL}/ffmpeg-core.wasm`,
        "application/wasm"
      ),
  });

  ffmpegInstance =
    ffmpeg;

  return ffmpeg;
}

/* =========================
   CREATE DUBBING
========================= */

async function createDubbingVideo(
  videoFile,
  caption,
  setStatus
) {
  if (!caption.length) {
    throw new Error(
      "មិនមាន Caption សម្រាប់ Dubbing"
    );
  }

  const duration =
    await getVideoDuration(
      videoFile
    );

  const sampleRate =
    22050;

  const totalSamples =
    Math.ceil(
      duration *
        sampleRate
    );

  const master =
    new Float32Array(
      totalSamples
    );

  for (
    let i = 0;
    i < caption.length;
    i++
  ) {
    const item =
      caption[i];

    const khmer =
      item.khmer?.trim();

    if (
      !khmer ||
      khmer.includes(
        "⚠️ បកប្រែមិនបាន"
      )
    ) {
      continue;
    }

    setStatus(
      `🗣️ បង្កើតសំឡេងខ្មែរ ${i + 1}/${caption.length}...`
    );

    const pcm =
      await synthesizeKhmer(
        khmer,
        setStatus
      );

    if (!pcm.length) {
      continue;
    }

    const start =
      Math.max(
        0,
        Number(item.start) || 0
      );

    const end =
      Math.max(
        start + 0.35,
        Number(item.end) ||
          start + 1
      );

    const targetDuration =
      Math.max(
        0.35,
        end - start
      );

    const fitted =
      fitAudioToDuration(
        pcm,
        sampleRate,
        targetDuration
      );

    const startSample =
      Math.floor(
        start *
          sampleRate
      );

    for (
      let j = 0;
      j < fitted.length;
      j++
    ) {
      const index =
        startSample + j;

      if (
        index >=
        master.length
      ) {
        break;
      }

      master[index] +=
        fitted[j];
    }
  }

  /* Prevent clipping */

  let peak = 0;

  for (
    let i = 0;
    i < master.length;
    i++
  ) {
    peak =
      Math.max(
        peak,
        Math.abs(
          master[i]
        )
      );
  }

  if (peak > 0.95) {
    const scale =
      0.95 / peak;

    for (
      let i = 0;
      i < master.length;
      i++
    ) {
      master[i] *=
        scale;
    }
  }

  setStatus(
    "🎵 កំពុងបញ្ចូល Khmer Voice ជាមួយសំឡេងដើម..."
  );

  const wav =
    floatToWav(
      master,
      sampleRate
    );

  const ffmpeg =
    await getFFmpeg(
      setStatus
    );

  const extension =
    videoFile.name
      .split(".")
      .pop()
      ?.toLowerCase() ||
    "mp4";

  const inputName =
    `input.${extension}`;

  const voiceName =
    "khmer_voice.wav";

  const outputName =
    "smey-khmer-dubbing.mp4";

  await ffmpeg.writeFile(
    inputName,
    await fetchFile(
      videoFile
    )
  );

  await ffmpeg.writeFile(
    voiceName,
    wav
  );

  setStatus(
    "🎬 កំពុងបង្កើត MP4..."
  );

  try {
    /*
      Original audio = 30%
      Khmer voice = 100%
      Original music/audio is preserved.
    */

    await ffmpeg.exec([
      "-i",
      inputName,

      "-i",
      voiceName,

      "-filter_complex",
      "[0:a]volume=0.30[original];[1:a]volume=1.00[voice];[original][voice]amix=inputs=2:duration=first:dropout_transition=0[a]",

      "-map",
      "0:v:0",

      "-map",
      "[a]",

      "-c:v",
      "copy",

      "-c:a",
      "aac",

      "-b:a",
      "192k",

      "-shortest",

      outputName,
    ]);
  } catch (error) {
    console.warn(
      "Video copy failed. Trying H264..."
    );

    await ffmpeg.exec([
      "-i",
      inputName,

      "-i",
      voiceName,

      "-filter_complex",
      "[0:a]volume=0.30[original];[1:a]volume=1.00[voice];[original][voice]amix=inputs=2:duration=first:dropout_transition=0[a]",

      "-map",
      "0:v:0",

      "-map",
      "[a]",

      "-c:v",
      "libx264",

      "-preset",
      "ultrafast",

      "-crf",
      "28",

      "-c:a",
      "aac",

      "-b:a",
      "192k",

      "-shortest",

      outputName,
    ]);
  }

  const output =
    await ffmpeg.readFile(
      outputName
    );

  const blob =
    new Blob(
      [output.buffer],
      {
        type:
          "video/mp4",
      }
    );

  const url =
    URL.createObjectURL(
      blob
    );

  const link =
    document.createElement(
      "a"
    );

  link.href = url;
  link.download =
    "smey-khmer-dubbing.mp4";

  document.body.appendChild(
    link
  );

  link.click();

  link.remove();

  setTimeout(
    () =>
      URL.revokeObjectURL(
        url
      ),
    30000
  );

  setStatus(
    "✅ Khmer Dubbing MP4 រួចរាល់ — Download បានហើយ!"
  );

  return blob;
}

/* =========================
   TIME
========================= */

function formatTime(seconds) {
  if (
    seconds == null ||
    Number.isNaN(seconds)
  ) {
    return "00:00";
  }

  const total =
    Math.max(
      0,
      Math.floor(seconds)
    );

  const minutes =
    Math.floor(
      total / 60
    );

  const secs =
    total % 60;

  return `${String(
    minutes
  ).padStart(
    2,
    "0"
  )}:${String(
    secs
  ).padStart(
    2,
    "0"
  )}`;
}

/* =========================
   APP
========================= */

function App() {
  const [video, setVideo] =
    useState(null);

  const [videoUrl, setVideoUrl] =
    useState("");

  const [status, setStatus] =
    useState(
      "រង់ចាំវីដេអូ..."
    );

  const [busy, setBusy] =
    useState(false);

  const [dubbingBusy, setDubbingBusy] =
    useState(false);

  const [text, setText] =
    useState("");

  const [
    translatedText,
    setTranslatedText,
  ] = useState("");

  const [caption, setCaption] =
    useState([]);

  const [sourceLang, setSourceLang] =
    useState(
      "eng_Latn"
    );

  useEffect(() => {
    if (!video) {
      setVideoUrl("");
      return;
    }

    const url =
      URL.createObjectURL(
        video
      );

    setVideoUrl(url);

    return () =>
      URL.revokeObjectURL(
        url
      );
  }, [video]);

  function handleVideo(event) {
    const file =
      event.target.files?.[0];

    if (!file) return;

    setVideo(file);

    setText("");
    setTranslatedText("");
    setCaption([]);

    setStatus(
      `បានជ្រើស៖ ${file.name}`
    );
  }

  /* =========================
     WHISPER + TRANSLATION
  ========================= */

  async function start() {
    if (!video) {
      setStatus(
        "សូមបញ្ចូលវីដេអូជាមុនសិន។"
      );

      return;
    }

    setBusy(true);

    setText("");
    setTranslatedText("");
    setCaption([]);

    try {
      setStatus(
        "🎧 កំពុងអានសំឡេងពីវីដេអូ..."
      );

      const audio =
        await videoToAudio(
          video
        );

      if (
        !(audio instanceof
          Float32Array)
      ) {
        throw new Error(
          "Audio format មិនត្រឹមត្រូវ"
        );
      }

      if (!audio.length) {
        throw new Error(
          "មិនមានសំឡេងក្នុងវីដេអូ"
        );
      }

      setStatus(
        "🤖 កំពុងដំណើរការ Whisper Tiny..."
      );

      const whisper =
        await getTranscriber(
          setStatus
        );

      const result =
        await whisper(
          audio,
          {
            return_timestamps:
              true,

            chunk_length_s:
              30,

            stride_length_s:
              5,
          }
        );

      const whisperText =
        result.text?.trim() ||
        "";

      const chunks =
        (result.chunks || [])
          .map(
            (item) => ({
              start:
                item.timestamp?.[0] ??
                0,

              end:
                item.timestamp?.[1] ??
                0,

              text:
                item.text?.trim() ||
                "",

              khmer: "",
            })
          )
          .filter(
            (item) =>
              item.text
          );

      setText(
        whisperText
      );

      setCaption(
        chunks
      );

      if (!whisperText) {
        setStatus(
          "⚠️ Whisper មិនរកឃើញសំឡេងនិយាយទេ។"
        );

        return;
      }

      setStatus(
        "🇰🇭 កំពុងបកប្រែទៅខ្មែរ..."
      );

      const translatedParts =
        [];

      for (
        let i = 0;
        i < chunks.length;
        i++
      ) {
        setStatus(
          `🇰🇭 បកប្រែទៅខ្មែរ ${i + 1}/${chunks.length}...`
        );

        let khmer =
          "";

        try {
          khmer =
            await translateText(
              chunks[i].text,
              sourceLang
            );
        } catch (error) {
          console.error(
            error
          );

          khmer =
            "⚠️ បកប្រែមិនបាន";
        }

        translatedParts.push(
          khmer
        );

        setCaption(
          (old) =>
            old.map(
              (
                item,
                index
              ) =>
                index === i
                  ? {
                      ...item,
                      khmer,
                    }
                  : item
            )
        );
      }

      setTranslatedText(
        translatedParts
          .filter(Boolean)
          .join(" ")
      );

      setStatus(
        `✅ Caption + Khmer Translation រួចរាល់ — ${chunks.length} ផ្នែក`
      );
    } catch (error) {
      console.error(
        error
      );

      setStatus(
        `❌ មានបញ្ហា៖ ${
          error?.message ||
          "Unknown error"
        }`
      );
    } finally {
      setBusy(false);
    }
  }

  /* =========================
     DUBBING BUTTON
  ========================= */

  async function handleDubbing() {
    if (!video) {
      setStatus(
        "សូមបញ្ចូលវីដេអូជាមុនសិន។"
      );

      return;
    }

    if (!caption.length) {
      setStatus(
        "សូមចុច ចាប់ផ្ដើម ជាមុនសិន។"
      );

      return;
    }

    setDubbingBusy(true);

    try {
      await createDubbingVideo(
        video,
        caption,
        setStatus
      );
    } catch (error) {
      console.error(
        error
      );

      setStatus(
        `❌ Dubbing error៖ ${
          error?.message ||
          "Unknown error"
        }`
      );
    } finally {
      setDubbingBusy(false);
    }
  }

  return (
    <main className="app">
      <section className="card">

        <h1>
          🇰🇭 Smey Auto Caption V2
        </h1>

        <p className="subtitle">
          Browser AI • Whisper • Khmer Caption • Translation • Khmer Voice • MP4
        </p>

        <label className="upload">
          <span>
            🎥 បញ្ចូលវីដេអូ
          </span>

          <input
            type="file"
            accept="video/mp4,video/mov,video/webm,video/mkv"
            onChange={
              handleVideo
            }
          />
        </label>

        {video && (
          <>
            <div className="file">
              <strong>
                📁 {video.name}
              </strong>

              <small>
                {(
                  video.size /
                  1024 /
                  1024
                ).toFixed(
                  1
                )}{" "}
                MB
              </small>
            </div>

            {videoUrl && (
              <video
                src={videoUrl}
                controls
                playsInline
                style={{
                  width:
                    "100%",
                  marginTop:
                    "16px",
                  borderRadius:
                    "14px",
                  display:
                    "block",
                  background:
                    "#000",
                }}
              />
            )}
          </>
        )}

        <div
          style={{
            marginTop:
              "14px",
            padding:
              "12px",
            borderRadius:
              "12px",
            background:
              "#334155",
          }}
        >
          <strong>
            🌐 ភាសាសំឡេងដើម
          </strong>

          <select
            value={
              sourceLang
            }
            onChange={(e) =>
              setSourceLang(
                e.target
                  .value
              )
            }
            disabled={
              busy ||
              dubbingBusy
            }
            style={{
              width:
                "100%",
              marginTop:
                "8px",
              padding:
                "10px",
              borderRadius:
                "10px",
              fontSize:
                "16px",
            }}
          >
            <option value="eng_Latn">
              English
            </option>

            <option value="zho_Hans">
              Chinese 简体中文
            </option>
          </select>
        </div>

        <button
          className="start"
          onClick={start}
          disabled={
            busy ||
            dubbingBusy
          }
        >
          {busy
            ? "⏳ កំពុងដំណើរការ..."
            : "▶️ ចាប់ផ្ដើម"}
        </button>

        {caption.length >
          0 && (
          <button
            className="start"
            onClick={
              handleDubbing
            }
            disabled={
              busy ||
              dubbingBusy
            }
            style={{
              marginTop:
                "10px",
            }}
          >
            {dubbingBusy
              ? "🎬 កំពុងបង្កើត MP4..."
              : "🗣️ បង្កើត Khmer Dubbing MP4"}
          </button>
        )}

        <div className="status">
          {status}
        </div>

        {text && (
          <div
            style={{
              marginTop:
                "16px",
              padding:
                "16px",
              borderRadius:
                "14px",
              background:
                "#0f172a",
              lineHeight:
                "1.8",
            }}
          >
            <strong>
              📝 Whisper Caption:
            </strong>

            <div
              style={{
                marginTop:
                  "10px",
              }}
            >
              {text}
            </div>
          </div>
        )}

        {translatedText && (
          <div
            style={{
              marginTop:
                "16px",
              padding:
                "16px",
              borderRadius:
                "14px",
              background:
                "#0f172a",
              lineHeight:
                "1.8",
            }}
          >
            <strong>
              🇰🇭 Khmer Translation:
            </strong>

            <div
              style={{
                marginTop:
                  "10px",
              }}
            >
              {
                translatedText
              }
            </div>
          </div>
        )}

        {caption.length >
          0 && (
          <div
            style={{
              marginTop:
                "16px",
            }}
          >
            <h3>
              ⏱️ Caption Timeline
            </h3>

            {caption.map(
              (
                item,
                index
              ) => (
                <div
                  key={
                    index
                  }
                  style={{
                    marginBottom:
                      "8px",
                    padding:
                      "10px 12px",
                    borderRadius:
                      "10px",
                    background:
                      "#334155",
                  }}
                >
                  <small>
                    {
                      formatTime(
                        item.start
                      )
                    }{" "}
                    →{" "}
                    {
                      formatTime(
                        item.end
                      )
                    }
                  </small>

                  <div
                    style={{
                      marginTop:
                        "4px",
                    }}
                  >
                    {
                      item.text
                    }
                  </div>

                  {item.khmer && (
                    <div
                      style={{
                        marginTop:
                          "6px",
                        fontWeight:
                          "700",
                      }}
                    >
                      🇰🇭{" "}
                      {
                        item.khmer
                      }
                    </div>
                  )}
                </div>
              )
            )}
          </div>
        )}

        <div className="features">
          <div>
            🎙️ Whisper Speech → Caption
          </div>

          <div>
            🇰🇭 Khmer Caption
          </div>

          <div>
            🌐 Translation
          </div>

          <div>
            🗣️ Khmer Voice
          </div>

          <div>
            🎬 Video Sync
          </div>

          <div>
            🎵 Original Audio/Music Preserved
          </div>

          <div>
            📥 MP4 Export
          </div>
        </div>

      </section>
    </main>
  );
}

createRoot(
  document.getElementById(
    "root"
  )
).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
