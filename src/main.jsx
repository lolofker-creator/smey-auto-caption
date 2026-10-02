import React, { useEffect, useState } from "react";
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

async function getTranscriber(setStatus) {
  if (transcriber) return transcriber;

  setStatus("⏳ កំពុងទាញ Whisper Tiny...");

  transcriber = await pipeline(
    "automatic-speech-recognition",
    "onnx-community/whisper-tiny"
  );

  return transcriber;
}

async function translateText(text, sourceLang) {
  if (!text?.trim()) return "";

  const source =
    sourceLang === "zho_Hans" ? "zh-CN" : "en";

  const params = new URLSearchParams({
    q: text.trim(),
    langpair: `${source}|km-KM`,
    mt: "1",
  });

  const response = await fetch(
    `https://api.mymemory.translated.net/get?${params.toString()}`
  );

  if (!response.ok) {
    throw new Error(`Translation API error ${response.status}`);
  }

  const data = await response.json();

  return data?.responseData?.translatedText?.trim() || "";
}

function resampleAudio(input, fromRate, toRate) {
  if (fromRate === toRate) return input;

  const ratio = fromRate / toRate;
  const newLength = Math.round(input.length / ratio);
  const output = new Float32Array(newLength);

  for (let i = 0; i < newLength; i++) {
    const position = i * ratio;
    const left = Math.floor(position);
    const right = Math.min(left + 1, input.length - 1);
    const weight = position - left;

    output[i] =
      input[left] * (1 - weight) +
      input[right] * weight;
  }

  return output;
}

async function videoToAudio(videoFile) {
  const arrayBuffer = await videoFile.arrayBuffer();
  const audioContext = new AudioContext();
  const audioBuffer =
    await audioContext.decodeAudioData(arrayBuffer);

  const channels = audioBuffer.numberOfChannels;
  const mono = new Float32Array(audioBuffer.length);

  for (let channel = 0; channel < channels; channel++) {
    const data = audioBuffer.getChannelData(channel);

    for (let i = 0; i < mono.length; i++) {
      mono[i] += data[i] / channels;
    }
  }

  const audio16k = resampleAudio(
    mono,
    audioBuffer.sampleRate,
    16000
  );

  await audioContext.close();
  return audio16k;
}

/* =========================
   KHMER TTS
========================= */

const TTS_BASE =
  "https://huggingface.co/sengtha/khmer-tts-female-v2/resolve/main";

async function loadKhmerTTS(setStatus) {
  if (ttsSession && ttsMeta) {
    return { session: ttsSession, meta: ttsMeta };
  }

  setStatus("🗣️ កំពុងទាញ Khmer Voice...");

  const metaResponse =
    await fetch(`${TTS_BASE}/tts_meta.json`);

  if (!metaResponse.ok) {
    throw new Error("មិនអាចទាញ Khmer TTS metadata");
  }

  ttsMeta = await metaResponse.json();

  ttsSession =
    await ort.InferenceSession.create(
      `${TTS_BASE}/khmer_tts_v3.onnx`,
      { executionProviders: ["wasm"] }
    );

  return {
    session: ttsSession,
    meta: ttsMeta,
  };
}

function textToIds(text, meta) {
  const vocab = meta.vocab || [];

  const idOf = new Map(
    vocab.map((char, index) => [char, index])
  );

  const ids = [];

  for (const char of text) {
    if (idOf.has(char)) {
      ids.push(idOf.get(char));
    }
  }

  if (!meta.add_blank) return ids;

  const blankId = idOf.get(meta.blank);
  const output = [blankId];

  for (const id of ids) {
    output.push(id);
    output.push(blankId);
  }

  return output;
}

async function synthesizeKhmer(text, setStatus) {
  const { session, meta } =
    await loadKhmerTTS(setStatus);

  const ids = textToIds(text, meta);

  if (!ids.length) {
    return new Float32Array(0);
  }

  const inputIds = new BigInt64Array(
    ids.map((x) => BigInt(x))
  );

  const lengths = new BigInt64Array([
    BigInt(ids.length),
  ]);

  const feeds = {};

  feeds[session.inputNames[0]] =
    new ort.Tensor(
      "int64",
      inputIds,
      [1, ids.length]
    );

  feeds[session.inputNames[1]] =
    new ort.Tensor(
      "int64",
      lengths,
      [1]
    );

  const output = await session.run(feeds);

  const tensor =
    output[session.outputNames[0]];

  return new Float32Array(tensor.data);
}

function fitAudioToDuration(
  audio,
  sampleRate,
  duration
) {
  if (!audio.length) return audio;

  const targetLength = Math.max(
    1,
    Math.floor(duration * sampleRate)
  );

  const output =
    new Float32Array(targetLength);

  for (let i = 0; i < targetLength; i++) {
    const position =
      i *
      (audio.length - 1) /
      Math.max(1, targetLength - 1);

    const left = Math.floor(position);
    const right =
      Math.min(left + 1, audio.length - 1);

    const weight = position - left;

    output[i] =
      audio[left] * (1 - weight) +
      audio[right] * weight;
  }

  return output;
}

/* =========================
   WAV
========================= */

function floatToWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(
    44 + samples.length * 2
  );

  const view = new DataView(buffer);

  const writeString = (offset, value) => {
    for (let i = 0; i < value.length; i++) {
      view.setUint8(
        offset + i,
        value.charCodeAt(i)
      );
    }
  };

  writeString(0, "RIFF");
  view.setUint32(
    4,
    36 + samples.length * 2,
    true
  );

  writeString(8, "WAVE");
  writeString(12, "fmt ");

  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);

  writeString(36, "data");

  view.setUint32(
    40,
    samples.length * 2,
    true
  );

  let offset = 44;

  for (let i = 0; i < samples.length; i++) {
    const sample = Math.max(
      -1,
      Math.min(1, samples[i])
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

  return new Uint8Array(buffer);
}

function getVideoDuration(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const video = document.createElement("video");

    video.preload = "metadata";

    video.onloadedmetadata = () => {
      const duration = video.duration;
      URL.revokeObjectURL(url);
      resolve(duration);
    };

    video.onerror = () => {
      URL.revokeObjectURL(url);
      reject(
        new Error("មិនអាចអាន video duration")
      );
    };

    video.src = url;
  });
}

/* =========================
   FFMPEG
========================= */

async function getFFmpeg(setStatus) {
  if (ffmpegInstance) return ffmpegInstance;

  setStatus("🎬 កំពុងផ្ទុក FFmpeg...");

  const ffmpeg = new FFmpeg();

  const baseURL =
    "https://cdn.jsdelivr.net/npm/@ffmpeg/core@0.12.10/dist/esm";

  await ffmpeg.load({
    coreURL: await toBlobURL(
      `${baseURL}/ffmpeg-core.js`,
      "text/javascript"
    ),

    wasmURL: await toBlobURL(
      `${baseURL}/ffmpeg-core.wasm`,
      "application/wasm"
    ),
  });

  ffmpegInstance = ffmpeg;

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
  const duration =
    await getVideoDuration(videoFile);

  const sampleRate = 22050;

  const master =
    new Float32Array(
      Math.ceil(duration * sampleRate)
    );

  for (let i = 0; i < caption.length; i++) {
    const item = caption[i];
    const khmer = item.khmer?.trim();

    if (!khmer) continue;
    if (khmer.startsWith("⚠️")) continue;

    setStatus(
      `🗣️ បង្កើតសំឡេងខ្មែរ ${i + 1}/${caption.length}...`
    );

    const pcm =
      await synthesizeKhmer(
        khmer,
        setStatus
      );

    if (!pcm.length) continue;

    const start =
      Math.max(0, Number(item.start) || 0);

    const end =
      Math.max(
        start + 0.35,
        Number(item.end) || start + 1
      );

    const fitted =
      fitAudioToDuration(
        pcm,
        sampleRate,
        end - start
      );

    const startSample =
      Math.floor(start * sampleRate);

    for (let j = 0; j < fitted.length; j++) {
      const index = startSample + j;

      if (index >= master.length) break;

      master[index] += fitted[j];
    }
  }

  let peak = 0;

  for (const sample of master) {
    peak = Math.max(
      peak,
      Math.abs(sample)
    );
  }

  if (peak > 0.95) {
    const scale = 0.95 / peak;

    for (let i = 0; i < master.length; i++) {
      master[i] *= scale;
    }
  }

  const wav =
    floatToWav(
      master,
      sampleRate
    );

  const ffmpeg =
    await getFFmpeg(setStatus);

  const extension =
    videoFile.name
      .split(".")
      .pop()
      ?.toLowerCase() || "mp4";

  const inputName =
    `input.${extension}`;

  await ffmpeg.writeFile(
    inputName,
    await fetchFile(videoFile)
  );

  await ffmpeg.writeFile(
    "khmer_voice.wav",
    wav
  );

  setStatus("🎬 កំពុងបង្កើត MP4...");

  const outputName =
    "smey-khmer-dubbing.mp4";

  await ffmpeg.exec([
    "-i",
    inputName,

    "-i",
    "khmer_voice.wav",

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

  const output =
    await ffmpeg.readFile(
      outputName
    );

  return new Blob(
    [output.buffer],
    { type: "video/mp4" }
  );
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
    Math.max(0, Math.floor(seconds));

  const minutes =
    Math.floor(total / 60);

  const secs =
    total % 60;

  return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

/* =========================
   REMOVE DUPLICATES
========================= */

function cleanChunks(chunks) {
  const result = [];
  let previous = "";

  for (const item of chunks) {
    const normalized =
      item.text
        .toLowerCase()
        .replace(/\s+/g, " ")
        .trim();

    if (!normalized) continue;

    if (normalized === previous) {
      continue;
    }

    previous = normalized;
    result.push(item);
  }

  return result;
}

/* =========================
   APP
========================= */

function App() {
  const [video, setVideo] = useState(null);
  const [videoUrl, setVideoUrl] = useState("");
  const [status, setStatus] =
    useState("រង់ចាំវីដេអូ...");

  const [busy, setBusy] = useState(false);
  const [dubbingBusy, setDubbingBusy] =
    useState(false);

  const [caption, setCaption] = useState([]);
  const [sourceLang, setSourceLang] =
    useState("eng_Latn");

  const [downloadUrl, setDownloadUrl] =
    useState("");

  useEffect(() => {
    if (!video) {
      setVideoUrl("");
      return;
    }

    const url =
      URL.createObjectURL(video);

    setVideoUrl(url);

    return () =>
      URL.revokeObjectURL(url);
  }, [video]);

  function handleVideo(event) {
    const file =
      event.target.files?.[0];

    if (!file) return;

    if (downloadUrl) {
      URL.revokeObjectURL(downloadUrl);
    }

    setDownloadUrl("");
    setVideo(file);
    setCaption([]);

    setStatus(
      `បានជ្រើស៖ ${file.name}`
    );
  }

  async function start() {
    if (!video) {
      setStatus(
        "សូមបញ្ចូលវីដេអូជាមុនសិន។"
      );
      return;
    }

    setBusy(true);
    setCaption([]);

    try {
      setStatus(
        "🎧 កំពុងអានសំឡេង..."
      );

      const audio =
        await videoToAudio(video);

      setStatus(
        "🤖 កំពុងដំណើរការ Whisper Tiny..."
      );

      const whisper =
        await getTranscriber(setStatus);

      const result =
        await whisper(audio, {
          return_timestamps: true,
          chunk_length_s: 30,
          stride_length_s: 5,
        });

      let chunks =
        (result.chunks || [])
          .map((item) => ({
            start:
              item.timestamp?.[0] ?? 0,

            end:
              item.timestamp?.[1] ?? 0,

            text:
              item.text?.trim() || "",

            khmer: "",
          }))
          .filter(
            (item) => item.text
          );

      chunks = cleanChunks(chunks);

      setCaption(chunks);

      if (!chunks.length) {
        setStatus(
          "⚠️ មិនរកឃើញសំឡេងនិយាយទេ។"
        );
        return;
      }

      for (
        let i = 0;
        i < chunks.length;
        i++
      ) {
        setStatus(
          `🇰🇭 បកប្រែ ${i + 1}/${chunks.length}...`
        );

        try {
          const khmer =
            await translateText(
              chunks[i].text,
              sourceLang
            );

          chunks[i].khmer = khmer;

          setCaption(
            [...chunks]
          );
        } catch {
          chunks[i].khmer =
            "⚠️ បកប្រែមិនបាន";

          setCaption(
            [...chunks]
          );
        }
      }

      setStatus(
        `✅ Caption + Khmer រួចរាល់ — ${chunks.length} ផ្នែក`
      );
    } catch (error) {
      console.error(error);

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

  async function handleDubbing() {
    if (!video || !caption.length) {
      setStatus(
        "សូមបង្កើត Caption ជាមុនសិន។"
      );
      return;
    }

    setDubbingBusy(true);

    try {
      if (downloadUrl) {
        URL.revokeObjectURL(downloadUrl);
        setDownloadUrl("");
      }

      const blob =
        await createDubbingVideo(
          video,
          caption,
          setStatus
        );

      const url =
        URL.createObjectURL(blob);

      setDownloadUrl(url);

      setStatus(
        "✅ MP4 រួចរាល់ — ចុច Download MP4 ខាងក្រោម!"
      );
    } catch (error) {
      console.error(error);

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
          Whisper • Khmer Translation • Khmer Voice • MP4
        </p>

        <label className="upload">
          <span>
            🎥 បញ្ចូលវីដេអូ
          </span>

          <input
            type="file"
            accept="video/mp4,video/mov,video/webm,video/mkv"
            onChange={handleVideo}
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
                ).toFixed(1)} MB
              </small>
            </div>

            {videoUrl && (
              <video
                src={videoUrl}
                controls
                playsInline
                style={{
                  width: "100%",
                  marginTop: "16px",
                  borderRadius: "14px",
                  background: "#000",
                }}
              />
            )}
          </>
        )}

        <div
          style={{
            marginTop: "14px",
            padding: "12px",
            borderRadius: "12px",
            background: "#334155",
          }}
        >
          <strong>
            🌐 ភាសាសំឡេងដើម
          </strong>

          <select
            value={sourceLang}
            onChange={(e) =>
              setSourceLang(e.target.value)
            }
            disabled={busy || dubbingBusy}
            style={{
              width: "100%",
              marginTop: "8px",
              padding: "10px",
              borderRadius: "10px",
              fontSize: "16px",
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
          disabled={busy || dubbingBusy}
        >
          {busy
            ? "⏳ កំពុងដំណើរការ..."
            : "▶️ ចាប់ផ្ដើម"}
        </button>

        {caption.length > 0 && (
          <button
            className="start"
            onClick={handleDubbing}
            disabled={busy || dubbingBusy}
            style={{
              marginTop: "10px",
            }}
          >
            {dubbingBusy
              ? "🎬 កំពុងបង្កើត MP4..."
              : "🗣️ បង្កើត Khmer Dubbing MP4"}
          </button>
        )}

        {downloadUrl && (
          <a
            href={downloadUrl}
            download="smey-khmer-dubbing.mp4"
            style={{
              display: "block",
              marginTop: "10px",
              padding: "14px",
              borderRadius: "12px",
              background: "#16a34a",
              color: "#fff",
              textAlign: "center",
              textDecoration: "none",
              fontWeight: "800",
              fontSize: "17px",
            }}
          >
            📥 Download MP4
          </a>
        )}

        <div className="status">
          {status}
        </div>

        {caption.length > 0 && (
          <div style={{ marginTop: "18px" }}>
            <h3>
              ⏱️ Caption Timeline
            </h3>

            {caption.map(
              (item, index) => (
                <div
                  key={index}
                  style={{
                    marginBottom: "10px",
                    padding: "13px",
                    borderRadius: "12px",
                    background: "#334155",
                  }}
                >
                  <small
                    style={{
                      opacity: 0.8,
                    }}
                  >
                    {formatTime(item.start)}
                    {" → "}
                    {formatTime(item.end)}
                  </small>

                  <div
                    style={{
                      marginTop: "7px",
                      fontWeight: "600",
                    }}
                  >
                    {item.text}
                  </div>

                  {item.khmer && (
                    <div
                      style={{
                        marginTop: "8px",
                        paddingTop: "8px",
                        borderTop:
                          "1px solid rgba(255,255,255,.15)",
                        fontWeight: "700",
                      }}
                    >
                      🇰🇭 {item.khmer}
                    </div>
                  )}
                </div>
              )
            )}
          </div>
        )}

        <div className="features">
          <div>🎙️ Whisper Speech → Caption</div>
          <div>🇰🇭 Khmer Translation</div>
          <div>🗣️ Khmer Voice</div>
          <div>🎬 Timing Sync</div>
          <div>🎵 Original Audio/Music</div>
          <div>📥 MP4 Download</div>
        </div>

      </section>
    </main>
  );
}

createRoot(
  document.getElementById("root")
).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
