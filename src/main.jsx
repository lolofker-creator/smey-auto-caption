import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { pipeline } from "@huggingface/transformers";
import "./style.css";

let transcriber = null;
let translator = null;

async function getTranscriber(setStatus) {
  if (transcriber) return transcriber;

  setStatus("⏳ កំពុងទាញ Whisper Tiny...");

  transcriber = await pipeline(
    "automatic-speech-recognition",
    "onnx-community/whisper-tiny"
  );

  return transcriber;
}

async function getTranslator(setStatus) {
  if (translator) return translator;

  setStatus("🌐 កំពុងទាញ Khmer Translation model...");

  translator = await pipeline(
    "translation",
    "Xenova/nllb-200-distilled-600M"
  );

  return translator;
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
  const length = audioBuffer.length;
  const mono = new Float32Array(length);

  for (let channel = 0; channel < channels; channel++) {
    const data = audioBuffer.getChannelData(channel);

    for (let i = 0; i < length; i++) {
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

function formatTime(seconds) {
  if (seconds == null || Number.isNaN(seconds)) {
    return "00:00";
  }

  const total = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(total / 60);
  const secs = total % 60;

  return `${String(minutes).padStart(2, "0")}:${String(
    secs
  ).padStart(2, "0")}`;
}

function App() {
  const [video, setVideo] = useState(null);
  const [videoUrl, setVideoUrl] = useState("");
  const [status, setStatus] = useState("រង់ចាំវីដេអូ...");
  const [busy, setBusy] = useState(false);

  const [text, setText] = useState("");
  const [translatedText, setTranslatedText] = useState("");
  const [caption, setCaption] = useState([]);

  const [sourceLang, setSourceLang] = useState("eng_Latn");

  useEffect(() => {
    if (!video) {
      setVideoUrl("");
      return;
    }

    const url = URL.createObjectURL(video);
    setVideoUrl(url);

    return () => URL.revokeObjectURL(url);
  }, [video]);

  function handleVideo(event) {
    const file = event.target.files?.[0];

    if (!file) return;

    setVideo(file);
    setText("");
    setTranslatedText("");
    setCaption([]);
    setStatus(`បានជ្រើស៖ ${file.name}`);
  }

  async function start() {
    if (!video) {
      setStatus("សូមបញ្ចូលវីដេអូជាមុនសិន។");
      return;
    }

    setBusy(true);
    setText("");
    setTranslatedText("");
    setCaption([]);

    try {
      setStatus("🎧 កំពុងអានសំឡេងពីវីដេអូ...");

      const audio = await videoToAudio(video);

      if (!(audio instanceof Float32Array)) {
        throw new Error("Audio format មិនត្រឹមត្រូវ");
      }

      if (audio.length === 0) {
        throw new Error("មិនមានសំឡេងក្នុងវីដេអូ");
      }

      setStatus("🤖 កំពុងដំណើរការ Whisper Tiny...");

      const whisper = await getTranscriber(setStatus);

      const result = await whisper(audio, {
        return_timestamps: true,
        chunk_length_s: 30,
        stride_length_s: 5,
      });

      const whisperText = result.text?.trim() || "";

      const chunks = (result.chunks || [])
        .map((item) => ({
          start: item.timestamp?.[0] ?? 0,
          end: item.timestamp?.[1] ?? 0,
          text: item.text?.trim() || "",
          khmer: "",
        }))
        .filter((item) => item.text);

      setText(whisperText);
      setCaption(chunks);

      if (!whisperText) {
        setStatus("⚠️ Whisper មិនរកឃើញសំឡេងនិយាយទេ។");
        return;
      }

      setStatus("🌐 កំពុងបកប្រែទៅខ្មែរ...");

      const translate = await getTranslator(setStatus);

      const translatedParts = [];

      for (let i = 0; i < chunks.length; i++) {
        setStatus(
          `🌐 បកប្រែទៅខ្មែរ ${i + 1}/${chunks.length}...`
        );

        const resultKhmer = await translate(chunks[i].text, {
          src_lang: sourceLang,
          tgt_lang: "khm_Khmr",
        });

        const khmer =
          resultKhmer?.[0]?.translation_text?.trim() || "";

        translatedParts.push(khmer);

        setCaption((old) =>
          old.map((item, index) =>
            index === i
              ? { ...item, khmer }
              : item
          )
        );
      }

      setTranslatedText(
        translatedParts.filter(Boolean).join(" ")
      );

      setStatus(
        `✅ Caption + Khmer Translation រួចរាល់ — ${chunks.length} ផ្នែក`
      );
    } catch (error) {
      console.error(error);

      setStatus(
        `❌ មានបញ្ហា៖ ${
          error?.message || "Unknown error"
        }`
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="app">
      <section className="card">
        <h1>🇰🇭 Smey Auto Caption V2</h1>

        <p className="subtitle">
          Browser AI • Whisper • Khmer Caption • Translation • Khmer Voice
        </p>

        <label className="upload">
          <span>🎥 បញ្ចូលវីដេអូ</span>

          <input
            type="file"
            accept="video/mp4,video/mov,video/webm,video/mkv"
            onChange={handleVideo}
          />
        </label>

        {video && (
          <>
            <div className="file">
              <strong>📁 {video.name}</strong>

              <small>
                {(video.size / 1024 / 1024).toFixed(1)} MB
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
                  display: "block",
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
          <strong>🌐 ភាសាសំឡេងដើម</strong>

          <select
            value={sourceLang}
            onChange={(e) => setSourceLang(e.target.value)}
            disabled={busy}
            style={{
              width: "100%",
              marginTop: "8px",
              padding: "10px",
              borderRadius: "10px",
              fontSize: "16px",
            }}
          >
            <option value="eng_Latn">English</option>
            <option value="zho_Hans">
              Chinese 简体中文
            </option>
          </select>
        </div>

        <button
          className="start"
          onClick={start}
          disabled={busy}
        >
          {busy
            ? "⏳ កំពុងដំណើរការ..."
            : "▶️ ចាប់ផ្ដើម"}
        </button>

        <div className="status">
          {status}
        </div>

        {text && (
          <div
            style={{
              marginTop: "16px",
              padding: "16px",
              borderRadius: "14px",
              background: "#0f172a",
              lineHeight: "1.8",
            }}
          >
            <strong>📝 Whisper Caption:</strong>

            <div style={{ marginTop: "10px" }}>
              {text}
            </div>
          </div>
        )}

        {translatedText && (
          <div
            style={{
              marginTop: "16px",
              padding: "16px",
              borderRadius: "14px",
              background: "#0f172a",
              lineHeight: "1.8",
            }}
          >
            <strong>🇰🇭 Khmer Translation:</strong>

            <div style={{ marginTop: "10px" }}>
              {translatedText}
            </div>
          </div>
        )}

        {caption.length > 0 && (
          <div style={{ marginTop: "16px" }}>
            <h3>⏱️ Caption Timeline</h3>

            {caption.map((item, index) => (
              <div
                key={index}
                style={{
                  marginBottom: "8px",
                  padding: "10px 12px",
                  borderRadius: "10px",
                  background: "#334155",
                }}
              >
                <small>
                  {formatTime(item.start)} →{" "}
                  {formatTime(item.end)}
                </small>

                <div style={{ marginTop: "4px" }}>
                  {item.text}
                </div>

                {item.khmer && (
                  <div
                    style={{
                      marginTop: "6px",
                      fontWeight: "700",
                    }}
                  >
                    🇰🇭 {item.khmer}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        <div className="features">
          <div>🎙️ Whisper Speech → Caption</div>
          <div>🇰🇭 Khmer Caption</div>
          <div>🌐 Translation</div>
          <div>🗣️ Khmer Voice</div>
          <div>🎬 Video Sync</div>
          <div>📥 MP4 Export</div>
        </div>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
