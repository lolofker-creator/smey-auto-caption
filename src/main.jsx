import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { pipeline } from "@huggingface/transformers";
import "./style.css";

let transcriber = null;

async function getTranscriber(setStatus) {
  if (transcriber) return transcriber;

  setStatus("⏳ កំពុងទាញ Whisper Tiny លើកដំបូង...");

  transcriber = await pipeline(
    "automatic-speech-recognition",
    "onnx-community/whisper-tiny"
  );

  return transcriber;
}

async function videoToAudio(videoFile) {
  const arrayBuffer = await videoFile.arrayBuffer();

  const audioContext = new AudioContext();
  const source = audioContext.createBufferSource();

  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

  source.buffer = audioBuffer;

  const channels = audioBuffer.numberOfChannels;
  const length = audioBuffer.length;

  const mono = new Float32Array(length);

  for (let ch = 0; ch < channels; ch++) {
    const data = audioBuffer.getChannelData(ch);

    for (let i = 0; i < length; i++) {
      mono[i] += data[i] / channels;
    }
  }

  await audioContext.close();

  return {
    audio: mono,
    sampling_rate: audioBuffer.sampleRate,
  };
}

function formatTime(seconds) {
  if (seconds == null || Number.isNaN(seconds)) {
    return "00:00";
  }

  const total = Math.floor(seconds);
  const minutes = Math.floor(total / 60);
  const secs = total % 60;

  return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(
    2,
    "0"
  )}`;
}

function App() {
  const [video, setVideo] = useState(null);
  const [videoUrl, setVideoUrl] = useState("");
  const [status, setStatus] = useState("រង់ចាំវីដេអូ...");
  const [busy, setBusy] = useState(false);
  const [caption, setCaption] = useState([]);
  const [text, setText] = useState("");

  useEffect(() => {
    if (!video) {
      setVideoUrl("");
      return;
    }

    const url = URL.createObjectURL(video);
    setVideoUrl(url);

    return () => URL.revokeObjectURL(url);
  }, [video]);

  function handleVideo(e) {
    const file = e.target.files?.[0];

    if (!file) return;

    setVideo(file);
    setCaption([]);
    setText("");
    setStatus(`បានជ្រើស៖ ${file.name}`);
  }

  async function start() {
    if (!video) {
      setStatus("សូមបញ្ចូលវីដេអូជាមុនសិន។");
      return;
    }

    setBusy(true);
    setCaption([]);
    setText("");

    try {
      setStatus("🎧 កំពុងយកសំឡេងចេញពីវីដេអូ...");

      const audio = await videoToAudio(video);

      setStatus("🤖 កំពុងដំណើរការ Whisper Tiny...");

      const pipe = await getTranscriber(setStatus);

      const result = await pipe(audio, {
        return_timestamps: true,
        chunk_length_s: 30,
        stride_length_s: 5,
      });

      const chunks = (result.chunks || []).map((item) => ({
        start: item.timestamp?.[0] ?? 0,
        end: item.timestamp?.[1] ?? 0,
        text: item.text?.trim() || "",
      }));

      setCaption(chunks);
      setText(result.text?.trim() || "");

      if (chunks.length > 0) {
        setStatus(`✅ Caption រួចរាល់ ${chunks.length} ផ្នែក`);
      } else {
        setStatus("⚠️ Whisper មិនរកឃើញសំឡេងនិយាយទេ។");
      }
    } catch (error) {
      console.error(error);
      setStatus(`❌ មានបញ្ហា៖ ${error.message}`);
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

        <button
          className="start"
          onClick={start}
          disabled={busy}
        >
          {busy ? "⏳ កំពុងដំណើរការ..." : "▶️ ចាប់ផ្ដើម"}
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
            <strong>📝 អត្ថបទដែល Whisper ស្គាល់៖</strong>

            <div style={{ marginTop: "10px" }}>
              {text}
            </div>
          </div>
        )}

        {caption.length > 0 && (
          <div style={{ marginTop: "16px" }}>
            <h3>🇰🇭 Caption Timeline</h3>

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
                  {formatTime(item.start)} → {formatTime(item.end)}
                </small>

                <div style={{ marginTop: "4px" }}>
                  {item.text}
                </div>
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
