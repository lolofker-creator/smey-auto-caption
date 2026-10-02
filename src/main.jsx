import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

function App() {
  const [video, setVideo] = useState(null);
  const [status, setStatus] = useState("រង់ចាំវីដេអូ...");
  const [busy, setBusy] = useState(false);

  function handleVideo(e) {
    const file = e.target.files?.[0];
    if (!file) return;

    setVideo(file);
    setStatus(`បានជ្រើស៖ ${file.name}`);
  }

  async function start() {
    if (!video) {
      setStatus("សូមបញ្ចូលវីដេអូជាមុនសិន។");
      return;
    }

    setBusy(true);
    setStatus("កំពុងរៀបចំ Browser AI...");

    /*
      V2 browser-processing engine នឹងដំណើរការនៅទីនេះ។
      វាមិនប្រើ Streamlit server CPU ទេ។
    */

    try {
      await new Promise((resolve) => setTimeout(resolve, 1000));

      setStatus(
        "វីដេអូរួចរាល់សម្រាប់ Browser AI។ យើងនឹងបន្ថែម Whisper + Khmer TTS engine បន្ទាប់។"
      );
    } catch (error) {
      setStatus(`មានបញ្ហា៖ ${error.message}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="app">
      <section className="card">
        <h1>🇰🇭 Smey Auto Caption V2</h1>

        <p className="subtitle">
          Browser AI • Khmer Caption • Translation • Khmer Voice
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
          <div className="file">
            <strong>📁 {video.name}</strong>
            <small>
              {(video.size / 1024 / 1024).toFixed(1)} MB
            </small>
          </div>
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

        <div className="features">
          <div>🎙️ Speech → Caption</div>
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
