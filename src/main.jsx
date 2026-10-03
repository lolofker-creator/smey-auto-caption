import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { pipeline } from "@huggingface/transformers";
import * as ort from "onnxruntime-web";
import { FFmpeg } from "@ffmpeg/ffmpeg";
import { fetchFile, toBlobURL } from "@ffmpeg/util";
import "./style.css";

let whisper = null;
let ttsSession = null;
let ttsMeta = null;
let ffmpegInstance = null;

const TTS_BASE =
  "https://huggingface.co/sengtha/khmer-tts-female-v2/resolve/main";

async function getWhisper(setStatus) {
  if (whisper) return whisper;
  setStatus("🤖 កំពុងផ្ទុក Whisper...");
  whisper = await pipeline(
    "automatic-speech-recognition",
    "onnx-community/whisper-tiny"
  );
  return whisper;
}

async function translateText(text, sourceLang = "eng_Latn") {
  const source = sourceLang === "zho_Hans" ? "zh-CN" : "en";
  const params = new URLSearchParams({
    q: text.trim(),
    langpair: `${source}|km-KM`,
    mt: "1",
  });
  const response = await fetch(
    `https://api.mymemory.translated.net/get?${params.toString()}`
  );
  if (!response.ok) throw new Error(`Translation API ${response.status}`);
  const data = await response.json();
  return data?.responseData?.translatedText?.trim() || "";
}

function resampleAudio(input, fromRate, toRate) {
  if (fromRate === toRate) return input;
  const ratio = fromRate / toRate;
  const output = new Float32Array(Math.round(input.length / ratio));
  for (let i = 0; i < output.length; i++) {
    const p = i * ratio;
    const l = Math.floor(p);
    const r = Math.min(l + 1, input.length - 1);
    const w = p - l;
    output[i] = input[l] * (1 - w) + input[r] * w;
  }
  return output;
}

async function videoToAudio(file) {
  const ctx = new AudioContext();
  const buffer = await ctx.decodeAudioData(await file.arrayBuffer());
  const mono = new Float32Array(buffer.length);

  for (let c = 0; c < buffer.numberOfChannels; c++) {
    const data = buffer.getChannelData(c);
    for (let i = 0; i < mono.length; i++) mono[i] += data[i] / buffer.numberOfChannels;
  }

  const audio = resampleAudio(mono, buffer.sampleRate, 16000);
  await ctx.close();
  return audio;
}

async function loadTTS(setStatus) {
  if (ttsSession && ttsMeta) return { session: ttsSession, meta: ttsMeta };

  setStatus("🗣️ កំពុងផ្ទុក Khmer Voice...");
  const metaResponse = await fetch(`${TTS_BASE}/tts_meta.json`);
  if (!metaResponse.ok) throw new Error("មិនអាចទាញ Khmer TTS metadata");

  ttsMeta = await metaResponse.json();
  ttsSession = await ort.InferenceSession.create(
    `${TTS_BASE}/khmer_tts_v3.onnx`,
    { executionProviders: ["wasm"] }
  );
  return { session: ttsSession, meta: ttsMeta };
}

function textToIds(text, meta) {
  const map = new Map((meta.vocab || []).map((x, i) => [x, i]));
  const ids = [];
  for (const ch of text) if (map.has(ch)) ids.push(map.get(ch));

  if (!meta.add_blank) return ids;
  const blank = map.get(meta.blank);
  if (blank === undefined) return ids;

  const out = [blank];
  for (const id of ids) out.push(id, blank);
  return out;
}

async function synthesizeKhmer(text, setStatus) {
  const { session, meta } = await loadTTS(setStatus);
  const ids = textToIds(text, meta);
  if (!ids.length) return { samples: new Float32Array(0), sampleRate: 22050 };

  const names = session.inputNames;
  const xName = names.find(x => x === "x") || names[0];
  const lenName = names.find(x => x === "x_lengths") || names[1];

  const feeds = {};
  feeds[xName] = new ort.Tensor(
    "int64",
    new BigInt64Array(ids.map(x => BigInt(x))),
    [1, ids.length]
  );
  feeds[lenName] = new ort.Tensor(
    "int64",
    new BigInt64Array([BigInt(ids.length)]),
    [1]
  );

  const result = await session.run(feeds);
  return {
    samples: new Float32Array(result[session.outputNames[0]].data),
    sampleRate: Number(meta.sample_rate) || 22050,
  };
}

function floatToWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const str = (o, s) => [...s].forEach((c, i) => view.setUint8(o + i, c.charCodeAt(0)));

  str(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  str(8, "WAVE");
  str(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  str(36, "data");
  view.setUint32(40, samples.length * 2, true);

  let o = 44;
  for (const value of samples) {
    const s = Math.max(-1, Math.min(1, value));
    view.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    o += 2;
  }
  return new Uint8Array(buffer);
}

async function getFFmpeg(setStatus) {
  if (ffmpegInstance) return ffmpegInstance;
  setStatus("🎬 កំពុងផ្ទុក FFmpeg...");

  const ffmpeg = new FFmpeg();
  const base =
    "https://cdn.jsdelivr.net/npm/@ffmpeg/core@0.12.10/dist/esm";

  await ffmpeg.load({
    coreURL: await toBlobURL(`${base}/ffmpeg-core.js`, "text/javascript"),
    wasmURL: await toBlobURL(`${base}/ffmpeg-core.wasm`, "application/wasm"),
  });

  ffmpegInstance = ffmpeg;
  return ffmpeg;
}

function atempo(factor) {
  let x = Number(factor);
  if (!Number.isFinite(x) || x <= 0) return "atempo=1";
  const parts = [];
  while (x > 2) {
    parts.push("atempo=2");
    x /= 2;
  }
  while (x < 0.5) {
    parts.push("atempo=0.5");
    x /= 0.5;
  }
  parts.push(`atempo=${x.toFixed(6)}`);
  return parts.join(",");
}

async function makeDubbing(videoFile, chunks, setStatus) {
  const ffmpeg = await getFFmpeg(setStatus);
  const duration = await new Promise((resolve, reject) => {
    const url = URL.createObjectURL(videoFile);
    const v = document.createElement("video");
    v.preload = "metadata";
    v.onloadedmetadata = () => {
      URL.revokeObjectURL(url);
      resolve(v.duration);
    };
    v.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("មិនអាចអាន video duration"));
    };
    v.src = url;
  });

  const ext = videoFile.name.split(".").pop()?.toLowerCase() || "mp4";
  const input = `input.${ext}`;
  const output = "smey-khmer-dubbing.mp4";

  await ffmpeg.writeFile(input, await fetchFile(videoFile));

  const valid = [];

  for (let i = 0; i < chunks.length; i++) {
    const text = chunks[i].khmer?.trim();
    if (!text || text.startsWith("⚠️")) continue;

    let start = Math.max(0, Number(chunks[i].start) || 0);
    let end = Math.min(duration, Number(chunks[i].end) || start + 1);
    end = Math.max(start + 0.25, end);
    if (start >= duration || end <= start) continue;

    setStatus(`🗣️ Khmer Voice ${i + 1}/${chunks.length}...`);

    const voice = await synthesizeKhmer(text, setStatus);
    if (!voice.samples.length) continue;

    const sourceDuration = voice.samples.length / voice.sampleRate;
    const targetDuration = end - start;
    const filter = [
      atempo(sourceDuration / targetDuration),
      "afade=t=in:st=0:d=0.015",
      `afade=t=out:st=${Math.max(0.02, targetDuration - 0.02).toFixed(3)}:d=0.015`,
    ].join(",");

    const wav = `voice_${i}.wav`;
    const fit = `voice_fit_${i}.wav`;

    await ffmpeg.writeFile(wav, floatToWav(voice.samples, voice.sampleRate));
    await ffmpeg.exec([
      "-i", wav,
      "-filter:a", filter,
      "-ar", "22050",
      "-ac", "1",
      "-y", fit,
    ]);

    valid.push({ file: fit, start });
  }

  if (!valid.length) throw new Error("មិនអាចបង្កើត Khmer Voice បានទេ");

  setStatus("🎚️ កំពុង Sync Voice...");

  const inputs = ["-i", input];
  for (const item of valid) inputs.push("-i", item.file);

  const filters = ["[0:a]volume=0.30[orig]"];
  const labels = [];

  for (let i = 0; i < valid.length; i++) {
    const delay = Math.round(valid[i].start * 1000);
    const label = `v${i}`;
    filters.push(
      `[${i + 1}:a]adelay=${delay}:all=1,volume=1.0[${label}]`
    );
    labels.push(`[${label}]`);
  }

  filters.push(
    `[orig]${labels.join("")}amix=inputs=${valid.length + 1}:duration=first:dropout_transition=0:normalize=0[mix]`
  );

  setStatus("🎬 កំពុងបង្កើត MP4...");

  try {
    await ffmpeg.exec([
      ...inputs,
      "-filter_complex", filters.join(";"),
      "-map", "0:v:0",
      "-map", "[mix]",
      "-c:v", "copy",
      "-c:a", "aac",
      "-b:a", "192k",
      "-ar", "48000",
      "-ac", "2",
      "-shortest",
      "-movflags", "+faststart",
      "-y", output,
    ]);
  } catch {
    await ffmpeg.exec([
      ...inputs,
      "-filter_complex", filters.join(";"),
      "-map", "0:v:0",
      "-map", "[mix]",
      "-c:v", "libx264",
      "-preset", "ultrafast",
      "-crf", "28",
      "-pix_fmt", "yuv420p",
      "-c:a", "aac",
      "-b:a", "192k",
      "-ar", "48000",
      "-ac", "2",
      "-shortest",
      "-movflags", "+faststart",
      "-y", output,
    ]);
  }

  const data = await ffmpeg.readFile(output);
  return new Blob([data], { type: "video/mp4" });
}

function App() {
  const [video, setVideo] = useState(null);
  const [videoUrl, setVideoUrl] = useState("");
  const [sourceLang, setSourceLang] = useState("eng_Latn");
  const [status, setStatus] = useState("🎥 បញ្ចូលវីដេអូ...");
  const [busy, setBusy] = useState(false);
  const [outputUrl, setOutputUrl] = useState("");

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
    if (outputUrl) URL.revokeObjectURL(outputUrl);
    setOutputUrl("");
    setVideo(file);
    setStatus(`បានជ្រើស៖ ${file.name}`);
  }

  async function startDubbing() {
    if (!video) {
      setStatus("សូមបញ្ចូលវីដេអូជាមុនសិន។");
      return;
    }

    setBusy(true);
    try {
      if (outputUrl) URL.revokeObjectURL(outputUrl);
      setOutputUrl("");

      setStatus("🎧 កំពុងអានសំឡេងពីវីដេអូ...");
      const audio = await videoToAudio(video);
      if (!audio.length) throw new Error("វីដេអូមិនមានសំឡេង");

      const model = await getWhisper(setStatus);
      setStatus("🤖 កំពុងស្តាប់សំឡេង...");

      const result = await model(audio, {
        return_timestamps: true,
        chunk_length_s: 30,
        stride_length_s: 5,
      });

      const chunks = (result.chunks || [])
        .map(x => ({
          start: x.timestamp?.[0] ?? 0,
          end: x.timestamp?.[1] ?? 0,
          text: x.text?.trim() || "",
          khmer: "",
        }))
        .filter(x => x.text);

      if (!chunks.length) throw new Error("រកមិនឃើញសំឡេងនិយាយទេ");

      for (let i = 0; i < chunks.length; i++) {
        const next = chunks[i + 1];
        if (!Number.isFinite(chunks[i].end) || chunks[i].end <= chunks[i].start) {
          chunks[i].end = next
            ? Math.max(chunks[i].start + 0.35, next.start)
            : chunks[i].start + 1;
        }

        setStatus(`🇰🇭 បកប្រែ ${i + 1}/${chunks.length}...`);
        try {
          chunks[i].khmer = await translateText(chunks[i].text, sourceLang);
        } catch {
          chunks[i].khmer = "";
        }
      }

      setStatus("🗣️ បកប្រែរួច — កំពុងបង្កើត Khmer Dubbing...");
      const blob = await makeDubbing(video, chunks, setStatus);
      const url = URL.createObjectURL(blob);

      setOutputUrl(url);
      setStatus(`✅ Dubbing រួចរាល់ — ${(blob.size / 1024 / 1024).toFixed(1)} MB`);
    } catch (error) {
      console.error(error);
      setStatus(`❌ ${error?.message || "Dubbing error"}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="app">
      <section className="card">
        <h1>🇰🇭 Smey Khmer Dubbing</h1>
        <p className="subtitle">
          Video → Auto Speech → Khmer Voice → Sync → MP4
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
              <small>{(video.size / 1024 / 1024).toFixed(1)} MB</small>
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
            onChange={e => setSourceLang(e.target.value)}
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
            <option value="zho_Hans">Chinese 简体中文</option>
          </select>
        </div>

        <button
          className="start"
          onClick={startDubbing}
          disabled={busy}
        >
          {busy ? "⏳ កំពុង Dubbing..." : "🗣️ បង្កើត Khmer Dubbing MP4"}
        </button>

        <div className="status">{status}</div>

        {outputUrl && (
          <div style={{ marginTop: "18px" }}>
            <h3>🎬 Khmer Dubbing</h3>
            <video
              src={outputUrl}
              controls
              playsInline
              style={{
                width: "100%",
                borderRadius: "14px",
                display: "block",
                background: "#000",
              }}
            />
            <a
              href={outputUrl}
              download="smey-khmer-dubbing.mp4"
              style={{
                display: "block",
                marginTop: "12px",
                padding: "15px",
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
          </div>
        )}

        <div className="features">
          <div>🤖 Auto Speech</div>
          <div>🇰🇭 Khmer Voice</div>
          <div>🎚️ Auto Timing Sync</div>
          <div>🎵 Original Audio/Music 30%</div>
          <div>🎬 MP4 Preview</div>
          <div>📥 MP4 Download</div>
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
