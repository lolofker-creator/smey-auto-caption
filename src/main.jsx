import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import * as ort from "onnxruntime-web";
import { FFmpeg } from "@ffmpeg/ffmpeg";
import { fetchFile, toBlobURL } from "@ffmpeg/util";
import "./style.css";

let ttsSession = null;
let ttsMeta = null;
let ffmpegInstance = null;

const TTS_BASE =
  "https://huggingface.co/sengtha/khmer-tts-female-v2/resolve/main";

async function loadKhmerTTS(setStatus) {
  if (ttsSession && ttsMeta) return { session: ttsSession, meta: ttsMeta };

  setStatus("🗣️ កំពុងទាញ Khmer Voice...");
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
  const vocab = meta.vocab || [];
  const idOf = new Map(vocab.map((char, index) => [char, index]));
  const ids = [];

  for (const char of text) {
    if (idOf.has(char)) ids.push(idOf.get(char));
  }

  if (!meta.add_blank) return ids;

  const blankId = idOf.get(meta.blank);
  if (blankId === undefined) return ids;

  const out = [blankId];
  for (const id of ids) out.push(id, blankId);
  return out;
}

async function synthesizeKhmer(text, setStatus) {
  const { session, meta } = await loadKhmerTTS(setStatus);
  const ids = textToIds(text, meta);

  if (!ids.length) {
    return {
      samples: new Float32Array(0),
      sampleRate: Number(meta.sample_rate) || 22050,
    };
  }

  const inputIds = new BigInt64Array(ids.map(x => BigInt(x)));
  const lengths = new BigInt64Array([BigInt(ids.length)]);
  const feeds = {};

  const inputNames = session.inputNames;
  const xName = inputNames.find(name => name === "x") || inputNames[0];
  const xLengthName =
    inputNames.find(name => name === "x_lengths") || inputNames[1];

  feeds[xName] = new ort.Tensor("int64", inputIds, [1, ids.length]);
  feeds[xLengthName] = new ort.Tensor("int64", lengths, [1]);

  const output = await session.run(feeds);
  const tensor = output[session.outputNames[0]];

  return {
    samples: new Float32Array(tensor.data),
    sampleRate: Number(meta.sample_rate) || 22050,
  };
}

function floatToWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  const writeString = (offset, value) => {
    for (let i = 0; i < value.length; i++) {
      view.setUint8(offset + i, value.charCodeAt(i));
    }
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
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
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }

  return new Uint8Array(buffer);
}

function parseSrtTime(value) {
  const m = value.trim().replace(",", ".").match(
    /^(?:(\d+):)?(\d{1,2}):(\d{2})\.(\d{3})$/
  );
  if (!m) return NaN;

  const hours = Number(m[1] || 0);
  const minutes = Number(m[2]);
  const seconds = Number(m[3]);
  const millis = Number(m[4]);

  return hours * 3600 + minutes * 60 + seconds + millis / 1000;
}

function parseSrt(text) {
  const blocks = text.replace(/\r/g, "").trim().split(/\n\s*\n/);
  const result = [];

  for (const block of blocks) {
    const lines = block.split("\n").map(x => x.trim()).filter(Boolean);
    if (!lines.length) continue;

    const timeIndex = lines.findIndex(x => x.includes("-->"));
    if (timeIndex < 0) continue;

    const [a, b] = lines[timeIndex].split("-->").map(x => x.trim());
    const start = parseSrtTime(a);
    const end = parseSrtTime(b);
    const textLines = lines.slice(timeIndex + 1);

    if (
      Number.isFinite(start) &&
      Number.isFinite(end) &&
      end > start &&
      textLines.length
    ) {
      result.push({
        start,
        end,
        khmer: textLines.join(" ").trim(),
      });
    }
  }

  return result;
}

async function getVideoDuration(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const video = document.createElement("video");
    video.preload = "metadata";

    video.onloadedmetadata = () => {
      URL.revokeObjectURL(url);
      resolve(video.duration);
    };

    video.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("មិនអាចអាន video duration"));
    };

    video.src = url;
  });
}

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

function makeAtempoFilters(factor) {
  let value = Number(factor);
  if (!Number.isFinite(value) || value <= 0) return ["atempo=1"];

  const filters = [];
  while (value > 2) {
    filters.push("atempo=2");
    value /= 2;
  }
  while (value < 0.5) {
    filters.push("atempo=0.5");
    value /= 0.5;
  }
  filters.push(`atempo=${value.toFixed(6)}`);
  return filters;
}

async function createDubbingVideo(videoFile, segments, setStatus) {
  const duration = await getVideoDuration(videoFile);
  const ffmpeg = await getFFmpeg(setStatus);

  const extension =
    videoFile.name.split(".").pop()?.toLowerCase() || "mp4";
  const inputName = `input.${extension}`;
  const outputName = "smey-khmer-dubbing.mp4";

  await ffmpeg.writeFile(inputName, await fetchFile(videoFile));

  const valid = [];
  const batchSize = 3;

  for (let base = 0; base < segments.length; base += batchSize) {
    const batch = segments.slice(base, base + batchSize);

    setStatus(
      `🗣️ បង្កើត Khmer Voice ${Math.min(
        base + batch.length,
        segments.length
      )}/${segments.length}...`
    );

    const results = await Promise.all(
      batch.map(async (item, localIndex) => {
        const i = base + localIndex;
        const khmer = item.khmer.trim();

        let start = Math.max(0, Number(item.start) || 0);
        let end = Math.min(duration, Number(item.end) || start + 1);
        end = Math.max(start + 0.25, end);

        if (start >= duration || end <= start) return null;

        const result = await synthesizeKhmer(khmer, setStatus);
        if (!result.samples.length) return null;

        const sourceDuration = result.samples.length / result.sampleRate;
        const targetDuration = end - start;
        const factor = sourceDuration / targetDuration;
        const filter = [
          ...makeAtempoFilters(factor),
          "afade=t=in:st=0:d=0.015",
          `afade=t=out:st=${Math.max(
            0.02,
            targetDuration - 0.02
          ).toFixed(3)}:d=0.015`,
        ].join(",");

        const voiceFile = `voice_${i}.wav`;
        const fittedFile = `voice_fit_${i}.wav`;

        await ffmpeg.writeFile(
          voiceFile,
          floatToWav(result.samples, result.sampleRate)
        );

        await ffmpeg.exec([
          "-i", voiceFile,
          "-filter:a", filter,
          "-ar", "22050",
          "-ac", "1",
          "-y", fittedFile,
        ]);

        return { file: fittedFile, start, end };
      })
    );

    for (const item of results) if (item) valid.push(item);
  }

  if (!valid.length) {
    throw new Error("មិនមាន Khmer Voice សម្រាប់ Dubbing ទេ");
  }

  setStatus("🎚️ កំពុង Sync Khmer Voice តាម timestamp...");

  const inputArgs = ["-i", inputName];
  for (const segment of valid) inputArgs.push("-i", segment.file);

  const filterParts = ["[0:a]volume=0.30[orig]"];
  const labels = [];

  for (let i = 0; i < valid.length; i++) {
    const segment = valid[i];
    const delay = Math.max(0, Math.round(segment.start * 1000));
    const label = `v${i}`;

    filterParts.push(
      `[${i + 1}:a]adelay=${delay}:all=1,volume=1.0[${label}]`
    );
    labels.push(`[${label}]`);
  }

  filterParts.push(
    `[orig]${labels.join("")}amix=inputs=${
      valid.length + 1
    }:duration=first:dropout_transition=0:normalize=0[mix]`
  );

  const filterComplex = filterParts.join(";");

  setStatus("🎬 កំពុងបង្កើត MP4 និងរក្សា Original Music...");

  try {
    await ffmpeg.exec([
      ...inputArgs,
      "-filter_complex", filterComplex,
      "-map", "0:v:0",
      "-map", "[mix]",
      "-c:v", "copy",
      "-c:a", "aac",
      "-b:a", "192k",
      "-ar", "48000",
      "-ac", "2",
      "-shortest",
      "-movflags", "+faststart",
      "-y", outputName,
    ]);
  } catch {
    await ffmpeg.exec([
      ...inputArgs,
      "-filter_complex", filterComplex,
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
      "-y", outputName,
    ]);
  }

  const output = await ffmpeg.readFile(outputName);
  return new Blob([output], { type: "video/mp4" });
}

function App() {
  const [video, setVideo] = useState(null);
  const [srt, setSrt] = useState(null);
  const [segments, setSegments] = useState([]);
  const [videoUrl, setVideoUrl] = useState("");
  const [outputUrl, setOutputUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("រង់ចាំវីដេអូ និង Khmer SRT...");

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
    setStatus(`បានជ្រើសវីដេអូ៖ ${file.name}`);
  }

  async function handleSrt(e) {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      const text = await file.text();
      const parsed = parseSrt(text);

      if (!parsed.length) {
        throw new Error("SRT មិនមាន timestamp/text ត្រឹមត្រូវ");
      }

      setSrt(file);
      setSegments(parsed);
      setStatus(`បានអាន Khmer SRT — ${parsed.length} ផ្នែក`);
    } catch (error) {
      setSrt(null);
      setSegments([]);
      setStatus(`❌ ${error.message}`);
    }
  }

  async function startDubbing() {
    if (!video) {
      setStatus("សូមបញ្ចូលវីដេអូជាមុនសិន។");
      return;
    }

    if (!segments.length) {
      setStatus("សូមបញ្ចូល Khmer SRT ជាមុនសិន។");
      return;
    }

    setBusy(true);

    try {
      if (outputUrl) URL.revokeObjectURL(outputUrl);
      setOutputUrl("");

      const blob = await createDubbingVideo(
        video,
        segments,
        setStatus
      );

      const url = URL.createObjectURL(blob);
      setOutputUrl(url);
      setStatus(
        `✅ Khmer Dubbing រួចរាល់ — ${(blob.size / 1024 / 1024).toFixed(1)} MB`
      );
    } catch (error) {
      console.error(error);
      setStatus(`❌ Dubbing error៖ ${error?.message || "Unknown error"}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="app">
      <section className="card">
        <h1>🇰🇭 Smey Khmer Dubbing</h1>
        <p className="subtitle">
          Video + Khmer SRT → Khmer Voice → Sync → MP4
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
                preload="metadata"
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

        <label className="upload" style={{ marginTop: "14px" }}>
          <span>📄 បញ្ចូល Khmer SRT</span>
          <input
            type="file"
            accept=".srt,text/plain"
            onChange={handleSrt}
          />
        </label>

        {srt && (
          <div className="file">
            <strong>📄 {srt.name}</strong>
            <small>{segments.length} ផ្នែក</small>
          </div>
        )}

        <button
          className="start"
          onClick={startDubbing}
          disabled={busy}
        >
          {busy ? "🎬 កំពុង Dubbing..." : "🗣️ បង្កើត Khmer Dubbing MP4"}
        </button>

        <div className="status">{status}</div>

        {outputUrl && (
          <div style={{ marginTop: "18px" }}>
            <h3>🎬 Khmer Dubbing MP4</h3>

            <video
              src={outputUrl}
              controls
              playsInline
              preload="metadata"
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

        {segments.length > 0 && (
          <div style={{ marginTop: "18px" }}>
            <h3>⏱️ Dubbing Timeline</h3>

            {segments.map((item, index) => (
              <div
                key={index}
                style={{
                  marginBottom: "10px",
                  padding: "13px",
                  borderRadius: "12px",
                  background: "#334155",
                }}
              >
                <small>
                  {new Date(item.start * 1000)
                    .toISOString()
                    .slice(14, 19)}
                  {" → "}
                  {new Date(item.end * 1000)
                    .toISOString()
                    .slice(14, 19)}
                </small>

                <div style={{ marginTop: "7px", fontWeight: "600" }}>
                  🇰🇭 {item.khmer}
                </div>
              </div>
            ))}
          </div>
        )}

        <div className="features">
          <div>🗣️ Khmer Voice</div>
          <div>🎚️ Timestamp Sync</div>
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
