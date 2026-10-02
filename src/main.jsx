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

/* =========================================================
   WHISPER
========================================================= */

async function getTranscriber(setStatus) {
  if (transcriber) return transcriber;

  setStatus("⏳ កំពុងទាញ Whisper Tiny...");

  transcriber = await pipeline(
    "automatic-speech-recognition",
    "onnx-community/whisper-tiny"
  );

  return transcriber;
}

/* =========================================================
   TRANSLATION
========================================================= */

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

  return (
    data?.responseData?.translatedText?.trim() || ""
  );
}

/* =========================================================
   AUDIO
========================================================= */

function resampleAudio(input, fromRate, toRate) {
  if (fromRate === toRate) {
    return input;
  }

  const ratio = fromRate / toRate;
  const newLength = Math.round(
    input.length / ratio
  );

  const output =
    new Float32Array(newLength);

  for (let i = 0; i < newLength; i++) {
    const position = i * ratio;

    const left =
      Math.floor(position);

    const right =
      Math.min(
        left + 1,
        input.length - 1
      );

    const weight =
      position - left;

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

  const mono =
    new Float32Array(
      audioBuffer.length
    );

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
      i < mono.length;
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

/* =========================================================
   KHMER TTS
========================================================= */

const TTS_BASE =
  "https://huggingface.co/sengtha/khmer-tts-female-v2/resolve/main";

async function loadKhmerTTS(setStatus) {
  if (
    ttsSession &&
    ttsMeta
  ) {
    return {
      session: ttsSession,
      meta: ttsMeta,
    };
  }

  setStatus(
    "🗣️ កំពុងទាញ Khmer Voice..."
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

  ttsSession =
    await ort.InferenceSession.create(
      `${TTS_BASE}/khmer_tts_v3.onnx`,
      {
        executionProviders: [
          "wasm",
        ],
      }
    );

  return {
    session: ttsSession,
    meta: ttsMeta,
  };
}

function textToIds(
  text,
  meta
) {
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

  const ids = [];

  for (const char of text) {
    if (idOf.has(char)) {
      ids.push(
        idOf.get(char)
      );
    }
  }

  if (!meta.add_blank) {
    return ids;
  }

  const blankId =
    idOf.get(meta.blank);

  if (
    blankId === undefined
  ) {
    return ids;
  }

  const output = [
    blankId,
  ];

  for (const id of ids) {
    output.push(id);
    output.push(blankId);
  }

  return output;
}

async function synthesizeKhmer(
  text,
  setStatus
) {
  const {
    session,
    meta,
  } =
    await loadKhmerTTS(
      setStatus
    );

  const ids =
    textToIds(
      text,
      meta
    );

  if (!ids.length) {
    return {
      samples:
        new Float32Array(0),
      sampleRate:
        Number(
          meta.sample_rate
        ) || 22050,
    };
  }

  const inputIds =
    new BigInt64Array(
      ids.map(
        (x) => BigInt(x)
      )
    );

  const lengths =
    new BigInt64Array([
      BigInt(ids.length),
    ]);

  const feeds = {};

  /*
    Model uses:
    x
    x_lengths
  */

  const inputNames =
    session.inputNames;

  const xName =
    inputNames.find(
      (name) =>
        name === "x"
    ) ||
    inputNames[0];

  const xLengthName =
    inputNames.find(
      (name) =>
        name === "x_lengths"
    ) ||
    inputNames[1];

  feeds[xName] =
    new ort.Tensor(
      "int64",
      inputIds,
      [1, ids.length]
    );

  feeds[xLengthName] =
    new ort.Tensor(
      "int64",
      lengths,
      [1]
    );

  const output =
    await session.run(
      feeds
    );

  const outputName =
    session.outputNames[0];

  const tensor =
    output[outputName];

  let samples =
    new Float32Array(
      tensor.data
    );

  /*
    Some ONNX outputs can be
    [1,1,T] or [1,T].
    The underlying data is still
    the same continuous PCM array.
  */

  const sampleRate =
    Number(
      meta.sample_rate
    ) || 22050;

  return {
    samples,
    sampleRate,
  };
}

/* =========================================================
   WAV
========================================================= */

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
    new DataView(
      buffer
    );

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

/* =========================================================
   VIDEO DURATION
========================================================= */

function getVideoDuration(
  file
) {
  return new Promise(
    (resolve, reject) => {
      const url =
        URL.createObjectURL(
          file
        );

      const video =
        document.createElement(
          "video"
        );

      video.preload =
        "metadata";

      video.onloadedmetadata =
        () => {
          const duration =
            video.duration;

          URL.revokeObjectURL(
            url
          );

          resolve(
            duration
          );
        };

      video.onerror = () => {
        URL.revokeObjectURL(
          url
        );

        reject(
          new Error(
            "មិនអាចអាន video duration"
          )
        );
      };

      video.src = url;
    }
  );
}

/* =========================================================
   FFMPEG
========================================================= */

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

/* =========================================================
   ATEMPO FILTER
   Preserve pitch while changing speed.
========================================================= */

function makeAtempoFilters(
  factor
) {
  let value =
    Number(factor);

  if (
    !Number.isFinite(value) ||
    value <= 0
  ) {
    return [
      "atempo=1",
    ];
  }

  const filters = [];

  /*
    FFmpeg atempo is safest
    inside 0.5 - 2.0.

    Split large changes into
    multiple atempo filters.
  */

  while (value > 2) {
    filters.push(
      "atempo=2"
    );

    value /= 2;
  }

  while (value < 0.5) {
    filters.push(
      "atempo=0.5"
    );

    value /= 0.5;
  }

  filters.push(
    `atempo=${value.toFixed(6)}`
  );

  return filters;
}

/* =========================================================
   CREATE DUBBING VIDEO
========================================================= */

async function createDubbingVideo(
  videoFile,
  caption,
  setStatus
) {
  const duration =
    await getVideoDuration(
      videoFile
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

  const outputName =
    "smey-khmer-dubbing.mp4";

  await ffmpeg.writeFile(
    inputName,
    await fetchFile(
      videoFile
    )
  );

  /*
    Remove old temporary
    voice files from previous
    generation.
  */

  for (
    let i = 0;
    i < caption.length;
    i++
  ) {
    try {
      await ffmpeg.deleteFile(
        `voice_${i}.wav`
      );
    } catch {}

    try {
      await ffmpeg.deleteFile(
        `voice_fit_${i}.wav`
      );
    } catch {}
  }

  const validSegments = [];

  /* =======================================================
     STEP 1
     Generate each Khmer voice separately.
  ======================================================= */

  for (
    let i = 0;
    i < caption.length;
    i++
  ) {
    const item =
      caption[i];

    const khmer =
      item.khmer?.trim();

    if (!khmer) {
      continue;
    }

    if (
      khmer.startsWith(
        "⚠️"
      )
    ) {
      continue;
    }

    let start =
      Number(item.start);

    let end =
      Number(item.end);

    if (
      !Number.isFinite(start)
    ) {
      start = 0;
    }

    if (
      !Number.isFinite(end)
    ) {
      end =
        start + 1;
    }

    start =
      Math.max(
        0,
        start
      );

    end =
      Math.max(
        start + 0.25,
        end
      );

    /*
      Never allow segment
      outside video.
    */

    start =
      Math.min(
        start,
        Math.max(
          0,
          duration - 0.05
        )
      );

    end =
      Math.min(
        end,
        duration
      );

    if (
      end <= start
    ) {
      continue;
    }

    const targetDuration =
      end - start;

    setStatus(
      `🗣️ បង្កើត Khmer Voice ${i + 1}/${caption.length}...`
    );

    const result =
      await synthesizeKhmer(
        khmer,
        setStatus
      );

    if (
      !result.samples.length
    ) {
      continue;
    }

    const sourceDuration =
      result.samples.length /
      result.sampleRate;

    const voiceFile =
      `voice_${i}.wav`;

    const fittedFile =
      `voice_fit_${i}.wav`;

    const wav =
      floatToWav(
        result.samples,
        result.sampleRate
      );

    await ffmpeg.writeFile(
      voiceFile,
      wav
    );

    /*
      atempo factor:

      source 2 sec
      target 1 sec
      => factor 2
      => speak faster

      source 1 sec
      target 2 sec
      => factor 0.5
      => speak slower
    */

    const factor =
      sourceDuration /
      targetDuration;

    const atempoFilters =
      makeAtempoFilters(
        factor
      );

    /*
      Add tiny fade to prevent
      clicks at boundaries.
    */

    const filter =
      [
        ...atempoFilters,
        "afade=t=in:st=0:d=0.015",
        `afade=t=out:st=${Math.max(
          0.02,
          targetDuration - 0.02
        ).toFixed(3)}:d=0.015`,
      ].join(",");

    /*
      If voice is already close
      to target, atempo still
      preserves pitch and gives
      better timing than raw
      resampling.
    */

    await ffmpeg.exec([
      "-i",
      voiceFile,

      "-filter:a",
      filter,

      "-ar",
      "22050",

      "-ac",
      "1",

      "-y",

      fittedFile,
    ]);

    validSegments.push({
      file:
        fittedFile,

      start,

      end,
    });
  }

  if (
    !validSegments.length
  ) {
    throw new Error(
      "មិនមាន Khmer Voice សម្រាប់ Dubbing ទេ"
    );
  }

  /* =======================================================
     STEP 2
     Build one FFmpeg filter graph.

     Original audio:
       30%

     Khmer voices:
       100%

     Every voice:
       exact Whisper timestamp
  ======================================================= */

  setStatus(
    "🎚️ កំពុង Sync Khmer Voice តាម timestamp..."
  );

  const inputArgs = [
    "-i",
    inputName,
  ];

  for (
    const segment of validSegments
  ) {
    inputArgs.push(
      "-i",
      segment.file
    );
  }

  const filterParts = [];

  /*
    Original video audio
  */

  filterParts.push(
    "[0:a]volume=0.30[orig]"
  );

  const voiceLabels = [];

  for (
    let i = 0;
    i < validSegments.length;
    i++
  ) {
    const segment =
      validSegments[i];

    const inputIndex =
      i + 1;

    const label =
      `v${i}`;

    const delay =
      Math.max(
        0,
        Math.round(
          segment.start * 1000
        )
      );

    filterParts.push(
      `[${inputIndex}:a]adelay=${delay}:all=1,volume=1.0[${label}]`
    );

    voiceLabels.push(
      `[${label}]`
    );
  }

  /*
    Mix original + all Khmer voices.
  */

  filterParts.push(
    `[orig]${voiceLabels.join("")}amix=inputs=${
      validSegments.length + 1
    }:duration=first:dropout_transition=0:normalize=0[mix]`
  );

  const filterComplex =
    filterParts.join(";");

  setStatus(
    "🎬 កំពុងបង្កើត MP4 និងរក្សា Original Music..."
  );

  try {
    await ffmpeg.exec([
      ...inputArgs,

      "-filter_complex",
      filterComplex,

      "-map",
      "0:v:0",

      "-map",
      "[mix]",

      "-c:v",
      "copy",

      "-c:a",
      "aac",

      "-b:a",
      "192k",

      "-ar",
      "48000",

      "-ac",
      "2",

      "-shortest",

      "-movflags",
      "+faststart",

      "-y",

      outputName,
    ]);
  } catch (error) {
    console.warn(
      "Video copy failed. Re-encoding..."
    );

    await ffmpeg.exec([
      ...inputArgs,

      "-filter_complex",
      filterComplex,

      "-map",
      "0:v:0",

      "-map",
      "[mix]",

      "-c:v",
      "libx264",

      "-preset",
      "ultrafast",

      "-crf",
      "28",

      "-pix_fmt",
      "yuv420p",

      "-c:a",
      "aac",

      "-b:a",
      "192k",

      "-ar",
      "48000",

      "-ac",
      "2",

      "-shortest",

      "-movflags",
      "+faststart",

      "-y",

      outputName,
    ]);
  }

  const output =
    await ffmpeg.readFile(
      outputName
    );

  /*
    IMPORTANT:
    Use output directly.
    Do not use output.buffer,
    because Uint8Array can have
    a non-zero byteOffset.
  */

  return new Blob(
    [output],
    {
      type:
        "video/mp4",
    }
  );
}

/* =========================================================
   TIME
========================================================= */

function formatTime(
  seconds
) {
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

/* =========================================================
   CLEAN CHUNKS
========================================================= */

function cleanChunks(
  chunks
) {
  const result = [];

  let previous = "";

  for (
    const item of chunks
  ) {
    const normalized =
      item.text
        .toLowerCase()
        .replace(
          /\s+/g,
          " "
        )
        .trim();

    if (!normalized) {
      continue;
    }

    /*
      Only remove exact consecutive
      duplicates.
    */

    if (
      normalized ===
      previous
    ) {
      continue;
    }

    previous =
      normalized;

    result.push(
      item
    );
  }

  return result;
}

/* =========================================================
   APP
========================================================= */

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

  const [caption, setCaption] =
    useState([]);

  const [sourceLang, setSourceLang] =
    useState(
      "eng_Latn"
    );

  const [downloadUrl, setDownloadUrl] =
    useState("");

  const [outputVideoUrl, setOutputVideoUrl] =
    useState("");

  /* =======================================================
     VIDEO PREVIEW
  ======================================================= */

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

    return () => {
      URL.revokeObjectURL(
        url
      );
    };
  }, [video]);

  /* =======================================================
     FILE SELECT
  ======================================================= */

  function handleVideo(
    event
  ) {
    const file =
      event.target.files?.[0];

    if (!file) {
      return;
    }

    if (downloadUrl) {
      URL.revokeObjectURL(
        downloadUrl
      );
    }

    if (outputVideoUrl) {
      URL.revokeObjectURL(
        outputVideoUrl
      );
    }

    setDownloadUrl("");
    setOutputVideoUrl("");

    setVideo(file);

    setCaption([]);

    setStatus(
      `បានជ្រើស៖ ${file.name}`
    );
  }

  /* =======================================================
     START
  ======================================================= */

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
      /* -----------------------------------------------
         AUDIO
      ----------------------------------------------- */

      setStatus(
        "🎧 កំពុងអានសំឡេង..."
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

      /* -----------------------------------------------
         WHISPER
      ----------------------------------------------- */

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

      let chunks =
        (
          result.chunks ||
          []
        )
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

              khmer:
                "",
            })
          )
          .filter(
            (item) =>
              item.text
          );

      chunks =
        cleanChunks(
          chunks
        );

      /*
        Fix missing/zero end timestamps.
      */

      for (
        let i = 0;
        i < chunks.length;
        i++
      ) {
        const current =
          chunks[i];

        const next =
          chunks[i + 1];

        if (
          !Number.isFinite(
            current.start
          )
        ) {
          current.start = 0;
        }

        if (
          !Number.isFinite(
            current.end
          ) ||
          current.end <=
            current.start
        ) {
          if (next) {
            current.end =
              Math.max(
                current.start +
                  0.35,
                next.start
              );
          } else {
            current.end =
              current.start +
              1;
          }
        }

        /*
          Don't let a segment
          become ridiculously long.
        */

        current.start =
          Math.max(
            0,
            current.start
          );

        current.end =
          Math.max(
            current.start +
              0.25,
            current.end
          );
      }

      setCaption(
        chunks
      );

      if (!chunks.length) {
        setStatus(
          "⚠️ មិនរកឃើញសំឡេងនិយាយទេ។"
        );

        return;
      }

      /* -----------------------------------------------
         TRANSLATION
      ----------------------------------------------- */

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

          chunks[i].khmer =
            khmer;
        } catch (
          error
        ) {
          console.error(
            error
          );

          chunks[i].khmer =
            "⚠️ បកប្រែមិនបាន";
        }

        setCaption(
          [...chunks]
        );
      }

      setStatus(
        `✅ Caption + Khmer រួចរាល់ — ${chunks.length} ផ្នែក`
      );
    } catch (
      error
    ) {
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
      setBusy(
        false
      );
    }
  }

  /* =======================================================
     DUBBING
  ======================================================= */

  async function handleDubbing() {
    if (
      !video ||
      !caption.length
    ) {
      setStatus(
        "សូមបង្កើត Caption ជាមុនសិន។"
      );

      return;
    }

    setDubbingBusy(
      true
    );

    try {
      if (downloadUrl) {
        URL.revokeObjectURL(
          downloadUrl
        );
      }

      if (outputVideoUrl) {
        URL.revokeObjectURL(
          outputVideoUrl
        );
      }

      setDownloadUrl("");
      setOutputVideoUrl("");

      const blob =
        await createDubbingVideo(
          video,
          caption,
          setStatus
        );

      if (
        !blob ||
        blob.size === 0
      ) {
        throw new Error(
          "MP4 ទទេ ឬបង្កើតមិនបាន"
        );
      }

      const url =
        URL.createObjectURL(
          blob
        );

      setDownloadUrl(
        url
      );

      setOutputVideoUrl(
        url
      );

      setStatus(
        `✅ MP4 រួចរាល់ — ${(blob.size / 1024 / 1024).toFixed(1)} MB`
      );
    } catch (
      error
    ) {
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
      setDubbingBusy(
        false
      );
    }
  }

  /* =======================================================
     UI
  ======================================================= */

  return (
    <main className="app">
      <section className="card">

        <h1>
          🇰🇭 Smey Auto Caption V2
        </h1>

        <p className="subtitle">
          Whisper • Khmer Translation • Khmer Voice • MP4
        </p>

        {/* =================================================
            UPLOAD
        ================================================= */}

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
                src={
                  videoUrl
                }
                controls
                playsInline
                preload="metadata"
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

        {/* =================================================
            SOURCE LANGUAGE
        ================================================= */}

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
                e.target.value
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

        {/* =================================================
            START
        ================================================= */}

        <button
          className="start"
          onClick={
            start
          }
          disabled={
            busy ||
            dubbingBusy
          }
        >
          {busy
            ? "⏳ កំពុងដំណើរការ..."
            : "▶️ ចាប់ផ្ដើម"}
        </button>

        {/* =================================================
            DUBBING BUTTON
        ================================================= */}

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

        {/* =================================================
            STATUS
        ================================================= */}

        <div className="status">
          {status}
        </div>

        {/* =================================================
            GENERATED VIDEO
        ================================================= */}

        {outputVideoUrl && (
          <div
            style={{
              marginTop:
                "18px",
            }}
          >
            <h3>
              🎬 វីដេអូ Khmer Dubbing
            </h3>

            <video
              src={
                outputVideoUrl
              }
              controls
              playsInline
              preload="metadata"
              style={{
                width:
                  "100%",

                borderRadius:
                  "14px",

                display:
                  "block",

                background:
                  "#000",
              }}
            />

            {downloadUrl && (
              <a
                href={
                  downloadUrl
                }
                download="smey-khmer-dubbing.mp4"
                style={{
                  display:
                    "block",

                  marginTop:
                    "12px",

                  padding:
                    "15px",

                  borderRadius:
                    "12px",

                  background:
                    "#16a34a",

                  color:
                    "#fff",

                  textAlign:
                    "center",

                  textDecoration:
                    "none",

                  fontWeight:
                    "800",

                  fontSize:
                    "17px",
                }}
              >
                📥 Download MP4
              </a>
            )}
          </div>
        )}

        {/* =================================================
            TIMELINE
        ================================================= */}

        {caption.length >
          0 && (
          <div
            style={{
              marginTop:
                "18px",
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
                      "10px",

                    padding:
                      "13px",

                    borderRadius:
                      "12px",

                    background:
                      "#334155",
                  }}
                >
                  <small
                    style={{
                      opacity:
                        0.8,
                    }}
                  >
                    {
                      formatTime(
                        item.start
                      )
                    }
                    {" → "}
                    {
                      formatTime(
                        item.end
                      )
                    }
                  </small>

                  <div
                    style={{
                      marginTop:
                        "7px",

                      fontWeight:
                        "600",
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
                          "8px",

                        paddingTop:
                          "8px",

                        borderTop:
                          "1px solid rgba(255,255,255,.15)",

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

        {/* =================================================
            FEATURES
        ================================================= */}

        <div className="features">

          <div>
            🎙️ Whisper Speech → Caption
          </div>

          <div>
            🇰🇭 Khmer Translation
          </div>

          <div>
            🗣️ Khmer Voice
          </div>

          <div>
            🎬 Timing Sync
          </div>

          <div>
            🎵 Original Audio/Music
          </div>

          <div>
            📥 MP4 Download
          </div>

        </div>

      </section>
    </main>
  );
}

/* =========================================================
   START APP
========================================================= */

createRoot(
  document.getElementById(
    "root"
  )
).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
