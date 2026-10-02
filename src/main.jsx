async function createDubbingVideo(
  videoFile,
  caption,
  setStatus
) {
  const duration =
    await getVideoDuration(videoFile);

  const ffmpeg =
    await getFFmpeg(setStatus);

  const extension =
    videoFile.name
      .split(".")
      .pop()
      ?.toLowerCase() || "mp4";

  const inputName =
    `input.${extension}`;

  const outputName =
    "smey-khmer-dubbing.mp4";

  await ffmpeg.writeFile(
    inputName,
    await fetchFile(videoFile)
  );

  const jobs = caption
    .map((item, index) => ({
      item,
      index,
    }))
    .filter(({ item }) => {
      const khmer =
        item.khmer?.trim();

      return (
        khmer &&
        !khmer.startsWith("⚠️")
      );
    });

  if (!jobs.length) {
    throw new Error(
      "មិនមាន Khmer Voice សម្រាប់ Dubbing ទេ"
    );
  }

  /*
    បង្កើត Khmer Voice 3 ផ្នែកក្នុងពេលតែមួយ
    ដើម្បីឱ្យ PROCESSING លឿនជាងមុន។
    ល្បឿនការនិយាយរបស់ AI Voice មិនត្រូវបានប្តូរ។
  */

  const generated = [];

  for (
    let batchStart = 0;
    batchStart < jobs.length;
    batchStart += 3
  ) {
    const batch =
      jobs.slice(
        batchStart,
        batchStart + 3
      );

    const batchEnd =
      Math.min(
        batchStart + batch.length,
        jobs.length
      );

    setStatus(
      `🗣️ បង្កើត Khmer Voice ${batchEnd}/${jobs.length}...`
    );

    const results =
      await Promise.all(
        batch.map(
          async ({ item, index }) => {
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
              !Number.isFinite(end) ||
              end <= start
            ) {
              end =
                start + 1;
            }

            start =
              Math.max(
                0,
                Math.min(
                  start,
                  Math.max(
                    0,
                    duration - 0.05
                  )
                )
              );

            end =
              Math.min(
                duration,
                Math.max(
                  start + 0.25,
                  end
                )
              );

            if (end <= start) {
              return null;
            }

            const tts =
              await synthesizeKhmer(
                item.khmer.trim(),
                setStatus
              );

            if (
              !tts?.samples?.length
            ) {
              return null;
            }

            return {
              item,
              index,
              start,
              end,
              tts,
            };
          }
        )
      );

    generated.push(
      ...results.filter(Boolean)
    );
  }

  if (!generated.length) {
    throw new Error(
      "មិនអាចបង្កើត Khmer Voice បានទេ"
    );
  }

  /*
    សរសេរ WAV ទាំងអស់ជាមុន
    ហើយ FFmpeg រត់តែ 1 ដងនៅចុងក្រោយ។
  */

  const inputArgs = [
    "-i",
    inputName,
  ];

  const valid = [];

  for (
    let i = 0;
    i < generated.length;
    i++
  ) {
    const segment =
      generated[i];

    const voiceName =
      `khmer_voice_${i}.wav`;

    const wav =
      floatToWav(
        segment.tts.samples,
        segment.tts.sampleRate
      );

    await ffmpeg.writeFile(
      voiceName,
      wav
    );

    inputArgs.push(
      "-i",
      voiceName
    );

    valid.push({
      ...segment,
      voiceName,
    });
  }

  setStatus(
    "🎚️ កំពុង Sync Khmer Voice តាម timestamp..."
  );

  const filterParts = [];

  /*
    Original Music / Audio = 30%
  */

  filterParts.push(
    "[0:a]volume=0.30[orig]"
  );

  const labels = [];

  for (
    let i = 0;
    i < valid.length;
    i++
  ) {
    const segment =
      valid[i];

    const inputIndex =
      i + 1;

    const targetDuration =
      Math.max(
        0.25,
        segment.end -
          segment.start
      );

    const sourceDuration =
      segment.tts.samples.length /
      segment.tts.sampleRate;

    let factor =
      sourceDuration /
      targetDuration;

    if (
      !Number.isFinite(factor) ||
      factor <= 0
    ) {
      factor = 1;
    }

    /*
      atempo:
      - រក្សា pitch
      - កែ duration ឱ្យត្រូវ timestamp
    */

    const tempoFilters = [];

    while (factor > 2) {
      tempoFilters.push(
        "atempo=2"
      );

      factor /= 2;
    }

    while (factor < 0.5) {
      tempoFilters.push(
        "atempo=0.5"
      );

      factor /= 0.5;
    }

    tempoFilters.push(
      `atempo=${factor.toFixed(6)}`
    );

    const delay =
      Math.max(
        0,
        Math.round(
          segment.start * 1000
        )
      );

    const label =
      `v${i}`;

    filterParts.push(
      `[${inputIndex}:a]${tempoFilters.join(",")},adelay=${delay}:all=1,volume=1.0[${label}]`
    );

    labels.push(
      `[${label}]`
    );
  }

  /*
    Mix Original + Khmer Voice
  */

  filterParts.push(
    `[orig]${labels.join("")}amix=inputs=${valid.length + 1}:duration=first:dropout_transition=0:normalize=0[mix]`
  );

  const filterComplex =
    filterParts.join(";");

  setStatus(
    "🎬 កំពុងបង្កើត MP4... (FFmpeg 1 ដង)"
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
      "Video copy failed. Re-encoding...",
      error
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

  if (
    !output ||
    !output.length
  ) {
    throw new Error(
      "MP4 មិនបានបង្កើត"
    );
  }

  return new Blob(
    [output],
    {
      type: "video/mp4",
    }
  );
}
