async function start() {
  if (!video) {
    setStatus(
      "សូមបញ្ចូលវីដេអូជាមុនសិន។"
    );

    return;
  }

  setBusy(true);
  setDubbingBusy(true);

  setCaption([]);

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

  try {
    /* =========================================
       1. AUDIO
    ========================================= */

    setStatus(
      "🎧 កំពុងអានសំឡេង..."
    );

    const audio =
      await videoToAudio(
        video
      );

    if (
      !(audio instanceof Float32Array) ||
      !audio.length
    ) {
      throw new Error(
        "មិនមានសំឡេងក្នុងវីដេអូ"
      );
    }

    /* =========================================
       2. WHISPER
    ========================================= */

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

    /* =========================================
       3. FIX TIMESTAMP
    ========================================= */

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

    if (!chunks.length) {
      throw new Error(
        "មិនរកឃើញសំឡេងនិយាយទេ។"
      );
    }

    setCaption(
      chunks
    );

    /* =========================================
       4. TRANSLATE
    ========================================= */

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

    /* =========================================
       5. AUTO DUBBING
       មិនចាំបាច់ចុចប៊ូតុងទី២
    ========================================= */

    setStatus(
      "🗣️ Caption រួចរាល់ — កំពុងបង្កើត Khmer Dubbing..."
    );

    const blob =
      await createDubbingVideo(
        video,
        chunks,
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

    /* =========================================
       6. SHOW MP4
    ========================================= */

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
      `✅ រួចរាល់! MP4 ${(blob.size / 1024 / 1024).toFixed(1)} MB`
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
    setBusy(false);
    setDubbingBusy(false);
  }
}
