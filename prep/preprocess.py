#!/usr/bin/env python3
"""
Audio Dataset Preprocessing Pipeline
Splits continuous studio audio takes into phonetically aligned segments (LJSpeech format).
Outputs:
  - wavs/<clip_id>.wav (22,050 Hz, 16-bit mono PCM)
  - metadata.csv (<clip_id>|<text>|<text>)
"""

import os
import sys
import re
import json
import argparse
import subprocess

def parse_time(ts):
    if isinstance(ts, (int, float)):
        return float(ts) / 1000.0 if ts > 1000 else float(ts)
    parts = ts.replace(",", ".").split(":")
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    return float(parts[0])

def clean_text(text):
    text = re.sub(r"\[.*?\]", "", text).strip()
    text = re.sub(r"\(.*?\)", "", text).strip()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    parser = argparse.ArgumentParser(description="Preprocess long audio into Piper-compatible training dataset.")
    parser.add_argument("--input-audio", required=True, help="Path to raw source audio file")
    parser.add_argument("--output-dir", required=True, help="Destination directory for dataset")
    parser.add_argument("--whisper-bin", default="whisper-cli", help="Path to whisper-cli executable")
    parser.add_argument("--whisper-model", required=True, help="Path to whisper ggml model")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Target sample rate (Hz)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for Whisper")
    args = parser.parse_args()

    wavs_dir = os.path.join(args.output_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    temp_16k = os.path.join(args.output_dir, "temp_16k.wav")
    whisper_out = os.path.join(args.output_dir, "whisper_transcription")

    print(f"1. Resampling input to 16kHz mono for Whisper transcription: {temp_16k}...")
    cmd_resample = f"ffmpeg -y -i '{args.input_audio}' -ar 16000 -ac 1 -c:a pcm_s16le '{temp_16k}' -v quiet"
    subprocess.run(cmd_resample, shell=True, check=True)

    print(f"2. Transcribing with Whisper...")
    cmd_whisper = [
        args.whisper_bin,
        "-m", args.whisper_model,
        "-f", temp_16k,
        "--output-json",
        "--output-file", whisper_out,
        "-l", "en",
        "-t", str(args.threads)
    ]
    subprocess.run(cmd_whisper, check=True)

    json_path = whisper_out + ".json"
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        whisper_data = json.load(f)

    segments = whisper_data.get("transcription", [])
    print(f"Extracted {len(segments)} candidate segments.")

    metadata_rows = []
    clip_idx = 0

    for seg in segments:
        text = clean_text(seg.get("text", ""))
        if not text or len(text) < 4:
            continue

        t_from = seg.get("timestamps", {}).get("from", "")
        t_to = seg.get("timestamps", {}).get("to", "")
        start_sec = parse_time(t_from)
        end_sec = parse_time(t_to)
        duration = end_sec - start_sec

        # Keep duration in standard Piper sweet spot (1.0s - 12.0s)
        if duration < 1.0 or duration > 12.0:
            continue

        clip_id = f"clip_{clip_idx:05d}"
        clip_path = os.path.join(wavs_dir, f"{clip_id}.wav")

        # Slice and resample to target rate (e.g. 22.05kHz 16-bit PCM)
        cmd_slice = (
            f"ffmpeg -y -ss {start_sec:.3f} -to {end_sec:.3f} -i '{args.input_audio}' "
            f"-ar {args.sample_rate} -ac 1 -c:a pcm_s16le '{clip_path}' -v quiet"
        )
        res = subprocess.run(cmd_slice, shell=True)
        if res.returncode == 0 and os.path.exists(clip_path) and os.path.getsize(clip_path) > 1000:
            metadata_rows.append(f"{clip_id}|{text}|{text}")
            clip_idx += 1

    metadata_path = os.path.join(args.output_dir, "metadata.csv")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("\n".join(metadata_rows) + "\n")

    if os.path.exists(temp_16k):
        os.remove(temp_16k)

    print(f"Preprocessing complete: Generated {clip_idx} clips in {wavs_dir}")
    print(f"Metadata index written to: {metadata_path}")

if __name__ == "__main__":
    main()
