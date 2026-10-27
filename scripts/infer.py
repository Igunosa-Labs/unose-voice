#!/usr/bin/env python3
"""
Synthesize speech from text using the exported ONNX model.
"""

import sys
import os
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Synthesize speech with Piper ONNX model.")
    parser.add_argument("--model", required=True, help="Path to .onnx model file")
    parser.add_argument("--config", default=None, help="Path to .onnx.json config (defaults to <model>.json)")
    parser.add_argument("--text", required=True, help="Text to speak")
    parser.add_argument("--output", default="output.wav", help="Output WAV path")
    args = parser.parse_args()

    config_path = args.config if args.config else f"{args.model}.json"
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}", file=sys.stderr)
        sys.exit(1)

    print(f"Synthesizing: \"{args.text}\"")
    cmd = f"echo '{args.text}' | piper -m '{args.model}' -c '{config_path}' -f '{args.output}'"
    res = subprocess.run(cmd, shell=True)

    if res.returncode == 0 and os.path.exists(args.output):
        print(f"Generated audio: {args.output} ({os.path.getsize(args.output)} bytes)")
    else:
        print("Inference failed.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
