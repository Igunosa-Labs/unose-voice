#!/usr/bin/env python3
"""
Export trained PyTorch Lightning VITS checkpoint to a standalone ONNX graph.
"""

import sys
import os
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Export Piper checkpoint to ONNX.")
    parser.add_argument("--checkpoint", required=True, help="Path to input .ckpt file")
    parser.add_argument("--output", required=True, help="Destination .onnx path")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    print(f"Exporting {args.checkpoint} to {args.output}...")
    cmd = [sys.executable, "-m", "piper_train.export_onnx", args.checkpoint, args.output]
    res = subprocess.run(cmd)

    if res.returncode == 0 and os.path.exists(args.output):
        size_mb = os.path.getsize(args.output) / (1024 * 1024)
        print(f"Successfully exported ONNX model ({size_mb:.2f} MB): {args.output}")
    else:
        print("ONNX export failed.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
