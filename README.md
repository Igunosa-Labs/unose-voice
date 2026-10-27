# Unose-Voice: Neural Voice Synthesis & Adaptation Toolkit

**Unose-Voice** provides an end-to-end pipeline for preparing speech datasets, fine-tuning VITS acoustic models via Piper, and exporting lightweight standalone ONNX graphs for low-latency CPU and edge inference.

Developed by **Igunosa Labs**.

---

## Pretrained Models

| Model | Architecture | Checkpoint | Target | Hugging Face Hub |
| :--- | :--- | :--- | :--- | :--- |
| **Unose-Voice-v1.0** | VITS ONNX (60.5 MB) | `en_US-ryan-medium` adaptation | CPU / GPU / edge | [igunosalabs/Unose-Voice-v1.0](https://huggingface.co/igunosalabs/Unose-Voice-v1.0) |

---

## Requirements & Setup

### Prerequisites
- Python 3.10+
- FFmpeg
- eSpeak NG

```bash
# Clone the repository
git clone https://github.com/Igunosa-Labs/unose-voice.git
cd unose-voice

# Install piper CLI for inference
pip install piper-tts

# For training & export environments (GPU recommended)
pip install piper-train
```

---

## Workflow

### 1. Dataset Preprocessing
Split long continuous studio or interview recordings into phonetically aligned, normalized audio segments (22,050 Hz 16-bit mono PCM):

```bash
python3 prep/preprocess.py \
    --input-audio /path/to/raw_audio.wav \
    --output-dir ./dataset \
    --whisper-model /path/to/ggml-small.bin
```

This outputs:
- `dataset/wavs/`: Sliced audio segments (1.0s to 12.0s duration).
- `dataset/metadata.csv`: LJSpeech-format alignment index (`clip_id|text|text`).

### 2. Model Fine-Tuning
Adapt a base VITS checkpoint to the custom speaker:

```bash
bash recipes/train.sh ./dataset ./training_dir ./checkpoints/en_US-ryan-medium.ckpt 16 500
```

### 3. Export to Standalone ONNX
Convert the PyTorch Lightning checkpoint to an optimized ONNX computational graph:

```bash
python3 scripts/export_onnx.py \
    --checkpoint ./training_dir/lightning_logs/version_0/checkpoints/best.ckpt \
    --output ./models/unose_voice.onnx
```

### 4. Inference

#### Command-Line Inference:
```bash
echo "Hello, this is a test of the neural voice model." | \
  piper -m ./models/unose_voice.onnx -f output.wav
```

#### Python Inference:
```bash
python3 scripts/infer.py \
    --model ./models/unose_voice.onnx \
    --text "Hello, this is a test of the neural voice model." \
    --output output.wav
```

---

## License

Developed by **Igunosa Labs**. Released under the MIT License.
Base Piper architecture copyright Rhasspy / Michael Hansen.
