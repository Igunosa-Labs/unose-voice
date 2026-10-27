# Unose Voice 🎙️🏛️
*Edge Neural Voice Synthesis Research & Sovereign Accented Models by Igunosa Labs*

[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-igunosalabs-yellow)](https://huggingface.co/igunosalabs)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Unose Voice** is an open neural voice cloning and speech research suite created by **Patrick Aigbogun** and the **Igunosa Labs** research team. It focuses on zero-latency, ultra-low memory footprint, sovereign voice synthesis running directly on commodity edge CPU and ARM64 servers.

---

## 🚀 Model Releases

| Model | Checkpoint Architecture | Fine-Tuning Duration | Key Milestones | Hugging Face Repository |
| :--- | :--- | :--- | :--- | :--- |
| **`Unose-Voice-v1.0`** | Native Male Base (`en_US-ryan-medium`) | **500 Epochs** | Zero transposition tremor, full baritone fundamental frequency ($F_0$), 12-min studio take | [igunosalabs/Unose-Voice-v1.0](https://huggingface.co/igunosalabs/Unose-Voice-v1.0) |
| **`Unose-Voice-v0.5`** | Female Base (`en_US-lessac-medium`) | 250 Epochs | Proof of concept adaptation, 60.5MB ONNX graph | [igunosalabs/Unose-Voice-v0.5](https://huggingface.co/igunosalabs/Unose-Voice-v0.5) |

---

## 🔬 Empirical Architecture & Science

### Acoustic Transposition Ceiling (The Female Base Post-Mortem)
In `v0.5`, adaptation was performed on `en_US-lessac-medium` ($F_0 \approx 210–240\text{ Hz}$). Adapting down by $>100\text{ Hz}$ to a natural male baritone forced the HiFi-GAN vocoder upsampling convolutions to extrapolate outside their learned receptive field, generating pitch tremor and harmonic buzz. 

In `v1.0`, transitioning to `en_US-ryan-medium` ($F_0 \approx 110–130\text{ Hz}$) resolved the pitch mismatch natively. Coupled with a 132-slice phonetically balanced master studio dataset, `v1.0` delivers natural timbre and cadence at 60.5 MB.

---

## 💻 Quickstart (Piper Inference)

```bash
# 1. Install Piper
curl -sSL https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz | tar -xz

# 2. Download Model & Config
wget https://huggingface.co/igunosalabs/Unose-Voice-v1.0/resolve/main/Unose-Voice-v1.0.onnx
wget https://huggingface.co/igunosalabs/Unose-Voice-v1.0/resolve/main/Unose-Voice-v1.0.onnx.json

# 3. Synthesize Speech
echo "Intelligence is kinetic. It moves, observes, and manipulates the physical universe." | \
  ./piper/piper -m Unose-Voice-v1.0.onnx -f output.wav
```

---

## 📜 License
MIT License. Copyright (c) 2026 Igunosa Labs.
