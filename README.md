# Mandible Clarity Sidecar

Headless studio AI vocal de-noiser and acoustic dereverberation sidecar for [Mandible](https://github.com/Xen10-Industries/Mandible).

Built for real-time 48 kHz vocal speech enhancement, room reverberation cancellation, and mechanical keyboard/HVAC suppression using DeepFilterNet3 and ONNX Runtime.

---

## Features

- **Full-band 48 kHz audio processing:** Pristine vocal clarity without telecom-muffled bandwidth.
- **Acoustic Dereverberation:** Strips out room reflections and hard-wall flutter echo from untreated recording spaces.
- **Deep Spectral Filtering:** Removes non-stationary noises (mechanical clicks, fans, street traffic) with high SNR improvement.
- **Adjustable Attenuation Limit:** Lets users specify maximum background cut (e.g. -35 dB) for natural ambience without harsh noise gating.
- **Hardware Acceleration:** Supports NVIDIA CUDA GPU, DirectML, or CPU execution via ONNX Runtime.
- **Zero-GUI Headless Design:** Clean CLI and streaming protocol for integration with host racks like Mandible.

---

## Installation & Setup

```powershell
git clone https://github.com/Xen10-Industries/mandible-clarity.git
cd mandible-clarity
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

---

## CLI Usage

### Enhance an audio file
```powershell
python -m mandible_clarity enhance --input "raw_mic.wav" --output "clean_mic.wav" --atten-limit 40 --dereverb
```

### Inspect Hardware & Engine Status
```powershell
python -m mandible_clarity info
```

---

## License

This project is licensed under the [MIT License](LICENSE). DeepFilterNet architecture by Rasti et al.
