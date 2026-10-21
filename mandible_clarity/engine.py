"""Core AI enhancement engine: DeepFilterNet3 ONNX neural inference & Decision-Directed spectral filtering."""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .dfn3 import DFN3Stream, find_weights_dir


@dataclass
class HardwareInfo:
    available: bool
    device_name: str
    backend: str
    supports_cuda: bool
    supports_directml: bool


def detect_hardware() -> HardwareInfo:
    """Detect available hardware accelerator and ONNX execution providers."""
    backend = "NumPy Decision-Directed ERB Engine"
    has_cuda = False
    has_dml = False
    gpu_name = ""

    try:
        import onnxruntime as ort
        providers = list(ort.get_available_providers())
        if "CUDAExecutionProvider" in providers:
            has_cuda = True
            backend = "ONNX Runtime (CUDA GPU)"
        elif "DmlExecutionProvider" in providers:
            has_dml = True
            backend = "ONNX Runtime (DirectML GPU)"
        else:
            backend = "ONNX Runtime (CPU)"
    except Exception:
        pass

    # Detect GPU model name via nvidia-smi
    if sys.platform == "win32":
        try:
            import subprocess
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=1.5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if res.returncode == 0 and res.stdout.strip():
                gpu_name = res.stdout.strip().splitlines()[0].strip()
        except Exception:
            pass

    device_name = gpu_name if gpu_name else ("DirectML / GPU" if has_dml else "Standard CPU")
    return HardwareInfo(
        available=True,
        device_name=device_name,
        backend=backend,
        supports_cuda=has_cuda,
        supports_directml=has_dml,
    )


class ClarityEngine:
    """DeepFilterNet3 neural voice cleaner and dereverberation processor.

    Operates at 48 kHz with 960-sample STFT (20 ms) and 480-sample hop (10 ms).
    Uses DeepFilterNet3 (`DFN3Stream`) when ONNX weights are available, with a
    COLA spectral fallback otherwise.
    """

    FRAME_SIZE: int = 960       # 20 ms at 48 kHz
    HOP_SIZE: int = 480         # 10 ms at 48 kHz (50% overlap)
    SAMPLE_RATE: int = 48000
    NUM_ERB: int = 32

    def __init__(
        self,
        sample_rate: int = 48000,
        model_dir: str | None = None,
        atten_limit_db: float = 100.0,
        dereverb: bool = True,
    ):
        self.sample_rate = sample_rate
        self.atten_limit_db = atten_limit_db
        self.dereverb = dereverb
        self.model_dir = model_dir or self._default_weights_dir()

        self._dfn3: DFN3Stream | None = None

        # Exact COLA periodic sqrt-Hann window for fallback path
        n = np.arange(self.FRAME_SIZE, dtype=np.float32)
        hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * n / self.FRAME_SIZE))
        self._window = np.sqrt(np.maximum(hann, 0.0)).astype(np.float32)

        self._in_buffer = np.zeros(self.FRAME_SIZE, dtype=np.float32)
        self._out_buffer = np.zeros(self.FRAME_SIZE, dtype=np.float32)

        n_bins = self.FRAME_SIZE // 2 + 1
        self._noise_psd = np.ones(n_bins, dtype=np.float32) * 1e-4
        self._prior_speech = np.zeros(n_bins, dtype=np.float32)
        self._reverb_tail = np.zeros(n_bins, dtype=np.float32)
        self._calibrated_frames = 0

        self._init_neural_models()

    @staticmethod
    def _default_weights_dir() -> str:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")

    def _init_neural_models(self) -> None:
        """Attempt to load DeepFilterNet3 ONNX models if available."""
        wdir = find_weights_dir([self.model_dir]) if self.model_dir else None
        if wdir is not None:
            try:
                self._dfn3 = DFN3Stream(wdir)
            except Exception:
                self._dfn3 = None

    def reset(self) -> None:
        """Clear buffers and reset filter tracking state."""
        if self._dfn3 is not None:
            self._dfn3.reset()
        self._in_buffer.fill(0.0)
        self._out_buffer.fill(0.0)
        self._noise_psd.fill(1e-4)
        self._prior_speech.fill(0.0)
        self._reverb_tail.fill(0.0)
        self._calibrated_frames = 0

    def process_chunk(
        self,
        chunk: np.ndarray,
        wet: float = 1.0,
        atten_limit_db: float | None = None,
        dereverb: bool | None = None,
    ) -> np.ndarray:
        """Process 1D float32 audio block with DeepFilterNet3 spectral enhancement."""
        if len(chunk) == 0 or wet <= 0.001:
            return chunk

        atten_lim = self.atten_limit_db if atten_limit_db is None else atten_limit_db
        do_dereverb = self.dereverb if dereverb is None else dereverb
        min_gain = 10.0 ** (-float(atten_lim) / 20.0)

        samples = chunk.astype(np.float32, copy=False)
        out = np.zeros_like(samples)

        pos = 0
        hop = self.HOP_SIZE
        win = self._window
        n_fft = self.FRAME_SIZE

        if self._dfn3 is not None:
            while pos + hop <= len(samples):
                out[pos : pos + hop] = self._dfn3.process_hop(
                    samples[pos : pos + hop],
                    atten_limit_db=atten_lim,
                    wet=wet,
                    pf=do_dereverb,
                )
                pos += hop
        else:
            while pos + hop <= len(samples):
                self._in_buffer[:-hop] = self._in_buffer[hop:]
                self._in_buffer[-hop:] = samples[pos : pos + hop]

                frame = self._in_buffer * win
                spec = np.fft.rfft(frame)
                mag = np.abs(spec)
                psd = mag ** 2
                phase = np.angle(spec)

                self._calibrated_frames += 1
                if self._calibrated_frames < 20:
                    self._noise_psd = 0.8 * self._noise_psd + 0.2 * psd
                    gain = np.ones_like(mag) * min_gain
                else:
                    post_snr = np.maximum(psd / np.maximum(self._noise_psd, 1e-9), 1e-4)
                    mean_vocal_snr = np.mean(post_snr[4:70])
                    is_speaking = mean_vocal_snr > 3.0

                    if not is_speaking:
                        self._noise_psd = 0.92 * self._noise_psd + 0.08 * psd
                    else:
                        quiet_bins = post_snr < 1.8
                        self._noise_psd = np.where(quiet_bins, 0.98 * self._noise_psd + 0.02 * psd, self._noise_psd)

                    gain = np.sqrt(np.maximum(1.0 - 2.2 / np.maximum(post_snr, 1.0), 0.0))

                    if do_dereverb:
                        self._reverb_tail = 0.82 * self._reverb_tail + 0.18 * (mag * gain)
                        rev_ratio = (0.35 * self._reverb_tail) / np.maximum(mag, 1e-8)
                        gain = gain * np.clip(1.0 - rev_ratio, 0.35, 1.0)

                    gain = np.convolve(gain, np.array([0.15, 0.70, 0.15], dtype=np.float32), mode="same")
                    gain = np.clip(gain, min_gain, 1.0)

                    if wet < 0.999:
                        gain = (1.0 - wet) + wet * gain

                    self._prior_speech = (mag * gain) ** 2

                enhanced_spec = (mag * gain) * np.exp(1j * phase)
                enhanced_frame = np.fft.irfft(enhanced_spec, n=n_fft).real.astype(np.float32) * win

                self._out_buffer += enhanced_frame
                out[pos : pos + hop] = self._out_buffer[:hop]
                self._out_buffer[:-hop] = self._out_buffer[hop:]
                self._out_buffer[-hop:] = 0.0

                pos += hop

        if pos < len(samples):
            out[pos:] = samples[pos:] * (1.0 - wet * 0.5)

        return out.astype(np.float32)

    def enhance_file(
        self,
        input_path: str,
        output_path: str,
        atten_limit_db: float = 100.0,
        dereverb: bool = True,
        progress_cb: Callable[[float], None] | None = None,
    ) -> None:
        """Enhance an audio file and write the de-noised vocal audio to output_path."""
        import pedalboard.io

        with pedalboard.io.AudioFile(input_path) as fin:
            sr = fin.samplerate
            channels = fin.num_channels
            audio = fin.read(fin.frames)

        mono = np.mean(audio, axis=0) if channels > 1 else audio[0]
        if sr != self.SAMPLE_RATE:
            resampler = pedalboard.io.StreamResampler(sr, self.SAMPLE_RATE, 1)
            mono_48k = resampler.process(mono.reshape(1, -1))[0]
        else:
            mono_48k = mono

        self.reset()
        block_size = 4800
        enhanced_chunks: list[np.ndarray] = []
        total_len = len(mono_48k)
        for i in range(0, total_len, block_size):
            block = mono_48k[i : i + block_size]
            cleaned = self.process_chunk(block, wet=1.0, atten_limit_db=atten_limit_db, dereverb=dereverb)
            enhanced_chunks.append(cleaned)
            if progress_cb:
                progress_cb(min(1.0, (i + len(block)) / float(total_len)))

        cleaned_full = np.concatenate(enhanced_chunks)
        if sr != self.SAMPLE_RATE:
            up_resampler = pedalboard.io.StreamResampler(self.SAMPLE_RATE, sr, 1)
            out_audio = up_resampler.process(cleaned_full.reshape(1, -1))
        else:
            out_audio = cleaned_full.reshape(1, -1)

        if channels > 1:
            out_audio = np.repeat(out_audio, channels, axis=0)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with pedalboard.io.AudioFile(output_path, "w", samplerate=sr, num_channels=channels) as fout:
            fout.write(out_audio)
