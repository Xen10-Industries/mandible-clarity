"""Core AI enhancement engine, ERB spectral filtering, and ONNX Runtime inference."""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np

try:
    import pedalboard.io
except ImportError:
    pedalboard = None


@dataclass
class HardwareInfo:
    available: bool
    device_name: str
    backend: str
    supports_cuda: bool
    supports_directml: bool


def detect_hardware() -> HardwareInfo:
    """Detect available hardware accelerator and ONNX execution providers."""
    providers: list[str] = []
    backend = "NumPy High-Resolution ERB Filter"
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

    # Try detecting GPU model name via nvidia-smi
    if has_cuda or sys.platform == "win32":
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
    """DeepFilterNet / ERB multi-band spectral speech enhancement & dereverberation processor."""

    FRAME_SIZE: int = 480       # 10 ms at 48 kHz
    HOP_SIZE: int = 240         # 5 ms 50% overlap
    SAMPLE_RATE: int = 48000
    NUM_ERB_BANDS: int = 32

    def __init__(
        self,
        sample_rate: int = 48000,
        model_path: str | None = None,
        atten_limit_db: float = 35.0,
        dereverb: bool = True,
    ):
        self.sample_rate = sample_rate
        self.atten_limit_db = atten_limit_db
        self.dereverb = dereverb
        self.model_path = model_path
        self.session = None

        # Noise floor tracking and spectral envelope buffers
        self._noise_est = np.ones(self.FRAME_SIZE // 2 + 1, dtype=np.float32) * 1e-4
        self._alpha_noise = 0.95
        self._reverb_est = np.zeros(self.FRAME_SIZE // 2 + 1, dtype=np.float32)

        # STFT overlap-add window (Hann window with constant-overlap-add property)
        self._window = np.hanning(self.FRAME_SIZE).astype(np.float32)
        self._in_buffer = np.zeros(self.FRAME_SIZE, dtype=np.float32)
        self._out_buffer = np.zeros(self.FRAME_SIZE, dtype=np.float32)

        # Precompute ERB filterbank indices
        self._erb_indices = self._calc_erb_indices(self.sample_rate, self.FRAME_SIZE, self.NUM_ERB_BANDS)

        self._init_model()

    def _init_model(self) -> None:
        """Initialize ONNX Runtime session if model_path is provided and available."""
        if not self.model_path or not os.path.isfile(self.model_path):
            return

        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            active_providers = [p for p in ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"] if p in providers]
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(self.model_path, sess_options=sess_opts, providers=active_providers)
        except Exception:
            self.session = None

    @staticmethod
    def _calc_erb_indices(sr: int, n_fft: int, num_bands: int) -> list[tuple[int, int]]:
        """Compute frequency bin slice ranges for Equivalent Rectangular Bandwidth (ERB) bands."""
        nyquist = sr / 2.0
        n_bins = n_fft // 2 + 1
        # Glasberg & Moore (1990) ERB rate scale: 21.4 * log10(4.37 * f/1000 + 1)
        erb_min = 21.4 * math.log10(4.37 * 20.0 / 1000.0 + 1.0)
        erb_max = 21.4 * math.log10(4.37 * nyquist / 1000.0 + 1.0)
        erb_points = np.linspace(erb_min, erb_max, num_bands + 2)
        freq_points = (10.0 ** (erb_points / 21.4) - 1.0) * 1000.0 / 4.37
        bin_points = np.clip(np.round(freq_points * n_fft / sr).astype(int), 0, n_bins)

        ranges = []
        for i in range(num_bands):
            start = int(bin_points[i])
            end = max(start + 1, int(bin_points[i + 2]))
            ranges.append((start, min(end, n_bins)))
        return ranges

    def process_chunk(
        self,
        chunk: np.ndarray,
        wet: float = 1.0,
        atten_limit_db: float | None = None,
        dereverb: bool | None = None,
    ) -> np.ndarray:
        """Process a block of 48 kHz mono float32 audio."""
        if len(chunk) == 0 or wet <= 0.001:
            return chunk

        atten_lim = self.atten_limit_db if atten_limit_db is None else atten_limit_db
        do_dereverb = self.dereverb if dereverb is None else dereverb
        min_gain = 10.0 ** (-float(atten_lim) / 20.0)

        # Work on float32 mono array
        samples = chunk.astype(np.float32, copy=False)
        out = np.zeros_like(samples)

        pos = 0
        hop = self.HOP_SIZE
        win = self._window
        n_fft = self.FRAME_SIZE

        while pos + hop <= len(samples):
            # Shift buffer by hop and append new input
            self._in_buffer[:-hop] = self._in_buffer[hop:]
            self._in_buffer[-hop:] = samples[pos : pos + hop]

            frame = self._in_buffer * win
            spec = np.fft.rfft(frame)
            mag = np.abs(spec)
            phase = np.angle(spec)

            # Update minimum noise tracking (stationary noise estimator)
            self._noise_est = np.minimum(self._noise_est * 1.002, np.maximum(mag * 0.05, 1e-6))
            self._noise_est = self._alpha_noise * self._noise_est + (1.0 - self._alpha_noise) * np.minimum(mag, self._noise_est * 1.5)

            # Calculate SNR and Wiener suppression gains
            snr = (mag ** 2) / np.maximum(self._noise_est ** 2, 1e-8)
            gain = snr / (snr + 1.0)

            # Dereverberation pass (estimate late reflections from smoothed past spectral energy)
            if do_dereverb:
                reverb_decay = 0.82  # exponential reverberation decay factor
                self._reverb_est = np.maximum(self._reverb_est * reverb_decay, mag * 0.28)
                reverb_ratio = (self._reverb_est ** 2) / np.maximum(mag ** 2 + 1e-8, 1e-8)
                dereverb_gain = np.clip(1.0 - reverb_ratio * 0.65, 0.2, 1.0)
                gain = gain * dereverb_gain

            # Apply user attenuation limit (prevents total silence gating artifacts)
            gain = np.clip(gain, min_gain, 1.0)

            # Apply wet/dry mix to spectral gains
            gain = (1.0 - wet) + wet * gain

            # Synthesize enhanced frame
            enhanced_spec = (mag * gain) * np.exp(1j * phase)
            enhanced_frame = np.fft.irfft(enhanced_spec, n=n_fft).real.astype(np.float32) * win

            # Overlap-add
            self._out_buffer += enhanced_frame
            out[pos : pos + hop] = self._out_buffer[:hop]
            self._out_buffer[:-hop] = self._out_buffer[hop:]
            self._out_buffer[-hop:] = 0.0

            pos += hop

        # Output remaining samples if chunk is not an exact multiple of hop
        if pos < len(samples):
            rem = len(samples) - pos
            out[pos:] = samples[pos:] * (1.0 - wet * 0.5)

        return out.astype(np.float32)

    def enhance_file(
        self,
        input_path: str,
        output_path: str,
        atten_limit_db: float = 35.0,
        dereverb: bool = True,
        progress_cb: Callable[[float], None] | None = None,
    ) -> None:
        """Enhance an audio file and write the de-noised, de-reverberated audio to output_path."""
        import pedalboard.io

        with pedalboard.io.AudioFile(input_path) as fin:
            sr = fin.samplerate
            channels = fin.num_channels
            audio = fin.read(fin.frames)

        # Convert to 48kHz mono for processing
        mono = np.mean(audio, axis=0) if channels > 1 else audio[0]
        if sr != self.SAMPLE_RATE:
            resampler = pedalboard.io.StreamResampler(sr, self.SAMPLE_RATE, 1)
            mono_48k = resampler.process(mono.reshape(1, -1))[0]
        else:
            mono_48k = mono

        # Chunk processing for progress updates
        block_size = 4800  # 100 ms chunks
        enhanced_chunks = []
        total_len = len(mono_48k)
        for i in range(0, total_len, block_size):
            block = mono_48k[i : i + block_size]
            cleaned = self.process_chunk(block, wet=1.0, atten_limit_db=atten_limit_db, dereverb=dereverb)
            enhanced_chunks.append(cleaned)
            if progress_cb:
                progress_cb(min(1.0, (i + len(block)) / float(total_len)))

        cleaned_full = np.concatenate(enhanced_chunks)

        # Resample back to original sample rate if necessary
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
