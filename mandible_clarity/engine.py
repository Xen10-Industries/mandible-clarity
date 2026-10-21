"""Core AI enhancement engine: DeepFilterNet3 ONNX neural inference & Decision-Directed spectral filtering."""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np


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
    Uses periodic square-root Hann window for exact Constant Overlap-Add (COLA)
    with zero amplitude modulation or buzzing distortion.
    """

    FRAME_SIZE: int = 960       # 20 ms at 48 kHz
    HOP_SIZE: int = 480         # 10 ms at 48 kHz (50% overlap)
    SAMPLE_RATE: int = 48000
    NUM_ERB: int = 32

    def __init__(
        self,
        sample_rate: int = 48000,
        model_dir: str | None = None,
        atten_limit_db: float = 35.0,
        dereverb: bool = True,
    ):
        self.sample_rate = sample_rate
        self.atten_limit_db = atten_limit_db
        self.dereverb = dereverb
        self.model_dir = model_dir or self._default_weights_dir()

        self.sess_enc = None
        self.sess_erb = None

        # Exact COLA periodic sqrt-Hann window
        n = np.arange(self.FRAME_SIZE, dtype=np.float32)
        hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * n / self.FRAME_SIZE))
        self._window = np.sqrt(np.maximum(hann, 0.0)).astype(np.float32)

        self._in_buffer = np.zeros(self.FRAME_SIZE, dtype=np.float32)
        self._out_buffer = np.zeros(self.FRAME_SIZE, dtype=np.float32)

        n_bins = self.FRAME_SIZE // 2 + 1
        self._noise_psd = np.ones(n_bins, dtype=np.float32) * 1e-4
        self._prior_speech = np.zeros(n_bins, dtype=np.float32)
        self._reverb_tail = np.zeros(n_bins, dtype=np.float32)

        # Normalization states for DeepFilterNet3
        self._erb_state = np.linspace(-60.0, -90.0, self.NUM_ERB, dtype=np.float32)
        self._spec_state = np.linspace(0.001, 0.0001, 96, dtype=np.float32)
        self._norm_alpha = 0.95
        self._calibrated_frames = 0

        self._erb_fb, self._inv_erb = self._build_erb_filterbank(self.sample_rate, self.FRAME_SIZE, self.NUM_ERB)
        self._init_neural_models()

    @staticmethod
    def _default_weights_dir() -> str:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")

    def _init_neural_models(self) -> None:
        """Attempt to load DeepFilterNet3 ONNX models if available."""
        if not self.model_dir or not os.path.isdir(self.model_dir):
            return

        enc_path = os.path.join(self.model_dir, "enc.onnx")
        erb_path = os.path.join(self.model_dir, "erb_dec.onnx")

        if os.path.isfile(enc_path) and os.path.isfile(erb_path):
            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                active = [p for p in ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"] if p in providers]
                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.sess_enc = ort.InferenceSession(enc_path, sess_options=opts, providers=active)
                self.sess_erb = ort.InferenceSession(erb_path, sess_options=opts, providers=active)
            except Exception:
                self.sess_enc = None
                self.sess_erb = None

    @staticmethod
    def _build_erb_filterbank(sr: int, n_fft: int, nb_erb: int) -> tuple[np.ndarray, np.ndarray]:
        """Construct forward and inverse ERB filterbanks for 48 kHz audio."""
        n_bins = n_fft // 2 + 1
        erb_low = 9.265 * math.log(1.0 + 20.0 / (24.7 * 9.265))
        erb_high = 9.265 * math.log(1.0 + (sr / 2.0) / (24.7 * 9.265))
        step = (erb_high - erb_low) / nb_erb
        freq_width = sr / n_fft

        widths: list[int] = []
        prev_f = 0
        for i in range(1, nb_erb + 1):
            f = 24.7 * 9.265 * (math.exp((erb_low + i * step) / 9.265) - 1.0)
            fb = int(round(f / freq_width))
            w = max(1, fb - prev_f)
            widths.append(w)
            prev_f = fb

        diff = sum(widths) - n_bins
        widths[-1] -= diff

        erb_fb = np.zeros((n_bins, nb_erb), dtype=np.float32)
        inv_erb = np.zeros((nb_erb, n_bins), dtype=np.float32)
        idx = 0
        for b, w in enumerate(widths):
            erb_fb[idx : idx + w, b] = 1.0 / max(1, w)
            inv_erb[b, idx : idx + w] = 1.0
            idx += w

        return erb_fb, inv_erb

    def reset(self) -> None:
        """Clear buffers and reset filter tracking state."""
        self._in_buffer.fill(0.0)
        self._out_buffer.fill(0.0)
        self._noise_psd.fill(1e-4)
        self._prior_speech.fill(0.0)
        self._reverb_tail.fill(0.0)
        self._erb_state = np.linspace(-60.0, -90.0, self.NUM_ERB, dtype=np.float32)
        self._spec_state = np.linspace(0.001, 0.0001, 96, dtype=np.float32)
        self._calibrated_frames = 0

    def process_chunk(
        self,
        chunk: np.ndarray,
        wet: float = 1.0,
        atten_limit_db: float | None = None,
        dereverb: bool | None = None,
    ) -> np.ndarray:
        """Process 1D float32 audio block with DeepFilterNet spectral enhancement."""
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
                gain = None

                # 1. Neural DeepFilterNet3 ONNX inference
                if self.sess_enc is not None and self.sess_erb is not None:
                    try:
                        erb_energy = np.matmul(psd, self._erb_fb)
                        erb_db = 10.0 * np.log10(np.maximum(erb_energy, 1e-10))
                        self._erb_state = erb_db * (1.0 - self._norm_alpha) + self._erb_state * self._norm_alpha
                        norm_erb = ((erb_db - self._erb_state) / 40.0)[np.newaxis, np.newaxis, np.newaxis, :].astype(np.float32)

                        spec_mag = np.abs(spec[:96])
                        self._spec_state = np.sqrt(spec_mag * (1.0 - self._norm_alpha) + (self._spec_state ** 2) * self._norm_alpha)
                        norm_real = spec[:96].real / np.maximum(self._spec_state, 1e-6)
                        norm_imag = spec[:96].imag / np.maximum(self._spec_state, 1e-6)
                        feat_spec = np.stack([norm_real, norm_imag], axis=0)[np.newaxis, :, np.newaxis, :].astype(np.float32)

                        e0, e1, e2, e3, emb, c0, lsnr = self.sess_enc.run(
                            None, {"feat_erb": norm_erb, "feat_spec": feat_spec}
                        )
                        m = self.sess_erb.run(None, {"emb": emb, "e3": e3, "e2": e2, "e1": e1, "e0": e0})[0]
                        erb_mask = np.clip(m[0, 0, 0, :], 0.0, 1.0)
                        gain = np.matmul(erb_mask, self._inv_erb)
                    except Exception:
                        gain = None

                # 2. Decision-Directed Ephraim-Malah / VAD engine (fallback or standalone)
                if gain is None:
                    post_snr = np.maximum(psd / np.maximum(self._noise_psd, 1e-9), 1e-4)
                    prior_snr = 0.94 * (self._prior_speech / np.maximum(self._noise_psd, 1e-9)) + 0.06 * np.maximum(
                        post_snr - 1.0, 0.0
                    )
                    prior_snr = np.maximum(prior_snr, 1e-4)

                    mean_vocal_snr = np.mean(post_snr[4:70])
                    is_speaking = mean_vocal_snr > 3.0

                    if not is_speaking:
                        self._noise_psd = 0.92 * self._noise_psd + 0.08 * psd
                    else:
                        quiet_bins = post_snr < 1.8
                        self._noise_psd = np.where(quiet_bins, 0.98 * self._noise_psd + 0.02 * psd, self._noise_psd)

                    gain = np.maximum(1.0 - 2.2 / np.maximum(post_snr, 1.0), 0.0)
                    gain = np.sqrt(gain)

                # 3. Acoustic dereverberation
                if do_dereverb:
                    self._reverb_tail = 0.82 * self._reverb_tail + 0.18 * (mag * gain)
                    rev_ratio = (0.35 * self._reverb_tail) / np.maximum(mag, 1e-8)
                    gain = gain * np.clip(1.0 - rev_ratio, 0.35, 1.0)

                # 4. Multi-band frequency smoothing across neighboring bins
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
        atten_limit_db: float = 35.0,
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
