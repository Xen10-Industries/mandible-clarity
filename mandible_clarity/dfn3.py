"""Streaming DeepFilterNet3 inference (ONNX Runtime) at 48 kHz.

Follows the libDF reference pipeline:
- Vorbis window, 960-sample FFT, 480-sample hop, spectrum scaled by ``wnorm``.
- 32 ERB band energies in dB with exponential mean normalisation (alpha 0.99).
- First 96 complex bins with exponential unit normalisation.
- ERB gain mask on the whole spectrum (with sinusoidal post-filter), plus 5-tap complex deep filter
  on the lowest 96 bins.
- Local-SNR gating: frames the model rates as pure noise (lsnr < -10 dB) are attenuated to the
  floor, with a smooth ramp up to 0 dB and sub-80 Hz thump damping during low-SNR frames. This gate
  is what removes keyboard taps, clicks and mic bumps between words.

The exported encoder/decoders contain GRUs but don't expose their hidden state, so each hop runs the
graph over a sliding window of the last ``context`` feature frames and keeps only the newest output.
The encoder expects features shifted by ``conv_lookahead`` (2 frames), so the newest output lines up
with the frame 2 hops back: the algorithmic latency is 20 ms window + 20 ms lookahead (~40 ms).

This module is shared with Mandible (``mandible/clarity_dfn3.py``). Keep the two copies identical.
"""
from __future__ import annotations

import math
import os

import numpy as np

SR = 48000
FFT = 960
HOP = 480
NB_BINS = FFT // 2 + 1
NB_ERB = 32
NB_DF = 96
DF_ORDER = 5
LOOKAHEAD = 2             # conv_lookahead == df_lookahead == 2 in the DeepFilterNet3 config
NORM_ALPHA = 0.99         # exp(-hop / sr / tau), tau = 1 s, rounded as libDF does
MIN_DB_THRESH = -10.0     # lsnr below this: frame is noise only, attenuate to the floor
MAX_DB_ERB_THRESH = 30.0  # lsnr above this: clean speech, pass through untouched
MAX_DB_DF_THRESH = 20.0   # lsnr above this: ERB mask only, skip the deep filter

MODEL_FILES = ("enc.onnx", "erb_dec.onnx", "df_dec.onnx")


def _freq2erb(f: float) -> float:
    return 9.265 * math.log1p(f / (24.7 * 9.265))


def _erb2freq(e: float) -> float:
    return 24.7 * 9.265 * (math.exp(e / 9.265) - 1.0)


def erb_widths(sr: int = SR, fft: int = FFT, nb_bands: int = NB_ERB, min_nb_freqs: int = 2) -> list[int]:
    """Number of FFT bins in each ERB band (port of libDF ``erb_fb``)."""
    freq_width = sr / fft
    erb_low, erb_high = _freq2erb(0.0), _freq2erb(sr / 2)
    step = (erb_high - erb_low) / nb_bands
    widths: list[int] = []
    prev, over = 0, 0
    for i in range(1, nb_bands + 1):
        fb = int(round(_erb2freq(erb_low + i * step) / freq_width))
        n = fb - prev - over
        if n < min_nb_freqs:
            over = min_nb_freqs - n
            n = min_nb_freqs
        else:
            over = 0
        widths.append(n)
        prev = fb
    widths[-1] += 1
    too_large = sum(widths) - (fft // 2 + 1)
    if too_large > 0:
        widths[-1] -= too_large
    return widths


def vorbis_window(n: int = FFT) -> np.ndarray:
    k = np.arange(n, dtype=np.float64)
    return np.sin(0.5 * np.pi * np.sin(np.pi * (k + 0.5) / n) ** 2).astype(np.float32)


def find_weights_dir(candidates: list[str]) -> str | None:
    for c in candidates:
        if c and all(os.path.isfile(os.path.join(c, f)) for f in MODEL_FILES):
            return c
    return None


class DFN3Stream:
    """Hop-in / hop-out DeepFilterNet3 enhancer for 48 kHz mono float32 audio.

    ``process_hop`` takes exactly ``HOP`` samples and returns ``HOP`` enhanced samples, delayed by
    ``latency_samples``.
    """

    latency_samples = HOP + LOOKAHEAD * HOP  # window overlap + model lookahead

    def __init__(self, weights_dir: str, context: int = 24, threads: int = 1):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, threads)
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Tiny per-hop graphs: the CPU is faster and steadier than a GPU round trip.
        prov = ["CPUExecutionProvider"]
        self._enc = ort.InferenceSession(os.path.join(weights_dir, "enc.onnx"), opts, providers=prov)
        self._erb = ort.InferenceSession(os.path.join(weights_dir, "erb_dec.onnx"), opts, providers=prov)
        self._df = ort.InferenceSession(os.path.join(weights_dir, "df_dec.onnx"), opts, providers=prov)

        self.context = max(LOOKAHEAD + 1, int(context))
        self._win = vorbis_window()
        self._wnorm = np.float32(2.0 * HOP / (FFT * FFT))

        widths = erb_widths()
        self._band_of_bin = np.repeat(np.arange(NB_ERB), widths)
        self._inv_width = 1.0 / np.asarray(widths, dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        self._analysis = np.zeros(FFT, dtype=np.float32)
        self._synthesis = np.zeros(HOP, dtype=np.float32)
        self._erb_state = np.linspace(-60.0, -90.0, NB_ERB, dtype=np.float32)
        self._spec_state = np.linspace(0.001, 0.0001, NB_DF, dtype=np.float32)
        self._feat_erb = np.zeros((self.context, NB_ERB), dtype=np.float32)
        self._feat_spec = np.zeros((self.context, 2, NB_DF), dtype=np.float32)
        self._spec_buf = np.zeros((DF_ORDER, NB_BINS), dtype=np.complex64)  # oldest first, newest last
        self._primed = False
        self.last_lsnr = 0.0

    # ------------------------------------------------------------------ features
    def _features(self, spec: np.ndarray) -> None:
        power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
        band = np.bincount(self._band_of_bin, weights=power, minlength=NB_ERB).astype(np.float32) * self._inv_width
        erb_db = 10.0 * np.log10(band + 1e-10)
        low = spec[:NB_DF]
        low_abs = np.abs(low)
        if not self._primed:
            self._erb_state = erb_db.copy()
            self._spec_state = np.maximum(low_abs, 1e-4).astype(np.float32)
            self._primed = True
        else:
            self._erb_state = erb_db * (1.0 - NORM_ALPHA) + self._erb_state * NORM_ALPHA
            self._spec_state = low_abs * (1.0 - NORM_ALPHA) + self._spec_state * NORM_ALPHA

        erb_feat = (erb_db - self._erb_state) / 40.0
        low_n = low / np.sqrt(self._spec_state + 1e-10)

        self._feat_erb = np.roll(self._feat_erb, -1, axis=0)
        self._feat_erb[-1] = erb_feat
        self._feat_spec = np.roll(self._feat_spec, -1, axis=0)
        self._feat_spec[-1, 0] = low_n.real
        self._feat_spec[-1, 1] = low_n.imag

    @staticmethod
    def _post_filter(m: np.ndarray, beta: float = 0.04) -> np.ndarray:
        """DeepFilterNet sinusoidal post-filter (port of libDF ``post_filter``)."""
        m_c = np.clip(m, 1e-4, 1.0)
        m_sin = m_c * np.sin(0.5 * np.pi * m_c)
        return ((1.0 + beta) * m_c / (1.0 + beta * (m_c / np.maximum(m_sin, 1e-6)) ** 2)).astype(np.float32)

    # ------------------------------------------------------------------ main hop
    def process_hop(
        self,
        hop: np.ndarray,
        atten_limit_db: float = 100.0,
        wet: float = 1.0,
        pf: bool = True,
    ) -> np.ndarray:
        self._analysis[:HOP] = self._analysis[HOP:]
        self._analysis[HOP:] = hop
        spec = (np.fft.rfft(self._analysis * self._win) * self._wnorm).astype(np.complex64)

        self._features(spec)
        self._spec_buf = np.roll(self._spec_buf, -1, axis=0)
        self._spec_buf[-1] = spec

        fe = self._feat_erb[np.newaxis, np.newaxis]                                 # [1,1,S,32]
        fs = self._feat_spec.transpose(1, 0, 2)[np.newaxis]                         # [1,2,S,96]
        e0, e1, e2, e3, emb, c0, lsnr = self._enc.run(None, {"feat_erb": fe, "feat_spec": fs})
        lsnr_now = float(lsnr[0, -1, 0])
        self.last_lsnr = lsnr_now

        # The newest output belongs to the frame LOOKAHEAD hops back.
        noisy = self._spec_buf[DF_ORDER - 1 - LOOKAHEAD].copy()

        if lsnr_now < MIN_DB_THRESH:
            enhanced = np.zeros_like(noisy)
        elif lsnr_now > MAX_DB_ERB_THRESH:
            enhanced = noisy
        else:
            m_raw = self._erb.run(None, {"emb": emb, "e3": e3, "e2": e2, "e1": e1, "e0": e0})[0][0, 0, -1]
            m = self._post_filter(m_raw, beta=0.04) if pf else m_raw
            enhanced = noisy * m[self._band_of_bin]
            if lsnr_now <= MAX_DB_DF_THRESH:
                coefs = self._df.run(None, {"emb": emb, "c0": c0})[0][0, -1]           # [96, 10]
                c = coefs.reshape(NB_DF, DF_ORDER, 2)
                c = (c[..., 0] + 1j * c[..., 1]).T.astype(np.complex64)             # [order, 96]
                enhanced[:NB_DF] = np.sum(c * self._spec_buf[:, :NB_DF], axis=0)

            # Soft local-SNR gate for negative-SNR transient bursts (-10 dB .. 0 dB)
            if lsnr_now < 0.0:
                vad_g = ((lsnr_now - MIN_DB_THRESH) / (0.0 - MIN_DB_THRESH)) ** 2
                enhanced *= np.float32(vad_g)

            # Sub-80 Hz mic-bump / desk-thump damping when voice harmonic SNR isn't strong
            if lsnr_now < 6.0:
                enhanced[0] = 0.0
                enhanced[1] *= 0.1

        # Attenuation limit + wet/dry: keep a fraction of the noisy signal.
        lim = 10.0 ** (-max(0.0, float(atten_limit_db)) / 20.0)
        keep = 1.0 - float(np.clip(wet, 0.0, 1.0)) * (1.0 - lim)
        out_spec = noisy * keep + enhanced * (1.0 - keep)

        frame = np.fft.irfft(out_spec, n=FFT).astype(np.float32) * np.float32(FFT) * self._win
        out = self._synthesis + frame[:HOP]
        self._synthesis = frame[HOP:].copy()
        return out
