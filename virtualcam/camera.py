#!/usr/bin/env python3
import argparse
import os
import signal
import time
from dataclasses import dataclass

import cv2
import numpy as np
import pyfakewebcam

running = True


def stop(sig, frame):
    global running
    running = False


@dataclass
class RenderConfig:
    width: int
    height: int
    fps: float
    fft_size: int
    avg: int
    overlap: float
    spec_height: int

    # dB scaling
    db_min: float
    db_max: float
    auto_range: bool
    auto_lo_pct: float
    auto_hi_pct: float
    auto_smooth: float  # EMA smoothing for auto range

    # PSD smoothing / hold
    psd_ema: float       # 0..1, higher=more smoothing
    peak_hold: bool
    peak_decay: float    # 0..1 per frame (0=no decay, 0.98 slow decay)

    show_preview: bool


class RTLSDRSource:
    def __init__(self, center_freq, sample_rate, gain):
        from rtlsdr import RtlSdr

        self.sdr = RtlSdr()
        self.sdr.sample_rate = sample_rate
        self.sdr.center_freq = center_freq

        if gain is None or gain < 0:
            try:
                self.sdr.gain = "auto"
            except Exception:
                self.sdr.gain = 0
        else:
            self.sdr.gain = gain

        self.sample_rate = float(sample_rate)

    def read(self, n: int) -> np.ndarray:
        x = self.sdr.read_samples(n)
        return np.asarray(x, dtype=np.complex64)

    def close(self):
        try:
            self.sdr.close()
        except Exception:
            pass


class SoapySDRSource:
    """
    HackRF (or other SDRs) via SoapySDR:
    Example: --backend soapy --soapy-args "driver=hackrf"
    """
    def __init__(self, soapy_args: str, center_freq, sample_rate, gain):
        import SoapySDR
        from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32

        dev = SoapySDR.Device(soapy_args)
        self.dev = dev
        self.sample_rate = float(sample_rate)

        dev.setSampleRate(SOAPY_SDR_RX, 0, self.sample_rate)
        dev.setFrequency(SOAPY_SDR_RX, 0, float(center_freq))

        if gain is not None and gain >= 0:
            try:
                dev.setGain(SOAPY_SDR_RX, 0, float(gain))
            except Exception:
                try:
                    dev.setGain(SOAPY_SDR_RX, 0, "LNA", float(gain))
                except Exception:
                    pass

        self.stream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0])
        dev.activateStream(self.stream)

        self.SoapySDR = SoapySDR
        self.SOAPY_SDR_RX = SOAPY_SDR_RX

    def read(self, n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.complex64)
        got = 0
        while got < n and running:
            sr = self.dev.readStream(self.stream, [out[got:]], n - got, timeoutUs=200000)
            if sr.ret > 0:
                got += sr.ret
            else:
                time.sleep(0.001)
        return out[:got] if got < n else out

    def close(self):
        try:
            self.dev.deactivateStream(self.stream)
            self.dev.closeStream(self.stream)
        except Exception:
            pass


class WaterfallRing:
    """Constant-time scrolling waterfall using a ring buffer."""
    def __init__(self, height: int, width: int):
        self.h = height
        self.w = width
        self.buf = np.zeros((height, width), dtype=np.uint8)
        self.write_row = 0  # next row index to write (wraps)

    def push(self, row_u8: np.ndarray):
        self.buf[self.write_row, :] = row_u8
        self.write_row = (self.write_row + 1) % self.h

    def view(self) -> np.ndarray:
        """Return a view ordered from oldest->newest rows (top->bottom)."""
        # Oldest is write_row, newest is write_row-1
        if self.write_row == 0:
            return self.buf
        return np.vstack((self.buf[self.write_row:, :], self.buf[:self.write_row, :]))


def compute_psd_welch(iq: np.ndarray, fft_size: int, avg: int, overlap: float, window: np.ndarray) -> np.ndarray:
    """
    Welch-like PSD:
    - segment length = fft_size
    - hop = fft_size*(1-overlap)
    - avg segments up to `avg`
    Returns dB (fft_size,).
    """
    if iq.size < fft_size:
        iq = np.pad(iq, (0, fft_size - iq.size))

    hop = max(1, int(fft_size * (1.0 - overlap)))
    needed = fft_size + (avg - 1) * hop
    if iq.size < needed:
        iq = np.pad(iq, (0, needed - iq.size))

    # Build segments by striding
    segs = []
    start = 0
    for _ in range(avg):
        segs.append(iq[start:start + fft_size])
        start += hop
    blocks = np.stack(segs, axis=0)  # (avg, fft)

    blocks = blocks * window[None, :]
    spec = np.fft.fftshift(np.fft.fft(blocks, axis=1), axes=1)
    mag = np.abs(spec)
    p_db = 20.0 * np.log10(mag + 1e-12)
    return p_db.mean(axis=0)


def psd_to_row(psd_db: np.ndarray, out_w: int, db_min: float, db_max: float) -> np.ndarray:
    x = (psd_db - db_min) / max(1e-6, (db_max - db_min))
    x = np.clip(x, 0.0, 1.0)
    row = (x * 255.0).astype(np.uint8)
    if row.size != out_w:
        row = cv2.resize(row.reshape(1, -1), (out_w, 1), interpolation=cv2.INTER_LINEAR).reshape(-1)
    return row


def draw_freq_axis(img, center_freq, sample_rate, spec_h, color=(180, 180, 180)):
    """
    Draw ticks & MHz labels at bottom of spectrum panel.
    """
    H, W = img.shape[:2]
    y0 = spec_h - 1
    bw = sample_rate
    f_start = center_freq - bw / 2.0
    f_end = center_freq + bw / 2.0

    # choose ~6 ticks
    ticks = 6
    for k in range(ticks + 1):
        x = int(W * k / ticks)
        f = f_start + (f_end - f_start) * (k / ticks)
        label = f"{f/1e6:.3f}"
        cv2.line(img, (x, y0 - 8), (x, y0), (70, 70, 70), 1)
        cv2.putText(img, label, (max(0, x - 25), y0 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # center marker
    cx = W // 2
    cv2.line(img, (cx, 0), (cx, spec_h - 1), (90, 90, 90), 1)


def render_frame(
    psd_db: np.ndarray,
    wf: WaterfallRing,
    cfg: RenderConfig,
    center_freq: float,
    sample_rate: float,
    db_min: float,
    db_max: float,
) -> np.ndarray:
    W, H = cfg.width, cfg.height
    spec_h = cfg.spec_height
    wf_h = H - spec_h

    # Waterfall update
    row = psd_to_row(psd_db, W, db_min, db_max)
    wf.push(row)

    # Colorize waterfall
    wf_gray = wf.view()
    wf_bgr = cv2.applyColorMap(wf_gray, cv2.COLORMAP_TURBO)
    wf_rgb = cv2.cvtColor(wf_bgr, cv2.COLOR_BGR2RGB)

    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[spec_h:H, :, :] = wf_rgb

    spec_bg = frame[:spec_h, :, :]

    # Grid
    for k in range(1, 5):
        x = int(W * k / 5)
        cv2.line(spec_bg, (x, 0), (x, spec_h - 1), (40, 40, 40), 1)
    for k in range(1, 4):
        y = int(spec_h * k / 4)
        cv2.line(spec_bg, (0, y), (W - 1, y), (40, 40, 40), 1)

    # Spectrum polyline
    yvals = (psd_to_row(psd_db, W, db_min, db_max).astype(np.float32) / 255.0)
    y = (spec_h - 20) - (yvals * (spec_h - 30))
    y = np.clip(y, 0, spec_h - 21)
    pts = np.stack([np.arange(W, dtype=np.int32), y.astype(np.int32)], axis=1)
    cv2.polylines(spec_bg, [pts], isClosed=False, color=(230, 230, 230), thickness=2)

    # Axis labels
    draw_freq_axis(spec_bg, center_freq, sample_rate, spec_h)

    cf_mhz = center_freq / 1e6
    sr_mhz = sample_rate / 1e6
    txt = f"CF: {cf_mhz:.6f} MHz | SR: {sr_mhz:.3f} MHz | dB [{db_min:.0f}, {db_max:.0f}]"
    cv2.putText(spec_bg, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(spec_bg, "ubermensch (SDR Canvas)", (10, spec_h - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)

    return frame


def main():
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    ap = argparse.ArgumentParser(description="SDR -> Spectrum/Waterfall -> V4L2 Virtual Camera")
    ap.add_argument("--device", default="/dev/video2", help="v4l2loopback device path (e.g. /dev/video2)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=float, default=30.0)

    ap.add_argument("--backend", choices=["rtlsdr", "soapy"], default="rtlsdr")
    ap.add_argument("--soapy-args", default="driver=hackrf", help='SoapySDR device args, e.g. "driver=hackrf"')

    ap.add_argument("--freq", type=float, default=100e6, help="Center frequency (Hz)")
    ap.add_argument("--rate", type=float, default=2.4e6, help="Sample rate (Hz)")
    ap.add_argument("--gain", type=float, default=-1, help="Gain in dB (use -1 for auto when available)")

    ap.add_argument("--fft", type=int, default=2048, help="FFT size")
    ap.add_argument("--avg", type=int, default=8, help="Segments to average per frame")
    ap.add_argument("--overlap", type=float, default=0.5, help="Welch overlap [0..0.9]")
    ap.add_argument("--spec-height", type=int, default=240, help="Top spectrum panel height (px)")

    ap.add_argument("--db-min", type=float, default=-80.0)
    ap.add_argument("--db-max", type=float, default=-20.0)

    ap.add_argument("--auto-range", action="store_true", help="Auto dB range via percentiles")
    ap.add_argument("--auto-lo", type=float, default=5.0, help="Low percentile for auto-range (e.g. 5)")
    ap.add_argument("--auto-hi", type=float, default=98.0, help="High percentile for auto-range (e.g. 98)")
    ap.add_argument("--auto-smooth", type=float, default=0.90, help="EMA smoothing for auto-range (0..0.99)")

    ap.add_argument("--psd-ema", type=float, default=0.60, help="EMA smoothing for PSD (0..0.95)")
    ap.add_argument("--peak-hold", action="store_true", help="Enable peak hold trace")
    ap.add_argument("--peak-decay", type=float, default=0.985, help="Peak decay per frame (0..1)")

    ap.add_argument("--preview", action="store_true", help="Show local preview window (keys: q quit, f freeze, a auto-range)")
    args = ap.parse_args()

    if not os.path.exists(args.device):
        raise SystemExit(
            f"Device not found: {args.device}\n"
            f"Load v4l2loopback, e.g.:\n"
            f"  sudo modprobe v4l2loopback video_nr=2 card_label=\"ubermensch\" exclusive_caps=1"
        )

    cfg = RenderConfig(
        width=args.width,
        height=args.height,
        fps=args.fps,
        fft_size=args.fft,
        avg=max(1, args.avg),
        overlap=float(np.clip(args.overlap, 0.0, 0.9)),
        spec_height=min(args.spec_height, args.height - 10),
        db_min=args.db_min,
        db_max=args.db_max,
        auto_range=bool(args.auto_range),
        auto_lo_pct=args.auto_lo,
        auto_hi_pct=args.auto_hi,
        auto_smooth=float(np.clip(args.auto_smooth, 0.0, 0.99)),
        psd_ema=float(np.clip(args.psd_ema, 0.0, 0.95)),
        peak_hold=bool(args.peak_hold),
        peak_decay=float(np.clip(args.peak_decay, 0.0, 1.0)),
        show_preview=args.preview,
    )

    # SDR init
    if args.backend == "rtlsdr":
        src = RTLSDRSource(center_freq=args.freq, sample_rate=args.rate, gain=args.gain)
    else:
        src = SoapySDRSource(soapy_args=args.soapy_args, center_freq=args.freq, sample_rate=args.rate, gain=args.gain)

    cam = pyfakewebcam.FakeWebcam(args.device, args.width, args.height)

    wf_h = args.height - cfg.spec_height
    wf = WaterfallRing(wf_h, args.width)

    # Precompute window
    window = np.hanning(cfg.fft_size).astype(np.float32)

    # Timing
    frame_interval = 1.0 / max(1e-6, cfg.fps)

    # Choose enough samples to supply Welch segments
    hop = max(1, int(cfg.fft_size * (1.0 - cfg.overlap)))
    needed = cfg.fft_size + (cfg.avg - 1) * hop
    # Also ensure we keep up with fps (at least rate/fps)
    samples_per_frame = int(max(needed, args.rate / max(1.0, args.fps)))

    print(f"Streaming SDR canvas -> {args.device} ({args.width}x{args.height} @ {cfg.fps} fps)")
    if cfg.show_preview:
        print("Preview controls: q quit | f freeze | a toggle auto-range")

    # State for smoothing
    psd_smoothed = None
    psd_peak = None
    db_min = cfg.db_min
    db_max = cfg.db_max
    freeze = False

    try:
        while running:
            t0 = time.perf_counter()

            if not freeze:
                iq = src.read(samples_per_frame)
                if iq.size == 0:
                    time.sleep(0.01)
                    continue

                psd_db = compute_psd_welch(iq, cfg.fft_size, cfg.avg, cfg.overlap, window)

                # PSD EMA smoothing
                if psd_smoothed is None:
                    psd_smoothed = psd_db
                else:
                    a = cfg.psd_ema
                    psd_smoothed = a * psd_smoothed + (1.0 - a) * psd_db

                # Peak hold
                if cfg.peak_hold:
                    if psd_peak is None:
                        psd_peak = psd_smoothed.copy()
                    else:
                        psd_peak = np.maximum(psd_peak * cfg.peak_decay, psd_smoothed)

                # Auto-range (percentile-based) with smoothing
                if cfg.auto_range:
                    lo = float(np.percentile(psd_smoothed, cfg.auto_lo_pct))
                    hi = float(np.percentile(psd_smoothed, cfg.auto_hi_pct))
                    # guard
                    if hi - lo < 10.0:
                        hi = lo + 10.0
                    s = cfg.auto_smooth
                    db_min = s * db_min + (1.0 - s) * lo
                    db_max = s * db_max + (1.0 - s) * hi
                else:
                    db_min, db_max = cfg.db_min, cfg.db_max

            # choose what to render: peak or smoothed
            render_psd = psd_peak if (cfg.peak_hold and psd_peak is not None) else psd_smoothed
            if render_psd is None:
                time.sleep(0.01)
                continue

            frame_rgb = render_frame(render_psd, wf, cfg, center_freq=args.freq, sample_rate=args.rate,
                                     db_min=db_min, db_max=db_max)
            cam.schedule_frame(frame_rgb)

            if cfg.show_preview:
                cv2.imshow("ubermensch preview", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                key = (cv2.waitKey(1) & 0xFF)
                if key == ord("q"):
                    break
                elif key == ord("f"):
                    freeze = not freeze
                elif key == ord("a"):
                    cfg.auto_range = not cfg.auto_range

            dt = time.perf_counter() - t0
            if frame_interval - dt > 0:
                time.sleep(frame_interval - dt)

    finally:
        src.close()
        if cfg.show_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
