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
    spec_height: int
    db_min: float
    db_max: float
    show_preview: bool


class RTLSDRSource:
    def __init__(self, center_freq, sample_rate, gain):
        from rtlsdr import RtlSdr

        self.sdr = RtlSdr()
        self.sdr.sample_rate = sample_rate
        self.sdr.center_freq = center_freq
        # gain: -1 => auto in many setups; otherwise dB value
        if gain is None or gain < 0:
            try:
                self.sdr.gain = "auto"
            except Exception:
                # fallback if "auto" is not supported in your pyrtlsdr build
                self.sdr.gain = 0
        else:
            self.sdr.gain = gain

        self.sample_rate = float(sample_rate)

    def read(self, n: int) -> np.ndarray:
        # returns complex64
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
    Example args: --backend soapy --soapy-args "driver=hackrf"
    """
    def __init__(self, soapy_args: str, center_freq, sample_rate, gain):
        import SoapySDR
        from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32

        dev = SoapySDR.Device(soapy_args)
        self.dev = dev
        self.sample_rate = float(sample_rate)

        dev.setSampleRate(SOAPY_SDR_RX, 0, self.sample_rate)
        dev.setFrequency(SOAPY_SDR_RX, 0, float(center_freq))

        # Gain handling varies by driver; try a few common approaches
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

        self._buf = np.empty(0, dtype=np.complex64)

        self.SoapySDR = SoapySDR
        self.SOAPY_SDR_RX = SOAPY_SDR_RX

    def read(self, n: int) -> np.ndarray:
        # Read n complex64 samples
        out = np.empty(n, dtype=np.complex64)
        got = 0
        while got < n and running:
            sr = self.dev.readStream(self.stream, [out[got:]], n - got, timeoutUs=200000)
            if sr.ret > 0:
                got += sr.ret
            else:
                # brief backoff on underflow/timeouts
                time.sleep(0.001)
        if got < n:
            out = out[:got]
        return out

    def close(self):
        try:
            self.dev.deactivateStream(self.stream)
            self.dev.closeStream(self.stream)
        except Exception:
            pass


def compute_psd(iq: np.ndarray, fft_size: int, avg: int) -> np.ndarray:
    if iq.size < fft_size:
        iq = np.pad(iq, (0, fft_size - iq.size))

    need = fft_size * max(1, avg)
    if iq.size < need:
        iq = np.pad(iq, (0, need - iq.size))

    blocks = iq[:need].reshape(-1, fft_size)
    w = np.hanning(fft_size).astype(np.float32)
    blocks = blocks * w[None, :]

    spec = np.fft.fftshift(np.fft.fft(blocks, axis=1), axes=1)
    p = 20.0 * np.log10(np.abs(spec) + 1e-12)
    return p.mean(axis=0)  # (fft_size,)


def psd_to_row(psd_db: np.ndarray, out_w: int, db_min: float, db_max: float) -> np.ndarray:
    x = (psd_db - db_min) / max(1e-6, (db_max - db_min))
    x = np.clip(x, 0.0, 1.0)
    row = (x * 255.0).astype(np.uint8)

    if row.size != out_w:
        row = cv2.resize(row.reshape(1, -1), (out_w, 1), interpolation=cv2.INTER_LINEAR).reshape(-1)
    return row


def render_frame(
    psd_db: np.ndarray,
    waterfall_gray: np.ndarray,
    cfg: RenderConfig,
    center_freq: float,
    sample_rate: float,
) -> np.ndarray:
    W, H = cfg.width, cfg.height
    spec_h = cfg.spec_height
    wf_h = H - spec_h

    # Update waterfall (scroll up, add new row at bottom)
    row = psd_to_row(psd_db, W, cfg.db_min, cfg.db_max)
    waterfall_gray[:-1, :] = waterfall_gray[1:, :]
    waterfall_gray[-1, :] = row

    # Colorize waterfall
    wf_bgr = cv2.applyColorMap(waterfall_gray, cv2.COLORMAP_TURBO)
    wf_rgb = cv2.cvtColor(wf_bgr, cv2.COLOR_BGR2RGB)

    # Canvas RGB
    frame = np.zeros((H, W, 3), dtype=np.uint8)

    # Place spectrum background (top) and waterfall (bottom)
    frame[spec_h:H, :, :] = wf_rgb

    # Draw spectrum line on top region
    spec_bg = frame[:spec_h, :, :]
    # Light grid
    for k in range(1, 5):
        x = int(W * k / 5)
        cv2.line(spec_bg, (x, 0), (x, spec_h - 1), (40, 40, 40), 1)
    for k in range(1, 4):
        y = int(spec_h * k / 4)
        cv2.line(spec_bg, (0, y), (W - 1, y), (40, 40, 40), 1)

    yvals = (psd_to_row(psd_db, W, cfg.db_min, cfg.db_max).astype(np.float32) / 255.0)
    y = (spec_h - 1) - (yvals * (spec_h - 1))
    pts = np.stack([np.arange(W, dtype=np.int32), y.astype(np.int32)], axis=1)
    cv2.polylines(spec_bg, [pts], isClosed=False, color=(220, 220, 220), thickness=2)

    # Labels
    cf_mhz = center_freq / 1e6
    sr_mhz = sample_rate / 1e6
    txt = f"CF: {cf_mhz:.6f} MHz | SR: {sr_mhz:.3f} MHz | dB [{cfg.db_min:.0f}, {cfg.db_max:.0f}]"
    cv2.putText(spec_bg, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(spec_bg, "ubermensch (SDR Canvas)", (10, spec_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)

    return frame  # RGB uint8


def main():
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    ap = argparse.ArgumentParser(description="SDR -> Spectrum/Waterfall Canvas -> V4L2 Virtual Camera")
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
    ap.add_argument("--avg", type=int, default=4, help="Averaging blocks per frame")
    ap.add_argument("--spec-height", type=int, default=240, help="Top spectrum panel height (px)")
    ap.add_argument("--db-min", type=float, default=-80.0)
    ap.add_argument("--db-max", type=float, default=-20.0)

    ap.add_argument("--preview", action="store_true", help="Show local preview window (press q to quit)")
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
        spec_height=min(args.spec_height, args.height - 10),
        db_min=args.db_min,
        db_max=args.db_max,
        show_preview=args.preview,
    )

    # SDR init
    if args.backend == "rtlsdr":
        src = RTLSDRSource(center_freq=args.freq, sample_rate=args.rate, gain=args.gain)
    else:
        src = SoapySDRSource(soapy_args=args.soapy_args, center_freq=args.freq, sample_rate=args.rate, gain=args.gain)

    cam = pyfakewebcam.FakeWebcam(args.device, args.width, args.height)

    wf_h = args.height - cfg.spec_height
    waterfall_gray = np.zeros((wf_h, args.width), dtype=np.uint8)

    # For each video frame, read enough samples for averaging
    samples_per_frame = int(max(cfg.fft_size * cfg.avg, (args.rate / max(1.0, args.fps))))
    frame_interval = 1.0 / max(1e-6, cfg.fps)

    print(f"Streaming SDR canvas -> {args.device} ({args.width}x{args.height} @ {cfg.fps} fps)")
    print("Ctrl+C to stop. If --preview is enabled, press 'q' to quit preview.")

    try:
        while running:
            t0 = time.perf_counter()

            iq = src.read(samples_per_frame)
            if iq.size == 0:
                time.sleep(0.01)
                continue

            psd_db = compute_psd(iq, cfg.fft_size, cfg.avg)
            frame_rgb = render_frame(psd_db, waterfall_gray, cfg, center_freq=args.freq, sample_rate=args.rate)

            cam.schedule_frame(frame_rgb)

            if cfg.show_preview:
                cv2.imshow("ubermensch preview", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            dt = time.perf_counter() - t0
            if frame_interval - dt > 0:
                time.sleep(frame_interval - dt)

    finally:
        src.close()
        if cfg.show_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
