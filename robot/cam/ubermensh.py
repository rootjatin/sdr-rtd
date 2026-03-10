#!/usr/bin/env python3
import argparse
import os
import signal
import time

import cv2
import numpy as np
import pyfakewebcam
from rtlsdr import RtlSdr

running = True


def stop(sig, frame):
    global running
    running = False


def parse_freq(text: str) -> float:
    s = text.strip().lower().replace("_", "")
    mult = 1.0

    if s.endswith("ghz"):
        mult, s = 1e9, s[:-3]
    elif s.endswith("g"):
        mult, s = 1e9, s[:-1]
    elif s.endswith("mhz"):
        mult, s = 1e6, s[:-3]
    elif s.endswith("m"):
        mult, s = 1e6, s[:-1]
    elif s.endswith("khz"):
        mult, s = 1e3, s[:-3]
    elif s.endswith("k"):
        mult, s = 1e3, s[:-1]
    elif s.endswith("hz"):
        mult, s = 1.0, s[:-2]

    return float(s) * mult


def parse_gain(text: str):
    s = str(text).strip().lower()
    if s == "auto":
        return "auto"
    return float(s)


def open_fakecam(device: str, width: int, height: int):
    attempts = [(width, height), (640, 480), (1280, 720)]
    seen = set()

    for w, h in attempts:
        if (w, h) in seen:
            continue
        seen.add((w, h))
        try:
            cam = pyfakewebcam.FakeWebcam(device, w, h)
            return cam, w, h
        except OSError as e:
            print(f"Fake webcam init failed for {w}x{h}: {e}")

    raise SystemExit(
        "Could not open virtual camera. Recreate /dev/video2 with v4l2loopback and try 640x480 first."
    )


class FalseColorRadioView:
    def __init__(
        self,
        width: int,
        height: int,
        fft_size: int = 1024,
        spectrum_alpha: float = 0.70,
        history: int = 720,
        rng_seed: int = 1234,
    ):
        self.width = int(width)
        self.height = int(height)
        self.fft_size = int(fft_size)
        self.spectrum_alpha = float(spectrum_alpha)
        self.history = int(max(128, history))

        self.window = np.hanning(self.fft_size).astype(np.float32)
        self.spectrum_ema = None

        self.waterfall = np.zeros((self.history, self.fft_size), dtype=np.float32)
        self.rng = np.random.default_rng(rng_seed)

    def spectrum_from_samples(self, samples: np.ndarray) -> np.ndarray:
        if samples.size < self.fft_size:
            buf = np.zeros(self.fft_size, dtype=np.complex64)
            buf[:samples.size] = samples
            chunks = buf[None, :]
        else:
            usable = (samples.size // self.fft_size) * self.fft_size
            chunks = samples[:usable].reshape(-1, self.fft_size)

        acc = np.zeros(self.fft_size, dtype=np.float32)

        for chunk in chunks:
            spec = np.fft.fftshift(np.fft.fft(chunk * self.window, n=self.fft_size))
            mag = np.abs(spec) + 1e-12
            power_db = 20.0 * np.log10(mag)
            acc += power_db.astype(np.float32)

        db = acc / max(1, len(chunks))

        if self.spectrum_ema is None:
            self.spectrum_ema = db
        else:
            a = self.spectrum_alpha
            self.spectrum_ema = a * self.spectrum_ema + (1.0 - a) * db

        return self.spectrum_ema.copy()

    def _normalize_row(self, db: np.ndarray) -> np.ndarray:
        lo = np.percentile(db, 8)
        hi = np.percentile(db, 99.2)
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        norm = (db - lo) / (hi - lo)
        norm = np.clip(norm, 0.0, 1.0)

        # gamma for more visible contrast in weak signals
        norm = np.power(norm, 0.72)
        return norm.astype(np.float32)

    def update(self, samples: np.ndarray) -> np.ndarray:
        db = self.spectrum_from_samples(samples)
        row = self._normalize_row(db)

        # shift waterfall
        self.waterfall[:-1] = self.waterfall[1:]
        self.waterfall[-1] = row

        return self.render(row)

    def render(self, current_row: np.ndarray) -> np.ndarray:
        # resize waterfall to frame
        wf = cv2.resize(self.waterfall, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        # moving grain so it doesn't look solid
        grain = self.rng.normal(0.0, 0.035, wf.shape).astype(np.float32)
        wf = np.clip(wf + grain, 0.0, 1.0)

        # emphasize vertical streaks / radio texture
        blur_small = cv2.GaussianBlur(wf, (0, 0), 1.0)
        blur_large = cv2.GaussianBlur(wf, (0, 0), 3.5)
        bandpass = np.clip((blur_small - 0.7 * blur_large) + 0.5, 0.0, 1.0)

        # edge texture
        gx = cv2.Sobel(wf, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(wf, cv2.CV_32F, 0, 1, ksize=3)
        edges = cv2.magnitude(gx, gy)
        if np.max(edges) > 1e-6:
            edges = edges / np.max(edges)
        edges = np.power(edges, 0.8)

        # current spectrum line blended into the image
        line_layer = np.zeros((self.height, self.width), dtype=np.float32)
        xs = np.linspace(0, self.width - 1, self.fft_size).astype(np.int32)
        ys = ((1.0 - current_row) * (self.height - 1)).astype(np.int32)
        pts = np.column_stack([xs, ys]).reshape(-1, 1, 2)
        cv2.polylines(line_layer, [pts], False, 1.0, 2, cv2.LINE_AA)
        line_layer = cv2.GaussianBlur(line_layer, (0, 0), 1.6)

        # combine layers
        img = (
            0.58 * wf +
            0.34 * bandpass +
            0.28 * edges +
            0.52 * line_layer
        )
        img = np.clip(img, 0.0, 1.0)

        # add faint vertical modulation to mimic "scan" feel
        x = np.linspace(0, 1, self.width, dtype=np.float32)
        stripe = (0.96 + 0.04 * np.sin(2 * np.pi * (x * 23.0)))
        img *= stripe[None, :]
        img = np.clip(img, 0.0, 1.0)

        gray = (img * 255.0).astype(np.uint8)

        # false color, then invert
        color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        color = cv2.bitwise_not(color)

        # soften slightly so it feels more organic
        color = cv2.GaussianBlur(color, (0, 0), 0.8)

        # extra texture after invert so it doesn't become flat
        noise = self.rng.normal(0.0, 6.0, color.shape).astype(np.int16)
        color = np.clip(color.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def main():
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    ap = argparse.ArgumentParser(description="RTL-SDR false-color virtual camera")
    ap.add_argument("--device", default="/dev/video2")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=30.0)

    ap.add_argument("--freq", type=parse_freq, default=parse_freq("100.1M"))
    ap.add_argument("--sample-rate", type=parse_freq, default=parse_freq("2.048M"))
    ap.add_argument("--gain", default="auto")
    ap.add_argument("--ppm", type=int, default=0)
    ap.add_argument("--device-index", type=int, default=0)

    ap.add_argument("--fft-size", type=int, default=1024)
    ap.add_argument("--samples-per-frame", type=int, default=0)
    args = ap.parse_args()

    if not os.path.exists(args.device):
        raise SystemExit(f"Device not found: {args.device}")

    cam, cam_w, cam_h = open_fakecam(args.device, args.width, args.height)
    print(f"Virtual camera opened at {cam_w}x{cam_h}")

    renderer = FalseColorRadioView(
        width=cam_w,
        height=cam_h,
        fft_size=args.fft_size,
        history=max(cam_h * 2, 720),
    )

    gain = parse_gain(args.gain)

    sdr = RtlSdr(device_index=args.device_index)
    sdr.sample_rate = int(args.sample_rate)
    sdr.center_freq = int(args.freq)

    ppm = int(args.ppm)
    if ppm != 0:
        try:
            sdr.freq_correction = ppm
        except Exception as e:
            print(f"Warning: could not set PPM correction to {ppm}: {e}")

    sdr.gain = gain

    if args.samples_per_frame > 0:
        samples_per_frame = int(args.samples_per_frame)
    else:
        samples_per_frame = int(args.sample_rate / max(args.fps, 1.0))
        samples_per_frame = max(samples_per_frame, args.fft_size)

    samples_per_frame = max(args.fft_size, samples_per_frame)
    samples_per_frame = min(samples_per_frame, 262144)
    samples_per_frame = (samples_per_frame // args.fft_size) * args.fft_size
    samples_per_frame = max(samples_per_frame, args.fft_size)

    frame_interval = 1.0 / max(args.fps, 1e-6)

    print(f"Streaming false-color radio view to {args.device}")
    print("Press Ctrl+C to stop.")

    try:
        while running:
            t0 = time.perf_counter()

            samples = sdr.read_samples(samples_per_frame)
            rgb = renderer.update(samples)
            cam.schedule_frame(rgb)

            dt = time.perf_counter() - t0
            remain = frame_interval - dt
            if remain > 0:
                time.sleep(remain)

    finally:
        try:
            sdr.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
