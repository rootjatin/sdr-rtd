#!/usr/bin/env python3
"""
Real-time spectrum + waterfall renderer for RTL-SDR (including RTL-SDR v3).

Usage examples:
  python rtl_waterfall.py --freq 100e6 --rate 2.4e6 --gain auto
  python rtl_waterfall.py --freq 1090e6 --rate 2.4e6 --gain 40 --ppm 0
"""

import argparse
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

try:
    from rtlsdr import RtlSdr
except Exception as e:
    print("ERROR: Could not import pyrtlsdr. Install with: pip install pyrtlsdr")
    print("Details:", e)
    sys.exit(1)


def dbfs_from_iq(samples: np.ndarray, fft_size: int) -> np.ndarray:
    """Compute FFT power (dB) from complex IQ samples."""
    # Take exactly fft_size samples (or pad if short)
    if len(samples) < fft_size:
        samples = np.pad(samples, (0, fft_size - len(samples)), mode="constant")
    else:
        samples = samples[:fft_size]

    # Window to reduce spectral leakage
    window = np.hanning(fft_size)
    x = samples * window

    # FFT and power
    spec = np.fft.fftshift(np.fft.fft(x, n=fft_size))
    power = np.abs(spec) ** 2

    # Convert to dB (relative)
    power_db = 10.0 * np.log10(power + 1e-12)
    return power_db


def main():
    p = argparse.ArgumentParser(description="RTL-SDR real-time spectrum + waterfall renderer")
    p.add_argument("--freq", type=float, required=True, help="Center frequency in Hz (e.g. 100e6)")
    p.add_argument("--rate", type=float, default=2.4e6, help="Sample rate in Hz (default: 2.4e6)")
    p.add_argument("--gain", type=str, default="auto", help='Gain in dB (e.g. 40) or "auto"')
    p.add_argument("--ppm", type=int, default=0, help="Frequency correction in PPM (default: 0)")
    p.add_argument("--fft", type=int, default=2048, help="FFT size (default: 2048)")
    p.add_argument("--waterfall", type=int, default=300, help="Waterfall history rows (default: 300)")
    p.add_argument("--interval", type=int, default=60, help="Update interval in ms (default: 60)")
    p.add_argument("--vmin", type=float, default=None, help="Waterfall min dB (optional)")
    p.add_argument("--vmax", type=float, default=None, help="Waterfall max dB (optional)")
    args = p.parse_args()

    # --- Open SDR ---
    try:
        sdr = RtlSdr()
    except Exception as e:
        print("ERROR: Could not open RTL-SDR device.")
        print("Tip (Linux): check permissions/udev rules or try running with sudo.")
        print("Details:", e)
        sys.exit(1)

    sdr.sample_rate = args.rate
    sdr.center_freq = args.freq
    sdr.freq_correction = args.ppm

    if args.gain.strip().lower() == "auto":
        sdr.gain = "auto"
    else:
        sdr.gain = float(args.gain)

    # Frequency axis for plotting (Hz)
    freqs = np.linspace(args.freq - args.rate / 2, args.freq + args.rate / 2, args.fft)

    # Waterfall buffer: rows x cols
    wf = np.full((args.waterfall, args.fft), -120.0, dtype=np.float32)

    # --- Matplotlib UI ---
    plt.rcParams["figure.figsize"] = (12, 7)
    fig = plt.figure()
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.4], hspace=0.15)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Spectrum line
    (line,) = ax1.plot(freqs / 1e6, np.zeros_like(freqs), lw=1)
    ax1.set_title("RTL-SDR Spectrum")
    ax1.set_xlabel("Frequency (MHz)")
    ax1.set_ylabel("Power (dB)")
    ax1.grid(True, alpha=0.3)

    # Waterfall image
    img = ax2.imshow(
        wf,
        aspect="auto",
        origin="lower",
        extent=[(freqs[0] / 1e6), (freqs[-1] / 1e6), 0, args.waterfall],
        interpolation="nearest",
        vmin=args.vmin,
        vmax=args.vmax,
    )
    ax2.set_title("Waterfall / Spectrogram")
    ax2.set_xlabel("Frequency (MHz)")
    ax2.set_ylabel("Time (new →)")
    cbar = fig.colorbar(img, ax=ax2)
    cbar.set_label("Power (dB)")

    # Read chunk a bit larger than FFT for smoother updates
    read_len = max(args.fft * 2, 4096)

    last_ok = time.time()

    def update(_frame_idx):
        nonlocal wf, last_ok
        try:
            samples = sdr.read_samples(read_len)
            spec_db = dbfs_from_iq(samples, args.fft)

            # Update spectrum
            line.set_ydata(spec_db)
            ax1.relim()
            ax1.autoscale_view(scalex=False, scaley=True)

            # Update waterfall (scroll)
            wf[:-1, :] = wf[1:, :]
            wf[-1, :] = spec_db.astype(np.float32)
            img.set_data(wf)

            last_ok = time.time()
        except Exception as e:
            # If reads fail, keep UI alive and show a hint occasionally
            if time.time() - last_ok > 2:
                ax1.set_title(f"RTL-SDR Spectrum (read error: {e})")
                last_ok = time.time()

        return line, img

    ani = FuncAnimation(fig, update, interval=args.interval, blit=False)

    try:
        plt.show()
    finally:
        try:
            sdr.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
