import argparse
import numpy as np
import sounddevice as sd
from rtlsdr import RtlSdr
from scipy.signal import butter, lfilter, resample_poly

def main():
    ap = argparse.ArgumentParser(description="RTL-SDR RX: receive HackRF NBFM and play audio.")
    ap.add_argument("--freq", type=float, default=433.92e6, help="HackRF center frequency (Hz), default 433.92e6")
    ap.add_argument("--offset", type=float, default=100e3, help="TX offset (Hz), default 100e3")
    ap.add_argument("--sr", type=float, default=1.536e6, help="RTL sample rate (Hz). Default 1.536e6 (exact /32 -> 48k)")
    ap.add_argument("--gain", type=float, default=30.0, help="RTL gain dB, or -1 for auto (default 30)")
    ap.add_argument("--ppm", type=int, default=0, help="Frequency correction PPM (default 0)")
    ap.add_argument("--audio_rate", type=int, default=48000, help="Audio rate (default 48000)")
    ap.add_argument("--squelch_db", type=float, default=None,
                    help="Optional squelch threshold in dBFS-ish (e.g. -35). If None, squelch off.")
    args = ap.parse_args()

    tune_freq = args.freq + args.offset  # main signal is at center+offset
    fs = float(args.sr)
    audio_rate = int(args.audio_rate)

    # Choose rational resampling from fs -> audio_rate
    g = np.gcd(int(fs), audio_rate)
    up = audio_rate // g
    down = int(fs) // g

    # RTL-SDR setup
    sdr = RtlSdr()
    sdr.sample_rate = fs
    sdr.center_freq = tune_freq
    sdr.gain = "auto" if args.gain < 0 else args.gain

    print(f"[RX] Tuning to {tune_freq/1e6:.6f} MHz (freq+offset)")
    print(f"[RX] SR={fs/1e6:.3f} Msps, Gain={sdr.gain}, PPM={args.ppm}")
    print("[RX] Mode: NBFM demod. Ctrl+C to stop.")
    if args.squelch_db is not None:
        print(f"[RX] Squelch ON at {args.squelch_db} dB (approx).")
    else:
        print("[RX] Squelch OFF.")

    # Lowpass for NBFM audio-ish (keep ~12 kHz)
    # Butterworth IIR; keep filter state between blocks
    b, a = butter(5, 12000.0 / (0.5 * fs))
    zi = np.zeros(max(len(a), len(b)) - 1, dtype=np.float32)

    block = 256 * 1024

    # Audio output stream
    stream = sd.OutputStream(
        samplerate=audio_rate,
        channels=1,
        dtype="float32",
        blocksize=0  # let sounddevice choose
    )
    stream.start()

    try:
        while True:
            x = sdr.read_samples(block).astype(np.complex64)

            # Optional squelch based on average power
            if args.squelch_db is not None:
                pwr = 10.0 * np.log10(np.mean(np.abs(x) ** 2) + 1e-12)
                if pwr < args.squelch_db:
                    # output silence
                    out = np.zeros(int(block * (audio_rate / fs)), dtype=np.float32)
                    stream.write(out.reshape(-1, 1))
                    continue

            # NBFM discriminator
            y = np.angle(x[1:] * np.conj(x[:-1])).astype(np.float32)

            # Lowpass filter (with state)
            y, zi[:] = lfilter(b, a, y, zi=zi)

            # Resample to audio_rate (fs -> audio_rate)
            y_audio = resample_poly(y, up, down).astype(np.float32)

            # Remove DC + simple level control
            y_audio -= np.mean(y_audio)
            mx = np.max(np.abs(y_audio)) + 1e-6
            y_audio = (y_audio / mx) * 0.3  # 0.3 for comfortable volume

            stream.write(y_audio.reshape(-1, 1))

    except KeyboardInterrupt:
        print("\n[RX] Stopping...")

    finally:
        stream.stop()
        stream.close()
        sdr.close()


if __name__ == "__main__":
    main()
