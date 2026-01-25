import argparse
import signal
import subprocess
import sys
import time
import numpy as np
from scipy.signal import resample_poly

def decode_mp3_to_mono_float(mp3_path: str, target_rate: int, start_s: float, dur_s: float) -> np.ndarray:
    """
    Decode MP3 -> mono float32 [-1..1] at target_rate using ffmpeg.
    Extracts [start_s, start_s+dur_s] so you can loop the 'HELLO' part.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", str(start_s),
        "-i", mp3_path,
        "-t", str(dur_s),
        "-ac", "1",
        "-ar", str(target_rate),
        "-f", "s16le",
        "pipe:1"
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw, err = p.communicate()
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed:\n{err.decode(errors='ignore')}")

    audio_i16 = np.frombuffer(raw, dtype=np.int16)
    if audio_i16.size == 0:
        raise RuntimeError("Decoded audio is empty. Check --start/--dur or MP3 file.")
    return (audio_i16.astype(np.float32) / 32768.0).astype(np.float32)

def fm_modulate(audio: np.ndarray, fs: float, deviation_hz: float) -> np.ndarray:
    audio = np.clip(audio, -1.0, 1.0)
    phase = np.cumsum(2.0 * np.pi * deviation_hz * audio / fs).astype(np.float64)
    return np.exp(1j * phase).astype(np.complex64)

def write_cs8_iq(path: str, x: np.ndarray, amp: float) -> None:
    amp = float(np.clip(amp, 0.0, 0.99))
    y = x * amp
    i = np.clip(np.real(y), -0.99, 0.99)
    q = np.clip(np.imag(y), -0.99, 0.99)
    i8 = np.round(i * 127.0).astype(np.int8)
    q8 = np.round(q * 127.0).astype(np.int8)
    out = np.empty(i8.size * 2, dtype=np.int8)
    out[0::2] = i8
    out[1::2] = q8
    with open(path, "wb") as f:
        f.write(out.tobytes())

def main():
    ap = argparse.ArgumentParser(description="HackRF TX: loop MP3 snippet (HELLO) as NBFM.")
    ap.add_argument("--mp3", default="silence.mp3", help="MP3 file path")
    ap.add_argument("--start", type=float, default=0.0, help="Start time in seconds (default 0)")
    ap.add_argument("--dur", type=float, default=3.0, help="Duration seconds to loop (default 3)")
    ap.add_argument("--freq", type=float, default=433.92e6, help="RF center freq (Hz)")
    ap.add_argument("--offset", type=float, default=100e3, help="Offset from center (Hz) to avoid DC")
    ap.add_argument("--hackrf_sr", type=float, default=8e6, help="HackRF SR (8e6/10e6/12.5e6/16e6/20e6)")
    ap.add_argument("--base_sr", type=float, default=500e3, help="Internal baseband SR (Hz)")
    ap.add_argument("--deviation", type=float, default=5e3, help="NBFM deviation Hz (default 5k)")
    ap.add_argument("--amp", type=float, default=0.25, help="IQ amplitude 0..0.99 (default 0.25)")
    ap.add_argument("--txvga", type=int, default=10, help="TXVGA 0..47 (default 10)")
    ap.add_argument("--amp_enable", type=int, default=0, help="HackRF amp enable 0/1 (default 0)")
    ap.add_argument("--iq", default="tx_hello_cs8.iq", help="IQ output file")
    args = ap.parse_args()

    if args.hackrf_sr not in (8e6, 10e6, 12.5e6, 16e6, 20e6):
        print("ERROR: --hackrf_sr must be 8e6/10e6/12.5e6/16e6/20e6", file=sys.stderr)
        sys.exit(1)

    # 1) Decode MP3 snippet -> 48k mono float
    audio48 = decode_mp3_to_mono_float(args.mp3, target_rate=48000, start_s=args.start, dur_s=args.dur)

    # 2) Resample audio 48k -> base_sr
    base_sr = int(args.base_sr)
    g = np.gcd(48000, base_sr)
    audio_base = resample_poly(audio48, base_sr // g, 48000 // g).astype(np.float32)

    # 3) NBFM modulate at base_sr
    x = fm_modulate(audio_base, base_sr, args.deviation)

    # 4) Mix up by offset
    if args.offset != 0:
        n = x.size
        t = np.arange(n, dtype=np.float64) / base_sr
        x = (x * np.exp(1j * 2*np.pi*args.offset*t)).astype(np.complex64)

    # 5) Resample complex -> hackrf_sr
    hackrf_sr = int(args.hackrf_sr)
    g2 = np.gcd(base_sr, hackrf_sr)
    up = hackrf_sr // g2
    down = base_sr // g2
    xi = resample_poly(np.real(x), up, down).astype(np.float32)
    xq = resample_poly(np.imag(x), up, down).astype(np.float32)
    xh = (xi + 1j*xq).astype(np.complex64)

    write_cs8_iq(args.iq, xh, args.amp)

    rf_line = args.freq + args.offset
    print(f"[TX] Looping snippet: start={args.start}s dur={args.dur}s")
    print(f"[TX] Center={args.freq/1e6:.6f} MHz offset={args.offset/1e3:.1f} kHz => RX tune {rf_line/1e6:.6f} MHz")
    print("[TX] Ctrl+C to stop.")

    cmd = [
        "hackrf_transfer",
        "-t", args.iq,
        "-f", str(int(args.freq)),
        "-s", str(int(hackrf_sr)),
        "-x", str(int(args.txvga)),
        "-a", str(int(args.amp_enable)),
    ]

    stop = False
    def _stop(_sig, _frm):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    while not stop:
        # show errors if any (DON'T hide stderr)
        p = subprocess.Popen(cmd)
        while p.poll() is None and not stop:
            time.sleep(0.1)
        if p.poll() is None:
            p.terminate()
        time.sleep(0.05)

    print("\n[TX] Stopped.")

if __name__ == "__main__":
    main()
