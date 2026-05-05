#!/usr/bin/env python3
"""
Record audio from an ALSA capture device and optionally save a single channel.

This script is meant for quick microphone bring-up when working with the
stereo contact-microphone setup. It records a WAV file with `arecord` and can:

- keep the full multichannel capture as-is
- extract one channel from a multichannel capture and save it as mono
- print simple RMS / peak statistics for the saved signal
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile
import wave
from typing import Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires numpy. Install it with `python3 -m pip install numpy`."
    ) from exc


PCM_SCALE_BY_WIDTH = {
    2: 32768.0,
    4: 2147483648.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record audio from ALSA and optionally export one channel as mono."
    )
    parser.add_argument(
        "--device",
        default="hw:1,0",
        help="ALSA capture device passed to arecord. Default: hw:1,0",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Recording duration in seconds. Default: 5.0",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        help="Recording sample rate. Default: 48000",
    )
    parser.add_argument(
        "--sample-format",
        default="S32_LE",
        choices=["S16_LE", "S24_3LE", "S32_LE"],
        help="ALSA sample format passed to arecord. Default: S32_LE",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Total number of channels to record from ALSA. Default: 1",
    )
    parser.add_argument(
        "--channel-index",
        type=int,
        default=0,
        help=(
            "Zero-based channel index to export from the recorded stream. "
            "Default: 0"
        ),
    )
    parser.add_argument(
        "--output-wav",
        default="single_channel_recording.wav",
        help="Path to the final saved WAV file. Default: single_channel_recording.wav",
    )
    parser.add_argument(
        "--keep-multichannel-wav",
        default=None,
        help="Optional path to keep the raw multichannel capture before channel extraction.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print ALSA capture devices using arecord -l and exit.",
    )
    return parser.parse_args()


def list_capture_devices() -> int:
    result = subprocess.run(["arecord", "-l"], check=False)
    return result.returncode


def record_wav(
    device: str,
    duration: float,
    sample_rate: int,
    channels: int,
    sample_format: str,
    output_path: str,
) -> None:
    cmd = [
        "arecord",
        "-D",
        device,
        "-f",
        sample_format,
        "-r",
        str(sample_rate),
        "-c",
        str(channels),
        "-d",
        str(max(1, int(math.ceil(duration)))),
        output_path,
    ]
    subprocess.run(cmd, check=True)


def load_wav(path: str) -> Tuple[np.ndarray, int, int]:
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        frames = wf.readframes(n_frames)

    if sample_width not in PCM_SCALE_BY_WIDTH:
        raise ValueError(
            f"Unsupported PCM sample width {sample_width} bytes. Expected 2 or 4."
        )

    dtype = np.int16 if sample_width == 2 else np.int32
    audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    audio /= PCM_SCALE_BY_WIDTH[sample_width]

    if audio.size % n_channels != 0:
        raise ValueError("WAV data size is not divisible by the channel count.")

    return audio.reshape(-1, n_channels), sample_rate, sample_width


def save_wav(path: str, audio: np.ndarray, sample_rate: int, sample_width: int) -> None:
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]

    clipped = np.clip(audio, -1.0, 1.0 - np.finfo(np.float32).eps)
    if sample_width == 2:
        pcm = (clipped * PCM_SCALE_BY_WIDTH[sample_width]).astype(np.int16)
    elif sample_width == 4:
        pcm = (clipped * PCM_SCALE_BY_WIDTH[sample_width]).astype(np.int32)
    else:
        raise ValueError(
            f"Unsupported PCM sample width {sample_width} bytes. Expected 2 or 4."
        )

    with wave.open(path, "wb") as wf:
        wf.setnchannels(audio.shape[1])
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def compute_stats(signal: np.ndarray) -> Tuple[float, float]:
    rms = float(np.sqrt(np.mean(np.square(signal))))
    peak = float(np.max(np.abs(signal)))
    return rms, peak


def main() -> int:
    args = parse_args()

    if args.list_devices:
        return list_capture_devices()

    if args.channels < 1:
        print("--channels must be at least 1.", file=sys.stderr)
        return 1
    if not (0 <= args.channel_index < args.channels):
        print("--channel-index must be in [0, channels).", file=sys.stderr)
        return 1

    output_wav = os.path.abspath(args.output_wav)
    os.makedirs(os.path.dirname(output_wav) or ".", exist_ok=True)

    raw_capture_path = args.keep_multichannel_wav
    temp_dir = None
    if raw_capture_path is None and args.channels > 1:
        temp_dir = tempfile.TemporaryDirectory()
        raw_capture_path = os.path.join(temp_dir.name, "multichannel_capture.wav")
    elif raw_capture_path is not None:
        raw_capture_path = os.path.abspath(raw_capture_path)
        os.makedirs(os.path.dirname(raw_capture_path) or ".", exist_ok=True)
    else:
        raw_capture_path = output_wav

    try:
        record_wav(
            device=args.device,
            duration=args.duration,
            sample_rate=args.sample_rate,
            channels=args.channels,
            sample_format=args.sample_format,
            output_path=raw_capture_path,
        )

        audio, sample_rate, sample_width = load_wav(raw_capture_path)
        if audio.shape[1] != args.channels:
            raise ValueError(
                f"Expected {args.channels} channels, but WAV contains {audio.shape[1]}."
            )

        selected = audio[:, args.channel_index]
        if args.channels == 1 and os.path.abspath(raw_capture_path) == output_wav:
            saved_audio = audio
        else:
            saved_audio = selected
            save_wav(output_wav, saved_audio, sample_rate, sample_width)

        rms, peak = compute_stats(selected)
    except subprocess.CalledProcessError as exc:
        print(f"Recording failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Failed to process recording: {exc}", file=sys.stderr)
        return 1
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    print(f"Saved WAV: {output_wav}")
    print(f"Sample rate: {sample_rate}")
    print(f"Recorded channels: {args.channels}")
    print(f"Exported channel index: {args.channel_index}")
    print(f"Duration: {len(selected) / sample_rate:.3f} s")
    print(f"RMS: {rms:.6f}")
    print(f"Peak: {peak:.6f}")

    if args.keep_multichannel_wav is not None:
        print(f"Raw multichannel capture: {os.path.abspath(args.keep_multichannel_wav)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())