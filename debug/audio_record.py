"""Record contact audio from a microphone into a WAV file.

Examples:
    python debug/audio_record.py --list-devices
    python debug/audio_record.py --seconds 10 --output logs/contact.wav
    python debug/audio_record.py --device 6 --seconds 5 --samplerate 48000
"""

from __future__ import annotations

import argparse
from pathlib import Path
import queue
import signal
import sys
import time
import wave

import numpy as np
import sounddevice as sd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record contact audio from an input device.")
    parser.add_argument("--output", type=Path, default=Path("logs/contact_audio.wav"), help="Path to the output WAV file.")
    parser.add_argument("--seconds", type=float, default=None, help="Recording duration in seconds. Omit to record until Ctrl+C.")
    parser.add_argument(
        "--device",
        default=None,
        help="Input device index or case-insensitive name substring. Defaults to the current sounddevice input device.",
    )
    parser.add_argument("--samplerate", type=int, default=48000, help="Requested sample rate in Hz.")
    parser.add_argument("--channels", type=int, default=1, help="Number of input channels to record.")
    parser.add_argument("--blocksize", type=int, default=0, help="Frames per callback block. Use 0 for the backend default.")
    parser.add_argument("--list-devices", action="store_true", help="Print available audio devices and exit.")
    parser.add_argument("--print-level-every", type=float, default=0.5, help="Seconds between simple level meter updates.")
    return parser.parse_args()


def iter_input_devices() -> list[tuple[int, dict]]:
    devices = sd.query_devices()
    return [(index, device) for index, device in enumerate(devices) if int(device["max_input_channels"]) > 0]


def print_input_devices() -> None:
    default_input, _ = sd.default.device
    print("Available input devices:")
    for index, device in iter_input_devices():
        default_marker = " [default]" if index == default_input else ""
        print(
            f"  {index}: {device['name']} | inputs={device['max_input_channels']} "
            f"| default_samplerate={device['default_samplerate']}{default_marker}"
        )


def resolve_input_device(device_arg: str | None) -> int:
    if device_arg is None:
        default_input, _ = sd.default.device
        if default_input is not None and int(default_input) >= 0:
            return int(default_input)
        input_devices = iter_input_devices()
        if input_devices:
            return int(input_devices[0][0])
        raise RuntimeError("No input audio devices are available.")

    try:
        return int(device_arg)
    except (TypeError, ValueError):
        pass

    lowered = str(device_arg).strip().lower()
    matches = []
    for index, device in iter_input_devices():
        if lowered in str(device["name"]).lower():
            matches.append((index, device))

    if not matches:
        raise RuntimeError(f"No input device matched {device_arg!r}. Use --list-devices to inspect available inputs.")
    if len(matches) > 1:
        names = ", ".join(f"{index}:{device['name']}" for index, device in matches)
        raise RuntimeError(f"Multiple input devices matched {device_arg!r}: {names}")
    return int(matches[0][0])


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_wav(path: Path, audio_blocks: list[np.ndarray], samplerate: int, channels: int) -> int:
    if audio_blocks:
        audio = np.concatenate(audio_blocks, axis=0)
    else:
        audio = np.zeros((0, channels), dtype=np.int16)

    ensure_parent_dir(path)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(samplerate)
        wav_file.writeframes(audio.astype(np.int16, copy=False).tobytes())
    return int(audio.shape[0])


def compute_level_percent(block: np.ndarray) -> float:
    if block.size == 0:
        return 0.0
    peak = np.max(np.abs(block.astype(np.float32)))
    return float(100.0 * peak / 32768.0)


def record_audio(
    output_path: Path,
    *,
    seconds: float | None,
    device: int,
    samplerate: int,
    channels: int,
    blocksize: int,
    print_level_every: float,
) -> None:
    audio_queue: queue.Queue[np.ndarray | None] = queue.Queue()
    recorded_blocks: list[np.ndarray] = []
    should_stop = False
    next_meter_time = 0.0

    def request_stop(signum: int, frame: object) -> None:
        del signum, frame
        nonlocal should_stop
        should_stop = True

    def callback(indata: np.ndarray, frames: int, callback_time: object, status: sd.CallbackFlags) -> None:
        del frames, callback_time
        if status:
            print(f"[audio] stream status: {status}", file=sys.stderr)
        audio_queue.put(indata.copy())

    previous_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, request_stop)
    start_time = time.time()
    print(f"[audio] recording from device {device} to {output_path}")
    if seconds is None:
        print("[audio] press Ctrl+C to stop")
    else:
        print(f"[audio] duration={seconds:.2f}s")

    try:
        with sd.InputStream(
            device=device,
            samplerate=samplerate,
            channels=channels,
            dtype="int16",
            blocksize=blocksize,
            callback=callback,
        ):
            while True:
                timeout_s = min(max(float(print_level_every), 0.1), 1.0)
                try:
                    block = audio_queue.get(timeout=timeout_s)
                except queue.Empty:
                    block = None

                if block is not None:
                    recorded_blocks.append(block)
                    now = time.time()
                    if now >= next_meter_time:
                        print(f"[audio] level={compute_level_percent(block):5.1f}%")
                        next_meter_time = now + max(float(print_level_every), 0.1)

                elapsed_s = time.time() - start_time
                if seconds is not None and elapsed_s >= float(seconds):
                    break
                if should_stop:
                    break
    finally:
        signal.signal(signal.SIGINT, previous_sigint)

    frame_count = write_wav(output_path, recorded_blocks, samplerate=samplerate, channels=channels)
    duration_s = frame_count / float(samplerate) if samplerate > 0 else 0.0
    print(f"[audio] saved {frame_count} frames ({duration_s:.2f}s) to {output_path}")


def main() -> int:
    args = parse_args()

    if args.list_devices:
        print_input_devices()
        return 0

    if args.seconds is not None and float(args.seconds) <= 0.0:
        raise SystemExit("--seconds must be positive when provided.")
    if int(args.channels) <= 0:
        raise SystemExit("--channels must be positive.")
    if int(args.samplerate) <= 0:
        raise SystemExit("--samplerate must be positive.")
    if int(args.blocksize) < 0:
        raise SystemExit("--blocksize cannot be negative.")

    try:
        device_index = resolve_input_device(args.device)
        device_info = sd.query_devices(device_index, "input")
        print(
            f"[audio] using device {device_index}: {device_info['name']} "
            f"(max_input_channels={device_info['max_input_channels']}, "
            f"default_samplerate={device_info['default_samplerate']})"
        )
        record_audio(
            args.output,
            seconds=args.seconds,
            device=device_index,
            samplerate=int(args.samplerate),
            channels=int(args.channels),
            blocksize=int(args.blocksize),
            print_level_every=float(args.print_level_every),
        )
    except KeyboardInterrupt:
        print("\n[audio] interrupted before recording finished", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"[audio] error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
