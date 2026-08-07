"""Tests for the standalone Hands23 transport and latest-only client queue."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from perception.hands23_ipc import (
    Hands23IPCError,
    Hands23SidecarClient,
    receive_packet,
    send_packet,
    validate_hands23_assets,
)


class _AliveProcess:
    def poll(self):
        return None


class _AliveThread:
    def is_alive(self):
        return True


class _MemorySocket:
    def __init__(self):
        self.buffer = bytearray()

    def sendall(self, payload):
        self.buffer.extend(payload)

    def recv(self, size):
        if not self.buffer:
            return b""
        chunk = bytes(self.buffer[:size])
        del self.buffer[:size]
        return chunk


class Hands23IPCTest(unittest.TestCase):
    @staticmethod
    def _client(socket_path):
        return Hands23SidecarClient(
            python_interpreter="/tmp/python",
            sidecar_script="/tmp/sidecar.py",
            config_path="/tmp/config.yaml",
            repo_path="/tmp/repo",
            detector_config_path="/tmp/detector.yaml",
            weights_path="/tmp/weights.pth",
            socket_path=socket_path,
            max_input_hz=0.0,
        )

    def test_packet_round_trip_preserves_header_and_binary_payload(self):
        connection = _MemorySocket()
        payload = bytes(range(64))
        send_packet(connection, {"type": "frame", "frame_seq": 7}, payload)
        header, received = receive_packet(connection)
        self.assertEqual(header, {"type": "frame", "frame_seq": 7})
        self.assertEqual(received, payload)

    def test_submit_replaces_pending_frame_for_same_camera(self):
        client = self._client("/tmp/test-hands23.sock")
        client._process = _AliveProcess()
        client._thread = _AliveThread()
        client.reset(3)
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        self.assertTrue(
            client.submit(0, image, task_id=3, frame_seq=10, capture_time_s=1.0)
        )
        self.assertTrue(
            client.submit(0, image + 1, task_id=3, frame_seq=11, capture_time_s=2.0)
        )
        self.assertEqual(len(client._pending), 1)
        self.assertEqual(client._pending[0].frame_seq, 11)
        self.assertEqual(int(client._pending[0].image_bgr[0, 0, 0]), 1)

    def test_socket_preflight_never_deletes_a_regular_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hands23.sock"
            path.write_text("operator data", encoding="utf-8")
            client = self._client(path)
            with self.assertRaisesRegex(Hands23IPCError, "non-socket"):
                client._prepare_socket_path()
            self.assertEqual(path.read_text(encoding="utf-8"), "operator data")

    def test_asset_preflight_rejects_weights_only_repo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "hands23"
            repo.mkdir()
            sidecar = root / "sidecar.py"
            config = root / "runtime.yaml"
            detector = repo / "detector.yaml"
            weights = repo / "weights.pth"
            for path in (sidecar, config, detector, weights):
                path.write_text("fixture", encoding="utf-8")
            kwargs = dict(
                python_interpreter=sys.executable,
                sidecar_script=sidecar,
                config_path=config,
                repo_path=repo,
                detector_config_path=detector,
                weights_path=weights,
            )
            with self.assertRaisesRegex(FileNotFoundError, "weights-only"):
                validate_hands23_assets(**kwargs)

            marker = repo / "hodetector" / "modeling" / "roi_heads" / "__init__.py"
            marker.parent.mkdir(parents=True)
            marker.write_text("", encoding="utf-8")
            validated = validate_hands23_assets(**kwargs)
            self.assertEqual(validated["Hands23 source package"], marker)

            detector.write_text("_BASE_: ./Base-RCNN-FPN.yaml\n", encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "base config"):
                validate_hands23_assets(**kwargs)
            base_config = repo / "Base-RCNN-FPN.yaml"
            base_config.write_text("MODEL: {}\n", encoding="utf-8")
            validated = validate_hands23_assets(**kwargs)
            self.assertEqual(validated["Hands23 base config"], base_config)


if __name__ == "__main__":
    unittest.main()
