#!/usr/bin/env python3
"""Inspect the local persistent SAM3D model server."""

from __future__ import annotations

import argparse
import json
import sys

if __package__:
    from .sam3d_ipc import DEFAULT_SOCKET_PATH, Sam3DIPCError, status
else:
    from sam3d_ipc import DEFAULT_SOCKET_PATH, Sam3DIPCError, status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default=DEFAULT_SOCKET_PATH)
    parser.add_argument("operation", choices=("status",), default="status", nargs="?")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    response = status(args.socket)
    print(json.dumps(response, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Sam3DIPCError as exc:
        print(f"[sam3d_server_client] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
