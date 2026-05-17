"""Create a standalone HTML view of a template point cloud and local axes.

The template .npy files are stored in their local coordinate frame. This script
renders the points together with the local x/y/z axes so the axis convention can
be checked without Open3D or Matplotlib.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_TEMPLATE_PATH = Path("shape_fitting/poteau.npy")
DEFAULT_OUTPUT_PATH = Path("output/template_axes/poteau_axes.html")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a .npy template point cloud and local x/y/z axes.")
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE_PATH), help="Path to the template .npy file.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output HTML path.")
    parser.add_argument("--unit-scale", type=float, default=0.001, help="Scale applied before display, e.g. 0.001 for mm->m.")
    parser.add_argument("--max-points", type=int, default=5000, help="Maximum number of points to embed in the HTML.")
    parser.add_argument(
        "--axis-origin",
        choices=["centroid", "min-corner", "zero"],
        default="centroid",
        help="Where to draw the local axes.",
    )
    parser.add_argument("--axis-length", type=float, default=None, help="Axis length after scaling. Default uses 35%% of max extent.")
    return parser.parse_args()


def load_points(path: str | Path, *, unit_scale: float, max_points: int) -> np.ndarray:
    points = np.asarray(np.load(path), dtype=np.float64).reshape((-1, 3))
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        raise ValueError(f"No finite points in template: {path}")
    points = points * float(unit_scale)
    if max_points > 0 and len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, int(max_points), dtype=np.int32)
        points = points[indices]
    return points


def axis_origin(points: np.ndarray, mode: str) -> np.ndarray:
    if mode == "zero":
        return np.zeros((3,), dtype=np.float64)
    if mode == "min-corner":
        return np.min(points, axis=0)
    return np.mean(points, axis=0)


def make_html(points: np.ndarray, *, title: str, axis_origin_mode: str, axis_length: float | None) -> str:
    point_min = np.min(points, axis=0)
    point_max = np.max(points, axis=0)
    extent = point_max - point_min
    origin = axis_origin(points, axis_origin_mode)
    length = float(axis_length) if axis_length is not None else float(np.max(extent) * 0.35)
    if not np.isfinite(length) or length <= 0.0:
        length = 0.05

    axes = {
        "x": {"end": origin + np.asarray([length, 0.0, 0.0]), "color": "#e53935"},
        "y": {"end": origin + np.asarray([0.0, length, 0.0]), "color": "#43a047"},
        "z": {"end": origin + np.asarray([0.0, 0.0, length]), "color": "#1e88e5"},
    }
    payload = {
        "title": title,
        "points": points.round(6).tolist(),
        "origin": origin.round(6).tolist(),
        "axes": {
            name: {"end": axis["end"].round(6).tolist(), "color": axis["color"]}
            for name, axis in axes.items()
        },
        "bounds": {"min": point_min.round(6).tolist(), "max": point_max.round(6).tolist(), "extent": extent.round(6).tolist()},
    }
    payload_json = json.dumps(payload)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    html, body {{ margin: 0; width: 100%; height: 100%; font-family: Arial, sans-serif; background: #f7f7f4; }}
    #info {{ position: fixed; top: 12px; left: 12px; padding: 10px 12px; background: rgba(255,255,255,0.88); border: 1px solid #ddd; font-size: 13px; line-height: 1.35; }}
    #info b {{ display: inline-block; min-width: 18px; }}
    #canvas {{ width: 100vw; height: 100vh; display: block; }}
  </style>
</head>
<body>
  <canvas id="canvas"></canvas>
  <div id="info"></div>
  <script>
    const DATA = {payload_json};
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const info = document.getElementById("info");
    let yaw = -0.7, pitch = 0.45, zoom = 1.0;
    let dragging = false, lastX = 0, lastY = 0;

    info.innerHTML = `
      <strong>${{DATA.title}}</strong><br>
      points: ${{DATA.points.length}}<br>
      extent: ${{DATA.bounds.extent.map(v => v.toFixed(4)).join(", ")}}<br>
      <b style="color:#e53935">X</b> local x axis<br>
      <b style="color:#43a047">Y</b> local y axis<br>
      <b style="color:#1e88e5">Z</b> local z axis<br>
      drag: rotate, wheel: zoom
    `;

    function resize() {{
      canvas.width = window.innerWidth * devicePixelRatio;
      canvas.height = window.innerHeight * devicePixelRatio;
      draw();
    }}

    function matProject(point) {{
      const center = [
        0.5 * (DATA.bounds.min[0] + DATA.bounds.max[0]),
        0.5 * (DATA.bounds.min[1] + DATA.bounds.max[1]),
        0.5 * (DATA.bounds.min[2] + DATA.bounds.max[2])
      ];
      let x = point[0] - center[0];
      let y = point[1] - center[1];
      let z = point[2] - center[2];
      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      const x1 = cy * x + sy * z;
      const z1 = -sy * x + cy * z;
      const y1 = cp * y - sp * z1;
      const z2 = sp * y + cp * z1;
      const maxExtent = Math.max(...DATA.bounds.extent, 1e-6);
      const scale = Math.min(canvas.width, canvas.height) * 0.72 * zoom / maxExtent;
      return [
        canvas.width * 0.5 + x1 * scale,
        canvas.height * 0.52 - y1 * scale,
        z2
      ];
    }}

    function drawLine(a, b, color, width) {{
      const pa = matProject(a), pb = matProject(b);
      ctx.strokeStyle = color;
      ctx.lineWidth = width * devicePixelRatio;
      ctx.beginPath();
      ctx.moveTo(pa[0], pa[1]);
      ctx.lineTo(pb[0], pb[1]);
      ctx.stroke();
    }}

    function drawLabel(text, point, color) {{
      const p = matProject(point);
      ctx.fillStyle = color;
      ctx.font = `${{15 * devicePixelRatio}}px Arial`;
      ctx.fillText(text, p[0] + 7 * devicePixelRatio, p[1] - 7 * devicePixelRatio);
    }}

    function draw() {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const projected = DATA.points.map(p => [p, matProject(p)]).sort((a, b) => a[1][2] - b[1][2]);
      for (const [, p] of projected) {{
        ctx.fillStyle = "rgba(30,30,30,0.55)";
        ctx.beginPath();
        ctx.arc(p[0], p[1], 1.6 * devicePixelRatio, 0, Math.PI * 2);
        ctx.fill();
      }}
      for (const name of ["x", "y", "z"]) {{
        const axis = DATA.axes[name];
        drawLine(DATA.origin, axis.end, axis.color, 4);
        drawLabel(name.toUpperCase(), axis.end, axis.color);
      }}
    }}

    canvas.addEventListener("mousedown", e => {{ dragging = true; lastX = e.clientX; lastY = e.clientY; }});
    window.addEventListener("mouseup", () => dragging = false);
    window.addEventListener("mousemove", e => {{
      if (!dragging) return;
      yaw += (e.clientX - lastX) * 0.01;
      pitch += (e.clientY - lastY) * 0.01;
      pitch = Math.max(-1.45, Math.min(1.45, pitch));
      lastX = e.clientX; lastY = e.clientY;
      draw();
    }});
    canvas.addEventListener("wheel", e => {{
      e.preventDefault();
      zoom *= Math.exp(-e.deltaY * 0.001);
      zoom = Math.max(0.25, Math.min(6, zoom));
      draw();
    }}, {{ passive: false }});
    window.addEventListener("resize", resize);
    resize();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    template_path = Path(args.template)
    output_path = Path(args.output)
    points = load_points(template_path, unit_scale=args.unit_scale, max_points=args.max_points)
    html = make_html(
        points,
        title=f"{template_path.name} local axes",
        axis_origin_mode=args.axis_origin,
        axis_length=args.axis_length,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    extent = np.max(points, axis=0) - np.min(points, axis=0)
    print(f"wrote {output_path}")
    print(f"points={len(points)}, extent_scaled={extent}")
    print("local axes: x=red, y=green, z=blue")


if __name__ == "__main__":
    main()
