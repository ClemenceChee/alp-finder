"""Pack the built scene into self-contained HTML viewers.

Two packs from one scene: a lite one small enough to email to the search team,
and a full one for local work. Both inline everything - three.js, the scene, and
every photograph as a data URI - so they open by double-click with no network
and nothing to install. That constraint comes from the recipients, who are the
ones likely to be offline.

Run from the repo root, after scripts/build_scene3d.py:

    .venv/bin/python scripts/pack_scene3d.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out/scene3d"
VENDOR = OUT / "vendor"
TEMPLATE = Path(__file__).resolve().parent / "viewer_template.html"

PACKS = {
    "lite": {"dir": "lite", "out": "kurumdy-3d-lite.html", "max_mb": 6.0},
    "full": {"dir": "full", "out": "kurumdy-3d.html", "max_mb": 20.0},
}


class TooBig(RuntimeError):
    """A pack exceeded its size budget."""


def data_uri(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def pack(name: str, spec: dict, template: str, three: str, orbit: str, scene: dict) -> Path:
    image_dir = OUT / spec["dir"]

    # Each pack carries only its own surface geometry. Shipping both would put a
    # 0.5 m mesh inside the file meant to be emailed.
    scene = dict(scene)
    surfaces = scene.pop("surface", {})
    if name in surfaces:
        scene["surface"] = surfaces[name]

    wanted = [c["image"] for c in scene["cameras"] if c.get("image")]
    wanted += [q["image"] for q in scene.get("quads", [])]
    if scene.get("surface", {}).get("ortho"):
        wanted.append(scene["surface"]["ortho"])

    images = {}
    for asset in dict.fromkeys(wanted):
        path = image_dir / asset
        if path.exists():
            images[asset] = data_uri(path)

    html = template
    html = html.replace("%%SCENE%%", json.dumps(scene, separators=(",", ":")))
    html = html.replace("%%THREE%%", three)
    html = html.replace("%%ORBIT%%", orbit)
    html = html.replace("%%IMAGES%%", json.dumps(images, separators=(",", ":")))

    target = OUT / spec["out"]
    target.write_text(html, encoding="utf-8")
    size_mb = target.stat().st_size / 1e6
    if size_mb > spec["max_mb"]:
        raise TooBig(
            f"{target.name} is {size_mb:.1f} MB, over its {spec['max_mb']} MB budget. "
            "Lower the ortho width or drop a shipped frame rather than raising this."
        )
    return target


def main() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")
    three = (VENDOR / "three.min.js").read_text(encoding="utf-8")
    orbit = (VENDOR / "OrbitControls.js").read_text(encoding="utf-8")
    scene = json.loads((OUT / "scene.json").read_text())

    for name, spec in PACKS.items():
        target = pack(name, spec, template, three, orbit, scene)
        size = target.stat().st_size
        shipped = sum(1 for c in scene["cameras"]
                      if c.get("image") and (OUT / spec["dir"] / c["image"]).exists())
        surface = scene["surface"][name]
        print(f"{target.name:<24} {size/1e6:5.1f} MB   {shipped} photographs, "
              f"{surface['triangles']} tris at {surface['posting_m']} m, "
              f"ortho {surface['rect']['metres_per_texel']*100:.1f} cm/texel")


if __name__ == "__main__":
    main()
