"""Fuse PatchMatch depth maps into a dense cloud, georeferenced into the scene frame.

This is the step where the photographs stop being texture and become geometry. Up to
here the surface was interpolated between 14,835 sparse feature points about a metre
apart; after this it is computed from the images themselves, one depth per pixel,
fused across every view that saw it.

The dense cloud lands in the sparse model's own coordinate frame, because the dense
workspace was undistorted from that model. So the similarity transform already fitted
from telemetry applies unchanged, and the laser points stay honest checks: they were
never used to build or place either cloud.

Run after `colmap patch_match_stereo`:
    .venv/bin/python scripts/fuse_dense.py [--name site]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pycolmap  # noqa: E402

from image_lab.kurumdy import scene3d  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "data/kurumdy/new"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="site")
    ap.add_argument("--min-num-pixels", type=int, default=5,
                    help="views that must agree before a point is kept")
    ap.add_argument("--max-reproj-error", type=float, default=2.0)
    args = ap.parse_args()

    work = ROOT / f"out/dense/{args.name}"
    dense = work / "dense"
    if not (dense / "stereo" / "depth_maps").exists():
        raise SystemExit(f"no depth maps in {dense}; run patch_match_stereo first")

    options = pycolmap.StereoFusionOptions()
    options.min_num_pixels = args.min_num_pixels
    options.max_reproj_error = args.max_reproj_error

    t0 = time.time()
    fused = pycolmap.stereo_fusion(
        output_path=str(work / "fused.ply"),
        workspace_path=str(dense),
        workspace_format="COLMAP",
        input_type="geometric",
        options=options,
        output_type="PLY",
    )
    print(f"fused {len(fused.points3D):,} points in {time.time() - t0:.0f}s", flush=True)

    xyz = np.array([p.xyz for p in fused.points3D.values()])
    rgb = np.array([p.color for p in fused.points3D.values()], dtype=np.uint8)

    # Reuse the sparse model's alignment: the dense cloud is in its coordinate frame.
    sparse = pycolmap.Reconstruction(str(work / "sparse"))
    origin = scene3d.default_origin()
    poses = scene3d.poses_from_telemetry(
        [im.name for im in sparse.images.values()], VIDEO_DIR)
    alignment = scene3d.align_to_telemetry(sparse, poses, origin, model_name=args.name)
    if alignment is None or not np.isfinite(alignment.scale):
        raise SystemExit("the sparse model could not be georeferenced, "
                         "so neither can the dense one")

    cloud = alignment.apply(xyz)
    np.savez_compressed(work / "dense_enu.npz",
                        xyz=cloud.astype(np.float32), rgb=rgb)

    checks = scene3d.check_objects(cloud, origin)
    extent = np.ptp(cloud, axis=0)
    lo, hi = np.percentile(cloud, [1, 99], axis=0)
    report = {
        "points": int(len(cloud)),
        "scale": round(alignment.scale, 4),
        "camera_residual_median_m": round(alignment.residual_median_m, 2),
        "extent_m": [round(float(v), 1) for v in extent],
        "core_extent_m": [round(float(v), 1) for v in (hi - lo)],
        "min_num_pixels": args.min_num_pixels,
        "held_out_laser_check": [
            {"object": c.name, "points_within_3m": c.n_points_within,
             "surface_error_m": None if not c.measured else round(c.surface_error_m, 2)}
            for c in checks
        ],
    }
    (work / "dense_report.json").write_text(json.dumps(report, indent=1))

    print(f"  scale {alignment.scale:.4f}, camera residual {alignment.residual_median_m:.2f} m")
    print(f"  extent {extent[0]:.0f} x {extent[1]:.0f} x {extent[2]:.0f} m")
    area = float(extent[0] * extent[1])
    print(f"  density about {len(cloud)/max(area,1):.0f} points per square metre "
          f"over the bounding footprint")
    for c in checks:
        if c.measured:
            print(f"  {c.name:9} held out: {c.surface_error_m:+.2f} m "
                  f"({c.n_points_within} points within 3 m)")
        else:
            print(f"  {c.name:9} held out: not covered")


if __name__ == "__main__":
    main()
