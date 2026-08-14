"""Build the 3D scene description for the Kurumdy object site.

Reads the DEM, the georeferenced reconstruction and the flight telemetry, and
writes ``out/scene3d/scene.json`` plus the image assets the viewer needs.

Everything in the output is in one local ENU metric frame with ellipsoidal
heights (see ``image_lab.kurumdy.scene3d``). Terrain is converted on the way in.
Nothing downstream has to think about datums, which is the point.

Run from the repo root:

    .venv/bin/python scripts/build_scene3d.py
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pycolmap  # noqa: E402

from image_lab.kurumdy import drape, georef, scene3d, stills, telemetry, terrain_prior  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out/scene3d"
FRAMES = OUT / "frames"
VIDEO_DIR = ROOT / "data/kurumdy/new"
MODEL = OUT / "sfm/model0"

HERO = "DJI_20260813163855_0001_Z"
HERO_STEM = "0813163855_0001_Z"

#: Context window: wide enough to hold the whole traced fall line.
CONTEXT_BOUNDS = (39.4700, 39.5100, 73.5700, 73.6050)
#: Site window: the debris scatter and the ground immediately below it.
SITE_BOUNDS = (39.4795, 39.4865, 73.5835, 73.5885)

#: Frames shipped as full photographs you can jump into. The five wide frames that
#: used to be here (3525, 3930, 4200, 4470, 2355) are now baked into the orthophoto and
#: are near-duplicates of 3705, so shipping them again only costs bytes.
SHIP_FRAMES = [1410, 2265, 3415, 3705, 4785, 5640, 13215]

#: The lite pack drops 1410 and 5640: both look 31 to 35 m off the site centre, both
#: are already baked into the orthophoto, and the pack has a hard size budget that
#: the dense mesh now spends most of.
LITE_SHIP_FRAMES = [2265, 3415, 3705, 4785, 13215]

#: Frames painted onto the surface, coarsest ground sample distance first so the widest
#: lays a complete base and every sharper frame overwrites inside its own footprint.
#: 3705 alone covers the whole site at 8.0 cm/px.
DRAPE_FRAMES = [3705, 3525, 3930, 4200, 4470, 2265, 2355, 4785, 5640, 1410]

#: Stills that look at a known object, each carrying its own laser range. These become
#: photo quads hung at the measured target, which is how the rucksack enters the scene
#: at all: every clip that sees it is a hover, so no reconstruction is possible there.
STILL_QUADS = {
    "rucksack": "DJI_20260813185530_0004_Z.JPG",
    "lid": "DJI_20260813183044_0005_Z.JPG",
    "pole": "DJI_20260813183142_0013_Z.JPG",
}

#: Texture width per pack. Lite is sized so the encoded orthophoto stays under a
#: megabyte; full matches the source imagery's own 3.5 cm/px.
ORTHO_WIDTH = {"lite": 2048, "full": 4096}
ORTHO_QUALITY = {"lite": 74, "full": 82}
SURFACE_POSTING_M = {"lite": 1.0, "full": 0.5}
#: A dense cloud supports far finer ground. These are the postings used when one
#: exists; the support gate in drape still drops any cell the cloud cannot carry.
DENSE_POSTING_M = {"lite": 0.5, "full": 0.25}

#: Frames posed from telemetry rather than by the reconstruction, with the focal
#: measured by chaining SIFT scale ratios back to a frame the reconstruction
#: solved. Both are far above ``selfcal``'s optical-only ceiling of 14315 px,
#: because the operator was using digital zoom on top of full optical zoom.
TELEMETRY_FOCALS = {3415: 65984.0, 13215: 17400.0}

#: Frames in which the orange object was seen, with a seed pixel to pick the
#: right blob. Seeds come from reading the frames; the position is then solved,
#: not asserted, so the coordinate in the deliverable is reproducible.
ORANGE_SIGHTINGS = {
    2265: (380, 806),
    2355: (662, 591),
    4200: (959, 558),
    4470: (959, 500),
    4785: (901, 459),
}

MAX_CLOUD_POINTS = 60_000
MIN_TRACK_LENGTH = 3

#: A dense cloud from PatchMatch stereo, if one has been fused. When present the
#: surface is built from it rather than from the sparse feature points, which is the
#: difference between terrain computed from the photographs and terrain interpolated
#: between points a metre apart.
DENSE_CLOUD = ROOT / "out/dense/site/dense_enu.npz"


def load_cloud(rec, alignment) -> tuple[np.ndarray, str]:
    """The point cloud the surface is built from, dense if one exists."""
    if DENSE_CLOUD.exists():
        data = np.load(DENSE_CLOUD)
        return data["xyz"].astype(float), "dense"
    tracks_len = np.array([len(p.track.elements) for p in rec.points3D.values()])
    sparse = alignment.apply(np.array([p.xyz for p in rec.points3D.values()]))
    return sparse[tracks_len >= MIN_TRACK_LENGTH], "sparse"


def b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def terrain_payload(layer: scene3d.TerrainLayer, origin: scene3d.SceneOrigin) -> dict:
    """A heightfield the viewer can build a mesh from, as int16 metres."""
    z = np.round(layer.z_ellipsoidal).astype(np.int16)
    rows, cols = z.shape
    # Corner positions in scene metres, so the viewer never touches degrees.
    sw = origin.enu(layer.lat0, layer.lon0, 0.0)
    ne = origin.enu(layer.lat1, layer.lon1, 0.0)
    return {
        "name": layer.name,
        "rows": rows,
        "cols": cols,
        "east_m": [float(sw[0]), float(ne[0])],
        "north_m": [float(sw[1]), float(ne[1])],
        "row0_is_north": True,
        "datum": "WGS84 ellipsoidal, metres",
        "geoid_separation_m": layer.separation_m,
        "posting_m": abs(layer.dlat) * np.pi / 180.0 * scene3d.EARTH_R,
        "z_offset_m": float(origin.alt_ellipsoidal),
        "z_int16_base64": b64(z),
    }


def load_model() -> tuple[pycolmap.Reconstruction, scene3d.Alignment]:
    rec = pycolmap.Reconstruction(str(MODEL))
    names = [im.name for im in rec.images.values()]
    poses = scene3d.poses_from_telemetry(names, VIDEO_DIR)
    alignment = scene3d.align_to_telemetry(rec, poses, scene3d.default_origin(),
                                           model_name="model0")
    if alignment is None or not np.isfinite(alignment.scale):
        raise SystemExit("model0 could not be georeferenced")
    return rec, alignment


def cloud_payload(rec, alignment: scene3d.Alignment) -> dict:
    """The reconstructed surface, thinned and colour-carrying."""
    keep = [p for p in rec.points3D.values() if len(p.track.elements) >= MIN_TRACK_LENGTH]
    xyz = alignment.apply(np.array([p.xyz for p in keep]))
    rgb = np.array([p.color for p in keep], dtype=np.uint8)
    if len(xyz) > MAX_CLOUD_POINTS:
        idx = np.linspace(0, len(xyz) - 1, MAX_CLOUD_POINTS).astype(int)
        xyz, rgb = xyz[idx], rgb[idx]
    return {
        "n": int(len(xyz)),
        "min_track_length": MIN_TRACK_LENGTH,
        "xyz_float32_base64": b64(xyz.astype(np.float32)),
        "rgb_uint8_base64": b64(rgb),
    }


def camera_payload(node: scene3d.CameraNode, extra: dict | None = None) -> dict:
    payload = {
        "id": node.ident,
        "source": node.source,
        "clip": node.clip,
        "frame": node.frame,
        "t_s": round(node.frame / 29.97, 2),
        "position_m": [round(float(v), 3) for v in node.position],
        "right": [round(float(v), 6) for v in node.right],
        "up": [round(float(v), 6) for v in node.up],
        "forward": [round(float(v), 6) for v in node.forward],
        "f_px": round(node.f_px, 1),
        "width": node.width,
        "height": node.height,
        "hfov_deg": round(node.hfov_deg, 3),
        "vfov_deg": round(node.vfov_deg, 3),
    }
    if node.image:
        payload["image"] = node.image
    if node.notes:
        payload["notes"] = node.notes
    if extra:
        payload.update(extra)
    return payload


def build_cameras(rec, alignment, origin, track) -> list[dict]:
    """One entry per reconstruction camera, plus the two telemetry-posed frames."""
    by_frame = {f.frame: f for f in track.positioned()}
    out: list[dict] = []
    solved = set()
    for image in sorted(rec.images.values(), key=lambda i: i.name):
        camera = rec.cameras[image.camera_id]
        node = scene3d.camera_from_reconstruction(image, camera, alignment, origin,
                                                  clip=HERO)
        solved.add(node.frame)
        if node.frame in SHIP_FRAMES:
            node.image = f"163855_{node.frame:06d}.jpg"
        out.append(camera_payload(node))

    for frame, f_px in TELEMETRY_FOCALS.items():
        if frame in solved:
            continue
        fix = by_frame.get(frame)
        if fix is None:
            continue
        node = scene3d.camera_from_pose(fix, f_px, origin, source="telemetry",
                                        clip=HERO, ident=f"{HERO_STEM}_{frame:06d}")
        node.image = f"163855_{frame:06d}.jpg"
        node.notes.append(
            "focal measured by chaining SIFT scale ratios to a reconstructed frame; "
            "it exceeds the optical maximum because digital zoom was in use"
        )
        out.append(camera_payload(node))
    return out


def build_surface(rec, alignment, origin) -> tuple:
    """Fit the site plane, build the mesh, and bake the orthophoto for each pack."""
    cloud, kind = load_cloud(rec, alignment)
    posting = SURFACE_POSTING_M if kind == "sparse" else DENSE_POSTING_M
    print(f"  surface from the {kind} cloud: {len(cloud):,} points")
    frame = drape.fit_surface_frame(cloud)

    by_name = {im.name: im for im in rec.images.values()}
    cameras, images = [], {}
    for number in DRAPE_FRAMES:
        image = by_name.get(f"{HERO_STEM}_{number:06d}.jpg")
        path = FRAMES / f"163855_{number:06d}.jpg"
        if image is None or not path.exists():
            continue
        cameras.append((number, scene3d.camera_from_reconstruction(
            image, rec.cameras[image.camera_id], alignment, origin, clip=HERO)))
        images[number] = cv2.imread(str(path))[:, :, ::-1]

    packs = {}
    for label, width in ORTHO_WIDTH.items():
        rect = drape.texture_rect(frame, width)
        surface = drape.heightfield(cloud, frame, rect,
                                    posting_m=posting[label])
        ortho = drape.orthophoto(surface, frame, cloud, cameras, images)
        target = OUT / label / "ortho.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), ortho.rgb[:, :, ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, ORTHO_QUALITY[label]])
        packs[label] = (surface, ortho, target.stat().st_size)
    return frame, cloud, packs


def surface_payload(frame, packs, label: str) -> dict:
    """Geometry, texture coordinates and per-vertex confidence for one pack."""
    surface, ortho, size = packs[label]
    drawn = np.unique(surface.indices)
    return {
        "frame": frame.as_dict(),
        "rect": surface.rect.as_dict(),
        "posting_m": surface.posting_m,
        "rows": surface.rows,
        "cols": surface.cols,
        "area_m2": round(surface.area_m2, 1),
        "vertices_drawn": int(len(drawn)),
        "triangles": int(len(surface.indices)),
        "measured_fraction": round(surface.measured_fraction, 4),
        "max_support_m": drape.MAX_SUPPORT_M,
        "measured_support_m": drape.MEASURED_SUPPORT_M,
        "ortho_covered_fraction": round(ortho.covered_fraction, 4),
        "ortho_frames": ortho.frames,
        "ortho_bytes": size,
        "ortho": "ortho.jpg",
        "note": ("heights interpolated from the reconstruction over a plane fitted to "
                 "it; cells with no reconstructed point within "
                 f"{drape.MAX_SUPPORT_M} m are not drawn at all"),
        # Texture coordinates are omitted on purpose: they are an exact function of
        # position, the frozen surface frame and the texture rectangle, so shipping
        # them would be a megabyte of base64 restating what the viewer can derive.
        "uv_from": "position projected on surface.frame, normalised over surface.rect",
        "support_scale_m": round(drape.MAX_SUPPORT_M / 255.0, 6),
        "vertices_float32_base64": b64(surface.vertices.astype(np.float32)),
        "support_uint8_base64": b64(
            np.clip(surface.support_m / drape.MAX_SUPPORT_M * 255.0,
                    0, 255).astype(np.uint8)),
        "indices_uint32_base64": b64(surface.indices.astype(np.uint32)),
    }


def still_quads(origin) -> list[dict]:
    """Photo quads hung at each still's own laser target.

    This is the only honest way to put the rucksack in the model. Every clip that looks
    at it is a hover, so there is no baseline and no reconstruction; what there is, is a
    3840x2160 photograph whose laser rangefinder measured the target at 145.0 m. The
    quad states direction and range, which are measured, and asserts nothing about the
    surface, which is not.
    """
    out = []
    for name, filename in STILL_QUADS.items():
        path = VIDEO_DIR / filename
        if not path.exists():
            continue
        pose = stills.read_still(path)
        if pose.lrf is None or not pose.lrf.ok:
            continue
        node = scene3d.camera_from_pose(pose.fix, pose.f_px, origin, source="still",
                                        clip=filename, ident=filename,
                                        width=pose.width, height=pose.height)
        centre = origin.enu(pose.lrf.lat, pose.lrf.lon, pose.lrf.alt_msl)
        rng = float(pose.lrf.distance_m)
        half_w = 0.5 * pose.width * rng / pose.f_px
        half_h = 0.5 * pose.height * rng / pose.f_px
        corners = [centre - half_w * node.right + half_h * node.up,
                   centre + half_w * node.right + half_h * node.up,
                   centre + half_w * node.right - half_h * node.up,
                   centre - half_w * node.right - half_h * node.up]
        out.append({
            "name": name,
            "source_still": filename,
            "image": f"still_{name}.jpg",
            "centre_m": [round(float(v), 3) for v in centre],
            "camera_m": [round(float(v), 3) for v in node.position],
            "corners_m": [[round(float(v), 3) for v in c] for c in corners],
            "range_m": round(rng, 1),
            "width_m": round(2 * half_w, 2),
            "height_m": round(2 * half_h, 2),
            "gsd_cm_px": round(rng / pose.f_px * 100, 4),
            "note": "faces the camera, not the ground; direction and range are measured",
        })
    return out


def write_still_images(quads: list[dict]) -> dict:
    """Re-encode each still quad's photograph at a size proportional to its detail."""
    written = {}
    for quad in quads:
        source = VIDEO_DIR / quad["source_still"]
        if not source.exists():
            continue
        img = cv2.imread(str(source))
        for label, width in (("full", 1600), ("lite", 1100)):
            dst = OUT / label / quad["image"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            scale = width / img.shape[1]
            small = cv2.resize(img, (width, int(round(img.shape[0] * scale))),
                               interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dst), small, [cv2.IMWRITE_JPEG_QUALITY, 80])
            written.setdefault(label, 0)
            written[label] += dst.stat().st_size
    return written


def orange_blob(frame: int, seed: tuple[float, float]) -> tuple[np.ndarray, int] | None:
    """Centroid and area of the orange component nearest ``seed`` in a frame.

    Orange against snow-dusted scree is a strong Lab signal, so a chroma and hue
    gate finds it without a detector. The seed only disambiguates which blob.
    """
    path = FRAMES / f"163855_{frame:06d}.jpg"
    if not path.exists():
        return None
    img = cv2.imread(str(path))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32)
    lightness, a, b = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    chroma = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a)) % 360
    mask = ((chroma > 30) & (hue > 25) & (hue < 75) & (lightness > 60)).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    found = [(stats[i, cv2.CC_STAT_AREA], centroids[i]) for i in range(1, count)
             if stats[i, cv2.CC_STAT_AREA] >= 4]
    if not found:
        return None
    area, centre = min(found, key=lambda t: float(np.hypot(t[1][0] - seed[0],
                                                          t[1][1] - seed[1])))
    return centre, int(area)


def solve_orange(rec, alignment, origin) -> tuple[tuple[float, float, float], dict]:
    """Triangulate the orange object from the frames that show it.

    Uses the reconstruction's own cameras, whose pose and focal are solved, so
    this needs no terrain model and inherits no DEM error.
    """
    by_name = {im.name: im for im in rec.images.values()}
    rays, evidence = [], []
    for frame, seed in ORANGE_SIGHTINGS.items():
        image = by_name.get(f"{HERO_STEM}_{frame:06d}.jpg")
        blob = orange_blob(frame, seed)
        if image is None or blob is None:
            continue
        centre, area = blob
        node = scene3d.camera_from_reconstruction(image, rec.cameras[image.camera_id],
                                                  alignment, origin, clip=HERO)
        rays.append((f"frame {frame}", node.position,
                     scene3d.pixel_direction(node, centre[0], centre[1])))
        evidence.append({"frame": frame, "pixel": [round(float(v), 1) for v in centre],
                         "blob_px": area, "f_px": round(node.f_px, 1)})
    if len(rays) < 2:
        raise SystemExit("not enough sightings of the orange object to triangulate")

    result = scene3d.triangulate(rays)
    for item, (_, along, miss) in zip(evidence, result.per_ray, strict=True):
        item["range_m"] = round(along, 1)
        item["ray_miss_m"] = round(miss, 3)
        gsd_cm = along / item["f_px"] * 100
        item["gsd_cm_px"] = round(gsd_cm, 2)
        item["size_cm"] = round(float(np.sqrt(item["blob_px"])) * gsd_cm, 1)
    lat, lon, alt = origin.geodetic(result.position)
    report = {
        "method": "least-squares ray intersection, reconstruction cameras only",
        "rays": result.n_rays,
        "max_baseline_m": round(result.max_baseline_m, 1),
        "ray_miss_median_m": round(result.miss_median_m, 3),
        "ray_miss_max_m": round(result.miss_max_m, 3),
        "well_conditioned": result.well_conditioned,
        "caveat": ("the miss distances describe internal consistency; absolute "
                   "accuracy is limited by the georeferencing, whose camera "
                   "residual is about 2 m and which reads about 3 m high at the "
                   "held-out laser points"),
        "sightings": evidence,
    }
    return (lat, lon, alt), report


def object_payload(origin, terrain, layer_sep: float, orange) -> list[dict]:
    """The laser objects and the triangulated one, each with its DEM disagreement."""
    entries = []
    catalogue = dict(scene3d.OBJECTS)
    catalogue["orange object"] = orange
    described = {
        "rucksack": "Blue pack on a rock slab, straps loose. Laser range 145.0 m.",
        "pole": "Trekking pole, cork grip, teal shaft. Laser range 43.7 m.",
        "lid": "Round turquoise plastic rim, 45 mm across. Laser range 43.7 m.",
        "orange object": ("Bright orange item, 35 to 40 cm across, on dark scree. "
                          "Triangulated from five reconstructed frames."),
    }
    for name, (lat, lon, alt) in catalogue.items():
        dem_ortho = terrain.elevation(lat, lon)
        dem_ellip = None if dem_ortho is None else dem_ortho - layer_sep
        entries.append({
            "name": name,
            "lat": lat,
            "lon": lon,
            "elev_ellipsoidal_m": alt,
            "elev_orthometric_m": round(alt + layer_sep, 1),
            "position_m": [round(float(v), 3) for v in origin.enu(lat, lon, alt)],
            "dem_ellipsoidal_m": None if dem_ellip is None else round(dem_ellip, 1),
            "dem_minus_object_m": None if dem_ellip is None else round(dem_ellip - alt, 1),
            "source": "laser rangefinder" if name in scene3d.OBJECTS else "triangulated",
            "description": described.get(name, ""),
        })
    return entries


def fall_line_payload(origin, terrain, sep: float) -> dict:
    """Steepest descent from the rucksack, in scene metres, split at the lid."""
    grid = terrain_prior.load_grid(terrain, CONTEXT_BOUNDS[0], CONTEXT_BOUNDS[2],
                                   CONTEXT_BOUNDS[1], CONTEXT_BOUNDS[3], pad_cells=20)
    fdir = terrain_prior.flow_direction(grid)
    lat, lon, _ = scene3d.OBJECTS["rucksack"]
    path = terrain_prior.trace_fall_line(grid, lat, lon, fdir)
    pts = np.array([origin.enu(la, lo, elev - sep) for la, lo, elev in path])
    lowest_searched = scene3d.OBJECTS["lid"][2]
    searched = [i for i, (_, _, e) in enumerate(path) if (e - sep) >= lowest_searched]
    split = max(searched) if searched else 0
    return {
        "n": len(path),
        "split_index": int(split),
        "split_reason": (
            "nothing has been searched below the lid at "
            f"{lowest_searched:.1f} m ellipsoidal"
        ),
        "xyz_float32_base64": b64(pts.astype(np.float32)),
        "end_lat": path[-1][0] if path else None,
        "end_lon": path[-1][1] if path else None,
        "end_elev_ellipsoidal_m": round(path[-1][2] - sep, 1) if path else None,
    }


def write_images(rec, alignment, origin) -> dict:
    """Re-encode the shipped frames at two sizes and report what was written."""
    written = {}
    for frame in SHIP_FRAMES:
        src = FRAMES / f"163855_{frame:06d}.jpg"
        if not src.exists():
            continue
        img = cv2.imread(str(src))
        for label, width, quality in (("full", 1600, 82), ("lite", 1280, 78)):
            if label == "lite" and frame not in LITE_SHIP_FRAMES:
                continue
            dst = OUT / label / f"163855_{frame:06d}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            scale = width / img.shape[1]
            small = cv2.resize(img, (width, int(round(img.shape[0] * scale))),
                               interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dst), small, [cv2.IMWRITE_JPEG_QUALITY, quality])
            written.setdefault(label, []).append(dst.stat().st_size)
    return {k: {"count": len(v), "bytes": int(sum(v))} for k, v in written.items()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    origin = scene3d.default_origin()
    terrain = georef.load_terrain()
    sep = scene3d.geoid_separation(origin.lat, origin.lon)

    rec, alignment = load_model()
    track = telemetry.read_track(VIDEO_DIR / f"{HERO}.MP4")

    context = scene3d.build_terrain_layer(terrain, "context", CONTEXT_BOUNDS[0],
                                          CONTEXT_BOUNDS[2], CONTEXT_BOUNDS[1],
                                          CONTEXT_BOUNDS[3])
    site = scene3d.build_terrain_layer(terrain, "site", SITE_BOUNDS[0], SITE_BOUNDS[2],
                                       SITE_BOUNDS[1], SITE_BOUNDS[3], pad_cells=6)

    cloud = cloud_payload(rec, alignment)
    checks = scene3d.check_objects(alignment.apply(
        np.array([p.xyz for p in rec.points3D.values()])), origin)
    orange, orange_report = solve_orange(rec, alignment, origin)
    surface_frame, _, packs = build_surface(rec, alignment, origin)
    quads = still_quads(origin)

    scene = {
        "generated_by": "scripts/build_scene3d.py",
        "site": "Kurumdy, Pamir",
        "frame": {
            "kind": "local East-North-Up, metres",
            "origin_lat": origin.lat,
            "origin_lon": origin.lon,
            "origin_alt_ellipsoidal_m": origin.alt_ellipsoidal,
            "origin_is": "the trekking pole, a laser-measured point",
            "vertical_datum": "WGS84 ellipsoidal throughout",
            "geoid_separation_m": round(sep, 3),
            "note": ("the DEM is EGM2008 orthometric and was converted on import; "
                     "laser object elevations are already ellipsoidal"),
        },
        "terrain": [terrain_payload(context, origin), terrain_payload(site, origin)],
        "surface": {label: surface_payload(surface_frame, packs, label) for label in packs},
        "quads": quads,
        "reconstruction": {
            "model": "model0",
            "clip": HERO,
            "registered_images": rec.num_reg_images(),
            "points3D": len(rec.points3D),
            "mean_reprojection_error_px": round(rec.compute_mean_reprojection_error(), 3),
            "scale": round(alignment.scale, 4),
            "camera_residual_median_m": round(alignment.residual_median_m, 2),
            "camera_residual_p90_m": round(alignment.residual_p90_m, 2),
            "inliers": f"{alignment.n_inliers}/{alignment.n_cameras}",
            "cloud": cloud,
        },
        "objects": object_payload(origin, terrain, sep, orange),
        "fall_line": fall_line_payload(origin, terrain, sep),
        "cameras": build_cameras(rec, alignment, origin, track),
        "verification": {
            "held_out_laser_check": [
                {"object": c.name,
                 "surface_error_m": None if not c.measured else round(c.surface_error_m, 2),
                 "points_within_3m": c.n_points_within}
                for c in checks
            ],
            "note": ("the laser points were never used to build or georeference the "
                     "reconstruction, so these are independent"),
            "orange_object_triangulation": orange_report,
        },
    }
    scene["assets"] = write_images(rec, alignment, origin)
    still_bytes = write_still_images(quads)

    (OUT / "scene.json").write_text(json.dumps(scene, indent=1))
    size = (OUT / "scene.json").stat().st_size
    print(f"scene.json  {size/1024:.0f} KB")
    print(f"  terrain   context {context.shape}  site {site.shape}")
    print(f"  cloud     {cloud['n']} points")
    print(f"  cameras   {len(scene['cameras'])}")
    print(f"  objects   {len(scene['objects'])}")
    described = ", ".join(f"{q['name']} {q['gsd_cm_px']} cm/px" for q in quads)
    print(f"  quads     {described}")
    print(f"  fall line {scene['fall_line']['n']} points, "
          f"split at index {scene['fall_line']['split_index']}")
    for label, payload in scene["surface"].items():
        print(f"  surface   {label}: {payload['posting_m']} m posting, "
              f"{payload['area_m2']:.0f} m2, {payload['triangles']} tris, "
              f"{payload['measured_fraction']*100:.0f}% measured, "
              f"ortho {payload['rect']['width_px']}x{payload['rect']['height_px']} "
              f"({payload['rect']['metres_per_texel']*100:.1f} cm/texel, "
              f"{payload['ortho_bytes']/1e6:.2f} MB, "
              f"{payload['ortho_covered_fraction']*100:.0f}% covered)")
    for label, info in scene["assets"].items():
        extra = still_bytes.get(label, 0)
        print(f"  images    {label}: {info['count']} frames {info['bytes']/1e6:.1f} MB "
              f"+ stills {extra/1e6:.1f} MB")


if __name__ == "__main__":
    main()
