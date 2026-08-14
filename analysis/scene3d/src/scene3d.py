"""A 3D scene of the object site: terrain, reconstruction, cameras and photographs.

What this is for.

``out/finds/kurumdy-objects.html`` states where the three objects are. It cannot
show where they sit on the mountain, what each camera actually saw, or which
ground was never imaged. This module builds the scene description that can, and
does it in one metric frame so that every entity in it is comparable.

The frame.

Everything is local East-North-Up in metres, origin at the pole's coordinate.
Metres rather than degrees, because at 39.48 N a degree of longitude is 0.77 of
a degree of latitude: a mesh built in raw degrees is stretched 29 percent
east-west, and every slope and fall-line bearing computed on it is wrong while
looking entirely normal.

The datum, which is the part that bites.

Three vertical datums are in play and mixing them silently moves things by tens
of metres:

  Fix.alt_msl          WGS84 ellipsoidal
  LrfFix.alt_msl       WGS84 ellipsoidal too, because it is the camera altitude
                       plus range times sine of pitch (see stills.LrfFix)
  the Copernicus DEM   EGM2008 orthometric, 33.16 m below the ellipsoid here

So the scene is built ellipsoidal throughout: cameras, reconstruction and
objects already share that frame, which is why georeferencing from telemetry
alone lands within metres of the laser points rather than 33 m away. Terrain
elevations are converted on the way in, and only there. ``elev_ellipsoidal_m``
and ``elev_orthometric_m`` are never both called ``elev``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EARTH_R = 6_371_000.0

#: The pole, and the origin of the scene frame. Chosen because it is a measured
#: point rather than a round number, so the frame is reproducible from the
#: published coordinates alone.
SITE_ORIGIN = (39.483176, 73.585463, 4530.0)

#: The three laser-measured objects. Elevations are ELLIPSOIDAL: they come from
#: LRFTargetAbsAlt, which is derived from the drone's own ellipsoidal altitude.
#: out/finds/OBJECT_COORDINATES.md publishes these without naming the datum.
OBJECTS: dict[str, tuple[float, float, float]] = {
    "rucksack": (39.482656, 73.586792, 4663.0),
    "pole": (39.483176, 73.585463, 4529.3),
    "lid": (39.483144, 73.585443, 4530.8),
}


@dataclass(frozen=True)
class SceneOrigin:
    """Origin of the local ENU frame. Height is ellipsoidal, like everything else."""

    lat: float
    lon: float
    alt_ellipsoidal: float

    @property
    def m_per_deg_lat(self) -> float:
        return math.pi / 180.0 * EARTH_R

    @property
    def m_per_deg_lon(self) -> float:
        return math.pi / 180.0 * EARTH_R * math.cos(math.radians(self.lat))

    def enu(self, lat: float, lon: float, alt: float) -> np.ndarray:
        """Geodetic to local ENU metres. ``alt`` must be ellipsoidal."""
        return np.array([
            (lon - self.lon) * self.m_per_deg_lon,
            (lat - self.lat) * self.m_per_deg_lat,
            alt - self.alt_ellipsoidal,
        ])

    def enu_many(self, latlonalt: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`enu` over an (n, 3) array of lat, lon, alt."""
        out = np.empty_like(np.asarray(latlonalt, dtype=float))
        arr = np.asarray(latlonalt, dtype=float)
        out[:, 0] = (arr[:, 1] - self.lon) * self.m_per_deg_lon
        out[:, 1] = (arr[:, 0] - self.lat) * self.m_per_deg_lat
        out[:, 2] = arr[:, 2] - self.alt_ellipsoidal
        return out

    def geodetic(self, enu: np.ndarray) -> tuple[float, float, float]:
        """Local ENU metres back to lat, lon, ellipsoidal alt."""
        e, n, u = float(enu[0]), float(enu[1]), float(enu[2])
        return (
            self.lat + n / self.m_per_deg_lat,
            self.lon + e / self.m_per_deg_lon,
            self.alt_ellipsoidal + u,
        )


def default_origin() -> SceneOrigin:
    return SceneOrigin(*SITE_ORIGIN)


def camera_centre(image) -> np.ndarray:
    """Camera centre of a registered pycolmap Image, in model coordinates.

    ``cam_from_world`` is a method in pycolmap 4.x and was a property earlier;
    ``projection_center`` is likewise a method here. Reading the pose through
    one helper keeps that version skew in a single place instead of scattering
    ``AttributeError`` risk through every caller.
    """
    centre = getattr(image, "projection_center", None)
    if centre is not None:
        value = centre() if callable(centre) else centre
        return np.asarray(value, dtype=float)
    pose = image.cam_from_world
    pose = pose() if callable(pose) else pose
    return np.asarray(pose.inverse().translation, dtype=float)


@dataclass
class Alignment:
    """A reconstruction's similarity transform into the scene frame, with its error.

    ``residual_median_m`` is the whole point of keeping this as a value rather
    than applying the transform and forgetting it: a model that aligns to within
    half a metre and one that aligns to within twenty look identical once the
    points are drawn.
    """

    model: str
    scale: float
    rotation: np.ndarray             # 3x3, model axes to scene axes
    translation: np.ndarray          # 3, metres
    n_cameras: int
    n_inliers: int
    residual_median_m: float
    residual_p90_m: float
    notes: list[str] = field(default_factory=list)

    @property
    def inlier_fraction(self) -> float:
        return self.n_inliers / self.n_cameras if self.n_cameras else 0.0

    def apply(self, xyz: np.ndarray) -> np.ndarray:
        """Transform model coordinates (n, 3) into scene ENU metres."""
        arr = np.asarray(xyz, dtype=float).reshape(-1, 3)
        return self.scale * (arr @ self.rotation.T) + self.translation


#: Least camera spread, in metres, that can pin the scale of a similarity fit.
#: A hover gives the fit nothing to hold on to: every camera sits in the same
#: place, so any scale reproduces them equally well.
MIN_CAMERA_SPREAD_M = 8.0

#: Least ratio of smallest to largest principal spread of the camera positions.
#: Cameras strung along a line, or lying in a plane with no vertical spread,
#: let scale trade against rotation. The fit then converges on a tiny scale that
#: collapses the point cloud while still reporting a sub-metre camera residual.
MIN_CAMERA_CONDITION = 0.02


def telemetry_degeneracy(camera_enu: np.ndarray) -> str | None:
    """Say why these camera positions cannot pin a similarity transform, or None.

    This exists because the failure it catches is invisible downstream. Fitting
    a similarity to 22 cameras that are all within 30 cm of each other yields a
    scale of 0.03, a point cloud shrunk to nothing, and a reported residual of
    0.07 m - which reads as the best model in the set rather than the worst.
    """
    if len(camera_enu) < 4:
        return f"only {len(camera_enu)} positioned cameras, need at least 4"
    spread = float(np.linalg.norm(np.ptp(camera_enu, axis=0)))
    if spread < MIN_CAMERA_SPREAD_M:
        return (f"camera positions span only {spread:.1f} m: a hover cannot "
                "constrain scale, so any alignment is arbitrary")
    sv = np.linalg.svd(camera_enu - camera_enu.mean(axis=0), compute_uv=False)
    condition = float(sv[-1] / sv[0]) if sv[0] > 0 else 0.0
    if condition < MIN_CAMERA_CONDITION:
        return (f"camera positions are near-degenerate (axis ratio {condition:.4f}): "
                "collinear or coplanar cameras let scale trade against rotation")
    return None


def align_to_telemetry(
    rec,
    poses: dict[str, list[float]],
    origin: SceneOrigin,
    *,
    max_error_m: float = 5.0,
    model_name: str = "",
) -> Alignment | None:
    """Fit a robust similarity transform from a reconstruction into the scene frame.

    The correspondence is one camera centre per registered image against that
    frame's telemetry position. RANSAC rather than a plain least-squares fit,
    because a single frame that registered into the wrong place drags an
    unweighted Kabsch fit by metres and leaves no trace that it did.

    Returns None when too few registered images carry telemetry to fit at all.
    """
    import pycolmap

    src, dst = [], []
    for image in rec.images.values():
        p = poses.get(image.name)
        if p is None:
            continue
        src.append(camera_centre(image))
        dst.append(origin.enu(*p))
    if len(src) < 4:
        return None

    src_a = np.asarray(src, dtype=float)
    dst_a = np.asarray(dst, dtype=float)
    degenerate = telemetry_degeneracy(dst_a)
    if degenerate is not None:
        return Alignment(model=model_name, scale=float("nan"), rotation=np.eye(3),
                         translation=np.zeros(3), n_cameras=len(src_a), n_inliers=0,
                         residual_median_m=float("nan"), residual_p90_m=float("nan"),
                         notes=[degenerate])

    opts = pycolmap.RANSACOptions()
    opts.max_error = max_error_m
    result = pycolmap.estimate_sim3d_robust(src_a, dst_a, opts)
    if result is None:
        return None
    # pycolmap 4.1 returns {"tgt_from_src", "num_inliers", "inlier_mask"};
    # the type stub still advertises a bare Sim3d, so accept either.
    sim3 = result["tgt_from_src"] if isinstance(result, dict) else result
    if sim3 is None:
        return None

    scale = float(sim3.scale)
    rot = np.asarray(sim3.rotation.matrix(), dtype=float)
    trans = np.asarray(sim3.translation, dtype=float)
    resid = np.linalg.norm(dst_a - (scale * (src_a @ rot.T) + trans), axis=1)
    inliers = int(result["num_inliers"]) if isinstance(result, dict) else int(
        (resid <= max_error_m).sum())

    notes = []
    if scale <= 0:
        notes.append("non-positive scale: the fit is degenerate")
    if inliers < 0.5 * len(src_a):
        notes.append(
            f"only {inliers}/{len(src_a)} cameras within {max_error_m:.0f} m; "
            "the model may be split or mis-registered"
        )
    return Alignment(
        model=model_name,
        scale=scale,
        rotation=rot,
        translation=trans,
        n_cameras=len(src_a),
        n_inliers=inliers,
        residual_median_m=float(np.median(resid)),
        residual_p90_m=float(np.percentile(resid, 90)),
        notes=notes,
    )


@dataclass
class ObjectCheck:
    """How close a reconstruction's surface comes to one laser-measured object."""

    name: str
    n_points_within: int
    surface_error_m: float | None     # median reconstructed height minus laser height
    nearest_point_m: float

    @property
    def measured(self) -> bool:
        return self.surface_error_m is not None


def check_objects(
    cloud_enu: np.ndarray,
    origin: SceneOrigin,
    *,
    radius_m: float = 3.0,
    objects: dict[str, tuple[float, float, float]] | None = None,
) -> list[ObjectCheck]:
    """Compare a reconstructed surface against the laser points held out of the fit.

    The comparison is local median HEIGHT within a horizontal radius, not
    distance to the nearest point. Nearest-point distance is the tempting
    version and it is meaningless: in a cloud of 70,000 points spread over
    100 m, the expected distance to the nearest point is a couple of metres for
    any target whatsoever, so it reports a pass wherever you aim it.
    """
    objs = objects if objects is not None else OBJECTS
    out: list[ObjectCheck] = []
    for name, (lat, lon, alt) in objs.items():
        target = origin.enu(lat, lon, alt)
        horiz = np.linalg.norm(cloud_enu[:, :2] - target[:2], axis=1)
        near = horiz <= radius_m
        n = int(near.sum())
        nearest = (float(np.linalg.norm(cloud_enu - target, axis=1).min())
                   if len(cloud_enu) else float("inf"))
        err = float(np.median(cloud_enu[near, 2]) - target[2]) if n >= 4 else None
        out.append(ObjectCheck(name=name, n_points_within=n, surface_error_m=err,
                               nearest_point_m=nearest))
    return out


@dataclass
class CameraNode:
    """One photograph placed in the scene: where it was taken from and where it looked.

    ``right``, ``up`` and ``forward`` are unit vectors in scene ENU. They are
    stored explicitly rather than as a quaternion so that a reader can check the
    handedness by eye, and so that the two very different sources of pose - a
    bundle-adjusted reconstruction and raw gimbal telemetry - land in exactly
    the same representation and can be compared.

    ``source`` is load-bearing: "sfm" poses are solved to sub-pixel and carry a
    solved focal, "telemetry" poses inherit the gimbal's own accuracy and have
    no roll term at all, and "still" poses come from XMP with an EXIF focal.
    Anything reading this file needs to know which it is holding.
    """

    ident: str
    source: str                      # "sfm" | "telemetry" | "still"
    clip: str
    frame: int
    t_s: float | None
    position: np.ndarray             # scene ENU metres
    right: np.ndarray
    up: np.ndarray
    forward: np.ndarray
    f_px: float
    width: int
    height: int
    image: str | None = None         # asset filename, when a photograph ships
    range_m: float | None = None
    gsd_cm_px: float | None = None
    notes: list[str] = field(default_factory=list)

    def basis_matrix(self) -> np.ndarray:
        """Right-handed camera basis as columns, for a renderer to use directly.

        ``forward`` is the viewing direction, so ``right``, ``up``, ``forward``
        is LEFT-handed (right x up = -forward). Graphics conventions - three.js,
        OpenGL - put the camera looking down its own -Z, so the renderable basis
        is ``[right, up, -forward]``, which is right-handed. Exporting the raw
        triple without this would mirror every photograph, and a mirrored snow
        slope looks exactly like a snow slope.
        """
        return np.column_stack([self.right, self.up, -self.forward])

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.height / (2.0 * self.f_px)))

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.width / (2.0 * self.f_px)))


def camera_from_reconstruction(image, camera, alignment: Alignment, origin: SceneOrigin,
                               *, clip: str = "", ident: str = "") -> CameraNode:
    """Turn a registered COLMAP image into a scene camera.

    COLMAP's camera axes are x right, y DOWN, z forward. The scene keeps y up,
    so ``up`` is the negated second row. Getting this backwards renders every
    photograph upside down, which is at least obvious; getting the handedness
    backwards mirrors them, which is not.
    """
    pose = image.cam_from_world
    pose = pose() if callable(pose) else pose
    rot_cam_from_world = np.asarray(pose.rotation.matrix(), dtype=float)
    # Model axes to scene axes, then camera axes read off the rows.
    rot_cam_from_scene = rot_cam_from_world @ alignment.rotation.T
    right = rot_cam_from_scene[0]
    down = rot_cam_from_scene[1]
    forward = rot_cam_from_scene[2]
    centre = alignment.apply(camera_centre(image))[0]
    frame = _frame_from_name(image.name)
    return CameraNode(
        ident=ident or image.name,
        source="sfm",
        clip=clip,
        frame=frame,
        t_s=None,
        position=centre,
        right=right / np.linalg.norm(right),
        up=-down / np.linalg.norm(down),
        forward=forward / np.linalg.norm(forward),
        f_px=float(camera.params[0]),
        width=int(camera.width),
        height=int(camera.height),
    )


def camera_from_pose(fix, f_px: float, origin: SceneOrigin, *, source: str,
                     clip: str = "", ident: str = "", width: int = 1920,
                     height: int = 1080) -> CameraNode:
    """Place a camera from gimbal telemetry or still XMP, via ``georef.camera_basis``.

    Reuses the repo's basis rather than rebuilding it: yaw is a compass bearing
    clockwise from north and pitch is positive upward, and a second
    implementation of that convention would be a second chance to get a sign
    wrong. There is no roll field on this aircraft, so roll is zero here by
    necessity, not by choice.
    """
    from .georef import camera_basis

    forward, right, down = camera_basis(fix.gimbal_yaw, fix.gimbal_pitch)
    return CameraNode(
        ident=ident,
        source=source,
        clip=clip,
        frame=int(fix.frame),
        t_s=fix.t_s if getattr(fix, "t_us", None) is not None else None,
        position=origin.enu(fix.lat, fix.lon, fix.alt_msl),
        right=np.asarray(right, dtype=float),
        up=-np.asarray(down, dtype=float),
        forward=np.asarray(forward, dtype=float),
        f_px=float(f_px),
        width=width,
        height=height,
        notes=["gimbal roll is not reported by this aircraft and is taken as zero"],
    )


def pixel_direction(node: CameraNode, x: float, y: float) -> np.ndarray:
    """Unit ray in scene ENU through pixel (x, y) of this camera.

    Pinhole, principal point at the image centre, no distortion - the same model
    ``georef.pixel_ray`` uses, expressed against a camera whose axes are already
    in scene coordinates. Image y runs down, and ``up`` points up, hence the sign.
    """
    dx = x - node.width / 2.0
    dy = y - node.height / 2.0
    d = dx * node.right - dy * node.up + node.f_px * node.forward
    return d / np.linalg.norm(d)


@dataclass
class Triangulation:
    """A point solved from two or more rays, with the evidence that it is real."""

    position: np.ndarray                 # scene ENU metres
    n_rays: int
    miss_median_m: float                 # how far each ray passes from the solution
    miss_max_m: float
    max_baseline_m: float
    per_ray: list[tuple[str, float, float]] = field(default_factory=list)

    @property
    def well_conditioned(self) -> bool:
        """Rays from effectively one place cannot fix depth, however well they agree."""
        return self.max_baseline_m >= MIN_CAMERA_SPREAD_M


def triangulate(rays: list[tuple[str, np.ndarray, np.ndarray]]) -> Triangulation:
    """Least-squares intersection of rays given as (label, origin, unit direction).

    Reports how far each ray passes from the solution rather than a covariance.
    A miss distance is in metres and can be compared directly against the size of
    the thing being located, which is the question that actually matters.
    """
    if len(rays) < 2:
        raise ValueError("need at least two rays to triangulate")
    a = np.zeros((3, 3))
    b = np.zeros(3)
    for _, origin, direction in rays:
        unit = np.asarray(direction, dtype=float)
        unit = unit / np.linalg.norm(unit)
        proj = np.eye(3) - np.outer(unit, unit)
        a += proj
        b += proj @ np.asarray(origin, dtype=float)
    position = np.linalg.solve(a, b)

    per_ray, misses = [], []
    for label, origin, direction in rays:
        unit = np.asarray(direction, dtype=float)
        unit = unit / np.linalg.norm(unit)
        offset = position - np.asarray(origin, dtype=float)
        along = float(np.dot(offset, unit))
        miss = float(np.linalg.norm(offset - along * unit))
        per_ray.append((label, along, miss))
        misses.append(miss)

    centres = np.array([np.asarray(o, dtype=float) for _, o, _ in rays])
    baseline = float(max(
        np.linalg.norm(centres[i] - centres[j])
        for i in range(len(centres)) for j in range(i + 1, len(centres))
    ))
    return Triangulation(
        position=position,
        n_rays=len(rays),
        miss_median_m=float(np.median(misses)),
        miss_max_m=float(np.max(misses)),
        max_baseline_m=baseline,
        per_ray=per_ray,
    )


def _frame_from_name(name: str) -> int:
    """Frame index out of ``<clip>_<frame:06d>.jpg``; -1 when the name has none."""
    stem = str(name).rsplit(".", 1)[0]
    tail = stem.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else -1


def geoid_separation(lat: float, lon: float) -> float:
    """Orthometric height minus ellipsoidal height at a point, in metres.

    Measured here as -33.150 m, i.e. the EGM2008 geoid lies 33.15 m below the
    WGS84 ellipsoid. Raises rather than defaulting: a silent zero would put the
    DEM 33 m out and every terrain comparison with it.
    """
    from .georef import to_orthometric

    ortho = to_orthometric(lat, lon, 0.0)
    if ortho is None:
        raise DatumError(
            "the EGM2008 geoid grid is unavailable, so orthometric and ellipsoidal "
            "heights cannot be reconciled; refusing to guess a separation"
        )
    return float(ortho)


class DatumError(RuntimeError):
    """Raised when heights cannot be put in a common vertical datum."""


@dataclass
class TerrainLayer:
    """A DEM window as a heightfield in the scene's ellipsoidal frame.

    ``z_ellipsoidal`` is the DEM's orthometric elevation converted once, on the
    way in, so that nothing downstream has to remember which datum it is holding.
    Row 0 is the northern edge, matching ``terrain_prior.TerrainGrid``.
    """

    name: str
    z_ellipsoidal: np.ndarray        # (rows, cols) metres
    lat0: float                      # southern edge
    lon0: float                      # western edge
    lat1: float                      # northern edge
    lon1: float
    dlat: float                      # degrees per row, negative going south
    dlon: float                      # degrees per column
    separation_m: float              # orthometric minus ellipsoidal, for the record

    @property
    def shape(self) -> tuple[int, int]:
        return self.z_ellipsoidal.shape

    def enu_corners(self, origin: SceneOrigin) -> dict[str, float]:
        """Scene-frame bounds of this layer, so the viewer can place it without a mesh."""
        sw = origin.enu(self.lat0, self.lon0, 0.0)
        ne = origin.enu(self.lat1, self.lon1, 0.0)
        return {
            "east_min_m": float(sw[0]), "east_max_m": float(ne[0]),
            "north_min_m": float(sw[1]), "north_max_m": float(ne[1]),
            "up_min_m": float(np.nanmin(self.z_ellipsoidal) - origin.alt_ellipsoidal),
            "up_max_m": float(np.nanmax(self.z_ellipsoidal) - origin.alt_ellipsoidal),
        }


def build_terrain_layer(
    terrain,
    name: str,
    lat0: float,
    lon0: float,
    lat1: float,
    lon1: float,
    *,
    pad_cells: int = 20,
) -> TerrainLayer:
    """Cut a DEM window and convert it into the scene's ellipsoidal frame.

    The DEM is left at its native 30 m posting deliberately. Upsampling or
    smoothing it would manufacture the appearance of detail on the one surface
    here that is known to be 24 to 41 m wrong against the laser points.
    """
    from .terrain_prior import load_grid

    grid = load_grid(terrain, lat0, lon0, lat1, lon1, pad_cells=pad_cells)
    sep = geoid_separation((grid.lat0 + grid.lat1) / 2, (grid.lon0 + grid.lon1) / 2)
    return TerrainLayer(
        name=name,
        z_ellipsoidal=np.asarray(grid.z, dtype=float) - sep,
        lat0=grid.lat0, lon0=grid.lon0, lat1=grid.lat1, lon1=grid.lon1,
        dlat=grid.dlat, dlon=grid.dlon,
        separation_m=sep,
    )


def load_poses(path: str | Path) -> dict[str, list[float]]:
    """Read an image-name to [lat, lon, ellipsoidal alt] map written by an SfM run."""
    import json

    data = json.loads(Path(path).read_text())
    # Later runs group frames by role ("structure", "localise", ...); earlier ones
    # wrote a flat map. Flatten either into the same thing.
    if data and all(isinstance(v, dict) for v in data.values()):
        flat: dict[str, list[float]] = {}
        for group in data.values():
            flat.update(group)
        return flat
    return data


def parse_asset_name(name: str) -> tuple[str, int]:
    """Split ``0813163855_0001_Z_002355.jpg`` into its clip stem and frame index."""
    stem = str(name).rsplit(".", 1)[0]
    clip, _, tail = stem.rpartition("_")
    if not tail.isdigit():
        raise ValueError(f"{name!r} does not end in a frame index")
    return clip, int(tail)


def poses_from_telemetry(
    names, video_dir: str | Path, *, prefix: str = "DJI_2026"
) -> dict[str, list[float]]:
    """Camera positions for SfM image names, read from the videos themselves.

    Deliberately does not use the pose file an SfM run leaves behind. Those live
    in ``out/``, which is gitignored and is being rewritten by whichever job is
    running; the videos are the source of truth and do not move. One decode-free
    telemetry pass per clip.
    """
    from .telemetry import read_track

    wanted: dict[str, list[str]] = {}
    for name in names:
        clip, _ = parse_asset_name(name)
        wanted.setdefault(clip, []).append(name)

    out: dict[str, list[float]] = {}
    for clip, members in wanted.items():
        matches = sorted(Path(video_dir).glob(f"{prefix}*{clip}*.MP4"))
        if not matches:
            matches = sorted(Path(video_dir).glob(f"*{clip}*.MP4"))
        if not matches:
            raise FileNotFoundError(f"no video in {video_dir} for clip {clip!r}")
        track = read_track(matches[0])
        by_frame = {f.frame: f for f in track.positioned()}
        for name in members:
            _, frame = parse_asset_name(name)
            fix = by_frame.get(frame)
            if fix is not None:
                out[name] = [fix.lat, fix.lon, fix.alt_msl]
    return out
