"""A topographic surface for the object site, with the photographs baked onto it.

Why a baked orthophoto rather than runtime projective texturing.

The site is a plane. An SVD of the reconstructed cloud gives extents 33.5 by 19.7 by
5.6 m and a normal dipping 41.9 degrees, and the frames that cover it look almost
straight down that normal: median incidence 16 degrees, and only 0.12 percent of the
surface is backfacing to the widest frame. Reprojecting into a fixed grid therefore
costs about four percent of resolution, and in exchange the mosaic can be inspected
before it ships rather than being assembled at runtime inside a shader.

Why the grid is aligned to the plane rather than to the horizon.

On a 42 degree face a horizontal 2.5D grid wastes half its cells on ground the surface
does not pass over. Measured on this cloud: a horizontal grid has 50 percent of its
bounding box inside the hull, a plane-aligned one has 89 percent, and the fraction of
cells with a cloud point within a metre rises from 74 to 83 percent.

What stops this inventing terrain.

Two gates, both structural rather than advisory. A cell is drawn only if it lies inside
the convex hull of the cloud in the plane frame, and only if a cloud point sits within
``MAX_SUPPORT_M`` of it. A cell that fails is dropped from the index buffer, so it
cannot be drawn, textured, or painted on by anything downstream. The distance to the
nearest cloud point travels with each vertex as ``support_m`` so the viewer can show
which ground is measured and which is interpolated across a gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: A cell further than this from any cloud point is not surface, it is a guess.
#: Set at the knee of the measured curve: raising the gate from 1.5 to 2.0 m adds
#: 974 m2 of surface, 2.0 to 2.5 adds 497, and 2.5 to 3.0 adds 166. Past the knee the
#: gate is buying holes bridged over nothing.
MAX_SUPPORT_M = 2.0

#: Below this the cell is drawn at full saturation as measured ground. Above it, and
#: up to MAX_SUPPORT_M, the viewer desaturates: interpolated across a gap, not seen.
#: At 1.0 m posting this leaves 85.6 percent of drawn ground measured.
MEASURED_SUPPORT_M = 1.0

#: Beyond this angle between the viewing ray and the local surface normal, a frame
#: contributes nothing usable: the ground is nearly edge-on to it.
MAX_INCIDENCE_DEG = 70.0

#: A texel further behind the nearest surface seen along the same camera pixel than
#: this is occluded, and painting it would smear foreground onto hidden ground.
OCCLUSION_TOLERANCE_M = 1.5


@dataclass(frozen=True)
class SurfaceFrame:
    """A plane fitted to the cloud, and the 2D coordinate system on it.

    Frozen into the scene description on purpose. A denser cloud later would move an
    SVD refit by a fraction of a degree, which is enough to shift every texture
    coordinate and silently invalidate the orthophoto built against the old fit.
    """

    origin: np.ndarray               # scene ENU metres
    u: np.ndarray                    # unit, in-plane
    v: np.ndarray                    # unit, in-plane
    n: np.ndarray                    # unit normal, n_z > 0
    u_min: float
    u_max: float
    v_min: float
    v_max: float

    def to_plane(self, xyz: np.ndarray) -> np.ndarray:
        """Scene ENU (n, 3) to plane coordinates (n, 3) as (u, v, height above plane)."""
        d = np.asarray(xyz, dtype=float).reshape(-1, 3) - self.origin
        return np.column_stack([d @ self.u, d @ self.v, d @ self.n])

    def to_scene(self, uvw: np.ndarray) -> np.ndarray:
        """Plane coordinates (n, 3) back to scene ENU (n, 3)."""
        a = np.asarray(uvw, dtype=float).reshape(-1, 3)
        return (self.origin
                + a[:, 0:1] * self.u
                + a[:, 1:2] * self.v
                + a[:, 2:3] * self.n)

    def as_dict(self) -> dict:
        return {
            "origin_m": [round(float(x), 4) for x in self.origin],
            "u": [round(float(x), 6) for x in self.u],
            "v": [round(float(x), 6) for x in self.v],
            "n": [round(float(x), 6) for x in self.n],
            "u_range_m": [round(self.u_min, 3), round(self.u_max, 3)],
            "v_range_m": [round(self.v_min, 3), round(self.v_max, 3)],
            "dip_deg": round(float(np.degrees(np.arccos(abs(self.n[2])))), 2),
        }


def fit_surface_frame(cloud: np.ndarray, *, margin_m: float = 2.0) -> SurfaceFrame:
    """Fit a plane to the cloud and build a right-handed frame on it.

    Two sign conventions are pinned here rather than left to the SVD, whose output
    signs are arbitrary: the normal points up, and (u, v, n) is right-handed. Without
    both, the orthophoto comes out mirrored or upside down, and a mirrored snow slope
    looks exactly like a snow slope.
    """
    pts = np.asarray(cloud, dtype=float).reshape(-1, 3)
    if len(pts) < 3:
        raise ValueError("need at least three points to fit a plane")
    origin = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - origin, full_matrices=False)
    u, v, n = vt[0], vt[1], vt[2]

    if n[2] < 0:
        n = -n
    if np.dot(np.cross(u, v), n) < 0:
        v = -v

    plane = np.column_stack([(pts - origin) @ u, (pts - origin) @ v])
    return SurfaceFrame(
        origin=origin, u=u, v=v, n=n,
        u_min=float(plane[:, 0].min() - margin_m),
        u_max=float(plane[:, 0].max() + margin_m),
        v_min=float(plane[:, 1].min() - margin_m),
        v_max=float(plane[:, 1].max() + margin_m),
    )


@dataclass(frozen=True)
class TextureRect:
    """The plane rectangle the orthophoto covers, and the raster that covers it.

    The raster matches the plane rectangle's aspect exactly, so one texel is the same
    distance in u as in v and the ground scale printed in the viewer is honest.

    Sizes are multiples of four rather than powers of two. Squaring a 146 by 90 m site
    up to a power-of-two raster would spend 38 percent of the texture on ground the
    surface does not cover, which at 4096 wide is megabytes of blank in a file meant to
    be emailed. three.js r128 requests a WebGL2 context first, and WebGL2 mipmaps
    non-power-of-two textures without complaint; the viewer's existing capability check
    covers the case where only WebGL1 is available.
    """

    u0: float
    u1: float
    v0: float
    v1: float
    width: int
    height: int

    @property
    def metres_per_texel(self) -> float:
        return (self.u1 - self.u0) / self.width

    def texel_centres(self) -> tuple[np.ndarray, np.ndarray]:
        """(u, v) of every texel centre, row 0 at the top, i.e. at v1."""
        du = (self.u1 - self.u0) / self.width
        dv = (self.v1 - self.v0) / self.height
        u = self.u0 + (np.arange(self.width) + 0.5) * du
        v = self.v1 - (np.arange(self.height) + 0.5) * dv
        return np.meshgrid(u, v)

    def as_dict(self) -> dict:
        return {
            "u_range_m": [round(self.u0, 3), round(self.u1, 3)],
            "v_range_m": [round(self.v0, 3), round(self.v1, 3)],
            "width_px": self.width,
            "height_px": self.height,
            "metres_per_texel": round(self.metres_per_texel, 4),
        }


def texture_rect(frame: SurfaceFrame, width: int) -> TextureRect:
    """Choose a raster of the given width whose texels are square on the ground."""
    span_u = frame.u_max - frame.u_min
    span_v = frame.v_max - frame.v_min
    width = max(4, (int(width) // 4) * 4)
    height = max(4, (int(round(width * span_v / span_u)) // 4) * 4)
    # Grow whichever axis is short so metres per texel is identical in u and v.
    target = width / height
    if span_u / span_v < target:
        span_u = span_v * target
    else:
        span_v = span_u / target
    mid_u = 0.5 * (frame.u_min + frame.u_max)
    mid_v = 0.5 * (frame.v_min + frame.v_max)
    return TextureRect(
        u0=mid_u - span_u / 2, u1=mid_u + span_u / 2,
        v0=mid_v - span_v / 2, v1=mid_v + span_v / 2,
        width=width, height=height,
    )


@dataclass
class Surface:
    """A drawn topographic surface: geometry, texture coordinates and confidence."""

    vertices: np.ndarray             # (n, 3) scene ENU metres
    uv: np.ndarray                   # (n, 2) in [0, 1], matching the TextureRect
    indices: np.ndarray              # (m, 3) triangle vertex indices
    support_m: np.ndarray            # (n,) distance to the nearest cloud point
    normals: np.ndarray              # (n, 3) unit, scene ENU
    cell_drawn: np.ndarray           # (rows-1, cols-1) bool, which quads are drawn
    rows: int
    cols: int
    posting_m: float
    rect: TextureRect

    def texel_mask(self) -> np.ndarray:
        """Which texels lie under a drawn triangle, at the texture's resolution.

        The orthophoto must paint exactly the ground the mesh draws. Re-testing support
        per texel instead leaves unpainted specks scattered through every triangle whose
        interior happens to sit further from a cloud point than its corners do.
        """
        rect = self.rect
        cols = np.clip(((np.arange(rect.width) + 0.5) / rect.width
                        * (self.cols - 1)).astype(int), 0, self.cols - 2)
        rows = np.clip(((np.arange(rect.height) + 0.5) / rect.height
                        * (self.rows - 1)).astype(int), 0, self.rows - 2)
        return self.cell_drawn[np.ix_(rows, cols)]

    @property
    def area_m2(self) -> float:
        a = self.vertices[self.indices[:, 0]]
        b = self.vertices[self.indices[:, 1]]
        c = self.vertices[self.indices[:, 2]]
        return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())

    @property
    def measured_fraction(self) -> float:
        drawn = np.unique(self.indices)
        if not len(drawn):
            return 0.0
        return float((self.support_m[drawn] <= MEASURED_SUPPORT_M).mean())


def _grid_heights(cloud_plane: np.ndarray, uu: np.ndarray, vv: np.ndarray,
                  *, neighbours: int = 6):
    """Inverse-distance height and nearest-point distance on a (u, v) grid."""
    from scipy.spatial import cKDTree

    tree = cKDTree(cloud_plane[:, :2])
    query = np.column_stack([uu.ravel(), vv.ravel()])
    k = min(neighbours, len(cloud_plane))
    dist, idx = tree.query(query, k=k)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    weight = 1.0 / np.maximum(dist, 1e-3) ** 2
    height = (cloud_plane[idx, 2] * weight).sum(axis=1) / weight.sum(axis=1)
    return height.reshape(uu.shape), dist[:, 0].reshape(uu.shape)


def heightfield(cloud: np.ndarray, frame: SurfaceFrame, rect: TextureRect,
                *, posting_m: float = 1.0,
                max_support_m: float = MAX_SUPPORT_M) -> Surface:
    """Build a plane-aligned surface over the cloud, drawing only supported ground.

    A cell survives only if it is inside the convex hull of the cloud in the plane
    frame and has a cloud point within ``max_support_m``. Failing cells are left out
    of the index buffer entirely, so nothing downstream can draw or texture them. That
    is the whole defence against inventing terrain, and it is structural rather than a
    matter of remembering to check.
    """
    from scipy.spatial import Delaunay

    plane = frame.to_plane(cloud)
    cols = max(2, int(round((rect.u1 - rect.u0) / posting_m)) + 1)
    rows = max(2, int(round((rect.v1 - rect.v0) / posting_m)) + 1)
    u = np.linspace(rect.u0, rect.u1, cols)
    v = np.linspace(rect.v1, rect.v0, rows)        # row 0 at the top, matching the rect
    uu, vv = np.meshgrid(u, v)

    height, support = _grid_heights(plane, uu, vv)
    hull = Delaunay(plane[:, :2])
    inside = hull.find_simplex(np.column_stack([uu.ravel(), vv.ravel()])) >= 0
    inside = inside.reshape(uu.shape)
    valid = inside & (support <= max_support_m)

    vertices = frame.to_scene(np.column_stack([uu.ravel(), vv.ravel(), height.ravel()]))
    uv = np.column_stack([
        (uu.ravel() - rect.u0) / (rect.u1 - rect.u0),
        (vv.ravel() - rect.v0) / (rect.v1 - rect.v0),
    ])

    # Surface normal from the local gradient of height over the plane, rotated back
    # into scene axes. Needed to reject frames that see a cell nearly edge-on.
    dv_du, dv_dv = np.gradient(height, u[1] - u[0], v[0] - v[1])
    normals = (-dv_dv[..., None] * frame.u
               - dv_du[..., None] * frame.v
               + frame.n)
    normals = normals.reshape(-1, 3)
    normals /= np.linalg.norm(normals, axis=1)[:, None]

    index = np.arange(rows * cols).reshape(rows, cols)
    a, b = index[:-1, :-1], index[:-1, 1:]
    c, d = index[1:, :-1], index[1:, 1:]
    quad_ok = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    tris = np.concatenate([
        np.stack([a[quad_ok], c[quad_ok], b[quad_ok]], axis=1),
        np.stack([b[quad_ok], c[quad_ok], d[quad_ok]], axis=1),
    ])

    return Surface(vertices=vertices, uv=uv, indices=tris,
                   support_m=support.ravel(), normals=normals,
                   cell_drawn=quad_ok,
                   rows=rows, cols=cols, posting_m=posting_m, rect=rect)


def project_points(node, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project scene points into a camera. Returns (pixels, range, in_front).

    The inverse of ``scene3d.pixel_direction``, and it must stay that way: image y runs
    down while ``node.up`` points up, so the vertical term is negated. Deriving this
    afresh rather than mirroring that function is how a texture ends up flipped.
    """
    pts = np.asarray(xyz, dtype=float).reshape(-1, 3)
    d = pts - np.asarray(node.position, dtype=float)
    z = d @ np.asarray(node.forward, dtype=float)
    in_front = z > 1e-6
    safe = np.where(in_front, z, 1.0)
    x = node.f_px * (d @ np.asarray(node.right, dtype=float)) / safe + node.width / 2.0
    y = -node.f_px * (d @ np.asarray(node.up, dtype=float)) / safe + node.height / 2.0
    return np.column_stack([x, y]), np.linalg.norm(d, axis=1), in_front


def inside_image(pixels: np.ndarray, width: int, height: int,
                 *, margin_px: float = 2.0) -> np.ndarray:
    return ((pixels[:, 0] >= margin_px) & (pixels[:, 0] < width - margin_px)
            & (pixels[:, 1] >= margin_px) & (pixels[:, 1] < height - margin_px))


def sample_image(image: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Nearest-neighbour sample of an (h, w, 3) image at float pixel coordinates.

    Nearest rather than bilinear: at 7 cm per pixel against a 1 m grid the texel is
    coarser than the source pixel, so interpolation would only blur, and nearest keeps
    the sampled value traceable to one identifiable pixel of one identifiable frame.
    """
    h, w = image.shape[:2]
    cols = np.clip(np.round(pixels[:, 0]).astype(int), 0, w - 1)
    rows = np.clip(np.round(pixels[:, 1]).astype(int), 0, h - 1)
    return image[rows, cols]


def visible_mask(node, xyz: np.ndarray, normals: np.ndarray | None = None,
                 *, image_width: int | None = None,
                 image_height: int | None = None) -> np.ndarray:
    """Which points a camera can usefully see: in front, in frame, not edge-on."""
    width = image_width if image_width is not None else node.width
    height = image_height if image_height is not None else node.height
    pixels, ranges, in_front = project_points(node, xyz)
    ok = in_front & inside_image(pixels, width, height)
    if normals is not None:
        d = np.asarray(xyz, dtype=float).reshape(-1, 3) - np.asarray(node.position, float)
        rng = np.linalg.norm(d, axis=1)
        rng[rng == 0] = 1.0
        cos_incidence = np.abs(np.sum((d / rng[:, None]) * normals, axis=1))
        ok &= cos_incidence > np.cos(np.radians(MAX_INCIDENCE_DEG))
    del ranges
    return ok


@dataclass
class Orthophoto:
    """The baked mosaic, plus a record of which photograph each texel came from."""

    rgb: np.ndarray                  # (h, w, 3) uint8
    provenance: np.ndarray           # (h, w) uint8, 0 = unpainted, else 1-based frame slot
    frames: list[int]                # frame numbers, in the order they were painted
    covered_fraction: float          # of the supported surface
    per_frame: list[dict]

    def legend(self) -> dict[int, int]:
        return {slot + 1: frame for slot, frame in enumerate(self.frames)}


def _lab_gain_offset(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-channel linear map putting ``source`` on ``target``'s tone, in Lab.

    Tone only, never geometry. Without it the mosaic reads as a patchwork of separately
    exposed photographs, which is exactly what it is; with it, it reads as one picture.
    """
    import cv2

    if len(source) < 64:
        return np.array([[1.0, 0.0]] * 3)
    src = cv2.cvtColor(source.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2Lab)
    dst = cv2.cvtColor(target.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2Lab)
    src = src.reshape(-1, 3).astype(float)
    dst = dst.reshape(-1, 3).astype(float)
    out = []
    for c in range(3):
        sd = src[:, c].std()
        gain = float(dst[:, c].std() / sd) if sd > 1e-6 else 1.0
        gain = float(np.clip(gain, 0.5, 2.0))
        out.append([gain, float(dst[:, c].mean() - gain * src[:, c].mean())])
    return np.array(out)


def _apply_lab_gain(rgb: np.ndarray, gain_offset: np.ndarray) -> np.ndarray:
    import cv2

    lab = cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.uint8),
                       cv2.COLOR_RGB2Lab).reshape(-1, 3).astype(float)
    for c in range(3):
        lab[:, c] = lab[:, c] * gain_offset[c, 0] + gain_offset[c, 1]
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab.reshape(-1, 1, 3), cv2.COLOR_Lab2RGB).reshape(-1, 3)


def _feather(mask: np.ndarray, width_px: int) -> np.ndarray:
    """Alpha that is 1 well inside a footprint and ramps to 0 over its last few pixels.

    Feathering only at the boundary, never averaging whole overlaps: a fine frame and a
    coarse one blended everywhere would drag the fine one's resolution down across the
    exact region someone is zooming into.
    """
    import cv2

    if width_px <= 0 or not mask.any():
        return mask.astype(np.float32)
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    return np.clip(dist / float(width_px), 0.0, 1.0).astype(np.float32)


def orthophoto(surface: Surface, frame: SurfaceFrame, cloud: np.ndarray,
               cameras: list, images: dict, *, feather_px: int = 6) -> Orthophoto:
    """Bake the photographs onto the surface, coarsest first so the finest wins.

    ``cameras`` is a list of (frame_number, CameraNode) already sorted coarsest ground
    sample distance first. Painting in that order with last-write-wins means the widest
    frame lays a complete base and every sharper frame overwrites inside its own
    footprint, which is both the simplest rule and the one that keeps resolution.
    """
    rect = surface.rect
    uu, vv = rect.texel_centres()
    height, _ = _grid_heights(frame.to_plane(cloud), uu, vv)
    # Paint exactly the ground the mesh draws, no more and no less.
    paintable = surface.texel_mask().ravel()

    pts = frame.to_scene(np.column_stack([uu.ravel(), vv.ravel(), height.ravel()]))
    normals = _texel_normals(height, rect, frame)

    out = np.zeros((rect.height * rect.width, 3), dtype=np.float32)
    provenance = np.zeros(rect.height * rect.width, dtype=np.uint8)
    painted = np.zeros(rect.height * rect.width, dtype=bool)
    base_sample: np.ndarray | None = None
    base_mask: np.ndarray | None = None
    frames_used, per_frame = [], []

    for number, node in cameras:
        image = images.get(number)
        if image is None:
            continue
        ok = paintable.copy()
        ok &= visible_mask(node, pts, normals,
                           image_width=image.shape[1], image_height=image.shape[0])
        if not ok.any():
            continue
        ok &= occlusion_mask(node, pts)
        if ok.sum() < 64:
            continue

        scaled = _scaled_node(node, image.shape[1], image.shape[0])
        pixels, _, _ = project_points(scaled, pts[ok])
        sample = sample_image(image, pixels).astype(np.float32)

        if base_sample is None:
            base_sample, base_mask = np.zeros_like(out), ok.copy()
            base_sample[ok] = sample
        else:
            overlap = ok & base_mask
            if overlap.sum() >= 64:
                idx = overlap[ok]
                gain = _lab_gain_offset(sample[idx], base_sample[overlap])
                sample = _apply_lab_gain(sample, gain).astype(np.float32)

        alpha_full = _feather(ok.reshape(rect.height, rect.width), feather_px).ravel()
        alpha = alpha_full[ok]
        slot = len(frames_used) + 1
        out[ok] = out[ok] * (1.0 - alpha)[:, None] + sample * alpha[:, None]
        provenance[ok] = np.where(alpha > 0.5, slot, provenance[ok])
        painted |= ok & (alpha_full > 0.0)
        frames_used.append(number)
        per_frame.append({"frame": number, "texels": int(ok.sum()),
                          "share_of_surface": round(float(ok.sum() / max(1, paintable.sum())), 4)})

    covered = float((painted & paintable).sum() / max(1, paintable.sum()))
    holes = paintable & ~painted
    if holes.any():
        # Ground the mesh draws but no frame could see: occluded, or off the edge of
        # every footprint. Filling from the nearest painted texel keeps it from reading
        # as a black void; the provenance raster still records it as unattributed, so
        # nobody can mistake a filled hole for imagery.
        from scipy.ndimage import distance_transform_edt

        shape = (rect.height, rect.width)
        _, (rr, cc) = distance_transform_edt(
            ~painted.reshape(shape), return_indices=True)
        filled = out.reshape(*shape, 3)[rr, cc].reshape(-1, 3)
        out[holes] = filled[holes]

    return Orthophoto(
        rgb=np.clip(out, 0, 255).astype(np.uint8).reshape(rect.height, rect.width, 3),
        provenance=provenance.reshape(rect.height, rect.width),
        frames=frames_used, covered_fraction=covered, per_frame=per_frame,
    )


def _scaled_node(node, width: int, height: int):
    """The same camera against a resized copy of its image.

    Frames are re-encoded smaller for shipping, so the focal in pixels has to follow.
    Sampling a 1600 px JPEG with a focal measured at 1920 px puts every texel in the
    wrong place by a smoothly varying amount, which looks like a plausible photograph.
    """
    from dataclasses import replace

    if width == node.width and height == node.height:
        return node
    return replace(node, f_px=node.f_px * (width / node.width),
                   width=width, height=height)


def _texel_normals(height: np.ndarray, rect: TextureRect, frame: SurfaceFrame) -> np.ndarray:
    du = (rect.u1 - rect.u0) / rect.width
    dv = (rect.v1 - rect.v0) / rect.height
    d_dv, d_du = np.gradient(height, dv, du)
    normals = (-d_du[..., None] * frame.u - d_dv[..., None] * frame.v + frame.n)
    normals = normals.reshape(-1, 3)
    return normals / np.linalg.norm(normals, axis=1)[:, None]


def occlusion_mask(node, xyz: np.ndarray, *, tolerance_m: float = OCCLUSION_TOLERANCE_M,
                   bin_px: int = 4) -> np.ndarray:
    """Reject points hidden behind nearer surface along the same camera pixel.

    A depth buffer built with ``np.minimum.at`` rather than a GL render: bin every
    point into an integer pixel cell, keep the nearest range per cell, and drop anything
    sitting more than ``tolerance_m`` behind it. On a 42 degree face viewed 16 degrees
    off-normal this catches very little, which is the point of checking rather than
    assuming.
    """
    pixels, ranges, in_front = project_points(node, xyz)
    keep = in_front.copy()
    if not keep.any():
        return keep
    cols = (pixels[:, 0] / bin_px).astype(int)
    rows = (pixels[:, 1] / bin_px).astype(int)
    cols = np.clip(cols, 0, node.width // bin_px)
    rows = np.clip(rows, 0, node.height // bin_px)
    key = rows.astype(np.int64) * (node.width // bin_px + 1) + cols
    nearest = np.full(key.max() + 1, np.inf)
    np.minimum.at(nearest, key[keep], ranges[keep])
    keep &= ranges <= nearest[key] + tolerance_m
    return keep
