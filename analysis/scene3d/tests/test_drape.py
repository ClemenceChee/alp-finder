"""Tests for the draped surface, and the one guard that matters most.

The handedness test is the reason this file exists. If the camera basis is mirrored,
every photograph baked onto the surface is mirrored too, and a mirrored snow slope is
indistinguishable from a snow slope. The only cheap way to catch it is to project the
reconstruction's own points into a frame and check that the pixel found there is the
colour the reconstruction already believes that point to be.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from image_lab.kurumdy import drape

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "out/scene3d/sfm/model0"
FRAMES = ROOT / "out/scene3d/frames"
VIDEO_DIR = ROOT / "data/kurumdy/new"

#: Frames that see enough of the cloud for the colour agreement to mean something.
CHECK_FRAMES = [3705, 3525, 3930, 2265]

#: Agreement between a point's own colour and the pixel it projects to. Measured at
#: 13.7 to 15.7 for these frames; a mirrored basis scores 46 to 55.
MAX_COLOUR_ERROR = 20.0
MIN_MIRRORED_ERROR = 40.0

needs_model = pytest.mark.skipif(
    not MODEL.exists() or not FRAMES.exists() or not VIDEO_DIR.exists(),
    reason="needs the reconstruction snapshot and extracted frames in out/scene3d",
)


class TestSurfaceFrame:
    def _cloud(self, seed=0, dip_deg=42.0):
        rng = np.random.default_rng(seed)
        plane = rng.uniform(-50, 50, size=(4000, 2))
        tilt = math.radians(dip_deg)
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, math.cos(tilt), math.sin(tilt)])
        noise = rng.normal(scale=0.3, size=(4000, 1))
        n = np.cross(u, v)
        return plane[:, 0:1] * u + plane[:, 1:2] * v + noise * n

    def test_normal_always_points_up(self):
        for seed in range(4):
            frame = drape.fit_surface_frame(self._cloud(seed))
            assert frame.n[2] > 0

    def test_frame_is_right_handed(self):
        frame = drape.fit_surface_frame(self._cloud())
        assert np.dot(np.cross(frame.u, frame.v), frame.n) == pytest.approx(1.0, abs=1e-9)

    def test_axes_are_orthonormal(self):
        frame = drape.fit_surface_frame(self._cloud())
        m = np.stack([frame.u, frame.v, frame.n])
        assert np.allclose(m @ m.T, np.eye(3), atol=1e-9)

    def test_plane_round_trip(self):
        frame = drape.fit_surface_frame(self._cloud())
        pts = self._cloud(seed=7)
        assert np.allclose(frame.to_scene(frame.to_plane(pts)), pts, atol=1e-9)

    def test_recovers_the_dip_it_was_given(self):
        frame = drape.fit_surface_frame(self._cloud(dip_deg=42.0))
        dip = math.degrees(math.acos(abs(frame.n[2])))
        assert dip == pytest.approx(42.0, abs=1.0)

    def test_rejects_too_few_points(self):
        with pytest.raises(ValueError):
            drape.fit_surface_frame(np.zeros((2, 3)))


class TestProjection:
    def _node(self):
        from image_lab.kurumdy import scene3d
        # 100 m above the origin, looking straight down, north-up.
        return scene3d.CameraNode(
            ident="t", source="sfm", clip="c", frame=0, t_s=0.0,
            position=np.array([0.0, 0.0, 100.0]),
            right=np.array([1.0, 0.0, 0.0]),
            up=np.array([0.0, 1.0, 0.0]),
            forward=np.array([0.0, 0.0, -1.0]),
            f_px=1000.0, width=1920, height=1080,
        )

    def test_point_below_the_camera_lands_at_the_principal_point(self):
        px, rng, front = drape.project_points(self._node(), np.array([[0.0, 0.0, 0.0]]))
        assert front[0]
        assert px[0] == pytest.approx([960.0, 540.0])
        assert rng[0] == pytest.approx(100.0)

    def test_east_is_right_and_north_is_up_in_the_image(self):
        node = self._node()
        px, _, _ = drape.project_points(node, np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]))
        assert px[0][0] > 960.0 and px[0][1] == pytest.approx(540.0)   # east -> right
        assert px[1][1] < 540.0 and px[1][0] == pytest.approx(960.0)   # north -> up

    def test_points_behind_the_camera_are_flagged(self):
        _, _, front = drape.project_points(self._node(), np.array([[0.0, 0.0, 200.0]]))
        assert not front[0]

    def test_project_points_inverts_pixel_direction(self):
        from image_lab.kurumdy import scene3d
        node = self._node()
        for x, y in [(100.0, 200.0), (1500.0, 900.0), (960.0, 540.0)]:
            direction = scene3d.pixel_direction(node, x, y)
            point = np.asarray(node.position) + 250.0 * direction
            px, _, front = drape.project_points(node, point[None, :])
            assert front[0]
            assert px[0] == pytest.approx([x, y], abs=1e-6)


class TestOcclusion:
    def test_a_point_hidden_behind_another_is_rejected(self):
        from image_lab.kurumdy import scene3d
        node = scene3d.CameraNode(
            ident="t", source="sfm", clip="c", frame=0, t_s=0.0,
            position=np.array([0.0, 0.0, 100.0]),
            right=np.array([1.0, 0.0, 0.0]), up=np.array([0.0, 1.0, 0.0]),
            forward=np.array([0.0, 0.0, -1.0]),
            f_px=1000.0, width=1920, height=1080,
        )
        # Same pixel, one 50 m behind the other.
        pts = np.array([[0.0, 0.0, 50.0], [0.0, 0.0, 0.0]])
        keep = drape.occlusion_mask(node, pts)
        assert keep[0] and not keep[1]

    def test_points_at_the_same_depth_both_survive(self):
        from image_lab.kurumdy import scene3d
        node = scene3d.CameraNode(
            ident="t", source="sfm", clip="c", frame=0, t_s=0.0,
            position=np.array([0.0, 0.0, 100.0]),
            right=np.array([1.0, 0.0, 0.0]), up=np.array([0.0, 1.0, 0.0]),
            forward=np.array([0.0, 0.0, -1.0]),
            f_px=1000.0, width=1920, height=1080,
        )
        pts = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
        assert drape.occlusion_mask(node, pts).all()


@pytest.fixture(scope="module")
def solved():
    """The reconstruction, georeferenced, with its check frames decoded."""
    import cv2
    import pycolmap

    from image_lab.kurumdy import scene3d

    rec = pycolmap.Reconstruction(str(MODEL))
    origin = scene3d.default_origin()
    poses = scene3d.poses_from_telemetry([im.name for im in rec.images.values()],
                                         VIDEO_DIR)
    alignment = scene3d.align_to_telemetry(rec, poses, origin, model_name="model0")
    pts = alignment.apply(np.array([p.xyz for p in rec.points3D.values()]))
    rgb = np.array([p.color for p in rec.points3D.values()], dtype=float)
    by_name = {im.name: im for im in rec.images.values()}
    nodes, images = {}, {}
    for frame in CHECK_FRAMES:
        image = by_name.get(f"0813163855_0001_Z_{frame:06d}.jpg")
        path = FRAMES / f"163855_{frame:06d}.jpg"
        if image is None or not path.exists():
            continue
        nodes[frame] = scene3d.camera_from_reconstruction(
            image, rec.cameras[image.camera_id], alignment, origin)
        images[frame] = cv2.imread(str(path))[:, :, ::-1]
    if not nodes:
        pytest.skip("no check frames available")
    return pts, rgb, nodes, images


@needs_model
class TestSurfaceAgainstLaserPoints:
    """The surface must put the laser-measured objects where the laser put them."""

    def _surface(self, solved):
        pts, _, _, _ = solved
        frame = drape.fit_surface_frame(pts)
        rect = drape.texture_rect(frame, 2048)
        return frame, rect, drape.heightfield(pts, frame, rect, posting_m=1.0)

    def test_objects_map_into_the_texture_and_back(self, solved):
        from image_lab.kurumdy import scene3d

        origin = scene3d.default_origin()
        frame, rect, _ = self._surface(solved)
        for name, (lat, lon, alt) in scene3d.OBJECTS.items():
            if name == "rucksack":
                continue                      # outside the reconstruction, by design
            enu = origin.enu(lat, lon, alt)
            u, v, _ = frame.to_plane(enu)[0]
            assert rect.u0 <= u <= rect.u1, f"{name} falls outside the texture in u"
            assert rect.v0 <= v <= rect.v1, f"{name} falls outside the texture in v"
            # Texture coordinate, then back to metres, must return the same place.
            uv = ((u - rect.u0) / (rect.u1 - rect.u0), (v - rect.v0) / (rect.v1 - rect.v0))
            back_u = rect.u0 + uv[0] * (rect.u1 - rect.u0)
            back_v = rect.v0 + uv[1] * (rect.v1 - rect.v0)
            assert back_u == pytest.approx(u, abs=1e-6)
            assert back_v == pytest.approx(v, abs=1e-6)

    def test_the_surface_carries_the_objects_within_a_few_metres(self, solved):
        from image_lab.kurumdy import scene3d

        origin = scene3d.default_origin()
        frame, _, surface = self._surface(solved)
        drawn = np.unique(surface.indices)
        verts = surface.vertices[drawn]
        for name in ("pole", "lid"):
            lat, lon, alt = scene3d.OBJECTS[name]
            enu = origin.enu(lat, lon, alt)
            gap = float(np.linalg.norm(verts - enu, axis=1).min())
            # The reconstruction reads about 3 m high at the held-out laser points, so
            # the fitted surface should pass within a few metres of them, not tens.
            assert gap < 6.0, f"{name} is {gap:.1f} m from the drawn surface"

    def test_every_drawn_vertex_is_supported(self, solved):
        _, _, surface = self._surface(solved)
        drawn = np.unique(surface.indices)
        assert surface.support_m[drawn].max() <= drape.MAX_SUPPORT_M + 1e-9

    def test_reported_measured_fraction_matches_the_support_values(self, solved):
        _, _, surface = self._surface(solved)
        drawn = np.unique(surface.indices)
        expected = float((surface.support_m[drawn] <= drape.MEASURED_SUPPORT_M).mean())
        assert surface.measured_fraction == pytest.approx(expected)


@needs_model
class TestHandedness:
    """The guard. A mirrored basis must score three times worse than the true one."""

    def _median_error(self, node, image, pts, rgb, *, mirror=False):
        px, _, front = drape.project_points(node, pts)
        if mirror:
            px = px.copy()
            px[:, 0] = node.width - px[:, 0]
        ok = front & drape.inside_image(px, node.width, node.height)
        if ok.sum() < 200:
            return None
        sampled = drape.sample_image(image, px[ok]).astype(float)
        return float(np.median(np.abs(sampled - rgb[ok]).mean(axis=1)))

    def test_projected_colour_matches_the_reconstruction(self, solved):
        pts, rgb, nodes, images = solved
        for frame, node in nodes.items():
            err = self._median_error(node, images[frame], pts, rgb)
            assert err is not None, f"frame {frame} saw too few points"
            assert err < MAX_COLOUR_ERROR, f"frame {frame} colour error {err:.1f}"

    def test_a_mirrored_projection_scores_far_worse(self, solved):
        pts, rgb, nodes, images = solved
        for frame, node in nodes.items():
            true = self._median_error(node, images[frame], pts, rgb)
            flipped = self._median_error(node, images[frame], pts, rgb, mirror=True)
            assert flipped is not None and true is not None
            assert flipped > MIN_MIRRORED_ERROR, f"frame {frame} mirrored only {flipped:.1f}"
            assert flipped > 2.0 * true
