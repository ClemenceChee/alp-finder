"""Tests for the scene frame and the guards that keep a bad alignment out of it."""

from __future__ import annotations

import math

import numpy as np
import pytest

from image_lab.kurumdy import scene3d


@pytest.fixture
def origin() -> scene3d.SceneOrigin:
    return scene3d.default_origin()


class TestSceneOrigin:
    def test_origin_maps_to_zero(self, origin):
        assert np.allclose(origin.enu(origin.lat, origin.lon, origin.alt_ellipsoidal), 0.0)

    def test_round_trip_returns_the_original_coordinate(self, origin):
        lat, lon, alt = 39.483144, 73.585443, 4530.8
        back = origin.geodetic(origin.enu(lat, lon, alt))
        assert back[0] == pytest.approx(lat, abs=1e-9)
        assert back[1] == pytest.approx(lon, abs=1e-9)
        assert back[2] == pytest.approx(alt, abs=1e-6)

    def test_north_is_positive_y_and_east_is_positive_x(self, origin):
        north = origin.enu(origin.lat + 0.001, origin.lon, origin.alt_ellipsoidal)
        east = origin.enu(origin.lat, origin.lon + 0.001, origin.alt_ellipsoidal)
        assert north[1] > 0 and abs(north[0]) < 1e-9
        assert east[0] > 0 and abs(east[1]) < 1e-9

    def test_a_degree_of_longitude_is_shorter_than_a_degree_of_latitude(self, origin):
        # The whole reason the scene is metric: at 39.48 N the ratio is cos(lat),
        # so a mesh built in raw degrees is stretched east-west by 29 percent.
        assert origin.m_per_deg_lon / origin.m_per_deg_lat == pytest.approx(
            math.cos(math.radians(origin.lat)), rel=1e-12
        )

    def test_enu_many_agrees_with_enu(self, origin):
        pts = np.array([
            [39.482656, 73.586792, 4663.0],
            [39.483176, 73.585463, 4529.3],
            [39.483144, 73.585443, 4530.8],
        ])
        batch = origin.enu_many(pts)
        one_by_one = np.array([origin.enu(*p) for p in pts])
        assert np.allclose(batch, one_by_one)

    def test_enu_many_does_not_mutate_its_input(self, origin):
        pts = np.array([[39.4831, 73.5854, 4530.0]])
        before = pts.copy()
        origin.enu_many(pts)
        assert np.array_equal(pts, before)


class TestTelemetryDegeneracy:
    def test_well_spread_cameras_are_accepted(self):
        rng = np.random.default_rng(0)
        cams = rng.normal(scale=40.0, size=(50, 3))
        assert scene3d.telemetry_degeneracy(cams) is None

    def test_a_hover_is_rejected_for_spread(self):
        rng = np.random.default_rng(1)
        cams = rng.normal(scale=0.1, size=(22, 3))
        why = scene3d.telemetry_degeneracy(cams)
        assert why is not None and "span only" in why

    def test_collinear_cameras_are_rejected(self):
        # This is model 4: 164 images spread 97 m east, 17 m north, 0.1 m up.
        # It reported a 0.54 m residual while collapsing its cloud to a 1 m box.
        t = np.linspace(0, 97, 164)
        cams = np.column_stack([t, t * 0.18, np.full_like(t, 0.0)])
        why = scene3d.telemetry_degeneracy(cams)
        assert why is not None and "near-degenerate" in why

    def test_too_few_cameras_is_rejected(self):
        assert "at least 4" in scene3d.telemetry_degeneracy(np.zeros((3, 3)))


class TestAlignment:
    def _alignment(self, scale, rot, trans):
        return scene3d.Alignment(
            model="t", scale=scale, rotation=rot, translation=trans,
            n_cameras=10, n_inliers=10, residual_median_m=0.0, residual_p90_m=0.0,
        )

    def test_apply_matches_the_convention_used_to_fit(self):
        rng = np.random.default_rng(2)
        src = rng.normal(size=(20, 3))
        angle = 0.7
        rot = np.array([
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        trans = np.array([10.0, -20.0, 5.0])
        expected = 2.5 * (src @ rot.T) + trans
        assert np.allclose(self._alignment(2.5, rot, trans).apply(src), expected)

    def test_apply_accepts_a_single_point(self):
        out = self._alignment(1.0, np.eye(3), np.zeros(3)).apply(np.array([1.0, 2.0, 3.0]))
        assert out.shape == (1, 3)

    def test_inlier_fraction_is_zero_when_no_cameras(self):
        a = scene3d.Alignment("t", 1.0, np.eye(3), np.zeros(3), 0, 0, 0.0, 0.0)
        assert a.inlier_fraction == 0.0


class TestCameraNode:
    def _node(self):
        # A camera at 100 m up, looking due north and level.
        return scene3d.CameraNode(
            ident="t", source="telemetry", clip="c", frame=0, t_s=0.0,
            position=np.array([0.0, 0.0, 100.0]),
            right=np.array([1.0, 0.0, 0.0]),
            up=np.array([0.0, 0.0, 1.0]),
            forward=np.array([0.0, 1.0, 0.0]),
            f_px=7179.0, width=1920, height=1080,
        )

    def test_basis_matrix_is_right_handed(self):
        # The raw right/up/forward triple is left-handed by construction; the
        # renderable one must not be, or every photograph comes out mirrored.
        node = self._node()
        assert np.linalg.det(np.stack([node.right, node.up, node.forward])) < 0
        assert np.linalg.det(node.basis_matrix()) == pytest.approx(1.0)

    def test_basis_matrix_is_orthonormal(self):
        m = self._node().basis_matrix()
        assert np.allclose(m @ m.T, np.eye(3), atol=1e-12)

    def test_field_of_view_matches_the_pinhole_focal(self):
        node = self._node()
        assert node.hfov_deg == pytest.approx(
            math.degrees(2 * math.atan(1920 / (2 * 7179.0))), rel=1e-12
        )
        assert node.vfov_deg < node.hfov_deg

    def test_frame_index_is_parsed_from_the_asset_name(self):
        assert scene3d._frame_from_name("0813163855_0001_Z_002355.jpg") == 2355
        assert scene3d._frame_from_name("still_A.JPG") == -1


class TestCheckObjects:
    def test_measures_median_height_not_distance_to_nearest_point(self, origin):
        # A surface 5 m above the lid, sampled densely. Nearest-point distance
        # would report a fraction of a metre; the height error is what matters.
        lat, lon, alt = scene3d.OBJECTS["lid"]
        target = origin.enu(lat, lon, alt)
        rng = np.random.default_rng(3)
        offsets = rng.uniform(-2.0, 2.0, size=(400, 2))
        cloud = np.column_stack([
            target[0] + offsets[:, 0],
            target[1] + offsets[:, 1],
            np.full(400, target[2] + 5.0),
        ])
        result = {c.name: c for c in scene3d.check_objects(cloud, origin)}
        assert result["lid"].surface_error_m == pytest.approx(5.0, abs=1e-6)
        assert result["lid"].n_points_within > 100

    def test_reports_no_measurement_when_the_object_is_not_covered(self, origin):
        cloud = np.zeros((10, 3)) + 1000.0
        for check in scene3d.check_objects(cloud, origin):
            assert not check.measured
            assert check.surface_error_m is None

    def test_published_object_elevations_are_treated_as_ellipsoidal(self):
        # Guards the datum note in the module docstring: these are LRFTargetAbsAlt
        # values, derived from the drone's ellipsoidal altitude, and converting
        # them as if they were orthometric would move every object by 33 m.
        assert scene3d.OBJECTS["lid"][2] == pytest.approx(4530.8)
        assert scene3d.OBJECTS["pole"][2] == pytest.approx(4529.3)
        assert scene3d.OBJECTS["rucksack"][2] == pytest.approx(4663.0)
