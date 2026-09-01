import numpy as np
import pytest
from affine import Affine

from propagator.io.writer.isochrones_geojson import extract_isochrone

propagator_rust = pytest.importorskip("propagator_rust")


def _block(dtype=np.float64):
    return np.pad(np.ones((3, 3), dtype=dtype), 2)


def test_extract_isochrone_matches_legacy_reference_coordinates():
    result = extract_isochrone(
        _block(), Affine.identity(), thresholds=[0.5, 1.1]
    )

    assert result.keys() == {0.5}
    assert len(result[0.5].geoms) == 1
    line = np.asarray(result[0.5].geoms[0].coords)
    assert line.shape == (13, 2)
    np.testing.assert_allclose(
        line[0],
        [3.2734535749019926, 2.02279180090523],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_array_equal(line[0], line[-1])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_python_wrapper_matches_native_binding(dtype):
    values = _block(dtype)
    transform = Affine(20, 3, 500_000, -2, -20, 4_900_000)
    kwargs = {
        "thresholds": [0.25, 0.5, 1.1],
        "med_filt_val": 3,
        "min_length": 0.001,
        "smooth_sigma": 1.25,
        "simp_fact": 123.0,
    }
    wrapped = extract_isochrone(values, transform, **kwargs)
    native = propagator_rust.extract_isochrone(
        values,
        (
            transform.a,
            transform.b,
            transform.c,
            transform.d,
            transform.e,
            transform.f,
        ),
        **kwargs,
    )

    assert wrapped.keys() == native.keys()
    for threshold, lines in native.items():
        assert len(wrapped[threshold].geoms) == len(lines)
        for geometry, coordinates in zip(wrapped[threshold].geoms, lines):
            np.testing.assert_allclose(
                np.asarray(geometry.coords), coordinates, rtol=0, atol=0
            )


def test_affine_transform_supports_rotation_and_shear():
    identity = extract_isochrone(
        _block(), Affine.identity(), thresholds=[0.5], min_length=0
    )[0.5]
    transform = Affine(20, 3, 500_000, -2, -20, 4_900_000)
    transformed = extract_isochrone(
        _block(), transform, thresholds=[0.5], min_length=0
    )[0.5]

    expected = np.asarray(
        [transform * tuple(xy) for xy in identity.geoms[0].coords]
    )
    np.testing.assert_allclose(
        np.asarray(transformed.geoms[0].coords),
        expected,
        rtol=1e-12,
        atol=1e-9,
    )


def test_threshold_can_survive_with_empty_boundary_geometry():
    result = extract_isochrone(
        np.ones((7, 7)), Affine.identity(), thresholds=[0.5]
    )
    assert result.keys() == {0.5}
    assert result[0.5].is_empty


def test_native_binding_preserves_defaults_and_validates_shape():
    result = propagator_rust.extract_isochrone(
        _block(), (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    )
    assert result.keys() == {0.5, 0.75, 0.9}

    with pytest.raises(ValueError, match="2-D"):
        propagator_rust.extract_isochrone(
            _block().ravel(), (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        )
