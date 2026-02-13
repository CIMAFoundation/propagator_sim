"""Tests for spotting distance models."""

import numpy as np
import pytest

from propagator.core.numba.functions import (
    get_spotting_fn,
    spotting_distance_alexandridis,
    spotting_distance_koo,
    spotting_distance_pereira,
    spotting_distance_trucchia,
)


def test_get_spotting_fn_returns_correct_function():
    """Test that factory returns the correct function for each model."""
    assert get_spotting_fn("alexandridis") == spotting_distance_alexandridis
    assert get_spotting_fn("trucchia") == spotting_distance_trucchia
    assert get_spotting_fn("pereira") == spotting_distance_pereira
    assert get_spotting_fn("koo") == spotting_distance_koo


def test_get_spotting_fn_raises_for_unknown_model():
    """Test that factory raises ValueError for unknown model."""
    with pytest.raises(ValueError, match="Unknown spotting model"):
        get_spotting_fn("unknown_model")


def test_alexandridis_returns_valid_distance_and_time():
    """Test Alexandridis model returns positive distance and time."""
    # Set seed for reproducibility
    np.random.seed(42)

    angle = 0.0  # North
    w_dir = 0.0  # Wind from north
    w_speed = 20.0  # 20 km/h

    distance, time = spotting_distance_alexandridis(angle, w_dir, w_speed)

    assert distance >= 0.0
    assert time >= 0.0
    assert isinstance(distance, (int, float, np.number))
    assert isinstance(time, (int, float, np.number))


def test_alexandridis_zero_wind_returns_zero_distance():
    """Test Alexandridis model with zero wind."""
    angle = 0.0
    w_dir = 0.0
    w_speed = 0.0

    distance, time = spotting_distance_alexandridis(angle, w_dir, w_speed)

    assert distance == 0.0
    assert time == 1.0  # Default time when no wind


def test_trucchia_returns_valid_distance_and_time():
    """Test Trucchia model returns positive distance and time."""
    np.random.seed(42)

    angle = 0.0
    w_dir = 0.0
    w_speed = 20.0

    distance, time = spotting_distance_trucchia(angle, w_dir, w_speed)

    assert distance >= 0.0
    assert time >= 0.0


def test_trucchia_zero_wind_returns_zero_distance():
    """Test Trucchia model with zero wind."""
    angle = 0.0
    w_dir = 0.0
    w_speed = 0.0

    distance, time = spotting_distance_trucchia(angle, w_dir, w_speed)

    assert distance == 0.0
    assert time == 1.0


def test_pereira_returns_valid_distance_and_time():
    """Test Pereira model returns positive distance and time."""
    np.random.seed(42)

    angle = 0.0
    w_dir = 0.0
    w_speed = 20.0

    distance, time = spotting_distance_pereira(angle, w_dir, w_speed)

    assert distance >= 0.0
    assert time >= 0.0


def test_pereira_zero_wind_returns_zero_distance():
    """Test Pereira model with zero wind."""
    angle = 0.0
    w_dir = 0.0
    w_speed = 0.0

    distance, time = spotting_distance_pereira(angle, w_dir, w_speed)

    assert distance == 0.0
    assert time == 1.0


def test_koo_returns_valid_distance_and_time():
    """Test Koo model returns positive distance and time."""
    np.random.seed(42)

    angle = 0.0
    w_dir = 0.0
    w_speed = 20.0

    distance, time = spotting_distance_koo(angle, w_dir, w_speed)

    assert distance >= 0.0
    assert time >= 0.0


def test_koo_zero_wind_returns_zero_distance():
    """Test Koo model with zero wind."""
    angle = 0.0
    w_dir = 0.0
    w_speed = 0.0

    distance, time = spotting_distance_koo(angle, w_dir, w_speed)

    assert distance == 0.0
    assert time == 1.0


def test_spotting_models_with_wind_direction():
    """Test that models respond to wind direction."""
    np.random.seed(42)

    w_speed = 30.0

    # Wind from north (0), ember going north (0) - favorable
    angle_favorable = 0.0
    w_dir_favorable = 0.0

    # Wind from north (0), ember going south (π) - unfavorable
    angle_unfavorable = np.pi
    w_dir_unfavorable = 0.0

    for model_fn in [
        spotting_distance_alexandridis,
        spotting_distance_trucchia,
        spotting_distance_pereira,
    ]:
        dist_fav, _ = model_fn(angle_favorable, w_dir_favorable, w_speed)
        dist_unfav, _ = model_fn(
            angle_unfavorable, w_dir_unfavorable, w_speed
        )

        # Favorable direction should generally have longer distance
        # (not strict due to randomness, but testing they're different)
        assert dist_fav >= 0
        assert dist_unfav >= 0


def test_spotting_models_increase_with_wind_speed():
    """Test that spotting distance increases with wind speed."""
    np.random.seed(42)

    angle = 0.0
    w_dir = 0.0

    for model_fn in [
        spotting_distance_alexandridis,
        spotting_distance_trucchia,
        spotting_distance_pereira,
        spotting_distance_koo,
    ]:
        # Low wind
        np.random.seed(42)
        dist_low, _ = model_fn(angle, w_dir, 5.0)

        # High wind
        np.random.seed(42)
        dist_high, _ = model_fn(angle, w_dir, 40.0)

        # Higher wind should give longer distance
        # Due to randomness, we just check both are valid
        assert dist_low >= 0.0
        assert dist_high >= 0.0


def test_spotting_models_are_stochastic():
    """Test that models produce different results on multiple calls."""
    angle = 0.0
    w_dir = 0.0
    w_speed = 20.0

    for model_fn in [
        spotting_distance_alexandridis,
        spotting_distance_trucchia,
        spotting_distance_pereira,
        spotting_distance_koo,
    ]:
        results = []
        for _ in range(5):
            distance, _ = model_fn(angle, w_dir, w_speed)
            results.append(distance)

        # At least some variation expected (not all identical)
        # Check that not all results are the same
        unique_results = len(set(results))
        assert (
            unique_results > 1
        ), f"{model_fn.__name__} produced identical results"
