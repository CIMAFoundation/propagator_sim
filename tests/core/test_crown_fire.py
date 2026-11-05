"""Crown fire module: off by default, stochastic initiation, selectable
models, and numba/Rust parity."""

from __future__ import annotations

import numpy as np
import pytest

from propagator.core import BoundaryConditions, Propagator  # type: ignore
from propagator.core.constants import (  # type: ignore
    FUEL_SYSTEM_LEGACY_DICT,
)
from propagator.core.numba import (  # type: ignore
    active_crowning_alexander,
    get_active_crowning_fn,
    get_p_crowning_initiation_fn,
    p_crowning_initiation_cruz,
    p_crowning_initiation_perrakis,
)

SIZE = 48
CONIFERS = 5
GRASSLAND = 4
WIND_SPEED = 10.0  # km/h
MOISTURE = 6.0  # %
# fuel strata gaps giving a crowning probability close to 0.5 at the
# conditions above (conifers: surface fuel consumption d0 = 1 kg/m2)
FSG_HALF = {"cruz": 1.7, "perrakis": 5.4}
CBD_ACTIVE = 0.3  # active crown fire at the conditions above
CBD_PASSIVE = 0.1  # passive crown fire at the conditions above


def canopy(fsg: float, cbd: float):
    return (
        np.full((SIZE, SIZE), fsg, dtype=np.float32),
        np.full((SIZE, SIZE), cbd, dtype=np.float32),
    )


def make_simulator(
    core: str,
    *,
    realizations: int = 8,
    seed: int = 7,
    veg: np.ndarray | None = None,
    model: str = "cruz",
    **kwargs,
):
    if veg is None:
        veg = np.full((SIZE, SIZE), CONIFERS, dtype=np.int32)
    dem = np.zeros((SIZE, SIZE), dtype=np.float32)
    common = dict(
        veg=veg,
        dem=dem,
        realizations=realizations,
        do_spotting=False,
        seed=seed,
        out_of_bounds_mode="ignore",
    )
    if core == "rust":
        pytest.importorskip("propagator_rust")
        from propagator.rust_core import Propagator as RustPropagator

        simulator = RustPropagator(
            fuels_dict=FUEL_SYSTEM_LEGACY_DICT,
            crowning_initiation_model=model,
            **common,
            **kwargs,
        )
    else:
        simulator = Propagator(
            p_crowning_init_fn=get_p_crowning_initiation_fn(model),
            **common,
            **kwargs,
        )
    simulator.set_boundary_conditions(
        BoundaryConditions(
            time=0,
            moisture=MOISTURE,
            wind_dir=0.0,
            wind_speed=WIND_SPEED,
            ignitions=[(SIZE // 2, SIZE // 2)],
        )
    )
    return simulator


def run(simulator, seconds: int = 3 * 3600):
    simulator.step(seconds=seconds)
    return simulator.get_output(), simulator.get_fire_status()


def crown_fraction(status: np.ndarray) -> tuple[float, int]:
    """Fraction of front-reached cells that crowned (ignition excluded)."""
    status = status.copy()
    status[:, SIZE // 2, SIZE // 2] = 0
    burnt = int((status > 0).sum())
    return float((status >= 2).sum()) / burnt, burnt


# --- model functions -------------------------------------------------------


def test_cruz_matches_logistic_model():
    g = 4.236 + 0.357 * 20 - 0.710 * 2 - 4.613 - 0.331 * 6
    expected = 1 / (1 + np.exp(-g))
    assert p_crowning_initiation_cruz(20.0, 2.0, 1.0, 0.06) == pytest.approx(
        expected
    )
    # surface fuel consumption classes
    g_mid = 4.236 + 0.357 * 20 - 0.710 * 2 - 1.856 - 0.331 * 6
    assert p_crowning_initiation_cruz(20.0, 2.0, 1.5, 0.06) == pytest.approx(
        1 / (1 + np.exp(-g_mid))
    )


def test_perrakis_matches_logistic_model():
    g = (
        -3.5550
        + 1.4407 * 10
        - 0.53211 * 2.0**1.5
        - 0.07019 * 10 * 6
        + 2.4897 * np.log(1.5)
    )
    assert p_crowning_initiation_perrakis(
        10.0, 2.0, 1.5, 0.06
    ) == pytest.approx(1 / (1 + np.exp(-g)))
    assert p_crowning_initiation_perrakis(10.0, 2.0, 0.0, 0.06) == 0.0


def test_logistic_models_do_not_overflow():
    assert p_crowning_initiation_cruz(1e4, 0.0, 3.0, 0.0) == 1.0
    assert p_crowning_initiation_cruz(0.0, 1e4, 0.5, 1.0) == 0.0


def test_active_crowning_criterion():
    assert active_crowning_alexander(WIND_SPEED, CBD_ACTIVE, 0.06)
    assert not active_crowning_alexander(WIND_SPEED, CBD_PASSIVE, 0.06)


@pytest.mark.parametrize("model", ["cruz", "perrakis"])
def test_initiation_models_are_selectable(model):
    fn = get_p_crowning_initiation_fn(model)
    p = fn(WIND_SPEED, FSG_HALF[model], 1.0, MOISTURE / 100)
    assert 0.4 < p < 0.6
    assert get_active_crowning_fn("alexander") is active_crowning_alexander


def test_unknown_models_raise():
    with pytest.raises(ValueError):
        get_p_crowning_initiation_fn("nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        get_active_crowning_fn("nope")  # type: ignore[arg-type]


# --- crowning disabled: baseline unchanged -------------------------------


@pytest.mark.parametrize("core", ["numba", "rust"])
def test_crowning_disabled_leaves_outputs_unchanged(core):
    fsg, cbd = canopy(FSG_HALF["cruz"], CBD_ACTIVE)
    base_out, base_status = run(make_simulator(core))
    # canopy given but module off; module on without any canopy cell
    # (numba RNGs are per thread, not per simulator: build each run's
    # simulator right before running it)
    variants = [
        dict(fsg=fsg, cbd=cbd),
        dict(do_crowning=True, fsg=fsg, cbd=cbd * 0),
    ]
    for kwargs in variants:
        out, status = run(make_simulator(core, **kwargs))
        np.testing.assert_array_equal(status, base_status)
        np.testing.assert_array_equal(
            out.fire_probability, base_out.fire_probability
        )
        np.testing.assert_array_equal(out.fli_max, base_out.fli_max)
        np.testing.assert_array_equal(
            out.min_arrival_time, base_out.min_arrival_time
        )
        assert not out.crowning_probability.any()
    assert set(np.unique(base_status)) <= {0, 1}


def test_do_crowning_requires_canopy():
    veg = np.full((SIZE, SIZE), CONIFERS, dtype=np.int32)
    dem = np.zeros((SIZE, SIZE), dtype=np.float32)
    with pytest.raises(ValueError):
        Propagator(veg=veg, dem=dem, do_crowning=True)
    with pytest.raises(ValueError):
        Propagator(veg=veg, dem=dem, fsg=np.zeros((SIZE, SIZE)))


# --- crowning enabled ----------------------------------------------------


@pytest.mark.parametrize("core", ["numba", "rust"])
def test_crowning_only_in_prone_fuels_with_canopy(core):
    veg = np.full((SIZE, SIZE), CONIFERS, dtype=np.int32)
    veg[:, : SIZE // 2] = GRASSLAND  # not crowning-prone
    fsg, cbd = canopy(0.0, CBD_ACTIVE)
    cbd[: SIZE // 4] = np.nan  # no canopy (nodata)
    out, status = run(
        make_simulator(core, veg=veg, do_crowning=True, fsg=fsg, cbd=cbd)
    )
    crowned = out.crowning_probability > 0
    assert crowned.any()
    assert not crowned[:, : SIZE // 2].any()
    assert not crowned[: SIZE // 4].any()
    assert np.all(out.crowning_probability <= out.fire_probability)
    assert np.all(status[status >= 2] <= 3)


@pytest.mark.parametrize("core", ["numba", "rust"])
def test_active_or_passive_follows_canopy_bulk_density(core):
    fsg, cbd = canopy(0.0, CBD_ACTIVE)
    cbd[:, SIZE // 2 :] = CBD_PASSIVE
    out, status = run(make_simulator(core, do_crowning=True, fsg=fsg, cbd=cbd))
    left = status[:, :, : SIZE // 2]
    right = status[:, :, SIZE // 2 :]
    assert (left == 3).any() and not (left == 2).any()
    assert (right == 2).any() and not (right == 3).any()
    np.testing.assert_array_equal(
        out.active_crowning_probability[:, SIZE // 2 :], 0
    )


@pytest.mark.parametrize("core", ["numba", "rust"])
def test_crowning_is_reproducible_with_a_fixed_seed(core):
    fsg, cbd = canopy(FSG_HALF["cruz"], CBD_ACTIVE)
    _, first = run(make_simulator(core, do_crowning=True, fsg=fsg, cbd=cbd))
    _, second = run(make_simulator(core, do_crowning=True, fsg=fsg, cbd=cbd))
    np.testing.assert_array_equal(first, second)
    assert (first == 1).any() and (first == 3).any()


# --- numba / Rust parity -------------------------------------------------


@pytest.mark.parametrize("model", ["cruz", "perrakis"])
def test_crown_fraction_parity_between_cores(model):
    """The cores use independent RNG streams, so (as for spotting) parity is
    statistical: the fraction of front-reached cells that crown must match
    the initiation probability on both cores, within Monte-Carlo noise."""
    fsg, cbd = canopy(FSG_HALF[model], CBD_ACTIVE)
    p = get_p_crowning_initiation_fn(model)(
        WIND_SPEED, FSG_HALF[model], 1.0, MOISTURE / 100
    )
    fractions = {}
    for core in ("numba", "rust"):
        simulator = make_simulator(
            core,
            realizations=16,
            model=model,
            do_crowning=True,
            fsg=fsg,
            cbd=cbd,
        )
        _, status = run(simulator, seconds=4 * 3600)
        fraction, n = crown_fraction(status)
        assert n > 2000
        # 5 binomial standard deviations
        assert abs(fraction - p) < 5 * np.sqrt(p * (1 - p) / n)
        fractions[core] = fraction
    assert abs(fractions["numba"] - fractions["rust"]) < 0.05


@pytest.mark.parametrize("core", ["numba", "rust"])
def test_crowning_probability_output_matches_status(core):
    fsg, cbd = canopy(FSG_HALF["cruz"], CBD_ACTIVE)
    simulator = make_simulator(core, do_crowning=True, fsg=fsg, cbd=cbd)
    out, status = run(simulator)
    np.testing.assert_allclose(
        out.crowning_probability, (status >= 2).mean(axis=0), rtol=1e-6
    )
    np.testing.assert_allclose(
        out.active_crowning_probability,
        (status == 3).mean(axis=0),
        rtol=1e-6,
    )


# --- state handling (numba core) -----------------------------------------


def test_crowning_survives_freeze_and_checkpoint(tmp_path):
    fsg, cbd = canopy(FSG_HALF["cruz"], CBD_ACTIVE)
    # numba RNGs are per thread: run the two simulators one after the other
    frozen = make_simulator(
        "numba",
        do_crowning=True,
        fsg=fsg,
        cbd=cbd,
        freeze_dir=tmp_path / "tiles",
    )
    frozen_tiles = 0
    for _ in range(8):
        frozen.step(seconds=3600)
        frozen_tiles += frozen.freeze_inactive_tiles()
    assert frozen_tiles > 0
    frozen_out = frozen.get_output()
    reference = make_simulator("numba", do_crowning=True, fsg=fsg, cbd=cbd)
    for _ in range(8):
        reference.step(seconds=3600)
    assert frozen_out.crowning_probability.any()
    np.testing.assert_array_equal(
        reference.get_output().crowning_probability,
        frozen_out.crowning_probability,
    )
    np.testing.assert_array_equal(
        reference.get_fire_status(), frozen.get_fire_status()
    )

    checkpoint = reference.checkpoint()
    path = tmp_path / "ckpt"
    checkpoint.save(path)
    loaded = type(checkpoint).load(path)
    assert loaded.do_crowning
    np.testing.assert_array_equal(loaded.cbd, reference.cbd)
    resumed = Propagator.from_checkpoint(
        loaded, seed=3, out_of_bounds_mode="ignore"
    )
    assert resumed.do_crowning
    np.testing.assert_array_equal(
        resumed.get_fire_status(), reference.get_fire_status()
    )
    resumed.step(seconds=600)

    no_crowning = make_simulator("numba")
    with pytest.raises(ValueError, match="do_crowning"):
        no_crowning.restore(checkpoint)
