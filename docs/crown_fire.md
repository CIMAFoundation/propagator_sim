# Crown Fire

The crown fire module flags, for every cell a surface fire reaches, whether
the fire also moves into the canopy and, if it does, whether the crown fire
is **passive** (torching) or **active** (continuous crown fire). It is off
by default. When it is off, both cores give the same results as before,
bit for bit.

The module only adds information about the fire. Spread, rate of spread
and fireline intensity are the same with or without crowning, so the new
outputs are the crowning maps.

## Inputs

Crowning needs two canopy rasters, aligned with the DEM, plus a flag on
the fuel types that can crown:

| Input | Units | Meaning |
| --- | --- | --- |
| `fsg` | m | Fuel strata gap: the vertical distance from the surface fuel to the canopy base. Cruz et al. (2004) use this instead of canopy base height. |
| `cbd` | kg/m³ | Canopy bulk density. |
| `crowning` (fuel option) | bool | The fuel type can crown. In the legacy fuel table only conifers (`5`) have it. |

A cell with a non-finite value, a negative `fsg` or a `cbd` ≤ 0 counts as
having **no canopy** and never crowns, so nodata needs no special handling.
The cell's surface fuel load `d0` (kg/m²) stands in for surface fuel
consumption.

## Model

Each time a spreading surface front ignites a cell whose fuel can crown
and that has a canopy, the core does the following:

1. **Initiation (stochastic).** It computes the crowning probability `p`
   from the cell's 10-m wind speed `U` (km/h), `fsg`, surface fuel
   consumption `SFC` (kg/m²) and fine dead fuel moisture `M` (%). It then
   draws one uniform number per realization, and the cell crowns when
   `p > u`.
2. **Active or passive (deterministic).** A crowning cell is an *active*
   crown fire when the active crown rate of spread `R_a` reaches Van
   Wagner's critical rate `3 / CBD` (m/min). Otherwise it is *passive*:

    ```text
    R_a = 11.02 · U^0.9 · CBD^0.9 · exp(-0.17 · M)        (m/min)
    active  ⇔  R_a / (3 / CBD) ≥ 1
    ```

Cells ignited directly (boundary-condition ignitions) or by embers start
as surface fires. Only arrival by spread can make a cell crown, as in the
original module.

### Initiation models

Both models are logistic, `p = 1 / (1 + exp(-g))`:

- **`cruz`** (default), Cruz et al. (2004):

    ```text
    g = 4.236 + 0.357·U − 0.710·FSG − 0.331·M + b_SFC
    b_SFC = −4.613 if SFC ≤ 1, −1.856 if 1 < SFC ≤ 2, 0 otherwise
    ```

- **`perrakis`**, Perrakis et al. (2023), model 11:

    ```text
    g = −3.5550 + 1.4407·U − 0.53211·FSG^1.5 − 0.07019·U·M + 2.4897·ln(SFC)
    ```

    Here `p = 0` when there is no surface fuel.

The active criterion has one model, **`alexander`**.

## Usage

Python, numba core:

```python
from propagator.core import Propagator
from propagator.core.numba import get_p_crowning_initiation_fn

simulator = Propagator(
    veg=veg, dem=dem,
    do_crowning=True, fsg=fsg, cbd=cbd,
    p_crowning_init_fn=get_p_crowning_initiation_fn("perrakis"),
)
```

On the Rust core (`propagator.rust_core.Propagator`) you pass the same
`do_crowning`, `fsg` and `cbd` arguments, and select models by name with
`crowning_initiation_model` and `active_crowning_model`.

CLI (`geotiff` mode): set the options in the configuration file and pass
the rasters on the command line:

```json
{"do_crowning": true, "crowning_initiation_model": "cruz",
 "active_crowning_model": "alexander"}
```

```bash
propagator --config example/crowning/config_crowning.json --mode geotiff \
    --dem dem.tif --fuel fuel.tif --fsg fsg.tif --cbd cbd.tif --output out/
```

To make a fuel type in a custom fuel table crowning-prone, add
`crowning: true` to it. Runnable examples are in `example/crowning/`.

## Outputs

| Field | Description |
| --- | --- |
| `crowning_probability` | Fraction of realizations where the cell burned as a crown fire, passive or active. |
| `active_crowning_probability` | Fraction of realizations where the cell burned as an active crown fire. |

Both fields are on `PropagatorOutput`. The CLI writes them as rasters when
`do_crowning` is set. `Propagator.get_fire_status()` returns the
per-realization status for each cell: `0` unburnt, `1` surface fire, `2`
passive crown fire, `3` active crown fire.

## Implementation notes

- The crown status is stored in two bits of the per-cell tile flags
  (`FLAG_CROWN`, `FLAG_CROWN_ACTIVE`). It therefore carries over through
  tile freezing, checkpoints (`do_crowning`, `fsg` and `cbd` are saved
  too) and domain growth. Cells added by growth have no canopy unless you
  pass new canopy rasters.
- No random number is drawn for a cell that cannot crown. With the module
  off, or with no canopy anywhere, both cores give the same results as
  before, bit for bit.
- The numba and Rust cores use the same formulas. They draw from separate
  RNG streams, so, as with spotting, they agree statistically rather than
  bit for bit (see [Rust vs numba Core](rust-vs-numba.md)).

## References

- M. G. Cruz, M. E. Alexander and R. H. Wakimoto. "Modeling the likelihood
  of crown fire occurrence in conifer forest stands." *Forest Science*
  50(5):640–658, 2004.
- D. D. B. Perrakis et al. "Improved logistic models of crown fire
  probability in Canadian conifer forests." *International Journal of
  Wildland Fire*, 2023.
- M. E. Alexander and M. G. Cruz. "Evaluating a model for predicting active
  crown fire rate of spread using wildfire observations." *Canadian
  Journal of Forest Research* 36(11):3015–3028, 2006.
- C. E. Van Wagner. "Conditions for the start and spread of crown fire."
  *Canadian Journal of Forest Research* 7(1):23–34, 1977.
