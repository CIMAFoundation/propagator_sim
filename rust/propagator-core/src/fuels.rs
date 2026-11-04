//! Fuel system: dense per-fuel parameter table plus the id -> index map,
//! and the legacy 7-fuel default table.

use std::collections::HashMap;

use crate::error::{PropagatorError, Result};
use crate::grid::Grid2;

/// Reserved vegetation code: not combustible, blocks spread.
pub const NO_FUEL: i32 = 0;

/// Sentinel for "no live fuel moisture defined" (matches the Python core).
pub const HUMIDITY_UNSET: f64 = -9999.0;

/// Definition of one fuel type, in *config units* (v0 in m/h, humidity in
/// percent), mirroring the Python configuration dictionaries.
#[derive(Clone, Debug)]
pub struct FuelDef {
    pub id: i32,
    pub name: String,
    /// nominal rate of spread, m/h (converted to m/min internally)
    pub v0: f64,
    /// dead fuel load, kg/m^2
    pub d0: f64,
    /// live (canopy) fuel load, kg/m^2
    pub d1: f64,
    /// higher heating value, kJ/kg
    pub hhv: f64,
    /// live fuel moisture, percent; `None` when the fuel has no live load
    pub humidity: Option<f64>,
    pub spotting: bool,
    pub prob_ign_by_embers: f64,
    pub burn: bool,
    /// base spread probability toward other fuel ids
    pub spread_probability: Vec<(i32, f64)>,
}

/// Dense fuel table used by the hot kernel. All per-fuel quantities are in
/// internal units (v0 m/min, humidity fraction or `HUMIDITY_UNSET`).
#[derive(Clone, Debug)]
pub struct FuelSystem {
    ids: HashMap<i32, usize>,
    names: Vec<String>,
    pub(crate) v0: Vec<f64>,
    pub(crate) d0: Vec<f64>,
    pub(crate) d1: Vec<f64>,
    pub(crate) hhv: Vec<f64>,
    pub(crate) humidity: Vec<f64>,
    pub(crate) spotting: Vec<bool>,
    pub(crate) prob_ign_by_embers: Vec<f64>,
    pub(crate) burn: Vec<bool>,
    /// row-major (n, n): spread_probability[from * n + to]
    spread_probability: Vec<f64>,
    non_vegetated: Option<i32>,
}

impl FuelSystem {
    /// Build the dense table from per-fuel definitions, applying the
    /// config->internal unit conversions (v0 m/h -> m/min, humidity percent
    /// -> fraction). Errors on duplicate ids, spread-probability references
    /// to unknown ids, or a live fuel load (`d1 != 0`) without a humidity.
    /// The last non-burnable fuel becomes the unknown-code fallback.
    pub fn from_defs(defs: &[FuelDef]) -> Result<FuelSystem> {
        let n = defs.len();
        let mut system = FuelSystem {
            ids: HashMap::new(),
            names: Vec::with_capacity(n),
            v0: Vec::with_capacity(n),
            d0: Vec::with_capacity(n),
            d1: Vec::with_capacity(n),
            hhv: Vec::with_capacity(n),
            humidity: Vec::with_capacity(n),
            spotting: Vec::with_capacity(n),
            prob_ign_by_embers: Vec::with_capacity(n),
            burn: Vec::with_capacity(n),
            spread_probability: vec![0.0; n * n],
            non_vegetated: None,
        };
        for def in defs {
            if system.ids.contains_key(&def.id) {
                return Err(PropagatorError::FuelSystem(format!(
                    "fuel id {} already exists",
                    def.id
                )));
            }
            let humidity = match def.humidity {
                // percent -> fraction
                Some(h) => h / 100.0,
                None => {
                    if def.d1 != 0.0 {
                        return Err(PropagatorError::FuelSystem(format!(
                            "inconsistent fuel {}: d1 != 0 without humidity",
                            def.id
                        )));
                    }
                    HUMIDITY_UNSET
                }
            };
            let index = system.ids.len();
            system.ids.insert(def.id, index);
            system.names.push(def.name.clone());
            system.v0.push(def.v0 / 60.0); // m/h -> m/min
            system.d0.push(def.d0);
            system.d1.push(def.d1);
            system.hhv.push(def.hhv);
            system.humidity.push(humidity);
            system.spotting.push(def.spotting);
            system.prob_ign_by_embers.push(def.prob_ign_by_embers);
            system.burn.push(def.burn);
            if !def.burn {
                system.non_vegetated = Some(def.id);
            }
        }
        for def in defs {
            for &(to_id, prob) in &def.spread_probability {
                let from = system.ids[&def.id];
                let to = *system.ids.get(&to_id).ok_or_else(|| {
                    PropagatorError::FuelSystem(format!(
                        "spread probability references unknown fuel id {to_id}"
                    ))
                })?;
                system.spread_probability[from * n + to] = prob;
            }
        }
        Ok(system)
    }

    /// Number of fuels in the table.
    pub fn n_fuels(&self) -> usize {
        self.v0.len()
    }

    /// Human-readable name of the fuel at a dense index.
    pub fn name(&self, index: usize) -> &str {
        &self.names[index]
    }

    /// Id of the non-vegetated fallback fuel, if one is defined.
    pub fn non_vegetated(&self) -> Option<i32> {
        self.non_vegetated
    }

    /// Base spread probability from one fuel to another (both dense indices).
    #[inline]
    pub fn transition_probability(&self, from_index: usize, to_index: usize) -> f64 {
        self.spread_probability[from_index * self.v0.len() + to_index]
    }

    /// Clear spotting for every fuel (used when `do_spotting == false`).
    pub fn disable_spotting(&mut self) {
        for flag in &mut self.spotting {
            *flag = false;
        }
        for prob in &mut self.prob_ign_by_embers {
            *prob = 0.0;
        }
    }

    /// Map vegetation codes to dense fuel indices once, so the kernel
    /// indexes arrays directly. Unknown codes fall back to the
    /// non-vegetated fuel; error if there is none and unknown codes exist.
    pub fn build_fuel_index_grid(&self, veg: &Grid2<i32>) -> Result<Grid2<i32>> {
        let default_index = self
            .non_vegetated
            .map(|id| self.ids[&id] as i32)
            .unwrap_or(-1);
        let mut fuel_idx = Grid2::filled(veg.rows(), veg.cols(), default_index);
        for (out, code) in fuel_idx
            .as_mut_slice()
            .iter_mut()
            .zip(veg.as_slice().iter())
        {
            if let Some(&index) = self.ids.get(code) {
                *out = index as i32;
            } else if default_index == -1 {
                return Err(PropagatorError::FuelSystem(
                    "vegetation map contains codes not present in the fuel \
                     system and no non-vegetated (burn=false) fallback fuel \
                     is defined"
                        .into(),
                ));
            }
        }
        Ok(fuel_idx)
    }
}

/// The legacy 7-fuel table (`FUEL_SYSTEM_LEGACY` in the Python core).
pub fn legacy_fuel_defs() -> Vec<FuelDef> {
    fn probs(values: [(i32, f64); 7]) -> Vec<(i32, f64)> {
        values.to_vec()
    }
    vec![
        FuelDef {
            id: 1,
            name: "broadleaves".into(),
            v0: 140.0,
            d0: 1.5,
            d1: 3.0,
            hhv: 20000.0,
            humidity: Some(60.0),
            spotting: false,
            prob_ign_by_embers: 0.0,
            burn: true,
            spread_probability: probs([
                (1, 0.3),
                (2, 0.375),
                (3, 0.005),
                (4, 0.45),
                (5, 0.225),
                (6, 0.25),
                (7, 0.075),
            ]),
        },
        FuelDef {
            id: 2,
            name: "shrubs".into(),
            v0: 140.0,
            d0: 1.0,
            d1: 3.0,
            hhv: 21000.0,
            humidity: Some(45.0),
            spotting: false,
            prob_ign_by_embers: 0.0,
            burn: true,
            spread_probability: probs([
                (1, 0.375),
                (2, 0.375),
                (3, 0.005),
                (4, 0.475),
                (5, 0.325),
                (6, 0.25),
                (7, 0.1),
            ]),
        },
        FuelDef {
            id: 3,
            name: "non-vegetated".into(),
            v0: 20.0,
            d0: 0.1,
            d1: 0.0,
            hhv: 100.0,
            humidity: None,
            spotting: false,
            prob_ign_by_embers: 0.0,
            burn: false,
            spread_probability: probs([
                (1, 0.005),
                (2, 0.005),
                (3, 0.005),
                (4, 0.005),
                (5, 0.005),
                (6, 0.005),
                (7, 0.005),
            ]),
        },
        FuelDef {
            id: 4,
            name: "grassland".into(),
            v0: 120.0,
            d0: 0.5,
            d1: 0.0,
            hhv: 17000.0,
            humidity: None,
            spotting: false,
            prob_ign_by_embers: 0.0,
            burn: true,
            spread_probability: probs([
                (1, 0.25),
                (2, 0.35),
                (3, 0.005),
                (4, 0.475),
                (5, 0.1),
                (6, 0.3),
                (7, 0.075),
            ]),
        },
        FuelDef {
            id: 5,
            name: "conifers".into(),
            v0: 200.0,
            d0: 1.0,
            d1: 4.0,
            hhv: 21000.0,
            humidity: Some(55.0),
            spotting: true,
            prob_ign_by_embers: 0.4,
            burn: true,
            spread_probability: probs([
                (1, 0.275),
                (2, 0.4),
                (3, 0.005),
                (4, 0.475),
                (5, 0.35),
                (6, 0.475),
                (7, 0.275),
            ]),
        },
        FuelDef {
            id: 6,
            name: "agro-forestry areas".into(),
            v0: 120.0,
            d0: 0.5,
            d1: 2.0,
            hhv: 19000.0,
            humidity: Some(60.0),
            spotting: false,
            prob_ign_by_embers: 0.0,
            burn: true,
            spread_probability: probs([
                (1, 0.25),
                (2, 0.3),
                (3, 0.005),
                (4, 0.375),
                (5, 0.2),
                (6, 0.35),
                (7, 0.075),
            ]),
        },
        FuelDef {
            id: 7,
            name: "non-fire prone forests".into(),
            v0: 60.0,
            d0: 1.0,
            d1: 2.0,
            hhv: 18000.0,
            humidity: Some(65.0),
            spotting: false,
            prob_ign_by_embers: 0.0,
            burn: true,
            spread_probability: probs([
                (1, 0.25),
                (2, 0.375),
                (3, 0.005),
                (4, 0.475),
                (5, 0.35),
                (6, 0.25),
                (7, 0.075),
            ]),
        },
    ]
}

/// Build the default (legacy) fuel system.
pub fn legacy_fuel_system() -> FuelSystem {
    FuelSystem::from_defs(&legacy_fuel_defs())
        .expect("legacy fuel table is well-formed")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_table_units() {
        let fuels = legacy_fuel_system();
        assert_eq!(fuels.n_fuels(), 7);
        // v0 converted m/h -> m/min
        let broadleaves = 0usize;
        assert!((fuels.v0[broadleaves] - 140.0 / 60.0).abs() < 1e-12);
        // humidity converted percent -> fraction
        assert!((fuels.humidity[broadleaves] - 0.60).abs() < 1e-12);
        // non-vegetated fallback
        assert_eq!(fuels.non_vegetated(), Some(3));
        // conifers spotting
        assert!(fuels.spotting[4]);
        assert!((fuels.prob_ign_by_embers[4] - 0.4).abs() < 1e-12);
        // transition probability lookup
        let from = 0; // broadleaves
        let to = 4; // conifers
        assert!((fuels.transition_probability(from, to) - 0.225).abs() < 1e-12);
    }

    #[test]
    fn unknown_codes_fall_back_to_non_vegetated() {
        let fuels = legacy_fuel_system();
        let veg = Grid2::from_vec(1, 3, vec![1, 99, 5]);
        let idx = fuels.build_fuel_index_grid(&veg).unwrap();
        assert_eq!(idx[(0, 0)], 0);
        assert_eq!(idx[(0, 1)], 2); // non-vegetated dense index
        assert_eq!(idx[(0, 2)], 4);
    }
}
