# Phase 2 — Direct BigQuery access without local scientific-data files

**Important data policy:** MAVEN MAG and SWEA measurements remain in BigQuery. Python queries only small intervals and processes the returned results **temporarily in RAM**. There is no write to `data/raw/`, `data/processed/`, `.npz`, `.csv` or `.json` for the scientific observations.

A local Python model cannot read remote BigQuery data without transmitting some measurements over the network. The design avoids persistent downloads and keeps network transfer bounded; completely remote computation would require a cloud training environment.

## Pilot

`scripts/1_bigquery/1c_bigquery_event_pilot.py` examines a single labeled 600-second window for MAG and SWEA. Default: **dry-run only**. No raw observations are queried unless `--execute` is specified.

```bash
git pull origin main

# Estimate BigQuery bytes; no scientific data transferred:
python scripts/1_bigquery/1c_bigquery_event_pilot.py

# If the estimate is acceptable, process one event in RAM and print diagnostics:
python scripts/1_bigquery/1c_bigquery_event_pilot.py --execute

# Process only SWEA, e.g. after MAG has already completed successfully:
python scripts/1_bigquery/1c_bigquery_event_pilot.py --instrument swea --execute
```

Use `--limit-events 2` or `3` for the small optional pilot; to protect against accidental scans, only one to three events are supported. BigQuery partition pruning by event date and the ±300-second timestamp interval are mandatory. The pilot has query and total byte limits, and sends `maximum_bytes_billed` only on execution (never `None` on a dry run).

**MAG:** returns approximately 601 timestamped vectors `(Bx, By, Bz, |B|)` at the native ~1 s cadence into RAM.

**SWEA:** use the fixed **30–100 eV inclusive energy band**, selected with the true `energy_ev` axis from `maven.swea_svy3d_axes`. This excludes all energy bins outside that range regardless of their index. The original `diff_en_fluxes` arrays are processed **within BigQuery** by grouping array positions via `energy_index = MOD(flat_array_offset, n_energy)`. Only the angular average and valid-bin count of the **selected** energies are returned to Python, along with the energy axis metadata and instrument timestamps. BigQuery can still scan the entire repeated flux field internally; the narrower band primarily reduces network transfer and CNN input size, not necessarily bytes billed.

No `quality`-flag exclusion is applied. The exploratory mean includes finite, nonnegative angular flux values at each selected energy and does **not** claim to be a calibrated omnidirectional flux. The energy band is a modeling choice, not proof that all retained data are physically correct.

Printed diagnostics include temporal sample count and cadence, original SWEA cube dimensions, selected energy-channel count and actual energies, reduced energy-time shape, number of selected energy-time cells with no valid angular measurements, and source-array length consistency.

## Training design after pilot

- Build **two independent** in-memory data loaders, one for MAG and another for SWEA.
- Fetch moderately sized observation blocks once, rather than issuing a BigQuery query for **each** overlapping sliding window.
- Within each block, create 600-second windows offset by **one native sample**, yielding two separate detection-score curves.
- Reuse each in-memory block for multiple training batches/epochs when feasible. If a block is reread in later epochs, it incurs new network and potentially billed BigQuery work; plan for that cost.
- Train/validation/test splits are by event/orbit or contiguous time blocks to avoid overlapping-window leakage.
- Only model weights, small summaries, and plots may be written locally if desired; measurements remain remotely stored.

The earlier version of this script wrote `.npz` files. It has been replaced and **should not be used** if the objective is to avoid local scientific-data copies.
