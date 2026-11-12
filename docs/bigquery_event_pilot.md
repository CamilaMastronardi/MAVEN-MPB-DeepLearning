# Phase 2 — Direct BigQuery access without local scientific-data files

**Important data policy:** MAVEN MAG and SWEA measurements remain in BigQuery. Python queries only small intervals and processes the returned results **temporarily in RAM**. There is no write to `data/raw/`, `data/processed/`, `.npz`, `.csv` or `.json` for the scientific observations.

A local Python model cannot read remote BigQuery data without transmitting some measurements over the network. The design avoids persistent downloads and keeps network transfer bounded; completely remote computation would require a cloud training environment.

## Pilot

`scripts/02_bigquery_event_pilot.py` examines a single labeled 600-second window for MAG and SWEA. Default: **dry-run only**. No raw observations are queried unless `--execute` is specified.

```bash
git pull origin main

# Estimate BigQuery bytes; no scientific data transferred:
python scripts/02_bigquery_event_pilot.py

# If the estimate is acceptable, process one event in RAM and print diagnostics:
python scripts/02_bigquery_event_pilot.py --execute

# Process only SWEA, e.g. after MAG has already completed successfully:
python scripts/02_bigquery_event_pilot.py --instrument swea --execute
```

Use `--limit-events 2` or `3` for the small optional pilot; to protect against accidental scans, only one to three events are supported. BigQuery partition pruning by event date and the ±300-second timestamp interval are mandatory. The pilot has query and total byte limits, and sends `maximum_bytes_billed` only on execution (never `None` on a dry run).

**MAG:** returns approximately 601 timestamped vectors `(Bx, By, Bz, |B|)` at the native ~1 s cadence into RAM.

**SWEA:** the original `diff_en_fluxes` arrays are **processed within BigQuery**, using the mapping `energy_index = MOD(flat_array_offset, n_energy)` inherited from the CDF flattening order. The query returns the valid-value angular mean and valid-bin count **per energy and timestamp**, with the energy axis and quality flag. This reduces network payload relative to transferring the full 3D angular-energy flux arrays. BigQuery can still scan the full repeated field internally, so bytes billed may remain significant; always review the dry-run estimate.

The angular average is an **exploratory** energy-time representation, not validated omnidirectional flux. No assumption is made yet about the physical interpretation of SWEA `quality` codes; these are reported for review. It may be necessary to replace this reduction with an instrument-aware treatment before training.

Printed diagnostics include number of samples, cadence, missing/invalid values, original SWEA dimensions, energy-time dimensions, energy range, and quality-flag distribution.

## Training design after pilot

- Build **two independent** in-memory data loaders, one for MAG and another for SWEA.
- Fetch moderately sized observation blocks once, rather than issuing a BigQuery query for **each** overlapping sliding window.
- Within each block, create 600-second windows offset by **one native sample**, yielding two separate detection-score curves.
- Reuse each in-memory block for multiple training batches/epochs when feasible. If a block is reread in later epochs, it incurs new network and potentially billed BigQuery work; plan for that cost.
- Train/validation/test splits are by event/orbit or contiguous time blocks to avoid overlapping-window leakage.
- Only model weights, small summaries, and plots may be written locally if desired; measurements remain remotely stored.

The earlier version of this script wrote `.npz` files. It has been replaced and **should not be used** if the objective is to avoid local scientific-data copies.
