# Phase 2 — Pilot retrieval of actual MAG and SWEA measurements

Script: `scripts/02_bigquery_event_pilot.py`.

This step follows the timestamp-only diagnostics. The 569 labels show initial timestamp coverage of 569/569 MAG and 559/569 SWEA, with median per-event cadences of **1 second (MAG)** and **16 seconds (SWEA)**.

## Design

The two models will use separate centered **600-second** windows, shifted by one **native measurement sample** at inference:

- MAG: approximately 601 field vectors in a regular 1-second window.
- SWEA: approximately 37–38 timestamped angular/energy distributions in a regular 16-second window.

Neither instrument is resampled to the cadence of the other. SWEA's 16-second cadence limits the time-grid resolution of its predicted detection curve.

The pilot uses a single *label-centered* window per event to check real physical data before building many sliding windows. **It is not yet a training dataset**.

## Commands

From the repository root with `.venv` activated:

```bash
git pull origin main

# Dry run: estimate BigQuery scanned bytes; nothing downloaded.
python scripts/02_bigquery_event_pilot.py

# Execute only if the estimated bytes are acceptable.
python scripts/02_bigquery_event_pilot.py --execute
```

Default: **one** chronological labeled event. `--limit-events 2` or `3` can be used for a larger pilot (hard cap of three to prevent accidental large scans). The script validates datetime partitioning and enforces default limits of **1 GiB estimated per query and 2 GiB total**; the same per-query limit is sent to BigQuery as `maximum_bytes_billed` on actual execution. Dry-run query jobs omit that parameter to avoid client compatibility errors.

This pilot reads the **actual SWEA flux arrays**, unlike the timestamp-only coverage diagnostic, so its scan estimate can be substantially larger.

## Saved results

The script saves under `data/raw/bigquery_pilot/` (excluded from Git):

- `event_XX_YYYYMMDDTHHMMSS_mag.npz`: native UTC timestamps stored as Unix seconds, `field_nt` with `Bx, By, Bz, |B|`, and MAG `range_b`.
- `event_XX_YYYYMMDDTHHMMSS_swea.npz`: native SWEA timestamps, raw flux arrays reshaped as `[time, elevation, azimuth, energy]`, energy/angle coordinate arrays, quality values, a provisional energy-time angular mean, and counts of valid angular values.
- `pilot_summary.json`: sample counts, per-event time spacing, MAG nonfinite counts, SWEA invalid-flux fraction and observed quality flags.

For exploratory purposes only, the SWEA mean is calculated over available finite, nonnegative angular flux values at each energy. **This is not a validated omnidirectional flux estimate, and no quality-flag semantics are assumed.** We will assess and choose the correct physical SWEA representation before training.

All source-file axis dimensions and energy/angle axes are checked for consistency. An inconsistent shape/energy grid causes the script to stop for that event rather than silently mix incompatible measurements.

## After the pilot

1. Review `pilot_summary.json` for MAG and SWEA sample counts, SWEA quality flags and array dimensions.
2. Establish valid SWEA flux selection and physically justified angular aggregation or other representation.
3. Establish a gap policy and, if required, instrument-specific fixed-length padding/masking for CNNs without cross-instrument resampling.
4. Retrieve broader event neighborhoods (beyond ±300 s) for **multiple sliding-window centers** used in training, with train/validation/test split defined by event/orbit.
5. Use labeled times to construct center-proximity targets and verify the model learns event localization; separately assess latent-space groupings.

No signal values have been fetched inside ChatGPT; this script must run on the authenticated local environment.
