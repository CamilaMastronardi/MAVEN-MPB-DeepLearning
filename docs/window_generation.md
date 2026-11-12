# Step 5 — Fold-aware sliding windows, direct from BigQuery to RAM

Status: **pilot code created, awaiting execution on WSL**. It builds MAG and SWEA windows and supervision scores, **not a trained neural network**.

## Input contract

Use the **versioned 4-fold split** in \`data/splits/cv4_monthly_manifest.csv\`. A given month's events and all their windows remain in one fold. For a requested \`--fold K\`, \`--subset val\` selects held-out events and \`--subset train\` selects events from the other three folds. Ten known SWEA-unavailable events remain available for MAG and are skipped for SWEA.

The first event pilot uses a fixed time neighborhood:

- MPB label at \`t_MPB\`.
- Candidate window **centers**: every native instrument timestamp in \`[t_MPB - 30 min, t_MPB + 30 min]\`.
- Each input window: all native instrument measurements in \`[t_center - 5 min, t_center + 5 min]\`.
- Prefetch **one BigQuery block per event and instrument** over \`[t_MPB - 35 min, t_MPB + 35 min]\`, then generate all overlapping windows from that same in-RAM block. Do not query BigQuery separately for every window.
- **MAG**: channels \`Bx, By, Bz, |B|\`, with shape \`(4, N_t)\` for a model.
- **SWEA**: angular-mean spectra only for physical energies **30 to 100 eV inclusive**, with shape \`(1, N_E, N_t)\` for a model. The mapping uses the \`energy_ev\` values stored in BigQuery, not assumed index positions. No quality-code exclusion, per the selected input policy. The angular mean remains exploratory rather than a calibrated omnidirectional flux.

The sample count \`N_t\` is **not fixed**. SWEA can have native cadences of 16 or 32 seconds in the labeled material; MAG is usually near 1 second. No MAG/SWEA resampling, temporal interpolation, or uniform-grid conversion is performed by this pilot.

### Window acceptance criteria

A candidate window is rejected when:

- Its measurement support does not reach the ends of the 600-second interval within **1.75 ×** the event block's median sample cadence (so missing portions at the edges are not treated as real coverage);
- It contains an internal gap greater than **2.5 ×** the median native cadence;
- Its selected input values contain NaN or infinity;
- It contains fewer than three time samples.

These tolerances are preliminary data-quality choices. They must be evaluated across events before finalizing the data loader. Window outputs are views into the event block: no thousands-of-windows materialization or local measurement files.

## Center-proximity target

For each native center time \`t_c\`, define a *supervised temporal-proximity score*:

\`y(t_c) = exp(-0.5 * ((t_c - nearest_known_t_MPB) / sigma)^2)\`

The nearest known MPB timestamp is chosen from **the same split pool** to avoid assigning contradictory scores around closely spaced annotated events (including the catalog's close pairs). Initial \`sigma = 60 s\`, adjustable with \`--sigma-seconds\`. **This is not an intrinsic MPB thickness, a calibrated crossing probability, or a guarantee that distant times are physical non-events.**

Negative windows are **not yet sampled or certified**. Some periods far from the catalog labels may contain unlabeled boundaries; do not train a binary classifier that silently assumes all small-Gaussian-target centers are true negatives. Balanced sampling and negative-window validity require a separate scientific decision.

## How to run on WSL

From the repository root, with the Python virtual environment active:

\`\`\`bash
git pull origin main

# Offline synthetic unit tests (no Google credentials / BigQuery):
python -m unittest discover -s tests -v

# Dry run for the FIRST event in held-out fold 1.
python scripts/05_bigquery_window_pilot.py --fold 1 --subset val

# Execute the 2 bounded MAG/SWEA queries for that one event.
# No raw data or derived arrays are saved to disk.
python scripts/05_bigquery_window_pilot.py --fold 1 --subset val --execute
\`\`\`

For a cheaper MAG-only run:

\`\`\`bash
python scripts/05_bigquery_window_pilot.py --instrument mag --fold 1 --subset val --execute
\`\`\`

To inspect SWEA only, or use training events:

\`\`\`bash
python scripts/05_bigquery_window_pilot.py --instrument swea --fold 1 --subset val --execute
python scripts/05_bigquery_window_pilot.py --instrument swea --fold 1 --subset train --execute
\`\`\`

Defaults are dry-run only, **one event maximum unless explicitly selected**, \`--max-query-gib 1\`, \`--max-total-gib 2\`, \`--max-returned-rows 25000\`. Use \`--event-index N\` to choose an event from the sorted fold/subset list, and \`--limit-events N\` up to three to inspect consecutive events. Query estimates can remain similar even after selecting only 30–100 eV because BigQuery scans the full repeated flux array internally.

## What to inspect after executing

The pilot prints: BigQuery bytes estimated, number of source epochs, native cadence, selected energy channels, invalid input cells, number of eligible center timestamps, accepted/rejected windows and reasons, variable input dimensions, target-score range and the native center nearest the MPB label.

The next phase is to build **batching and training-only normalization** from these checked in-RAM windows, retaining masks/actual time offsets where the native sample counts vary. Do not train on validation statistics, and do not store raw MAVEN data in local CSV/NPZ files.

**Temporal leakage constraint:** the saved month-grouped CV split was audited only for centers within ±30 minutes of the label and 10-minute windows, implying ±35 minutes total support and a required inter-fold event gap >70 minutes. If expanding the context or adding other negatives, recheck all physical window supports against the held-out folds.
