# Step 6 — Train-only normalization and native-window minibatches

This stage prepares inputs for the two CNN autoencoders. **No model has been trained yet.**

## Observed first event

The real-data `3b_bigquery_window_pilot.py` produced:

| | MAG | SWEA (30–100 eV) |
|---|---:|---:|
| Queried epochs | 4,200 | 132 |
| Native cadence | 1 s | 32 s |
| Accepted windows / candidate centers | 3,600 / 3,600 | 112 / 112 |
| Model input per window | `(4,601)` | `(1,11,19)` |
| Scores `y >= 0.5` | 142 | 4 |
| Missing/invalid selected input cells | 0 | 0 |

The selected 11 SWEA energy channels are approximately `[98.93, 88.05, 78.36, 69.74, 62.07, 55.24, 49.17, 43.76, 38.95, 34.66, 30.85]` eV. The original energy ordering is retained.

Only around 4% of candidate window centers in this event are within the high-Gaussian-target neighborhood. This **does not** mean the other windows are confirmed non-MPB.

## Feature transformations

Apply separate normalization for each `(held_out_fold, instrument)` pair, using measurements from that fold's **training months only**:

- **MAG:** `Bx,By,Bz,|B|` in nT. Raw values may be negative, so no log is applied. Per-channel standardization is `(x - train_mean) / train_std`.
- **SWEA:** retain **30–100 eV** channels from the original energy axis, already angular-reduced in BigQuery. Compute `log10(1 + flux)` for each nonnegative finite channel, then standardize each energy channel using **training-only** mean/std.

Fit the normalizer on **unique raw measurement epochs** (training blocks with overlap deduplicated), **not on every overlapping training window**; otherwise samples repeated across many overlapping windows would distort the statistics. Implement this deduplication before fitting a full-fold scaler.

The `FoldChannelNormalizer` implementation in `scripts/4_preprocessing/4a_native_batches.py` *requires an explicit source fold* when updating statistics, and rejects a source fold matching the held-out fold. Once frozen, the same transformation can be applied to either training or validation samples.

The transformation is a provisional default, not a validated choice for every instrument/channel.

## Padding and masks

Event cadence may vary (SWEA 16 or 32 seconds, MAG typically 1 second), and some accepted 600-second windows contain different numbers of native samples.

`collate_native_windows(windows, normalizer=...)` returns:

- MAG tensor `x`: `(batch,4,max_time_samples)`.
- SWEA tensor `x`: `(batch,1,n_selected_energies,max_time_samples)`.
- Boolean `time_mask`: `(batch,max_time_samples)`, identifying genuine observations.
- `rel_time_s`: native observation times relative to each center, padded with zero where the mask is false.
- `y`: Gaussian temporal-proximity target `(batch,1)`.
- Per-sample center and nearest-MPB offsets.

Normalized padding is zero. The provided `masked_reconstruction_mse` excludes those padded positions from the reconstruction loss, while `weighted_proximity_mse` is an **optional** weighted score-regression loss (`near_label_weight = 10` provisional) emphasizing high-proximity windows. This is NOT a validated negative classification loss.

**Critical model constraint:** the neural network itself must also avoid treating padded samples as physical measurements in its convolutions/readout; the reconstruction loss mask alone is insufficient to guarantee that. Use mask-aware readout/pooling and account for actual `rel_time_s` or group windows by cadence rather than pretending every time index is equally spaced in physical seconds.

## Offline tests

From the project root, after updating the repository:

```bash
python -m unittest discover -s tests -v
```

The new unit tests cover train/validation separation of scaler statistics, MAG z-scores, SWEA log/standardization, variable-length tensor padding, valid-sample reconstruction masking and invalid flux rejection. No BigQuery queries are performed.

## Real-data integration pilot

`scripts/4_preprocessing/4b_fold_preprocessing_pilot.py` executes a **small, dry-run-first** demonstration using one training event and one validation event from the selected fold. It fits provisional statistics from **one training event only**, then applies them to the held-out validation event. This is for testing integration, **not** a final fold normalizer, and no model weights are trained.

```bash
# Estimate two 70-minute MAG queries; no measurement transfer.
python scripts/4_preprocessing/4b_fold_preprocessing_pilot.py --fold 1 --instrument mag

# Run once the estimate is acceptable.
python scripts/4_preprocessing/4b_fold_preprocessing_pilot.py --fold 1 --instrument mag --execute

# SWEA (two 70-minute blocks, energy range 30–100 eV):
python scripts/4_preprocessing/4b_fold_preprocessing_pilot.py --fold 1 --instrument swea
python scripts/4_preprocessing/4b_fold_preprocessing_pilot.py --fold 1 --instrument swea --execute
```

Defaults: dry-run unless `--execute`; capped at **1 GiB per query and 1 GiB total**, 25,000 rows per query; the first eligible train/validation event. It prints source fold IDs, query cost estimates, training-only channel statistics, actual padded batch shapes, masks and baseline loss values. No raw data or arrays are written to disk.

## Full-fold statistics script

The single-event MAG/SWEA integration pilots have both succeeded. For full training-fold statistics, see [Step 4c: resumable fold-wide normalization](full_fold_normalization.md). Only the final JSON after all training UTC days are processed should be used for model training.

## Before full training

1. Merge overlapping training event blocks in each month to avoid repeated raw epochs in normalization and reduce BigQuery scans.
2. Verify that the 11 SWEA 30–100 eV channels have compatible **physical energy axes** across all training/validation blocks; do not silently interpret mismatched channels as equal.
3. Accumulate training-only channel statistics across all eligible training observations, streaming blocks rather than materializing the full archive locally.
4. Implement a mask-aware convolutional reconstruction/detection network for each instrument, treating actual native times explicitly when cadences vary.
5. Set a cautious near-boundary and far-from-annotation sampling/weighting policy; remote/unlabeled samples cannot automatically be labeled negative.
6. Choose model hyperparameters using training-only inner splits or keep a separate untouched held-out test group if an unbiased final performance estimate is needed.

These scientific measurement arrays are never saved locally by the pilots.
