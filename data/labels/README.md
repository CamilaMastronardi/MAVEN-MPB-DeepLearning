# MPB crossing annotations

This directory contains the event annotations supplied for the Machine Learning in Geofluids project.

## Files

- `group1.csv`: 108 rows (2014–2015).
- `group2.csv`: 219 rows (2018–2020).
- `group3.csv`: 70 annotated rows (2015–2016). The source CSV also contains 16 blank rows, which are excluded from the unified table.
- `group4.csv`: 172 rows (2017–2019).
- `mpb_crossings.csv`: unified, chronologically sorted table with **569 distinct MPB timestamps**.

The two uploaded Group 1 CSV copies were identical, so only one was retained. The Group 3 source is preserved as received, including a seventh *unnamed* column of additional time-like strings; this column has not been interpreted or included in the normalized table.

## Unified-table columns

- `mpb_timestamp_utc`: datetime from `yyyy-mm-dd` + `MPB time`.
- `bs_timestamp_utc`: datetime from `yyyy-mm-dd` + `BS time`.
- `source_group`: supplied group name.
- `source_row`: row number in the original CSV, counting the header as row 1 (including blank source rows).
- `theta_bn_deg`: original ThetaBn value when available.
- `beta_proton`, `pdyn_proton`: original ancillary values when available.

Times are **assumed** to be UTC, but the input files do not explicitly state their time standard. BS timestamps were constructed using the date recorded for the MPB observation; verify day crossings before using BS–MPB time differences.

**Supervised target limitation:** `MPB time` is a single annotated timestamp per event; it is not enough to infer four times or the physical duration of the MPB. Confirm whether the marker is onset, center, or another crossing reference.

## Quality-control notes

- Group 2 has a pair of records separated by 16 seconds (2018-07-04T11:08:20Z and 2018-07-04T11:08:36Z). Both remain pending review.
- Group 4 has an unusually large BS–MPB separation on 2019-06-10 (BS 13:08:46; MPB 21:59:27). Both remain unchanged pending review.
- Group 3 has 16 blank rows and an unlabeled seventh column with time-like text (including a value containing a slash). Only explicitly labeled variables are standardized; the source is kept intact.
- These are positive-event annotations, not a guaranteed exhaustive crossing catalog. Unannotated windows must not be automatically considered confirmed negatives.
- Gaussian label width is a training hyperparameter controlling tolerance around the annotated time, **not** a physically measured MPB duration.

Large time series from BigQuery and model weights are excluded from Git.
