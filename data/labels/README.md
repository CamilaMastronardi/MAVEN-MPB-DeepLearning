# MPB crossing annotations

This directory contains the event annotations supplied for the Machine Learning in Geofluids project.

## Files

- `group1.csv`: 108 rows (2014–2015).
- `group2.csv`: 219 rows (2018–2020).
- `group4.csv`: 172 rows (2017–2019).
- `mpb_crossings.csv`: unified, time-sorted table with 499 distinct MPB timestamps.

The uploaded `MAVEN_MPB_Data_Group1(5).csv` and `MAVEN_MPB_Data_Group1(6).csv` files had exactly identical bytes, so only one source copy is included here.

## Unified-table columns

- `mpb_timestamp_utc`: datetime from `yyyy-mm-dd` + `MPB time`.
- `bs_timestamp_utc`: datetime from `yyyy-mm-dd` + `BS time`.
- `source_group`: supplied file group.
- `source_row`: original CSV row number, counting the header as row 1.
- `theta_bn_deg`: original ThetaBn value (when supplied).
- `beta_proton`, `pdyn_proton`: original ancillary columns when identifiable.

Times are assumed to represent UTC, but the source files do not specify the time standard explicitly. The date of each BS event is taken from the same date column as the MPB event.

**Crucial labeling limitation:** `MPB time` is one annotated timestamp per event. It is not enough to infer the start/end extent of each crossing or four characteristic times. Verify whether these annotations represent onset, crossing center, or another convention before assigning a physical meaning to localization errors.

## Quality-control notes

- A pair in Group2 is separated by 16 seconds: 2018-07-04T11:08:20Z and 2018-07-04T11:08:36Z. Both are retained pending manual review.
- Group4 has an unusually large BS–MPB separation on 2019-06-10 (BS 13:08:46; MPB 21:59:27). Both values are preserved as provided. Check the source before using BS times as constraints.
- These are *positive event annotations*, not a fully enumerated list of all crossings. Unannotated time periods must not automatically be considered confirmed negatives.
- The Gaussian width used to create training targets is a *label-smoothing hyperparameter*, not the measured physical crossing duration.

Large time series fetched from BigQuery and trained weights remain excluded from Git.
