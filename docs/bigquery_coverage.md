# Phase 1: MAG/SWEA coverage around MPB labels

This phase checks whether each of the **569** labeled MPB times has usable time sampling for a centered **600-second** window in MAG and SWEA, without loading the magnetic-field values or electron-flux arrays yet.

## BigQuery source tables

Project: `maven-vignes-crossings`

- `maven.mag_fields`: `datetime_utc`, `bx_ss_nt`, `by_ss_nt`, `bz_ss_nt`, `b_total_nt`, ...
- `maven.swea_svy3d`: `datetime_utc`, `diff_en_fluxes`, `quality`, `source_file`.
- `maven.swea_svy3d_axes`: SWEA energy and angular dimensions.
- `maven.positions`: positions for future physical interpretation.

MAG and SWEA are assessed separately with their **own native timestamps**. No resampling or interpolation between instruments is performed.

## First run on Ubuntu

From the repository root, with the Python 3.11 virtual environment activated:

```bash
git pull origin main
uv pip install -r requirements.txt
python scripts/1_bigquery/1a_bigquery_crossing_coverage.py --metadata-only
```

The metadata command prints BigQuery schemas and partitioning without scanning table rows.

Start with a **dry-run**, which is the default:

```bash
python scripts/1_bigquery/1a_bigquery_crossing_coverage.py --limit-events 5
```

This estimates processed bytes and does **not** execute the data query. If the estimates are acceptable, run the same five-event pilot:

```bash
python scripts/1_bigquery/1a_bigquery_crossing_coverage.py --limit-events 5 --execute
```

Then assess all labels, again estimating cost before executing:

```bash
python scripts/1_bigquery/1a_bigquery_crossing_coverage.py
python scripts/1_bigquery/1a_bigquery_crossing_coverage.py --execute
```

The script refuses to run if a table is not partitioned by `datetime_utc`. The query uses a **date-partition filter**, reads only `datetime_utc`, limits each event to ±300 seconds, estimates bytes before execution, and enforces default caps of 1 GiB per query and 5 GiB total estimated bytes. If the estimate exceeds either cap, review the estimate before choosing explicit higher values:

```bash
python scripts/1_bigquery/1a_bigquery_crossing_coverage.py --max-billed-gib 2 --max-total-gib 8
```

The previous command is still a dry run without `--execute`.

Other options: `--instrument mag`, `--instrument swea`, `--batch-size 40`, `--limit-events N`.

If Application Default Credentials are missing, authenticate locally with `gcloud auth application-default login` and check your BigQuery project access and query quota. Do not commit credential files.

## Output

When `--execute` is supplied, the script writes:

- `results/metrics/bigquery_coverage/crossing_coverage_by_instrument.csv`: one row per labeled event/instrument, including sample count, actual first/last timestamps, median/mean/maximum positive cadence, duplicate intervals, edge distances, and preliminary timestamp continuity status.
- `results/metrics/bigquery_coverage/crossing_coverage_summary.csv`: counts with any samples and counts passing preliminary timestamp checks for MAG, SWEA, and both.

The preliminary `preliminary_time_coverage_ok` flag checks:

- At least three timestamps and a positive median cadence;
- No exact duplicate timestamps in the interval;
- Maximum consecutive timestamp difference ≤ 2.5 × median cadence;
- Distance from each nominal window edge to the nearest sample ≤ 2.5 × median cadence.

This is an **initial timestamp-only diagnostic**, not a guarantee of physical data validity or instrument quality. In particular, it does **not** check SWEA `quality` values, energy-axis stability, negative/nonfinite flux values, angular coverage, or the full MAG numerical data.

## Next tasks after reviewing output

1. Inspect how many labels have valid temporal coverage per instrument and their overlap.
2. Define instrument-specific gap policy and SWEA energy-time projection (e.g. angular aggregation) for fixed-shape CNN inputs.
3. Fetch/copy limited event-centered raw windows to local cached files, avoiding repeated BigQuery scans.
4. Define training target `y(t_c)`, negative/offset examples, and event-wise train/validation/test splits **before** normalization.
5. Train a MAG autoencoder + detection head, then a separate SWEA autoencoder + detection head; compare native-time score curves and latent representations.

Remember: a timestamp catalog with single `t_MPB` labels does not establish the full physical duration of the MPB and may not exhaustively label all events.
