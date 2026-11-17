"""SQL MAG/SWEA y validacion de fuentes MAVEN en BigQuery (sin entrenamiento).

La energia SWEA en la consulta base es 30–100 eV, pero V7b cambia
los limites de lectura a 25–330 eV para interpolar sus 12 energias 30–300 eV.
"""
from __future__ import annotations
from datetime import datetime, timezone
from google.cloud import bigquery

PROJECT = "maven-vignes-crossings"
MAG = f"{PROJECT}.maven.mag_fields"
SWEA = f"{PROJECT}.maven.swea_svy3d"
AXES = f"{PROJECT}.maven.swea_svy3d_axes"
GIB = 1024 ** 3
SWEA_ENERGY_MIN_EV = 30.0
SWEA_ENERGY_MAX_EV = 100.0

MAG_SQL = """
SELECT datetime_utc, bx_ss_nt, by_ss_nt, bz_ss_nt, b_total_nt, range_b
FROM `{mag}`
WHERE DATE(datetime_utc) IN UNNEST(@days)
  AND datetime_utc BETWEEN @start AND @end
ORDER BY datetime_utc
"""

# Angular bins stay in BigQuery. The client receives one angular-mean flux
# and valid-bin count only for 30–100 eV channels at each SWEA epoch.
# Using the physical energy axis (rather than hardcoded channel indices)
# handles logarithmic channel placement and possible grid changes.
SWEA_SQL = """
SELECT d.datetime_utc, d.source_file,
       a.energy_ev, a.n_elev, a.n_azim, a.n_energy,
       ARRAY_LENGTH(d.diff_en_fluxes) AS original_flux_count,
       ARRAY(
         SELECT AS STRUCT
           MOD(offset_index, a.n_energy) AS energy_index,
           AVG(IF(
             flux IS NOT NULL AND NOT IS_NAN(flux)
             AND NOT IS_INF(flux) AND flux >= 0,
             flux, NULL
           )) AS mean_flux,
           COUNTIF(
             flux IS NOT NULL AND NOT IS_NAN(flux)
             AND NOT IS_INF(flux) AND flux >= 0
           ) AS valid_count
         FROM UNNEST(d.diff_en_fluxes) AS flux WITH OFFSET AS offset_index
         WHERE a.energy_ev[SAFE_OFFSET(MOD(offset_index, a.n_energy))]
               BETWEEN @energy_min_ev AND @energy_max_ev
         GROUP BY energy_index
         ORDER BY energy_index
       ) AS energy_bins
FROM `{swea}` AS d
JOIN `{axes}` AS a USING (source_file)
WHERE DATE(d.datetime_utc) IN UNNEST(@days)
  AND d.datetime_utc BETWEEN @start AND @end
ORDER BY d.datetime_utc
"""

def to_utc(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("Time must have an explicit UTC offset.")
    return value.astimezone(timezone.utc)

def query_config(start, end, *, dry_run, max_bytes):
    options = {
        "dry_run": dry_run,
        "use_query_cache": not dry_run,
        "query_parameters": [
            bigquery.ArrayQueryParameter(
                "days", "DATE", sorted({start.date(), end.date()})
            ),
            bigquery.ScalarQueryParameter("start", "TIMESTAMP", start),
            bigquery.ScalarQueryParameter("end", "TIMESTAMP", end),
            bigquery.ScalarQueryParameter(
                "energy_min_ev", "FLOAT64", SWEA_ENERGY_MIN_EV
            ),
            bigquery.ScalarQueryParameter(
                "energy_max_ev", "FLOAT64", SWEA_ENERGY_MAX_EV
            ),
        ],
    }
    # Never send maximum_bytes_billed=None to BigQuery in dry-run mode.
    if not dry_run:
        options["maximum_bytes_billed"] = int(max_bytes)
    return bigquery.QueryJobConfig(**options)

def validate_schema(client):
    for name in (MAG, SWEA):
        table = client.get_table(name)
        partitioning = table.time_partitioning
        if partitioning is None or partitioning.field != "datetime_utc":
            raise ValueError(f"Unsafe query: {name} not partitioned by datetime_utc")
    axis_fields = {field.name for field in client.get_table(AXES).schema}
    expected = {"source_file", "energy_ev", "n_elev", "n_azim", "n_energy"}
    if not expected <= axis_fields:
        raise ValueError(f"Missing SWEA axis columns: {sorted(expected - axis_fields)}")

