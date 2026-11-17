"""Persistent cache of measured SWEA electron spectra for dual-sensor v7.

Data originate exclusively from BigQuery SWEA angular-mean fluxes and the
physical energy axis. A FIXED log-energy grid (30--300 eV, 12 bins) is
interpolated within the measured support and log1p transformed. No temporal
interpolation, labels or fold statistics are stored.

This cache is intentionally distinct from the MAG raw-value cache.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ENERGY_EV = np.geomspace(30.0, 300.0, 12).astype(np.float64)


def energy_features(rows, *, low_ev=25.0, high_ev=330.0):
    """Return measured timestamps and interpolated log1p electron spectra.

    SQL must select physical energies in [25,330] eV; interpolation is only
    permitted inside the support of at least 2 independently measured bins.
    Invalid or insufficient frames are omitted (never filled with zeros).
    """
    times, values = [], []
    for row in rows:
        dims = tuple(int(v) for v in
                     (row.n_elev, row.n_azim, row.n_energy))
        if int(row.original_flux_count) != int(np.prod(dims)):
            raise ValueError("SWEA flux count is inconsistent with its axes.")
        energies = np.asarray(row.energy_ev, dtype=np.float64)
        if (len(energies) != dims[-1] or
                not np.isfinite(energies).all() or (energies <= 0).any()):
            raise ValueError("Invalid physical SWEA energy axis.")
        measured_energy = []
        measured_flux = []
        for item in row.energy_bins:
            j = int(item["energy_index"])
            if j < 0 or j >= len(energies):
                raise ValueError("Invalid SWEA physical channel index.")
            value = item["mean_flux"]
            if (value is not None and np.isfinite(float(value)) and
                    float(value) >= 0 and
                    low_ev <= float(energies[j]) <= high_ev):
                measured_energy.append(float(energies[j]))
                measured_flux.append(float(value))
        if len(measured_energy) < 4:
            continue
        ordered = np.argsort(measured_energy)
        x = np.asarray(measured_energy)[ordered]
        y = np.log1p(np.asarray(measured_flux)[ordered])
        if (np.any(np.diff(x) <= 0) or
                x[0] > ENERGY_EV[0] or x[-1] < ENERGY_EV[-1]):
            continue
        interpolated = np.interp(np.log(ENERGY_EV), np.log(x), y)
        if not np.isfinite(interpolated).all():
            continue
        timestamp = row.datetime_utc
        if timestamp.tzinfo is None:
            raise ValueError("SWEA timestamps require a timezone.")
        times.append(timestamp.astimezone(timezone.utc).timestamp())
        values.append(interpolated)
    t = np.asarray(times, dtype=np.float64)
    x = np.asarray(values, dtype=np.float64).reshape(-1, len(ENERGY_EV))
    if len(t) > 1 and np.any(np.diff(t) <= 0):
        raise ValueError("SWEA timestamps must be unique and increasing.")
    return t, x


class SweaEventCache:
    """Atomic, source-bound compressed SWEA files; corruption is an error."""

    def __init__(self, root, *, project, table, sql):
        self.root = Path(root)
        self.project = str(project)
        self.table = str(table)
        self.sql_sha = hashlib.sha256(sql.encode()).hexdigest()

    @staticmethod
    def _utc(t):
        if not isinstance(t, datetime) or t.tzinfo is None:
            raise ValueError("Event cache uses timezone-aware datetimes.")
        return t.astimezone(timezone.utc).isoformat(timespec="microseconds")

    def specification(self, event, start, end):
        if not start < event < end:
            raise ValueError("Invalid SWEA cache time support.")
        return {
            "schema": 1, "project": self.project, "table": self.table,
            "sql_sha256": self.sql_sha,
            "event": self._utc(event), "start": self._utc(start),
            "end": self._utc(end),
            "grid_ev": ENERGY_EV.tolist(),
            "value_transform": "log1p angular mean; log-energy interpolation",
            "source_energy_filter_ev": [25., 330.],
        }

    def path_for(self, event, start, end):
        spec = self.specification(event, start, end)
        digest = hashlib.sha256(
            json.dumps(spec, sort_keys=True).encode()
        ).hexdigest()[:20]
        prefix = event.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S")
        return self.root / f"swea_{prefix}_{digest}.npz"

    @staticmethod
    def _validate(times, values, start, end, max_rows):
        if (times.dtype != np.dtype("float64") or
                values.dtype != np.dtype("float64") or
                times.ndim != 1 or
                values.shape != (len(times), len(ENERGY_EV))):
            raise ValueError("Incorrect SWEA cache dimensions or precision.")
        if (not np.isfinite(times).all() or
                not np.isfinite(values).all()):
            raise ValueError("Nonfinite SWEA cached arrays.")
        if len(times) > 1 and np.any(np.diff(times) <= 0):
            raise ValueError("SWEA event timestamps are not monotonic.")
        if len(times) and (
                times[0] < start.timestamp() - .001 or
                times[-1] > end.timestamp() + .001):
            raise ValueError("SWEA cache timestamps exceed event support.")
        if max_rows is not None and len(times) > max_rows:
            raise ValueError("SWEA cached block exceeds row cap.")

    def _digest(self, t, x):
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(t, dtype="<f8").tobytes())
        h.update(np.ascontiguousarray(x, dtype="<f8").tobytes())
        return h.hexdigest()

    def load(self, event, start, end, *, max_rows=None):
        path = self.path_for(event, start, end)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {"times", "values", "metadata"}:
                    raise ValueError("Unexpected SWEA archive members.")
                t, x = archive["times"], archive["values"]
                metadata = json.loads(str(archive["metadata"].item()))
        except (OSError, EOFError, ValueError, KeyError) as err:
            raise ValueError(f"Corrupt SWEA cache: {path}") from err
        if metadata.get("spec") != self.specification(event, start, end):
            raise ValueError("SWEA cache provenance mismatch.")
        self._validate(t, x, start, end, max_rows)
        if (metadata.get("n_rows") != len(t) or
                metadata.get("sha256") != self._digest(t, x)):
            raise ValueError("SWEA cache checksum or row count mismatch.")
        return t, x

    def save(self, event, start, end, times, values):
        t = np.ascontiguousarray(times, dtype=np.float64)
        x = np.ascontiguousarray(values, dtype=np.float64)
        self._validate(t, x, start, end, None)
        metadata = {
            "spec": self.specification(event, start, end),
            "n_rows": len(t), "sha256": self._digest(t, x),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.path_for(event, start, end)
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=self.root, prefix=".swea_", delete=False
            ) as handle:
                tmp = handle.name
                np.savez_compressed(
                    handle, times=t, values=x,
                    metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, destination)
        finally:
            if tmp is not None and os.path.exists(tmp):
                os.unlink(tmp)
        return destination
