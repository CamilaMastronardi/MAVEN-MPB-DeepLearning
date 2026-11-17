"""Persistent, integrity-checked cache of original BigQuery MAG event blocks.

This cache stores measured float64 [Bx,By,Bz,|B|] in nT and UTC Unix
timestamps, BEFORE any fold normalization or window generation. It never
stores or computes model labels, train/test partitions or scientific fakes.

Each file is bound to a full query specification including table, project,
SQL and the exact UTC event support. The cache is a LOCAL snapshot: upstream
BigQuery updates are NOT automatically detected; use --refresh-mag-cache to
refresh source observations. Files are not intended for Git.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = 1


def utc_string(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("MAG cache requires timezone-aware UTC event times.")
    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds")


def array_hash(times: np.ndarray, values: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(times, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(values, dtype="<f8").tobytes())
    return digest.hexdigest()


class MagEventCache:
    """Atomic compressed NPZ files indexed by source and query interval."""

    def __init__(self, root: Path, *, project: str, table: str, sql: str):
        if not project or not table or not sql:
            raise ValueError("MAG cache requires source project, table, and SQL.")
        self.root = Path(root)
        self.project = project
        self.table = table
        self.sql_sha256 = hashlib.sha256(sql.encode("utf-8")).hexdigest()

    def specification(
        self, event_time: datetime, start: datetime, end: datetime
    ) -> dict:
        if not start < end or not start <= event_time <= end:
            raise ValueError("Invalid MAG event time support.")
        return {
            "schema": SCHEMA,
            "project": self.project,
            "table": self.table,
            "sql_sha256": self.sql_sha256,
            "event_utc": utc_string(event_time),
            "start_utc": utc_string(start),
            "end_utc": utc_string(end),
            "channels": ["Bx_ss_nT", "By_ss_nT", "Bz_ss_nT", "|B|_nT"],
        }

    def path_for(self, event_time, start, end) -> Path:
        spec = self.specification(event_time, start, end)
        key = hashlib.sha256(
            json.dumps(spec, sort_keys=True).encode("utf-8")
        ).hexdigest()[:20]
        prefix = event_time.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S")
        return self.root / f"mag_{prefix}_{key}.npz"

    @staticmethod
    def _validate_arrays(times, values, start, end):
        if (times.dtype != np.dtype("float64")
                or values.dtype != np.dtype("float64")):
            raise ValueError("Cached MAG arrays must preserve float64 precision.")
        if times.ndim != 1 or values.shape != (len(times), 4):
            raise ValueError("Invalid MAG cache array shapes.")
        if not np.isfinite(times).all():
            raise ValueError("Nonfinite MAG cache timestamps.")
        if len(times) > 1 and not np.all(np.diff(times) > 0):
            raise ValueError("MAG cache timestamps must be strictly increasing.")
        if len(times) and (
            times[0] < start.timestamp() - 0.001
            or times[-1] > end.timestamp() + 0.001
        ):
            raise ValueError("MAG cache observation outside requested interval.")

    def load(self, event_time, start, end, *, max_rows=None):
        """Return exact measured arrays or raise; missing != corrupt.

        Missing cache returns None, but a mismatching/corrupt cache FAILS
        loudly, never silently changes training inputs.
        """
        path = self.path_for(event_time, start, end)
        if not path.exists():
            return None
        expected_spec = self.specification(event_time, start, end)
        try:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {"times", "values", "metadata"}:
                    raise ValueError("Unexpected MAG cache contents.")
                times = archive["times"]
                values = archive["values"]
                raw_meta = archive["metadata"]
                if raw_meta.shape != () or raw_meta.dtype.kind != "U":
                    raise ValueError("Invalid MAG cache metadata type.")
                meta = json.loads(str(raw_meta.item()))
        except (OSError, EOFError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid or damaged MAG cache at {path}; "
                "use --refresh-mag-cache to overwrite it."
            ) from exc
        if meta.get("spec") != expected_spec:
            raise ValueError(
                f"Cache provenance mismatch: {path}. "
                "Use --refresh-mag-cache."
            )
        self._validate_arrays(times, values, start, end)
        if max_rows is not None and len(times) > max_rows:
            raise ValueError("Cached MAG block exceeds --max-returned-rows.")
        if int(meta.get("n_rows", -1)) != len(times):
            raise ValueError("MAG cache row count mismatch.")
        if meta.get("array_sha256") != array_hash(times, values):
            raise ValueError(
                f"MAG cache checksum mismatch: {path}; "
                "use --refresh-mag-cache."
            )
        return times, values

    def save(self, event_time, start, end, times, values) -> Path:
        """Write a complete event atomically: interruption leaves no partial hit."""
        t = np.ascontiguousarray(times, dtype=np.float64)
        v = np.ascontiguousarray(values, dtype=np.float64)
        self._validate_arrays(t, v, start, end)
        spec = self.specification(event_time, start, end)
        metadata = {
            "spec": spec,
            "n_rows": int(len(t)),
            "array_sha256": array_hash(t, v),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.path_for(event_time, start, end)
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=self.root, prefix=".mag_cache_",
                suffix=".tmp", delete=False
            ) as handle:
                temp_name = handle.name
                np.savez_compressed(
                    handle, times=t, values=v,
                    metadata=np.asarray(
                        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
                    ),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, destination)
        finally:
            if temp_name is not None and os.path.exists(temp_name):
                os.unlink(temp_name)
        return destination
