"""Offline filesystem tests for the fifth MAG measured-event cache.

Numerical placeholders check byte-for-byte transport and corruption only;
no synthetic MAVEN measurements, physics training or BigQuery calls.
"""
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from src.data.mag_event_cache import MagEventCache


class MagCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache = MagEventCache(
            Path(self.tmp.name), project="test-project",
            table="test-project.example.mag", sql="SELECT measured MAG"
        )
        self.event = datetime(2020, 3, 3, 12, 0, tzinfo=timezone.utc)
        self.start = self.event - timedelta(minutes=35)
        self.end = self.event + timedelta(minutes=35)
        self.times = np.asarray([
            self.event.timestamp() - 2.,
            self.event.timestamp() - 1.,
            self.event.timestamp(),
            self.event.timestamp() + 1.,
        ], dtype=np.float64)
        self.values = np.arange(16, dtype=np.float64).reshape(4, 4)

    def test_missing_is_none_and_save_roundtrip_bitwise(self):
        self.assertIsNone(
            self.cache.load(self.event, self.start, self.end)
        )
        path = self.cache.save(
            self.event, self.start, self.end, self.times, self.values
        )
        self.assertTrue(path.is_file())
        saved = self.cache.load(
            self.event, self.start, self.end, max_rows=25
        )
        self.assertIsNotNone(saved)
        np.testing.assert_array_equal(saved[0], self.times)
        np.testing.assert_array_equal(saved[1], self.values)
        self.assertFalse(any(
            ".mag_cache_" in entry.name for entry in Path(self.tmp.name).iterdir()
        ))

    def test_different_source_and_support_not_reused(self):
        self.cache.save(
            self.event, self.start, self.end, self.times, self.values
        )
        shifted_start = self.start - timedelta(seconds=1)
        self.assertIsNone(self.cache.load(
            self.event, shifted_start, self.end
        ))
        different_query = MagEventCache(
            Path(self.tmp.name), project="test-project",
            table="test-project.example.mag", sql="SELECT different columns"
        )
        self.assertIsNone(different_query.load(
            self.event, self.start, self.end
        ))

    def test_bad_checksum_fails_not_silent_reuse(self):
        original = self.cache.save(
            self.event, self.start, self.end, self.times, self.values
        )
        with np.load(original, allow_pickle=False) as archive:
            t = archive["times"].copy()
            v = archive["values"].copy()
            meta = archive["metadata"].copy()
        v[0, 0] += 999.
        with original.open("wb") as handle:
            np.savez_compressed(handle, times=t, values=v, metadata=meta)
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.cache.load(self.event, self.start, self.end)

    def test_bad_timestamp_or_outside_support_rejected(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            self.cache.save(
                self.event, self.start, self.end, self.times[::-1], self.values
            )
        out_of_range = self.times.copy()
        out_of_range[0] = self.start.timestamp() - 5.
        with self.assertRaisesRegex(ValueError, "outside requested interval"):
            self.cache.save(
                self.event, self.start, self.end, out_of_range, self.values
            )

    def test_corrupt_archive_and_row_cap(self):
        path = self.cache.save(
            self.event, self.start, self.end, self.times, self.values
        )
        with self.assertRaisesRegex(ValueError, "--max-returned-rows"):
            self.cache.load(self.event, self.start, self.end, max_rows=2)
        path.write_bytes(b"NOT AN NPZ")
        with self.assertRaises((ValueError, OSError, EOFError)):
            self.cache.load(self.event, self.start, self.end)


if __name__ == "__main__":
    unittest.main()
