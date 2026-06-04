"""DuckDB + Parquet store for the observation log (the moat).

The log is held in a DuckDB table (the working set) and mirrored to Parquet on
disk (the durable copy), so the moat is no longer ephemeral. Features
(recent_hit_rate, days_since_seen, ...) are computed as windowed SQL over the
log, not in-memory Python. tests/test_store.py proves these SQL features equal
predict/features.py exactly. At scale the same Parquet log sits downstream of a
Postgres system-of-record (see ARCHITECTURE.md).

Parity notes (why the SQL looks the way it does):
- DuckDB isodow (Mon=1..Sun=7) minus 1 matches Python date.weekday() (Mon=0).
- "strictly before this row" = ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING.
- no prior history -> SQL NULL -> NumPy NaN (HistGradientBoosting routes NaN).
- in_shortage uses a contiguous window (date >= shortage_start), which equals
  features.py's set membership because load_shortage_calendar is contiguous.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from ..predict.features import FEATURE_NAMES
from ..schemas import Pharmacy, StockObservation


class ObservationStore:
    """Append-only observation log on DuckDB + Parquet, with SQL feature queries."""

    def __init__(self, parquet_dir, pharmacies: list[Pharmacy], shortage_start: date) -> None:
        self.dir = Path(parquet_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.parquet = self.dir / "observations.parquet"
        self.shortage_start = str(shortage_start)
        self.con = duckdb.connect()
        # Determinism: single-threaded execution pins float-summation order, and
        # the ORDER BY clauses below pin row order, so a fixed log always yields
        # the same feature matrix (the auto-research journal depends on this).
        self.con.execute("SET threads TO 1")
        pharm_df = pd.DataFrame(
            [{"pharmacy_npi": p.npi, "chain": p.chain, "throughput_prior": p.throughput_prior}
             for p in pharmacies]
        )
        self.con.register("pharm_df", pharm_df)
        self.con.execute("CREATE TABLE pharmacies AS SELECT * FROM pharm_df")
        self.con.execute(
            "CREATE TABLE observations (pharmacy_npi TEXT, observed_at TIMESTAMP, in_stock TEXT)"
        )

    def load(self, observations: list[StockObservation]) -> None:
        """Replace the log with these observations (the deterministic seed)."""
        self.con.execute("DELETE FROM observations")
        self._insert(observations)
        self._flush()

    def append(self, obs: StockObservation) -> None:
        self._insert([obs])
        self._flush()

    def count(self) -> int:
        return self.con.execute("SELECT count(*) FROM observations").fetchone()[0]

    def _insert(self, observations: list[StockObservation]) -> None:
        if not observations:
            return
        df = pd.DataFrame(
            [
                {"pharmacy_npi": o.pharmacy_npi, "observed_at": o.observed_at,
                 "in_stock": str(o.in_stock)}
                for o in observations
            ]
        )
        self.con.register("ins_df", df)
        self.con.execute("INSERT INTO observations SELECT * FROM ins_df")
        self.con.unregister("ins_df")

    def _flush(self) -> None:
        self.con.execute(f"COPY observations TO '{self.parquet}' (FORMAT parquet)")

    def feature_matrix(self):
        """Training matrix: one row per observation, features from its strict past."""
        sql = f"""
        SELECT
          p.throughput_prior AS throughput_prior,
          CASE WHEN o.observed_at::date >= DATE '{self.shortage_start}' THEN 1.0 ELSE 0.0 END
              AS in_shortage,
          CAST(isodow(o.observed_at) - 1 AS DOUBLE) AS dow,
          CASE WHEN p.chain IS NOT NULL THEN 1.0 ELSE 0.0 END AS is_chain,
          avg(CASE WHEN o.in_stock = 'yes' THEN 1.0 ELSE 0.0 END) OVER w AS recent_hit_rate,
          date_diff('day', max(o.observed_at::date) OVER w, o.observed_at::date) AS days_since_seen,
          CASE WHEN o.in_stock = 'yes' THEN 1 ELSE 0 END AS y
        FROM observations o JOIN pharmacies p USING (pharmacy_npi)
        WINDOW w AS (PARTITION BY o.pharmacy_npi ORDER BY o.observed_at
                     ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
        ORDER BY o.pharmacy_npi, o.observed_at
        """
        df = self.con.execute(sql).df()
        return df[FEATURE_NAMES].to_numpy(dtype=float), df["y"].to_numpy(dtype=int)

    def request_features(self, as_of: date):
        """Features for every pharmacy as of a request day (all history is prior)."""
        sql = f"""
        SELECT
          p.pharmacy_npi AS npi,
          p.throughput_prior AS throughput_prior,
          CASE WHEN DATE '{as_of}' >= DATE '{self.shortage_start}'
               THEN 1.0 ELSE 0.0 END AS in_shortage,
          CAST(isodow(DATE '{as_of}') - 1 AS DOUBLE) AS dow,
          CASE WHEN p.chain IS NOT NULL THEN 1.0 ELSE 0.0 END AS is_chain,
          avg(CASE WHEN o.in_stock = 'yes' THEN 1.0 ELSE 0.0 END) AS recent_hit_rate,
          date_diff('day', max(o.observed_at::date), DATE '{as_of}') AS days_since_seen
        FROM pharmacies p LEFT JOIN observations o USING (pharmacy_npi)
        GROUP BY p.pharmacy_npi, p.throughput_prior, p.chain
        ORDER BY p.pharmacy_npi
        """
        df = self.con.execute(sql).df()
        return df["npi"].tolist(), df[FEATURE_NAMES].to_numpy(dtype=float)
