"""Parity test: the DuckDB/Parquet store's SQL features must equal predict/features.py
exactly. This is the safety net that lets the engine migrate to the store without
behavior drift. Comparison is NaN-aware and order-independent.

Store path is under the repo (gitignored data/observations/), not /tmp.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from callbudget.data.identity import resolve
from callbudget.data.loaders import (
    load_arcos_throughput,
    load_boston_pharmacies,
    load_shortage_calendar,
)
from callbudget.data.store import ObservationStore
from callbudget.predict.features import FEATURE_NAMES, build_request_features, build_training_matrix
from callbudget.sim.callgen import generate_world

STORE_DIR = Path(__file__).resolve().parents[1] / "data" / "observations" / "_paritytest"


def _world_and_store():
    pharmacies = resolve(load_boston_pharmacies(limit=15))
    load_arcos_throughput(pharmacies)
    shortage = load_shortage_calendar()
    world = generate_world(
        history_per_pharmacy=8, seed=7, pharmacies=pharmacies, shortage_dates=shortage
    )
    store = ObservationStore(STORE_DIR, world.pharmacies, min(shortage))
    store.load(world.observations)
    return world, shortage, store


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(list(df.columns)).reset_index(drop=True)


def test_training_features_match_features_py():
    world, shortage, store = _world_and_store()
    x_py, y_py = build_training_matrix(world.observations, world.pharmacies, shortage)
    x_sql, y_sql = store.feature_matrix()
    f_py = pd.DataFrame(x_py, columns=FEATURE_NAMES).assign(y=y_py)
    f_sql = pd.DataFrame(x_sql, columns=FEATURE_NAMES).assign(y=y_sql)
    pd.testing.assert_frame_equal(
        _sorted(f_py), _sorted(f_sql), check_exact=False, atol=1e-9, check_dtype=False
    )


def test_request_features_match_features_py():
    world, _shortage, store = _world_and_store()
    npis_py, x_py = build_request_features(
        world.pharmacies, world.observations, world.as_of, _shortage
    )
    npis_sql, x_sql = store.request_features(world.as_of)
    f_py = pd.DataFrame(x_py, columns=FEATURE_NAMES).assign(npi=npis_py)
    f_sql = pd.DataFrame(x_sql, columns=FEATURE_NAMES).assign(npi=npis_sql)
    pd.testing.assert_frame_equal(
        f_py.sort_values("npi").reset_index(drop=True),
        f_sql.sort_values("npi").reset_index(drop=True),
        check_exact=False, atol=1e-9, check_dtype=False,
    )
