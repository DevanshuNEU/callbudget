"""Configuration + reproducibility.

Two jobs:
1. Hold the knobs (paths, the canonical scenario, model + sim seeds).
2. Produce a `run_id` by hashing the *resolved* config, so any result can be
   traced back to the exact settings that produced it, and a re-run with
   different settings produces a different id.

THEORY (content-addressing / reproducibility): a result is only trustworthy if
you can answer "what produced this?" By deriving the run id from a hash of the
config content (not a timestamp or a hand-typed name), the identity IS the
content. You cannot accidentally reuse an id for a different configuration, and
two machines with the same config compute the same id.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel

# Repo root = three parents up from this file (src/callbudget/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"


class Settings(BaseModel):
    """Resolved run configuration. Defaults encode the locked canonical scenario."""

    # Canonical scenario (FOUNDATION.md section 7): Adderall XR 20mg, Boston.
    drug_name: str = "Adderall XR"
    dose: str = "20 mg"
    city: str = "Boston"
    radius_km: float = 10.0

    # Determinism: seeds make the synthetic world + model splits reproducible.
    sim_seed: int = 7
    model_seed: int = 7

    # Prediction
    n_pharmacies: int = 60  # size of the curated Boston subset for v1
    cv_folds: int = 5  # held-out evaluation (non-circularity guard)

    # Voice: the locked single stack.
    stt_vendor: str = "deepgram"
    llm_vendor: str = "anthropic"
    tts_vendor: str = "cartesia"

    def run_id(self) -> str:
        """Stable 12-char id derived from the canonical JSON of this config.

        `sort_keys=True` makes the serialization order-independent, so the hash
        depends only on the values, not on dict insertion order.
        """
        canonical = json.dumps(self.model_dump(), sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]


DEFAULT = Settings()
