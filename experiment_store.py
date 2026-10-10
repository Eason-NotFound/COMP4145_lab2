"""Experiment favorites: serialization, format versioning and validation.

Pure functions (no Streamlit import) so they can be unit-tested.
The favorites list itself lives in st.session_state on the app side;
cross-session persistence is done by exporting to a JSON file and
importing it later.

JSON layout (format_version = 1):
  single experiment: {"format_version": 1, "id", "name", "saved_at", "kind",
                      "config", "metrics", "curves"}
  bundle:            {"format_version": 1, "experiments": [ ... ]}

Curves are stored as ordered [date_string, value] pairs so that ordering
and float precision survive the JSON round trip.
"""

import json
import math
import uuid
from datetime import datetime

import numpy as np
import pandas as pd

FORMAT_VERSION = 1
KINDS = ("single", "portfolio", "comparison")

REQUIRED_TOP = ("format_version", "name", "kind", "saved_at", "config",
                "metrics", "curves")
REQUIRED_CONFIG = {
    "single": ("source", "ticker", "period", "initial_capital"),
    "portfolio": ("source", "tickers", "period", "weights_pct", "initial_capital"),
    "comparison": ("source", "ticker", "period", "initial_capital", "fee_pct"),
}


# ---------------------------------------------------------------------------
# JSON-safe conversion
# ---------------------------------------------------------------------------
def jsonable(obj):
    """Recursively convert numpy/pandas values to plain JSON-safe Python."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, pd.Series):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return [jsonable(r) for r in obj.to_dict("records")]
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)  # fallback: never lose an experiment to an odd type


def series_to_pairs(series):
    """Equity Series -> ordered [[date, value], ...] with date-only strings."""
    return [[str(pd.Timestamp(d).date()), jsonable(v)]
            for d, v in zip(series.index, series.tolist())]


def pairs_to_series(pairs):
    """Inverse of series_to_pairs (used after importing)."""
    return pd.Series(
        [float(v) for _, v in pairs],
        index=pd.to_datetime([d for d, _ in pairs]),
        name="Equity",
    )


# ---------------------------------------------------------------------------
# Validation & construction
# ---------------------------------------------------------------------------
def validate_experiment(exp):
    """Raise ValueError with a clear message if the experiment is malformed."""
    if not isinstance(exp, dict):
        raise ValueError("an experiment must be a JSON object.")
    label = exp.get("name", "?")
    missing = [f for f in REQUIRED_TOP if f not in exp]
    if missing:
        raise ValueError(
            f"experiment '{label}' is missing required field(s): {', '.join(missing)}."
        )
    if exp["kind"] not in KINDS:
        raise ValueError(
            f"experiment '{label}' has unknown kind {exp['kind']!r}; "
            f"expected one of: {', '.join(KINDS)}."
        )
    if not str(exp["name"]).strip():
        raise ValueError("experiment name must not be empty.")

    cfg = exp["config"]
    if not isinstance(cfg, dict):
        raise ValueError(f"experiment '{label}': 'config' must be a JSON object.")
    missing_cfg = [f for f in REQUIRED_CONFIG[exp["kind"]] if f not in cfg]
    if missing_cfg:
        raise ValueError(
            f"experiment '{label}': config is missing {', '.join(missing_cfg)}."
        )
    if not isinstance(cfg["initial_capital"], (int, float)) or cfg["initial_capital"] <= 0:
        raise ValueError(f"experiment '{label}': initial_capital must be positive.")

    if exp["kind"] == "portfolio":
        tickers = cfg["tickers"]
        weights = cfg["weights_pct"]
        if not isinstance(tickers, list) or not tickers:
            raise ValueError(f"experiment '{label}': config.tickers must be a non-empty list.")
        if not isinstance(weights, dict) or any(t not in weights for t in tickers):
            raise ValueError(
                f"experiment '{label}': config.weights_pct must contain an entry for every ticker."
            )
        total = sum(float(weights[t]) for t in tickers)
        if abs(total - 100.0) > 1e-6:
            raise ValueError(
                f"experiment '{label}': weights_pct must sum to 100, got {total:.2f}."
            )

    if not isinstance(exp["metrics"], dict):
        raise ValueError(f"experiment '{label}': 'metrics' must be a JSON object.")
    if not isinstance(exp["curves"], dict):
        raise ValueError(f"experiment '{label}': 'curves' must be a JSON object.")
    return exp


def build_experiment(name, kind, config, metrics, curves):
    """Create a validated, JSON-safe experiment dict."""
    if kind not in KINDS:
        raise ValueError(f"Unknown experiment kind {kind!r}; expected one of: {', '.join(KINDS)}.")
    name = str(name).strip()
    if not name:
        raise ValueError("Experiment name must not be empty.")
    now = datetime.now()
    exp = {
        "format_version": FORMAT_VERSION,
        "id": f"exp-{now:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}",
        "name": name,
        "saved_at": now.isoformat(timespec="seconds"),
        "kind": kind,
        "config": jsonable(config),
        "metrics": jsonable(metrics),
        "curves": jsonable(curves),
    }
    return validate_experiment(exp)


# ---------------------------------------------------------------------------
# Export / import
# ---------------------------------------------------------------------------
def to_json(obj):
    """Serialize an experiment, a list of experiments, or a bundle dict."""
    return json.dumps(jsonable(obj), indent=2, ensure_ascii=False, allow_nan=False)


def from_json(text):
    """Parse and validate imported JSON. Returns a list of experiments.

    Accepts a single experiment object or a bundle
    {"format_version": 1, "experiments": [...]}. Raises ValueError with a
    clear message for bad JSON, wrong/missing version, or invalid content.
    """
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"the file is not valid JSON ({exc}).")

    if isinstance(parsed, dict) and isinstance(parsed.get("experiments"), list):
        items = parsed["experiments"]
    elif isinstance(parsed, dict):
        items = [parsed]
    else:
        raise ValueError(
            "top-level JSON must be an experiment object or a bundle "
            "{'format_version': 1, 'experiments': [...]}."
        )

    version = parsed.get("format_version")
    if not isinstance(version, int) or version < 1:
        raise ValueError("missing or invalid 'format_version' field.")
    if version > FORMAT_VERSION:
        raise ValueError(
            f"unsupported format_version {version}; this app supports up to "
            f"{FORMAT_VERSION}. Please upgrade the app or re-export the file."
        )
    if not items:
        raise ValueError("the file contains no experiments.")

    out = []
    for i, exp in enumerate(items):
        try:
            validate_experiment(exp)
        except ValueError as exc:
            raise ValueError(f"invalid experiment #{i + 1}: {exc}")
        out.append(exp)
    return out


def upsert(experiments, imported):
    """Insert/replace imported experiments by id. Returns (n_added, n_updated)."""
    by_id = {e["id"]: i for i, e in enumerate(experiments)}
    added = updated = 0
    for exp in imported:
        if exp["id"] in by_id:
            experiments[by_id[exp["id"]]] = exp
            updated += 1
        else:
            experiments.append(exp)
            by_id[exp["id"]] = len(experiments) - 1
            added += 1
    return added, updated
