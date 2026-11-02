"""Paths and YAML loaders for experiment0."""
import re
from pathlib import Path

import yaml

EXP_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = EXP_DIR / "configs"
OUT_DIR = EXP_DIR / "outputs"
RAW_DIR = OUT_DIR / "raw"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

TOY_RUNGS = (0, 1, 2)


def _load(name):
    with open(CONFIG_DIR / name) as fh:
        return yaml.safe_load(fh)


def protocol():
    return _load("protocol.yaml")


def toy_function():
    return _load("toy_function.yaml")


def toy_rung(k):
    return _load(f"toy_rung{k}.yaml")


def methods():
    return _load("methods.yaml")


def write_toy_seed(seed):
    """Record the curated seed in toy_function.yaml, editing only the `toy_seed:` line (comments survive)."""
    path = CONFIG_DIR / "toy_function.yaml"
    text = path.read_text()
    new, k = re.subn(r"(?m)^toy_seed:\s*null\b", f"toy_seed: {int(seed)}", text)
    assert k == 1, "toy_function.yaml has no `toy_seed: null` line: the seed is already frozen"
    path.write_text(new)


def toy_raw_path(method, rung_k, n):
    return RAW_DIR / "toy" / method / f"rung{rung_k}_n{n}.parquet"


def toy_extras_path(method, rung_k, n):
    return RAW_DIR / "toy" / method / f"rung{rung_k}_n{n}_extras.npz"
