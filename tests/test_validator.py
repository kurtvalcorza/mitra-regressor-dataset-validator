"""Unit tests for the Mitra regressor dataset validator."""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validator as V  # noqa: E402


def _zip(tmp: Path, members: dict[str, pd.DataFrame], name: str = "d.zip") -> Path:
    p = tmp / name
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
        for arc, df in members.items():
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            zf.writestr(arc, buf.getvalue())
    return p


def _train(n=60):
    return pd.DataFrame({"f1": range(n), "f2": np.arange(n) * 1.5, "target": np.arange(n) - 30.0})


def _run(tmp: Path, members: dict[str, pd.DataFrame], target="target", drop=""):
    _zip(tmp, members)
    cfg = V.Config(
        dataset_dir=tmp, result_path=tmp / "out.json", done_callback="", callback_timeout=1.0,
        max_sample_files=25, pipeline_metadata={}, target_column=target,
        drop_columns=[c.strip() for c in drop.split(",") if c.strip()],
    )
    src = V.DatasetSource(tmp)
    try:
        checks, meta = V._build_checks(cfg, src)
    finally:
        src.close()
    return {c["name"]: c["successful"] for c in checks}, meta


def test_happy_path_negative_targets_ok(tmp_path):
    checks, meta = _run(tmp_path, {"train.csv": _train(60)})  # targets include negatives
    assert all(checks.values()), checks
    assert meta["usableRowCount"] == 60


def test_duplicate_train_rejected(tmp_path):
    checks, _ = _run(tmp_path, {"train.csv": _train(), "dataset/train.csv": _train()})
    assert checks["no_duplicate_tables"] is False


def test_non_numeric_target_fails(tmp_path):
    df = _train(60)
    df["target"] = ["x"] * 60
    checks, _ = _run(tmp_path, {"train.csv": df})
    assert checks["target_is_numeric"] is False


def test_null_targets_reduce_usable_rows(tmp_path):
    df = _train(60)
    df.loc[df.index[:20], "target"] = np.nan  # 40 finite
    checks, meta = _run(tmp_path, {"train.csv": df})
    assert meta["usableRowCount"] == 40
    assert checks["minimum_rows"] is False


def test_feature_limit(tmp_path):
    wide = pd.DataFrame({f"f{i}": range(60) for i in range(510)})
    wide["target"] = np.arange(60, dtype=float)
    checks, _ = _run(tmp_path, {"train.csv": wide})
    assert checks["feature_limit"] is False


def test_val_schema_mismatch(tmp_path):
    train = _train(60)
    val = _train(20).rename(columns={"f2": "other"})
    checks, _ = _run(tmp_path, {"train.csv": train, "val.csv": val})
    assert checks["val_schema_matches_train"] is False


def test_zip_bomb_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "MAX_COMPRESSION_RATIO", 2.0)
    big = pd.DataFrame({"f1": [0] * 5000, "target": [0.0] * 5000})
    _zip(tmp_path, {"train.csv": big})
    with pytest.raises(ValueError):
        V.DatasetSource(tmp_path)


def test_target_in_drop_columns_fails(tmp_path):
    checks, _ = _run(tmp_path, {"train.csv": _train(60)}, drop="target")
    assert checks["target_not_dropped"] is False


def test_val_with_no_usable_targets_fails(tmp_path):
    train = _train(60)
    val = _train(20).copy()
    val["target"] = "x"  # non-numeric -> zero usable
    checks, _ = _run(tmp_path, {"train.csv": train, "val.csv": val})
    assert checks["val_has_usable_targets"] is False


def test_safe_parse_bad_env():
    assert V._safe_int("nope", 123) == 123
    assert V._safe_float(None, 1.5) == 1.5
