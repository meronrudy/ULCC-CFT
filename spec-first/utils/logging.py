"""Structured logging utilities (dependency-free).

Invariants
- Idempotent handler installation per logger.
- Pure functions aside from I/O; no hidden global RNG or state.
- Validation: values and step (if provided) must be finite floats.

Public API
- get_logger(name="spec-first", level=logging.INFO) -> logging.Logger
- log_metric(name, value, step=None, logger=None) -> None
- log_metrics(metrics: dict[str, float], step=None, logger=None) -> None
- csv_logger(path: str) -> Callable[[dict[str, float]], None]
"""
from __future__ import annotations

import logging
import math
import os
from typing import Callable, Mapping, Optional


def get_logger(name: str = "spec-first", level: int = logging.INFO) -> logging.Logger:
    """
    Return a configured logger with concise formatter.

    Idempotent: installs at most one StreamHandler marked by _spec_first_handler.
    """
    logger = logging.getLogger(name)
    logger.setLevel(int(level))
    logger.propagate = False  # avoid duplicate logs through root

    # Install handler once
    has_handler = any(getattr(h, "_spec_first_handler", False) for h in logger.handlers)
    if not has_handler:
        handler = logging.StreamHandler()
        handler._spec_first_handler = True  # type: ignore[attr-defined]
        handler.setLevel(int(level))
        formatter = logging.Formatter(
            fmt="%(asctime)s %(name)s %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _ensure_finite_float(x: object, name: str) -> float:
    try:
        val = float(x)  # type: ignore[arg-type]
    except Exception as e:
        raise TypeError(f"{name} must be a real number convertible to float") from e
    if not math.isfinite(val):
        raise ValueError(f"{name} must be finite, got {val}")
    return val


def _format_float(x: float) -> str:
    # Deterministic concise formatting
    return f"{x:.10g}"


def log_metric(
    name: str,
    value: float,
    step: int | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """
    Log a single metric as: "metric name=value step=step".

    Validation
    - value must be finite float
    - step (if provided) must be finite and integral/float
    """
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")
    v = _ensure_finite_float(value, "value")
    s_str = ""
    if step is not None:
        s = _ensure_finite_float(step, "step")
        s_str = f" step={int(s)}"
    lg = logger if logger is not None else get_logger()
    lg.info(f"metric {name}={_format_float(v)}{s_str}")


def log_metrics(
    metrics: Mapping[str, float],
    step: int | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """
    Log a dictionary of metrics as: "metrics k1=v1 k2=v2 ... step=step".

    - Keys are sorted for deterministic ordering.
    - Values validated to be finite floats.
    """
    if not isinstance(metrics, Mapping) or len(metrics) == 0:
        raise ValueError("metrics must be a non-empty mapping of str->float")
    parts: list[str] = []
    for k in sorted(metrics.keys()):
        if not isinstance(k, str) or not k:
            raise ValueError("metric keys must be non-empty strings")
        v = _ensure_finite_float(metrics[k], f"value for '{k}'")
        parts.append(f"{k}={_format_float(v)}")
    s_str = ""
    if step is not None:
        s = _ensure_finite_float(step, "step")
        s_str = f" step={int(s)}"
    lg = logger if logger is not None else get_logger()
    lg.info("metrics " + " ".join(parts) + s_str)


def csv_logger(path: str) -> Callable[[Mapping[str, float]], None]:
    """
    Return a callable that appends metrics rows into a CSV at `path`.

    Behavior
    - Header auto-created on first write (file missing or empty).
    - Columns are sorted by key for deterministic order.
    - Rows are newline-terminated.
    - Values validated to be finite floats.
    """
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")
    abs_path = os.path.abspath(path)

    def _writer(metrics: Mapping[str, float]) -> None:
        if not isinstance(metrics, Mapping) or len(metrics) == 0:
            raise ValueError("metrics must be a non-empty mapping")
        keys = sorted(metrics.keys())
        for k in keys:
            if not isinstance(k, str) or not k:
                raise ValueError("metric keys must be non-empty strings")
            _ = _ensure_finite_float(metrics[k], f"value for '{k}'")  # validate
        header_needed = not os.path.exists(abs_path) or os.path.getsize(abs_path) == 0
        # Write header and row
        with open(abs_path, "a", encoding="utf-8") as f:
            if header_needed:
                f.write(",".join(keys) + "\n")
            row = ",".join(_format_float(float(metrics[k])) for k in keys)
            f.write(row + "\n")

    return _writer


__all__ = ["get_logger", "log_metric", "log_metrics", "csv_logger"]
