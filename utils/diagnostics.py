"""Shared helpers for warnings, strict mode and bench-data paths.

Every place where the simulator falls back to a default (missing benchmark
file, missing row, KV cache that does not fit, SLO miss) records a warning
here instead of only printing it. ``main.py`` prints a summary at the end and,
with ``--strict``, exits non-zero if any warning was recorded, so a result that
silently depends on fallback values cannot be mistaken for a calibrated one.
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_warnings = []


def bench_path(relative_path):
    """Resolve a bench_data path relative to the repo, not the CWD."""
    return os.path.join(REPO_ROOT, relative_path)


def warn(message):
    _warnings.append(message)
    print(f"Warning: {message}")


def get_warnings():
    return list(_warnings)


def reset_warnings():
    _warnings.clear()
