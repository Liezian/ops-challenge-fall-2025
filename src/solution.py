import numpy as np
import pandas as pd

try:
    from numba import njit
except Exception:  # pragma: no cover - numba optional
    njit = None


def _rolling_rank_impl(values: np.ndarray, window: int) -> np.ndarray:
    """Core rolling-rank logic written in a Numba-friendly subset of Python."""
    n = values.size
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        if i + 1 < window:
            start = 0
            length = i + 1
        else:
            start = i - window + 1
            length = window

        current = values[i]
        count = 0
        valid = 0

        for j in range(length):
            val = values[start + j]
            if not np.isnan(val):
                valid += 1
                if not np.isnan(current) and val <= current:
                    count += 1

        if valid == 0:
            out[i] = np.nan
        else:
            out[i] = np.float32(count / length)

    return out


if njit is not None:
    _rolling_rank_1d = njit(cache=False)(_rolling_rank_impl)
else:  # pragma: no cover - fallback when numba missing
    _rolling_rank_1d = _rolling_rank_impl


def ops_rolling_rank(input_path: str, window: int = 20) -> np.ndarray:
    window = int(window)
    if window <= 0:
        raise ValueError("window must be a positive integer")

    df = pd.read_parquet(input_path, columns=["symbol", "Close"])

    closes = df["Close"].to_numpy(dtype=np.float32, copy=False)
    result = np.empty_like(closes, dtype=np.float32)

    # groupby(...).indices preserves the original order for each symbol
    indices_by_symbol = df.groupby("symbol", sort=False).indices

    for indexer in indices_by_symbol.values():
        idx = np.asarray(indexer, dtype=np.int64)
        symbol_values = np.ascontiguousarray(closes[idx])
        result[idx] = _rolling_rank_1d(symbol_values, window)

    return result.reshape(-1, 1)
