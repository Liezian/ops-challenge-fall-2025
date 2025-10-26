# retry 
import numpy as np
import pandas as pd

try:
    from numba import njit, prange
except Exception:  # pragma: no cover - numba optional
    njit = None
    prange = range  # type: ignore[assignment]


def _load_columns(input_path: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    df = pd.read_parquet(input_path, columns=["symbol", "Close"])
    closes = df["Close"].to_numpy(dtype=np.float32, copy=False)
    indices = df.groupby("symbol", sort=False).indices
    return closes, indices


def _prepare_groups(
    indices_by_symbol: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    group_arrays = []
    total = 0
    for arr in indices_by_symbol.values():
        idx = np.asarray(arr, dtype=np.int64)
        if idx.size == 0:
            continue
        group_arrays.append(idx)
        total += idx.size

    group_count = len(group_arrays)
    offsets = np.empty(group_count, dtype=np.int64)
    lengths = np.empty(group_count, dtype=np.int64)
    scatter = np.empty(total, dtype=np.int64)

    cursor = 0
    for i, idx in enumerate(group_arrays):
        length = idx.size
        offsets[i] = cursor
        lengths[i] = length
        scatter[cursor : cursor + length] = idx
        cursor += length

    return scatter, offsets, lengths


def _rolling_rank_symbol_impl(
    values: np.ndarray,
    start: int,
    length: int,
    window: int,
    result: np.ndarray,
) -> None:
    for local_idx in range(length):
        idx = start + local_idx
        current = values[idx]
        current_is_nan = np.isnan(current)

        tail = idx - window + 1
        if tail < start:
            tail = start

        segment_size = idx - tail + 1
        valid = 0
        count = 0

        for offset in range(segment_size):
            val = values[tail + offset]
            if not np.isnan(val):
                valid += 1
                if not current_is_nan and val <= current:
                    count += 1

        if valid == 0:
            result[idx] = np.nan
        elif current_is_nan:
            result[idx] = np.float32(0.0)
        else:
            result[idx] = np.float32(count / segment_size)


def _compute_all_impl(
    values: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    window: int,
    out: np.ndarray,
) -> None:
    for group_idx in range(offsets.size):
        start = int(offsets[group_idx])
        length = int(lengths[group_idx])
        _rolling_rank_symbol_impl(values, start, length, window, out)


if njit is not None:
    _rolling_rank_symbol_impl = njit(cache=False)(_rolling_rank_symbol_impl)

    @njit(parallel=True, cache=False)
    def _compute_all_impl_numba(
        values: np.ndarray,
        offsets: np.ndarray,
        lengths: np.ndarray,
        window: int,
        out: np.ndarray,
    ) -> None:
        for group_idx in prange(offsets.size):
            start = int(offsets[group_idx])
            length = int(lengths[group_idx])
            _rolling_rank_symbol_impl(values, start, length, window, out)

    _compute_all_impl = _compute_all_impl_numba  # type: ignore[assignment]


def ops_rolling_rank(input_path: str, window: int = 20) -> np.ndarray:
    window = int(window)
    if window <= 0:
        raise ValueError("window must be a positive integer")

    closes, indices = _load_columns(input_path)
    scatter, offsets, lengths = _prepare_groups(indices)

    if scatter.size == 0:
        return np.empty((0, 1), dtype=np.float32)

    sorted_closes = closes[scatter]

    sorted_result = np.empty_like(sorted_closes, dtype=np.float32)
    _compute_all_impl(sorted_closes, offsets, lengths, window, sorted_result)

    result = np.empty_like(sorted_closes, dtype=np.float32)
    result[scatter] = sorted_result

    return result.reshape(-1, 1)
