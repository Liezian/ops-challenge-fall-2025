import numpy as np
import pandas as pd

try:
    from numba import njit, prange
except Exception:  # pragma: no cover - numba optional
    njit = None
    prange = range  # type: ignore[assignment]


def _rolling_rank_segment_py(
    values: np.ndarray,
    start: int,
    length: int,
    window: int,
    out: np.ndarray,
) -> None:
    # Sliding comparisons for indices [start, start + length)
    max_index = start + length
    for idx in range(start, max_index):
        tail_start = idx - window + 1
        if tail_start < start:
            tail_start = start
        segment_size = idx - tail_start + 1

        current = values[idx]
        valid = 0
        count = 0

        for offset in range(segment_size):
            val = values[tail_start + offset]
            if not np.isnan(val):
                valid += 1
                if not np.isnan(current) and val <= current:
                    count += 1

        if valid == 0:
            out[idx] = np.nan
        else:
            out[idx] = np.float32(count / segment_size)


def _compute_all_py(
    values: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    window: int,
    out: np.ndarray,
) -> None:
    for i in range(offsets.size):
        _rolling_rank_segment_py(values, int(offsets[i]), int(lengths[i]), window, out)


if njit is not None:

    @njit(cache=False)
    def _rolling_rank_segment_numba(
        values: np.ndarray,
        start: int,
        length: int,
        window: int,
        out: np.ndarray,
    ) -> None:
        max_index = start + length
        for idx in range(start, max_index):
            tail_start = idx - window + 1
            if tail_start < start:
                tail_start = start
            segment_size = idx - tail_start + 1

            current = values[idx]
            valid = 0
            count = 0

            for offset in range(segment_size):
                val = values[tail_start + offset]
                if not np.isnan(val):
                    valid += 1
                    if not np.isnan(current) and val <= current:
                        count += 1

            if valid == 0:
                out[idx] = np.nan
            else:
                out[idx] = np.float32(count / segment_size)

    @njit(parallel=True, cache=False)
    def _compute_all_numba(
        values: np.ndarray,
        offsets: np.ndarray,
        lengths: np.ndarray,
        window: int,
        out: np.ndarray,
    ) -> None:
        for i in prange(offsets.size):
            start = int(offsets[i])
            length = int(lengths[i])
            _rolling_rank_segment_numba(values, start, length, window, out)

    _compute_all = _compute_all_numba
else:  # pragma: no cover - fallback when numba missing
    _compute_all = _compute_all_py


def ops_rolling_rank(input_path: str, window: int = 20) -> np.ndarray:
    window = int(window)
    if window <= 0:
        raise ValueError("window must be a positive integer")

    df = pd.read_parquet(input_path, columns=["symbol", "Close"])
    closes = df["Close"].to_numpy(dtype=np.float32, copy=False)

    indices_by_symbol = df.groupby("symbol", sort=False).indices
    group_arrays = []
    offsets_list = []
    lengths_list = []
    contiguous = True

    for idx_array in indices_by_symbol.values():
        arr = np.asarray(idx_array, dtype=np.int64)
        if arr.size == 0:
            continue
        group_arrays.append(arr)
        offsets_list.append(arr[0])
        lengths_list.append(arr.size)
        if contiguous and arr[-1] - arr[0] + 1 != arr.size:
            contiguous = False

    group_count = len(group_arrays)
    if group_count == 0:
        return np.empty((0, 1), dtype=np.float32)

    if contiguous:
        offsets = np.asarray(offsets_list, dtype=np.int64)
        lengths = np.asarray(lengths_list, dtype=np.int64)
        result = np.empty_like(closes, dtype=np.float32)
        _compute_all(closes, offsets, lengths, window, result)
    else:
        offsets = np.empty(group_count, dtype=np.int64)
        lengths = np.empty(group_count, dtype=np.int64)
        reordered = np.empty_like(closes, dtype=np.float32)
        scatter_index = np.empty(closes.size, dtype=np.int64)

        cursor = 0
        for group_id, arr in enumerate(group_arrays):
            length = arr.size
            offsets[group_id] = cursor
            lengths[group_id] = length
            reordered[cursor : cursor + length] = closes[arr]
            scatter_index[cursor : cursor + length] = arr
            cursor += length

        temp_out = np.empty_like(closes, dtype=np.float32)
        _compute_all(reordered, offsets, lengths, window, temp_out)

        result = np.empty_like(closes, dtype=np.float32)
        result[scatter_index] = temp_out

    return result.reshape(-1, 1)
