import numpy as np


def obs_to_str(arr: np.array) -> str:
    rows = []
    for row in arr:
        if row.dtype == bool:
            rows.append(''.join(str(int(d)) for d in row))
        else:
            rows.append(str(row.tolist()))
    return '\n'.join(rows)


def arr_to_str(arr: np.array) -> str:
    if arr.dtype == bool:
        return ''.join(str(int(d)) for d in arr)
    else:
        return str(arr.tolist())


def two_d_arr_to_str(arr: np.array) -> str:
    return '\n'.join(arr_to_str(arr[i, :]) for i in range(arr.shape[0]))