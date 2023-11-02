"""
Corrections for plate positional artefacts.
"""
from functools import reduce
from string import ascii_uppercase

import numpy as np
import pandas as pd
from numpy._typing import _UnknownType


def median_smooth_df(
    data: pd.DataFrame,
    fcols: list[str],
    plate_id_col: str,
    well_col: str,
    max_iterations: int = 10,
    epsilon: float = 0.01,
) -> pd.DataFrame:
    """
    2-way median smooth/polish on each plate
    This returns the polished feature columns with well and plate columns,
    it's up to the caller to then merge this back in with any additional
    metadata.
    """
    collection = []
    for plate_name, group in data.groupby(plate_id_col):
        arr = df_to_arr_multi_feature(group, well_col, fcols)
        arr_smoothed = np.stack(
            [median_smooth_arr(i, epsilon, max_iterations) for i in arr]
        )
        df_smoothed = arr_to_df_multi_feature(arr_smoothed, well_col, fcols)
        df_smoothed[plate_id_col] = plate_name
        collection.append(df_smoothed)
    df_smoothed_all = pd.concat(collection, axis=0)
    # FIXME: pretty sure missing data / missing rows will break this
    #        as the arr_to_df function will create all rows in the plate
    return df_smoothed_all


def median_smooth_arr(
    x: np.ndarray,
    epsilon: float = 0.01,
    max_iterations: int = 10,
    verbose: bool = False,
) -> np.ndarray:
    """2-way median polish on a single 2D numpy.ndarray"""
    assert epsilon > 0.0
    assert max_iterations > 0
    x = x.astype("float64")
    overall_effect = 0.0
    row_delta, col_delta = np.inf, np.inf
    median_row_medians, median_col_medians = np.nan, np.nan
    prev_median_col_medians, prev_median_row_medians = np.nan, np.nan
    i = 0
    while i <= max_iterations:
        row_medians = np.nanmedian(x, axis=1, keepdims=True)
        if i > 0:
            prev_median_row_medians = median_row_medians
        median_row_medians = np.median(row_medians)
        overall_effect += median_row_medians
        x = x - row_medians
        row_medians -= overall_effect
        col_medians = np.nanmedian(x, axis=0, keepdims=True)
        if i > 0:
            prev_median_col_medians = median_col_medians
        median_col_medians = np.median(col_medians)
        if i > 0:
            col_delta = abs(median_col_medians - prev_median_col_medians)
            row_delta = abs(median_row_medians - prev_median_row_medians)
            if col_delta <= epsilon or row_delta <= epsilon:
                if verbose:
                    print(f"Iteration: {i+1}, epsilon: {min([col_delta, row_delta])}")
                break
        overall_effect += median_col_medians
        x -= col_medians
        col_medians -= overall_effect
        i += 1
        if i >= max_iterations:
            break
        if verbose:
            print(f"Iteration: {i+1}, epsilon: {min([col_delta, row_delta])}")
    return x


def positional_norm(data: pd.DataFrame) -> pd.DataFrame:
    """outer-product correction of plate averages"""
    # TODO: make this
    # TODO: for each plate, convert to matrix per feature
    # TODO: correct matrix
    # TODO: convert back to well-ID -> value
    pass


def polynomial(data: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    """subtracting the polynomial surface calculated on each plate"""
    # TODO: make this
    # TODO: for each plate, convert to matrix per feature
    # TODO: fit surface to each matrix
    # TODO: correct matrix
    # TODO: convert back to well-ID -> value
    pass


def guess_plate_dims(wells: list[str] | pd.Series) -> tuple[int, int]:
    """
    Guess the plate dimensions from well labels. This is trivial if there are
    no missing wells, but more difficult if wells are missing from the well
    list.
    """
    wells = [well.upper() for well in wells]
    for well in wells:
        assert well[0] in ascii_uppercase and all([i.isdigit() for i in well[1:]])
    if all([well[0] <= "H" and int(well[1:]) <= 12 for well in wells]):
        return (8, 12)
    elif all([well[0] <= "P" and int(well[1:]) <= 24 for well in wells]):
        return (16, 24)
    else:
        return (32, 48)


def df_to_arr(df: pd.DataFrame, well_col: str, fcol: str) -> np.ndarray:
    """
    given a single plate, this converts it from a dataframe to an array
    in the format of a plate, with them dimensions [row, col].
    --
    FIXME: how to handle duplicate wells? this should be at the well-averaged level?
           just assume it's at the well-level already and the user should deal
           with their own mess?
    """
    # FIXME: dont' need to guess the plate dims for every plate and feature
    nrow, ncol = guess_plate_dims(df[well_col])
    arr = np.empty([nrow, ncol], dtype=float)
    arr[:] = np.nan
    for _, well, val in df[[well_col, fcol]].itertuples():
        r, c = well_to_zero_row_col(well)
        arr[r, c] = val
    return arr


def df_to_arr_multi_feature(
    df: pd.DataFrame, well_col: str, fcols: list[str]
) -> np.ndarray:
    """
    given a single plate dataframe of multiple features, this returns
    an array in the format of a plate stacked by features, with the dimensions
    [feature, row, col]
    """
    return np.stack([df_to_arr(df, well_col, f) for f in fcols])


def well_to_zero_row_col(well: str) -> tuple[int, int]:
    """return zero-indexed (row, column) indices from a well label"""
    return ord(well[0].upper()) - 65, int(well[1:]) - 1


def zero_row_col_to_well(row: int, column: int) -> str:
    return f"{ascii_uppercase[row]}{column+1:02}"


def arr_to_df(arr: np.ndarray, well_col: str, fcol: str) -> pd.DataFrame:
    """
    convert an array of dimensions [row, col] into a dataframe.
    Reverses df_to_arr().
    """
    wells = []
    vals = []
    nrow, ncol = arr.shape
    for r in range(nrow):
        for c in range(ncol):
            well = zero_row_col_to_well(r, c)
            val = arr[r, c]
            wells.append(well)
            vals.append(val)
    return pd.DataFrame({well_col: wells, fcol: vals})


def arr_to_df_multi_feature(
    arr: np.ndarray, well_col: str, fcols: list[str]
) -> pd.DataFrame:
    df_collection = []
    for idx, fcol in enumerate(fcols):
        arr_f = arr[idx]
        df_f = arr_to_df(arr_f, well_col, fcol)
        df_collection.append(df_f)
    df_merged = reduce(
        lambda left, right: pd.merge(left, right, on=[well_col], how="left"),
        df_collection,
    )
    return df_merged
