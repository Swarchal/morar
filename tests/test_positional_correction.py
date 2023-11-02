from string import ascii_uppercase

import numpy as np
import pandas as pd

from morar import positional_correction


def test_morar_dataframe_median_polish():
    # TODO: make this
    pass


def test_median_smooth_arr():
    arr = np.random.randn(16, 24)
    # add strip to rows
    arr[::2] += 5
    arr_out = positional_correction.median_smooth_arr(arr)
    print(arr)
    print(arr_out)
    # TODO: some actual tests other than if errors


def test_median_smooth_df():
    # create test data
    wells = [f"{r}{c+1:02}" for r in ascii_uppercase[:16] for c in range(24)]
    df_collection = []
    for i in range(4):
        vals = {f"val{i}": _make_random_vals() for i in range(10)}
        df = pd.DataFrame({"well": wells})
        df["plate"] = i
        for k, v in vals.items():
            df[k] = v
        df_collection.append(df)
    df_all = pd.concat(df_collection, axis=0)
    fcols = list(vals.keys())
    df_smooth = positional_correction.median_smooth_df(
        df_all, well_col="well", fcols=fcols, plate_id_col="plate"
    )
    assert df_smooth.shape[0] == df_all.shape[0]


def _make_random_vals(dims=(16, 24)):
    vals = np.random.randn(*dims)
    vals[::2] += 5
    return vals.flatten()
