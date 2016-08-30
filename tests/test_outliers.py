from morar import outliers
import pandas as pd
import numpy as np
from nose.tools import raises

# create simple dataframe with ImageQuality columns
x = [1, 2, 3]
y = [2, 4, 1]
z = [2, 5, 1]

df = pd.DataFrame(list(zip(x, y, z)))
df.columns = ["vals", "ImageQuality_test", "other"]


def test_get_image_quality():
    out = outliers.get_image_quality(df)
    print(out)
    assert out == ["ImageQuality_test"]


# column has ImageQuality in middle of string
df2 = pd.DataFrame(list(zip(x, y, z)))
df2.columns = ["vals", "ImageQuality_test", "Cells_ImageQuality"]


def test_get_image_quality_not_beginning():
    out = outliers.get_image_quality(df2)
    assert out == ["ImageQuality_test", "Cells_ImageQuality"]


test_list = df["ImageQuality_test"].tolist()

@raises(ValueError)
def test_get_image_quality_fails_non_dataframe():
    outliers.get_image_quality(test_list)


x = np.random.random(100).tolist()
y = np.random.random(100).tolist()
# add outlying values to last row
x.append(100)
y.append(500)
df3 = pd.DataFrame(list(zip(x, y)))
df3.columns = ["x", "y"]


@raises(ValueError)
def test_get_outlier_index_errors_wrong_method():
    outliers.get_outlier_index(df3, method="wrong")


@raises(ValueError)
def test_get_outlier_index_errors_non_dataframe():
    outliers.get_outlier_index(df3["x"].tolist())


def test_get_outlier_index_values():
    out = outliers.get_outlier_index(df3)
    assert len(out) == 1
    print(out)
    assert out == [100] # 0-based indexing
