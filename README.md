# morar

[![Build Status](https://travis-ci.org/Swarchal/morar.svg?branch=master)](https://travis-ci.org/Swarchal/morar)
[![Codecov branch](https://img.shields.io/codecov/c/github/Swarchal/morar/master.svg)](https://codecov.io/gh/Swarchal/morar)

Python package for phenotypic screening data.

### normalising plate and batch effects

Take an example multivariate dataset containing multiple plates/batches.

To remove any potential batch effects we want to normalise the values to the controls on each plate.

Now lets create some example data.
```python
val1 = [4,4,4,2,2,8,8,8,4,4]
val2 = [1,1,1,2,2,1,1,1,4,4]
plate = ["plate_1"]*5 + ["plate_2"]*5
compound = (["drug"]*3 + ["DMSO"]*2)*2
colnames = ["val1", "val2", "Metadata_plate", "Metadata_compound"]
data = pd.DataFrame(zip(val1, val2, plate, compound), columns=colnames)
print(data)
```
```
val1  val2     Metadata_plate       Metadata_compound
4     1        plate_1              drug
4     1        plate_1              drug
4     1        plate_1              drug
2     2        plate_1              DMSO
2     2        plate_1              DMSO
8     1        plate_2              drug
8     1        plate_2              drug
8     1        plate_2              drug
4     4        plate_2              DMSO
4     4        plate_2              DMSO

```

Now we can normalise it by plate. The `normalise` function takes arguments for the compound column and name of the negative control, though these have defaults of `Metadata_compound` and `DMSO` respectively.

The `normalise` function will normalise every numerical column that is not prefixed with a Metadata column tag.

```python
from morar.normalise import normalise

normalise(data, plate_id="Metadata_plate")
```

This returns the normalised data.

```
val1   val2
2.0   -1.0
2.0   -1.0
2.0   -1.0
0.0    0.0
0.0    0.0
4.0   -3.0
4.0   -3.0
4.0   -3.0
0.0    0.0
0.0    0.0

```

We can normalise this in place to keep the metadata.

```python
from morar.normalise import get_featuredata

data[get_featuredata(data)] = normalise(data, plate_id="Metadata_plate")
```

```
val1  val2       Metadata_plate       Metadata_compound
2.0  -1.0        plate_1              drug
2.0  -1.0        plate_1              drug
2.0  -1.0        plate_1              drug
0.0   0.0        plate_1              DMSO
0.0   0.0        plate_1              DMSO
4.0  -3.0        plate_2              drug
4.0  -3.0        plate_2              drug
4.0  -3.0        plate_2              drug
0.0   0.0        plate_2              DMSO
0.0   0.0        plate_2              DMSO

```

By default the normalise function subtracts the feature values by the median negative control value for that plate. Though we can also divide the median negative controls values using the `method` argument.

```python
normalise(data, plate_id="Metadata_plate", method="divide")
```
```
val1  val2
2.0   0.50
2.0   0.50
2.0   0.50
1.0   1.00
1.0   1.00
2.0   0.25
2.0   0.25
2.0   0.25
1.0   1.00
1.0   1.00
```

#### Robust methods

There is also `morar.normalise.robust_normalise`, which for each plate: subtracts the median negative control value from the sample values, divides by the median absolute deviation of the negative controls.

```python
from morar.normalise immport robust_normalise
robust_normalise(data, plate_id="Metadata_plate")
```


#### Z-scoring values

Z-scoring means each feature has a mean of 0 and standard deviation of 1.

```python
from morar import stats

data[get_featuredata(data)].apply(stats.z_score)
```

Or, a simplified method, that normalises feature data:


```python
stats.scale_features(data)
```

#### Transforming features

We can use the generalised logarithm to coerce features towards a normal distribution.

```python
data[get_featuredata(data)].applymap(stats.glog)
```

### Removing highly correlated measurements

To find which features are redundant with a pairwise correlation > threshold, find correlation returns a list of columns names, keeping one of pairs of highly correlated features.
```python
stats.find_correlation(data, threshold=0.8)
```

### Removing columns with low variance

We can remove feature columns that have little or zero variance with `morar.stats.find_low_var`.

```python
from morar import stats
# create example data
x = np.ones(100) # all 1's
y = np.random.random(100)
z = np.random.random(100)
meta = ["plate_1"] * 100
df = pd.DataFrame(list(zip(x, y, z, meta)))
df.columns = ["x", "y", "z", "Metadata_plate"]
df.head()
```

```
x    y         z           Metadata_plate
1.0  0.114824  0.210411    plate_1
1.0  0.477286  0.110609    plate_1
1.0  0.685683  0.641711    plate_1
1.0  0.744898  0.022308    plate_1
1.0  0.541362  0.988253    plate_1
```

```python
stats.find_low_var(df)
```
```
["x"]
```

We can change the threshold for what defines low variance with the threshold argument. The default is variance less than 1e-5.

```python
stats.find_low_var(df, threshold=0.001)
```

### Detecting outliers

We can detect outliers within the data either from unusual feature values, or out-of-focus images using `morar.outliers.get_outlier_index`.


The default is to use a hampel outlier test on the feature values, with a sigma of 6. That is any values that are beyond 6 median absolute deviations from the feature median are flagged as either positive or negative outliers.
```python
from morar import outliers
outliers.get_outlier_index(df)
```

We can change the sigma value to be more or less stringent.
```python
# less stringent
outliers.get_outlier_index(df, sigma=10)
```

Or, if we have used the ImageQuality module in CellProfiler, we can detect images with poor values that indicate out-of-focus images using FocusScore or PowerLogLogSlope.

```python
outliers.get_outlier(df, method="ImageQuality")
```