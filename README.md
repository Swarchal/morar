# morar

[![Build Status](https://travis-ci.org/Swarchal/morar.svg?branch=master)](https://travis-ci.org/Swarchal/morar)

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

We can normalise this in place to keep the metadata.

```python
from morar.normalise import get_featuredata

data[get_featuredata(data)] = normalise(data, plate_id="Metadata_plate")
```

```
val1  val2       Metadata_plate       Metadata_compound
2.0  0.50        plate_1              drug
2.0  0.50        plate_1              drug
2.0  0.50        plate_1              drug
1.0  1.00        plate_1              DMSO
1.0  1.00        plate_1              DMSO
2.0  0.25        plate_2              drug
2.0  0.25        plate_2              drug
2.0  0.25        plate_2              drug
1.0  1.00        plate_2              DMSO
1.0  1.00        plate_2              DMSO

```

By default the normalise function divides the feature values by the median negative control value for that plate. Though we can also subtract the median negative controls values using the `method` argument.

```python
normalise(data, plate_id="Metadata_plate", method="subtract")
```
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

#### Transforming features

We can use the generalised logarithm to coerce features towards a normal distribution.

```python
data[get_featuredata(data)].applymap(lambda x: stats.glog(x))
```

### Removing highly correlated measurements

To find which features are redundant with a pairwise correlation > threshold, find correlation returns a list of columns names, keeping one of pairs of highly correlated features.
```python
stats.find_correlation(data, threshold=0.8)
```
