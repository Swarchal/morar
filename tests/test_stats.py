from morar import stats

def test_mad_wikipedia_example():
    data_in = [1,1,2,2,4,6,9]
    correct = 1.0
    assert stats.mad(data_in) == correct

def test_mad_all_ones():
    data_in = [1,1,1,1,1,1]
    correct = 0.0
    assert stats.mad(data_in) == correct

def test_mad_skewed():
    data_in = [1,1,1,1,1,1,999]
    correct = 0.0
    assert stats.mad(data_in) == correct

def test_mad_negatives():
    data_in = [-1, -1, -1, -1, -1]
    correct = 0.0
    assert stats.mad(data_in) == correct


def test_glog():
    pass