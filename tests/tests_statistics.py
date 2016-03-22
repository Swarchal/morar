import morar.statistics as sts
import numpy as np

def test_median():
    x = [1,1,1,1,10]
    out = sts.median(x)
    assert out == 1

def test_median():
    x = [0,1,0,0,2,4,7]
    out = sts.median(x)
    assert out == 1

def test_mad():
    x = [1,1,2,2,4,6,9]
    out = sts.mad(x)
    assert out == 1

def test_hampel():
    x = [1,2,4,1,3,1,2,3,2,1,3,1,100]
    out = sts.hampel(x)
    ans = [0,0,0,0,0,0,0,0,0,0,0,0,1]
    assert out == ans
    
def test_winsorise():
     x = [1,2,4,1,3,1,2,3,2,1,3,1,100]
     out = list(sts.winsorise(x))
     assert len(x) == len(out)
     assert max(out) == 4

def test_iqr():
    x = range(1, 101)
    out = sts.iqr(x)
    assert out == 49.5

# def test_o_iqr():
#     x_ = list(np.random.normal(1, 0.01, 39))
#     x = x_ + [100]
#     out = sts.o_iqr(x)
#     print out
#     ans = [0] * 39 + [1]
#     assert out == ans
# 
# def test_u_iqr():
#     x_ = list(np.random.normal(100, 0.01, 39))
#     x = x_ + [1]
#     out = sts.u_iqr(x)
#     print out
#     ans = [0] * 39 + [1]
#     assert out == ans
