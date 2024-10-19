import numpy as np

def replace_nans(X: np.ndarray) -> np.ndarray:
    Y = X.copy()
    m = np.nanmedian(Y, axis=0)
    m[np.isnan(m)] = 0
    Y[np.isnan(Y)] = np.take(m, np.where(np.isnan(Y))[1])
    return Y

# ПРОВЕРКА:
from numpy.testing import assert_array_equal

######################################################
assert_array_equal(replace_nans(
    np.array([[np.nan], [np.nan],  [np.nan]])),
    np.array([[0. ],[ 0. ],[ 0. ]])
)
######################################################
assert_array_equal(replace_nans(
    np.array([[0, 42,  42]])),
    np.array([[0, 42 , 42]])
)
######################################################
assert_array_equal(replace_nans(
    np.array([[np.nan], [1], [np.nan]])),
    np.array([[1.] ,[ 1.] ,[ 1. ]])
)
######################################################
assert_array_equal(replace_nans(
    np.array([[4], [1],  [np.nan]])),
    np.array([[4 ], [1] ,[ 2.5]])
)
######################################################
assert_array_equal(replace_nans(
    np.array([[-8], [1],  [np.nan]])),
    np.array([[-8] , [1] , [-3.5]])
)
######################################################
assert_array_almost_equal(replace_nans(
    np.array([[-1.515], [2.252],  [np.nan]])),
    np.array([[-1.515], [2.252], [0.3685]])
)
######################################################
assert_array_equal(replace_nans(
    np.array([[np.nan, np.nan,  np.nan],
              [     4, np.nan,       5],
              [np.nan,      8,  np.nan]]).T),
    np.array([[0. , 0. , 0. ],
              [4. , 4.5, 5. ],
              [8. , 8. , 8. ]]).T
)
######################################################
assert_array_equal(replace_nans(
    np.array([[20., np.nan,  4., np.nan, 22., 14.],
       [np.nan, np.nan, 42., 30., np.nan, 26.],
       [np.nan, np.nan, np.nan, np.nan, 32., 36.],
       [ 6., 26., 36.,  6., np.nan,  8.],
       [np.nan, 30., np.nan, np.nan, np.nan, 36.],
       [22.,  4., 10., np.nan, 18.,  6.]]).T),
    np.array([[20., 17.,  4., 17., 22., 14.],
       [30., 30., 42., 30., 30., 26.],
       [34., 34., 34., 34., 32., 36.],
       [ 6., 26., 36.,  6.,  8.,  8.],
       [33., 30., 33., 33., 33., 36.],
       [22.,  4., 10., 10., 18.,  6.]]).T
)
######################################################
assert_array_equal(replace_nans(
    np.array([[ 82.,  np.nan, 182., 214.,  np.nan, 312.,  np.nan, 482.,  np.nan,  np.nan,  56.,
             np.nan,  np.nan,  np.nan, 274.,  np.nan, 388.,  np.nan, 364.,  np.nan,  np.nan, 394.,
             np.nan, 220., 190.,  98., 440., 376.,  np.nan, 200.,  np.nan,  np.nan, 326.,
             10.,  58., 294., 492.,  np.nan, 182., 410., 472., 126.,  np.nan, 498.,
            236.,  np.nan,  np.nan, 228.,  np.nan,  np.nan]]).T),
    np.array([[ 82., 255., 182., 214., 255., 312., 255., 482., 255., 255.,  56.,
        255., 255., 255., 274., 255., 388., 255., 364., 255., 255., 394.,
        255., 220., 190.,  98., 440., 376., 255., 200., 255., 255., 326.,
         10.,  58., 294., 492., 255., 182., 410., 472., 126., 255., 498.,
        236., 255., 255., 228., 255., 255.]]).T
)
######################################################
assert_array_equal(replace_nans(
    np.array([[np.nan,298.,380.,np.nan,104.]])),
   np.array([[0, 298., 380.,0,104.]])
)
######################################################