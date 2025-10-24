
Over the years, the following have been suggested as candidates for implementing
as a gufunc, perhaps with a Python wrapper for backwards compatbility and a nicer
user API.  (Note: this list is a snapshot from a search--not necessarily thorough--of
github issues as of October 2025.)

Existing functions:

* `searchsorted` (https://github.com/numpy/numpy/issues/4224)
* `argmin` (etc.) (https://github.com/numpy/numpy/issues/12516, https://github.com/numpy/numpy/issues/8710)
* `sort`, `argsort` (https://github.com/numpy/numpy/issues/12517)
* `interp` (https://github.com/numpy/numpy/issues/23434)
* `linalg.qr` (https://github.com/numpy/numpy/issues/7179)
* `vdot` (https://github.com/numpy/numpy/issues/21915)
* `mean` (https://github.com/numpy/numpy/issues/12901)
* `var` (https://github.com/numpy/numpy/issues/13199, https://github.com/numpy/numpy/issues/9631)
* `cross` (https://github.com/numpy/numpy/issues/13718, https://github.com/numpy/numpy/issues/2624)
* `unwrap` (https://github.com/numpy/numpy/issues/9959)
* `convolve` & `correlate` (https://github.com/numpy/numpy/issues/5624)
* `linalg.lstsq` (https://github.com/numpy/numpy/issues/8720)
* `median` (https://github.com/numpy/numpy/issues/18298)

New functions

* `find` or `first` (https://github.com/numpy/numpy/issues/2269)
* gufunc to test for all elements equal along axis (https://github.com/numpy/numpy/issues/8513)
* `minmax` (https://github.com/numpy/numpy/issues/9836)
* `linalg.gram` (https://github.com/numpy/numpy/issues/29559)
* stable sum (https://github.com/numpy/numpy/issues/8786)

Related

* ENH: add out arguments to linalg gufuncs (https://github.com/numpy/numpy/issues/11380)
* Expose inner1d and other generalized universal functions from numpy.core.umath_tests
  (https://github.com/numpy/numpy/issues/16983)
* ENH: Create a place for "optimization" ufuncs (https://github.com/numpy/numpy/issues/18483)