Creating a user-defined data type
---------------------------------

[To be written...]

Docs: https://numpy.org/devdocs/user/c-info.beyond-basics.html#user-defined-data-types


Questions about defining a user-defined dtype
---------------------------------------------

* How are the fields `kind` and `type` of the
  [`PyArray_Descr`](https://numpy.org/devdocs/reference/c-api/types-and-structures.html#c.PyArray_Descr)
  structure actually *used* by NumPy?  It isn't clear to me how these characters
  should be chosen for a user-defined dtype, and what the consequences of the
  choices are.
* The first step in creating a new dtype is to define a Python object for
  the data type.  For a number-like dtype, this involves implementing the
  Python number protocol for the object.  When using a NumPy array with the
  new data type, will NumPy code ever use the functions from the data type's
  implementation of the Python number protocol?  Or will NumPy only attempt
  to perform calculations with the new data type through ufuncs?
* Here's how the `rational` dtype defined in the file `_rational_tests.c.src`
  in `numpy/numpy/core/src/umath/` behaves when added to an integer numpy array
  with length greater than 1:

  ```
  In [13]: r = rational(3, 7)
  In [14]: a = np.array([1, 2])
  In [15]: a + r
  Out[15]: array([rational(10,7), rational(17,7)], dtype=rational)
  In [16]: r + a
  Out[16]: array([rational(10,7), rational(17,7)], dtype=rational)
  ```

  So that works as expected.  However, if the array has length 1:

  ```
  In [17]: a = np.array([1])
  In [18]: a + r
  Out[18]: array([rational(10,7)], dtype=rational)
  In [19]: r + a
  Out[19]: rational(10,7)
  ```

  `r + a` returns a scalar, not an array.

  What needs to be fixed in the implementation of `rational` so `r + a`
  is also an array with length 1?  Note that the builtin NumPy dtypes do
  not have this problem:

  ```
  In [20]: a + np.int32(1)
  Out[20]: array([2])
  In [21]: np.int32(1) + a
  Out[21]: array([2])
  ```
