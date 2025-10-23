ufunc and gufunc documentation and API quirks
---------------------------------------------

Note: I use "ufunc" to refer to the original universal functions
that operate on their arguments element-wise, and "gufunc" for the generalized
ufuncs.  When I write `ufunc` (i.e. with the backticks), I am referring to the
NumPy object; ufuncs and gufuncs are instances of the class `ufunc` (and that
is probably the source of several documentation and API warts).

Quirls, warts, etc.
-------------------
* All `ufunc` instances (ufuncs and gufuncs) have the `outer` method,
  but it raises an exception unless the object is an elementwise ufunc
  with two inputs and one output. 

* gufuncs do not accept the `where` parameter, but the class docstring says
  they do.
* gufuncs have the following methods, but they will raise an exception if called:
  * `accumulate`
  * `at`
  * `outer`
  * `reduce`
  * `reduceat`
