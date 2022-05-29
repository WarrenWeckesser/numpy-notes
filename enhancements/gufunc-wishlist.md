Wish list for NumPy gufuncs
===========================

* Shape-only parameters; see gufunc-shape-only-params.md.

* Expressions for output array sizes; see gufunc-output-size-exressions.md.

* Allow gufunc parameters to be keyword arguments with configurable names
  and with the option for default values.

* Provide an additional shape validation hook, so a gufunc can require, for
  example, that a certain core dimension is at least 1.  Currently, I have
  implemented this as a check in the loop functions; see, for example,

      https://github.com/WarrenWeckesser/ufunclab/blob/main/src/means/means_gufunc.c.src)

  It would be better if the gufunc machinery provided a way for this to be
  checked by the gufunc object, after it checks the shapes for broadcast
  compatibility but before it calls a loop function.  This could be an additional
  field of the ufunc object, say `shape_check`, that is normally NULL, but can
  be a C function that is called with the array arguments (or perhaps just
  their shapes) and that can set an error if it determines that the shapes are
  not allowed.

* Add a `__dict__` attribute to the ufunc object, so in Python one can assign
  attributes to the object (just like Python function objects).  This would
  allow a gufunc such as `ufunclab.first` to store the predefined symbols
  `LT`, `LE`, etc on the gufunc object instead of in a seperate object in the
  `ufunclab` namespace.  That is, instead of

      >>> from ufunclab import first, op

      >>> a = np.array([0, 0, 0, 0, 0, -0.5, 0, 1, 0.1])
      >>> first(a, op.NE, 0.0, 0.0)
      -0.5

  one could write

      >>> from ufunclab import first

      >>> a = np.array([0, 0, 0, 0, 0, -0.5, 0, 1, 0.1])
      >>> first(a, first.NE, 0.0, 0.0)
      -0.5
