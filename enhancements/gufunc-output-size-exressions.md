NumPy enhancement: size expressions in gufunc signatures
========================================================

*Proposal*

Allow the dimensions in the output of a gufunc signature to be
functions of the dimensions of the input parameters.

*Why?*

This would greatly expand the class of operations that could be
implemented as gufuncs.

The simplest way to introduce this proposal is with examples.

Examples
--------

* Compute all pairwise distances of a set of `n` vectors, each having
  length `d`:

      (n, d) -> (n*(n-1)//2)

* 1-d convolution (like `np.convolve`):

      mode='full':   (m), (n) -> (m + n - 1)
      mode='valid':  (m), (n) -> (max(m, n) - min(m, n) - 1)
      mode='same':   (m), (n) -> (max(m, n))

* A `mergesorted` gufunc that merges two sorted 1d arrays:

      (m), (n) -> (m + n)

* Singular value decomposition (as in the linalg gufuncs):

      (m, n) -> (min(m, n))

* Simple finite difference (like np.diff with n=1):

      (m) -> (m - 1)

  The general finite difference has an input `n`, and the
  signature would be

      (m) -> (m - n)

  if there was a way for the parameter `n` to be given that did not
  require an input parameter with shape `n`.  With the "shape-only"
  parameters described in a separate document, the signature for the
  general finite difference could be

      (m),<n> -> (m - n)


Implementation
--------------

Just ideas at the moment...

The allowed expressions are relatively simple.  They are *not*
arbitrary Python expressions.  Note that the operands are always
nonnegative integers, and expressions must always result in
nonnegative integers.

Expressions may contain literal integers and identifiers from
the left side of the gufunc signature.

At a minimum, the allowed operators would be:

* Infix integer arithmetic: `+`, `-`, `*`, `//`, `**`
* Predefined functions: `max`, `min`
* Maybe a ternary operator, like `value if cond else other` in Python
  or `cond ? value : other` in C.

When the gufunc is created, the expressions are parsed and compiled into
stack-oriented byte-code.  When a gufunc with expressions is called, the
byte-code is evaluated using the given array sizes as inputs.
An experiment for parsing and evaluating such expressions is in the
[c/gufunc-out-expr](https://github.com/WarrenWeckesser/experiments/tree/master/c/gufunc-out-expr)
directory of my [experiments](https://github.com/WarrenWeckesser/experiments)
repository.
