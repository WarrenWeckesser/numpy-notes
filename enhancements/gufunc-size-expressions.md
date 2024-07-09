*** UPDATE ***
While the capability proposed here is useful, the implementations details
(allowing *expressions* in the gufunc shape signature) is more complicated
than necessary. A user-defined function attached to the PyUFuncObject
that the gufunc author writes and that is passed the appropriate information
for the user to process is much simpler.  So everything after the "Examples"
section of this note can be ignored.


NumPy enhancement: size expressions in gufunc signatures
========================================================

*Proposal*

Allow some of the dimensions in the shape signature of a gufunc to
be functions of other dimensions in the signature.

*Why?*

This would greatly expand the class of operations that could be
implemented as gufuncs.

The simplest way to introduce this proposal is with examples.

Examples
--------

In these first examples, only the output shapes use expressions.

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

In general, it should be possible for *input* shapes to include
expressions, as long as each symbol used in the shape occurs at least
once as a singleton expression in one of the input dimensions.  So
something like `(n, n+2) -> (n+1)` would be allowed, but not
`(n-1),(n+1) -> n`.

Currently I have just one example of the general case:

* A boolean test for an `n`-dimensional point being within a
  simplex defined by `n+1` points.  The shape signature would be

      (n), (n+1, n) -> ()

Implementation
--------------

The allowed expressions are relatively simple.  They are *not*
arbitrary Python expressions.  Note that the operands are always
nonnegative integers, and expressions must always result in
nonnegative integers.

Expressions may contain literal integers and identifiers.

Each identifier must occur at least once as a singleton expression
in a dimension of the input shape.

At a minimum, the allowed operators in an expression would be:

* Infix integer arithmetic: `+`, `-`, `*`, `//`, `%`, `**`
* Predefined functions: `abs`, `max`, `min`
* Maybe a ternary operator, like `value if cond else other` in Python
  or `cond ? value : other` in C.

When the gufunc is created, the expressions are parsed and compiled into
stack-oriented byte-code (not necessarily Python byte code).  When a
gufunc with expressions is called, the byte-code is evaluated using the
given array sizes as inputs.

An experiment for parsing and evaluating such expressions is in the
[c/gufunc-out-expr](https://github.com/WarrenWeckesser/experiments/tree/master/c/gufunc-out-expr)
directory of my [experiments](https://github.com/WarrenWeckesser/experiments)
repository.
