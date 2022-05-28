

NumPy enhancement: extend gufuncs with shape-only parameters
============================================================

*Proposed enhancement*

Allow a parameter of a gufunc to be a "shape-only" parameter.
Instead of passing an array, one passes a shape tuple.

*Why?*

The current gufunc signature allows for functions that operate on given
arrays.  It does not provide a means for generating output whose shape
exceeds the (broadcast) shapes of the input arrays.  This means there
is no way that functions such as `linspace`, `geomspace` or `convolve1d`
can be implemented as gufuncs.

Signature change
----------------

The signature specification will be extended to allow parameters in
the input that are shape-only parameters.  Such parameters will be
delimited by angle brackets instead of parentheses.

For example, the gufunc signature for a simplified version of
`linspace`, with only the `start`, `stop` and `num` parameters, would
be `(),(),<n> -> (n)`.

Numerical literals are not accepted in the signature that defines a
shape-only parameter.  (Cf. the signature for array parameters, where
a signature such as `(n,3)` is allowed.)

The identifiers within the signature of a shape-only parameter must be
distinct, and not the same as any other identifiers used in other input
array or shape-only parameters.  For example, the following are not
allowed:

    (m),<n>,<n> -> (m,n)  # Error: can't use n twice in the input.
    (m),<m,n> -> (m,n)    # Error: can't use m twice in the input.

Shape-only parameters are not accepted in the output part of the
signature.

Identifiers within shape-only input parameters are allowed in the
output shape parameters; indeed, that is their primary reason to exist.

Shape-only signatures do not accept the `?` modifier.


gufunc user API
---------------

The argument associated with a shape-only parameter will accept tuples
of integers.  Scalar integers (i.e. anything that is accepted by
`operator.index`) are accepted, and are treated as a tuple of
length 1 containing the integer.  The tuple given by a user must
have at least as many elements as specified in the signature.

The shape-only signature may be empty angle brackets, `<>`. In this
case, the gufunc will accept an emtpy tuple `()` as an input, and all
the values in a non-empty tuple take part in broadcasting.

`None` is not accepted as the value of a shape-only parameter.
(Or perhaps `None` is treated as `()`?  This isn't really helpful;
what would be helpful is allowing gufunc parameters to have default
values.)


gufunc loop API
---------------

The shape remaining in the argument provided by a user after removing
the core shape as determined by the signature will be broadcast with
all the other arguments, just like the shape associated with an array
parameter.

The existing loop function signature for gufuncs is

    loop(char **args,
         npy_intp const *dimensions,
         npy_intp const *steps,
         void *data)

This does not change.  The only difference for a signature that
contains shape-only parameters is that those parameters do not show up
in the `args` array.  They *do* affect the `dimensions` and `steps`
arrays.

Consider the `linspace` example, with signature `(),(),<n> -> (n)`.
There is one core dimension, `n`.

In a call such as

    linspace(0.0, [1.0, 4.0], 5)

the `args` array passed to the loop function would have length 3
(two inputs, one output). The arrays passed to the loop function
would look like this:

    args = [  *  ,  *  ,  *  ]
              |     |     |
              |     |     +---> [---,---,---,---,---,
              |     |            ---,---,---,---,---]
              |     |
              |     +---> [1.0, 4.0]
              |
              +---> [0.0]

    dimensions[0] = 2  # Number of outer loops
    dimensions[1] = 5  # Inner loop `out` length (i.e. this is `n`)
    steps[0] = 0    # Outer loop `low` stride
    steps[1] = 8    # Outer loop `high` stride
    steps[2] = 8*5  # Outer loop `out` stride
    steps[3] = 8    # Inner stride for the `out` array

Examples
--------

* As already mentioned, a `linspace`-like gufunc with signature
  `(),(),<n> -> (n)`.  E.g.

      >>> linspace(0, [1, 10], 5)
      array([[ 0.  ,  0.25,  0.5 ,  0.75,  1.  ],
             [ 0.  ,  2.5 ,  5.  ,  7.5 , 10.  ]])

* A gufunc `convert_to_base(k, base, ndigits)` with signature
  `(),(),<n> -> (n)`.  The shape-only parameter specifies how
  many "digits" to include in the output.  In the following example,
  each value in the array `a` is converted to base 8, using 4 digits
  to represent the value.

      >>> a = np.array([3, 60, 129])
      >>> convert_to_base(a, 8, 4)
      array([[0, 0, 0, 3],
             [0, 0, 7, 4],
             [0, 2, 0, 1]])

  (An alternative version might reverse the order of the "digits".)

* `bincount(x, m)`, with gufunc signature `(n),<m> -> (m)`.  `m` is one
  more than the maximum value in the input array that should be counted.
  That is, the elements of the output array are the counts of the
  occurrences of the values [0, 1, 2, ..., m-1].

  For example, `bincount([0, 2, 8, 2, 2, 8, 3, 8, 8], 10)` returns
  `[1, 0, 3, 1, 0, 0, 0, 0, 4, 0]`.  `m` is like `minlength` of
  `numpy.bincount`, but in the gufunc version, the output always has
  size `m`; values in the input array that are greater than m-1 are ignored.

* Random variate generation.  In this case, the signature of the shape-only
  `size` parameter is `<>`, which means the user would pass `size=()` to
  generate a single variate, and `size=(n,)` (or `size=n`) to generate `n`
  variates.  Whatever is given, it must be consistent with the shapes
  of the other parameters for broadcasting.

      Function                              Signature
      ------------------------------------  ----------------------
      normal(loc, scale, size)              (),(),<> -> ()
      multinomial(n, pvals, size)           (),(m),<> -> (m)
      multivariate_normal(mean, cov, size)  (m),(m,m),<> -> (m)
      multivariate_hypergeometric(
          colors, nsample, size)            (m),(),<> -> (m)
      dirichlet(alpha, size)                (m),<> -> (m)
