`numpy.dual`
============

In response to some work on
[improving the information about `numpy.linalg`](https://github.com/numpy/numpy/pull/14988),
and how it compares to `scipy.linalg`, Kevin Sheppard suggested
that documentation of the module `numpy.dual` should also be improved.
When I mentioned this suggestion in the community meeting on December 11,
it was suggested that we should probably deprecate `numpy.dual`.

I think quite a few current NumPy developers (myself included) are
unfamiliar with the history and purpose of this module, so I spent a
little time reading code and github issues.  The following are a few
notes about `numpy.dual`.  Corrections and clarifications to
anything written here would be appreciated!


What is numpy.dual?
-------------------

The documentation for numpy.dual is at [https://numpy.org/devdocs/reference/routines.dual.html](https://numpy.org/devdocs/reference/routines.dual.html).

It is a namespace that exposes several linear algebra and FFT functions
(and a lone special function, `i0`).  By default,
these names are just aliases for NumPy's implementation of these functions.
The module also provides the function `register_func(name, func)` (with
no docstring and no mention in the online docs [*grumble, grumble*]) that
allows a program to replace the function in the `numpy.dual` namespace with
`func`.

To quote the documentation of `numpy.dual` linked above:

> Scipy can be built to use accelerated or otherwise improved
> libraries for FFTs, linear algebra, and special functions.
> This module allows developers to transparently support these
> accelerated functions when scipy is available but still support
> users who have only installed NumPy.

That leaves the intended configuration and use of this module somewhat
mysterious.  Does NumPy automatically check for an installed SciPy library
and update the names in `numpy.dual` accordingly?  No (and that is probably
a good thing).  It is up to SciPy (or any other library that chooses to do
so) to call `register_func(name, func)` to replace the functions in
`numpy.dual`.  E.g.


```
    In [1]: import numpy.dual

    In [2]: numpy.dual.inv
    Out[2]: <function numpy.linalg.inv(a)>

    In [3]: import scipy.linalg

    In [4]: numpy.dual.inv
    Out[4]: <function scipy.linalg.basic.inv(a, overwrite_a=False, check_finite=True)>
```

That means a user of the linear algebra function `inv` would probably not
use `numpy.dual.inv`.  They would have have to import  `scipy.linalg` for
the name in `numpy.dual` to have been replaced.  If they have to import
`scipy.linalg` to make the switch, why wouldn't they just use `scipy.linalg`
directly in their code instead of `numpy.dual`?  So that, apparently, is not
the intent of the module.

Based on the three uses of `numpy.dual` within the NumPy code, it appears
that the intent of the module is to allow a library such as SciPy to swap
out NumPy's implementation with its own versions, *to be used by NumPy in
NumPy's internal code*.

There are three uses of `numpy.dual` in NumPy that I could find:

1. In the code that generates random variates from the multivariate normal
   distribution, one of `svd`, `eigh` or `cholesky` are used from `numpy.dual`.
2. In `matrixlib/defmatrix.py`, the `.I` property of the `matrix` class
   uses either `inv` or `pinv` from `numpy.dual` to compute its value.
3. The window function `numpy.kaiser` uses `numpy.dual.i0`.

So if, for example, SciPy's implementation of `eigh` happens to be better
(by some metric) than NumPy's, the `numpy.dual` module allows the multivariate
normal code to benefit from SciPy's implementation, without having to
monkeypatch `numpy.linalg`.


Issues on the NumPy github site related to `numpy.dual`
-------------------------------------------------------
* [numpy.dual.cholesky behaves differently than numpy.linalg.cholesky](https://github.com/numpy/numpy/issues/5649)
* [Stochastic behaviour even after setting random.seed()](https://github.com/numpy/numpy/issues/8041)
* [multivariate_normal not consistent between mkl and nomkl numpy](https://github.com/numpy/numpy/issues/13358)
* [Deterministic (up to numerical error) multivariate_normal sampling](https://github.com/numpy/numpy/issues/13386)
* [ENH: Multivariate normal speedups](https://github.com/numpy/numpy/pull/14197)
* [numpy.linalg.eig crashes on Mandriva/OpenSuse for some matrices (Trac #1211)](https://github.com/numpy/numpy/issues/1809)


How is `numpy.dual` used in SciPy?
----------------------------------

Currently (git SHA `47ffc1ef...`) SciPy registers the following names
with `numpy.dual`:

    linalg:
        norm, inv, svd, solve, det, eig, eigh, eigvals, eigvalsh,
        lstsq, cholesky, pinv (using `scipy.linalg.pinv2`)
    fft:
        fft, ifft, fftn, ifftn, fft2, ifft2

Note that `scipy.special` does not register `numpy.dual.i0`.


SciPy *uses* `numpy.dual` as follows:

* In `scip.odrpack`, the method `_cov2wt` in the class `RealData` does
  `from numpy.dual import inv`.
* In `scipy.optimize`, the function `leastsq` conditionally does
  `from numpy.dual import inv` (depends on the arguments and results).
* In `scipy.signal`, the file `signal/wavelets.py` does
  `from numpy.dual import eig`.


In the following issue, created 11 July 2019, the removal of the use of
`numpy.dual` from SciPy was proposed:

* [Remove uses of `numpy.dual`?](https://github.com/scipy/scipy/issues/10441)


Related issues
--------------

A `statsmodels` issue:

* [Use numpy.dual?](https://github.com/statsmodels/statsmodels/issues/1218)
