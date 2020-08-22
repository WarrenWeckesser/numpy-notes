Ufunc Type Resolvers
====================

(This document is still a work in progress.)

The ufunc object has a component called `type_resolver`,
a C function with specific arguments and return value.
These notes attempt to explain the purpose of this function,
and explain the inputs and outputs of a TypeResolver function.


What is a type resolver?
------------------------

To implement a ufunc, the developer provides one or more *loop functions*
in C that compute the function for a specific set of input and output
C data types.  Depending on the expected behavior of the ufunc, one
might have a separate loop for each type of integer (e.g. `int8_t`,
`uint8_t`, `int16_t`, etc.), along with `float`, `double`, `long double`,
etc.  Or one might implement just one loop, say for `double`.

However many loops are implemented, it is still likely that a user of
the ufunc will try to call it with inputs or outputs that have data
types that do not exactly match the implemented loops.  What should
the ufunc calling code do with such input?  Part of the answer to
that question is provided by the type resolver.  The ufunc code will
have to decide if it is possible to cast the user's data to a data type
for which a loop is defined, and raise an error if it is not.


What are the inputs to a TypeResolver function?
-----------------------------------------------

For example,

```
    NPY_NO_EXPORT int
    PyUFunc_SubtractionTypeResolver(PyUFuncObject *ufunc,
                                    NPY_CASTING casting,
                                    PyArrayObject **operands,
                                    PyObject *type_tup,
                                    PyArray_Descr **out_dtypes)
    {
       ...
    }
```

The type `NPY_CASTING` is an enum defined in
`numpy/core/include/numpy/ndarraytypes.h`:

```
/* For specifying allowed casting in operations which support it */
typedef enum {
        /* Only allow identical types */
        NPY_NO_CASTING=0,
        /* Allow identical and byte swapped types */
        NPY_EQUIV_CASTING=1,
        /* Only allow safe casts */
        NPY_SAFE_CASTING=2,
        /* Allow safe casts or casts within the same kind */
        NPY_SAME_KIND_CASTING=3,
        /* Allow any casts */
        NPY_UNSAFE_CASTING=4
} NPY_CASTING;
```

Question: How does the code know when a casting is "safe"? Presumably
this is the casting table and the "can cast" information.  How does
casting take place "within the same kind" for types that wouldn't
necessarily be considered "safe" for casting?  (See below for some
brief notes about `kind`.)  Same question for when the types are not
within the same kind: if the casting is set to NPY_UNSAFE_CASTING,
how does the code actually *do* the casting?


*Parameters*

* `PyUFuncObject *ufunc` - The ufunc object.

* `NPY_CASTING casting` - The casting rule to follow; see the enum above.

* `PyArrayObject **operands` - The operands (input and output) to the ufunc.

* `PyObject *type_tup` - TO DO: find out what this is.

* `PyArray_Descr **out_dtypes` - This is the output of the resolver.



What does the function do with them?
------------------------------------


The return value is an int.  What is it?
----------------------------------------

It is a status result.  From the comments about `PyUFunc_DefaultTypeResolver`
in `numpy/core/src/umath/ufunc_type_resolution.c`:

```
 * Returns 0 on success, -1 on error.
```


What is the default type resolver for a ufunc created with PyUFunc_FromFuncAndData?
-----------------------------------------------------------------------------------

This is from `PyUFunc_FromFuncAndDataAndSignatureAndIdentity` in the file
`numpy/core/src/umath/ufunc_object.c`:

```
    /* Type resolution and inner loop selection functions */
    ufunc->type_resolver = &PyUFunc_DefaultTypeResolver;
    ufunc->legacy_inner_loop_selector = &PyUFunc_DefaultLegacyInnerLoopSelector;
    ufunc->masked_inner_loop_selector = &PyUFunc_DefaultMaskedInnerLoopSelector;
```

which shows the default type resolver and inner loop selectors.

This is the complete `PyUFunc_DefaultTypeResolver` from
`numpy/core/src/umath/ufunc_type_resolution.c`:

```
PyUFunc_DefaultTypeResolver(PyUFuncObject *ufunc,
                            NPY_CASTING casting,
                            PyArrayObject **operands,
                            PyObject *type_tup,
                            PyArray_Descr **out_dtypes)
{
    int i, nop = ufunc->nin + ufunc->nout;
    int retval = 0, any_object = 0;
    NPY_CASTING input_casting;

    for (i = 0; i < nop; ++i) {
        if (operands[i] != NULL &&
                PyTypeNum_ISOBJECT(PyArray_DESCR(operands[i])->type_num)) {
            any_object = 1;
            break;
        }
    }

    /*
     * Decide the casting rules for inputs and outputs.  We want
     * NPY_SAFE_CASTING or stricter, so that the loop selection code
     * doesn't choose an integer loop for float inputs, or a float32
     * loop for float64 inputs.
     */
    input_casting = (casting > NPY_SAFE_CASTING) ? NPY_SAFE_CASTING : casting;

    if (type_tup == NULL) {
        /* Find the best ufunc inner loop, and fill in the dtypes */
        retval = linear_search_type_resolver(ufunc, operands,
                        input_casting, casting, any_object,
                        out_dtypes);
    } else {
        /* Find the specified ufunc inner loop, and fill in the dtypes */
        retval = type_tuple_type_resolver(ufunc, type_tup,
                        operands, casting, any_object, out_dtypes);
    }

    return retval;
}
```

It calls out to `linear_search_type_resolver` or `type_tuple_type_resolver`
to do the real work.

Here's the signature for `linear_search_type_resolver` from
`numpy/core/src/umath/ufunc_type_resolution.c`:

```
NPY_NO_EXPORT int
linear_search_type_resolver(PyUFuncObject *self,
                            PyArrayObject **op,
                            NPY_CASTING input_casting,
                            NPY_CASTING output_casting,
                            int any_object,
                            PyArray_Descr **out_dtype)
```

Examples
========

An example of a ufunc that is implemented with just one C
loop is `ufunclab.logfactorial`.  The data type of the
input is `int64_t`, and the output has data type `double`.
The implementation uses the default type resolver.  The
ufunc casting/dispatch automatically handles data types
that can be safely cast to `int64_t`, which means that
users can call this function with most of the builtin NumPy
integer data types.  For example,

```
In [35]: from ufunclab import logfactorial

In [36]: logfactorial(np.int8(25))
Out[36]: 58.00360522298052

In [37]: a = np.array([10, 100, 1000], dtype=np.uint32)

In [38]: logfactorial(a)
Out[38]: array([  15.10441257,  363.73937556, 5912.12817849])
```

Not *all* the builtin integer types are handled.  The
implemented loop is for `int64_t`, which is a signed integer
type, and there is no safe casting from the corresponding
unsigned 64 bit integer (`uint64_t`) to `int64_t`, so
attempting to pass an array with dtype `np.uint64` will
raise an error:

```
In [39]: u = np.array([0, 1024], dtype=np.uint64)

In [40]: logfactorial(u)
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-40-89f97d2d439d> in <module>
----> 1 logfactorial(u)

TypeError: ufunc 'logfactorial' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''
```

Non-integer types are also not accepted, e.g.:

```
In [7]: logfactorial(np.array([1.0, 2.0, 4.0]))                                                                                                                                                
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-7-21afcddecc1e> in <module>
----> 1 logfactorial(np.array([1.0, 2.0, 4.0]))

TypeError: ufunc 'logfactorial' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''

In [8]: logfactorial(np.timedelta64(123456789, 's'))                                                                                                                                           
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-8-df670e485295> in <module>
----> 1 logfactorial(np.timedelta64(123456789, 's'))

TypeError: ufunc 'logfactorial' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''
```

-----

The `kind` of a dtype
=======================
In Python, the `kind` of a NumPy dtype is available as the `kind`
attribute; see

    https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html

For example, all the signed integer dtypes have `kind` `'i'`:

    >>> a = np.array([-100, 0, 100, 200, 500, 8000], dtype=np.int32)
    >>> a.dtype
    dtype('int32')
    >>> a.dtype.kind
    'i'

By default, the `astype` method allows unsafe casting, so we can do:

    >>> a.astype(np.int8)
    array([-100,    0,  100,  -56,  -12,   64], dtype=int8)

If we specify safe casting, we get an error:

    >>> a.astype(np.int8, casting="safe")
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: Cannot cast array data from dtype('int32') to dtype('int8') according to the rule 'safe'

Because the dtypes associated with `int8` and `int32` have the same
`kind`, we can (unsafely!) cast `a` to `int8`:

    >>> a.astype(np.int8, casting="same_kind")
    array([-100,    0,  100,  -56,  -12,   64], dtype=int8)
