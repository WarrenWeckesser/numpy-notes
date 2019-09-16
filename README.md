numpy-notes
===========

A collection of notes about NumPy.

This is not a really blog, nor a tutorial.  It is a place to record bits and
pieces of information about NumPy as I figure them out.  Initially the content
level and organization will be fairly random.  It will certainly be incomplete,
and there is a nonzero chance that parts of it may be wrong.  Also, these notes
will likely discuss NumPy implementation details that are not part of the
public API and may change in the future.  As I write this, I am working with
the pre-release development version of 1.18.

The C API for ufuncs
--------------------

The C API for NumPy's ufuncs is documented at

    https://numpy.org/devdocs/user/c-info.ufunc-tutorial.html

NumPy itself follows more than one convention for its own ufuncs.  Here I'll
look at one of the most fundamental ufuncs, `add`, and then at `hypot`, which
uses different convention for implementing the "inner loops".

An implementation of a ufunc includes one or more C functions that implement
the "inner loop".  Generally each function implements the same calculation but
for a different data type (e.g. `int32`, `int64`, `float`, `double`, etc.)  The
signature of these C functions are all the same.  They implement the ufunc's
calculation by interpreting the arguments differently.  Not surprisingly, then,
the code in each of these functions is almost the same.  Implementing all the
desired types "by hand" would be repetitive and tedious.  The code is written
in C, so we don't have the templating or generic data types available in higher
level languages to take care of the repetition.  Instead, for most (all?) of
its ufuncs, NumPy uses a Python script to generate the inner loops.

This scripts for generating NumPy's ufuncs are in

    numpy/numpy/core/code_generators

The `add` ufunc implements 22 inner loops.  This fact can be determined from
the public Python API by inspecting the `types` attribute of the ufunc:

    In [1]: import numpy as np
    In [2]: np.add.types
    Out[2]:
    ['??->?',
     'bb->b',
     'BB->B',
     'hh->h',
     'HH->H',
     'ii->i',
     'II->I',
     'll->l',
     'LL->L',
     'qq->q',
     'QQ->Q',
     'ee->e',
     'ff->f',
     'dd->d',
     'gg->g',
     'FF->F',
     'DD->D',
     'GG->G',
     'Mm->M',
     'mm->m',
     'mM->M',
     'OO->O']

    In [3]: len(np.add.types)
    Out[3]: 22

Let's look at the C implementation of those inner loops.

This is from `numpy/core/include/numpy/__umath_generated.c` (the lines have
been reformattd for clarity and some comments have been added):

    static PyUFuncGenericFunction add_functions[] = {
        // bool type
        BOOL_add,

        // integer types, signed and unsigned
        BYTE_add, UBYTE_add, SHORT_add, USHORT_add, INT_add, UINT_add,
        LONG_add, ULONG_add, LONGLONG_add, ULONGLONG_add,

        // floating point types
        HALF_add,   FLOAT_add,  DOUBLE_add, LONGDOUBLE_add,

        // complex types
        CFLOAT_add, CDOUBLE_add, CLONGDOUBLE_add,

        // datetime types
        DATETIME_Mm_M_add, TIMEDELTA_mm_m_add, DATETIME_mM_M_add,

        // object type (Why NULL? We'll get back to that.)
        NULL
    };

    static void * add_data[] = {
        (void *)NULL, (void *)NULL, (void *)NULL, (void *)NULL, (void *)NULL,
        (void *)NULL, (void *)NULL, (void *)NULL, (void *)NULL, (void *)NULL,
        (void *)NULL, (void *)NULL, (void *)NULL, (void *)NULL, (void *)NULL,
        (void *)NULL, (void *)NULL, (void *)NULL, (void *)NULL, (void *)NULL,
        (void *)NULL, (void *)NULL
    };

    static char add_signatures[] = {
        NPY_BOOL,          NPY_BOOL,           NPY_BOOL,
        NPY_BYTE,          NPY_BYTE,           NPY_BYTE,
        NPY_UBYTE,         NPY_UBYTE,          NPY_UBYTE,
        NPY_SHORT,         NPY_SHORT,          NPY_SHORT,
        NPY_USHORT,        NPY_USHORT,         NPY_USHORT,
        NPY_INT,           NPY_INT,            NPY_INT,
        NPY_UINT,          NPY_UINT,           NPY_UINT,
        NPY_LONG,          NPY_LONG,           NPY_LONG,
        NPY_ULONG,         NPY_ULONG,          NPY_ULONG,
        NPY_LONGLONG,      NPY_LONGLONG,       NPY_LONGLONG,
        NPY_ULONGLONG,     NPY_ULONGLONG,      NPY_ULONGLONG,
        NPY_HALF,          NPY_HALF,           NPY_HALF,
        NPY_FLOAT,         NPY_FLOAT,          NPY_FLOAT,
        NPY_DOUBLE,        NPY_DOUBLE,         NPY_DOUBLE,
        NPY_LONGDOUBLE,    NPY_LONGDOUBLE,     NPY_LONGDOUBLE,
        NPY_CFLOAT,        NPY_CFLOAT,         NPY_CFLOAT,
        NPY_CDOUBLE,       NPY_CDOUBLE,        NPY_CDOUBLE,
        NPY_CLONGDOUBLE,   NPY_CLONGDOUBLE,    NPY_CLONGDOUBLE,
        NPY_DATETIME,      NPY_TIMEDELTA,      NPY_DATETIME,
        NPY_TIMEDELTA,     NPY_TIMEDELTA,      NPY_TIMEDELTA,
        NPY_TIMEDELTA,     NPY_DATETIME,       NPY_DATETIME,
        NPY_OBJECT,        NPY_OBJECT,         NPY_OBJECT
    };

The first array shown, `add_functions`, is the array of "inner loop" functions.
The functions correspond to the short signatures in `numpy.add.types`.

The array `add_data` contains the array stored in the `data` field of the C
ufunc object, and `add_signatures` holds the `types` fields.  `add_signatures`
is best interpreted as a two-dimensional array.  Each row has three values.
The first two give the data types of the two input arguments, and the third
gives the data type of the output.  The rows also correspond one-to-one with
the list `numpy.add.types`.

The array `add_functions` is not necessarily the final version.  Some of those
functions might be replaced by implementation that take advantage of SIMD
instructions.  Further down in the file `__umath_generated.c`, we have

    static int
    InitOperators(PyObject *dictionary) {
        PyObject *f, *identity;
        
        _ones_like_functions[20] = PyUFunc_O_O;
        _ones_like_data[20] = (void *) Py_get_one;
        absolute_functions[19] = PyUFunc_O_O;
        absolute_data[19] = (void *) PyNumber_Absolute;
        #ifdef HAVE_ATTRIBUTE_TARGET_AVX2
        if (npy_cpu_supports("avx2")) {
            add_functions[1] = BYTE_add_avx2;
        }
        #endif
        
        #ifdef HAVE_ATTRIBUTE_TARGET_AVX2
        if (npy_cpu_supports("avx2")) {
            add_functions[2] = UBYTE_add_avx2;
        }
        #endif
        
        #ifdef HAVE_ATTRIBUTE_TARGET_AVX2
        if (npy_cpu_supports("avx2")) {
            add_functions[3] = SHORT_add_avx2;
        }
        #endif
        
        [...]
        
        #ifdef HAVE_ATTRIBUTE_TARGET_AVX2
        if (npy_cpu_supports("avx2")) {
            add_functions[10] = ULONGLONG_add_avx2;
        }
        #endif
        
        add_functions[21] = PyUFunc_OO_O;
        add_data[21] = (void *) PyNumber_Add;

So it is possible the some of the inner loop functions for the `add` ufunc
will use SIMD instructions.

Also note that at the end of `InitOperators`, the entries in `add_functions`
and `add_data` for the `object` data type are filled in.  The entry in `data`
is actually a function from the C API, `PyNumber_Add`.  We'll get back to
the significance of those functions later.

Let's look at one of the (non-SIMD) inner loop functions.  The generated
source code is in `numpy/core/src/umath/loops.c`.  Here is `UINT_add` from
that file:

    #if 1
    NPY_NO_EXPORT NPY_GCC_OPT_3  void
    UINT_add(char **args, npy_intp *dimensions, npy_intp *steps, void *NPY_UNUSED(func))
    {
        if (IS_BINARY_REDUCE) {
            BINARY_REDUCE_LOOP(npy_uint) {
                io1 += *(npy_uint *)ip2;
            }
            *((npy_uint *)iop1) = io1;
        }
        else {
            BINARY_LOOP_FAST(npy_uint, npy_uint, *out = in1 + in2);
        }
    }
    #endif

To make sense of that function, it will probably help to review the description
of the inner loop functions.  This is such a function, but the C loop code--which
is mostly boilerplate--is hidden inside a few macros.

For completeness, here's the definition of `UINT_add_avx2`, also from `loops.c`:

    #if HAVE_ATTRIBUTE_TARGET_AVX2
    NPY_NO_EXPORT NPY_GCC_OPT_3 NPY_GCC_TARGET_AVX2 void
    UINT_add_avx2(char **args, npy_intp *dimensions, npy_intp *steps, void *NPY_UNUSED(func))
    {
        if (IS_BINARY_REDUCE) {
            BINARY_REDUCE_LOOP(npy_uint) {
                io1 += *(npy_uint *)ip2;
            }
            *((npy_uint *)iop1) = io1;
        }
        else {
            BINARY_LOOP_FAST(npy_uint, npy_uint, *out = in1 + in2);
        }
    }
    #endif

The only difference is the addition of the macro `NPY_GCC_TARGET_AVX2`
to the qualifiers of the function definition.

More to come...
