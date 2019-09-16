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

This is from `numpy/core/include/numpy/__umath_generated.c` (the line breaks
have been added for clarity):

    static void * add_data[] = {
    	(void *)NULL, (void *)NULL, (void *)NULL, (void *)NULL, (void *)NULL,
    	(void *)NULL, (void *)NULL, (void *)NULL, (void *)NULL, (void *)NULL,
    	(void *)NULL, (void *)NULL, (void *)NULL, (void *)NULL, (void *)NULL,
    	(void *)NULL, (void *)NULL, (void *)NULL, (void *)NULL, (void *)NULL,
    	(void *)NULL, (void *)NULL
    };
    static char add_signatures[] = {
    	NPY_BOOL, NPY_BOOL, NPY_BOOL,
    	NPY_BYTE, NPY_BYTE, NPY_BYTE,
    	NPY_UBYTE, NPY_UBYTE, NPY_UBYTE,
    	NPY_SHORT, NPY_SHORT, NPY_SHORT,
    	NPY_USHORT, NPY_USHORT, NPY_USHORT,
    	NPY_INT, NPY_INT, NPY_INT,
    	NPY_UINT, NPY_UINT, NPY_UINT,
    	NPY_LONG, NPY_LONG, NPY_LONG,
    	NPY_ULONG, NPY_ULONG, NPY_ULONG,
    	NPY_LONGLONG, NPY_LONGLONG, NPY_LONGLONG,
    	NPY_ULONGLONG, NPY_ULONGLONG, NPY_ULONGLONG,
    	NPY_HALF, NPY_HALF, NPY_HALF,
    	NPY_FLOAT, NPY_FLOAT, NPY_FLOAT,
    	NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
    	NPY_LONGDOUBLE, NPY_LONGDOUBLE, NPY_LONGDOUBLE,
    	NPY_CFLOAT, NPY_CFLOAT, NPY_CFLOAT,
    	NPY_CDOUBLE, NPY_CDOUBLE, NPY_CDOUBLE,
    	NPY_CLONGDOUBLE, NPY_CLONGDOUBLE, NPY_CLONGDOUBLE,
    	NPY_DATETIME, NPY_TIMEDELTA, NPY_DATETIME,
    	NPY_TIMEDELTA, NPY_TIMEDELTA, NPY_TIMEDELTA,
    	NPY_TIMEDELTA, NPY_DATETIME, NPY_DATETIME,
    	NPY_OBJECT, NPY_OBJECT, NPY_OBJECT
    };

The array `add_data` contains the array stored in the `data` field of the C
ufunc object, and `add_signatures` holds the `types` fields.  `add_signatures`
is best interpreted as a two-dimensional array.  Each row has three values.
The first two gives the data types of the two input arguments, and the third
gives the data type of the output.

More to come...
