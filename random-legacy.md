Some notes on the change of the integer type in 1.17 random generator code
--------------------------------------------------------------------------

The new Generator class implemented in 1.17 uses int64_t for C integers.
The legacy RandomState class uses C long integers, which are typically either
32 or 64 bits, depending on the platorm or compiler.

The implementations of the distributions in the new code reuses much of the
old code.  To make this work, the macro RAND_INT_TYPE specifies the integer
type.  When the legacy code in `mtrand.pyx` is compiled, RAND_INT_TYPE is
`long`.  When it is compiled for the new Generator, RAND_INT_TYPE is `int64_t`.

From `numpy/random/src/distributions/distributions.h`:

```c
/*
 * RAND_INT_TYPE is used to share integer generators with RandomState which
 * used long in place of int64_t. If changing a distribution that uses
 * RAND_INT_TYPE, then the original unmodified copy must be retained for
 * use in RandomState by copying to the legacy distributions source file.
 */
#ifdef NP_RANDOM_LEGACY
#define RAND_INT_TYPE long
#define RAND_INT_MAX LONG_MAX
#else
#define RAND_INT_TYPE int64_t
#define RAND_INT_MAX INT64_MAX
#endif
```

`RandomState` refers to the legacy implementation class defined in
`numpy/random/mtrand.pyx`.

In the `configuration` function in `numpy/random/setup.py`, we have

```python
    # Use legacy integer variable sizes
    LEGACY_DEFS = [('NP_RANDOM_LEGACY', '1')]
```

which is used later in the configuration of the `mtrand` module--note
that `LEGACY_DEFS` is included in the `define_macros` argument:

```python
    config.add_extension('mtrand',
                         # mtrand does not depend on random_hypergeometric.c.
                         sources=['mtrand.c',
                                  'src/legacy/legacy-distributions.c',
                                  'src/distributions/logfactorial.c',
                                  'src/distributions/distributions.c'],
                         include_dirs=['.', 'src', 'src/legacy'],
                         libraries=EXTRA_LIBRARIES,
                         extra_compile_args=EXTRA_COMPILE_ARGS,
                         extra_link_args=EXTRA_LINK_ARGS,
                         depends=['mtrand.pyx'],
                         define_macros=defs + LEGACY_DEFS,
                         )
```

