Notes for NEP 34 (and possible alternatives)
============================================

*These notes are a work in progress.  There are still API issues
to be resolved.*

Currently, NumPy infers that an input must be an object array in
the following cases.

1. In nested sequences, some lengths are not compatible
   with a regular n-d array.  For example:

       np.array([1, [2, 3]])

2. A scalar was found that is not compatible with a known data
   type.  For example:

       np.array([1, None, 3])
       np.array([Fraction(1, 2), Fraction(3, 4)])

3. The scalars might be known, but are not castable to a
   common type.  For example:

       np.array([np.datetime64(2**31, 's'), np.uint8(99)])

(Maybe there are other cases.  I haven't read that part of the
NumPy code.)

My understanding is that with NEP 34, we're removing #1.
In the post-NEP 34 world, ragged nested sequences will not
be accepted by default.

That statement is not controversial.  What I propose next 
probably is controversial, because it amounts to an alternative
to NEP 34, and who wants to start this process all over again?
Well, here goes...

The NumPy code must infer the *shape* of the array, and it
must infer the *data type* of the elements in the array.
I think the shape should be inferred *only* from the lengths
of the nested sequences, and the data type should be
inferred *only* from the scalars found in those sequences.
That is, we have these two simple rules:

    nested sequence lengths => shape
    scalar values           => data type

The `dtype` argument should be used to override the
second rule *only*.  That means this example

    np.array([None, [None, None]], dtype=object)

should result in an error.  The input is a ragged nested
sequence, which we do not (by default) accept.  It doesn't
matter that `dtype=object` was given.  The `dtype` argument
only overrides the rule for how the scalar values are used
to infer the data type.

If `dtype` is only used to override the second rule, how
does one override the first rule that converts nested
sequence lengths to the shape?  Let's consider some use
cases:

1. Suppose I write

      x = np.array([1, [2, 3]])

   but I *want* that to result in a 1-d object array with
   length 2, containing the integer 1 and the list [2, 3].
   (In other words, I want the legacy behavior.)

2. More generally, I want to preserve the old behavior exactly.
   That is, create an array with as many dimensions as possible,
   and leave the remaining ragged inputs as lists.  For example,
   this input should result in a 2-d array:

       np.array([[[1],    [2, 3]],
                 [[3, 5], [6]   ]])


3. I want to create a 2-d array of Python lists.  The
   lengths of the lists can be arbitrary, and it is possible
   that all the lists will have the same length.  That is,
   in each of these cases, I want the result to be a 2-d
   object array:

       a = np.array([[[1], [2, 3]],
                     [[4], []    ]])
       b = np.array([[[1, 2], [3, 4]]
                      [5, 6], [7, 8]]])


How do we enable these cases?  Here are a couple alternatives:


### Add `ndim` parameter to `array`

In all three cases, we're interested in controlling
the number of dimensions of the result.  In a NumPy
array, that is given by the `ndim` attribute, so let's
add the argument `ndim` to the `array` function.  For
example 1, we'll use `ndim=1` and for example 3, we'll
use `ndim=2`.  For example 2, where we tell NumPy to 
create an array with as many dimensions as possible, we'll
adopt the convention that this behavior is specified as
`ndim=-1`.

The problem with `ndim` is that is has an unpleasant interaction
with the `dtype` parameter.  In some cases, given `ndim` makes
means that the `dtype` argument must be ignored (because the
result is forced to be `dtype=object`).  If a user actual gives, say,
`dtype=int` in such a case, should that be an error?  Or should that
mean "use int if possible, otherwise use objct"?

Questions:

* If `ndim` is given, does that imply that the data type *must* be
  object?  Wouldn't it be OK if `np.array([[1, 2], [3, 4]], ndim=2)`
  was an integer array?  Likewise for `np.array([[1, 2], [3, 4]], ndim=-1)`.
  If so, then is also makes 


### Create a new function for these cases.

I think this ends up looking like Eric Wieser's suggestion for the
`ragged_array_object` function in https://github.com/numpy/numpy/pull/14674:

> I'd consider `np.ragged_object_array(list_of_lists, depth=n)` or
> similar, to force a specific depth

In this case, we never allow `array()` to accept ragged input sequences.
For a user to handle such inputs, they must call a separate functon.

Questions:

* (Basically the same as the question above.) Does this function *always*
  return an object array?  If so, we can't use it to restore the legacy
  behavior.