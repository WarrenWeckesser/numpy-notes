For the moment, this is just a mini-rant, followed by bunch of links...

NumPy refers to its default casting rules as "safe".  The idea of "safeness"
is that information is not lost in the cast.  But in fact, some of the default
casting allowed by NumPy is not safe.  NumPy will cast from int32 to float32,
or from int64 to float64, but these type conversions do not preserve the values
of all the possible input values.

This is not a new observation, and I suspect most NumPy developers will be
quick to point it out in any new conversation about casting where the word
"safe" comes up.  The word *safe* often appears in quotes in the NumPy
documentation, probably for this reason.  Still, it would be nice if NumPy
could change the word to something with weaker connotations than "safe",
perhaps something like "allowed". *End of mini-rant*.

*Documentation*

* [Casting rules](https://numpy.org/devdocs/reference/ufuncs.html#casting-rules)

*Relevant functions*

* [numpy.result_type](https://numpy.org/devdocs/reference/generated/numpy.result_type.html)
* [numpy.promote_types](https://numpy.org/devdocs/reference/generated/numpy.promote_types.html)
* [numpy.min_scalar_type](https://numpy.org/devdocs/reference/generated/numpy.min_scalar_type.html)
* [numpy.can_cast](https://numpy.org/devdocs/reference/generated/numpy.can_cast.html)
