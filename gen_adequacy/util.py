#  -*- coding: utf-8 -*-
"""
GenAdequacy utility functions
"""
def _rng_interpreter(rng):
    if rng is None:
        return np.random.random.__self__
    elif isinstance(rng, np.random.RandomState):
        return rng
    else:
        return np.random.default_rng(rng)