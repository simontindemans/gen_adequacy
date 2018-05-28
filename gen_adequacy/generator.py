#  -*- coding: utf-8 -*-
"""
Generator class for generation adequacy assessment

author: Simon Tindemans
email: s.h.tindemans@tudelft.nl

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
import math
import numpy as np
from numba import jit
import random

import gen_adequacy.randomvar as rv


class Generator(object):
    """Class that represents a single generator or set of identical generators as a linear Markov chain

    """

    def __init__(self, unit_capacity, unit_availability, unit_mtbf, unit_count=1):
        """Initialises a Generator object, consisting of identical units.

        :param unit_capacity: capacity of each unit
        :param unit_availability: availability of each unit [0,1]
        :param unit_mtbf: mean time between failure, in the units of the simulation (usually hours)
        :param unit_count: number of identical units
        """
        assert (unit_capacity > 0), "Generator capacities must be larger than zero."
        assert (0 <= unit_availability <= 1), "Generator availability must be between zero and one."
        assert (unit_mtbf > 0), "Generator MTBF must be positive."
        assert (unit_count >= 1), "Generating unit count must be positive."

        self.unit_capacity = unit_capacity
        self.unit_count = unit_count
        self.unit_availability = unit_availability
        self.unit_MTBF = unit_mtbf

        # Compute the number of units in the 'up' state. This is used for sequential series generation.
        self.up_count = np.random.binomial(unit_count, unit_availability)
        self.unit_fail_rate = 1/(unit_availability * unit_mtbf)
        self.unit_repair_rate = 1/((1 - unit_availability) * unit_mtbf)

    def available_capacity(self):
        """Return sum of capacities of available units.

        :return: sum of capacities of available units.
        """
        return self.up_count * self.unit_capacity

    @jit
    def power_trace(self, num_steps=1, dt=1, random_start=True):
        """
        Return time series of available capacities of the aggregate units.

        :param num_steps: Number of time steps in the series
        :param dt: Size of each time step
        :param random_start: Option to create an independent sequence (True:default) or start from self.up_count (False)
        :return: numpy array of length num_steps and dtype float
        """

        initial_up_count = \
            np.random.binomial(self.unit_count, self.unit_availability) if random_start else self.up_count

        repair_prob = self.unit_repair_rate * dt
        fail_prob = self.unit_fail_rate * dt
        assert repair_prob < 1.0
        assert fail_prob < 1.0

        repair_factor = 1./math.log(1.0-repair_prob)
        fail_factor = 1./math.log(1.0-fail_prob)

        sum_factor = repair_factor + fail_factor

        adjust_trace = np.zeros(num_steps + 1, dtype=int)
        adjust_trace[0] = initial_up_count

        for i in range(self.unit_count):
            if i < initial_up_count:
                state = 1
                change_factor = fail_factor
            else:
                state = -1
                change_factor = repair_factor
            t = 0
            while True:
                t += math.ceil(change_factor * math.log(1.0 - random.random()))
                if t > num_steps:
                    break
                adjust_trace[t] -= state
                state = -state
                change_factor = sum_factor - change_factor

        trace = np.cumsum(adjust_trace)
        self.up_count = trace[-1]

        return self.unit_capacity * trace[:-1].astype(float)

    def rv(self, resolution=1):
        """Returns a random variable representing the available capacity of the generator set.

        :param resolution: desired resolution of the RandomVariable object
        :return: RandomVariable object corresponding to the generator(s) output

        This uses a simple iterative definition for multi-unit generators; this could easily be improved
        (e.g. using a binomial distribution), but is rarely on the critical path.

        If resolution is not a divisor of unit_capacity, the capacity is divided proportionally over
        neighbouring bins, depending on resolution
        """

        # Note that ceil_idx and floor_idx are either identical (if resolution matches) or one apart
        ceil_idx = math.ceil(self.unit_capacity / resolution)
        floor_idx = math.floor(self.unit_capacity / resolution)
        probability_array = np.zeros(int(ceil_idx + 1))

        probability_array[0] = 1.0 - self.unit_availability

        # allocate probability over two identical or neighbouring bins
        distance_to_end = ceil_idx - self.unit_capacity / resolution
        probability_array[floor_idx] += distance_to_end * self.unit_availability
        probability_array[ceil_idx] += (1-distance_to_end) * self.unit_availability

        # define a single unit rv
        unit_rv = rv.RandomVariable(resolution=resolution, base=0, probability_array=probability_array)
        # iterative convolution to determine aggregate rv
        result_rv = rv.RandomVariable(resolution=resolution)
        for i in range(self.unit_count):
            result_rv += unit_rv
        return result_rv
