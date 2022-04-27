#  -*- coding: utf-8 -*-
"""
Discretised random variable functions

author: Simon Tindemans
email: s.h.tindemans@tudelft.nl

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
import math
import numpy as np

import gen_adequacy.util as util

class RandomVariable(object):
    """Represents a random variable, defined by a probability mass function on a regular array."""

    def __init__(self, resolution=1, base=0, probability_array=np.array([1.0])):
        """Creates a Random Variable object with probability masses at regular intervals.

        :param resolution: step size of the PMF
        :param base: location of the first probability mass in the array
        :param probability_array: optional PMF array. Values should be positive and sum to 1
        """
        self.base = base
        self.step = resolution
        self.probability_array = np.array(probability_array)
        # compute cumulative distribution function
        self.cdf_array = np.cumsum(self.probability_array)

        self.value_pool = None
        self.value_pool_left = 0
        self.value_pool_random_state = None

    def load_data(self, data_array):
        """Initialise the random variable from an empirical data set.

        This implementation creates a PMF array with bins that are aligned with zero.

        :param data_array: empirical data set (1D array)
        :return: none
        """

        # determine lower and upper bounds for the bins, to compute array size and base
        # TODO: generalise this procedure to distribute probability mass over nearby points (interpolation)
        left_bound = (math.floor(min(data_array)/self.step + 0.5) - 0.5) * self.step
        right_bound = (math.ceil(max(data_array)/self.step - 0.5) + 0.5) * self.step
        num_bins = int(np.round((right_bound - left_bound)/self.step))
        self.base = left_bound + 0.5 * self.step
        # allocate that experimental data across the bins
        prob_array, bins = np.histogram(
            data_array,
            bins=np.linspace(start=left_bound, stop=right_bound, num=(num_bins+1)),
            density=True
        )
        # enforce normalisation as probability mass function, and compute the CDF
        self.probability_array = prob_array * self.step
        self.cdf_array = np.cumsum(self.probability_array)

    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other):
        """Overloads the '+' operator.

        Independence is assumed between the variables being added."""
        if type(other) == RandomVariable:
            if self.step != other.step:
                raise NotImplementedError("Cannot add RandomVariable objects with different step sizes.")

            return RandomVariable(
                self.step,
                self.base + other.base,
                np.convolve(self.probability_array, other.probability_array)
            )
        else:
            return RandomVariable(
                self.step,
                self.base + other,
                self.probability_array
            )

    def __neg__(self):
        """Implements the inversion parameter '-'."""
        return RandomVariable(
            base=-(self.base + self.step * (len(self.probability_array) - 1)),
            resolution=self.step,
            probability_array=self.probability_array[::-1]
        )

    def __rsub__(self, other):
        return self.__sub__(other)

    def __sub__(self, other):
        """Overloads the '-' operator.

        Assumes independence between variables."""
        if type(other) == RandomVariable:
            if self.step != other.step:
                raise NotImplementedError("Cannot subtract RandomVariable objects with different step sizes.")

            return RandomVariable(
                self.step,
                self.base - other.base - self.step * (len(other.probability_array)-1),
                np.convolve(self.probability_array, other.probability_array[::-1])
            )
        else:
            return RandomVariable(
                self.step,
                self.base - other,
                self.probability_array
            )

    def __rmul__(self, multiplier):
        return self.__mul__(multiplier)

    def __mul__(self, multiplier):
        """Overloads the '*' operator.

        Only valid for integer values of 'multiplier'."""
        if type(multiplier) == int:
            if multiplier == 0:
                return 0
            elif multiplier > 0:
                new_array = np.zeros(multiplier * (len(self.probability_array)-1) + 1)
                new_array[::multiplier] = self.probability_array
                new_base = self.base * multiplier
            else:
                new_array = np.zeros(- multiplier * (len(self.probability_array)-1) + 1)
                new_array[::multiplier] = self.probability_array
                new_base = multiplier * (self.base + self.step * (len(self.probability_array) - 1))
            return RandomVariable(self.step, base=new_base, probability_array=new_array)
        else:
            raise NotImplementedError("Can only multiply RandomVariable objects by integer values.")

    def __iadd__(self, other):
        """Overloads the '+=' operator."""
        if type(other) == RandomVariable:
            if self.step == other.step:
                self.base += other.base
                self.probability_array = np.convolve(self.probability_array, other.probability_array)
                self.cdf_array = np.cumsum(self.probability_array)
            else:
                raise NotImplementedError("Cannot add RandomVariable objects with different step sizes.")
        else:
            self.base += other
        return self

    def __isub__(self, other):
        """Overloads the '-=' operator."""
        if type(other) == RandomVariable:
            if self.step == other.step:
                self.base -= other.base + self.step * (len(other.probability_array)-1)
                self.probability_array = np.convolve(self.probability_array, other.probability_array[::-1])
                self.cdf_array = np.cumsum(self.probability_array)
            else:
                raise NotImplementedError("Cannot subtract RandomVariable objects with different step sizes.")
        else:
            self.base -= other
        return self

    def __imul__(self, multiplier):
        """Overloads the '*=' operator.

        Is only valid for positive integer values of the multiplier."""
        if type(multiplier) == int:
            if multiplier == 0:
                self.base = 0.0
                self.probability_array = np.numarray([1.0])
            elif multiplier > 0:
                new_array = np.zeros(multiplier * (len(self.probability_array)-1) + 1)
                new_array[::multiplier] = self.probability_array
                self.probability_array = new_array
                self.cdf_array = np.cumsum(self.probability_array)
                self.base = self.base * multiplier
            else:
                new_array = np.zeros(- multiplier * (len(self.probability_array)-1) + 1)
                new_array[::multiplier] = self.probability_array
                self.probability_array - new_array
                self.cdf_array = np.cumsum(self.probability_array)
                self.base = multiplier * (self.base + self.step * (len(self.probability_array) - 1))
        else:
            raise NotImplementedError("Can only multiply RandomVariable objects by integer values.")
        return self

    def random_value(self, number_of_items=1, dither=True, rng=None):
        """
        Generate random values according to the object's distribution

        :param number_of_items: number of random values to return
        :param dither: optional dithering to generate a piecewise-linear PDF (default: True)
        :param rng: optional numpy random number generator or seed
        :return: numpy.ndarray() of random values
        """

        use_rng=util._rng_interpreter(rng)

        res = self.step

        values = use_rng.choice(self.x_array(), size=number_of_items, p=self.probability_array)
        if dither:
            return values + use_rng.triangular(-res, 0, res, size=number_of_items)
        else:
            return values


    def random_value_from_pool(self, pool_size, dither=True, reset=False, rng=None):
        """
        Efficient sampler implementation that returns a single value from a pool each time it is called

        :param pool_size: pool size to use when new samples are required
        :param dither: optional dithering to generate a piecewise-linear PDF (default: True)
        :param reset: force repopulation of the pool (default: False)
        :param rng: optional numpy random number generator or seed
        :return: random value

        NOTE: the implementation is unsafe, in the sense that a manual reset is required to update the pool values
        each time the system parameters change.
        """
        if reset or (self.value_pool_left <= 0):
            self.value_pool = self.random_value(number_of_items=pool_size, dither=dither, rng=rng)
            self.value_pool_left = pool_size

        self.value_pool_left = self.value_pool_left - 1
        return self.value_pool[self.value_pool_left]


    def min_max(self):
        """Returns the minimum and maximum values for which probability masses are stored."""
        return self.base, self.base + self.step * (len(self.probability_array)-1)

    def x_array(self):
        """Returns an array of x-values for which probability masses are stored."""
        return np.linspace(*self.min_max(), self.probability_array.size)

    def mean(self):
        """Returns the expectation cdf_value of the probability distribution."""
        return np.dot(self.x_array(), self.probability_array)

    def truncate(self, tolerance=1e-10):
        """Truncates the probability mass function."""
        sum_array = np.cumsum(self.probability_array)
        boundary_indices = np.searchsorted(sum_array, [tolerance, 1.0 - tolerance], side='left')

        new_array = np.copy(self.probability_array[boundary_indices[0]:boundary_indices[1]+1])
        if boundary_indices[0] > 0:
            new_array[0] += sum_array[boundary_indices[0]-1]
        if boundary_indices[1] < len(new_array) - 1:
            new_array[-1] += 1.0 - sum_array[boundary_indices[1]+2]

        self.base += boundary_indices[0] * self.step
        self.probability_array = new_array         # NOTE: copy can be avoided, but this is clearer.
        self.cdf_array = np.cumsum(self.probability_array)

    def quantile(self, q, side='left'):
        """Returns the value associated with the quantile q in [0,1]

        :param q: single cdf_value or list of quantiles to return
        :param side: rounding down ('left') or up ('right').
        :return: returns the type of q
        """

        indices = np.searchsorted(self.cdf_array, q, side=side)
        return self.base + indices * self.step

    def quantile_interpolate(self, q):
        """Returns the value associated with the quantile q in [0,1]

        :param q: single cdf_value or list of quantiles to return
        :return: returns the type of q
        """

        # TODO: create and use interpolated function object.
        # TODO: use numpy.digitize() to perform operation on array indices in one go
        index = np.searchsorted(self.cdf_array, q)  # default 'left' side
        if index != 0:
            delta = (q - self.cdf_array[index - 1]) / (self.cdf_array[index] - self.cdf_array[index - 1])
        else:
            delta = q / self.cdf_array[0]
        return self.base + self.step * (index - 1 + delta + 0.5)

    def cdf_value(self, x):
        """Returns the CDF value [0,1] for a given x

        :param x:
        :return: CDF value [0,1]
        """

        # TODO: vectorise this in x
        if x < self.base:
            return 0.0
        elif x > self.base + self.step * (len(self.cdf_array) - 1):
            return 1.0
        else:
            return self.cdf_array[math.floor((x - self.base)/self.step)]

    def cdf_value_interpolate(self, x):
        """Returns the CDF value [0,1] for a given x

        :param x:
        :return: CDF value [0,1]
        """

        # TODO: vectorise this in x
        x_unit = (x - self.base) / self.step + 0.5
        idx = math.floor(x_unit)
        if idx < 0:
            return 0.0
        if idx >= len(self.cdf_array):
            return 1.0
        else:
            return self.cdf_array[idx] - (idx + 1 - x_unit) * self.probability_array[idx]

    def change_resolution(self, new_resolution):
        """Returns a RandomVariable with new resolution, and distributes the PMF across its bins

        :param new_resolution: new resolution
        """

        # identify lower and upper bounds, and number of bins
        new_base = math.floor(self.base / new_resolution) * new_resolution
        current_pmf_right = self.base + (self.probability_array.size - 1)*self.step
        new_pmf_right = (math.ceil(current_pmf_right / new_resolution)) * new_resolution
        new_num_bins = 1 + int(np.round((new_pmf_right - new_base) / new_resolution))

        # construct probability array
        new_probability_array = np.zeros(new_num_bins)
        for idx, x_value in enumerate(self.x_array()):
            x_unit = (x_value - new_base) / new_resolution
            left_idx = math.floor(x_unit)
            right_idx = math.ceil(x_unit)
            distance_from_end = right_idx - x_unit
            new_probability_array[left_idx] += distance_from_end * self.probability_array[idx]
            new_probability_array[right_idx] += (1 - distance_from_end) * self.probability_array[idx]

        return RandomVariable(
            resolution=new_resolution,
            base=new_base,
            probability_array=new_probability_array
        )
