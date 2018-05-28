#!/usr/bin/env python
#  -*- coding: utf-8 -*-
"""
Class and supporting function to instantiate a single node generation adequacy problem.

author: Simon Tindemans
email: s.h.tindemans@tudelft.nl

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
import math
import numpy as np
from numba import jit

import gen_adequacy.randomvar as rv
from gen_adequacy.generator import Generator


# TODO: fix smoothing in epns()

class SingleNodeSystem(object):
    """Class representing generic single bus system with two-state generators, and profiles for wind power and load."""

    def __init__(self, gen_list, load_profile, wind_profile=None, resolution=1, load_offset=None):
        """Initialises the System object with a supplied list of generators and a load profile.

        :param gen_list: list of Generator objects
        :param load_profile: array with load profile
        :param wind_profile: array with wind profile. Assumed independent from load.
        :param resolution: resolution for random variables
        :param load_offset: structural load offset

        It is assumed that a system is instantiated only once and is not modified afterwards.
        """
        self.resolution = resolution
        self.gen_list = gen_list
        self.load_profile = load_profile
        self.load_offset = load_offset

        # NOTE: wind_profile may be None
        self.wind_profile = wind_profile

    @property
    def generation_rv(self):
        """Return a random variable representing the available generation of 'areas' RTS areas.

        :return: RandomVariable object that represents the generating capacity
        """
        if not hasattr(self, '_gen_rv'):
            self._gen_rv = rv.RandomVariable(resolution=self.resolution)
            for generator in self.gen_list:
                self._gen_rv += generator.rv(resolution=self.resolution)
        return self._gen_rv

    @property
    def load_rv(self):
        """Return a random variable representing the load duration curve of 'areas' RTS areas.

        :return: RandomVariable object that represents the system load
        """
        if not hasattr(self, '_ld_rv'):
            self._ld_rv = rv.RandomVariable(resolution=self.resolution)
            self._ld_rv.load_data(self.load_profile)
            if self.load_offset is not None:
                self._ld_rv += self.load_offset
        return self._ld_rv

    @property
    def wind_rv(self):
        """Return a random variable representing the gen duration curve of wind power.

        :return: RandomVariable object that represents the system load
        """
        if self.wind_profile is None:
            return None
        if not hasattr(self, '_w_rv'):
            self._w_rv = rv.RandomVariable(resolution=self.resolution)
            self._w_rv.load_data(self.wind_profile)
        return self._w_rv

    @property
    def margin_rv(self):
        """Return a random variable representing the generation margin.

        :return: RandomVariable object that represents the system margin
        """
        if not hasattr(self, '_m_rv'):
            if self.wind_profile is None:
                self._m_rv = self.generation_rv - self.load_rv
            else:
                self._m_rv = self.generation_rv + self.wind_rv - self.load_rv
        return self._m_rv

    def lolp(self, load_offset=0, interpolation=True):
        """
        Compute system LOLP (loss of load probability).

        :param load_offset: Additional load added to the system prior to computing LOLP
        :param interpolation: Specify whether linear interpolation of probability mass is used.
        :return: LOLP value in range [0,1]
        """
        if interpolation:
            return self.margin_rv.cdf_value_interpolate(load_offset)
        else:
            return self.margin_rv.cdf_value(load_offset - 1E-10)

    def lole(self, load_offset=0):
        """
        Compute system LOLE (loss of load expectation).

        :param load_offset: Additional load added to the system prior to computing LOLE
        :return: LOLE value

        The LOLE is computed by summing loss-of-load probabilities for each load level in self.load_profile. If an
        hourly annual load profile is used, this results in LOLE (hr/year); when a profile of daily peak loads is used,
        it returns LOLE (days/year).

        Note that LOLE is approx. LOLP * len(load_profile), but there is a small numerical difference due to the fact
        that LOLP is based on the margin distribution, which uses load levels that have been discretised onto a grid.
        As a result, the lole() result should be used for comparison with LOLE values in the literature.
        """
        total_load_offset = (0 if self.load_offset is None else self.load_offset) + load_offset
        if self.wind_rv is None:
            gw_rv = self.generation_rv - total_load_offset
        else:
            gw_rv = self.generation_rv + self.wind_rv - total_load_offset
        return np.sum([gw_rv.cdf_value(load_level - 1E-10) for load_level in self.load_profile])

    def epns(self, load_offset=0, interpolation=True):
        """
        Compute EPNS (expected power not supplied).

        :param load_offset: Additional load added to the system prior to computing EPNS
        :param interpolation: Specify whether linear interpolation of probability mass is used.
        :return: EPNS value
        """
        # TODO: implement interpolation
        mrv = self.margin_rv
        # determine index corresponding to target offset (on left)
        ceil_idx = max(0, min(math.ceil((load_offset - mrv.base)/mrv.step), len(mrv.cdf_array)))

        return mrv.step * mrv.cdf_array[0:ceil_idx].sum() + \
            mrv.cdf_array[ceil_idx] * (load_offset - mrv.base - ceil_idx * mrv.step)

    def compute_load_offset(self, lolp_target):
        """
        Determine additional load that must be added to the system to achieve the target LOLP.

        :param lolp_target: LOLP target in [0,1]
        :return: additional load level

        The load addition is implemented as a constant addition (no profile). It is effectively the inverse of
        lolp_target = lolp(load, interpolation=True). Interpolation is used to ensure a unique solution.
        """
        return self.margin_rv.quantile_interpolate(lolp_target)

    def margin_sample(self, number_of_values=1):
        """
        Generate random samples from the margin distribution.

        :param number_of_values: number of samples
        :return: single value or numpy array of values
        """
        mrv = self.margin_rv
        return mrv.random_value(number_of_items=number_of_values)

    @jit
    def generation_trace(self, num_steps=None, dt=1, random_start=True):
        """
        Sample a random generation trace for all generators in the system.

        :param num_steps: Number of steps
        :param dt: Time step
        :param random_start: Whether to create an independent sequence (True; default) or start from previous state (False)
        :return: Sequence of available capacity values
        """

        # infer length of trace
        if num_steps is None:
            num_steps = len(self.load_profile)
            if self.wind_profile is not None:
                num_steps = max(num_steps, len(self.wind_profile))

        trace = np.zeros(num_steps)
        for gen in self.gen_list:
            trace += gen.power_trace(num_steps=num_steps, random_start=random_start)

        return trace


# TODO: pass on most arguments to make less fragile
def autogen_system(
        load_profile=None, peak_load=None, wind_profile=None, resolution=10,
        gen_availability=0.90,  MTBF=2000, LOLH=3, base_unit=None, max_unit=None, gen_set=None, num_sets=None,
        apply_load_offset=False):
        """Initialises the generators and load curve of the RTS object and calls the System initialiser.

        :param load_profile: optional numpy.ndarray load profile of arbitrary length
        :param peak_load: if specified, peak load of the system, in same units as generator capacities
        :param wind_profile: optinoal numpy.ndarray wind profile of arbitrary length. Assumed independent from load.
        :param resolution: resolution for the random variables
        :param gen_availability: availability of each generator
        :param MTBF: generator MTBF (not required for steady state analysis)
        :param LOLH: system risk standard (expected loss-of-load hours)
        :param base_unit: size of smallest unit in the system
        :param max_unit: upper bound for maximum unit size. Actual unit sizes are generated in powers of two from base_unit
        :param gen_set: array of generator sizes
        :param num_sets: number of generator sets to use ('None' for automatic optimisation)
        :param apply_load_offset: adjust load profile to hit LOLH target exactly
        """

        assert (max_unit is None and base_unit is None) or (gen_set is None)

        if gen_set is not None:
            gen_types = gen_set
        else:
            assert (max_unit >= base_unit), "autogen_system: max_unit should exceed base_unit"

            # Create a 'basic' generator set, starting from base_unit (in MW), increasing by factors of two until 'max_unit'
            gen_types = []
            current_size = base_unit
            while current_size <= max_unit:
                gen_types.append(current_size)
                current_size *= 2
        print('Generator set:', gen_types)


        # Check whether a load_profile has been supplied. If not, use a constant peak_load
        if load_profile is None:
            assert peak_load is not None
            load_profile = np.array([1.])
        else:
            assert isinstance(load_profile, (np.ndarray, np.generic))

        # Check whether peak_load has been supplied. If so, rescale load profile to target peak load
        if peak_load is not None:
            load_profile *= peak_load / load_profile.max()

        # Check whether the number of generator sets must be optimised.
        if num_sets is None:

            print("Optimising number of generator sets to LOLE target of", LOLH, "hours")

            # initialise generation portfolio
            gen_list = [
                Generator(unit_count=1, unit_capacity=power, unit_availability=gen_availability, unit_mtbf=MTBF)
                for (idx, power) in enumerate(gen_types)]
            num_sets = 0

            # create temporary load and gen rv's for iteration
            net_ld_rv = rv.RandomVariable(resolution=resolution)
            net_ld_rv.load_data(load_profile)

            if wind_profile is not None:
                wind_rv = rv.RandomVariable(resolution=resolution)
                wind_rv.load_data(wind_profile)
                net_ld_rv -= wind_rv

            gen_rv = rv.RandomVariable(resolution=resolution)

            # add sets of generators one by one, until the risk is less than the threshold
            while True:
                for generator in gen_list:
                    gen_rv += generator.rv(resolution=resolution)

                gen_rv.truncate()
                num_sets += 1

                if (gen_rv - net_ld_rv).cdf_value(-0.00001) * 8760 < LOLH:
                    break

            # Report number of generator sets and LOLH value
            print("{num} generator sets. Base LOLE={lole:.3g} h; determined load offset of {load:.4g} MW to reach LOLE target.\n".
                  format(num=num_sets,
                         lole=(gen_rv - net_ld_rv).cdf_value_interpolate(0.0) * 8760,
                         load=(gen_rv - net_ld_rv).quantile_interpolate(LOLH / 8760)
                         )
                  )

            if apply_load_offset is True:
                load_offset = (gen_rv - net_ld_rv).quantile_interpolate(LOLH/8760)
            else:
                load_offset = None

        else:
            load_offset = None

        # Generate generator list
        gen_list = [
            Generator(unit_count=num_sets, unit_capacity=power, unit_availability=gen_availability, unit_mtbf=MTBF)
            for (idx, power) in enumerate(gen_types)]

        # initialise parent object
        return SingleNodeSystem(gen_list=gen_list, load_profile=load_profile,
                                wind_profile=wind_profile, resolution=resolution,
                                load_offset=load_offset)


if __name__ == "__main__":
    print("Starting direct execution\n")

    system = autogen_system(load_profile=None, peak_load=10000, wind_profile=None, resolution=10,
                            gen_availability=0.90,  MTBF=2000, LOLH=3, base_unit=10, max_unit=1000, gen_set=None,
                            apply_load_offset=True)

    print("Properties after adjustment:")
    print("LOLE: {:.4f} h".format(8760 * system.lolp()))
    print("EPNS: {:.4g} h".format(system.epns()))
