#!/usr/bin/env python
#  -*- coding: utf-8 -*-
"""
IEEE RTS Generation System

author: Simon Tindemans
email: s.h.tindemans@tudelft.nl

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
import numpy as np

from gen_adequacy.system import SingleNodeSystem
from gen_adequacy.generator import Generator

# NOTE: the __init__ file imports ieee_rts directly into the gen_adequacy namespace


def ieee_rts(resolution=1, areas=1):
        """Initialises the generators and load curve of the RTS object and calls the GenLoadSystem initialiser.

        :param resolution: desired resolution for any random variables (generation or load)
        :param areas: number of RTS areas
        """

        # Initialise generation portfolio
        gen_list = [
            Generator(unit_count=5 * areas, unit_capacity=12, unit_availability=0.98, unit_mtbf=3000),
            Generator(unit_count=4 * areas, unit_capacity=20, unit_availability=0.9, unit_mtbf=500),
            Generator(unit_count=6 * areas, unit_capacity=50, unit_availability=0.99, unit_mtbf=2000),
            Generator(unit_count=4 * areas, unit_capacity=76, unit_availability=0.98, unit_mtbf=2000),
            Generator(unit_count=3 * areas, unit_capacity=100, unit_availability=0.96, unit_mtbf=1250),
            Generator(unit_count=4 * areas, unit_capacity=155, unit_availability=0.96, unit_mtbf=1000),
            Generator(unit_count=3 * areas, unit_capacity=197, unit_availability=0.95, unit_mtbf=1000),
            Generator(unit_count=1 * areas, unit_capacity=350, unit_availability=0.92, unit_mtbf=1250),
            Generator(unit_count=2 * areas, unit_capacity=400, unit_availability=0.88, unit_mtbf=1250)
        ]

        weekly_load_factors = np.array([86.2, 90.0, 87.8, 83.4, 88.0, 84.1, 83.2, 80.6, 74.0, 73.7,
                                        71.5, 72.7, 70.4, 75.0, 72.1, 80.0, 75.4, 83.7, 87.0, 88.0,
                                        85.6, 81.1, 90.0, 88.7, 89.6, 86.1, 75.5, 81.6, 80.1, 88.0,
                                        72.2, 77.6, 80.0, 72.9, 72.6, 70.5, 78.0, 69.5, 72.4, 72.4,
                                        74.3, 74.4, 80.0, 88.1, 88.5, 90.9, 94.0, 89.0, 94.2, 97.0,
                                       100.0, 95.2]) / 100
        daily_load_factors = np.array([93, 100, 98, 96, 94, 77, 75]) / 100
        winter_weekday_hourly = np.array([67, 63, 60, 59, 59, 60,
                                          74, 86, 95, 96, 96, 95,
                                          95, 95, 93, 94, 99, 100,
                                         100, 96, 91, 83, 73, 63]) / 100
        winter_weekend_hourly = np.array([78, 72, 68, 66, 64, 65,
                                          66, 70, 80, 88, 90, 91,
                                          90, 88, 87, 87, 91, 100,
                                          99, 97, 94, 92, 87, 81]) / 100
        summer_weekday_hourly = np.array([64, 60, 58, 56, 56, 58,
                                          64, 76, 87, 95, 99, 100,
                                          99, 100, 100, 97, 96, 96,
                                          93, 92, 92, 93, 87, 72]) / 100
        summer_weekend_hourly = np.array([74, 70, 66, 65, 64, 62,
                                          62, 66, 81, 86, 91, 93,
                                          93, 92, 91, 91, 92, 94,
                                          95, 95, 100, 93, 88, 80]) / 100
        spring_fall_weekday_hourly = np.array([63, 62, 60, 58, 59, 65,
                                               72, 85, 95, 99, 100, 99,
                                               93, 92, 90, 88, 90, 92,
                                               96, 98, 96, 90, 80, 70]) / 100
        spring_fall_weekend_hourly = np.array([75, 73, 69, 66, 65, 65,
                                               68, 74, 83, 89, 92, 94,
                                               91, 90, 90, 86, 85, 88,
                                               92, 100, 97, 95, 90, 85]) / 100

        winter_week = np.concatenate((np.tile(winter_weekday_hourly, (5,)), np.tile(winter_weekend_hourly, (2,)))) \
            * np.repeat(daily_load_factors, 24)
        summer_week = np.concatenate((np.tile(summer_weekday_hourly, (5,)), np.tile(summer_weekend_hourly, (2,)))) \
            * np.repeat(daily_load_factors, 24)
        spring_fall_week = np.concatenate((np.tile(spring_fall_weekday_hourly, (5,)),
                                           np.tile(spring_fall_weekend_hourly, (2,)))) \
            * np.repeat(daily_load_factors, 24)

        load_profile = areas * 2850 * np.repeat(weekly_load_factors, 7*24) * np.concatenate((
            np.tile(winter_week, (8,)),
            np.tile(spring_fall_week, (9,)),
            np.tile(summer_week, (13,)),
            np.tile(spring_fall_week, (13,)),
            np.tile(winter_week, (9,))
        ))

        # initialise parent object
        return SingleNodeSystem(gen_list=gen_list, load_profile=load_profile, resolution=resolution)


if __name__ == "__main__":
    print("Starting direct execution")

    rts1 = ieee_rts(areas=1, resolution=1)
    rts3 = ieee_rts(areas=3)

    print("\nIEEE RTS 1 Area:")
    print("LOLP [discrete levels]       : {:.6g}".format(rts1.lolp(interpolation=False)))
    print("LOLP [linear interpolation]  : {:.6g}".format(rts1.lolp(interpolation=True)))
    print("EENS [discrete levels]       : {:.6g} MWh".format(rts1.epns(interpolation=False) * 8736))
    print("EENS [linear interpolation]  : {:.6g} MWh".format(rts1.epns(interpolation=True) * 8736))
    print("LOLE                         : {:.6g} h".format(rts1.lole()))
    print("Load offset for LOLP==3/8736 : {:.6g} MW".format(rts1.compute_load_offset(lolp_target=3/8736)))

    print("\nIEEE RTS 3 Area (single node representation):")
    print("LOLP                         : {:.6g}".format(rts3.lolp()))
    print("EENS                         : {:.6g} MWh".format(rts3.epns() * 8736))
    print("LOLE                         : {:.6g} h".format(rts3.lole()))
    print("Load offset for LOLP==3/8736 : {:.6g} MW".format(rts3.compute_load_offset(lolp_target=3/8736)))
