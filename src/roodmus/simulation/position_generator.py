"""Class to generate positions for a given molecule
based on some specifications.

Delft University of Technology (TU Delft) hereby disclaims
all copyright interest in the program “Roodmus” written by
the Author(s).
Copyright (C) 2023  Joel Greer(UKRI), Tom Burnley (UKRI),
Maarten Joosten (TU Delft), Arjen Jakobi (TU Delft)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""


import os
import numpy as np

from parakeet.sample.distribute import (
    distribute_particles_uniformly,
    CuboidVolume,
)


class position_generator(object):
    def __init(self):
        pass


    @classmethod
    def generate_random(self, n: int = 1):
        """
        This is the main random generator for positions.
        It is a wrapper to the Parakeet function 'distribution_particles_uniformly'
        """


