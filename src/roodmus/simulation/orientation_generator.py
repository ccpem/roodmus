"""Class to generate orientations for a given molecule
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


class orientation_generator(object):
    def __init__(self):
        pass

    @classmethod
    def generate_inplane(self, n: int = 1):
        poses = []
        for i in range(n):
            phi = 0
            theta = 0
            psi = np.random.uniform(0, 2 * np.pi)
            poses.append((phi, theta, psi))
        return poses

    @classmethod
    def generate_discrete_tilt(
        self, n: int = 1, k: int = 3, save_to_file: bool = False
    ):
        # this generator creates particles with a
        # tilt angle sampled from K discrete values
        # the tilt angle is sampled from a uniform distribution
        # the azimuthal angle and the rotation angle are sampled
        # from a uniform distribution as in the random case

        # the euler angles have to be converted to axis-angle
        # representation for the simulation

        allowed_tilt_angles = np.linspace(0, np.pi, k, endpoint=False)

        poses = []
        for i in range(n):
            phi = 0
            theta = np.random.choice(allowed_tilt_angles, 1)
            psi = 0
            poses.append([phi, float(theta[0]), psi])

        if save_to_file:
            if not os.path.exists("angles.csv"):
                with open("angles.csv", "w") as f:
                    f.write("phi,theta,psi\n")

            with open("angles.csv", "a") as f:
                for pose in poses:
                    f.write(",".join([str(e) for e in pose]) + "\n")
        return poses
