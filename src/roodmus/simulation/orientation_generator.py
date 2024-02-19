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
from scipy.spatial.transform import Rotation as R


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
    

    @classmethod
    def generate_preferred_orientation(
        self, n: int = 1, maxtilt: float = 30, maxtries: int = 1000
    ):
        """
        This generator samples random orientations from SO(3), converts them
        to ZYZ (intrinsic) Euler angles and then checks if the tilt angle is
        within a certain range. If not, the orientation is discarded and a new
        one is sampled. This is repeated until n orientations are found or
        maxtries is reached. In the latter case, a warning is raised and the 
        orientation is set to the last one found.

        Parameters
        ----------
        n : int
            Number of orientations to generate
        maxtilt : float
            Maximum tilt angle in degrees
        maxtries : int
            Maximum number of tries to generate an orientation before giving up

        Returns
        -------
        list
            List of n orientations as rotation vectors  
        """

        def _sample_rotation_vector():
            """
            Sampling a unit vector uniformly on a sphere and a rotation angle uniformly on [0, 2pi]
            """
            u = np.random.uniform(0, 1)
            v = np.random.uniform(0, 1)
            theta = 2 * np.pi * u
            phi = np.arccos(2 * v - 1)
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            psi = np.random.uniform(0, 2 * np.pi)
            r = np.array([x, y, z]) * psi
            return r
        
        poses = []
        for _ in range(n):
            tries = 0
            while tries <= maxtries:
                r = _sample_rotation_vector()
                Rr = R.from_rotvec(r)
                euler = Rr.as_euler("ZYZ", degrees=True)
                if np.abs(euler[1]) <= maxtilt:
                    # convert euler back to rotation vector
                    rvec = R.from_euler("ZYZ", euler, degrees=True).as_rotvec()
                    poses.append(rvec.tolist())
                    break
                tries += 1

            if tries > maxtries:
                print(f"Warning: maximum number of tries ({maxtries}) reached. Returning last orientation found.")
                poses.append(rvec)
        return poses 
