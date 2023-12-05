"""Integration tests for the heterogeneity calculation utilities.

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

import unittest
import os
import shutil
import tempfile

from tests.integration import fixtures


class IntegrationTestHetMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.test_data = os.path.dirname(fixtures.__file__)
        self.test_dir = tempfile.mkdtemp()

        # Change to test directory
        self._orig_dir = os.getcwd()
        os.chdir(self.test_dir)

        self.oldpath = os.environ["PATH"]

        # for analysis code integration tests also need to set up
        # the data objects which analysis utilities are applied to
        self.config_dir = ""
        return super().setUp()

    def tearDown(self) -> None:
        os.chdir(self._orig_dir)
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.environ["PATH"] = self.oldpath
        return super().tearDown()

    def test_run_het_metric(self) -> None:
        """Use 2 conformations to test that dimension reduction and clustering
        via pca and kmeans
        Distance metric would reduce matrix to 1x1 so is not included
        """

        # set up args to pass to het_metrics utility
        # first move the conformations to run on to test
        pdb_dir = os.path.join(self.test_dir, "conformations")
        os.makedirs(pdb_dir)

        frames = os.listdir(self.test_data)
        frames = [frame for frame in frames if frame.endswith(".pdb")]
        frames = [
            frame for frame in frames if frame.startswith("conformation_")
        ]
        for frame in frames:
            shutil.copy(os.path.join(self.test_data, frame), pdb_dir)

        system_cmd = (
            "roodmus het_metrics"
            + " --conformations_dir {}".format(pdb_dir)
            + " --n_confs 2"
            + " --verbose"
            + " --alignment superpose"
            + " --dimension_reduction pca"
            + " --dimensions 2"
            + " --cluster_alg kmeans"
            + " --n_clusters 2"
            + " --output_dir {}".format(self.test_dir)
        )
        print("system cmd: {}".format(system_cmd))
        os.system(system_cmd)

        # check the outputs exist
        output_files = os.listdir(self.test_dir)
        assert "workflows.csv" in output_files
        assert "superpose_pca___kmeans_2_dr.png" in output_files
        assert "superpose_pca___kmeans_2_ca.png" in output_files
        assert "superpose_pca___kmeans_2.pkl" in output_files

        # may want to add loading from pkl with equality tests in future
