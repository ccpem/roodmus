"""unittest for ctf analysis
"""

import unittest
import os

import numpy as np


class test_ctf_analysis(unittest.TestCase):
    def setUp(self) -> None:
        roodmus_install_path = os.path.dirname(__file__)
        self.config_dir = os.path.join(roodmus_install_path, "data", "mrc")
        self.metadata_csparc_file = os.path.join(
            roodmus_install_path,
            "data",
            "cryoSPARC",
            "J293_picked_particles.cs",
        )
        self.metadata_relion_file = os.path.join(
            roodmus_install_path, "data", "RELION", "job006_topaz.star"
        )

    def test_loading_metadata(self):
        # this test is to check that the metadata is loaded correctly
        # for both cryoSPARC and RELION
        from roodmus.analysis.analyse_ctf import ctf_estimation

        # load the first metadata file (.cs)
        ctf_analysis = ctf_estimation(
            meta_file=self.metadata_csparc_file, config_dir=self.config_dir
        )

        # check that all particles have been loaded
        self.assertEqual(len(ctf_analysis.results["ugraph_filename"]), 5344)
        self.assertEqual(len(ctf_analysis.results["defocusU"]), 5344)
        self.assertEqual(len(ctf_analysis.results["defocusV"]), 5344)
        self.assertEqual(len(ctf_analysis.results["kV"]), 5344)
        self.assertEqual(len(ctf_analysis.results["Cs"]), 5344)
        self.assertEqual(len(ctf_analysis.results["amp"]), 5344)
        self.assertEqual(len(ctf_analysis.results["Bfac"]), 5344)
        self.assertEqual(len(ctf_analysis.results["defocus_truth"]), 5344)
        self.assertEqual(len(ctf_analysis.results["kV_truth"]), 5344)
        self.assertEqual(len(ctf_analysis.results["Cs_truth"]), 5344)

        # check that all micrographs have been loaded
        self.assertEqual(
            np.unique(ctf_analysis.results["ugraph_filename"]).size, 3
        )

        # check that the values for the first particle are correct
        self.assertEqual(
            ctf_analysis.results["ugraph_filename"][0], "000000.mrc"
        )
        self.assertAlmostEqual(
            ctf_analysis.results["defocusU"][0], 1544.8848, places=4
        )
        self.assertAlmostEqual(
            ctf_analysis.results["defocusV"][0], 1544.8848, places=4
        )
        self.assertAlmostEqual(ctf_analysis.results["kV"][0], 300.0, places=4)
        self.assertAlmostEqual(ctf_analysis.results["Cs"][0], 2.7, places=4)
        self.assertAlmostEqual(ctf_analysis.results["amp"][0], 0.07, places=4)
        self.assertAlmostEqual(ctf_analysis.results["Bfac"][0], 0.0, places=4)
        self.assertAlmostEqual(
            ctf_analysis.results["defocus_truth"][0], 1459.3485, places=4
        )
        self.assertAlmostEqual(
            ctf_analysis.results["kV_truth"][0], 300.0, places=4
        )
        self.assertAlmostEqual(
            ctf_analysis.results["Cs_truth"][0], 2.7, places=4
        )

        # load the second metadata file (.star)
        ctf_analysis = ctf_estimation(
            meta_file=self.metadata_relion_file, config_dir=self.config_dir
        )

        # check that all particles have been loaded
        self.assertEqual(len(ctf_analysis.results["ugraph_filename"]), 3906)
        self.assertEqual(len(ctf_analysis.results["defocusU"]), 3906)
        self.assertEqual(len(ctf_analysis.results["defocusV"]), 3906)
        self.assertEqual(len(ctf_analysis.results["kV"]), 3906)
        self.assertEqual(len(ctf_analysis.results["Cs"]), 3906)
        self.assertEqual(len(ctf_analysis.results["amp"]), 3906)
        self.assertEqual(len(ctf_analysis.results["Bfac"]), 3906)
        self.assertEqual(len(ctf_analysis.results["defocus_truth"]), 3906)
        self.assertEqual(len(ctf_analysis.results["kV_truth"]), 3906)
        self.assertEqual(len(ctf_analysis.results["Cs_truth"]), 3906)

        # check that all micrographs have been loaded
        self.assertEqual(
            np.unique(ctf_analysis.results["ugraph_filename"]).size, 3
        )

        # check that the values for the first particle are correct
        self.assertEqual(
            ctf_analysis.results["ugraph_filename"][0], "000000.mrc"
        )
        self.assertAlmostEqual(
            ctf_analysis.results["defocusU"][0], 1520.8881, places=4
        )
        self.assertAlmostEqual(
            ctf_analysis.results["defocusV"][0], 1312.2670, places=4
        )
        self.assertAlmostEqual(ctf_analysis.results["kV"][0], 300.0, places=4)
        self.assertAlmostEqual(ctf_analysis.results["Cs"][0], 2.7, places=4)
        self.assertAlmostEqual(ctf_analysis.results["amp"][0], 0.1, places=4)
        self.assertAlmostEqual(ctf_analysis.results["Bfac"][0], 0.0, places=4)
        self.assertAlmostEqual(
            ctf_analysis.results["defocus_truth"][0], 1459.3485, places=4
        )
        self.assertAlmostEqual(
            ctf_analysis.results["kV_truth"][0], 300.0, places=4
        )
        self.assertAlmostEqual(
            ctf_analysis.results["Cs_truth"][0], 2.7, places=4
        )

    def test_reloading_metadata(self):
        # this test checks if multiple files can be loaded together
        from roodmus.analysis.analyse_ctf import ctf_estimation

        # load the first metadata file (.cs)
        ctf_analysis = ctf_estimation(
            meta_file=self.metadata_csparc_file, config_dir=self.config_dir
        )
        # load the second metadata file (.cs)
        ctf_analysis.compute(
            meta_file=self.metadata_relion_file, config_dir=self.config_dir
        )

        # check that all particles have been loaded
        self.assertEqual(len(ctf_analysis.results["ugraph_filename"]), 9250)
        self.assertEqual(len(ctf_analysis.results["defocusU"]), 9250)
        self.assertEqual(len(ctf_analysis.results["defocusV"]), 9250)
        self.assertEqual(len(ctf_analysis.results["kV"]), 9250)
        self.assertEqual(len(ctf_analysis.results["Cs"]), 9250)
        self.assertEqual(len(ctf_analysis.results["amp"]), 9250)
        self.assertEqual(len(ctf_analysis.results["Bfac"]), 9250)
        self.assertEqual(len(ctf_analysis.results["defocus_truth"]), 9250)
        self.assertEqual(len(ctf_analysis.results["kV_truth"]), 9250)
        self.assertEqual(len(ctf_analysis.results["Cs_truth"]), 9250)

        # check that all micrographs have been loaded
        self.assertEqual(
            np.unique(ctf_analysis.results["ugraph_filename"]).size, 3
        )


if __name__ == "__main__":
    unittest.main()
