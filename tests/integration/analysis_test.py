import unittest
import os
import shutil
import tempfile
import filecmp  # noqa: F401
import hashlib

import mrcfile  # noqa: F401
import numpy as np  # noqa: F401

from tests.integration import fixtures


def md5_hash(fpath):
    """Grab binary data from file and calculate md5 hash to allow file
    comparison.

    Args:
        fpath (_type_): File to read binary data from.

    Returns:
        _type_: MD5 hash
    """
    md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for data_chunk in iter(lambda: f.read(4096), b""):
            md5.update(data_chunk)
    return md5.digest()


class IntegrationTestAnalysis(unittest.TestCase):
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

    """
    def test_plot_frames_star(self):
        config_dir = os.path.join(
            self.test_data, "analysis_test_inputs/relion_subset_ugraphs"
        )
        meta_file = os.path.join(
            self.test_data,
            "analysis_test_inputs/relion_subset/Select/job009/particles.star",
        )
        plot_dir = os.path.join(self.test_dir, "frames_star")
        job_types = "2D_class_selection"

        system_cmd = (
            "roodmus plot_frames"
            + " --config_dir {}".format(config_dir)
            + " --meta_file {}".format(meta_file)
            + " --plot_dir {}".format(plot_dir)
            + " --job_types {}".format(job_types)
        )
        print("system cmd: {}".format(system_cmd))
        os.system(system_cmd)

        # find the outputs
        output_files: list = os.listdir(plot_dir)
        output_files = sorted(output_files)
        output_files = [
            os.path.join(plot_dir, output_file) for output_file in output_files
        ]
        self.assertIsNotNone(output_files)
        assert isinstance(output_files, list)
        print("Output files: {}".format(output_files))

        # find the reference files
        ref_files: list = [
            os.path.join(
                self.test_data,
                "analysis_test_outputs/frames_star/particles_frame_"
                + "distribution.png",
            ),
        ]
        ref_files = sorted(ref_files)

        for output, ref in zip(output_files, ref_files):
            assert filecmp.cmp(ref, output)

    def test_extract_particles(self):
        config_dir = os.path.join(
            self.test_data, "analysis_test_inputs/relion_subset_ugraphs"
        )

        particle_dir = os.path.join(self.test_dir, "extract_star")

        system_cmd = (
            "roodmus extract_particles"
            + " --config_dir {}".format(config_dir)
            + " --particle_dir {}".format(particle_dir)
        )
        print("system cmd: {}".format(system_cmd))
        os.system(system_cmd)

        # find the outputs
        output_files: list = os.listdir(particle_dir)
        output_files = sorted(output_files)
        output_files = [
            os.path.join(particle_dir, output_file)
            for output_file in output_files
        ]
        self.assertIsNotNone(output_files)
        assert isinstance(output_files, list)
        print("Output files: {}".format(output_files))

        # find the reference files
        ref_files: list = [
            os.path.join(
                self.test_data,
                "analysis_test_outputs/extract_star/particle_stack.tif",
            ),
            os.path.join(
                self.test_data,
                "analysis_test_outputs/extract_star/particle_stack.mrc",
            ),
        ]
        ref_files = sorted(ref_files)

        for output, ref in zip(output_files, ref_files):
            # stop mrc files having a filecmp (is there a timestamp prob?)
            if ref.endswith(".mrc"):
                continue
            assert filecmp.cmp(ref, output)

        # load the mrcfiles and check that the stack data is the same
        for mrc_ref, mrc_out in zip(
            [mrc_ref for mrc_ref in output_files if mrc_ref.endswith(".mrc")],
            [mrc_out for mrc_out in ref_files if mrc_out.endswith(".mrc")],
        ):
            with mrcfile.open(mrc_ref) as ref_mrc, mrcfile.open(
                mrc_out
            ) as out_mrc:
                print(
                    "Image shapes are:\nref: {}\nout: {}".format(
                        ref_mrc.data.shape,
                        out_mrc.data.shape,
                    )
                )
                assert np.array_equal(ref_mrc.data, out_mrc.data)
    """

    """
    def test_plot_ctf_star(self):
        config_dir = os.path.join(
            self.test_data, "analysis_test_inputs/relion_subset_ugraphs"
        )
        meta_file = os.path.join(
            self.test_data,
            "analysis_test_inputs/relion_subset/Extract/job007/particles.star",
        )
        plot_dir = os.path.join(self.test_dir, "ctf_star")
        plot_types = "ctf"

        system_cmd = (
            "roodmus plot_ctf"
            + " --config_dir {}".format(config_dir)
            + " --meta_file {}".format(meta_file)
            + " --plot_dir {}".format(plot_dir)
            + " --plot_types {}".format(plot_types)
        )
        print("system cmd: {}".format(system_cmd))
        os.system(system_cmd)

        # find the outputs
        output_files: list = os.listdir(plot_dir)
        output_files = sorted(output_files)
        output_files = [
            os.path.join(plot_dir, output_file) for output_file in output_files
        ]
        self.assertIsNotNone(output_files)
        assert isinstance(output_files, list)
        print("Output files: {}".format(output_files))

        # find the reference files
        ref_files: list = [
            os.path.join(
                self.test_data, "analysis_test_outputs/ctf_star/ctf_0.png"
            ),
        ]
        ref_files = sorted(ref_files)

        for output, ref in zip(output_files, ref_files):
            assert filecmp.cmp(ref, output)
            assert md5_hash(ref) == md5_hash(output)
    """

    """
    def test_plot_ctf_scatter_star(self):
        config_dir = os.path.join(
            self.test_data, "analysis_test_inputs/relion_subset_ugraphs"
        )
        meta_file = os.path.join(
            self.test_data,
            "analysis_test_inputs/relion_subset/Extract/job007/particles.star",
        )
        plot_dir = os.path.join(self.test_dir, "ctf_star")
        plot_types = "scatter"

        system_cmd = (
            "roodmus plot_ctf"
            + " --config_dir {}".format(config_dir)
            + " --meta_file {}".format(meta_file)
            + " --plot_dir {}".format(plot_dir)
            + " --plot_types {}".format(plot_types)
        )
        print("system cmd: {}".format(system_cmd))
        os.system(system_cmd)

        # find the outputs
        output_files: list = os.listdir(plot_dir)
        output_files = sorted(output_files)
        output_files = [
            os.path.join(plot_dir, output_file) for output_file in output_files
        ]
        self.assertIsNotNone(output_files)
        assert isinstance(output_files, list)
        print("Output files: {}".format(output_files))

        # find the reference files
        ref_files: list = [
            os.path.join(
                self.test_data,
                "analysis_test_outputs/ctf_star/ctf_scatter.png",
            ),
        ]
        ref_files = sorted(ref_files)

        for output, ref in zip(output_files, ref_files):
            assert filecmp.cmp(ref, output)
            assert md5_hash(ref) == md5_hash(output)

    def test_plot_picking_star(self):
        config_dir = os.path.join(
            self.test_data, "analysis_test_inputs/relion_subset_ugraphs"
        )
        meta_file = os.path.join(
            self.test_data,
            "analysis_test_inputs/relion_subset/Extract/job007/particles.star",
        )
        job_types = "topaz_picking"
        plot_dir = os.path.join(self.test_dir, "picking_star")
        plot_types = (
            "label_truth label_picked label_truth_and_picked precision "
            + "boundary overlap"
        )

        system_cmd = (
            "roodmus plot_picking"
            + " --config_dir {}".format(config_dir)
            + " --meta_file {}".format(meta_file)
            + " --job_types {}".format(job_types)
            + " --plot_dir {}".format(plot_dir)
            + " --plot_types {}".format(plot_types)
        )
        print("system cmd: {}".format(system_cmd))
        os.system(system_cmd)

        # find the outputs
        output_files: list = os.listdir(plot_dir)
        output_files = sorted(output_files)
        output_files = [
            os.path.join(plot_dir, output_file) for output_file in output_files
        ]
        self.assertIsNotNone(output_files)
        assert isinstance(output_files, list)
        print("Output files: {}".format(output_files))

        # find the reference files
        ref_files: list = [
            os.path.join(
                self.test_data,
                "analysis_test_outputs/picking_star/000000_truth.png",
            ),
            os.path.join(
                self.test_data,
                "analysis_test_outputs/picking_star/000000_"
                + "particles_picked.png",
            ),
            os.path.join(
                self.test_data,
                "analysis_test_outputs/picking_star/000000_"
                + "particles_truth_and_picked.png",
            ),
            os.path.join(
                self.test_data,
                "analysis_test_outputs/picking_star/precision.png",
            ),
            os.path.join(
                self.test_data, "analysis_test_outputs/picking_star/recall.png"
            ),
            os.path.join(
                self.test_data,
                "analysis_test_outputs/picking_star/precision_and_recall.png",
            ),
            os.path.join(
                self.test_data,
                "analysis_test_outputs/picking_star/f1_score.png",
            ),
            os.path.join(
                self.test_data,
                "analysis_test_outputs/picking_star/particles_boundary_x.png",
            ),
            os.path.join(
                self.test_data,
                "analysis_test_outputs/picking_star/particles_boundary_y.png",
            ),
            os.path.join(
                self.test_data,
                "analysis_test_outputs/picking_star/particles_boundary_z.png",
            ),
            os.path.join(
                self.test_data,
                "analysis_test_outputs/picking_star/particles_overlap.png",
            ),
            os.path.join(
                self.test_data,
                "analysis_test_outputs/picking_star/overlap.png",
            ),
        ]
        ref_files = sorted(ref_files)

        for output, ref in zip(output_files, ref_files):
            assert filecmp.cmp(ref, output)

    def test_plot_classes_star(self):
        config_dir = os.path.join(
            self.test_data, "analysis_test_inputs/relion_subset_ugraphs"
        )
        meta_file = os.path.join(
            self.test_data,
            "analysis_test_inputs/relion_subset/Select/job009/particles.star",
        )
        job_types = "2D_class_selection"
        plot_dir = os.path.join(self.test_dir, "classes_star")
        plot_types = "precision frame_distribution"

        system_cmd = (
            "roodmus plot_classes"
            + " --config_dir {}".format(config_dir)
            + " --meta_file {}".format(meta_file)
            + " --job_types {}".format(job_types)
            + " --plot_dir {}".format(plot_dir)
            + " --plot_types {}".format(plot_types)
        )
        print("system cmd: {}".format(system_cmd))
        os.system(system_cmd)

        # find the outputs
        output_files: list = os.listdir(plot_dir)
        output_files = sorted(output_files)
        output_files = [
            os.path.join(plot_dir, output_file) for output_file in output_files
        ]
        self.assertIsNotNone(output_files)
        assert isinstance(output_files, list)
        print("Output files: {}".format(output_files))

        # find the reference files
        ref_files: list = [
            os.path.join(
                self.test_data,
                "analysis_test_outputs/classes_star/2D_class_selection_2Dclass"
                + "_precision.png",
            ),
            os.path.join(
                self.test_data,
                "analysis_test_outputs/classes_star/2D_class_selection_2Dclass"
                + "_frame_distribution.png",
            ),
        ]
        ref_files = sorted(ref_files)

        for output, ref in zip(output_files, ref_files):
            assert filecmp.cmp(ref, output)

    def test_plot_alignment_star(self):
        config_dir = os.path.join(
            self.test_data, "analysis_test_inputs/relion_subset_ugraphs"
        )
        meta_file = os.path.join(
            self.test_data,
            "analysis_test_inputs/relion_subset/Extract/job013/particles.star",
        )
        plot_dir = os.path.join(self.test_dir, "alignment_star")

        system_cmd = (
            "roodmus plot_alignment"
            + " --config_dir {}".format(config_dir)
            + " --meta_file {}".format(meta_file)
            + " --plot_dir {}".format(plot_dir)
        )
        print("system cmd: {}".format(system_cmd))
        os.system(system_cmd)

        # find the outputs
        output_files: list = os.listdir(plot_dir)
        output_files = sorted(output_files)
        output_files = [
            os.path.join(plot_dir, output_file) for output_file in output_files
        ]
        self.assertIsNotNone(output_files)
        assert isinstance(output_files, list)
        print("Output files: {}".format(output_files))

        # find the reference files
        ref_files: list = [
            os.path.join(
                self.test_data,
                "analysis_test_outputs/alignment_star/particles_picked_"
                + "pose_distribution.png",
            ),
            os.path.join(
                self.test_data,
                "analysis_test_outputs/alignment_star/true_pose_"
                + "distribution.png",
            ),
        ]
        ref_files = sorted(ref_files)

        for output, ref in zip(output_files, ref_files):
            assert filecmp.cmp(ref, output)
    """


if __name__ == "__main__":
    unittest.main()
