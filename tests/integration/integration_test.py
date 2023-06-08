import unittest
import os
import shutil
import tempfile
import filecmp
import difflib

import yaml
import numpy as np
import mrcfile

from tests.integration import fixtures


def remove_ts_text_file(filename: str) -> str:
    """Some decisions have been made which add date of generation to
    pdb files created via mdtraj and to mtf files created by parakeet.
    These mess up filecmp against a fixed date reference file.
    These dates and other superfluous info are add into first line of
    the text files.
    Therefore, this function creates temporary files which are 1 line
    shorter (removes the first line)

    Args:
        file (str): file

    Returns:
        str: Path of saved file with first line removed
    """
    # get the dirname
    which_dir = os.path.dirname(filename)

    # get the filename
    which_file = os.path.basename(filename)

    # make the new_filename
    file_name_parts = which_file.split(".")
    saved_file = "".join(file_name_parts[:-1])
    saved_file = saved_file + "_removed_line" + file_name_parts[-1]
    saved_file = os.path.join(which_dir, saved_file)

    # load text file into memory and remove first line
    with open(filename, "r") as infile:
        data = infile.read().splitlines(True)

    # save file after removing first line
    with open(saved_file, "w") as outfile:
        outfile.writelines(data[1:])
    print("Saved {} from {}".format(saved_file, filename))

    return saved_file


class IntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_data = os.path.dirname(fixtures.__file__)
        self.test_dir = tempfile.mkdtemp()

        # Change to test directory
        self._orig_dir = os.getcwd()
        os.chdir(self.test_dir)

        # set the path to find parakeet
        # see if there is a functioning copy of parakeet
        self.oldpath = os.environ["PATH"]
        return super().setUp()

    def tearDown(self) -> None:
        os.chdir(self._orig_dir)
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.environ["PATH"] = self.oldpath
        return super().tearDown()

    def test_conformations_sampling(self) -> None:
        """Test that the frames are sampled and the pdbs are as expected.
        This is done by asserting the pdb files produced are exactly
        identical to pre-determined references
        """

        # set up the args to pass to roodmus conformations_sampling
        traj_files_path = self.test_data
        top_file_path = os.path.join(
            self.test_data, "pdbfile_11021566_glyco.pdb"
        )
        sampling_method = "even_sampling"
        n_conformations = 2
        traj_extension = ".dcd"
        output_dir = os.path.join(self.test_dir, "conformations_sampling")

        # run the conformations_sampling
        system_cmd = (
            "roodmus conformations_sampling"
            + " --trajfiles_dir_path {}".format(traj_files_path)
            + " --topfile_path {}".format(top_file_path)
            + " --verbose"
            + " --sampling_method {}".format(sampling_method)
            + " --n_conformations {}".format(n_conformations)
            + " --traj_extension {}".format(traj_extension)
            + " --output_dir {}".format(output_dir)
        )
        print("system cmd: {}".format(system_cmd))
        os.system(system_cmd)

        # find the outputs
        output_files: list = os.listdir(output_dir)
        output_files = sorted(output_files)
        output_files = [
            os.path.join(output_dir, output_file)
            for output_file in output_files
        ]
        self.assertIsNotNone(output_files)
        assert isinstance(output_files, list)
        print("Output files: {}".format(output_files))

        # find the reference files
        ref_files: list = [
            os.path.join(self.test_data, "conformation_000000.pdb"),
            os.path.join(self.test_data, "conformation_000001.pdb"),
        ]
        ref_files = sorted(ref_files)
        self.assertIsNotNone(ref_files)
        assert isinstance(ref_files, list)
        print("Ref files: {}".format(ref_files))

        self.assertEqual(len(output_files), 2)

        self.assertEqual(len(ref_files), 2)

        # load the outputs and references into memory and
        # compare to reference and print and differences
        # Find and print the diff between new and reference starfiles
        for output_file, ref_file in zip(output_files, ref_files):
            # not bothered about removing timestamps when
            # simply printing differences
            with open(ref_file) as ref:
                ref_text = ref.readlines()

            with open(output_file) as new:
                new_text = new.readlines()

            for line in difflib.unified_diff(
                ref_text,
                new_text,
                fromfile=ref_file,
                tofile=output_file,
                lineterm="",
            ):
                print(line)

            # but do want to remove timestamps (first line) before
            # we do a filecmp between new and ref files
            ref_file_no_ts = remove_ts_text_file(ref_file)
            output_file_no_ts = remove_ts_text_file(output_file)

            # ensure they are the same
            assert filecmp.cmp(ref_file_no_ts, output_file_no_ts)

    def test_run_parakeet(self) -> None:
        """Test that the correct files are generated by run_parakeet.
        Note that due the implementation of random positions being left
        to Parakeet internally, the checks to be performed are:
          - whether yaml files can be loaded in as non-empty dicts via pyyaml
          without error
          - whether the mrcfiles data have the correct dimensions after
          loading them using mrcfile
        """

        # set up the args to pass to roodmus run_parakeet
        # create dir for pdbs files and move them in there
        pdb_dir = os.path.join(self.test_dir, "conformations")
        os.makedirs(pdb_dir)

        frames = os.listdir(self.test_data)
        frames = [frame for frame in frames if frame.endswith(".pdb")]
        frames = [
            frame for frame in frames if frame.startswith("conformation_")
        ]
        for frame in frames:
            shutil.copy(os.path.join(self.test_data, frame), pdb_dir)

        n_images = 1
        n_molecules = 10
        mrc_dir = os.path.join(self.test_dir, "Micrographs")
        electrons_per_angstrom = 45.0
        energy = 300.0
        nx = 4000
        ny = 4000
        pixel_size = 1.0
        c_10 = -20000.0
        c_10_stddev = 5000.0
        c_c = 2.7
        box_x = 4000.0
        box_y = 4000.0
        box_z = 500.0
        centre_x = 2000.0
        centre_y = 2000.0
        centre_z = 250.0
        sample_type = "cuboid"
        cuboid_length_x = 4000.0
        cuboid_length_y = 4000.0
        cuboid_length_z = 500.0
        simulation_margin = 0
        simulation_padding = 1

        # run the run_parakeet utility
        system_cmd = (
            "roodmus run_parakeet"
            + " --dqe"
            + " --fast_ice"
            + " --pdb_dir {}".format(pdb_dir)
            + " --mrc_dir {}".format(mrc_dir)
            + " --n_images {}".format(n_images)
            + " --n_molecules {}".format(n_molecules)
            + " --electrons_per_angstrom {}".format(electrons_per_angstrom)
            + " --energy {}".format(energy)
            + " --nx {}".format(nx)
            + " --ny {}".format(ny)
            + " --pixel_size {}".format(pixel_size)
            + " --c_10 {}".format(c_10)
            + " --c_10_stddev {}".format(c_10_stddev)
            + " --c_c {}".format(c_c)
            + " --box_x {}".format(box_x)
            + " --box_y {}".format(box_y)
            + " --box_z {}".format(box_z)
            + " --centre_x {}".format(centre_x)
            + " --centre_y {}".format(centre_y)
            + " --centre_z {}".format(centre_z)
            + " --type {}".format(sample_type)
            + " --cuboid_length_x {}".format(cuboid_length_x)
            + " --cuboid_length_y {}".format(cuboid_length_y)
            + " --cuboid_length_z {}".format(cuboid_length_z)
            + " --simulation_margin {}".format(simulation_margin)
            + " --simulation_padding {}".format(simulation_padding)
        )
        os.system(system_cmd)

        # find the outputs
        output_files = os.listdir(mrc_dir)
        output_files = [
            output_file
            for output_file in output_files
            if not os.path.isdir(output_file)
        ]
        output_files = [
            os.path.join(mrc_dir, output_file) for output_file in output_files
        ]
        print("Outputs from {} are:\n{}".format(mrc_dir, os.listdir(mrc_dir)))
        output_mtf_file: str = os.listdir(os.path.join(mrc_dir, "relion"))[0]
        output_mtf_file = os.path.join(
            os.path.join(mrc_dir, "relion"), output_mtf_file
        )

        # ensure the all expected output files exist
        assert (
            os.path.join(mrc_dir, "000000.mrc") in output_files
        ), "000000.mrc not found in {}".format(mrc_dir)
        assert (
            os.path.join(mrc_dir, "000000.yaml") in output_files
        ), "000000.yaml not found in {}".format(mrc_dir)
        assert (
            os.path.join(mrc_dir, os.path.join("relion", "mtf_300kV.star"))
            == output_mtf_file
        ), "mtf_300kV.star not found in {}".format(
            os.path.join(mrc_dir, "relion")
        )

        # load up reference mtf file
        # load the new mtf file and check it is identical to the reference
        ref_file = os.path.join(self.test_data, "mtf_300kV.star")
        with open(ref_file) as ref:
            ref_text = ref.readlines()

        with open(output_mtf_file) as new:
            new_text = new.readlines()

        # not worried about removing ts as this won't cause
        # a test failure until filecmp
        for line in difflib.unified_diff(
            ref_text,
            new_text,
            fromfile=ref_file,
            tofile=output_mtf_file,
            lineterm="",
        ):
            print(line)

        # removed ts line and
        # ensure they are the same
        ref_file_no_ts = remove_ts_text_file(ref_file)
        output_mtf_file_no_ts = remove_ts_text_file(output_mtf_file)
        assert filecmp.cmp(ref_file_no_ts, output_mtf_file_no_ts)

        # load the yaml files into pyyaml-created dicts and check that
        # at least the fields set above have the values expected
        for yaml_name in [
            yaml_file
            for yaml_file in output_files
            if yaml_file.endswith(".yaml")
        ]:
            with open(yaml_name) as yaml_stream:
                yaml_data = yaml.safe_load(yaml_stream)
                assert int(yaml_data["microscope"]["detector"]["dqe"]) == 1
                assert int(yaml_data["simulation"]["ice"]) == 1
                assert np.isclose(
                    yaml_data["microscope"]["beam"]["electrons_per_angstrom"],
                    45.0,
                    atol=1e-08,
                    equal_nan=False,
                )
                assert np.isclose(
                    yaml_data["microscope"]["beam"]["energy"],
                    300.0,
                    atol=1e-08,
                    equal_nan=False,
                )
                assert yaml_data["microscope"]["detector"]["nx"] == 4000
                assert yaml_data["microscope"]["detector"]["ny"] == 4000
                assert np.isclose(
                    yaml_data["microscope"]["detector"]["pixel_size"],
                    1.0,
                    atol=1e-08,
                    equal_nan=False,
                )
                assert np.isclose(
                    yaml_data["microscope"]["lens"]["c_c"],
                    2.7,
                    atol=1e-08,
                    equal_nan=False,
                )
                assert np.isclose(
                    yaml_data["sample"]["box"][0],
                    4000.0,
                    atol=1e-08,
                    equal_nan=False,
                )
                assert np.isclose(
                    yaml_data["sample"]["box"][1],
                    4000.0,
                    atol=1e-08,
                    equal_nan=False,
                )
                assert np.isclose(
                    yaml_data["sample"]["box"][2],
                    500.0,
                    atol=1e-08,
                    equal_nan=False,
                )
                assert np.isclose(
                    yaml_data["sample"]["centre"][0],
                    2000.0,
                    atol=1e-08,
                    equal_nan=False,
                )
                assert np.isclose(
                    yaml_data["sample"]["centre"][1],
                    2000.0,
                    atol=1e-08,
                    equal_nan=False,
                )
                assert np.isclose(
                    yaml_data["sample"]["centre"][2],
                    250.0,
                    atol=1e-08,
                    equal_nan=False,
                )
                assert yaml_data["sample"]["shape"]["type"] == "cuboid"
                assert np.isclose(
                    yaml_data["sample"]["shape"]["cuboid"]["length_x"],
                    4000.0,
                    atol=1e-08,
                    equal_nan=False,
                )
                assert np.isclose(
                    yaml_data["sample"]["shape"]["cuboid"]["length_y"],
                    4000.0,
                    atol=1e-08,
                    equal_nan=False,
                )
                assert np.isclose(
                    yaml_data["sample"]["shape"]["cuboid"]["length_z"],
                    500.0,
                    atol=1e-08,
                    equal_nan=False,
                )
                assert yaml_data["simulation"]["margin"] == 0
                assert yaml_data["simulation"]["padding"] == 1

                # check there are the correct number of particles in
                # the yaml files
                for image in yaml_data["sample"]["molecules"]["local"]:
                    self.assertEqual(len(image["instances"]), 5)

        # load the mrcfiles and check that each has the dimensions specified
        for mrc in [
            mrc_file for mrc_file in output_files if mrc_file.endswith(".mrc")
        ]:
            with mrcfile.open(mrc) as my_mrc:
                print("Image shape is: {}".format(my_mrc.data.shape))
                assert my_mrc.data.shape == (
                    1,
                    4000,
                    4000,
                ), "Shape not 1x4000x4000 as expected.\nIs: {}".format(
                    my_mrc.data.shape
                )

    # maybe should add filecmp for run_parakeet_output/ files????


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

    def test_plot_ctf_star(self):
        config_dir = os.path.join(
            self.test_dir, "analysis_test_inputs/relion_subset_ugraphs"
        )
        meta_file = os.path.join(
            self.test_dir,
            "analysis_test_inputs/relion_subset/Extract/job007/particles.star",
        )
        plot_dir = os.path.join(self.test_dir, "ctf_star")
        plot_types = "scatter ctf"

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
            os.path.join(
                self.test_data, "analysis_test_outputs/ctf_star/ctf_0.png"
            ),
        ]
        ref_files = sorted(ref_files)

        for output, ref in zip(output_files, ref_files):
            assert filecmp.cmp(ref, output)

    def test_plot_picking_star(self):
        config_dir = os.path.join(
            self.test_dir, "analysis_test_inputs/relion_subset_ugraphs"
        )
        meta_file = os.path.join(
            self.test_dir,
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
                "analysis_test_outputs/picking_star/000000_\
                    particles_picked.png",
            ),
            os.path.join(
                self.test_data,
                "analysis_test_outputs/picking_star/000000_\
                    particles_truth_and_picked.png",
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
            self.test_dir, "analysis_test_inputs/relion_subset_ugraphs"
        )
        meta_file = os.path.join(
            self.test_dir,
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
                "analysis_test_outputs/classes_star/2D_class_selection_2Dclass_\
                precision.png",
            ),
            os.path.join(
                self.test_data,
                "analysis_test_outputs/classes_star/2D_class_selection_2Dclass\
                    _frame_distribution.png",
            ),
        ]
        ref_files = sorted(ref_files)

        for output, ref in zip(output_files, ref_files):
            assert filecmp.cmp(ref, output)

    def test_plot_alignment_star(self):
        config_dir = os.path.join(
            self.test_dir, "analysis_test_inputs/relion_subset_ugraphs"
        )
        meta_file = os.path.join(
            self.test_dir,
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
                "analysis_test_outputs/alignment_star/particles_picked_\
                    pose_distribution.png",
            ),
            os.path.join(
                self.test_data,
                "analysis_test_outputs/alignment_star/true_pose_\
                    distribution.png",
            ),
        ]
        ref_files = sorted(ref_files)

        for output, ref in zip(output_files, ref_files):
            assert filecmp.cmp(ref, output)


if __name__ == "__main__":
    unittest.main()
