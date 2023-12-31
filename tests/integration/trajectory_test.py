"""Integration tests for the trajectory utilities.

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
import filecmp
import difflib

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
