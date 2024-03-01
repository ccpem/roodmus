"""Integration tests for the analysis (plotting) utilities.

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
import filecmp  # noqa: F401
import hashlib

import mrcfile  # noqa: F401
import numpy as np  # noqa: F401

from tests.integration import fixtures

from roodmus.analysis.plot_frames import plotFrameDistribution

from roodmus.analysis.plot_ctf import plotDefocusScatter

from roodmus.analysis.plot_picking import (
    plotLabelTruth,
    plotLabelPicked,
    plotLabelTruthAndPicked,
    plotPrecision,
    plotBoundaryInvestigation,
    plotOverlap,
)

from roodmus.analysis.plot_classes import plot2DClasses
from roodmus.analysis.plot_alignment import (
    plotTruePoseDistribution,
    plotPickedPoseDistribution,
)


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


def compare_dfs(
    ref_plot_data,
    out_plot_data,
):
    for (ref_plot, ref_dfs), (out_plot, out_dfs) in zip(
        ref_plot_data.items(), out_plot_data.items()
    ):
        for (_, ref_df), (_, out_df) in zip(ref_dfs.items(), out_dfs.items()):
            # print("o_ref_key: {}".format(o_ref_key))
            if "metadata_filename" in ref_df.columns:
                ref_df = ref_df.drop(columns=["metadata_filename"])
            if "metadata_filename" in out_df.columns:
                out_df = out_df.drop(columns=["metadata_filename"])
                print(
                    "Ref_df keys:{}\nOut_df_keys:{}\n".format(
                        ref_df.keys(), out_df.keys()
                    )
                )
            for ref_key, out_key in zip(
                sorted(ref_df.keys()), sorted(out_df.keys())
            ):
                assert ref_key == out_key
            # print("o_ref_df: {}".format(ref_df))
            # print("o_ref_df: {}".format(type(ref_df)))
            # print("o_out_key: {}".format(o_out_key))
            # print("o_out_df: {}".format(out_df))
            # print("o_out_df: {}".format(type(out_df)))
            # print("diff: {}".format(ref_df.compare(out_df)))
            assert ref_df.equals(out_df)


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
        pfd = "particles_frame_distribution.png"
        df_truth = os.path.join("frame_distribution", "df_truth.csv")
        df_picked = os.path.join("frame_distribution", "df_picked.csv")
        output_files: list = [
            os.path.join(plot_dir, pfd),
            os.path.join(plot_dir, df_truth),
            os.path.join(plot_dir, df_picked),
        ]
        output_files = sorted(output_files)
        output_files = [
            os.path.join(plot_dir, output_file) for output_file in output_files
        ]
        for output_file in output_files:
            assert os.path.isfile(output_file)
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
            os.path.join(
                self.test_data,
                "analysis_test_outputs/frames_star/"
                + "frame_distribution/df_truth.csv",
            ),
            os.path.join(
                self.test_data,
                "analysis_test_outputs/frames_star/"
                + "frame_distribution/df_picked.csv",
            ),
        ]
        ref_files = sorted(ref_files)

        ref_plot_frames = plotFrameDistribution(
            ["2D_class_selection"],
        )
        ref_plot_frames.setup_plot_data_empty()
        ref_plot_frames.load_dataframes(
            os.path.join(self.test_data, "analysis_test_outputs/frames_star")
        )

        output_plot_frames = plotFrameDistribution(
            ["2D_class_selection"],
        )
        output_plot_frames.setup_plot_data_empty()
        output_plot_frames.load_dataframes(plot_dir)
        print(os.listdir(plot_dir))
        print(os.listdir(os.path.join(plot_dir, "frame_distribution")))

        compare_dfs(
            ref_plot_frames.plot_data,
            output_plot_frames.plot_data,
        )

        # for (ref_plot, ref_dfs), (out_plot, out_dfs) in zip(
        #     ref_plot_frames.plot_data.items(),
        #     output_plot_frames.plot_data.items(),
        # ):
        #     for (_, ref_df), (_, out_df) in zip(
        #         ref_dfs.items(), out_dfs.items()
        #     ):
        #         # print("o_ref_key: {}".format(o_ref_key))
        #         if "metadata_filename" in ref_df.columns:
        #             ref_df = ref_df.drop(columns=["metadata_filename"])
        #         if "metadata_filename" in out_df.columns:
        #             out_df = out_df.drop(columns=["metadata_filename"])
        #         print("o_ref_df: {}".format(ref_df))
        #         print("o_ref_df: {}".format(type(ref_df)))
        #         # print("o_out_key: {}".format(o_out_key))
        #         print("o_out_df: {}".format(out_df))
        #         print("o_out_df: {}".format(type(out_df)))
        #         print("diff: {}".format(ref_df.compare(out_df)))
        #         assert ref_df.equals(out_df)

        # for output, ref in zip(output_files, ref_files):
        #     if output.endswith(".csv") and ref.endswith(".csv"):
        #         assert filecmp.cmp(ref, output)
        #     else:
        #         print("{} and {} are not compared".format(output, ref))

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

        ctf_scatter = "ctf_scatter.png"
        df_truth = os.path.join("defocus_scatter", "df_truth.csv")
        df_picked = os.path.join("defocus_scatter", "df_picked.csv")
        # find the outputs
        output_files: list = [
            os.path.join(plot_dir, ctf_scatter),
            os.path.join(plot_dir, df_truth),
            os.path.join(plot_dir, df_picked),
        ]
        output_files = sorted(output_files)
        output_files = [
            os.path.join(plot_dir, output_file) for output_file in output_files
        ]
        self.assertIsNotNone(output_files)
        assert isinstance(output_files, list)
        print("Output files: {}".format(output_files))

        # # find the reference files
        # ref_files: list = [
        #     os.path.join(
        #         self.test_data,
        #         "analysis_test_outputs/ctf_star/ctf_scatter.png",
        #     ),
        #     os.path.join(
        #         self.test_data,
        #         "analysis_test_outputs/ctf_star/defocus_scatter/df_truth.csv",
        #     ),
        #     os.path.join(
        #         self.test_data,
        #         "analysis_test_outputs/ctf_star/defocus_scatter/df_picked.csv",
        #     ),
        # ]
        # ref_files = sorted(ref_files)

        ref_plot_defocus_scatter = plotDefocusScatter("")
        ref_plot_defocus_scatter.setup_plot_data_empty()
        ref_plot_defocus_scatter.load_dataframes(
            os.path.join(self.test_data, "analysis_test_outputs/ctf_star")
        )

        out_plot_defocus_scatter = plotDefocusScatter("")
        out_plot_defocus_scatter.setup_plot_data_empty()
        out_plot_defocus_scatter.load_dataframes(plot_dir)

        compare_dfs(
            ref_plot_defocus_scatter.plot_data,
            out_plot_defocus_scatter.plot_data,
        )

        # for output, ref in zip(output_files, ref_files):
        #     if output.endswith(".csv") and ref.endswith(".csv"):
        # assert filecmp.cmp(ref, output)
        # assert md5_hash(ref) == md5_hash(output)
        # else:
        #     print("{} and {} are not compared".format(output, ref))

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
        # ref_files: list = [
        #     os.path.join(
        #         self.test_data,
        #         "analysis_test_outputs/picking_star/000000_truth.png",
        #     ),
        #     os.path.join(
        #         self.test_data,
        #         "analysis_test_outputs/picking_star/000000_"
        #         + "particles_picked.png",
        #     ),
        #     os.path.join(
        #         self.test_data,
        #         "analysis_test_outputs/picking_star/000000_"
        #         + "particles_truth_and_picked.png",
        #     ),
        #     os.path.join(
        #         self.test_data,
        #         "analysis_test_outputs/picking_star/precision.png",
        #     ),
        #     os.path.join(
        #         self.test_data,
        #           "analysis_test_outputs/picking_star/recall.png"
        #     ),
        #     os.path.join(
        #         self.test_data,
        #         "analysis_test_outputs/picking_star/precision_and_recall.png",
        #     ),
        #     os.path.join(
        #         self.test_data,
        #         "analysis_test_outputs/picking_star/f1_score.png",
        #     ),
        #     os.path.join(
        #         self.test_data,
        #         "analysis_test_outputs/picking_star/particles_boundary_x.png",
        #     ),
        #     os.path.join(
        #         self.test_data,
        #         "analysis_test_outputs/picking_star/particles_boundary_y.png",
        #     ),
        #     os.path.join(
        #         self.test_data,
        #         "analysis_test_outputs/picking_star/particles_boundary_z.png",
        #     ),
        #     os.path.join(
        #         self.test_data,
        #         "analysis_test_outputs/picking_star/particles_overlap.png",
        #     ),
        #     os.path.join(
        #         self.test_data,
        #         "analysis_test_outputs/picking_star/overlap.png",
        #     ),
        # ]
        # ref_files = sorted(ref_files)

        # plot_truth
        print("Running plot_truth_test")
        ref_plot_label_truth = plotLabelTruth(
            os.path.join(
                self.test_data,
                "tests/integration/fixtures/analysis_test_inputs/"
                "relion_subset_ugraphs",
            )
        )
        ref_plot_label_truth.setup_plot_data_empty()
        ref_plot_label_truth.load_dataframes(
            os.path.join(self.test_data, "analysis_test_outputs/picking_star")
        )

        out_plot_label_truth = plotLabelTruth(
            os.path.join(
                self.test_data,
                "tests/integration/fixtures/analysis_test_inputs/"
                "relion_subset_ugraphs",
            )
        )
        out_plot_label_truth.setup_plot_data_empty()
        out_plot_label_truth.load_dataframes(plot_dir)

        print(ref_plot_label_truth.plot_data)
        print(out_plot_label_truth.plot_data)

        compare_dfs(
            ref_plot_label_truth.plot_data,
            out_plot_label_truth.plot_data,
        )

        # plot_picked
        print("Running plot_picked_test")
        ref_plot_label_picked = plotLabelPicked(
            os.path.join(
                self.test_data,
                "tests/integration/fixtures/analysis_test_inputs/"
                "relion_subset_ugraphs",
            )
        )
        ref_plot_label_picked.setup_plot_data_empty()
        ref_plot_label_picked.load_dataframes(
            os.path.join(self.test_data, "analysis_test_outputs/picking_star")
        )

        out_plot_label_picked = plotLabelPicked(
            os.path.join(
                self.test_data,
                "tests/integration/fixtures/analysis_test_inputs/"
                "relion_subset_ugraphs",
            )
        )
        out_plot_label_picked.setup_plot_data_empty()
        out_plot_label_picked.load_dataframes(plot_dir)

        compare_dfs(
            ref_plot_label_picked.plot_data,
            out_plot_label_picked.plot_data,
        )

        # plot_truth_and_picked
        print("Running plot_truth_and_picked_test")
        ref_plot_label_truth_and_picked = plotLabelTruthAndPicked(
            os.path.join(
                self.test_data,
                "tests/integration/fixtures/analysis_test_inputs/"
                "relion_subset_ugraphs",
            )
        )
        ref_plot_label_truth_and_picked.setup_plot_data_empty()
        ref_plot_label_truth_and_picked.load_dataframes(
            os.path.join(self.test_data, "analysis_test_outputs/picking_star")
        )

        out_plot_label_truth_and_picked = plotLabelTruthAndPicked(
            os.path.join(
                self.test_data,
                "tests/integration/fixtures/analysis_test_inputs/"
                "relion_subset_ugraphs",
            )
        )
        out_plot_label_truth_and_picked.setup_plot_data_empty()
        out_plot_label_truth_and_picked.load_dataframes(plot_dir)

        # print("\n\n")
        # print(ref_plot_label_truth_and_picked.plot_data)
        # print("\n")
        # print(out_plot_label_truth_and_picked.plot_data)
        compare_dfs(
            ref_plot_label_truth_and_picked.plot_data,
            out_plot_label_truth_and_picked.plot_data,
        )

        # plot_precision
        print("Running plot_precision")
        ref_plot_precision = plotPrecision(
            # [job_types],
            # [meta_file],
            {},
            [],
        )
        ref_plot_precision.setup_plot_data_empty()
        ref_plot_precision.load_dataframes(
            os.path.join(self.test_data, "analysis_test_outputs/picking_star")
        )

        out_plot_precision = plotPrecision(
            # [job_types],
            # [meta_file],
            {},
            [],
        )
        out_plot_precision.setup_plot_data_empty()
        out_plot_precision.load_dataframes(plot_dir)

        compare_dfs(
            ref_plot_precision.plot_data,
            out_plot_precision.plot_data,
        )

        # plot_boundary_investigation
        print("Running boundary investigation test")
        ref_plot_boundary_investigation = plotBoundaryInvestigation(
            # job_types: dict[str, str],
            # bin_width: list[int],
            # axis: list[str],
            {},
            [],
            [],
        )
        ref_plot_boundary_investigation.setup_plot_data_empty()
        ref_plot_boundary_investigation.load_dataframes(
            os.path.join(self.test_data, "analysis_test_outputs/picking_star")
        )

        out_plot_boundary_investigation = plotBoundaryInvestigation(
            {},
            [],
            [],
        )
        out_plot_boundary_investigation.setup_plot_data_empty()
        out_plot_boundary_investigation.load_dataframes(plot_dir)

        compare_dfs(
            ref_plot_boundary_investigation.plot_data,
            out_plot_boundary_investigation.plot_data,
        )

        # plot_overlap
        print("Running plot overlap test")
        ref_plot_overlap = plotOverlap(
            # job_types: dict[str, str],
            {},
        )
        ref_plot_overlap.setup_plot_data_empty()
        ref_plot_overlap.load_dataframes(
            os.path.join(
                self.test_data,
                "analysis_test_outputs/picking_star",
            )
        )

        out_plot_overlap = plotOverlap(
            {},
        )
        out_plot_overlap.setup_plot_data_empty()
        out_plot_overlap.load_dataframes(plot_dir)

        compare_dfs(
            ref_plot_overlap.plot_data,
            out_plot_overlap.plot_data,
        )

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

        # plot_classes
        print("Running plot classes test")
        ref_plot_classes = plot2DClasses(
            {},
            [],
        )
        ref_plot_classes.setup_plot_data_empty()
        ref_plot_classes.load_dataframes(
            os.path.join(
                self.test_data,
                "analysis_test_outputs/classes_star",
            )
        )

        out_plot_classes = plot2DClasses(
            {},
            [],
        )
        out_plot_classes.setup_plot_data_empty()
        out_plot_classes.load_dataframes(plot_dir)

        compare_dfs(
            ref_plot_classes.plot_data,
            out_plot_classes.plot_data,
        )

    def test_plot_alignment_star(self):
        config_dir = os.path.join(
            self.test_data, "analysis_test_inputs/relion_subset_ugraphs"
        )
        meta_file = os.path.join(
            self.test_data,
            "analysis_test_inputs/relion_subset/Extract/job013/particles.star",
        )
        plot_dir = os.path.join(self.test_dir, "alignment_star")
        job_types = "3DRefinement"

        system_cmd = (
            "roodmus plot_alignment"
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

        # # find the reference files
        # ref_files: list = [
        #     os.path.join(
        #         self.test_data,
        #         "analysis_test_outputs/alignment_star/particles_picked_"
        #         + "pose_distribution.png",
        #     ),
        #     os.path.join(
        #         self.test_data,
        #         "analysis_test_outputs/alignment_star/true_pose_"
        #         + "distribution.png",
        #     ),
        # ]
        # ref_files = sorted(ref_files)

        # plot_picked alignment
        print("Running picked plot_alignment test")

        ref_picked_pose_distribution = plotPickedPoseDistribution()
        ref_picked_pose_distribution.setup_plot_data_empty()
        ref_picked_pose_distribution.load_dataframes(
            os.path.join(
                self.test_data,
                "analysis_test_outputs/alignment_star",
            )
        )

        out_picked_pose_distribution = plotPickedPoseDistribution()
        out_picked_pose_distribution.setup_plot_data_empty()
        out_picked_pose_distribution.load_dataframes(plot_dir)

        compare_dfs(
            ref_picked_pose_distribution.plot_data,
            out_picked_pose_distribution.plot_data,
        )

        # plot_truth alignment
        print("Running truth plot_alignment test")
        ref_truth_pose_distribution = plotTruePoseDistribution()
        ref_truth_pose_distribution.setup_plot_data_empty()
        ref_truth_pose_distribution.load_dataframes(
            os.path.join(
                self.test_data,
                "analysis_test_outputs/alignment_star",
            )
        )

        out_truth_pose_distribution = plotTruePoseDistribution()
        out_truth_pose_distribution.setup_plot_data_empty()
        out_truth_pose_distribution.load_dataframes(plot_dir)

        compare_dfs(
            ref_truth_pose_distribution.plot_data,
            out_truth_pose_distribution.plot_data,
        )

    # TODO for the per particle tests, need to add Polish mrc+yaml inputs
    # and rerun matching using only this data. Then overwrite the test
    # (reference) files with those new subset ones
    # TODO add plot_picking->label_matched_and_unmatched
    """
    def test_plot_picking_matching_star(self):
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
                analysis_test_outputs/picking_star/INSERT_HERE,
            ),
        ]
        ref_files = sorted(ref_files)
    # plotMatchedAndUnmatched
    # plotMatchedAndUnmatchedShiny
    """

    # TODO add plot_ctf->plot_defocus_scatter for perparticledefoci


if __name__ == "__main__":
    unittest.main()
