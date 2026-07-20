"""
computes the GMM overlap metric between a series of input pseudo-atomic models
in one directory and a set of ground-truth structures
"""

import argparse
import glob2
import logging
import os
import gemmi
import json
import math
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
from pathlib import Path
import torch
from torch import Tensor
from pykeops.torch import LazyTensor
from contextlib import contextmanager
from time import perf_counter
from mpi4py import MPI
from Lie_averaging import mean_se3


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the directory containing the input \
            pseudo-atomic models.",
    )
    parser.add_argument(
        "--ground_truth_dir",
        type=str,
        required=True,
        help="Path to the directory containing the ground-truth structures.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory where the GMM overlap \
            results will be saved.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional information during execution.",
    )
    parser.add_argument(
        "--gpu_ids",
        help="select the GPU ids to use for computation",
        type=int,
        default=[0],
        nargs="+",
    )
    parser.add_argument(
        "--N_sample",
        type=int,
        default=1000,
        help="Number of random pairs of pseudo-atomic and \
            ground-truth models to sample for alignment.",
    )
    return parser


def setup_logger(rank: int, log_dir: str = "logs"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"rank_{rank}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | rank=%(rank)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(f"{log_dir}/rank_{rank:04d}.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)

    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logging.LoggerAdapter(logger, {"rank": rank})


def parse_inputs(
    input_dir: str, ground_truth_dir: str, ll: logging.Logger
) -> Tuple[List[str], List[str]]:
    # load the first pseudo-atom model and ground-truth model
    # and compute the GMM overlap
    if args.input_dir.endswith(".pdb") or args.input_dir.endswith(".cif"):
        # user provided a pattern instead of a directory
        ps_models = glob2.glob(args.input_dir)
    else:
        ps_models = glob2.glob(args.input_dir + "/*.pdb")
    ll.info(
        f"Found {len(ps_models)} pseudo-atomic models in {args.input_dir}."
    )
    for ii in range(min(5, len(ps_models))):
        ll.debug(ps_models[ii])

    if args.ground_truth_dir.endswith(
        ".pdb"
    ) or args.ground_truth_dir.endswith(".cif"):
        gt_models = glob2.glob(args.ground_truth_dir)
    else:
        gt_models = glob2.glob(args.ground_truth_dir + "/*.pdb")
    ll.info(f"Found {len(gt_models)} \
            ground-truth models in {args.ground_truth_dir}.")
    for ii in range(min(5, len(gt_models))):
        ll.debug(gt_models[ii])

    return ps_models, gt_models


@dataclass
class RegistrationResult:
    rotation: Tensor  # [3, 3]
    translation: Tensor  # [3]
    l2_squared: Tensor  # scalar
    transformed_means: Tensor  # [M, 3]
    transformed_covariances: Tensor  # [M, 3, 3]


def _validate_gmm(weights: Tensor, means: Tensor, covariances: Tensor) -> None:
    """Validate a 3D Gaussian mixture."""
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError("means must have shape [K, 3].")
    if weights.shape != (means.shape[0],):
        raise ValueError("weights must have shape [K].")
    if covariances.shape != (means.shape[0], 3, 3) and covariances.shape != (
        means.shape[0],
    ):
        raise ValueError("covariances must have shape [K, 3, 3]. or [K,]")
    if not torch.isfinite(weights).all():
        raise ValueError("weights contain non-finite values.")
    if (weights < 0).any():
        raise ValueError("weights must be non-negative.")
    if weights.sum() <= 0:
        raise ValueError("weights must have a positive sum.")


def normalize_weights(weights: Tensor) -> Tensor:
    return weights / weights.sum()


def skew_symmetric(rotation_vector: Tensor) -> Tensor:
    """Convert a length-3 rotation vector to a skew-symmetric matrix."""
    if rotation_vector.shape != (3,):
        raise ValueError("rotation_vector must have shape [3].")

    x, y, z = rotation_vector.unbind()
    zero = torch.zeros(
        (), dtype=rotation_vector.dtype, device=rotation_vector.device
    )

    return torch.stack(
        (
            zero,
            -z,
            y,
            z,
            zero,
            -x,
            -y,
            x,
            zero,
        )
    ).reshape(3, 3)


def rotation_matrix(rotation_vector: Tensor) -> Tensor:
    """
    Exponential-map parameterization of SO(3).

    torch.linalg.matrix_exp keeps the operation differentiable and avoids
    manually handling the small-angle limit of Rodrigues' formula.
    """
    return torch.linalg.matrix_exp(skew_symmetric(rotation_vector))


def transform_gmm(
    means: Tensor,
    covariances: Tensor,
    rotation: Tensor,
    translation: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply x -> R x + t to Gaussian means and covariances."""
    transformed_means = means @ rotation.T + translation
    if covariances.ndim == 3 or covariances.shape[1:] == (3, 3):
        # if the covariance is a full matrix,
        # we need to apply the rotation to it
        transformed_covariances = (
            rotation.unsqueeze(0) @ covariances @ rotation.T.unsqueeze(0)
        )
    else:
        # the covariance is a scalar, isotropic value
        # so we don't need to apply the rotation
        transformed_covariances = covariances
    return transformed_means, transformed_covariances


def gmm_inner_product_iso(
    weights_a: Tensor,
    means_a: Tensor,
    variances_a: Tensor,
    weights_b: Tensor,
    means_b: Tensor,
    variances_b: Tensor,
    *,
    jitter: float = 1e-8,
) -> Tensor:
    # print("weights_a shape:", weights_a.shape)
    # print("variances_a shape:", variances_a.shape)
    mean_i = LazyTensor(means_a[:, None, :])  # [M, 1, 3]
    variance_i = LazyTensor(variances_a[:, None, None])  # [M, 1, 1]
    # print("variance_i shape:", variance_i.shape)
    weight_i = LazyTensor(weights_a[:, None, None])  # [M, 1, 1]
    # print("weight_i shape:", weight_i.shape)

    mean_j = LazyTensor(means_b[None, :, :])  # [1, N, 3]
    variance_j = LazyTensor(variances_b[None, :, None])  # [1, N, 1]
    weight_j = LazyTensor(weights_b[None, :, None])  # [1, N, 1]

    distance2_ij = ((mean_i - mean_j) ** 2).sum(-1)  # [M, N] symbolic
    # print("distance2_ij shape:", distance2_ij.shape)
    variance_ij = variance_i + variance_j + jitter  # [M, N] symbolic
    # print("variance_ij shape:", variance_ij.shape)
    log_overlap_ij = (
        -0.5 * distance2_ij / variance_ij
        - 1.5 * (2 * math.pi * variance_ij).log()
    )  # [M, N] symbolic
    # print("log_overlap_ij shape:", log_overlap_ij.shape)
    contribution_ij = (
        weight_i * weight_j * log_overlap_ij.exp()
    )  # [M, N] symbolic
    # print("contribution_ij shape:", contribution_ij.shape)
    contribution_i = contribution_ij.sum(dim=1)  # [M, 1]
    # print("contribution_i shape:", contribution_i.shape)
    return contribution_i.sum(dim=0).squeeze()  # scalar


def gmm_l2_squared(
    weights_a: Tensor,
    means_a: Tensor,
    covariances_a: Tensor,
    weights_b: Tensor,
    means_b: Tensor,
    covariances_b: Tensor,
    *,
    jitter: float = 1e-8,
    clamp: bool = False,
) -> Tensor:
    """Integrated squared L2 distance between two normalized GMM densities."""
    if covariances_a.ndim == 1 and covariances_b.ndim == 1:
        # The covariances are isotropic and represented as a scalar variance
        # for each component. Use the optimized inner product for isotropic
        # covariances. Much faster than the full covariance version.
        aa = gmm_inner_product_iso(
            weights_a,
            means_a,
            covariances_a,
            weights_a,
            means_a,
            covariances_a,
            jitter=jitter,
        )
        bb = gmm_inner_product_iso(
            weights_b,
            means_b,
            covariances_b,
            weights_b,
            means_b,
            covariances_b,
            jitter=jitter,
        )
        ab = gmm_inner_product_iso(
            weights_a,
            means_a,
            covariances_a,
            weights_b,
            means_b,
            covariances_b,
            jitter=jitter,
        )
    else:
        raise NotImplementedError(
            "Full covariance GMMs are not yet supported."
        )

    value = aa + bb - 2.0 * ab

    # Only use this for reporting. Do not clamp during optimization, because
    # clamping can suppress gradients near zero.
    return value.clamp_min(0.0) if clamp else value


def register_gmms_rigid(
    weights_moving: Tensor,
    means_moving: Tensor,
    covariances_moving: Tensor,
    weights_fixed: Tensor,
    means_fixed: Tensor,
    covariances_fixed: Tensor,
    *,
    ll: Optional[logging.Logger] = None,
    initial_rotation_vector: Optional[Tensor] = None,
    initial_translation: Optional[Tensor] = None,
    adam_steps: int = 300,
    adam_learning_rate: float = 5e-2,
    lbfgs_steps: int = 50,
    jitter: float = 1e-8,
) -> RegistrationResult:
    """
    Rigidly register the moving GMM to the fixed GMM.

    Uses Adam for basin finding, followed by L-BFGS refinement.
    All input tensors must be on the same device and use the same dtype.
    """
    _validate_gmm(weights_moving, means_moving, covariances_moving)
    _validate_gmm(weights_fixed, means_fixed, covariances_fixed)
    if ll is not None:
        ll.info(f"Registering moving GMM with {weights_moving.shape[0]} \
                components to fixed GMM with \
                    {weights_fixed.shape[0]} components.")

    weights_moving = normalize_weights(weights_moving)
    weights_fixed = normalize_weights(weights_fixed)

    device = means_moving.device
    dtype = means_moving.dtype

    if initial_rotation_vector is None:
        initial_rotation_vector = torch.zeros(3, device=device, dtype=dtype)

    if initial_translation is None:
        moving_centroid = torch.sum(
            weights_moving[:, None] * means_moving, dim=0
        )
        fixed_centroid = torch.sum(weights_fixed[:, None] * means_fixed, dim=0)
        initial_translation = fixed_centroid - moving_centroid

    if ll is not None:
        ll.info(
            f"Initial rotation vector: {initial_rotation_vector.cpu().numpy()}"
        )
        ll.info(f"Initial translation: {initial_translation.cpu().numpy()}")

    rotation_vector = torch.nn.Parameter(
        initial_rotation_vector.detach().clone()
    )
    translation = torch.nn.Parameter(initial_translation.detach().clone())

    def objective() -> Tensor:
        rotation = rotation_matrix(rotation_vector)
        transformed_means, transformed_covariances = transform_gmm(
            means_moving,
            covariances_moving,
            rotation,
            translation,
        )

        return gmm_l2_squared(
            weights_moving,
            transformed_means,
            transformed_covariances,
            weights_fixed,
            means_fixed,
            covariances_fixed,
            jitter=jitter,
        )

    # Adam is relatively forgiving when the initialization is not already
    # inside the local convergence basin.
    adam = torch.optim.Adam(
        [rotation_vector, translation],
        lr=adam_learning_rate,
    )
    if ll is not None:
        ll.info(f"Running {adam_steps} Adam steps with \
                learning rate {adam_learning_rate}.")

    for ii in range(adam_steps):
        adam.zero_grad(set_to_none=True)
        loss = objective()
        if not torch.isfinite(loss):
            raise RuntimeError("Registration objective became non-finite.")
        loss.backward()
        adam.step()

        # report on the progress every 10% of the total steps
        if ll is not None and (ii + 1) % max(1, adam_steps // 10) == 0:
            ll.info(f"Adam step {ii + 1}/{adam_steps}, loss={loss.item()}")

    # L-BFGS generally gives a cleaner final local optimum.
    lbfgs = torch.optim.LBFGS(
        [rotation_vector, translation],
        max_iter=lbfgs_steps,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
    )

    def closure() -> Tensor:
        lbfgs.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        return loss

    lbfgs.step(closure)

    with torch.no_grad():
        rotation = rotation_matrix(rotation_vector)
        transformed_means, transformed_covariances = transform_gmm(
            means_moving,
            covariances_moving,
            rotation,
            translation,
        )

        final_score = gmm_l2_squared(
            weights_moving,
            transformed_means,
            transformed_covariances,
            weights_fixed,
            means_fixed,
            covariances_fixed,
            jitter=jitter,
            clamp=True,
        )

    return RegistrationResult(
        rotation=rotation.detach(),
        translation=translation.detach(),
        l2_squared=final_score.detach(),
        transformed_means=transformed_means.detach(),
        transformed_covariances=transformed_covariances.detach(),
    )
    """
    # testing if the code L2 distance works
    rotation = rotation_matrix(rotation_vector)
    transformed_means, transformed_covariances = transform_gmm(
        means_moving,
        covariances_moving,
        rotation,
        translation,
    )

    test = gmm_l2_squared(
        weights_moving,
        transformed_means,
        transformed_covariances,
        weights_fixed,
        means_fixed,
        covariances_fixed,
        jitter=jitter,
    )
    return RegistrationResult(
        rotation=rotation.detach(),
        translation=translation.detach(),
        l2_squared=test.detach(),
        transformed_means=transformed_means.detach(),
        transformed_covariances=transformed_covariances.detach(),
    )
    """


def cif_to_gmm(
    filename: str,
    device: torch.device,
    ll: logging.Logger,
    full_covar: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert a CIF file to a GMM representation."""
    structure = gemmi.read_structure(filename)
    total_atoms = sum(model.count_atom_sites() for model in structure)
    ll.info(f"Converting {filename} to GMM representation \
            with {total_atoms} atoms.")

    weights = np.zeros(total_atoms, dtype=np.float32)
    means = np.zeros((total_atoms, 3), dtype=np.float32)
    if full_covar:
        covariances = np.zeros((total_atoms, 3, 3), dtype=np.float32)
    else:
        covariances = np.zeros(total_atoms, dtype=np.float32)

    atm_idx = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    weights[atm_idx] = atom.occ
                    means[atm_idx] = np.array(
                        [atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float32
                    )
                    bfac = atom.b_iso
                    if bfac <= 0:
                        bfac = 30  # avoid zero variance
                    if full_covar:
                        covariances[atm_idx] = (
                            np.eye(3) * bfac / 8.0 / np.pi**2
                        )
                    else:
                        covariances[atm_idx] = bfac / 8.0 / np.pi**2
                    if atm_idx == 0:
                        ll.info(f"Atom: {atom.name}, \
                                Type: {atom.element.name}, \
                                Occ: {atom.occ}, B_iso: {bfac}")
                    atm_idx += 1

    weights_tensor = torch.tensor(
        np.array(weights), dtype=torch.float32, device=device
    )
    ll.info(weights_tensor.shape)
    means_tensor = torch.tensor(means, dtype=torch.float32, device=device)
    ll.info(means_tensor.shape)
    covariances_tensor = torch.tensor(
        covariances, dtype=torch.float32, device=device
    )
    ll.info(covariances_tensor.shape)

    _validate_gmm(weights_tensor, means_tensor, covariances_tensor)

    return weights_tensor, means_tensor, covariances_tensor


def gmm_to_cif(
    weights: Tensor, means: Tensor, covariances: Tensor, filename: str
) -> None:
    """Convert a GMM representation to a CIF file."""
    _validate_gmm(weights, means, covariances)

    structure = gemmi.Structure()
    model = gemmi.Model("GMM")
    chain = gemmi.Chain("A")

    for i in range(means.shape[0]):
        atom = gemmi.Atom()
        atom.name = "O"
        atom.pos.x = float(means[i, 0].item())
        atom.pos.y = float(means[i, 1].item())
        atom.pos.z = float(means[i, 2].item())
        atom.occ = float(weights[i].item())
        if covariances.ndim == 1:
            atom.b_iso = float(covariances[i].item() * 8.0 * np.pi**2)
        else:
            atom.b_iso = float(
                torch.trace(covariances[i]).item() * 8.0 * np.pi**2
            )
        residue = gemmi.Residue()
        residue.name = "HOH"
        residue.seqid.num = i + 1
        residue.add_atom(atom)
        chain.add_residue(residue)

    model.add_chain(chain)
    structure.add_model(model)

    # structure.write_pdb(filename)
    structure_cif = structure.make_mmcif_document()
    structure_cif.write_file(filename)


def gen_example_gmms(
    N_gmm: int, device: torch.device, ll: logging.Logger
) -> Tuple[
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tuple[Tensor, Tensor]
]:
    """Generate example GMMs for testing."""
    # Define parameters for the first GMM (moving)
    # Note: I've simplified the problem to only allow isotropic covariances
    # for the moving GMM. This is always the case for a pseudo-atomic model,
    # and almost always for a ground-truth model.
    weights_moving = torch.ones(N_gmm, device=device) / N_gmm
    means_moving = torch.randn(N_gmm, 3, device=device) * 10.0
    covariances_moving = (
        torch.rand(N_gmm, device=device) * 0.5 + 0.1
    )  # Ensure positive variance

    # Define a known rotation and translation
    true_rotation_vector = torch.tensor(
        [0.1, 0.2, 0.3], device=device
    )  # Small rotation
    true_rotation = rotation_matrix(true_rotation_vector)

    true_translation = torch.tensor([5.0, -3.0, 2.0], device=device)

    # Apply the known transformation to generate the second GMM (fixed)
    means_fixed, covariances_fixed = transform_gmm(
        means_moving, covariances_moving, true_rotation, true_translation
    )
    weights_fixed = weights_moving.clone()  # Keep the same weights

    # add noise to the fixed means to simulate real-world data
    sigma = 1.0
    noise = torch.randn_like(means_fixed) * sigma
    means_fixed += noise

    # also add noise to the covariances to simulate real-world data
    covariances_moving = covariances_moving * (
        1.0 + 0.1 * torch.randn_like(covariances_moving)
    )

    return (
        weights_moving,
        means_moving,
        covariances_moving,
        weights_fixed,
        means_fixed,
        covariances_fixed,
        (true_rotation, true_translation),
    )


@contextmanager
def timed(logger, name, **fields):
    t0 = perf_counter()
    try:
        yield
    finally:
        dt = perf_counter() - t0
        extra = " ".join(f"{k}={v}" for k, v in fields.items())
        logger.info("TIMING %s seconds=%.6f %s", name, dt, extra)


def test_registration(device: torch.device, ll: logging.Logger) -> None:
    for N_gmm in [20000]:
        ll.info(
            f"Generating example GMMs with {N_gmm} components for testing."
        )
        (
            ps_weights,
            ps_means,
            ps_covariances,
            gt_weights,
            gt_means,
            gt_covariances,
            true_transformation,
        ) = gen_example_gmms(N_gmm, device, ll)
        ll.info(f"gt_weights shape: {gt_weights.shape}, \
                gt_means shape: {gt_means.shape}, \
                    gt_covariances shape: {gt_covariances.shape}")

        with timed(ll, f"register_gmms_rigid_{N_gmm}"):
            registration_result = register_gmms_rigid(
                weights_moving=ps_weights,
                means_moving=ps_means,
                covariances_moving=ps_covariances,
                weights_fixed=gt_weights,
                means_fixed=gt_means,
                covariances_fixed=gt_covariances,
                initial_rotation_vector=None,
                initial_translation=None,
                adam_steps=300,
                adam_learning_rate=5e-2,
                lbfgs_steps=50,
                jitter=1e-8,
            )

        ll.info(
            f"Rotation matrix:\n{registration_result.rotation.cpu().numpy()}"
        )
        ll.info(f"Translation vector:\
                \n{registration_result.translation.cpu().numpy()}")
        ll.info(
            f"L2 squared distance: {registration_result.l2_squared.item()}"
        )

        ll.info("True transformation for comparison:")
        ll.info(f"True rotation matrix:\
                \n{true_transformation[0].cpu().numpy()}")
        ll.info(f"True translation vector:\
                \n{true_transformation[1].cpu().numpy()}")


def main(args):
    """
    This script computes the GMM overlap between all pairs of models in the
    input directory and the ground-truth directory. The number of these
    models, especially of the ground-truth models can be very large and
    intractable to do alignment for. Instead the script first samples
    N_sample random pairs and performs alignment for only those. Then the
    average alignment between the pseudo-atomic models and the ground-truth
    models is computed using Lie algebraic averaging. It is assumed that the
    pseudo-atomic models and ground-truth models are aligned to their own set,
    so that the average alignment between the sets is good enough.

    After alignment, the L2 distance, without further alignemt is computed
    and reported as the GMM overlap metric. The results are saved in a .json
    file in the output directory.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    ll = setup_logger(rank=rank, log_dir=args.output_dir)
    ps_models, gt_models = parse_inputs(
        args.input_dir, args.ground_truth_dir, ll
    )

    # first the rank 0 process samples N_sample random pairs and chunks the
    # pairs over the available ranks to do alignment
    n_ps = len(ps_models)
    n_gt = len(gt_models)
    n_total = n_ps * n_gt

    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]  # assign GPU based on rank
    device = torch.device(f"cuda:{gpu_id}")
    ll.info(f"Using GPU: {torch.cuda.get_device_name(device)}")

    if rank == 0:
        ll.info(f"Total number of pseudo-atomic models: {n_ps}")
        ll.info(f"Total number of ground-truth models: {n_gt}")
        ll.info(f"Total number of pairs to process: {n_total}")
        sampled_ids = np.random.choice(
            n_total, size=min(args.N_sample, n_total), replace=False
        )
        sampled_ids_chunks = np.array_split(sampled_ids, size)
    else:
        sampled_ids_chunks = None

    sampled_ids_local = comm.scatter(sampled_ids_chunks, root=0)
    results_local = {}
    for sampled_id in sampled_ids_local:
        ii_ps, jj_gt = divmod(sampled_id, n_gt)
        ps_model = ps_models[ii_ps]
        gt_model = gt_models[jj_gt]

        ps_weights, ps_means, ps_covariances = cif_to_gmm(
            ps_model, device=device, ll=ll
        )
        gt_weights, gt_means, gt_covariances = cif_to_gmm(
            gt_model, device=device, ll=ll
        )

        ll.info(f"Rank {rank} processing sampled pair: pseudo-atomic \
                model {ps_model} and ground-truth model {gt_model}.")
        with timed(
            ll,
            f"register_gmms_rigid_{os.path.basename(ps_model)}"
            + f"_vs_{os.path.basename(gt_model)}",
        ):
            registration_result = register_gmms_rigid(
                weights_moving=ps_weights,
                means_moving=ps_means,
                covariances_moving=ps_covariances,
                weights_fixed=gt_weights,
                means_fixed=gt_means,
                covariances_fixed=gt_covariances,
                initial_rotation_vector=None,
                initial_translation=None,
                adam_steps=200,
                adam_learning_rate=0.05,
                lbfgs_steps=50,
                jitter=1e-8,
                ll=ll,
            )

        ll.info(f"Registration result for {ps_model} vs {gt_model}:")
        ll.info(
            f"Rotation matrix:\n{registration_result.rotation.cpu().numpy()}"
        )
        ll.info(f"Translation vector:\
                \n{registration_result.translation.cpu().numpy()}")
        ll.info(f"L2 squared distance:\
                 {registration_result.l2_squared.item()}")

        results_local[(ps_model, gt_model)] = {
            "rotation": registration_result.rotation.cpu().numpy(),
            "translation": registration_result.translation.cpu().numpy(),
            "l2_squared": registration_result.l2_squared.item(),
        }

    # now wait for all samples to finish being processed, gather the results
    # from all ranks compute the average rotation and translation using Lie
    # algebraic averaging and save the results to a .json file in the
    # output directory
    comm.Barrier()
    results_list = comm.gather(results_local, root=0)
    if rank == 0:
        # for Lie algebraic averaging, we need to construct a list of
        # matrices M = [R_i | t_i] where R_i is the rotation matrix and t_i
        # is the translation vector for each pair of models
        Lie_matrix = []
        results = {}
        for result in results_list:
            for key, sub_dict in result.items():
                # the key needs to be transformed to a string from a tuple
                # of strings to be JSON serializable
                key_str = f"{key[0]}|{key[1]}"
                results[key_str] = {}
                for sub_key, sub_value in sub_dict.items():
                    results[key_str][sub_key] = (
                        sub_value.tolist()
                        if isinstance(sub_value, np.ndarray)
                        else sub_value
                    )
                R = sub_dict["rotation"]
                t = sub_dict["translation"]
                M = np.eye(4)
                M[:3, :3] = R
                M[:3, 3] = t
                Lie_matrix.append(M)
        ll.info(f"Gathered results from all ranks. \
                Total number of results: {len(results)}")

        ll.info(f"Performing Lie algebraic averaging on \
                {len(Lie_matrix)} transformations.")
        Lie_matrix = np.stack(Lie_matrix, axis=0)
        Lie_tensor = torch.from_numpy(Lie_matrix).to(
            device=device, dtype=torch.float32
        )
        Lie_average, info = mean_se3(Lie_tensor)
        Lie_average_np = Lie_average.cpu().numpy()

        ll.info(
            f"Lie algebraic averaging result: \n{Lie_average.cpu().numpy()}"
        )
        ll.info(f"Lie averaging converged: {info['converged']}, iterations: \
                {info['iterations']}, residual_norm: {info['residual_norm']}")

        results["Lie_average"] = {
            "rotation": Lie_average_np[:3, :3].tolist(),
            "translation": Lie_average_np[:3, 3].tolist(),
            "converged": info["converged"],
            "iterations": info["iterations"],
            "residual_norm": info["residual_norm"],
        }

        with open(
            os.path.join(args.output_dir, "gmm_alignment.json"), "w"
        ) as f:
            json.dump(results, f, indent=4)
        ll.info(f"Saved GMM overlap results to \
                {os.path.join(args.output_dir, 'gmm_overlap_results.json')}")

        # apply the average transformation to an example
        # pseudo-atomic model and save the registered model
        # to the output directory
        example_ps_model = ps_models[0]
        ps_weights, ps_means, ps_covariances = cif_to_gmm(
            example_ps_model, device=device, ll=ll
        )
        ps_means_transformed, ps_covariances_transformed = transform_gmm(
            ps_means,
            ps_covariances,
            Lie_average[:3, :3],
            Lie_average[:3, 3],
        )

        # save the rotated and translated pseudo-atomic model to
        # the output directory
        outfilename = (
            "registered_"
            + os.path.basename(os.path.dirname(example_ps_model))
            + "_"
            + os.path.basename(example_ps_model)
        )
        outfilepath = os.path.join(args.output_dir, outfilename)
        ll.info(f"Saving registered pseudo-atomic model to {outfilepath}")
        gmm_to_cif(
            weights=ps_weights,
            means=registration_result.transformed_means,
            covariances=registration_result.transformed_covariances,
            filename=outfilepath,
        )

    else:
        Lie_average_np = None

    comm.Barrier()
    Lie_average_np = comm.bcast(Lie_average_np, root=0)
    ll.info(
        f"Rank {rank} received Lie average transformation: \n{Lie_average_np}"
    )

    # With the average transformation, we can compute the L2 distance
    # between all pseudo-atomic models and
    # ground-truth models without further alignment

    start = rank * n_total // size
    stop = (rank + 1) * n_total // size
    results_local = {}
    for ii in range(start, stop):
        ii_ps, jj_gt = divmod(ii, n_gt)
        ps_model = ps_models[ii_ps]
        gt_model = gt_models[jj_gt]

        ps_weights, ps_means, ps_covariances = cif_to_gmm(
            ps_model, device=device, ll=ll
        )
        ps_means_transformed, ps_covariances_transformed = transform_gmm(
            ps_means,
            ps_covariances,
            torch.from_numpy(Lie_average_np[:3, :3]).to(
                device=device, dtype=torch.float32
            ),
            torch.from_numpy(Lie_average_np[:3, 3]).to(
                device=device, dtype=torch.float32
            ),
        )
        ps_weights = normalize_weights(ps_weights)
        gt_weights, gt_means, gt_covariances = cif_to_gmm(
            gt_model, device=device, ll=ll
        )
        gt_weights = normalize_weights(gt_weights)
        with timed(
            ll,
            f"gmm_l2_squared_{os.path.basename(ps_model)}\
                _vs_{os.path.basename(gt_model)}",
        ):
            l2_squared = gmm_l2_squared(
                ps_weights,
                ps_means_transformed,
                ps_covariances_transformed,
                gt_weights,
                gt_means,
                gt_covariances,
                jitter=1e-8,
                clamp=True,
            )
            l2_squared_ctrl = gmm_l2_squared(
                ps_weights,
                ps_means,
                ps_covariances,
                gt_weights,
                gt_means,
                gt_covariances,
                jitter=1e-8,
                clamp=False,
            )  # compute the L2 distance without alignment as a sanity check
        ll.info(f"L2 squared distance between {ps_model} and \
                {gt_model}: {l2_squared.item()}")
        ll.info(f"L2 squared distance without alignment between \
                {ps_model} and {gt_model}: {l2_squared_ctrl.item()}")
        results_local[(f"{ps_model}|{gt_model}")] = l2_squared.item()

    # collect all results and save to a .json file in the output directory
    comm.Barrier()
    results_list = comm.gather(results_local, root=0)
    if rank == 0:
        results = {}
        for result in results_list:
            results.update(result)
        with open(
            os.path.join(args.output_dir, "gmm_l2_squared.json"), "w"
        ) as f:
            json.dump(results, f, indent=4)
        ll.info(f"Saved GMM L2 squared distance results to \
                {os.path.join(args.output_dir, 'gmm_l2_squared.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute GMM overlap metric.")
    parser = add_arguments(parser)
    args = parser.parse_args()
    main(args)
