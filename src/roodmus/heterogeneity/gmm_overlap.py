"""
computes the GMM overlap metric between a series of input pseudo-atomic models
in one directory and a set of ground-truth structures
"""

import argparse
import glob2
import logging
import os
import gemmi
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
import torch
from torch import Tensor
from contextlib import contextmanager
from time import perf_counter


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
        help="Path to the output directory where the GMM \
            overlap results will be saved.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional information during execution.",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="ID of the GPU to use for computation \
            (default: 0). ID -1 means CPU only.",
    )
    return parser


def start_logger(
    filename: str, rank: int, level: int = logging.INFO
) -> logging.Logger:
    """Start a logger that writes to a file and the console."""
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(filename)
    fh.setLevel(level)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def parse_inputs(
    input_dir: str, ground_truth_dir: str, ll: logging.Logger
) -> Tuple[List[str], List[str]]:
    # load the first pseudo-atom model and ground-truth\
    # model and compute the GMM overlap
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
    ll.info(f"Found {len(gt_models)} ground-truth \
            models in {args.ground_truth_dir}.")
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
    if covariances.shape != (means.shape[0], 3, 3):
        raise ValueError("covariances must have shape [K, 3, 3].")
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
    transformed_covariances = (
        rotation.unsqueeze(0) @ covariances @ rotation.T.unsqueeze(0)
    )
    return transformed_means, transformed_covariances


def gaussian_overlap_matrix(
    means_a: Tensor,
    covariances_a: Tensor,
    means_b: Tensor,
    covariances_b: Tensor,
    *,
    jitter: float = 1e-8,
) -> Tensor:
    """
    Compute pairwise Gaussian overlap integrals.

    result[i, j] =
        integral N(x | means_a[i], covariances_a[i])
                 N(x | means_b[j], covariances_b[j]) dx

    Shapes:
        means_a:       [M, 3]
        covariances_a: [M, 3, 3]
        means_b:       [N, 3]
        covariances_b: [N, 3, 3]
        result:        [M, N]
    """
    dimension = means_a.shape[-1]

    differences = means_a[:, None, :] - means_b[None, :, :]  # [M, N, 3]
    covariance_sums = (
        covariances_a[:, None, :, :] + covariances_b[None, :, :, :]
    )  # [M, N, 3, 3]

    identity = torch.eye(
        dimension,
        dtype=covariance_sums.dtype,
        device=covariance_sums.device,
    )
    covariance_sums = covariance_sums + jitter * identity

    # Cholesky is preferable to explicitly forming matrix inverses.
    chol, info = torch.linalg.cholesky_ex(covariance_sums)

    if torch.any(info != 0):
        raise RuntimeError(
            "A covariance sum was not positive definite. "
            "Increase jitter or regularize the input covariances."
        )

    rhs = differences.unsqueeze(-1)  # [M, N, 3, 1]
    solved = torch.cholesky_solve(rhs, chol)
    mahalanobis_squared = (
        (differences.unsqueeze(-2) @ solved).squeeze(-1).squeeze(-1)
    )

    # log(det(S)) = 2 * sum(log(diag(cholesky(S))))
    log_determinants = 2.0 * torch.log(
        torch.diagonal(chol, dim1=-2, dim2=-1)
    ).sum(dim=-1)

    log_normalization = (
        dimension
        * torch.log(
            torch.tensor(
                2.0 * torch.pi,
                dtype=means_a.dtype,
                device=means_a.device,
            )
        )
        + log_determinants
    )

    return torch.exp(-0.5 * (mahalanobis_squared + log_normalization))


def gmm_inner_product(
    weights_a: Tensor,
    means_a: Tensor,
    covariances_a: Tensor,
    weights_b: Tensor,
    means_b: Tensor,
    covariances_b: Tensor,
    *,
    jitter: float = 1e-8,
) -> Tensor:
    overlaps = gaussian_overlap_matrix(
        means_a,
        covariances_a,
        means_b,
        covariances_b,
        jitter=jitter,
    )
    return torch.sum(weights_a[:, None] * weights_b[None, :] * overlaps)


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
    aa = gmm_inner_product(
        weights_a,
        means_a,
        covariances_a,
        weights_a,
        means_a,
        covariances_a,
        jitter=jitter,
    )
    bb = gmm_inner_product(
        weights_b,
        means_b,
        covariances_b,
        weights_b,
        means_b,
        covariances_b,
        jitter=jitter,
    )
    ab = gmm_inner_product(
        weights_a,
        means_a,
        covariances_a,
        weights_b,
        means_b,
        covariances_b,
        jitter=jitter,
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

    for _ in range(adam_steps):
        adam.zero_grad(set_to_none=True)
        loss = objective()
        if not torch.isfinite(loss):
            raise RuntimeError("Registration objective became non-finite.")
        loss.backward()
        adam.step()

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


def cif_to_gmm(
    filename: str, device: torch.device, ll: logging.Logger
) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert a CIF file to a GMM representation."""
    structure = gemmi.read_structure(filename)
    total_atoms = sum(model.count_atom_sites() for model in structure)
    ll.info(f"Converting {filename} to GMM representation \
            with {total_atoms} atoms.")

    weights = np.zeros(total_atoms, dtype=np.float32)
    means = np.zeros((total_atoms, 3), dtype=np.float32)
    covariances = np.zeros((total_atoms, 3, 3), dtype=np.float32)

    atm_idx = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    weights[atm_idx] = atom.occ
                    means[atm_idx] = np.array(
                        [atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float32
                    )
                    covariances[atm_idx] = (
                        np.eye(3) * atom.b_iso / 8.0 / np.pi**2
                    )
                    if atm_idx == 0:
                        ll.info(f"Atom: {atom.name}, \
                            Type: {atom.element.name}, \
                            Occ: {atom.occ}, \
                            B_iso: {atom.b_iso}")

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
    model.add_chain(chain)
    structure.add_model(model)

    for i in range(means.shape[0]):
        atom = gemmi.Atom()
        atom.name = f"CA{i + 1}"
        atom.pos.x = float(means[i, 0].item())
        atom.pos.y = float(means[i, 1].item())
        atom.pos.z = float(means[i, 2].item())
        atom.occ = float(weights[i].item())
        atom.b_iso = float(torch.trace(covariances[i]).item() * 8.0 * np.pi**2)
        residue = gemmi.Residue()
        residue.name = "GLY"
        residue.seqid.num = i + 1
        residue.add_atom(atom)
        chain.add_residue(residue)

    structure.write_pdb(filename)


def gen_example_gmms(
    N_gmm: int, device: torch.device, ll: logging.Logger
) -> Tuple[
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tuple[Tensor, Tensor]
]:
    """Generate example GMMs for testing."""
    # Define parameters for the first GMM (moving)
    weights_moving = torch.ones(N_gmm, device=device) / N_gmm
    means_moving = torch.randn(N_gmm, 3, device=device) * 10.0
    covariances_moving = (
        torch.eye(3, device=device).unsqueeze(0).repeat(N_gmm, 1, 1) * 0.5
    )

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


def main(args):
    ll = start_logger(
        os.path.join(args.output_dir, "gmm_overlap.log"),
        rank=0,
        level=logging.INFO,
    )
    ps_models, gt_models = parse_inputs(
        args.input_dir, args.ground_truth_dir, ll
    )

    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        ll.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    elif args.gpu_id < 0:
        device = torch.device("cpu")
        ll.info("Using CPU for computation.")
    else:
        raise RuntimeError(f"GPU ID {args.gpu_id} is not available.")

    ii = 0
    jj = 0
    ps_model = ps_models[ii]
    gt_model = gt_models[jj]

    # ps_weights, ps_means, ps_covariances = cif_to_gmm(ps_model, device, ll)
    # gt_weights, gt_means, gt_covariances = cif_to_gmm(gt_model, device, ll)

    for N_gmm in [50, 100, 200, 400, 800, 1600]:
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

        ll.info(f"Registration result for {ps_model} vs {gt_model}:")
        ll.info(
            f"Rotation matrix:\n{registration_result.rotation.cpu().numpy()}"
        )
        ll.info(f"Translation vector:\n\
                {registration_result.translation.cpu().numpy()}")
        ll.info(
            f"L2 squared distance: {registration_result.l2_squared.item()}"
        )

        ll.info("True transformation for comparison:")
        ll.info(
            f"True rotation matrix:\n{true_transformation[0].cpu().numpy()}"
        )
        ll.info(
            f"True translation vector:\n{true_transformation[1].cpu().numpy()}"
        )

        # save the rotated and translated pseudo-atomic
        # model to the output directory
        outfilename = os.path.join(
            args.output_dir, f"registered_{os.path.basename(ps_model)}"
        )
        ll.info(f"Saving registered pseudo-atomic model to {outfilename}")
        gmm_to_cif(
            weights=ps_weights,
            means=registration_result.transformed_means,
            covariances=registration_result.transformed_covariances,
            filename=outfilename,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute GMM overlap metric.")
    parser = add_arguments(parser)
    args = parser.parse_args()
    main(args)
