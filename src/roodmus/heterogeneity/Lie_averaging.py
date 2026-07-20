import torch
from torch import Tensor


def skew(v: Tensor) -> Tensor:
    """
    Convert vectors (..., 3) to skew-symmetric matrices (..., 3, 3).
    """
    if v.shape[-1] != 3:
        raise ValueError("Expected shape (..., 3)")

    x, y, z = v.unbind(dim=-1)
    zeros = torch.zeros_like(x)

    return torch.stack(
        (
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ),
        dim=-1,
    ).reshape(v.shape[:-1] + (3, 3))


def vee(Omega: Tensor) -> Tensor:
    """
    Inverse of skew() for skew-symmetric matrices (..., 3, 3).
    """
    if Omega.shape[-2:] != (3, 3):
        raise ValueError("Expected shape (..., 3, 3)")

    return torch.stack(
        (
            Omega[..., 2, 1],
            Omega[..., 0, 2],
            Omega[..., 1, 0],
        ),
        dim=-1,
    )


def se3_inverse(T: Tensor) -> Tensor:
    """
    Batched inverse of homogeneous SE(3) matrices.
    """
    if T.shape[-2:] != (4, 4):
        raise ValueError("T must have shape (..., 4, 4)")

    R = T[..., :3, :3]
    t = T[..., :3, 3]

    R_transpose = R.transpose(-1, -2)
    inverse_t = -(R_transpose @ t[..., None]).squeeze(-1)

    result = torch.zeros_like(T)
    result[..., :3, :3] = R_transpose
    result[..., :3, 3] = inverse_t
    result[..., 3, 3] = 1.0

    return result


def se3_exp(xi: Tensor) -> Tensor:
    """
    Batched SE(3) exponential map.

    Parameters
    ----------
    xi:
        Twists with shape (..., 6), ordered as [rho, phi].

    Returns
    -------
    Homogeneous transformations with shape (..., 4, 4).
    """
    if xi.shape[-1] != 6:
        raise ValueError("xi must have shape (..., 6)")

    rho = xi[..., :3]
    phi = xi[..., 3:]

    theta2 = torch.sum(phi * phi, dim=-1, keepdim=True)
    theta = torch.sqrt(theta2)

    Omega = skew(phi)
    Omega2 = Omega @ Omega

    eps = torch.finfo(xi.dtype).eps
    safe_theta2 = theta2.clamp_min(eps)
    safe_theta3 = (theta2 * theta).clamp_min(eps)

    # SO(3) left-Jacobian coefficients.
    B_regular = (1.0 - torch.cos(theta)) / safe_theta2
    C_regular = (theta - torch.sin(theta)) / safe_theta3

    B_series = 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0
    C_series = 1.0 / 6.0 - theta2 / 120.0 + theta2 * theta2 / 5040.0

    small = theta2 < 1e-8
    B = torch.where(small, B_series, B_regular)
    C = torch.where(small, C_series, C_regular)

    eye3 = torch.eye(
        3,
        dtype=xi.dtype,
        device=xi.device,
    ).expand(xi.shape[:-1] + (3, 3))

    V = eye3 + B[..., None] * Omega + C[..., None] * Omega2

    R = so3_exp(phi)
    t = (V @ rho[..., None]).squeeze(-1)

    T = torch.zeros(
        xi.shape[:-1] + (4, 4),
        dtype=xi.dtype,
        device=xi.device,
    )

    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0

    return T


def se3_log(T: Tensor) -> Tensor:
    """
    Batched SE(3) logarithm map.

    Parameters
    ----------
    T:
        Homogeneous transformations with shape (..., 4, 4).

    Returns
    -------
    Twists with shape (..., 6), ordered as [rho, phi].
    """
    if T.shape[-2:] != (4, 4):
        raise ValueError("T must have shape (..., 4, 4)")

    R = T[..., :3, :3]
    t = T[..., :3, 3]

    phi = so3_log(R)

    theta2 = torch.sum(phi * phi, dim=-1, keepdim=True)
    theta = torch.sqrt(theta2)

    Omega = skew(phi)
    Omega2 = Omega @ Omega

    eps = torch.finfo(T.dtype).eps
    safe_theta2 = theta2.clamp_min(eps)

    half_theta = 0.5 * theta
    sin_half = torch.sin(half_theta).abs().clamp_min(eps)

    # Coefficient in:
    #
    # V^{-1} = I - 1/2 Omega + D Omega^2
    #
    # D = [1 - (theta/2) cot(theta/2)] / theta^2
    cot_half = torch.cos(half_theta) / sin_half
    D_regular = (1.0 - half_theta * cot_half) / safe_theta2

    D_series = 1.0 / 12.0 + theta2 / 720.0 + theta2 * theta2 / 30240.0

    D = torch.where(
        theta2 < 1e-8,
        D_series,
        D_regular,
    )

    eye3 = torch.eye(
        3,
        dtype=T.dtype,
        device=T.device,
    ).expand(T.shape[:-2] + (3, 3))

    V_inv = eye3 - 0.5 * Omega + D[..., None] * Omega2

    rho = (V_inv @ t[..., None]).squeeze(-1)

    return torch.cat((rho, phi), dim=-1)


def so3_exp(phi: Tensor) -> Tensor:
    """
    Batched SO(3) exponential map.

    Parameters
    ----------
    phi:
        Rotation vectors with shape (..., 3).

    Returns
    -------
    Rotation matrices with shape (..., 3, 3).
    """
    if phi.shape[-1] != 3:
        raise ValueError("phi must have shape (..., 3)")

    theta2 = torch.sum(phi * phi, dim=-1, keepdim=True)
    theta = torch.sqrt(theta2)

    Omega = skew(phi)
    Omega2 = Omega @ Omega

    # Safe denominators prevent invalid values in the branch not selected
    # by torch.where.
    eps = torch.finfo(phi.dtype).eps
    safe_theta = theta.clamp_min(eps)
    safe_theta2 = theta2.clamp_min(eps)

    # sin(theta) / theta
    A_regular = torch.sin(theta) / safe_theta
    A_series = 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0

    # (1 - cos(theta)) / theta^2
    B_regular = (1.0 - torch.cos(theta)) / safe_theta2
    B_series = 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0

    small = theta2 < 1e-8
    A = torch.where(small, A_series, A_regular)
    B = torch.where(small, B_series, B_regular)

    eye = torch.eye(
        3,
        dtype=phi.dtype,
        device=phi.device,
    ).expand(phi.shape[:-1] + (3, 3))

    return eye + A[..., None] * Omega + B[..., None] * Omega2


def so3_log(R: Tensor) -> Tensor:
    """
    Batched SO(3) logarithm map.

    Parameters
    ----------
    R:
        Rotation matrices with shape (..., 3, 3).

    Returns
    -------
    Rotation vectors with shape (..., 3).

    Notes
    -----
    This implementation is intended for relative rotations that remain
    comfortably below pi radians. That is normally true when averaging
    a compact cluster of transformations.
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError("R must have shape (..., 3, 3)")

    trace = torch.diagonal(
        R,
        dim1=-2,
        dim2=-1,
    ).sum(dim=-1)

    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    theta2 = theta * theta

    # vee((R - R^T) / 2) = sin(theta) * rotation_axis
    sin_axis = vee(0.5 * (R - R.transpose(-1, -2)))

    eps = torch.finfo(R.dtype).eps
    sin_theta = torch.sin(theta)
    safe_sin_theta = sin_theta.abs().clamp_min(eps)

    factor_regular = theta / safe_sin_theta

    # theta / sin(theta) near zero.
    factor_series = 1.0 + theta2 / 6.0 + 7.0 * theta2 * theta2 / 360.0

    factor = torch.where(
        theta2 < 1e-8,
        factor_series,
        factor_regular,
    )

    return factor[..., None] * sin_axis


@torch.no_grad()
def mean_se3(
    transforms: Tensor,
    weights: Tensor | None = None,
    tolerance: float | None = None,
    max_iterations: int = 100,
) -> tuple[Tensor, dict[str, float | int | bool]]:
    """
    Compute the iterative Lie-algebraic mean of SE(3) transforms.

    Parameters
    ----------
    transforms:
        Tensor with shape (N, 4, 4). It may already reside on a GPU.
    weights:
        Optional nonnegative tensor with shape (N,).
    tolerance:
        Convergence threshold for the mean residual twist. Defaults to
        1e-6 for float32 and 1e-10 for float64.
    max_iterations:
        Maximum number of fixed-point iterations.

    Returns
    -------
    mean:
        Mean transformation with shape (4, 4), on the same device and
        with the same dtype as transforms.
    information:
        Dictionary containing convergence information.
    """
    if transforms.ndim != 3 or transforms.shape[-2:] != (4, 4):
        raise ValueError("transforms must have shape (N, 4, 4)")

    number_of_transforms = transforms.shape[0]

    if number_of_transforms == 0:
        raise ValueError("At least one transformation is required")

    if not transforms.is_floating_point():
        raise TypeError("transforms must use a floating-point dtype")

    if tolerance is None:
        tolerance = 1e-10 if transforms.dtype == torch.float64 else 1e-6

    if weights is None:
        weights = torch.full(
            (number_of_transforms,),
            1.0 / number_of_transforms,
            dtype=transforms.dtype,
            device=transforms.device,
        )
    else:
        weights = weights.to(
            device=transforms.device,
            dtype=transforms.dtype,
        )

        if weights.shape != (number_of_transforms,):
            raise ValueError("weights must have shape (N,)")

        if torch.any(weights < 0):
            raise ValueError("weights must be nonnegative")

        weight_sum = weights.sum()

        if weight_sum <= 0:
            raise ValueError("At least one weight must be positive")

        weights = weights / weight_sum

    # Since the poses are assumed to be clustered, one observed pose is
    # normally a good initial estimate.
    current_mean = transforms[0].clone()

    final_error = float("inf")

    for iteration in range(max_iterations):
        current_inverse = se3_inverse(current_mean)

        # Broadcasting:
        # (4, 4) @ (N, 4, 4) -> (N, 4, 4)
        relative_transforms = current_inverse @ transforms

        # All N logarithms are evaluated in one batched GPU operation.
        residual_twists = se3_log(relative_transforms)

        mean_residual = torch.sum(
            weights[:, None] * residual_twists,
            dim=0,
        )

        error_tensor = torch.linalg.vector_norm(mean_residual)
        final_error = float(error_tensor.item())

        if final_error < tolerance:
            return current_mean, {
                "converged": True,
                "iterations": iteration,
                "residual_norm": final_error,
            }

        current_mean = current_mean @ se3_exp(mean_residual)

    return current_mean, {
        "converged": False,
        "iterations": max_iterations,
        "residual_norm": final_error,
    }
