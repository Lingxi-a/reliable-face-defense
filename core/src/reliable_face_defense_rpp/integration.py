from __future__ import annotations

import torch


def certificate_branch_from_blue_route(route) -> str:
    """
    Preserve the branch corresponding to the Blue-selected primitive.

      FacePoison -> detection certificate
      NullSwap   -> identity certificate

    This is sufficient for the project's OR task reliability:
      D_fail OR I_fail
    """
    value = getattr(route, "value", route)
    value = str(value).lower()
    if value == "facepoison":
        return "detection"
    if value == "nullswap":
        return "identity"
    raise ValueError(f"Unsupported Blue route for RPP certificate: {route!r}")


def resolve_rpp_epsilon_norm(
    *,
    clean01: torch.Tensor,
    protected01: torch.Tensor,
    blue_chosen,
) -> tuple[float, str]:
    """
    Resolve the RPP perturbation box.

    FacePoison:
      use Blue's verified sample-specific epsilon_px / 255.

    NullSwap:
      the paper-faithful NullSwap route has no invented inference epsilon.
      Therefore RPP uses the actual initial protected-image L_inf envelope.
      This does not create a new NullSwap strength knob; it prevents RPP from
      increasing the already-established image-space envelope.

    Returns:
      (epsilon_norm, provenance)
    """
    eps_px = getattr(blue_chosen, "epsilon_px", None)
    if eps_px is not None:
        return float(eps_px) / 255.0, "blue_verified_facepoison_epsilon"

    delta0 = protected01.detach() - clean01.detach()
    eps = float(delta0.abs().max().cpu())
    return eps, "nullswap_initial_actual_linf_envelope"
