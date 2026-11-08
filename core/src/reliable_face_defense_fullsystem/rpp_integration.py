from __future__ import annotations

from reliable_face_defense_rpp.contracts import RPPConfig
from reliable_face_defense_rpp.visual import (
    PerceptualObjective,
    LPIPSAlexNet01,
)
from reliable_face_defense_rpp.propagation import (
    DifferentiableTransformRegistry,
)

from .rpp_or import (
    ORConstraintSystem,
    ORRPPSolver,
    build_environment_certificate,
)


def resolve_epsilon_norm(clean01, blue_candidate):
    if blue_candidate.epsilon_px is not None:
        return float(blue_candidate.epsilon_px)/255.0
    # NullSwap has no paper-defined inference epsilon. Use actual initial
    # image-space envelope only, preventing RPP from expanding it.
    return float(
        (blue_candidate.protected-clean01)
        .detach().abs().max().cpu()
    )


class RPPFactory:
    def __init__(self, *, cfg, device):
        lpips_model = LPIPSAlexNet01(device=device)
        self.objective = PerceptualObjective(
            lambda_pix=cfg.lambda_pix,
            lambda_lpips=cfg.lambda_lpips,
            lpips_model=lpips_model,
        )
        self.solver = ORRPPSolver(
            config=RPPConfig(
                steps=cfg.steps,
                eta_visual=cfg.eta_visual,
                rho_repair=cfg.rho_repair,
                xi=cfg.xi,
                k_repair=cfg.k_repair,
                lambda_pix=cfg.lambda_pix,
                lambda_lpips=cfg.lambda_lpips,
                accept_only_if_visual_improves=cfg.accept_only_if_visual_improves,
            ),
            objective=self.objective,
        )
        self.diff_registry = (
            DifferentiableTransformRegistry
            .from_03bc_hard_red_pool_with_ste()
        )

    def refine(
        self,
        *,
        clean01,
        blue_candidate,
        context,
        reliability,
        hard_propagation,
    ):
        certificate = build_environment_certificate(blue_candidate)
        system = ORConstraintSystem(
            clean01=clean01,
            context=context,
            reliability=reliability,
            diff_transform_registry=self.diff_registry,
            hard_propagation=hard_propagation,
        )
        return self.solver.refine(
            system=system,
            protected01=blue_candidate.protected,
            epsilon_norm=resolve_epsilon_norm(clean01,blue_candidate),
            certificate=certificate,
        )
