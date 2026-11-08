from __future__ import annotations

import torch

from .contracts import (
    Branch,
    RPPConfig,
    RPPStepRecord,
    RPPResult,
)
from .constraints import snapshot_from_terms
from .projection import project_delta_box


class RPPSolver:
    r"""
    Complete Reliability-Preserving Projection solver.

    Per refinement step:

      (a) perceptual descent
          delta_tilde = delta - eta_v * grad D_per

      (b) if the Blue certificate is violated:
          sequentially project every violated g_m(delta)<=0 using

          delta <- delta
                   - rho * [g_m(delta)]_+
                           / (||grad g_m||_2^2 + xi)
                           * stopgrad(grad g_m)

          repeat up to K_repair passes and apply Pi_B after each projection.

      (c) hard recheck:
          accept repaired candidate iff the established Blue certificate still
          holds on CLEAN + all active Red constraints.

          If hard recheck fails -> rollback to previous reliable delta.

      Optional project safety gate:
          if visual objective did not improve, keep previous iterate.

    No QP solver and no second-order gradient are used.
    """

    def __init__(
        self,
        *,
        config: RPPConfig,
        objective,
        constraint_provider,
        reliability_checker,
    ):
        self.config = config
        self.objective = objective
        self.constraint_provider = constraint_provider
        self.reliability_checker = reliability_checker

    @staticmethod
    def _term_key(t):
        return (t.environment, t.branch, t.name)

    def _target_terms(
        self,
        *,
        clean01,
        delta,
        active_environments,
        certificate_branch: Branch,
    ):
        terms = self.constraint_provider.evaluate(
            clean01=clean01,
            delta=delta,
            active_environments=active_environments,
        )
        return tuple(t for t in terms if t.branch == certificate_branch)

    def _proxy_snapshot(
        self,
        *,
        clean01,
        delta,
        active_environments,
        certificate_branch,
        include_terms=False,
    ):
        terms = self.constraint_provider.evaluate(
            clean01=clean01,
            delta=delta,
            active_environments=active_environments,
        )
        return snapshot_from_terms(
            terms=terms,
            certificate_branch=certificate_branch,
            active_environments=active_environments,
            include_terms=include_terms,
        )

    def _hard_snapshot(
        self,
        *,
        clean01,
        delta,
        active_environments,
        certificate_branch,
    ):
        return self.reliability_checker.check(
            clean01=clean01,
            delta=delta,
            active_environments=active_environments,
            certificate_branch=certificate_branch,
        )

    def _visual_value(self, clean01, delta) -> float:
        with torch.no_grad():
            value = self.objective(clean01, delta)
        return float(value.detach().cpu())

    def _perceptual_descent(
        self,
        *,
        clean01,
        delta,
        epsilon_norm,
    ):
        d = delta.detach().clone().requires_grad_(True)
        loss = self.objective(clean01, d)
        grad = torch.autograd.grad(
            loss,
            d,
            create_graph=False,
            retain_graph=False,
        )[0]
        candidate = d.detach() - self.config.eta_visual * grad.detach()
        candidate = project_delta_box(
            clean01=clean01,
            delta=candidate,
            epsilon_norm=epsilon_norm,
        )
        return candidate.detach()

    def _sequential_repair(
        self,
        *,
        clean01,
        candidate,
        epsilon_norm,
        active_environments,
        certificate_branch,
    ):
        current = candidate.detach()
        projection_count = 0
        passes_used = 0

        for repair_pass in range(self.config.k_repair):
            passes_used = repair_pass + 1

            # Freeze the order from the currently violated certificate terms.
            target = self._target_terms(
                clean01=clean01,
                delta=current.detach().clone().requires_grad_(True),
                active_environments=active_environments,
                certificate_branch=certificate_branch,
            )
            violated_keys = [
                self._term_key(t)
                for t in target
                if float(t.value.detach().cpu()) > 0.0
            ]

            if not violated_keys:
                break

            for key in violated_keys:
                d = current.detach().clone().requires_grad_(True)
                fresh = self._target_terms(
                    clean01=clean01,
                    delta=d,
                    active_environments=active_environments,
                    certificate_branch=certificate_branch,
                )
                by_key = {self._term_key(t): t for t in fresh}
                if key not in by_key:
                    raise RuntimeError(f"Constraint disappeared during repair: {key}")

                term = by_key[key]
                g = term.value
                if float(g.detach().cpu()) <= 0.0:
                    continue

                grad = torch.autograd.grad(
                    g,
                    d,
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=False,
                )[0]

                # Eq.(25), with stop-gradient direction.
                grad_det = grad.detach()
                denom = grad_det.pow(2).sum() + self.config.xi
                scale = (
                    self.config.rho_repair
                    * torch.relu(g.detach())
                    / denom
                )
                repaired = d.detach() - scale * grad_det
                current = project_delta_box(
                    clean01=clean01,
                    delta=repaired,
                    epsilon_norm=epsilon_norm,
                ).detach()
                projection_count += 1

            # Early exit if proxy certificate is restored.
            snap = self._proxy_snapshot(
                clean01=clean01,
                delta=current,
                active_environments=active_environments,
                certificate_branch=certificate_branch,
            )
            if snap.certificate_reliable:
                break

        return current, passes_used, projection_count

    def refine(
        self,
        *,
        clean01: torch.Tensor,
        protected01: torch.Tensor,
        epsilon_norm: float,
        certificate_branch: Branch,
        active_environments: tuple[str, ...] = (),
    ) -> RPPResult:
        if clean01.shape != protected01.shape:
            raise ValueError("clean/protected shape mismatch")
        if clean01.ndim != 4:
            raise ValueError("Expected batched BCHW tensors")
        if certificate_branch not in {"detection", "identity"}:
            raise ValueError("certificate_branch must be detection or identity")

        delta0 = (protected01.detach() - clean01.detach())
        delta0 = project_delta_box(
            clean01=clean01,
            delta=delta0,
            epsilon_norm=epsilon_norm,
        ).detach()

        initial_protected = torch.clamp(clean01 + delta0, 0.0, 1.0).detach()

        initial_snapshot = self._hard_snapshot(
            clean01=clean01,
            delta=delta0,
            active_environments=active_environments,
            certificate_branch=certificate_branch,
        )

        # RPP is a refinement/preservation mechanism, not a coverage creator.
        if not initial_snapshot.certificate_reliable:
            return RPPResult(
                status="initial_unreliable_no_refinement",
                refined=initial_protected,
                initial=initial_protected,
                delta_initial=delta0,
                delta_final=delta0,
                epsilon_norm=float(epsilon_norm),
                certificate_branch=certificate_branch,
                active_environments=tuple(active_environments),
                initial_snapshot=initial_snapshot,
                final_snapshot=initial_snapshot,
                steps=(),
                accepted_steps=0,
                rollback_steps=0,
                total_sequential_projections=0,
            )

        current = delta0
        records = []
        accepted = 0
        rollbacks = 0
        total_projections = 0

        for step in range(self.config.steps):
            visual_before = self._visual_value(clean01, current)

            candidate = self._perceptual_descent(
                clean01=clean01,
                delta=current,
                epsilon_norm=epsilon_norm,
            )
            visual_after_descent = self._visual_value(clean01, candidate)

            snap_descent = self._proxy_snapshot(
                clean01=clean01,
                delta=candidate,
                active_environments=active_environments,
                certificate_branch=certificate_branch,
                include_terms=True,
            )

            repair_triggered = not snap_descent.certificate_reliable
            repair_passes_used = 0
            projection_count = 0

            if repair_triggered:
                repaired, repair_passes_used, projection_count = (
                    self._sequential_repair(
                        clean01=clean01,
                        candidate=candidate,
                        epsilon_norm=epsilon_norm,
                        active_environments=active_environments,
                        certificate_branch=certificate_branch,
                    )
                )
            else:
                repaired = candidate

            total_projections += projection_count
            visual_after_repair = self._visual_value(clean01, repaired)

            # Critical hard recheck after repair/descent.
            snap_repair = self._hard_snapshot(
                clean01=clean01,
                delta=repaired,
                active_environments=active_environments,
                certificate_branch=certificate_branch,
            )

            visual_ok = (
                visual_after_repair
                <= visual_before - self.config.visual_improvement_tolerance
            )

            if not snap_repair.certificate_reliable:
                accept = False
                reason = (
                    "hard recheck failed after repair; rolled back to previous "
                    "reliable iterate"
                )
            elif (
                self.config.accept_only_if_visual_improves
                and not visual_ok
            ):
                accept = False
                reason = (
                    "reliability preserved but perceptual objective did not "
                    "improve; kept previous reliable iterate"
                )
            else:
                accept = True
                reason = (
                    "candidate accepted: Blue certificate remains valid on "
                    "clean + all active Red environments"
                )

            if accept:
                current = repaired.detach()
                accepted += 1
            else:
                # Eq./algorithm acceptance-hold rule: exact rollback.
                current = current.detach()
                rollbacks += 1

            records.append(
                RPPStepRecord(
                    step=step,
                    visual_before=visual_before,
                    visual_after_perceptual_descent=visual_after_descent,
                    visual_after_repair=visual_after_repair,
                    accepted=accept,
                    rolled_back=not accept,
                    repair_triggered=repair_triggered,
                    repair_passes_used=repair_passes_used,
                    sequential_projection_count=projection_count,
                    snapshot_after_descent=snap_descent,
                    snapshot_after_repair=snap_repair,
                    reason=reason,
                )
            )

        final_snapshot = self._hard_snapshot(
            clean01=clean01,
            delta=current,
            active_environments=active_environments,
            certificate_branch=certificate_branch,
        )

        if not final_snapshot.certificate_reliable:
            # Defensive invariant: this should be unreachable because rollback
            # starts from a reliable point and only accepts reliable candidates.
            raise RuntimeError(
                "RPP invariant violated: final output lost Blue certificate."
            )

        refined = torch.clamp(clean01 + current, 0.0, 1.0).detach()

        return RPPResult(
            status="ok",
            refined=refined,
            initial=initial_protected,
            delta_initial=delta0,
            delta_final=current.detach(),
            epsilon_norm=float(epsilon_norm),
            certificate_branch=certificate_branch,
            active_environments=tuple(active_environments),
            initial_snapshot=initial_snapshot,
            final_snapshot=final_snapshot,
            steps=tuple(records),
            accepted_steps=accepted,
            rollback_steps=rollbacks,
            total_sequential_projections=total_projections,
        )
