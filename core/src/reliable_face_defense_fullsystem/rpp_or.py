from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal
import torch

from reliable_face_defense_agents.blue.contracts import Route
from reliable_face_defense_rpp.projection import project_delta_box


Branch = Literal["detection","identity"]


@dataclass(frozen=True)
class EnvironmentCertificate:
    branch_by_environment: dict[str,Branch]

    def branch(self, environment):
        return self.branch_by_environment[environment]


@dataclass(frozen=True)
class ORSnapshot:
    per_environment: dict
    task_reliable: bool
    certificate_reliable: bool
    certificate: dict[str,str]


@dataclass(frozen=True)
class ORRPPStep:
    step: int
    accepted: bool
    rolled_back: bool
    repair_triggered: bool
    projections: int
    repair_passes: int
    visual_before: float
    visual_candidate: float
    visual_after_repair: float
    snapshot_after_descent: ORSnapshot
    snapshot_after_repair: ORSnapshot


@dataclass(frozen=True)
class ORRPPResult:
    status: str
    refined: torch.Tensor
    delta_initial: torch.Tensor
    delta_final: torch.Tensor
    initial_snapshot: ORSnapshot
    final_snapshot: ORSnapshot
    certificate: EnvironmentCertificate
    steps: tuple[ORRPPStep,...]
    accepted_steps: int
    rollback_steps: int
    total_projections: int


def build_environment_certificate(blue_candidate) -> EnvironmentCertificate:
    """
    Build one sufficient protection certificate PER ENVIRONMENT.

    This fixes a subtle issue in the earlier RPP version: under OR task
    reliability, clean may be protected by Detection while an active JPEG
    condition is protected by Identity. Requiring one global branch for every
    environment would be unnecessarily strict and is not equivalent to the
    task-level OR rule.

    For each environment:
      - if only one branch is protected, preserve that branch;
      - if both are protected, prefer the branch targeted by the selected route.
    """
    preferred = (
        "detection"
        if blue_candidate.route is Route.FACEPOISON
        else "identity"
    )
    mapping = {}
    for item in blue_candidate.outcomes:
        ev = item.evaluation
        d = not ev.detection_success
        i = not ev.identity_success
        if not (d or i):
            raise ValueError(
                f"Cannot build RPP certificate: {item.environment} is unreliable."
            )
        if d and i:
            branch = preferred
        elif d:
            branch = "detection"
        else:
            branch = "identity"
        mapping[item.environment] = branch
    return EnvironmentCertificate(mapping)


class ORConstraintSystem:
    """
    Sample-specific differentiable constraints for clean + active Red envs.

    The reliability system owns actual model thresholds and clean references.
    The differentiable transform registry owns the propagation gradient path.
    """
    def __init__(
        self,
        *,
        clean01,
        context,
        reliability,
        diff_transform_registry,
        hard_propagation,
    ):
        self.clean01 = clean01
        self.context = context
        self.reliability = reliability
        self.diff = diff_transform_registry
        self.hard = hard_propagation

    def differentiable_terms(self, delta, environments):
        protected = torch.clamp(self.clean01 + delta, 0, 1)
        out = {}
        for env in environments:
            y = protected if env == "clean" else self.diff.apply(env, protected)
            vals = self.reliability.constraint_values(y, self.context)
            out[env] = {
                "detection": tuple(vals["detection"].items()),
                "identity": tuple(vals["identity"].items()),
            }
        return out

    def hard_snapshot(self, delta, certificate):
        protected = torch.clamp(self.clean01 + delta,0,1)
        per = {}
        cert_ok = True
        task_ok = True
        for env, branch in certificate.branch_by_environment.items():
            y = protected if env=="clean" else self.hard.apply(env, protected)
            ev = self.reliability.evaluate(y, self.context)
            dprot = not ev.detection_success
            iprot = not ev.identity_success
            env_task = dprot or iprot
            env_cert = dprot if branch=="detection" else iprot
            task_ok = task_ok and env_task
            cert_ok = cert_ok and env_cert
            per[env] = {
                "detection_protected": dprot,
                "identity_protected": iprot,
                "task_reliable": env_task,
                "certificate_branch": branch,
                "certificate_reliable": env_cert,
                "v_detection": ev.v_detection,
                "v_identity": ev.v_identity,
            }
        return ORSnapshot(
            per_environment=per,
            task_reliable=task_ok,
            certificate_reliable=cert_ok,
            certificate=dict(certificate.branch_by_environment),
        )

    def proxy_snapshot(self, delta, certificate):
        terms = self.differentiable_terms(
            delta,
            tuple(certificate.branch_by_environment),
        )
        per = {}
        task_ok = True
        cert_ok = True
        for env, branch in certificate.branch_by_environment.items():
            dvals = [v for _,v in terms[env]["detection"]]
            ivals = [v for _,v in terms[env]["identity"]]
            dprot = bool(dvals) and all(float(v.detach().cpu()) <= 0 for v in dvals)
            iprot = bool(ivals) and all(float(v.detach().cpu()) <= 0 for v in ivals)
            task = dprot or iprot
            cert = dprot if branch=="detection" else iprot
            task_ok = task_ok and task
            cert_ok = cert_ok and cert
            per[env] = {
                "detection_protected": dprot,
                "identity_protected": iprot,
                "task_reliable": task,
                "certificate_branch": branch,
                "certificate_reliable": cert,
            }
        return ORSnapshot(
            per_environment=per,
            task_reliable=task_ok,
            certificate_reliable=cert_ok,
            certificate=dict(certificate.branch_by_environment),
        )


class ORRPPSolver:
    """
    Complete RPP with environment-specific OR certificates.

    Every environment preserves one branch that was already sufficient in the
    selected Blue candidate. This guarantees the task-level OR condition for
    clean + every active Red constraint without requiring the same branch to be
    broken in all environments.
    """
    def __init__(self, *, config, objective):
        self.cfg = config
        self.objective = objective

    def _visual(self, clean, delta):
        with torch.no_grad():
            return float(self.objective(clean,delta).detach().cpu())

    def _descent(self, clean, delta, epsilon_norm):
        d = delta.detach().clone().requires_grad_(True)
        loss = self.objective(clean,d)
        grad = torch.autograd.grad(loss,d,create_graph=False)[0]
        cand = d.detach() - self.cfg.eta_visual * grad.detach()
        return project_delta_box(
            clean01=clean,
            delta=cand,
            epsilon_norm=epsilon_norm,
        ).detach()

    def _selected_terms(self, terms, certificate):
        selected = []
        for env, branch in certificate.branch_by_environment.items():
            for name, value in terms[env][branch]:
                selected.append((env,branch,name,value))
        return selected

    def _repair(self, *, system, delta, certificate, epsilon_norm):
        current = delta.detach()
        projections = 0
        passes = 0
        for p in range(self.cfg.k_repair):
            passes = p+1
            d = current.detach().clone().requires_grad_(True)
            terms = system.differentiable_terms(
                d,
                tuple(certificate.branch_by_environment),
            )
            selected = self._selected_terms(terms,certificate)
            violated_keys = [
                (e,b,n) for e,b,n,v in selected
                if float(v.detach().cpu()) > 0
            ]
            if not violated_keys:
                break

            for key in violated_keys:
                d = current.detach().clone().requires_grad_(True)
                fresh = system.differentiable_terms(
                    d,
                    tuple(certificate.branch_by_environment),
                )
                by_key = {
                    (e,b,n):v
                    for e,b,n,v in self._selected_terms(fresh,certificate)
                }
                g = by_key[key]
                if float(g.detach().cpu()) <= 0:
                    continue
                grad = torch.autograd.grad(g,d,create_graph=False)[0].detach()
                denom = grad.pow(2).sum() + self.cfg.xi
                scale = self.cfg.rho_repair * torch.relu(g.detach()) / denom
                current = project_delta_box(
                    clean01=system.clean01,
                    delta=d.detach()-scale*grad,
                    epsilon_norm=epsilon_norm,
                ).detach()
                projections += 1

            if system.proxy_snapshot(current,certificate).certificate_reliable:
                break
        return current, passes, projections

    def refine(
        self,
        *,
        system: ORConstraintSystem,
        protected01,
        epsilon_norm,
        certificate: EnvironmentCertificate,
    ):
        clean = system.clean01
        delta0 = project_delta_box(
            clean01=clean,
            delta=protected01.detach()-clean.detach(),
            epsilon_norm=epsilon_norm,
        ).detach()

        initial = system.hard_snapshot(delta0,certificate)
        if not initial.certificate_reliable:
            return ORRPPResult(
                status="initial_certificate_unreliable",
                refined=torch.clamp(clean+delta0,0,1),
                delta_initial=delta0,
                delta_final=delta0,
                initial_snapshot=initial,
                final_snapshot=initial,
                certificate=certificate,
                steps=(),
                accepted_steps=0,
                rollback_steps=0,
                total_projections=0,
            )

        current = delta0
        records = []
        accepted = rollback = total_proj = 0

        for step in range(self.cfg.steps):
            vb = self._visual(clean,current)
            cand = self._descent(clean,current,epsilon_norm)
            vc = self._visual(clean,cand)
            proxy = system.proxy_snapshot(cand,certificate)

            triggered = not proxy.certificate_reliable
            if triggered:
                repaired, passes, proj = self._repair(
                    system=system,
                    delta=cand,
                    certificate=certificate,
                    epsilon_norm=epsilon_norm,
                )
            else:
                repaired, passes, proj = cand,0,0
            total_proj += proj
            vr = self._visual(clean,repaired)

            hard = system.hard_snapshot(repaired,certificate)
            visual_ok = vr <= vb
            ok = hard.certificate_reliable and (
                (not self.cfg.accept_only_if_visual_improves) or visual_ok
            )
            if ok:
                current = repaired
                accepted += 1
            else:
                rollback += 1

            records.append(
                ORRPPStep(
                    step=step,
                    accepted=ok,
                    rolled_back=not ok,
                    repair_triggered=triggered,
                    projections=proj,
                    repair_passes=passes,
                    visual_before=vb,
                    visual_candidate=vc,
                    visual_after_repair=vr,
                    snapshot_after_descent=proxy,
                    snapshot_after_repair=hard,
                )
            )

        final = system.hard_snapshot(current,certificate)
        if not final.certificate_reliable:
            raise RuntimeError("RPP safety invariant violated.")

        return ORRPPResult(
            status="ok",
            refined=torch.clamp(clean+current,0,1).detach(),
            delta_initial=delta0,
            delta_final=current.detach(),
            initial_snapshot=initial,
            final_snapshot=final,
            certificate=certificate,
            steps=tuple(records),
            accepted_steps=accepted,
            rollback_steps=rollback,
            total_projections=total_proj,
        )
