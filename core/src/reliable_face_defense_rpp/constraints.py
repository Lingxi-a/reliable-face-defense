from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch

from .contracts import (
    Branch,
    ConstraintTerm,
    ReliabilitySnapshot,
)


@dataclass(frozen=True)
class NamedConstraintFunction:
    name: str
    branch: Branch
    fn: Callable[[torch.Tensor, torch.Tensor, str], torch.Tensor]


class CallableConstraintProvider:
    """
    Builds all differentiable g_m(delta)<=0 constraints for:

        clean + active Red environments

    Each supplied function must return a scalar tensor. Threshold calibration,
    locked clean geometry, proxy-model aggregation details, etc. remain inside
    the concrete model adapter and are NOT reinvented by RPP.

    Signature:
        fn(clean01, propagated_protected01, environment_name) -> scalar g

    Multiple model functions per branch are supported; RPP repairs them
    sequentially as Eq.(26) requires.
    """
    def __init__(
        self,
        *,
        transform_registry,
        constraint_functions: Iterable[NamedConstraintFunction],
    ):
        self.transform_registry = transform_registry
        self.constraint_functions = tuple(constraint_functions)
        if not self.constraint_functions:
            raise ValueError("At least one differentiable constraint is required.")

        names = [x.name for x in self.constraint_functions]
        if len(names) != len(set(names)):
            raise ValueError("Constraint function names must be unique.")

    def evaluate(
        self,
        *,
        clean01: torch.Tensor,
        delta: torch.Tensor,
        active_environments: tuple[str, ...],
    ) -> tuple[ConstraintTerm, ...]:
        protected = torch.clamp(clean01 + delta, 0.0, 1.0)
        environments = ("clean",) + tuple(active_environments)
        out = []

        for env in environments:
            y = self.transform_registry.apply(env, protected)
            for c in self.constraint_functions:
                value = c.fn(clean01, y, env)
                if not torch.is_tensor(value):
                    raise TypeError(
                        f"Constraint '{c.name}' must return a torch scalar tensor."
                    )
                if value.numel() != 1:
                    raise ValueError(
                        f"Constraint '{c.name}' returned {tuple(value.shape)}; scalar required."
                    )
                out.append(
                    ConstraintTerm(
                        name=c.name,
                        branch=c.branch,
                        environment=env,
                        value=value.reshape(()),
                    )
                )
        return tuple(out)


def _max_violation(terms, branch):
    xs = [
        max(float(t.value.detach().cpu()), 0.0)
        for t in terms
        if t.branch == branch
    ]
    if not xs:
        return float("inf")
    return max(xs)


def snapshot_from_terms(
    *,
    terms,
    certificate_branch: Branch,
    active_environments: tuple[str, ...],
    include_terms: bool = False,
) -> ReliabilitySnapshot:
    terms = tuple(terms)
    detection_terms = [x for x in terms if x.branch == "detection"]
    identity_terms = [x for x in terms if x.branch == "identity"]

    detection_protected = bool(detection_terms) and all(
        float(x.value.detach().cpu()) <= 0.0
        for x in detection_terms
    )
    identity_protected = bool(identity_terms) and all(
        float(x.value.detach().cpu()) <= 0.0
        for x in identity_terms
    )

    if certificate_branch == "detection":
        certificate = detection_protected
    elif certificate_branch == "identity":
        certificate = identity_protected
    else:
        raise ValueError(certificate_branch)

    return ReliabilitySnapshot(
        detection_protected=detection_protected,
        identity_protected=identity_protected,
        task_reliable=detection_protected or identity_protected,
        certificate_branch=certificate_branch,
        certificate_reliable=certificate,
        environments=("clean",) + tuple(active_environments),
        max_detection_violation=_max_violation(terms, "detection"),
        max_identity_violation=_max_violation(terms, "identity"),
        terms=tuple(
            x.detached_dict() for x in terms
        ) if include_terms else (),
    )


class ConstraintSignReliabilityChecker:
    """
    Exact recheck using the same scalar constraints' signs.

    Useful when the differentiable constraint implementation itself is the
    frozen evaluation protocol.
    """
    def __init__(self, provider, *, include_terms=False):
        self.provider = provider
        self.include_terms = bool(include_terms)

    def check(
        self,
        *,
        clean01,
        delta,
        active_environments,
        certificate_branch,
    ):
        terms = self.provider.evaluate(
            clean01=clean01,
            delta=delta,
            active_environments=active_environments,
        )
        return snapshot_from_terms(
            terms=terms,
            certificate_branch=certificate_branch,
            active_environments=active_environments,
            include_terms=self.include_terms,
        )


class CallableHardReliabilityChecker:
    """
    Hard recheck bridge for the actual detector/identity evaluation protocol.

    `check_fn` must return either:
      ReliabilitySnapshot

    or a dict with:
      detection_protected: bool
      identity_protected: bool
      max_detection_violation: float (optional)
      max_identity_violation: float (optional)

    This allows RPP gradients to use differentiable proxies while final
    accept/rollback uses the project's hard frozen evaluator.
    """
    def __init__(self, check_fn):
        self.check_fn = check_fn

    def check(
        self,
        *,
        clean01,
        delta,
        active_environments,
        certificate_branch,
    ):
        protected = torch.clamp(clean01 + delta, 0.0, 1.0)
        x = self.check_fn(
            clean01=clean01,
            protected01=protected,
            active_environments=active_environments,
        )
        if isinstance(x, ReliabilitySnapshot):
            return x

        d = bool(x["detection_protected"])
        i = bool(x["identity_protected"])
        cert = d if certificate_branch == "detection" else i

        return ReliabilitySnapshot(
            detection_protected=d,
            identity_protected=i,
            task_reliable=d or i,
            certificate_branch=certificate_branch,
            certificate_reliable=cert,
            environments=("clean",) + tuple(active_environments),
            max_detection_violation=float(
                x.get("max_detection_violation", 0.0 if d else float("inf"))
            ),
            max_identity_violation=float(
                x.get("max_identity_violation", 0.0 if i else float("inf"))
            ),
            terms=(),
        )
