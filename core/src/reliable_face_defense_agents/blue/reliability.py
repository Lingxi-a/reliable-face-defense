from __future__ import annotations

from .contracts import EnvironmentOutcome


def make_environment_outcome(
    environment: str,
    *,
    detection_success: bool,
    identity_success: bool,
    metadata=None,
) -> EnvironmentOutcome:
    """
    Current locked OR-defense logic:

      attack succeeds  = D_success AND I_success
      defense succeeds = D_fail OR I_fail
    """
    d = bool(detection_success)
    i = bool(identity_success)
    attack = d and i
    return EnvironmentOutcome(
        environment=environment,
        detection_success=d,
        identity_success=i,
        attack_success=attack,
        defense_success=not attack,
        metadata={} if metadata is None else dict(metadata),
    )
