from __future__ import annotations

from reliable_face_defense_agents.red.contracts import ViolationObservation


class ContextualRedViolationEvaluator:
    def __init__(self, *, reliability, contexts):
        self.reliability = reliability
        self.contexts = contexts

    def evaluate(self, *, sample_id, transform, transformed_protected):
        context = self.contexts[str(sample_id)]
        ev = self.reliability.evaluate(transformed_protected,context)
        return ViolationObservation(
            v_detection=ev.v_detection,
            v_identity=ev.v_identity,
            detection_success=ev.detection_success,
            identity_success=ev.identity_success,
            metadata=ev.to_dict(),
        )
