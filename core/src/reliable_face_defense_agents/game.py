from __future__ import annotations

from dataclasses import dataclass
import torch

from .blue.contracts import BlueState


@dataclass(frozen=True)
class BlueRoundRecord:
    sample_id: str
    blue_decision: object


class BlueRedFeedbackLoop:
    """
    Blue<->Red connection layer.

    This intentionally does NOT pretend RPP already exists.

    It closes the two-agent loop that was missing previously:

      current Red active set
        -> BlueState
        -> Blue route + sample-wise epsilon
        -> protected control images
        -> Red control-set worst-environment search
        -> accumulated active set
        -> next Blue round

    RPP will later be inserted between Blue output and Red evaluation by the
    overall orchestrator.
    """

    def __init__(self, *, blue_agent, red_agent):
        self.blue = blue_agent
        self.red = red_agent
        self.round_index = 0

    def blue_round(
        self,
        samples: list[tuple[str, torch.Tensor]],
    ):
        records = []
        protected = []

        active = self.red.active_constraints
        for sample_id, image in samples:
            state = BlueState(
                sample_id=sample_id,
                active_red_constraints=active,
                round_index=self.round_index,
            )
            decision = self.blue.act(
                state=state,
                image01=image,
            )
            records.append(BlueRoundRecord(sample_id, decision))
            if decision.feasible:
                protected.append(
                    (sample_id, decision.chosen.artifact)
                )

        return tuple(records), protected

    def red_update(
        self,
        protected_control_samples: list[tuple[str, torch.Tensor]],
    ):
        decision = self.red.search(
            control_samples=protected_control_samples,
        )
        self.round_index += 1
        return decision

    def run_one_feedback_round(
        self,
        control_samples: list[tuple[str, torch.Tensor]],
    ):
        blue_records, protected = self.blue_round(control_samples)
        if not protected:
            raise RuntimeError(
                "No feasible Blue protected control sample for Red update."
            )
        red_decision = self.red_update(protected)
        return {
            "blue_records": blue_records,
            "red_decision": red_decision,
            "next_active_constraints": red_decision.active_constraints,
            "next_round_index": self.round_index,
        }
