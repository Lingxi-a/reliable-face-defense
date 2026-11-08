from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import math
import torch
from .contracts import NullSwapPaperTrainingConfig, NullSwapEngineeringTrainingConfig


@dataclass(frozen=True)
class DLWResult:
    weighted_loss: torch.Tensor
    normalized_weights: tuple[float, ...]
    raw_weights: tuple[float, ...]
    variances: tuple[float, ...]
    relative_progress: tuple[float, ...]
    beta_epoch: float


class DynamicLossWeighting:
    def __init__(self, num_losses, paper=None, engineering=None):
        self.num_losses = int(num_losses)
        self.paper = paper or NullSwapPaperTrainingConfig()
        self.engineering = engineering or NullSwapEngineeringTrainingConfig()
        self.histories = [[] for _ in range(self.num_losses)]

    def beta(self, epoch):
        te = max(int(epoch), 0)
        if self.engineering.dlw_t_star is None:
            capped = te
        else:
            capped = min(te, int(self.engineering.dlw_t_star))
        return min(
            self.paper.dlw_beta_init
            + self.paper.dlw_gamma_schedule * capped,
            self.paper.dlw_beta_max,
        )

    @staticmethod
    def _variance(values):
        if len(values) <= 1:
            return 0.0
        m = sum(values) / len(values)
        return sum((v - m) ** 2 for v in values) / len(values)

    def compute(self, losses: Sequence[torch.Tensor], epoch: int, update_history=True):
        if len(losses) != self.num_losses:
            raise ValueError("DLW loss count mismatch")

        scalars = [float(x.detach().cpu()) for x in losses]
        beta = self.beta(epoch)

        variances, deltas, raw = [], [], []
        for i, current in enumerate(scalars):
            values = (self.histories[i] + [current])[-self.paper.dlw_k:]
            var = self._variance(values)

            if not self.histories[i]:
                delta = 0.0
            else:
                prev = self.histories[i][-1]
                delta = max(
                    (prev - current)
                    / (prev + self.engineering.dlw_epsilon_p),
                    -1.0,
                )

            denom = max(
                self.paper.dlw_alpha * var
                + beta * (1.0 + delta),
                self.paper.dlw_epsilon_d,
            )
            wi = max(
                1.0 / denom,
                self.paper.dlw_epsilon_w,
            )
            variances.append(var)
            deltas.append(delta)
            raw.append(wi)

        s = sum(raw)
        if not math.isfinite(s) or s <= 0:
            raise FloatingPointError("Invalid DLW weights")
        c = float(self.num_losses)
        normalized = [c * x / s for x in raw]

        weighted = sum(
            loss * loss.new_tensor(w)
            for loss, w in zip(losses, normalized)
        )

        if update_history:
            for i, v in enumerate(scalars):
                self.histories[i].append(v)
                self.histories[i] = self.histories[i][-max(self.paper.dlw_k, 2):]

        return DLWResult(
            weighted_loss=weighted,
            normalized_weights=tuple(normalized),
            raw_weights=tuple(raw),
            variances=tuple(variances),
            relative_progress=tuple(deltas),
            beta_epoch=float(beta),
        )

    def state_dict(self):
        return {
            "num_losses": self.num_losses,
            "histories": [list(x) for x in self.histories],
        }

    def load_state_dict(self, state):
        if int(state["num_losses"]) != self.num_losses:
            raise ValueError("DLW num_losses mismatch")
        self.histories = [list(map(float, x)) for x in state["histories"]]
