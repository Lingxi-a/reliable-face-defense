from __future__ import annotations

from pathlib import Path
import torch


def save_agents_state(
    path,
    *,
    budget_network,
    budget_optimizer,
    red_agent,
    round_index,
    extra=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "budget_network": budget_network.state_dict(),
        "budget_optimizer": None if budget_optimizer is None else budget_optimizer.state_dict(),
        "red_agent": red_agent.state_dict(),
        "round_index": int(round_index),
        "extra": extra or {},
    }, path)


def load_agents_state(
    path,
    *,
    budget_network,
    budget_optimizer,
    red_agent,
    map_location="cpu",
):
    state = torch.load(path, map_location=map_location, weights_only=False)
    budget_network.load_state_dict(state["budget_network"])
    if budget_optimizer is not None and state["budget_optimizer"] is not None:
        budget_optimizer.load_state_dict(state["budget_optimizer"])
    red_agent.load_state_dict(state["red_agent"])
    return {
        "round_index": int(state["round_index"]),
        "extra": state.get("extra", {}),
    }
