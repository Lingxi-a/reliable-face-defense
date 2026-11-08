from __future__ import annotations

from reliable_face_defense_agents.red.presets import (
    validation_red_pool,
    full_method_red_pool,
)
from reliable_face_defense_agents.red.transforms import apply_red_transform


class HardPropagationRegistry:
    def __init__(self, pool="validation"):
        if pool == "validation":
            specs = validation_red_pool()
        elif pool == "full":
            specs = full_method_red_pool()
        else:
            raise ValueError(pool)
        self.specs = {x.name:x for x in specs}

    @property
    def names(self):
        return tuple(self.specs)

    def apply(self, name, image01):
        if name == "clean":
            return image01
        if name not in self.specs:
            raise KeyError(f"Unknown propagation condition: {name}")
        return apply_red_transform(image01, self.specs[name])

    def transforms(self):
        return tuple(self.specs.values())
