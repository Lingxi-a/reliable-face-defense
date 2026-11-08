def apply_protected_range_policy(x, policy):
    # Engineering policy because paper does not report exact final range handling.
    if policy == "clamp_01":
        return x.clamp(0.0, 1.0)
    if policy == "identity":
        return x
    raise ValueError(f"Unknown range policy: {policy}")


def to_lpips_range(x01):
    # Adapter convention for lpips package: [0,1] -> [-1,1].
    return x01 * 2.0 - 1.0
