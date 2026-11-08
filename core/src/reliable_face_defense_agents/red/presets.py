from __future__ import annotations

from .contracts import RedTransformSpec


def validation_red_pool() -> tuple[RedTransformSpec, ...]:
    """
    Frozen first weaker validation setting from the current implementation plan.

    No Resize.
    OSN proxy:
      W  = JPEG80
      Q  = JPEG75
      WQ = JPEG80 -> JPEG75
      QW = JPEG75 -> JPEG80
    """
    return (
        RedTransformSpec("jpeg_q80", "jpeg", (80,)),
        RedTransformSpec("jpeg_q70", "jpeg", (70,)),
        RedTransformSpec("jpeg_q60", "jpeg", (60,)),
        RedTransformSpec("blur_sigma_0.5", "blur", (0.5,)),
        RedTransformSpec("blur_sigma_0.8", "blur", (0.8,)),
        RedTransformSpec("blur_sigma_1.0", "blur", (1.0,)),
        RedTransformSpec("osn_W", "osn", (80,)),
        RedTransformSpec("osn_Q", "osn", (75,)),
        RedTransformSpec("osn_WQ", "osn", (80, 75)),
        RedTransformSpec("osn_QW", "osn", (75, 80)),
    )


def full_method_red_pool() -> tuple[RedTransformSpec, ...]:
    """
    Full historical Dynamic-Red pool from the project method:
      JPEG J30/J50/J75/J95
      Blur 0.5/1.0/1.5/2.0
      OSN W/Q/WQ/QW

    The current CPU OSN proxy mapping is reused for executable testing.
    Real platform-calibrated proxies should replace these JPEG surrogates
    when those calibrated operators are available.
    """
    return (
        RedTransformSpec("jpeg_q30", "jpeg", (30,)),
        RedTransformSpec("jpeg_q50", "jpeg", (50,)),
        RedTransformSpec("jpeg_q75", "jpeg", (75,)),
        RedTransformSpec("jpeg_q95", "jpeg", (95,)),
        RedTransformSpec("blur_sigma_0.5", "blur", (0.5,)),
        RedTransformSpec("blur_sigma_1.0", "blur", (1.0,)),
        RedTransformSpec("blur_sigma_1.5", "blur", (1.5,)),
        RedTransformSpec("blur_sigma_2.0", "blur", (2.0,)),
        RedTransformSpec("osn_W", "osn", (80,)),
        RedTransformSpec("osn_Q", "osn", (75,)),
        RedTransformSpec("osn_WQ", "osn", (80, 75)),
        RedTransformSpec("osn_QW", "osn", (75, 80)),
    )
