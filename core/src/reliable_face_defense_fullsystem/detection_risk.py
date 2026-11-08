from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import torch

from .facepoison_vendor.retinaface_adapter import (
    RetinaFaceMobile025Adapter,
)
from .facepoison_rgb import rgb01_to_retina


@dataclass(frozen=True)
class DetectionContext:
    locked_anchor_indices: torch.Tensor
    clean_risk: float


def _box_iou_one_to_many(ref: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    # ref [4], boxes [N,4] in xyxy.
    lt = torch.maximum(ref[:2], boxes[:, :2])
    rb = torch.minimum(ref[2:], boxes[:, 2:])
    wh = (rb-lt).clamp_min(0)
    inter = wh[:,0] * wh[:,1]
    area_ref = (ref[2]-ref[0]).clamp_min(0) * (ref[3]-ref[1]).clamp_min(0)
    area_b = (
        (boxes[:,2]-boxes[:,0]).clamp_min(0)
        * (boxes[:,3]-boxes[:,1]).clamp_min(0)
    )
    return inter / (area_ref + area_b - inter + 1e-12)


class RetinaFaceLockedRisk:
    """
    Differentiable locked-anchor RetinaFace recovery risk.

    Clean image:
      - decode clean RetinaFace anchors;
      - take highest-confidence clean anchor as b_ref;
      - lock anchors whose clean decoded boxes have IoU >= eta_anchor_iou.

    Protected/transformed image:
      - use the SAME locked anchor indices;
      - aggregate face scores by log-sum-exp.

    This keeps geometry/candidate selection fixed from the clean source.
    """
    def __init__(
        self,
        *,
        deps_root,
        device,
        eta_anchor_iou,
        lse_alpha,
    ):
        deps_root = Path(deps_root)
        self.device = torch.device(device)
        self.eta = float(eta_anchor_iou)
        self.alpha = float(lse_alpha)

        self.adapter = RetinaFaceMobile025Adapter(
            deps_root/"source"/"Pytorch_Retinaface-master",
            deps_root/"weights"/"detection"/"mobilenet0.25_Final.pth",
            device=device,
        )
        self.model = self.adapter.model
        # RPP needs gradients w.r.t. the image, not detector weights.
        # Freezing parameters materially reduces autograd memory without
        # changing the detector function or image gradient.
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()
        self.cfg = self.adapter.cfg

        source_root = deps_root/"source"/"Pytorch_Retinaface-master"
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        from layers.functions.prior_box import PriorBox
        from utils.box_utils import decode
        self.PriorBox = PriorBox
        self.decode = decode

    def _forward_raw(self, image01):
        xret = rgb01_to_retina(image01.to(self.device))
        return self.model(xret)

    def _priors(self, h, w, dtype):
        p = self.PriorBox(self.cfg, image_size=(h,w)).forward()
        return p.to(device=self.device, dtype=dtype)

    def prepare_context(self, clean01: torch.Tensor) -> DetectionContext:
        x = clean01.to(self.device)
        h, w = x.shape[-2:]
        with torch.no_grad():
            loc, conf, _ = self._forward_raw(x)
            loc0 = loc[0]
            conf0 = conf[0]
            priors = self._priors(h,w,loc0.dtype)
            boxes = self.decode(loc0, priors, self.cfg["variance"])
            scale = torch.tensor(
                [w,h,w,h],
                device=self.device,
                dtype=boxes.dtype,
            )
            boxes = boxes * scale
            scores = conf0[:,1]
            top = int(torch.argmax(scores).item())
            ref = boxes[top]
            iou = _box_iou_one_to_many(ref, boxes)
            idx = torch.nonzero(iou >= self.eta, as_tuple=False).flatten()
            if idx.numel() == 0:
                idx = torch.tensor([top], device=self.device, dtype=torch.long)
            risk = self._risk_from_conf(conf0, idx)
        return DetectionContext(
            locked_anchor_indices=idx.detach(),
            clean_risk=float(risk.detach().cpu()),
        )

    def _risk_from_conf(self, conf0, indices):
        face_scores = conf0[indices,1]
        return torch.logsumexp(self.alpha*face_scores, dim=0) / self.alpha

    def risk(self, image01, context: DetectionContext):
        _, conf, _ = self._forward_raw(image01.to(self.device))
        return self._risk_from_conf(
            conf[0],
            context.locked_anchor_indices.to(self.device),
        )
