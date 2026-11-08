from __future__ import annotations

from pathlib import Path
import json

import cv2
import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .common import load_rgb01, write_json


class DetectionGate:
    def __init__(self, deps_root, device):
        from reliable_face_defense_fullsystem.facepoison_vendor.retinaface_adapter import (
            RetinaFaceMobile025Adapter,
        )
        self.device = torch.device(device)
        deps = Path(deps_root)
        adapter = RetinaFaceMobile025Adapter(
            deps / "source" / "Pytorch_Retinaface-master",
            deps / "weights" / "detection" / "mobilenet0.25_Final.pth",
            device=device,
        )
        self.model = adapter.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.threshold = None

    @torch.no_grad()
    def scores(self, images01):
        from reliable_face_defense_fullsystem.facepoison_rgb import rgb01_to_retina
        _, conf, _ = self.model(rgb01_to_retina(images01.to(self.device)))
        return conf[:, :, 1].amax(dim=1).detach().cpu()

    def calibrate(self, paths, *, target_tpr, cache_path, batch_size=16):
        cache_path = Path(cache_path)
        if cache_path.is_file():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            self.threshold = float(payload["threshold"])
            return payload
        scores = []
        batch = []
        for path in paths:
            batch.append(load_rgb01(path, 256))
            if len(batch) >= batch_size:
                scores.extend(self.scores(torch.cat(batch)).tolist())
                batch = []
        if batch:
            scores.extend(self.scores(torch.cat(batch)).tolist())
        self.threshold = float(np.quantile(np.asarray(scores), 1.0 - float(target_tpr)))
        payload = {
            "threshold": self.threshold,
            "target_clean_tpr": float(target_tpr),
            "num_official_valid_images": len(scores),
            "score_min": float(np.min(scores)),
            "score_mean": float(np.mean(scores)),
            "score_max": float(np.max(scores)),
            "provenance": "official-valid clean max RetinaFace face probability quantile",
        }
        write_json(cache_path, payload)
        return payload

    def success(self, images01):
        if self.threshold is None:
            raise RuntimeError("Detection gate is not calibrated.")
        scores = self.scores(images01)
        return scores >= self.threshold, scores


class VGGFaceNet(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [
            ("conv1_1", 3, 64), ("conv1_2", 64, 64),
            ("conv2_1", 64, 128), ("conv2_2", 128, 128),
            ("conv3_1", 128, 256), ("conv3_2", 256, 256), ("conv3_3", 256, 256),
            ("conv4_1", 256, 512), ("conv4_2", 512, 512), ("conv4_3", 512, 512),
            ("conv5_1", 512, 512), ("conv5_2", 512, 512), ("conv5_3", 512, 512),
        ]
        self.convs = nn.ModuleDict({
            name: nn.Conv2d(inp, out, 3, padding=1)
            for name, inp, out in channels
        })
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)

    def forward(self, x):
        for block in range(1, 6):
            count = 2 if block < 3 else 3
            for index in range(1, count + 1):
                x = F.relu(self.convs[f"conv{block}_{index}"](x), inplace=True)
            x = F.max_pool2d(x, 2, 2)
        x = x.flatten(1)
        x = F.relu(self.fc6(x), inplace=True)
        x = F.relu(self.fc7(x), inplace=True)
        return F.normalize(x, dim=1)


def _h5_datasets(group, prefix=""):
    out = {}
    for name, value in group.items():
        path = f"{prefix}/{name}" if prefix else name
        if isinstance(value, h5py.Dataset):
            out[path] = value
        else:
            out.update(_h5_datasets(value, path))
    return out


def _convert_weight(array, target_shape):
    target_shape = tuple(target_shape)
    if tuple(array.shape) == target_shape:
        return array
    if array.ndim == 4:
        value = np.transpose(array, (3, 2, 0, 1))
        if tuple(value.shape) == target_shape:
            return value
        if len(target_shape) == 2 and value.shape[0] == target_shape[0]:
            value = value.reshape(value.shape[0], -1)
            if tuple(value.shape) == target_shape:
                return value
    if array.ndim == 2 and tuple(array.T.shape) == target_shape:
        return array.T
    raise ValueError(f"Cannot convert H5 weight shape {array.shape} to {target_shape}")


def _compatible_shape(source_shape, target_shape):
    source_shape, target_shape = tuple(source_shape), tuple(target_shape)
    if source_shape == target_shape:
        return True
    if len(source_shape) == 2 and source_shape[::-1] == target_shape:
        return True
    if len(source_shape) == 4:
        transposed = (source_shape[3], source_shape[2], source_shape[0], source_shape[1])
        if transposed == target_shape:
            return True
        if len(target_shape) == 2 and transposed[0] == target_shape[0]:
            return int(np.prod(transposed[1:])) == target_shape[1]
    return False


class VGGFaceEvaluator:
    layer_names = [
        "conv1_1", "conv1_2", "conv2_1", "conv2_2",
        "conv3_1", "conv3_2", "conv3_3",
        "conv4_1", "conv4_2", "conv4_3",
        "conv5_1", "conv5_2", "conv5_3", "fc6", "fc7",
    ]

    def __init__(self, weight_path, device):
        self.device = torch.device(device)
        with torch.device(self.device):
            self.model = VGGFaceNet()
        self._load_h5(weight_path)
        self.model.eval()

    def _load_h5(self, path):
        used = set()
        with h5py.File(path, "r") as f:
            datasets = _h5_datasets(f)
            for layer_name in self.layer_names:
                layer = self.model.convs[layer_name] if layer_name.startswith("conv") else getattr(self.model, layer_name)
                for parameter_name, parameter in (("weight", layer.weight), ("bias", layer.bias)):
                    candidates = []
                    for ds_path, dataset in datasets.items():
                        if ds_path in used:
                            continue
                        lower = ds_path.lower()
                        if layer_name.lower() not in lower:
                            continue
                        if parameter_name == "bias" and len(dataset.shape) != 1:
                            continue
                        if parameter_name == "weight" and len(dataset.shape) == 1:
                            continue
                        if _compatible_shape(dataset.shape, parameter.shape):
                            candidates.append((ds_path, dataset))
                    if not candidates:
                        for ds_path, dataset in datasets.items():
                            if ds_path in used:
                                continue
                            if _compatible_shape(dataset.shape, parameter.shape):
                                candidates.append((ds_path, dataset))
                    if not candidates:
                        raise RuntimeError(
                            f"No compatible VGGFace H5 tensor for {layer_name}.{parameter_name} {tuple(parameter.shape)}"
                        )
                    ds_path, dataset = candidates[0]
                    array = np.asarray(dataset, dtype=np.float32)
                    array = _convert_weight(array, parameter.shape)
                    parameter.data.copy_(torch.from_numpy(array).to(self.device))
                    used.add(ds_path)

    @torch.no_grad()
    def embed(self, images01):
        x = F.interpolate(
            images01.to(self.device), (224, 224), mode="bilinear", align_corners=False
        )
        x = x[:, [2, 1, 0]] * 255.0
        mean = torch.tensor([93.5940, 104.7624, 129.1863], device=self.device).view(1, 3, 1, 1)
        return self.model(x - mean).detach().cpu()


class SFaceEvaluator:
    def __init__(self, model_path):
        creator = getattr(cv2, "FaceRecognizerSF_create", None)
        if creator is None:
            creator = getattr(cv2, "FaceRecognizerSF", None)
            creator = creator.create if creator is not None else None
        if creator is None:
            raise RuntimeError("OpenCV build lacks FaceRecognizerSF; install opencv-contrib-python-headless.")
        self.model = creator(str(model_path), "")

    def embed(self, images01):
        rows = []
        for image in images01.detach().cpu():
            rgb = np.rint(image.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            bgr = cv2.resize(bgr, (112, 112), interpolation=cv2.INTER_LINEAR)
            feature = np.asarray(self.model.feature(bgr), dtype=np.float32).reshape(-1)
            feature /= max(float(np.linalg.norm(feature)), 1e-12)
            rows.append(feature)
        return torch.from_numpy(np.stack(rows))


class HeldOutEvaluators:
    def __init__(self, cfg, device):
        self.vggface = VGGFaceEvaluator(cfg["evaluators"]["vggface"], device)
        self.sface = SFaceEvaluator(cfg["evaluators"]["sface"])

    def embed(self, images01):
        return {
            "vggface": self.vggface.embed(images01),
            "sface": self.sface.embed(images01),
        }


def build_gallery(evaluators, gallery, cache_path, batch_size=12):
    cache_path = Path(cache_path)
    if cache_path.is_file():
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    identities = sorted(int(x) for x in gallery)
    embeddings = {"vggface": [], "sface": []}
    for start in range(0, len(identities), batch_size):
        ids = identities[start:start + batch_size]
        images = torch.cat([load_rgb01(gallery[str(x)], 256) for x in ids])
        batch_embeddings = evaluators.embed(images)
        for name in embeddings:
            embeddings[name].append(batch_embeddings[name])
    payload = {
        "identities": identities,
        "embeddings": {
            name: F.normalize(torch.cat(parts).float(), dim=1)
            for name, parts in embeddings.items()
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


def rank_hits(query_embeddings, gallery_payload, true_identities):
    gallery_ids = gallery_payload["identities"]
    id_to_index = {int(value): index for index, value in enumerate(gallery_ids)}
    result = {}
    for evaluator_name, query in query_embeddings.items():
        gallery = gallery_payload["embeddings"][evaluator_name]
        scores = F.normalize(query.float(), dim=1) @ gallery.T
        top5 = scores.topk(k=min(5, scores.shape[1]), dim=1).indices
        true_indices = torch.tensor([id_to_index[int(x)] for x in true_identities]).view(-1, 1)
        result[evaluator_name] = {
            "acc1": (top5[:, :1] == true_indices).any(dim=1),
            "acc5": (top5 == true_indices).any(dim=1),
        }
    return result
