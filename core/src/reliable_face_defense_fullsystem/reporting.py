from __future__ import annotations

from pathlib import Path
import csv
import json
import numpy as np
from PIL import Image
import torch


def save_image01(x,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    arr=(x[0].detach().cpu().clamp(0,1).permute(1,2,0).numpy())
    Image.fromarray(np.rint(arr*255).astype(np.uint8)).save(path)


def save_perturbation(clean,protected,path,scale=8.0):
    d=(protected-clean)[0].detach().cpu().permute(1,2,0).numpy()
    vis=np.clip(127.5+d*255.0/max(float(scale),1e-6)*127.5,0,255).astype(np.uint8)
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(vis).save(path)


def json_dump(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2),encoding="utf-8")


def append_csv(path,row,fieldnames):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    exists=path.exists()
    with path.open("a",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)


def write_csv(path, rows, fieldnames):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fieldnames,extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
