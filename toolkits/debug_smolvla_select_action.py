#!/usr/bin/env python3

import glob
import pickle
import traceback
from pathlib import Path

import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

MODEL_PATH = Path("/home/chuhengzhang/RLinf/models/smolvla_libero")
PKL_DIR = Path("/home/chuhengzhang/RLinf/models/smolvla_debug")

assert MODEL_PATH.is_dir(), f"model path not found: {MODEL_PATH}"
pkl_files = sorted(glob.glob(str(PKL_DIR / "batch_before_select_action_*.pkl")))
assert pkl_files, f"no pkl found under: {PKL_DIR}"
PKL_PATH = Path(pkl_files[-1])

with open(PKL_PATH, "rb") as f:
    payload = pickle.load(f)

assert "batch_obs" in payload, f"invalid payload keys: {list(payload.keys())}"
batch_obs = payload["batch_obs"]
assert isinstance(batch_obs, dict), f"batch_obs type: {type(batch_obs)}"

print(f"[debug] model={MODEL_PATH}")
print(f"[debug] pkl={PKL_PATH}")
print(f"[debug] keys={list(batch_obs.keys())}")

policy = SmolVLAPolicy.from_pretrained(str(MODEL_PATH))
policy.eval()

try:
    with torch.no_grad():
        import pdb; pdb.set_trace()  # debug before select_action
        action = policy.select_action(batch_obs)
    print("[debug] select_action ok")
    if isinstance(action, torch.Tensor):
        print(f"[debug] action shape={tuple(action.shape)} dtype={action.dtype}")
except Exception:
    print("[debug] select_action failed:")
    traceback.print_exc()

import pdb; pdb.set_trace() 

