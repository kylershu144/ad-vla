"""Thin inference wrapper around the official AutoVLA model for closed-loop use.

Reuses autovla/models/autovla.py (prompt construction, action-token decoding)
unchanged; only stubs the RL-reward module, which pulls in nuplan/navsim.
"""
import os
import tempfile
import sys
import types
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
AUTOVLA_DIR = REPO / "autovla"
sys.path.insert(0, str(AUTOVLA_DIR))

# models.autovla imports models.utils.score (PDM reward, needs nuplan). Not used at inference.
_stub = types.ModuleType("models.utils.score")
_stub.PDM_Reward = _stub.TrajectorySampling = _stub.Trajectory = None
sys.modules["models.utils.score"] = _stub

from models.autovla import AutoVLA  # noqa: E402


def load_state_dict_cached(ckpt: Path):
    """Load the Lightning checkpoint; cache an fp16 safetensors copy for fast reloads."""
    from safetensors.torch import load_file, save_file
    cache = ckpt.with_suffix(".fp16.safetensors")
    if cache.exists():
        return load_file(str(cache))
    sd = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)["state_dict"]
    sd = {k.replace("autovla.", "", 1): v.half().contiguous() for k, v in sd.items()}
    save_file(sd, str(cache))
    return sd


class AutoVLAPolicy:
    def __init__(
        self,
        checkpoint=REPO / "models/AutoVLA_PDMS_89.ckpt",
        base_model=REPO / "models/Qwen2.5-VL-3B-Instruct",
        config=AUTOVLA_DIR / "config/training/qwen2.5-vl-3B-nuplan-grpo-cot.yaml",
        dtype=torch.float16,
        device="cuda",
        use_cot=True,
        max_new_tokens=1024,
    ):
        with open(config) as f:
            cfg = yaml.safe_load(f)
        cfg["model"]["pretrained_model_path"] = str(base_model)
        cfg["model"]["codebook_cache_path"] = str(AUTOVLA_DIR / "codebook_cache/agent_vocab.pkl")
        cfg["model"]["use_cot"] = use_cot

        self.model = AutoVLA(cfg, device="cpu")
        sd = load_state_dict_cached(Path(checkpoint))
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected[:5]
        assert not [m for m in missing if "vlm" in m], missing[:5]
        del sd
        self.model.vlm.to(device=device, dtype=dtype).eval()
        self.model.device = device
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._frame_dir = Path(tempfile.mkdtemp(prefix="autovla_", dir="/dev/shm"))
        self._path_cache, self._counter = {}, 0

    def _paths(self, imgs, cam):
        """get_prompt() expects file paths; write frames to RAM-backed storage as JPEG
        (nuPlan sensor blobs are JPEGs). Cache by object id so history frames are written once."""
        out = []
        for im in imgs:
            key = (cam, id(im))
            if key not in self._path_cache:
                self._counter += 1
                path = self._frame_dir / f"{cam}_{self._counter % 64:02d}.jpg"
                im.save(path, quality=95)
                self._path_cache[key] = (str(path), im)
            out.append(self._path_cache[key][0])
        # keep cache small: only frames currently referenced
        live = {(cam, id(im)) for im in imgs}
        for k in [k for k in self._path_cache if k[0] == cam and k not in live]:
            del self._path_cache[k]
        return out

    @torch.no_grad()
    def __call__(self, front, front_left, front_right, speed, accel, command):
        """front/front_left/front_right: lists of 4 PIL images (oldest -> newest, 2 Hz).

        Returns dict with `trajectory` (10x3 array: x fwd, y left, heading; 0.5 s steps,
        ego frame), `text` (decoded completion), and per-token action log-probs.
        """
        feats = {
            "images": {
                "front_camera": self._paths(front, "f"),
                "front_left_camera": self._paths(front_left, "fl"),
                "front_right_camera": self._paths(front_right, "fr"),
            },
            "sensor_data_path": None,
            "vehicle_velocity": float(speed),
            "vehicle_acceleration": float(accel),
            "driving_command": command,
        }
        inputs = self.model.get_prompt(feats)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        out = self.model.vlm.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
        gen = out.sequences[0, inputs["input_ids"].shape[1]:].cpu()
        logps = torch.stack([torch.log_softmax(s[0].float(), -1) for s in out.scores]).cpu()
        tok_logp = logps[torch.arange(len(gen)), gen]
        text = self.model.processor.decode(gen, skip_special_tokens=False)
        is_action = gen >= self.model.action_start_id
        action_ids = gen[is_action][:10]
        traj = self.model.action_tokenizer.decode_token_ids_to_trajectory(action_ids)[0, 1:]
        traj = traj.numpy() if isinstance(traj, torch.Tensor) else np.zeros((10, 3))
        return {
            "trajectory": traj,
            "text": text,
            "action_logprobs": tok_logp[is_action][:10].numpy(),
            "n_action_tokens": int(is_action.sum()),
        }
