# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from omegaconf import DictConfig


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.smolvlm_with_expert import (
        SmolVLMWithExpertModel,
    )

    from rlinf.models.embodiment.smolvla.smolvla_policy import (
        SmolVLAForRLActionPrediction,
    )

    # ---- Step 1: build architecture (random init, CPU) ----
    # Prevent the VLM backbone from loading via
    #   AutoModel.from_pretrained(..., device_map="auto", low_cpu_mem_usage=True)
    # which produces meta-device tensors that crash safetensors.torch.load_model.
    # Instead, create the VLM from config only (CPU random init).
    # This mirrors the OpenPI pattern where the architecture is created first
    # and weights are loaded separately.
    _orig_init = SmolVLMWithExpertModel.__init__

    def _cpu_only_init(self_inner, *args, **kwargs):
        kwargs["load_vlm_weights"] = False
        _orig_init(self_inner, *args, **kwargs)

    SmolVLMWithExpertModel.__init__ = _cpu_only_init
    try:
        # from_pretrained handles config parsing internally, creates the
        # architecture with load_vlm_weights=False, then loads ALL weights
        # (including VLM backbone) from the safetensors checkpoint.
        policy = SmolVLAPolicy.from_pretrained(cfg.model_path)
    finally:
        SmolVLMWithExpertModel.__init__ = _orig_init

    # ---- Step 2: wrap in RL model (value head, etc.) ----
    model = SmolVLAForRLActionPrediction(cfg, policy=policy)
    model.to(torch_dtype)
    return model
