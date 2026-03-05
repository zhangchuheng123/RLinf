import json

import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from rlinf_noray.runners.libero_ppo_ddp_runner import LiberoPPODDPNoRayRunner

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1", config_path="config", config_name="libero_10_ppo_openpi_pi05"
)
def main(cfg) -> None:
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2), flush=True)

    config_name = str(cfg.runner.logger.experiment_name).lower()
    allowed_configs = {
        "libero_10_ppo_openpi_pi0",
        "libero_10_ppo_openpi_pi05",
        "libero_10_ppo_smolvla",
    }
    assert config_name in allowed_configs, (
        f"Only {sorted(allowed_configs)} are supported in noray runner, got {config_name}"
    )

    model_type = str(cfg.actor.model.model_type).lower()
    assert model_type in {"openpi", "smolvla"}, (
        f"Only openpi/smolvla are supported in noray runner, got {model_type}"
    )

    env_type = str(cfg.env.train.env_type).lower()
    assert env_type == "libero", (
        f"Only libero env is supported in noray runner, got {env_type}"
    )

    runner = LiberoPPODDPNoRayRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
