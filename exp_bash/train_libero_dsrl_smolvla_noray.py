import hydra
import torch.multiprocessing as mp

from rlinf_noray.runners.libero_dsrl_ddp_runner import LiberoDSRLDDPNoRayRunner

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="20260331_baseline_scalar_mc_env16_exec4_mb2048",
)
def main(cfg) -> None:
    model_type = str(cfg.actor.model.model_type).lower()
    env_type = str(cfg.env.train.env_type).lower()

    assert model_type == "smolvla", (
        "exp_bash DSRL entry only supports smolvla model, got "
        f"{model_type}"
    )
    assert env_type == "libero", (
        "exp_bash DSRL entry only supports libero env, got "
        f"{env_type}"
    )

    runner = LiberoDSRLDDPNoRayRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
