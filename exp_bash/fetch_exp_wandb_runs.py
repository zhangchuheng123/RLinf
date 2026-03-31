from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import wandb
from omegaconf import OmegaConf


DEFAULT_METRICS = [
    "eval/success_rate",
    "rollout/success_rate",
    "rollout/recent_success",
    "train/value_loss",
    "train/value_mse",
    "train/ref_value_loss",
    "train/ratio",
    "train/ratio_std",
    "train/ratio_update0",
    "train/ratio_update0_abs_delta_from_1",
    "train/clip_fraction",
    "train/clip_fraction_update0",
    "train/approx_kl",
    "train/ppo_loss",
    "train/old_logprob_mean",
    "train/new_logprob_mean",
    "train/log_ratio_mean",
    "train/advantage_raw_mean",
    "train/advantage_raw_std",
    "train/advantage_raw_abs_mean",
    "train/advantage_raw_min",
    "train/advantage_raw_max",
    "train/advantage_pos_frac",
    "train/actor_grad_norm",
    "train/value_grad_norm",
    "train/value_target_oob_frac",
    "train/value_pred_mean",
    "train/value_pred_min",
    "train/value_pred_max",
    "train/value_target_mean",
    "train/value_target_min",
    "train/value_target_max",
    "train/entropy",
    "rollout/noise_latent_mean",
    "rollout/noise_latent_std",
    "rollout/actor_mean_mean",
    "rollout/actor_mean_std",
    "rollout/actor_logstd_mean",
    "rollout/actor_logstd_std",
]

CONFIG_PATHS = {
    "experiment_name": "runner.logger.experiment_name",
    "value_head_type": "actor.model.dsrl_value_head_type",
    "actor_lr": "actor.optim.dsrl_actor_lr",
    "value_lr": "actor.optim.dsrl_value_lr",
    "train_envs": "env.train.total_num_envs",
    "eval_envs": "env.eval.total_num_envs",
    "num_execute_steps": "runner.num_execute_steps",
    "rollout_epoch": "algorithm.rollout_epoch",
    "update_epoch": "algorithm.update_epoch",
    "pre_value_update_epoch": "algorithm.pre_value_update_epoch",
    "minibatch_size": "algorithm.dsrl_minibatch_size",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch and summarize WandB runs for self-contained exp_bash experiments. "
            "By default, the script discovers runs from exp_bash/config/*.yaml."
        )
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("exp_bash/config"),
        help="Directory containing self-contained exp_bash Hydra configs.",
    )
    parser.add_argument(
        "--config-name",
        action="append",
        default=[],
        help=(
            "Config basename or experiment name to include. Can be passed multiple times. "
            "Without this option, all exp_bash configs are scanned."
        ),
    )
    parser.add_argument(
        "--run-name",
        action="append",
        default=[],
        help="Extra exact WandB run name to include, even if it is not present in exp_bash/config/.",
    )
    parser.add_argument(
        "--name-contains",
        action="append",
        default=[],
        help="Include additional runs whose WandB names contain the given substring.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=100,
        help="Maximum number of matched runs to include in the output.",
    )
    parser.add_argument(
        "--recent-eval-window",
        type=int,
        default=20,
        help="Window size used when summarizing repeated evaluation stability.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        default=[],
        help="Extra metric key to fetch. Can be passed multiple times.",
    )
    parser.add_argument(
        "--include-history",
        action="store_true",
        help="Include full histories in the JSON output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def _nested_get(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _summarize_series(points: list[tuple[int, float]]) -> dict[str, float] | None:
    if not points:
        return None
    steps = [step for step, _ in points]
    values = [value for _, value in points]
    return {
        "step_first": float(steps[0]),
        "step_last": float(steps[-1]),
        "first": values[0],
        "last": values[-1],
        "min": min(values),
        "max": max(values),
        "delta_last_first": values[-1] - values[0],
    }


def _collect_series(run: wandb.apis.public.Run, metrics: list[str]) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = {metric: [] for metric in metrics}
    for metric in metrics:
        points: list[tuple[int, float]] = []
        for row in run.scan_history(keys=["_step", metric]):
            step_raw = row.get("_step")
            step = int(step_raw) if step_raw is not None else 0
            value = _to_float(row.get(metric))
            if value is None:
                continue
            points.append((step, value))
        series[metric] = points
    return series


def _load_exp_configs(config_dir: Path, selected_names: set[str]) -> tuple[list[dict[str, Any]], tuple[str, str, str, str]]:
    config_files = sorted(config_dir.glob("*.yaml"))
    if not config_files:
        raise RuntimeError(f"No YAML configs found under {config_dir}")

    loaded: list[dict[str, Any]] = []
    wandb_settings: tuple[str, str, str, str] | None = None

    for config_path in config_files:
        cfg = OmegaConf.load(config_path)
        experiment_name = str(cfg.runner.logger.experiment_name)
        config_name = config_path.stem
        if selected_names and config_name not in selected_names and experiment_name not in selected_names:
            continue

        current_wandb_settings = (
            str(cfg.wandb.host),
            str(cfg.wandb.entity),
            str(cfg.wandb.project),
            str(cfg.wandb.key),
        )
        if wandb_settings is None:
            wandb_settings = current_wandb_settings
        elif wandb_settings != current_wandb_settings:
            raise RuntimeError(
                "exp_bash configs must share the same wandb host/entity/project/key settings"
            )

        loaded.append(
            {
                "config_path": config_path,
                "config_name": config_name,
                "experiment_name": experiment_name,
            }
        )

    if not loaded:
        raise RuntimeError(
            f"No exp_bash configs matched selected names {sorted(selected_names)} under {config_dir}"
        )
    assert wandb_settings is not None
    return loaded, wandb_settings


def _metric_last(summary: dict[str, Any], metric: str) -> float | None:
    metric_summary = summary["metrics"].get(metric)
    if metric_summary is None:
        return None
    return metric_summary["last"]


def _metric_max(summary: dict[str, Any], metric: str) -> float | None:
    metric_summary = summary["metrics"].get(metric)
    if metric_summary is None:
        return None
    return metric_summary["max"]


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if value != 0.0 and abs(value) < 1.0e-3:
            return f"{value:.2e}"
        return f"{value:.4f}"
    return str(value)


def _longest_full_success_streak(values: list[float]) -> int:
    best = 0
    current = 0
    for value in values:
        if abs(value - 1.0) < 1e-12:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _recent_eval_summary(points: list[tuple[int, float]], window: int) -> dict[str, Any] | None:
    if not points:
        return None
    values = [value for _, value in points]
    tail = values[-window:] if len(values) >= window else values
    return {
        "window": len(tail),
        "mean": sum(tail) / float(len(tail)),
        "min": min(tail),
        "max": max(tail),
        "full_success_frac": sum(1 for value in tail if abs(value - 1.0) < 1e-12) / float(len(tail)),
        "longest_full_success_streak": _longest_full_success_streak(values),
    }


def _print_table(run_summaries: list[dict[str, Any]], recent_window: int) -> None:
    headers = [
        "config_name",
        "run_name",
        "state",
        "head",
        "actor_lr",
        "value_lr",
        "envs",
        "exec",
        "mb",
        f"eval_recent{recent_window}_mean",
        f"eval_recent{recent_window}_full_frac",
        "eval_best",
        "eval_last",
        "longest_eval_1.0_streak",
        "rollout_recent_last",
        "value_mse_last",
        "adv_abs_last",
        "actor_gn_last",
        "approx_kl_last",
    ]
    print(" | ".join(headers))
    for summary in run_summaries:
        recent_eval = summary.get("recent_eval") or {}
        row = [
            str(summary.get("config_name", "-")),
            summary["name"],
            summary["state"],
            _format_value(summary["config"]["value_head_type"]),
            _format_value(summary["config"]["actor_lr"]),
            _format_value(summary["config"]["value_lr"]),
            _format_value(summary["config"]["train_envs"]),
            _format_value(summary["config"]["num_execute_steps"]),
            _format_value(summary["config"]["minibatch_size"]),
            _format_value(recent_eval.get("mean")),
            _format_value(recent_eval.get("full_success_frac")),
            _format_value(_metric_max(summary, "eval/success_rate")),
            _format_value(_metric_last(summary, "eval/success_rate")),
            _format_value(recent_eval.get("longest_full_success_streak")),
            _format_value(_metric_last(summary, "rollout/recent_success")),
            _format_value(_metric_last(summary, "train/value_mse")),
            _format_value(_metric_last(summary, "train/advantage_raw_abs_mean")),
            _format_value(_metric_last(summary, "train/actor_grad_norm")),
            _format_value(_metric_last(summary, "train/approx_kl")),
        ]
        print(" | ".join(row))


def main() -> None:
    args = _parse_args()
    metrics = list(dict.fromkeys([*DEFAULT_METRICS, *args.metric]))

    selected_config_names = set(args.config_name)
    exp_configs, wandb_settings = _load_exp_configs(args.config_dir, selected_config_names)
    wandb_host, wandb_entity, wandb_project, wandb_key = wandb_settings

    exp_name_to_config: dict[str, dict[str, Any]] = {
        entry["experiment_name"]: entry for entry in exp_configs
    }
    explicit_run_names = set(args.run_name)
    managed_run_names = set(exp_name_to_config)

    os.environ["WANDB_API_KEY"] = wandb_key
    api = wandb.Api(overrides={"base_url": wandb_host})
    project_path = f"{wandb_entity}/{wandb_project}"

    matched_runs: list[wandb.apis.public.Run] = []
    seen_paths: set[str] = set()
    for run in api.runs(project_path):
        run_name = str(run.name or "")
        keep = False
        if run_name in managed_run_names or run_name in explicit_run_names:
            keep = True
        else:
            for substring in args.name_contains:
                if substring.lower() in run_name.lower():
                    keep = True
                    break
        if not keep:
            continue
        run_path = "/".join(run.path)
        if run_path in seen_paths:
            continue
        seen_paths.add(run_path)
        matched_runs.append(run)
        if len(matched_runs) >= args.max_runs:
            break

    if not matched_runs:
        raise RuntimeError(
            "No WandB runs matched the managed exp_bash configs and provided run filters. "
            "This usually means the exp_bash experiments have not been launched to WandB yet. "
            "Run the exp_bash experiments first, or use --run-name/--name-contains to include a legacy reference run."
        )

    run_summaries: list[dict[str, Any]] = []
    for run in matched_runs:
        resolved_run = api.run("/".join(run.path))
        series = _collect_series(resolved_run, metrics)
        metric_summaries = {
            metric: summarized
            for metric, points in series.items()
            if (summarized := _summarize_series(points)) is not None
        }
        run_config = dict(resolved_run.config)
        config_name = None
        local_config = exp_name_to_config.get(str(resolved_run.name or ""))
        if local_config is not None:
            config_name = local_config["config_name"]
        recent_eval = _recent_eval_summary(series.get("eval/success_rate", []), args.recent_eval_window)
        summary = {
            "id": resolved_run.id,
            "name": str(resolved_run.name or ""),
            "state": str(resolved_run.state or ""),
            "url": str(resolved_run.url),
            "project": project_path,
            "created_at": str(resolved_run.created_at),
            "config_name": config_name,
            "config": {
                alias: _nested_get(run_config, path)
                for alias, path in CONFIG_PATHS.items()
            },
            "metrics": metric_summaries,
            "recent_eval": recent_eval,
        }
        if args.include_history:
            summary["history"] = {
                metric: [{"step": step, "value": value} for step, value in points]
                for metric, points in series.items()
                if points
            }
        run_summaries.append(summary)

    run_summaries.sort(
        key=lambda item: (
            (item.get("recent_eval") or {}).get("mean") or float("-inf"),
            _metric_max(item, "eval/success_rate") or float("-inf"),
            item["created_at"],
        ),
        reverse=True,
    )

    result = {
        "config_dir": str(args.config_dir),
        "wandb_host": wandb_host,
        "project": project_path,
        "matched_runs": len(run_summaries),
        "recent_eval_window": args.recent_eval_window,
        "metrics": metrics,
        "runs": run_summaries,
    }

    _print_table(run_summaries, args.recent_eval_window)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nSaved JSON report to {args.output}")


if __name__ == "__main__":
    main()