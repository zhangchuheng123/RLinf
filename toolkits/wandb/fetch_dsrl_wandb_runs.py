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
    "train/ratio_update0",
    "train/ratio_update0_abs_delta_from_1",
    "train/clip_fraction",
    "train/clip_fraction_update0",
    "train/value_target_oob_frac",
    "train/value_pred_mean",
    "train/value_pred_min",
    "train/value_pred_max",
    "train/value_target_mean",
    "train/value_target_min",
    "train/value_target_max",
    "train/entropy",
]

CONFIG_PATHS = {
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
            "Fetch and summarize DSRL WandB runs using the credentials and host "
            "configured in the repository."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/embodiment/config/libero_10_dsrl_smolvla.yaml"),
        help="Hydra config file containing wandb.host/entity/project/key.",
    )
    parser.add_argument(
        "--name-contains",
        type=str,
        default="libero_10_dsrl_smolvla",
        help="Only keep runs whose WandB run name contains this substring.",
    )
    parser.add_argument(
        "--run-name",
        action="append",
        default=[],
        help="Exact run name to include. Can be passed multiple times.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=50,
        help="Maximum number of matching runs to fetch.",
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
        help="Include full metric histories in the JSON output.",
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
    keys = ["_step", *metrics]
    series: dict[str, list[tuple[int, float]]] = {metric: [] for metric in metrics}
    for row in run.scan_history(keys=keys):
        step_raw = row.get("_step")
        step = int(step_raw) if step_raw is not None else 0
        for metric in metrics:
            value = _to_float(row.get(metric))
            if value is None:
                continue
            series[metric].append((step, value))
    return series


def _load_wandb_settings(config_path: Path) -> tuple[str, str, str, str]:
    cfg = OmegaConf.load(config_path)
    return (
        str(cfg.wandb.host),
        str(cfg.wandb.entity),
        str(cfg.wandb.project),
        str(cfg.wandb.key),
    )


def _select_runs(
    runs: list[wandb.apis.public.Run],
    name_contains: str,
    explicit_names: set[str],
    max_runs: int,
) -> list[wandb.apis.public.Run]:
    selected: list[wandb.apis.public.Run] = []
    lowered_substring = name_contains.lower()
    for run in runs:
        run_name = str(run.name or "")
        if explicit_names:
            if run_name not in explicit_names:
                continue
        elif lowered_substring not in run_name.lower():
            continue
        selected.append(run)
        if len(selected) >= max_runs:
            break
    return selected


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
        return f"{value:.4f}"
    return str(value)


def _print_table(run_summaries: list[dict[str, Any]]) -> None:
    headers = [
        "run_name",
        "state",
        "head",
        "envs",
        "exec",
        "eval_best",
        "eval_last",
        "rollout_best",
        "recent_last",
        "value_mse_last",
        "ref_value_last",
        "ratio0_abs_last",
        "oob_last",
    ]
    print(" | ".join(headers))
    for summary in run_summaries:
        row = [
            summary["name"],
            summary["state"],
            _format_value(summary["config"]["value_head_type"]),
            _format_value(summary["config"]["train_envs"]),
            _format_value(summary["config"]["num_execute_steps"]),
            _format_value(_metric_max(summary, "eval/success_rate")),
            _format_value(_metric_last(summary, "eval/success_rate")),
            _format_value(_metric_max(summary, "rollout/success_rate")),
            _format_value(_metric_last(summary, "rollout/recent_success")),
            _format_value(_metric_last(summary, "train/value_mse")),
            _format_value(_metric_last(summary, "train/ref_value_loss")),
            _format_value(_metric_last(summary, "train/ratio_update0_abs_delta_from_1")),
            _format_value(_metric_last(summary, "train/value_target_oob_frac")),
        ]
        print(" | ".join(row))


def main() -> None:
    args = _parse_args()
    metrics = list(dict.fromkeys([*DEFAULT_METRICS, *args.metric]))
    wandb_host, wandb_entity, wandb_project, wandb_key = _load_wandb_settings(args.config)

    os.environ["WANDB_API_KEY"] = wandb_key
    api = wandb.Api(overrides={"base_url": wandb_host})
    project_path = f"{wandb_entity}/{wandb_project}"
    runs = list(api.runs(project_path))
    selected_runs = _select_runs(
        runs=runs,
        name_contains=args.name_contains,
        explicit_names=set(args.run_name),
        max_runs=args.max_runs,
    )

    if not selected_runs:
        raise RuntimeError(
            f"No runs matched name_contains={args.name_contains!r} and run_name={args.run_name!r}"
        )

    run_summaries: list[dict[str, Any]] = []
    for run in selected_runs:
        resolved_run = api.run("/".join(run.path))
        series = _collect_series(resolved_run, metrics)
        metric_summaries = {
            metric: summarized
            for metric, points in series.items()
            if (summarized := _summarize_series(points)) is not None
        }
        run_config = dict(resolved_run.config)
        summary = {
            "id": resolved_run.id,
            "name": str(resolved_run.name or ""),
            "state": str(resolved_run.state or ""),
            "url": str(resolved_run.url),
            "project": project_path,
            "created_at": str(resolved_run.created_at),
            "config": {
                alias: _nested_get(run_config, path)
                for alias, path in CONFIG_PATHS.items()
            },
            "metrics": metric_summaries,
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
            _metric_max(item, "eval/success_rate") or float("-inf"),
            item["created_at"],
        ),
        reverse=True,
    )

    result = {
        "config": str(args.config),
        "wandb_host": wandb_host,
        "project": project_path,
        "matched_runs": len(run_summaries),
        "metrics": metrics,
        "runs": run_summaries,
    }

    _print_table(run_summaries)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nSaved JSON report to {args.output}")


if __name__ == "__main__":
    main()