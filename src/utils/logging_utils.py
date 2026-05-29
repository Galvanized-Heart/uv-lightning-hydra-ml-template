import json, hashlib, os, socket, subprocess
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import wandb
from omegaconf import OmegaConf
from lightning.pytorch.utilities import rank_zero_only

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

MANIFEST = Path(os.environ.get("PROJECT_ROOT", ".")) / "runs_manifest.parquet"


def flatten_cfg(cfg) -> dict:
    resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    out = {}

    def _flat(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _flat(v, f"{prefix}{k}.")
        elif isinstance(obj, list):
            out[prefix.rstrip(".")] = json.dumps(obj)
        else:
            out[prefix.rstrip(".")] = obj

    _flat(resolved)
    return out


@rank_zero_only
def save_run_record(cfg, flat_cfg: dict, metrics: dict,
                    checkpoint_path: str, prediction_cache_path: str = "") -> None:
    run_id  = wandb.run.id  if wandb.run else "local"
    run_url = wandb.run.url if wandb.run else ""

    record = {
        "run_id":                run_id,
        "wandb_url":             run_url,
        "config_hash":           hashlib.md5(
                                     json.dumps(flat_cfg, sort_keys=True).encode()
                                 ).hexdigest(),
        "hydra_output_dir":      str(Path.cwd()),
        "checkpoint_path":       checkpoint_path,
        "prediction_cache_path": prediction_cache_path,
        "git_commit":            _git_hash(),
        "dataset_hash":          cfg.get("dataset_hash") or "",
        "hostname":              socket.gethostname(),
        "timestamp":             pd.Timestamp.now().isoformat(),
        **{f"cfg.{k}": v for k, v in flat_cfg.items()},
        **{f"metric.{k}": v for k, v in metrics.items()},
    }

    df = pd.DataFrame([record])
    if MANIFEST.exists():
        existing = pd.read_parquet(MANIFEST)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(MANIFEST, index=False)


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]
        ).decode().strip()
    except Exception:
        return ""


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    hparams = {}
    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]
    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")
    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)