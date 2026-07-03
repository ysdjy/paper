"""Training helpers for models_v2: multi-seed fit + seed ensembling + save/load.

Rules honored here:
  * train only on train (+val) session pairs; test is never seen at fit time.
  * torch models fit with >=5 seeds; numpy models are deterministic (single fit).
  * predictions are seed-ensembled (mean) for the reported metrics; per-seed metrics are
    also returned so seed variance can be reported.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import REGISTRY
from .numpy_models import B0  # noqa: F401 (registry import side effect)

_TORCH_NAMES = {"B2_deepsets", "B2_gru"}


def is_torch(name: str) -> bool:
    return name in _TORCH_NAMES


def train_seeds(model_name: str, train_pairs, val_pairs, seeds=(0, 1, 2, 3, 4), **hp):
    """Return a list of fitted models. numpy models ignore seeds (return one)."""
    cls = REGISTRY[model_name]
    if not is_torch(model_name):
        return [cls().fit(train_pairs, val_pairs)]
    models = []
    for s in seeds:
        m = cls(**hp) if hp else cls()
        m.fit(train_pairs, val_pairs=val_pairs, seed=s)
        models.append(m)
    return models


def ensemble_predict(models, e, H) -> dict:
    ps = np.array([m.predict(e, H)["p_success"] for m in models])
    es = np.array([m.predict(e, H)["pred_error"] for m in models])
    ts = np.array([m.predict(e, H)["pred_time"] for m in models])
    return {"p_success": float(ps.mean()), "pred_error": float(es.mean()),
            "pred_time": float(ts.mean()),
            "p_success_std": float(ps.std()), "n_seeds": len(models)}


def save_models(models, out_dir: str | Path, model_name: str) -> dict:
    """Persist config for every model; torch weights when available."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"model_name": model_name, "n_members": len(models), "members": []}
    for i, m in enumerate(models):
        entry = {"config": m.config()}
        if hasattr(m, "state_dict") and is_torch(model_name):
            import torch
            wpath = out / f"{model_name}_seed{i}.pt"
            torch.save(m.state_dict(), wpath)
            entry["weights"] = wpath.name
        manifest["members"].append(entry)
    (out / f"{model_name}_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
