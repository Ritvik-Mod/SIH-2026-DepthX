"""Config loading.  YAML + dotted CLI overrides, e.g.  train.batch_size=4"""
from __future__ import annotations
import sys
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

DEFAULT = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"


def load(path: str | Path | None = None, overrides: list[str] | None = None) -> DictConfig:
    cfg = _load_with_base(Path(path or DEFAULT))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    _validate(cfg)
    return cfg


def _load_with_base(path: Path, _seen=None) -> DictConfig:
    """Supports `_base_: other.yaml` so an ablation file only states its differences."""
    _seen = _seen or set()
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"circular _base_ chain at {path}")
    _seen.add(path)
    cfg = OmegaConf.load(str(path))
    base_ref = cfg.pop("_base_", None)
    if base_ref:
        base = _load_with_base((path.parent / str(base_ref)), _seen)
        cfg = OmegaConf.merge(base, cfg)
    return cfg


def load_from_argv(argv: list[str] | None = None) -> DictConfig:
    argv = list(sys.argv[1:] if argv is None else argv)
    path = None
    rest = []
    for a in argv:
        if a.startswith("--config="):
            path = a.split("=", 1)[1]
        elif a == "--config":
            raise SystemExit("use --config=PATH")
        else:
            rest.append(a)
    return load(path, rest)


def _validate(cfg: DictConfig) -> None:
    """Fail loudly on combinations that silently produce a worse model."""
    if cfg.loss.w_shadow > 0 and cfg.aug.geometric:
        raise ValueError(
            "loss.w_shadow>0 with aug.geometric=true: flips/rotations change the apparent "
            "sun direction while the supplied azimuth does not, so the shadow loss would "
            "penalise correct predictions.  Set aug.geometric=false for the shadow phase."
        )
    if cfg.loss.w_nll > 0 and not cfg.model.use_uncertainty:
        raise ValueError("loss.w_nll>0 requires model.use_uncertainty=true")
    if cfg.loss.w_sem > 0 and not cfg.model.use_semantic:
        raise ValueError("loss.w_sem>0 requires model.use_semantic=true")
    if cfg.model.head not in ("bins", "regression"):
        raise ValueError(f"unknown head {cfg.model.head!r}")
    if cfg.model.head == "regression" and (cfg.loss.w_bin_ce > 0 or cfg.loss.w_chamfer > 0):
        raise ValueError("bin losses require model.head=bins")
    if cfg.data.crop % 14 != 0:
        raise ValueError(f"data.crop must be a multiple of the patch size 14, got {cfg.data.crop}")
