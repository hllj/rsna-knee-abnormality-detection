"""
Configuration loading.

configs/default.yaml is the base; `exp=<name>` merges configs/exp/<name>.yaml on top, and any
other `a.b.c=value` CLI token overrides a single key (OmegaConf dotlist syntax).
"""
import os
import sys
from typing import List, Optional

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO_ROOT, "configs")
if not OmegaConf.has_resolver("repo"):
    OmegaConf.register_new_resolver("repo", lambda: REPO_ROOT)  # ${repo:} = repository root


def get_config(overrides: Optional[List[str]] = None, config_name: str = "default") -> DictConfig:
    """
    Load the base config, an optional experiment file and CLI overrides.

    Args:
        overrides: dotlist tokens, e.g. ["exp=coatnet_384", "fold=2", "training.lr=1e-4"]
        config_name: base config file name (without extension)

    Returns:
        Resolved DictConfig
    """
    overrides = list(sys.argv[1:] if overrides is None else overrides)
    cfg = OmegaConf.load(os.path.join(CONFIG_PATH, f"{config_name}.yaml"))

    exp = [o for o in overrides if o.startswith("exp=")]
    if exp:
        name = exp[-1].split("=", 1)[1]
        exp_cfg = OmegaConf.load(os.path.join(CONFIG_PATH, "exp", f"{name}.yaml"))
        cfg = OmegaConf.merge(cfg, exp_cfg)
        if OmegaConf.select(exp_cfg, "preprocess.slots") is not None:  # a slot layout replaces, never unions
            cfg.preprocess.slots = exp_cfg.preprocess.slots
    rest = [o for o in overrides if "=" in o and not o.startswith("exp=")]
    if rest:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(rest))
    OmegaConf.resolve(cfg)
    return cfg


if __name__ == "__main__":
    print(OmegaConf.to_yaml(get_config()))
