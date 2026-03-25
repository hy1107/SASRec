# sasrec/config.py
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

_CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def load_config(
    dataset: str | None = None,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Load base config, optionally merge dataset-specific overrides and CLI overrides.

    Args:
        dataset: Name of dataset (e.g. "beauty"). Looks for configs/datasets/{dataset}.yaml.
        overrides: List of OmegaConf dot-notation override strings, e.g. ["model.hidden_units=64"].

    Returns:
        Merged DictConfig.

    Raises:
        FileNotFoundError: If the dataset config file does not exist.
    """
    cfg = OmegaConf.load(_CONFIGS_DIR / "base.yaml")

    if dataset is not None:
        dataset_cfg_path = _CONFIGS_DIR / "datasets" / f"{dataset}.yaml"
        if not dataset_cfg_path.exists():
            raise FileNotFoundError(
                f"No config found for dataset '{dataset}'. "
                f"Expected: {dataset_cfg_path}"
            )
        cfg = OmegaConf.merge(cfg, OmegaConf.load(dataset_cfg_path))

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    return cfg
