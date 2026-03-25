from sasrec.config import load_config


def test_load_base_config():
    cfg = load_config()
    assert cfg.data.maxlen == 50
    assert cfg.model.hidden_units == 50
    assert cfg.train.seed == 42


def test_dataset_override():
    cfg = load_config(dataset="beauty")
    assert cfg.data.maxlen == 50  # beauty uses base default


def test_cli_override():
    cfg = load_config(overrides=["model.hidden_units=64"])
    assert cfg.model.hidden_units == 64


def test_unknown_dataset_raises():
    import pytest
    with pytest.raises(FileNotFoundError):
        load_config(dataset="nonexistent")
