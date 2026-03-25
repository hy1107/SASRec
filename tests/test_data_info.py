from sasrec.data.data_info import get_dataset_info, SUPPORTED_DATASETS


def test_supported_datasets_exist():
    for name in ["beauty", "video_games", "steam", "ml-1m"]:
        assert name in SUPPORTED_DATASETS


def test_get_dataset_info_returns_url_and_filename():
    info = get_dataset_info("beauty")
    assert "url" in info
    assert "raw_filename" in info
    assert isinstance(info["url"], str)
    assert len(info["url"]) > 0


def test_unknown_dataset_raises():
    import pytest
    with pytest.raises(KeyError):
        get_dataset_info("unknown_dataset")
