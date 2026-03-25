import pytest
from pathlib import Path
from sasrec.data.preprocessor import preprocess, load_processed_data


@pytest.fixture
def sample_csv(tmp_path):
    """Create a tiny CSV interaction file (5 users × 7 items so each item hits min_interactions=5)."""
    csv_file = tmp_path / "interactions.csv"
    rows = []
    for uid in range(1, 6):  # 5 users → each item appears 5 times
        for iid in range(1, 8):  # 7 interactions per user
            rows.append(f"{uid},{iid},{uid * 100 + iid}")
    csv_file.write_text("user_id,item_id,timestamp\n" + "\n".join(rows))
    return csv_file


def test_preprocess_csv_creates_output_files(sample_csv, tmp_path):
    out_dir = tmp_path / "processed"
    preprocess(input_path=sample_csv, output_dir=out_dir, fmt="csv", min_interactions=5)
    assert (out_dir / "train.txt").exists()
    assert (out_dir / "valid.txt").exists()
    assert (out_dir / "test.txt").exists()
    assert (out_dir / "item_count.txt").exists()


def test_preprocess_leave_one_out_split(sample_csv, tmp_path):
    out_dir = tmp_path / "processed"
    preprocess(input_path=sample_csv, output_dir=out_dir, fmt="csv", min_interactions=5)

    train_data, valid_data, test_data, item_count = load_processed_data(out_dir)

    assert set(train_data.keys()) == set(valid_data.keys()) == set(test_data.keys())

    for uid in train_data:
        full_seq = train_data[uid]
        val_item = valid_data[uid]
        test_item = test_data[uid]
        assert isinstance(val_item, int)
        assert isinstance(test_item, int)
        assert test_item not in full_seq
        assert val_item not in full_seq


def test_min_interactions_filter(tmp_path):
    """Users with fewer than min_interactions should be filtered out."""
    csv_file = tmp_path / "sparse.csv"
    rows = []
    # Users 1-6: each interacts with items 1-6 (6 interactions each → passes user filter;
    # each item appears 6 times → passes item filter with min_interactions=5)
    for uid in range(1, 7):
        for iid in range(1, 7):
            rows.append(f"{uid},{iid},{uid * 10 + iid}")
    # User 7: only 3 interactions → fails user filter
    for iid in range(1, 4):
        rows.append(f"7,{iid},{70 + iid}")
    csv_file.write_text("user_id,item_id,timestamp\n" + "\n".join(rows))

    out_dir = tmp_path / "processed"
    preprocess(input_path=csv_file, output_dir=out_dir, fmt="csv", min_interactions=5)

    train_data, _, _, _ = load_processed_data(out_dir)
    assert len(train_data) == 6  # users 1-6 pass; user 7 filtered


def test_item_ids_remapped_to_contiguous(sample_csv, tmp_path):
    out_dir = tmp_path / "processed"
    preprocess(input_path=sample_csv, output_dir=out_dir, fmt="csv", min_interactions=5)
    train_data, valid_data, test_data, item_count = load_processed_data(out_dir)
    all_items = (
        {item for seq in train_data.values() for item in seq}
        | set(valid_data.values())
        | set(test_data.values())
    )
    assert max(all_items) == item_count  # 1-indexed contiguous


def test_missing_raw_file_raises(tmp_path):
    out_dir = tmp_path / "processed"
    with pytest.raises(FileNotFoundError):
        preprocess(
            input_path=tmp_path / "nonexistent.csv",
            output_dir=out_dir,
            fmt="csv",
        )
