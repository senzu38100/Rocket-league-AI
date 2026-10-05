"""Tests for genericBot's checkpoint-folder completeness check.

rlgym_ppo's ppo_learner.load_from() does torch.load() on 4 specific files with
no existence check; a checkpoint folder straight out of a git checkout only
has BOOK_KEEPING_VARS.json (the *.pt weights are gitignored), so pointing
Learner at one crashes with FileNotFoundError before training starts.
"""

import genericBot

REQUIRED_FILES = (
    "PPO_POLICY.pt",
    "PPO_VALUE_NET.pt",
    "PPO_POLICY_OPTIMIZER.pt",
    "PPO_VALUE_NET_OPTIMIZER.pt",
)


def test_folder_with_only_bookkeeping_json_is_incomplete(tmp_path):
    (tmp_path / "BOOK_KEEPING_VARS.json").write_text("{}")
    assert not genericBot._checkpoint_is_complete(str(tmp_path))


def test_folder_with_all_four_weight_files_is_complete(tmp_path):
    for name in REQUIRED_FILES:
        (tmp_path / name).write_bytes(b"")
    assert genericBot._checkpoint_is_complete(str(tmp_path))


def test_folder_missing_one_weight_file_is_incomplete(tmp_path):
    for name in REQUIRED_FILES[:-1]:
        (tmp_path / name).write_bytes(b"")
    assert not genericBot._checkpoint_is_complete(str(tmp_path))


def test_nonexistent_folder_is_incomplete():
    assert not genericBot._checkpoint_is_complete("/nonexistent/path/xyz")
