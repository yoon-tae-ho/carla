from suspension_control.rl.policy import PolicyAdapter


def test_policy_import_and_missing_path_do_not_require_torch():
    policy = PolicyAdapter(policy_path="", action_dim=4, allow_dummy_zero=False)
    assert policy.is_available is False
    assert policy.status == "missing_path"
    assert policy.error == "policy_path_empty"


def test_missing_policy_file_marks_unavailable(tmp_path):
    policy = PolicyAdapter(
        policy_path=str(tmp_path / "missing.pt"),
        action_dim=4,
        allow_dummy_zero=False)
    assert policy.is_available is False
    assert policy.status == "missing_file"
    assert policy.error == "policy_file_missing"


def test_dummy_zero_policy_returns_correct_action_shape(tmp_path):
    policy = PolicyAdapter(
        policy_path=str(tmp_path / "missing.pt"),
        action_dim=4,
        allow_dummy_zero=True)
    assert policy.is_available is True
    assert policy.predict([1.0, 2.0]) == [0.0, 0.0, 0.0, 0.0]
