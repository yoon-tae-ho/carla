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


def test_builtin_dummy_zero_policy_is_explicitly_loaded():
    policy = PolicyAdapter(
        policy_path="dummy_zero",
        action_dim=4,
        allow_dummy_zero=False)
    assert policy.is_available is True
    assert policy.status == "dummy_zero_loaded"
    assert policy.predict([1.0, 2.0]) == [0.0, 0.0, 0.0, 0.0]


def test_builtin_constant_plus_policy_returns_small_nonzero_action():
    policy = PolicyAdapter(
        policy_path="const_action_plus_0p02",
        action_dim=4,
        allow_dummy_zero=False)
    assert policy.is_available is True
    assert policy.status == "const_action_plus_0p02_loaded"
    assert policy.predict([1.0, 2.0]) == [0.02, 0.02, 0.02, 0.02]
    assert policy.diagnostics()["policy_alias_deprecated"] == 0


def test_deprecated_constant_alias_still_loads_with_diagnostic_flag():
    policy = PolicyAdapter(
        policy_path="const_plus_0p02",
        action_dim=4,
        allow_dummy_zero=False)
    assert policy.is_available is True
    assert policy.status == "const_action_plus_0p02_loaded"
    assert policy.builtin_id == "const_action_plus_0p02"
    assert policy.diagnostics()["policy_alias_deprecated"] == 1
    assert policy.predict([1.0, 2.0]) == [0.02, 0.02, 0.02, 0.02]


def test_builtin_constant_action_policy_supports_larger_canary_actions():
    policy = PolicyAdapter(
        policy_path="const_action_minus_0p25",
        action_dim=4,
        allow_dummy_zero=False)
    assert policy.is_available is True
    assert policy.status == "const_action_minus_0p25_loaded"
    assert policy.predict([1.0, 2.0]) == [-0.25, -0.25, -0.25, -0.25]


def test_builtin_random_small_policy_is_deterministic_sequence():
    policy_a = PolicyAdapter(
        policy_path="random_action_0p02",
        action_dim=4,
        allow_dummy_zero=False)
    policy_b = PolicyAdapter(
        policy_path="random_action_0p02",
        action_dim=4,
        allow_dummy_zero=False)
    action_a = policy_a.predict([1.0, 2.0])
    action_b = policy_b.predict([1.0, 2.0])
    assert action_a == action_b
    assert any(abs(value) > 0.0 for value in action_a)
    assert all(-0.02 <= value <= 0.02 for value in action_a)


def test_deprecated_random_alias_still_loads_with_diagnostic_flag():
    policy = PolicyAdapter(
        policy_path="random_small_0p02",
        action_dim=4,
        allow_dummy_zero=False)
    assert policy.is_available is True
    assert policy.builtin_id == "random_action_0p02"
    assert policy.diagnostics()["policy_alias_deprecated"] == 1
