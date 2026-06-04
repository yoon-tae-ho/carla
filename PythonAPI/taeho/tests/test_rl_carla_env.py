from suspension_control.rl.carla_env import SuspensionCarlaEnv


def test_initial_transition_reward_mask_zeroes_first_three_seconds():
    env = SuspensionCarlaEnv(
        dry_run=True,
        max_steps=80,
        dt=0.05,
        initial_transition_skip_seconds=3.0)
    env.reset()

    _, reward, _, _, info = env.step([0.0, 0.0, 0.0, 0.0])

    assert reward == 0.0
    assert info["reward_diagnostics"]["reward_initial_masked"] == 1.0
    assert info["reward_diagnostics"]["reward_total"] == 0.0


def test_reward_unmasked_after_initial_three_seconds():
    env = SuspensionCarlaEnv(
        dry_run=True,
        max_steps=80,
        dt=0.05,
        initial_transition_skip_seconds=3.0)
    env.reset()

    info = None
    reward = 0.0
    for _ in range(60):
        _, reward, _, _, info = env.step([0.0, 0.0, 0.0, 0.0])

    assert info is not None
    assert info["reward_diagnostics"]["reward_initial_masked"] == 0.0
    assert reward == info["reward_diagnostics"]["reward_total"]
