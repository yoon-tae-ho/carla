import math

from suspension_control.rl.policy import PolicyAdapter
from suspension_control.rl.policy_export import export_torch_module


def test_phase4_policy_ts_contract_outputs_batch_actions(tmp_path):
    try:
        import torch
    except Exception:
        return

    actor = torch.nn.Sequential(
        torch.nn.Linear(5, 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 4))
    policy_path = export_torch_module(
        actor,
        output_path=str(tmp_path / "policy.ts"),
        observation_dim=5,
        action_dim=4)

    module = torch.jit.load(policy_path, map_location="cpu")
    single = module(torch.zeros(5))
    batch = module(torch.zeros((3, 5)))
    assert tuple(single.shape) == (4,)
    assert tuple(batch.shape) == (3, 4)
    assert float(batch.abs().max()) <= 1.0

    adapter = PolicyAdapter(policy_path=policy_path, action_dim=4)
    action = adapter.predict([0.0, 0.1, -0.1, 0.2, -0.2])
    assert len(action) == 4
    assert all(math.isfinite(value) for value in action)
    assert all(-1.0 <= value <= 1.0 for value in action)

