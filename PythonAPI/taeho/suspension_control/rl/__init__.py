"""RL helpers for residual suspension control.

The modules in this package intentionally keep optional ML dependencies behind
small adapters so the CARLA route suite can import without torch, gymnasium, or
stable-baselines3 installed.
"""

from .action_projection import (
    ProjectionResult,
    ResidualProjectionResult,
    ResidualActionProjector,
    ResidualActionProjectorConfig,
)
from .carla_env import SuspensionCarlaEnv, SuspensionSyntheticEnv
from .carla_backend import (
    BaseCarlaRouteBackend,
    BackendResetResult,
    BackendStepResult,
    FakeCarlaRouteBackend,
    FakeCarlaSuspensionBackend,
    LiveCarlaSuspensionBackend,
)
from .carla_online_env import CarlaSuspensionEnv
from .carla_route_env import CarlaRouteSuspensionEnv
from .carla_route_backend import LiveRouteProcessBackend
from .normalizer import FixedScaleNormalizer
from .observations import ObservationBuilder, ObservationSpec
from .policy import PolicyAdapter
from .policy_export import (
    export_sb3_actor_to_torchscript,
    export_torch_module,
    write_normalizer_metadata,
)
from .reward import RewardTransition, SuspensionReward, SuspensionRewardConfig
from .reward_calibration import (
    PHASE3B_REWARD_CALIBRATION_FIELDS,
    phase3b_overall_status,
    phase3b_reward_calibration_rows,
    write_phase3b_reward_calibration_report,
)
from .reward_sanity import (
    PHASE3_REWARD_SANITY_FIELDS,
    phase3_reward_sanity_rows,
    write_phase3_reward_sanity_report,
)
from .rollout_logger import RolloutLogger
from .task_info import (
    DEFAULT_TASK_INFO,
    TaskInfoBuilder,
    TaskInfoBuilderConfig,
    normalize_task_info,
    task_done_reason,
    task_summary_fields,
)

__all__ = [
    "FixedScaleNormalizer",
    "BackendResetResult",
    "BackendStepResult",
    "BaseCarlaRouteBackend",
    "CarlaSuspensionEnv",
    "CarlaRouteSuspensionEnv",
    "FakeCarlaRouteBackend",
    "FakeCarlaSuspensionBackend",
    "LiveCarlaSuspensionBackend",
    "LiveRouteProcessBackend",
    "ObservationBuilder",
    "ObservationSpec",
    "PolicyAdapter",
    "ProjectionResult",
    "ResidualActionProjector",
    "ResidualActionProjectorConfig",
    "ResidualProjectionResult",
    "RewardTransition",
    "RolloutLogger",
    "SuspensionCarlaEnv",
    "SuspensionSyntheticEnv",
    "SuspensionReward",
    "SuspensionRewardConfig",
    "DEFAULT_TASK_INFO",
    "PHASE3B_REWARD_CALIBRATION_FIELDS",
    "PHASE3_REWARD_SANITY_FIELDS",
    "TaskInfoBuilder",
    "TaskInfoBuilderConfig",
    "export_sb3_actor_to_torchscript",
    "export_torch_module",
    "normalize_task_info",
    "phase3b_overall_status",
    "phase3b_reward_calibration_rows",
    "phase3_reward_sanity_rows",
    "task_done_reason",
    "task_summary_fields",
    "write_phase3_reward_sanity_report",
    "write_phase3b_reward_calibration_report",
    "write_normalizer_metadata",
]
