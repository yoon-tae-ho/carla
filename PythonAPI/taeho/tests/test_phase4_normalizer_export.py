import json
import math

from suspension_control.rl.normalizer import FixedScaleNormalizer
from suspension_control.rl.policy_export import write_normalizer_metadata


def test_phase4_normalizer_json_reproduces_fixed_scale_normalization(tmp_path):
    feature_names = ("speed", "roll", "previous_action_fl", "preview_target_speed_now")
    path = write_normalizer_metadata(
        str(tmp_path / "normalizer.json"),
        feature_names=feature_names,
        clip=5.0,
        action_semantics="normalized_damper_residual_v1",
        max_damper_residual_scale=0.08)

    with open(path) as json_file:
        data = json.load(json_file)

    assert data["source"] == "fixed_scale_phase4"
    assert data["observation_clip"] == 5.0
    assert data["feature_names"] == list(feature_names)
    assert set(data["mean"]) == set(feature_names)
    assert set(data["std"]) == set(feature_names)

    default = FixedScaleNormalizer(clip=5.0)
    loaded = FixedScaleNormalizer(normalizer_path=path, clip=5.0)
    assert loaded.loaded is True
    for name in feature_names:
        for value in (-12.5, -1.0, 0.0, 2.5, 30.0):
            assert math.isclose(
                default.normalize(name, value),
                loaded.normalize(name, value),
                rel_tol=0.0,
                abs_tol=1.0e-12)

