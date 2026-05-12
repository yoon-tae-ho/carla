// Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include "carla/MsgPack.h"

#include <cmath>

namespace carla {
namespace rpc {

  class ChronoSuspensionControl {
  public:

    ChronoSuspensionControl() = default;

    ChronoSuspensionControl(
        float in_damping_fl,
        float in_damping_fr,
        float in_damping_rl,
        float in_damping_rr,
        float in_stiffness_fl,
        float in_stiffness_fr,
        float in_stiffness_rl,
        float in_stiffness_rr)
      : damping_fl(in_damping_fl),
        damping_fr(in_damping_fr),
        damping_rl(in_damping_rl),
        damping_rr(in_damping_rr),
        stiffness_fl(in_stiffness_fl),
        stiffness_fr(in_stiffness_fr),
        stiffness_rl(in_stiffness_rl),
        stiffness_rr(in_stiffness_rr) {}

    float damping_fl = 0.0f;
    float damping_fr = 0.0f;
    float damping_rl = 0.0f;
    float damping_rr = 0.0f;
    float stiffness_fl = 0.0f;
    float stiffness_fr = 0.0f;
    float stiffness_rl = 0.0f;
    float stiffness_rr = 0.0f;

    bool IsValid() const {
      return
          IsValidValue(damping_fl) &&
          IsValidValue(damping_fr) &&
          IsValidValue(damping_rl) &&
          IsValidValue(damping_rr) &&
          IsValidValue(stiffness_fl) &&
          IsValidValue(stiffness_fr) &&
          IsValidValue(stiffness_rl) &&
          IsValidValue(stiffness_rr);
    }

    MSGPACK_DEFINE_ARRAY(
        damping_fl,
        damping_fr,
        damping_rl,
        damping_rr,
        stiffness_fl,
        stiffness_fr,
        stiffness_rl,
        stiffness_rr);

  private:

    static bool IsValidValue(float Value) {
      return std::isfinite(Value) && Value >= 0.0f;
    }
  };

} // namespace rpc
} // namespace carla
