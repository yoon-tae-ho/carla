// Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include "carla/MsgPack.h"
#include "carla/rpc/String.h"
#include "carla/rpc/WheelSuspensionState.h"

#include <cstdint>
#include <string>
#include <vector>

namespace carla {
namespace rpc {

  class SuspensionState {
  public:

    SuspensionState() = default;

    const std::vector<WheelSuspensionState> &GetWheels() const {
      return wheels;
    }

    int64_t frame = 0;
    float timestamp = 0.0f;
    int32_t actor_id = 0;
    int32_t wheel_count = 0;
    std::string state_source;
    std::string failure_reason;
    bool state_valid = false;
    bool velocity_valid = false;
    bool compression_convention_validated = false;
    std::vector<WheelSuspensionState> wheels;

    bool operator!=(const SuspensionState &rhs) const {
      return
        frame != rhs.frame ||
        timestamp != rhs.timestamp ||
        actor_id != rhs.actor_id ||
        wheel_count != rhs.wheel_count ||
        state_source != rhs.state_source ||
        failure_reason != rhs.failure_reason ||
        state_valid != rhs.state_valid ||
        velocity_valid != rhs.velocity_valid ||
        compression_convention_validated != rhs.compression_convention_validated ||
        wheels != rhs.wheels;
    }

    bool operator==(const SuspensionState &rhs) const {
      return !(*this != rhs);
    }

#ifdef LIBCARLA_INCLUDED_FROM_UE4

    SuspensionState(const FSuspensionState &State)
      : frame(State.Frame),
        timestamp(State.Timestamp),
        actor_id(State.ActorId),
        wheel_count(State.WheelCount),
        state_source(FromFString(State.StateSource)),
        failure_reason(FromFString(State.FailureReason)),
        state_valid(State.bStateValid),
        velocity_valid(State.bVelocityValid),
        compression_convention_validated(State.bCompressionConventionValidated) {
      wheels = std::vector<WheelSuspensionState>();
      wheels.reserve(static_cast<size_t>(State.Wheels.Num()));
      for (const auto &Wheel : State.Wheels) {
        wheels.push_back(WheelSuspensionState(Wheel));
      }
    }

#endif

    MSGPACK_DEFINE_ARRAY(frame,
        timestamp,
        actor_id,
        wheel_count,
        state_source,
        failure_reason,
        state_valid,
        velocity_valid,
        compression_convention_validated,
        wheels)
  };

}
}
