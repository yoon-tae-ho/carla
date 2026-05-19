// Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include "carla/MsgPack.h"
#include "carla/rpc/WheelSuspensionPhysicsControl.h"

#include <vector>

namespace carla {
namespace rpc {

  class SuspensionPhysicsControl {
  public:

    SuspensionPhysicsControl() = default;

    explicit SuspensionPhysicsControl(
        std::vector<WheelSuspensionPhysicsControl> &in_wheels)
      : wheels(in_wheels) {}

    const std::vector<WheelSuspensionPhysicsControl> &GetWheels() const {
      return wheels;
    }

    void SetWheels(std::vector<WheelSuspensionPhysicsControl> &in_wheels) {
      wheels = in_wheels;
    }

    std::vector<WheelSuspensionPhysicsControl> wheels;

    bool operator!=(const SuspensionPhysicsControl &rhs) const {
      return wheels != rhs.wheels;
    }

    bool operator==(const SuspensionPhysicsControl &rhs) const {
      return !(*this != rhs);
    }
#ifdef LIBCARLA_INCLUDED_FROM_UE4

    SuspensionPhysicsControl(const FSuspensionPhysicsControl &Control) {
      wheels = std::vector<WheelSuspensionPhysicsControl>();
      for (const auto &Wheel : Control.Wheels) {
        wheels.push_back(WheelSuspensionPhysicsControl(Wheel));
      }
    }

    operator FSuspensionPhysicsControl() const {
      FSuspensionPhysicsControl Control;

      TArray<FWheelSuspensionPhysicsControl> Wheels;
      for (const auto &wheel : wheels) {
        Wheels.Add(FWheelSuspensionPhysicsControl(wheel));
      }
      Control.Wheels = Wheels;

      return Control;
    }

#endif

    MSGPACK_DEFINE_ARRAY(wheels)
  };

}
}
