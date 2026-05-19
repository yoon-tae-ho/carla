// Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include "carla/MsgPack.h"
#ifdef LIBCARLA_INCLUDED_FROM_UE4
#include <compiler/enable-ue4-macros.h>
#include "Vehicle/SuspensionPhysicsControl.h"
#include <compiler/disable-ue4-macros.h>
#endif

namespace carla {
namespace rpc {

  class WheelSuspensionPhysicsControl {
  public:

    WheelSuspensionPhysicsControl() = default;

    WheelSuspensionPhysicsControl(
        float in_spring_strength,
        float in_spring_damper_rate,
        float in_max_compression,
        float in_max_droop,
        float in_sprung_mass)
      : spring_strength(in_spring_strength),
        spring_damper_rate(in_spring_damper_rate),
        max_compression(in_max_compression),
        max_droop(in_max_droop),
        sprung_mass(in_sprung_mass) {}

    float spring_strength = 0.0f;
    float spring_damper_rate = 0.0f;
    float max_compression = 0.0f;
    float max_droop = 0.0f;
    float sprung_mass = 0.0f;

    bool operator!=(const WheelSuspensionPhysicsControl &rhs) const {
      return
        spring_strength != rhs.spring_strength ||
        spring_damper_rate != rhs.spring_damper_rate ||
        max_compression != rhs.max_compression ||
        max_droop != rhs.max_droop ||
        sprung_mass != rhs.sprung_mass;
    }

    bool operator==(const WheelSuspensionPhysicsControl &rhs) const {
      return !(*this != rhs);
    }
#ifdef LIBCARLA_INCLUDED_FROM_UE4

    WheelSuspensionPhysicsControl(const FWheelSuspensionPhysicsControl &Wheel)
      : spring_strength(Wheel.SpringStrength),
        spring_damper_rate(Wheel.SpringDamperRate),
        max_compression(Wheel.MaxCompression),
        max_droop(Wheel.MaxDroop),
        sprung_mass(Wheel.SprungMass) {}

    operator FWheelSuspensionPhysicsControl() const {
      FWheelSuspensionPhysicsControl Wheel;
      Wheel.SpringStrength = spring_strength;
      Wheel.SpringDamperRate = spring_damper_rate;
      Wheel.MaxCompression = max_compression;
      Wheel.MaxDroop = max_droop;
      Wheel.SprungMass = sprung_mass;
      return Wheel;
    }
#endif

    MSGPACK_DEFINE_ARRAY(spring_strength,
        spring_damper_rate,
        max_compression,
        max_droop,
        sprung_mass)
  };

}
}
