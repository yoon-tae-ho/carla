// Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include "carla/MsgPack.h"
#include "carla/rpc/String.h"
#ifdef LIBCARLA_INCLUDED_FROM_UE4
#include <compiler/enable-ue4-macros.h>
#include "Vehicle/SuspensionState.h"
#include <compiler/disable-ue4-macros.h>
#endif

#include <cstdint>
#include <string>

namespace carla {
namespace rpc {

  class WheelSuspensionState {
  public:

    WheelSuspensionState() = default;

    int32_t wheel_index_raw = -1;
    std::string wheel_name_canonical;
    float suspension_travel_m = 0.0f;
    float raw_suspension_offset_m = 0.0f;
    float suspension_compression_m = 0.0f;
    float suspension_velocity_mps = 0.0f;
    float normalized_travel = 0.0f;
    bool contact_valid = false;
    bool wheel_in_air = false;
    bool velocity_valid = false;
    bool normalized_tire_load_valid = false;
    float normalized_tire_load = 0.0f;
    bool field_valid = false;

    bool operator!=(const WheelSuspensionState &rhs) const {
      return
        wheel_index_raw != rhs.wheel_index_raw ||
        wheel_name_canonical != rhs.wheel_name_canonical ||
        suspension_travel_m != rhs.suspension_travel_m ||
        raw_suspension_offset_m != rhs.raw_suspension_offset_m ||
        suspension_compression_m != rhs.suspension_compression_m ||
        suspension_velocity_mps != rhs.suspension_velocity_mps ||
        normalized_travel != rhs.normalized_travel ||
        contact_valid != rhs.contact_valid ||
        wheel_in_air != rhs.wheel_in_air ||
        velocity_valid != rhs.velocity_valid ||
        normalized_tire_load_valid != rhs.normalized_tire_load_valid ||
        normalized_tire_load != rhs.normalized_tire_load ||
        field_valid != rhs.field_valid;
    }

    bool operator==(const WheelSuspensionState &rhs) const {
      return !(*this != rhs);
    }

#ifdef LIBCARLA_INCLUDED_FROM_UE4

    WheelSuspensionState(const FWheelSuspensionState &Wheel)
      : wheel_index_raw(Wheel.WheelIndexRaw),
        wheel_name_canonical(FromFString(Wheel.WheelNameCanonical)),
        suspension_travel_m(Wheel.SuspensionTravelM),
        raw_suspension_offset_m(Wheel.RawSuspensionOffsetM),
        suspension_compression_m(Wheel.SuspensionCompressionM),
        suspension_velocity_mps(Wheel.SuspensionVelocityMps),
        normalized_travel(Wheel.NormalizedTravel),
        contact_valid(Wheel.bContactValid),
        wheel_in_air(Wheel.bWheelInAir),
        velocity_valid(Wheel.bVelocityValid),
        normalized_tire_load_valid(Wheel.bNormalizedTireLoadValid),
        normalized_tire_load(Wheel.NormalizedTireLoad),
        field_valid(Wheel.bFieldValid) {}

#endif

    MSGPACK_DEFINE_ARRAY(wheel_index_raw,
        wheel_name_canonical,
        suspension_travel_m,
        raw_suspension_offset_m,
        suspension_compression_m,
        suspension_velocity_mps,
        normalized_travel,
        contact_valid,
        wheel_in_air,
        velocity_valid,
        normalized_tire_load_valid,
        normalized_tire_load,
        field_valid)
  };

}
}
