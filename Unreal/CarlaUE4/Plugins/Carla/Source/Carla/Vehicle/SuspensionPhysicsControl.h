// Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include "CoreMinimal.h"

#include "SuspensionPhysicsControl.generated.h"

USTRUCT(BlueprintType)
struct CARLA_API FWheelSuspensionPhysicsControl
{
  GENERATED_BODY()

  UPROPERTY(Category = "Suspension Physics Control", EditAnywhere, BlueprintReadWrite)
  float SpringStrength = 0.0f;

  UPROPERTY(Category = "Suspension Physics Control", EditAnywhere, BlueprintReadWrite)
  float SpringDamperRate = 0.0f;

  UPROPERTY(Category = "Suspension Physics Control", EditAnywhere, BlueprintReadWrite)
  float MaxCompression = 0.0f;

  UPROPERTY(Category = "Suspension Physics Control", EditAnywhere, BlueprintReadWrite)
  float MaxDroop = 0.0f;

  UPROPERTY(Category = "Suspension Physics Control", EditAnywhere, BlueprintReadWrite)
  float SprungMass = 0.0f;
};

USTRUCT(BlueprintType)
struct CARLA_API FSuspensionPhysicsControl
{
  GENERATED_BODY()

  // Wheel order: FL, FR, RL/BL, RR/BR. Values are raw PhysX/CARLA internals.
  UPROPERTY(Category = "Suspension Physics Control", EditAnywhere, BlueprintReadWrite)
  TArray<FWheelSuspensionPhysicsControl> Wheels;
};
