// Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include "CoreMinimal.h"

#include "SuspensionState.generated.h"

USTRUCT(BlueprintType)
struct CARLA_API FWheelSuspensionState
{
  GENERATED_BODY()

  UPROPERTY(Category = "Suspension State", EditAnywhere, BlueprintReadWrite)
  int32 WheelIndexRaw = -1;

  UPROPERTY(Category = "Suspension State", EditAnywhere, BlueprintReadWrite)
  FString WheelNameCanonical;

  UPROPERTY(Category = "Suspension State", EditAnywhere, BlueprintReadWrite)
  float SuspensionTravelM = 0.0f;

  UPROPERTY(Category = "Suspension State", EditAnywhere, BlueprintReadWrite)
  float RawSuspensionOffsetM = 0.0f;

  UPROPERTY(Category = "Suspension State", EditAnywhere, BlueprintReadWrite)
  float SuspensionCompressionM = 0.0f;

  UPROPERTY(Category = "Suspension State", EditAnywhere, BlueprintReadWrite)
  float SuspensionVelocityMps = 0.0f;

  UPROPERTY(Category = "Suspension State", EditAnywhere, BlueprintReadWrite)
  float NormalizedTravel = 0.0f;

  UPROPERTY(Category = "Suspension State", EditAnywhere, BlueprintReadWrite)
  bool bContactValid = false;

  UPROPERTY(Category = "Suspension State", EditAnywhere, BlueprintReadWrite)
  bool bWheelInAir = false;

  UPROPERTY(Category = "Suspension State", EditAnywhere, BlueprintReadWrite)
  bool bVelocityValid = false;

  UPROPERTY(Category = "Suspension State", EditAnywhere, BlueprintReadWrite)
  bool bNormalizedTireLoadValid = false;

  UPROPERTY(Category = "Suspension State", EditAnywhere, BlueprintReadWrite)
  float NormalizedTireLoad = 0.0f;

  UPROPERTY(Category = "Suspension State", EditAnywhere, BlueprintReadWrite)
  bool bFieldValid = false;
};

USTRUCT(BlueprintType)
struct CARLA_API FSuspensionState
{
  GENERATED_BODY()

  UPROPERTY(Category = "Suspension State", EditAnywhere, BlueprintReadWrite)
  int64 Frame = 0;

  UPROPERTY(Category = "Suspension State", EditAnywhere, BlueprintReadWrite)
  float Timestamp = 0.0f;

  UPROPERTY(Category = "Suspension State", EditAnywhere, BlueprintReadWrite)
  int32 ActorId = 0;

  UPROPERTY(Category = "Suspension State", EditAnywhere, BlueprintReadWrite)
  int32 WheelCount = 0;

  UPROPERTY(Category = "Suspension State", EditAnywhere, BlueprintReadWrite)
  FString StateSource;

  UPROPERTY(Category = "Suspension State", EditAnywhere, BlueprintReadWrite)
  FString FailureReason;

  UPROPERTY(Category = "Suspension State", EditAnywhere, BlueprintReadWrite)
  bool bStateValid = false;

  UPROPERTY(Category = "Suspension State", EditAnywhere, BlueprintReadWrite)
  bool bVelocityValid = false;

  UPROPERTY(Category = "Suspension State", EditAnywhere, BlueprintReadWrite)
  bool bCompressionConventionValidated = false;

  UPROPERTY(Category = "Suspension State", EditAnywhere, BlueprintReadWrite)
  TArray<FWheelSuspensionState> Wheels;
};
