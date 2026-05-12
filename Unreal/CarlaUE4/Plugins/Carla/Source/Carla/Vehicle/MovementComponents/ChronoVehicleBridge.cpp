// Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#include "ChronoMovementComponent.h"
#include "Carla/Vehicle/CarlaWheeledVehicle.h"

void ACarlaWheeledVehicle::EnableChronoPhysics(
    uint64_t MaxSubsteps,
    float MaxSubstepDeltaTime,
    const FString& VehicleJSON,
    const FString& PowertrainJSON,
    const FString& TireJSON,
    const FString& BaseJSONPath)
{
  UChronoMovementComponent::CreateChronoMovementComponent(
      this,
      MaxSubsteps,
      MaxSubstepDeltaTime,
      VehicleJSON,
      PowertrainJSON,
      TireJSON,
      BaseJSONPath);
}

bool ACarlaWheeledVehicle::HasChronoMovementComponent() const
{
  return GetCarlaMovementComponent<UChronoMovementComponent>() != nullptr;
}

bool ACarlaWheeledVehicle::ApplyChronoSuspensionControl(
    const TArray<float>& Damping,
    const TArray<float>& Stiffness)
{
  auto* ChronoMovementComponent = GetCarlaMovementComponent<UChronoMovementComponent>();
  return ChronoMovementComponent != nullptr &&
      ChronoMovementComponent->SetChronoSuspensionDamping(Damping) &&
      ChronoMovementComponent->SetChronoSuspensionStiffness(Stiffness);
}
