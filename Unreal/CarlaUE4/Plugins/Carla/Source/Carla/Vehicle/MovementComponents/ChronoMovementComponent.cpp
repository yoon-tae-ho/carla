// Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
// Copyright (c) 2019 Intel Corporation
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#include "ChronoMovementComponent.h"
#include "Carla/Vehicle/CarlaWheeledVehicle.h"
#include "Carla/Vehicle/MovementComponents/DefaultMovementComponent.h"
#include "Misc/Paths.h"

#include "compiler/disable-ue4-macros.h"
#include <carla/rpc/String.h>
#ifdef WITH_CHRONO
#include "chrono/physics/ChLinkTSDA.h"
#include "chrono_vehicle/utils/ChUtilsJSON.h"
#include "chrono_vehicle/wheeled_vehicle/suspension/ChDoubleWishbone.h"
#endif
#include "compiler/enable-ue4-macros.h"
#include "Carla/Util/RayTracer.h"


void UChronoMovementComponent::CreateChronoMovementComponent(
    ACarlaWheeledVehicle* Vehicle,
    uint64_t MaxSubsteps,
    float MaxSubstepDeltaTime,
    FString VehicleJSON,
    FString PowertrainJSON,
    FString TireJSON,
    FString BaseJSONPath)
{
  #ifdef WITH_CHRONO
  UChronoMovementComponent* ChronoMovementComponent = NewObject<UChronoMovementComponent>(Vehicle);
  if (!VehicleJSON.IsEmpty())
  {
    ChronoMovementComponent->VehicleJSON = VehicleJSON;
  }
  if (!PowertrainJSON.IsEmpty())
  {
    ChronoMovementComponent->PowertrainJSON = PowertrainJSON;
  }
  if (!TireJSON.IsEmpty())
  {
    ChronoMovementComponent->TireJSON = TireJSON;
  }
  if (!BaseJSONPath.IsEmpty())
  {
    ChronoMovementComponent->BaseJSONPath = BaseJSONPath;
  }
  ChronoMovementComponent->MaxSubsteps = MaxSubsteps;
  ChronoMovementComponent->MaxSubstepDeltaTime = MaxSubstepDeltaTime;
  Vehicle->SetCarlaMovementComponent(ChronoMovementComponent);
  ChronoMovementComponent->RegisterComponent();
  #else
  UE_LOG(LogCarla, Warning, TEXT("Error: Chrono is not enabled") );
  #endif
}

#ifdef WITH_CHRONO

class FChronoMutableTSDAForce : public chrono::ChLinkTSDA::ForceFunctor
{
public:
  enum class EMode
  {
    SpringStiffness,
    ShockDamping
  };

  FChronoMutableTSDAForce(EMode InMode, double InCoefficient)
    : Mode(InMode),
      Coefficient(InCoefficient) {}

  void SetCoefficient(double InCoefficient)
  {
    Coefficient = InCoefficient;
  }

  virtual double operator()(
      double time,
      double rest_length,
      double length,
      double vel,
      chrono::ChLinkTSDA* link) override
  {
    if (Mode == EMode::SpringStiffness)
    {
      return -Coefficient * (length - rest_length);
    }
    return -Coefficient * vel;
  }

private:
  EMode Mode;
  double Coefficient;
};

using namespace chrono;
using namespace chrono::vehicle;

namespace {

constexpr int32 ChronoSuspensionValueCount = 4;

struct FChronoSuspensionCorner
{
  int32 AxleIndex;
  VehicleSide Side;
  const TCHAR* Label;
};

const FChronoSuspensionCorner ChronoSuspensionCorners[ChronoSuspensionValueCount] =
{
  {0, LEFT, TEXT("FL")},
  {0, RIGHT, TEXT("FR")},
  {1, LEFT, TEXT("RL")},
  {1, RIGHT, TEXT("RR")}
};

} // namespace

constexpr double CMTOM = 0.01;
ChVector<> UE4LocationToChrono(const FVector& Location)
{
  return CMTOM*ChVector<>(Location.X, -Location.Y, Location.Z);
}
constexpr double MTOCM = 100;
FVector ChronoToUE4Location(const ChVector<>& position)
{
  return MTOCM*FVector(position.x(), -position.y(), position.z());
}
ChVector<> UE4DirectionToChrono(const FVector& Location)
{
  return ChVector<>(Location.X, -Location.Y, Location.Z);
}
FVector ChronoToUE4Direction(const ChVector<>& position)
{
  return FVector(position.x(), -position.y(), position.z());
}
ChQuaternion<> UE4QuatToChrono(const FQuat& Quat)
{
  return ChQuaternion<>(Quat.W, -Quat.X, Quat.Y, -Quat.Z);
}
FQuat ChronoToUE4Quat(const ChQuaternion<>& quat)
{
  return FQuat(-quat.e1(), quat.e2(), -quat.e3(), quat.e0());
}

UERayCastTerrain::UERayCastTerrain(
    ACarlaWheeledVehicle* UEVehicle,
    chrono::vehicle::ChVehicle* ChrVehicle)
    : CarlaVehicle(UEVehicle), ChronoVehicle(ChrVehicle) {}

std::pair<bool, FHitResult>
    UERayCastTerrain::GetTerrainProperties(const FVector &Location) const
{
  const double MaxDistance = 1000000;
  FVector StartLocation = Location;
  FVector EndLocation = Location + FVector(0,0,-1)*MaxDistance; // search downwards
  FHitResult Hit;
  FCollisionQueryParams CollisionQueryParams;
  CollisionQueryParams.AddIgnoredActor(CarlaVehicle);
  bool bDidHit = CarlaVehicle->GetWorld()->LineTraceSingleByChannel(
      Hit,
      StartLocation,
      EndLocation,
      ECC_GameTraceChannel2, // camera (any collision)
      CollisionQueryParams,
      FCollisionResponseParams()
  );
  return std::make_pair(bDidHit, Hit);
}

double UERayCastTerrain::GetHeight(const ChVector<>& loc) const
{
  FVector Location = ChronoToUE4Location(loc + ChVector<>(0,0,0.5)); // small offset to detect the ground properly
  auto point_pair = GetTerrainProperties(Location);
  if (point_pair.first)
  {
    double Height = CMTOM*static_cast<double>(point_pair.second.Location.Z);
    return Height;
  }
  return -1000000.0;
}
ChVector<> UERayCastTerrain::GetNormal(const ChVector<>& loc) const
{
  FVector Location = ChronoToUE4Location(loc);
  auto point_pair = GetTerrainProperties(Location);
  if (point_pair.first)
  {
    FVector Normal = point_pair.second.Normal;
    auto ChronoNormal = UE4DirectionToChrono(Normal);
    return ChronoNormal;
  }
  return UE4DirectionToChrono(FVector(0,0,1));
}
float UERayCastTerrain::GetCoefficientFriction(const ChVector<>& loc) const
{
  return 1;
}

void UChronoMovementComponent::BeginPlay()
{
  Super::BeginPlay();

  DisableUE4VehiclePhysics();

  // // // Chrono System
  Sys.Set_G_acc(ChVector<>(0, 0, -9.81));
  Sys.SetSolverType(ChSolver::Type::BARZILAIBORWEIN);
  Sys.SetSolverMaxIterations(150);
  Sys.SetMaxPenetrationRecoverySpeed(4.0);

  if (!InitializeChronoVehicle())
  {
    UE_LOG(LogCarla, Error, TEXT(
        "Error: Failed to initialize Chrono vehicle. Disabling chrono physics..."));
    UDefaultMovementComponent::CreateDefaultMovementComponent(CarlaVehicle);
    return;
  }

  // Create the terrain
  Terrain = chrono_types::make_shared<UERayCastTerrain>(CarlaVehicle, Vehicle.get());

  CarlaVehicle->OnActorHit.AddDynamic(
      this, &UChronoMovementComponent::OnVehicleHit);
  CarlaVehicle->GetMesh()->OnComponentBeginOverlap.AddDynamic(
      this, &UChronoMovementComponent::OnVehicleOverlap);
  CarlaVehicle->GetMesh()->SetCollisionResponseToChannel(
      ECollisionChannel::ECC_WorldStatic, ECollisionResponse::ECR_Overlap);
}

bool UChronoMovementComponent::InitializeChronoVehicle()
{
  // Initial location with small offset to prevent falling through the ground
  FVector VehicleLocation = CarlaVehicle->GetActorLocation() + FVector(0,0,25);
  FQuat VehicleRotation = CarlaVehicle->GetActorRotation().Quaternion();
  auto ChronoLocation = UE4LocationToChrono(VehicleLocation);
  auto ChronoRotation = UE4QuatToChrono(VehicleRotation);

  // Set base path for vehicle JSON files
  vehicle::SetDataPath(carla::rpc::FromFString(BaseJSONPath));

  std::string BasePath_string = carla::rpc::FromFString(BaseJSONPath);

  // Create full path for json files
  // Do NOT use vehicle::GetDataFile() as strings from chrono lib
  // messes with unreal's std lib
  std::string VehicleJSON_string = carla::rpc::FromFString(VehicleJSON);
  std::string VehiclePath_string = BasePath_string + VehicleJSON_string;
  FString VehicleJSONPath = carla::rpc::ToFString(VehiclePath_string);

  std::string PowerTrainJSON_string = carla::rpc::FromFString(PowertrainJSON);
  std::string PowerTrain_string = BasePath_string + PowerTrainJSON_string;
  FString PowerTrainJSONPath = carla::rpc::ToFString(PowerTrain_string);

  std::string TireJSON_string = carla::rpc::FromFString(TireJSON);
  std::string Tire_string = BasePath_string + TireJSON_string;
  FString TireJSONPath = carla::rpc::ToFString(Tire_string);

  UE_LOG(LogCarla, Log, TEXT("Loading Chrono files: Vehicle: %s, PowerTrain: %s, Tire: %s"),
      *VehicleJSONPath,
      *PowerTrainJSONPath,
      *TireJSONPath);

  if (!FPaths::FileExists(VehicleJSONPath))
  {
    UE_LOG(LogCarla, Error, TEXT("Error: Chrono vehicle JSON file does not exist: %s"), *VehicleJSONPath);
    return false;
  }
  if (!FPaths::FileExists(PowerTrainJSONPath))
  {
    UE_LOG(LogCarla, Error, TEXT("Error: Chrono powertrain JSON file does not exist: %s"), *PowerTrainJSONPath);
    return false;
  }
  if (!FPaths::FileExists(TireJSONPath))
  {
    UE_LOG(LogCarla, Error, TEXT("Error: Chrono tire JSON file does not exist: %s"), *TireJSONPath);
    return false;
  }

  // Create JSON vehicle
  ResetChronoSuspensionForceFunctors();
  Vehicle = chrono_types::make_shared<WheeledVehicle>(
      &Sys,
      VehiclePath_string);
  Vehicle->Initialize(ChCoordsys<>(ChronoLocation, ChronoRotation));
  Vehicle->GetChassis()->SetFixed(false);
  // Create and initialize the powertrain System
  auto powertrain = ReadPowertrainJSON(
      PowerTrain_string);
  Vehicle->InitializePowertrain(powertrain);
  // Create and initialize the tires
  for (auto& axle : Vehicle->GetAxles()) {
      for (auto& wheel : axle->GetWheels()) {
          auto tire = ReadTireJSON(Tire_string);
          Vehicle->InitializeTire(tire, wheel, VisualizationType::MESH);
      }
  }

  return true;
}

bool UChronoMovementComponent::SetChronoSuspensionDamping(const TArray<float>& Damping)
{
  return ApplyChronoSuspensionValues(Damping, false);
}

bool UChronoMovementComponent::SetChronoSuspensionStiffness(const TArray<float>& Stiffness)
{
  return ApplyChronoSuspensionValues(Stiffness, true);
}

std::shared_ptr<ChDoubleWishbone>
    UChronoMovementComponent::GetChronoDoubleWishboneSuspension(int32 AxleIndex) const
{
  if (!Vehicle)
  {
    UE_LOG(LogCarla, Warning, TEXT("Chrono suspension control requested before vehicle initialization."));
    return nullptr;
  }

  const auto& Axles = Vehicle->GetAxles();
  if (AxleIndex < 0 || static_cast<size_t>(AxleIndex) >= Axles.size())
  {
    UE_LOG(LogCarla, Warning, TEXT("Chrono suspension axle index %d is unavailable."), AxleIndex);
    return nullptr;
  }

  auto Suspension = Vehicle->GetSuspension(AxleIndex);
  auto DoubleWishbone = std::dynamic_pointer_cast<ChDoubleWishbone>(Suspension);
  if (!DoubleWishbone)
  {
    UE_LOG(LogCarla, Warning, TEXT("Chrono suspension axle %d is not a DoubleWishbone suspension."), AxleIndex);
    return nullptr;
  }

  return DoubleWishbone;
}

bool UChronoMovementComponent::ApplyChronoSuspensionValues(
    const TArray<float>& Values,
    bool bUseSpring)
{
  const TCHAR* ValueName = bUseSpring ? TEXT("stiffness") : TEXT("damping");
  if (Values.Num() != ChronoSuspensionValueCount)
  {
    UE_LOG(LogCarla, Warning, TEXT("Chrono suspension %s expects %d values in FL/FR/RL/RR order."),
        ValueName,
        ChronoSuspensionValueCount);
    return false;
  }

  for (int32 Index = 0; Index < ChronoSuspensionValueCount; ++Index)
  {
    if (!FMath::IsFinite(Values[Index]) || Values[Index] < 0.0f)
    {
      UE_LOG(LogCarla, Warning, TEXT("Invalid Chrono suspension %s value for %s: %f."),
          ValueName,
          ChronoSuspensionCorners[Index].Label,
          Values[Index]);
      return false;
    }
  }

  std::array<std::shared_ptr<ChDoubleWishbone>, 2> Suspensions =
  {
    GetChronoDoubleWishboneSuspension(0),
    GetChronoDoubleWishboneSuspension(1)
  };

  if (!Suspensions[0] || !Suspensions[1])
  {
    return false;
  }

  std::array<std::shared_ptr<ChLinkTSDA>, ChronoSuspensionValueCount> Links;
  for (int32 Index = 0; Index < ChronoSuspensionValueCount; ++Index)
  {
    const auto& Corner = ChronoSuspensionCorners[Index];
    Links[Index] = bUseSpring ?
        Suspensions[Corner.AxleIndex]->GetSpring(Corner.Side) :
        Suspensions[Corner.AxleIndex]->GetShock(Corner.Side);
    if (!Links[Index])
    {
      UE_LOG(LogCarla, Warning, TEXT("Chrono suspension %s link is unavailable for %s."),
          bUseSpring ? TEXT("spring") : TEXT("shock"),
          Corner.Label);
      return false;
    }
  }

  for (int32 Index = 0; Index < ChronoSuspensionValueCount; ++Index)
  {
    auto& Functor = bUseSpring ?
        ChronoSpringForceFunctors[Index] :
        ChronoDamperForceFunctors[Index];
    if (!Functor)
    {
      Functor = std::make_shared<FChronoMutableTSDAForce>(
          bUseSpring ?
              FChronoMutableTSDAForce::EMode::SpringStiffness :
              FChronoMutableTSDAForce::EMode::ShockDamping,
          Values[Index]);
    }
    else
    {
      Functor->SetCoefficient(Values[Index]);
    }
    Links[Index]->RegisterForceFunctor(Functor);
  }

  return true;
}

void UChronoMovementComponent::ResetChronoSuspensionForceFunctors()
{
  for (auto& Functor : ChronoSpringForceFunctors)
  {
    Functor.reset();
  }
  for (auto& Functor : ChronoDamperForceFunctors)
  {
    Functor.reset();
  }
}

void UChronoMovementComponent::ProcessControl(FVehicleControl &Control)
{
  VehicleControl = Control;
  auto PowerTrain = Vehicle->GetPowertrain();
  if (PowerTrain)
  {
    if (VehicleControl.bReverse)
    {
      PowerTrain->SetDriveMode(ChPowertrain::DriveMode::REVERSE);
    }
    else
    {
      PowerTrain->SetDriveMode(ChPowertrain::DriveMode::FORWARD);
    }
  }
}

void UChronoMovementComponent::TickComponent(float DeltaTime,
      ELevelTick TickType,
      FActorComponentTickFunction* ThisTickFunction)
{
  TRACE_CPUPROFILER_EVENT_SCOPE(UChronoMovementComponent::TickComponent);
  if (DeltaTime > MaxSubstepDeltaTime)
  {
    uint64_t NumberSubSteps =
        FGenericPlatformMath::FloorToInt(DeltaTime/MaxSubstepDeltaTime);
    if (NumberSubSteps < MaxSubsteps)
    {
      for (uint64_t i = 0; i < NumberSubSteps; ++i)
      {
        AdvanceChronoSimulation(MaxSubstepDeltaTime);
      }
      float RemainingTime = DeltaTime - NumberSubSteps*MaxSubstepDeltaTime;
      if (RemainingTime > 0)
      {
        AdvanceChronoSimulation(RemainingTime);
      }
    }
    else
    {
      double SubDelta = DeltaTime / MaxSubsteps;
      for (uint64_t i = 0; i < MaxSubsteps; ++i)
      {
        AdvanceChronoSimulation(SubDelta);
      }
    }
  }
  else
  {
    AdvanceChronoSimulation(DeltaTime);
  }

  const auto ChronoPositionOffset = ChVector<>(0,0,-0.25f);
  auto VehiclePos = Vehicle->GetVehiclePos() + ChronoPositionOffset;
  auto VehicleRot = Vehicle->GetVehicleRot();
  double Time = Vehicle->GetSystem()->GetChTime();

  FVector NewLocation = ChronoToUE4Location(VehiclePos);
  FQuat NewRotation = ChronoToUE4Quat(VehicleRot);
  if(NewLocation.ContainsNaN() || NewRotation.ContainsNaN())
  {
    UE_LOG(LogCarla, Warning, TEXT(
        "Error: Chrono vehicle position or rotation contains NaN. Disabling chrono physics..."));
    UDefaultMovementComponent::CreateDefaultMovementComponent(CarlaVehicle);
    return;
  }
  CarlaVehicle->SetActorLocation(NewLocation);
  FRotator NewRotator = NewRotation.Rotator();
  // adding small rotation to compensate chrono offset
  const float ChronoPitchOffset = 2.5f;
  NewRotator.Add(ChronoPitchOffset, 0.f, 0.f); 
  CarlaVehicle->SetActorRotation(NewRotator);
}

void UChronoMovementComponent::AdvanceChronoSimulation(float StepSize)
{
  double Time = Vehicle->GetSystem()->GetChTime();
  double Throttle = VehicleControl.Throttle;
  double Steering = -VehicleControl.Steer; // RHF to LHF
  double Brake = VehicleControl.Brake + VehicleControl.bHandBrake;
  Vehicle->Synchronize(Time, {Steering, Throttle, Brake}, *Terrain.get());
  Vehicle->Advance(StepSize);
  Sys.DoStepDynamics(StepSize);
}

FVector UChronoMovementComponent::GetVelocity() const
{
  if (Vehicle)
  {
    return ChronoToUE4Location(
        Vehicle->GetVehiclePointVelocity(ChVector<>(0,0,0)));
  }
  return FVector();
}

int32 UChronoMovementComponent::GetVehicleCurrentGear() const
{
  if (Vehicle)
  {
    auto PowerTrain = Vehicle->GetPowertrain();
    if (PowerTrain)
    {
      return PowerTrain->GetCurrentTransmissionGear();
    }
  }
  return 0;
}

float UChronoMovementComponent::GetVehicleForwardSpeed() const
{
  if (Vehicle)
  {
    return GetVelocity().X;
  }
  return 0.f;
}

void UChronoMovementComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
  if(!CarlaVehicle)
  {
    return;
  }
  // reset callbacks to react to collisions
  CarlaVehicle->OnActorHit.RemoveDynamic(
      this, &UChronoMovementComponent::OnVehicleHit);
  CarlaVehicle->GetMesh()->OnComponentBeginOverlap.RemoveDynamic(
      this, &UChronoMovementComponent::OnVehicleOverlap);
  CarlaVehicle->GetMesh()->SetCollisionResponseToChannel(
      ECollisionChannel::ECC_WorldStatic, ECollisionResponse::ECR_Block);
}
#else
bool UChronoMovementComponent::SetChronoSuspensionDamping(const TArray<float>& Damping)
{
  UE_LOG(LogCarla, Warning, TEXT("Chrono suspension damping control requested, but Chrono is not enabled."));
  return false;
}

bool UChronoMovementComponent::SetChronoSuspensionStiffness(const TArray<float>& Stiffness)
{
  UE_LOG(LogCarla, Warning, TEXT("Chrono suspension stiffness control requested, but Chrono is not enabled."));
  return false;
}
#endif

void UChronoMovementComponent::DisableChronoPhysics()
{
  this->SetComponentTickEnabled(false);
  EnableUE4VehiclePhysics(true);
  CarlaVehicle->OnActorHit.RemoveDynamic(this, &UChronoMovementComponent::OnVehicleHit);
  CarlaVehicle->GetMesh()->OnComponentBeginOverlap.RemoveDynamic(
      this, &UChronoMovementComponent::OnVehicleOverlap);
  CarlaVehicle->GetMesh()->SetCollisionResponseToChannel(
      ECollisionChannel::ECC_WorldStatic, ECollisionResponse::ECR_Block);
  UDefaultMovementComponent::CreateDefaultMovementComponent(CarlaVehicle);
  carla::log_warning("Chrono physics does not support collisions yet, reverting to default PhysX physics.");
}

void UChronoMovementComponent::OnVehicleHit(AActor *Actor,
    AActor *OtherActor,
    FVector NormalImpulse,
    const FHitResult &Hit)
{
  DisableChronoPhysics();
}

// On car mesh overlap, only works when carsim is enabled
// (this event triggers when overlapping with static environment)
void UChronoMovementComponent::OnVehicleOverlap(
    UPrimitiveComponent* OverlappedComponent,
    AActor* OtherActor,
    UPrimitiveComponent* OtherComp,
    int32 OtherBodyIndex,
    bool bFromSweep,
    const FHitResult & SweepResult)
{
  if (OtherComp->GetCollisionResponseToChannel(
      ECollisionChannel::ECC_WorldDynamic) ==
      ECollisionResponse::ECR_Block)
  {
    DisableChronoPhysics();
  }
}
