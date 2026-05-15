# Phase 3-B - Enable Chrono Physics Only

## 목표

TransFuser++ leaderboard가 spawn한 ego vehicle에 Chrono physics만 enable한다. suspension command는 아직 적용하지 않는다.

이 단계는 `carla_garage` 실행 환경에서 custom CARLA PythonAPI와 Chrono vehicle data 경로가 제대로 동작하는지 검증한다.

## 선행 조건

`01_log_only_planning_hook.md`가 통과되어야 한다.

특히 아래가 확인되어야 한다.

```text
TransFuser++ debug route 실행 가능
jsonl tick 로그 생성 가능
env var off 상태에서 기본 동작 보존
```

## 수정 범위

주요 대상:

```text
~/sim/e2e_models/carla_garage/team_code/sensor_agent.py
~/sim/e2e_models/carla_garage/team_code/chrono_suspension_bridge.py
```

## 환경변수

```bash
TFPP_CHRONO_ENABLE=1
TFPP_CHRONO_LOG_DIR=$HOME/sim/e2e_models/logs/transfuserpp_chrono
TFPP_CHRONO_DATA_ROOT=$HOME/sim/carla-0.9.15/Build/chrono-install/share/chrono/data/vehicle
TFPP_CHRONO_VEHICLE_JSON=sedan/vehicle/Sedan_Vehicle.json
TFPP_CHRONO_POWERTRAIN_JSON=sedan/powertrain/Sedan_SimpleMapPowertrain.json
TFPP_CHRONO_TIRE_JSON=sedan/tire/Sedan_TMeasyTire.json
```

기능은 `TFPP_CHRONO_ENABLE=1`일 때만 활성화한다.

## 구현 요구사항

`SensorAgent._init()` 이후 ego vehicle을 가져온다.

권장 접근:

```python
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
ego_vehicle = CarlaDataProvider.get_hero_actor()
```

ego가 없으면 route를 죽이지 말고 로그에 남긴다.

Chrono enable 호출은 helper에 둔다.

```python
bridge.enable_chrono_if_needed(ego_vehicle)
```

기대 호출 형태는 기존 probe/manual control 코드와 맞춘다.

```python
ego_vehicle.enable_chrono_physics(
    max_substeps,
    max_substep_delta_time,
    vehicle_json,
    powertrain_json,
    tire_json,
    base_json_path
)
```

정확한 인자 순서는 현재 custom PythonAPI와 기존 `chrono_suspension_probe.py`를 읽고 맞춘다.

## 로그 필드 추가

metadata:

```json
{"kind":"chrono_enable","enabled":true,"success":true}
```

실패 시:

```json
{"kind":"chrono_enable","enabled":true,"success":false,"error":"..."}
```

tick 로그에는 최소 아래를 추가한다.

```text
chrono_enabled
chrono_enable_success
```

## 테스트 명령

터미널 1: CARLA Editor safe mode 실행 후 Play.

터미널 2:

```bash
conda activate garage_2
source ~/sim/e2e_models/scripts/env_garage_2.sh

python - <<'PY'
import carla
print("has suspension setter:", hasattr(carla.Vehicle, "apply_chrono_suspension_control"))
print("has chrono enable:", hasattr(carla.Vehicle, "enable_chrono_physics"))
PY
```

이후:

```bash
TFPP_CHRONO_LOG_ONLY=1 \
TFPP_CHRONO_ENABLE=1 \
STOP_AFTER_METER=30 \
TEAM_CONFIG=$TFPP_CKPT_ROOT/pretrained_models/all_towns \
~/sim/e2e_models/scripts/run_tfpp_debug_route.sh
```

## 통과 기준

- `has suspension setter: True`
- `has chrono enable: True`
- route가 시작되고 최소 30m smoke test가 crash 없이 진행된다.
- jsonl 로그에 Chrono enable 성공/실패 record가 남는다.
- suspension apply는 아직 호출하지 않는다.

## 실패 시 분석

아래 경우는 각각 구분해서 기록한다.

```text
custom PythonAPI가 잡히지 않음
ego vehicle handle을 못 가져옴
Chrono vehicle data 경로가 틀림
enable_chrono_physics 호출 signature 불일치
Chrono enable 이후 차량 자세/물리가 불안정
```

## Codex 최종 요약 요구

- ego vehicle handle을 얻은 방식
- Chrono enable 호출 인자
- 로그 record 예시
- 실행 결과
- 다음 단계로 넘어가도 되는지 판단

