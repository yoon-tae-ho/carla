# Phase 3-A - TransFuser++ Planning Output Log-Only Hook

## 목표

TransFuser++ 내부의 실제 planning/control output을 jsonl로 저장한다. 이 단계에서는 Chrono physics와 suspension command를 전혀 적용하지 않는다.

이 단계의 목적은 `sensor_agent.py`의 어느 지점에서 어떤 planner output을 안정적으로 얻을 수 있는지 확인하는 것이다.

## 수정 범위

주요 대상:

```text
~/sim/e2e_models/carla_garage/team_code/sensor_agent.py
```

새 파일 권장:

```text
~/sim/e2e_models/carla_garage/team_code/chrono_suspension_bridge.py
```

`sensor_agent.py`에는 최소 hook만 넣고, jsonl writer와 helper는 `chrono_suspension_bridge.py`에 둔다.

## 환경변수

이 단계의 기능은 아래 환경변수가 켜졌을 때만 활성화한다.

```bash
TFPP_CHRONO_LOG_ONLY=1
TFPP_CHRONO_LOG_DIR=$HOME/sim/e2e_models/logs/transfuserpp_chrono
```

환경변수가 없거나 `0`이면 기존 TransFuser++와 완전히 같아야 한다.

## 구현 요구사항

`SensorAgent.setup()` 또는 `_init()`에서 logger helper를 초기화한다.

`SensorAgent.run_step()`에서 아래 값들이 만들어진 뒤 로그를 남긴다.

후보 위치:

```python
pred_checkpoints = torch.stack(...).mean(...).detach().cpu().numpy()
steer, throttle, brake = ...
control = carla.VehicleControl(...)
```

로그 필드:

```text
kind = "tick"
step
timestamp
speed
route command
pred_target_speed_scalar
pred_checkpoints summary or list
steer
throttle
brake
control.steer
control.throttle
control.brake
```

가능하면 ego vehicle 상태도 기록한다.

```text
ego_location
ego_rotation
ego_velocity
```

단, ego handle 접근 실패가 route를 죽이면 안 된다. 실패 시 `ego_state_available=false`로 기록하고 계속 진행한다.

## 로그 형식

jsonl을 사용한다.

위치 예:

```text
~/sim/e2e_models/logs/transfuserpp_chrono/tfpp_chrono_log_YYYYMMDD_HHMMSS.jsonl
```

첫 줄에는 metadata를 기록한다.

```json
{"kind":"metadata","mode":"log_only","enabled":true}
```

각 tick은 한 줄 하나의 JSON 객체로 저장한다.

## 테스트 명령

터미널 1에서 CARLA Editor safe mode 실행 후 Play.

터미널 2:

```bash
conda activate garage_2
source ~/sim/e2e_models/scripts/env_garage_2.sh

TFPP_CHRONO_LOG_ONLY=1 \
STOP_AFTER_METER=30 \
TEAM_CONFIG=$TFPP_CKPT_ROOT/pretrained_models/all_towns \
~/sim/e2e_models/scripts/run_tfpp_debug_route.sh
```

로그 확인:

```bash
ls -lt ~/sim/e2e_models/logs/transfuserpp_chrono | head
tail -n 5 ~/sim/e2e_models/logs/transfuserpp_chrono/*.jsonl
```

## 통과 기준

- env var off 상태에서 기존 debug route가 그대로 실행된다.
- `TFPP_CHRONO_LOG_ONLY=1`에서 debug route가 crash 없이 진행된다.
- jsonl 파일이 생성된다.
- tick record에 `pred_target_speed_scalar`, `steer`, `throttle`, `brake`가 기록된다.
- Chrono enable 또는 suspension apply는 아직 호출하지 않는다.

## 실패 시 중단 조건

- `sensor_agent.py` import 실패
- model inference 실패
- logger 예외 때문에 route가 종료됨
- env var off 상태에서도 동작이 바뀜

## Codex 최종 요약 요구

작업 후 다음을 요약한다.

- 수정/추가한 파일
- log-only hook 위치
- 로그 파일 경로와 예시 record
- 실행 명령
- 통과/실패 결과

