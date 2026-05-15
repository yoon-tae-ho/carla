# Phase 3-C - Apply Baseline Suspension Command Every Tick

## 목표

Chrono physics가 enabled 된 ego vehicle에 매 tick baseline suspension command를 적용한다.

이 단계에서는 planning 기반 제어를 하지 않는다. 목적은 TransFuser++ `run_step()` 루프 안에서 `apply_chrono_suspension_control(...)` 호출이 안정적으로 수행되는지 검증하는 것이다.

## 선행 조건

`02_chrono_enable_only.md`가 통과되어야 한다.

## 환경변수

```bash
TFPP_CHRONO_ENABLE=1
TFPP_CHRONO_BASELINE_CONTROL=1
TFPP_CHRONO_LOG_DIR=$HOME/sim/e2e_models/logs/transfuserpp_chrono
```

`TFPP_CHRONO_BASELINE_CONTROL=1`일 때만 suspension command를 적용한다.

## baseline 값

기존 probe에서 사용한 baseline 계수를 우선 사용한다.

```text
damping_FL = 3500.0
stiffness_FL = 70000.0
```

4개 wheel에 같은 baseline을 적용할지, 기존 API가 요구하는 vector shape에 맞춰 적용할지는 `chrono_suspension_probe.py`의 command 구조를 따른다.

중요:

- 이 setter는 damping ratio가 아니라 damper coefficient와 spring stiffness coefficient를 설정한다.
- desired force를 입력하는 API가 아니다.

## 구현 요구사항

`chrono_suspension_bridge.py`에 baseline command 생성 함수를 둔다.

```python
def baseline_command():
    ...
```

`SensorAgent.run_step()`에서 `control`을 만든 뒤, return 전에 적용한다.

권장 위치:

```python
control = carla.VehicleControl(...)
bridge.apply_suspension_if_needed(...)
return control
```

적용 실패가 route 전체를 죽이면 안 된다. 실패는 로그에 남기고 계속 진행한다.

단, 연속 실패 횟수는 기록한다.

## 로그 필드 추가

tick record에 아래를 추가한다.

```text
suspension_mode = "baseline"
suspension_applied = true/false
suspension_error = null or string
damping_command
stiffness_command
apply_failure_count
roll
pitch
speed
```

## 테스트 명령

터미널 1: CARLA Editor safe mode 실행 후 Play.

터미널 2:

```bash
conda activate garage_2
source ~/sim/e2e_models/scripts/env_garage_2.sh

TFPP_CHRONO_LOG_ONLY=1 \
TFPP_CHRONO_ENABLE=1 \
TFPP_CHRONO_BASELINE_CONTROL=1 \
STOP_AFTER_METER=30 \
TEAM_CONFIG=$TFPP_CKPT_ROOT/pretrained_models/all_towns \
~/sim/e2e_models/scripts/run_tfpp_debug_route.sh
```

짧은 test가 통과하면 `STOP_AFTER_METER` 없이 debug route 전체를 한 번 실행한다.

## 통과 기준

- `apply_chrono_suspension_control(...)` 호출이 tick마다 시도된다.
- failures가 0이거나 명확히 설명 가능한 수준이다.
- 차량이 ground penetration, pitch/roll 폭주, route 시작 즉시 crash를 보이지 않는다.
- TransFuser++ route가 최소 30m 이상 진행된다.
- jsonl 로그에서 baseline command 값이 확인된다.

## 실패 시 중단 조건

- `Chrono physics is not enabled for this vehicle`
- `apply_chrono_suspension_control` attribute 없음
- 차량 자세가 급격히 불안정해짐
- route 시작 직후 서버 crash

## Codex 최종 요약 요구

- baseline command 구조
- 적용 위치
- apply success/failure 통계
- 로그 예시
- 다음 단계 가능 여부

