# TransFuser++ Chrono Suspension Integration - Codex Execution Plan

이 문서 묶음은 TransFuser++가 CARLA에서 closed-loop로 주행하는 상태를 기반으로, Chrono suspension control을 단계적으로 붙이기 위한 Codex 작업 지시서이다.

각 단계는 독립적인 markdown 파일이다. 한 번에 전부 구현하지 말고, 반드시 `01 -> 02 -> 03 -> 04 -> 05` 순서로 진행한다. 각 단계에서 테스트가 통과한 뒤 다음 단계로 넘어간다.

## 현재 기준

이미 확인된 사실:

- CARLA custom PythonAPI에는 `carla.Vehicle.apply_chrono_suspension_control(...)`가 있다.
- Chrono probe에서 안정적인 조합은 다음이다.
  - CARLA blueprint: `vehicle.lincoln.mkz_2020`
  - Chrono vehicle: `sedan/vehicle/Sedan_Vehicle.json`
  - Chrono powertrain: `sedan/powertrain/Sedan_SimpleMapPowertrain.json`
  - Chrono tire: `sedan/tire/Sedan_TMeasyTire.json`
- TransFuser++ debug route는 Editor safe mode에서 closed-loop 완료가 가능하다.
- packaged Shipping 서버는 Vulkan sensor rendering crash가 반복되므로, 지금 단계에서는 사용하지 않는다.

## 공통 실행 조건

CARLA 서버는 Editor safe mode를 사용한다.

터미널 1:

```bash
conda activate carla-0.9.15
source ~/sim/e2e_models/scripts/env_carla_0915.sh
~/sim/e2e_models/scripts/launch_carla_editor_tfpp_safe.sh
```

Editor가 뜨면 Play를 누른다.

터미널 2:

```bash
conda activate garage_2
source ~/sim/e2e_models/scripts/env_garage_2.sh
python ~/sim/e2e_models/scripts/check_carla_server.py
```

## 공통 수정 원칙

- 기본 TransFuser++ 동작은 깨지지 않아야 한다.
- 모든 Chrono/suspension 기능은 환경변수로 켰을 때만 동작해야 한다.
- `sensor_agent.py`는 최소 hook만 넣고, 실제 로직은 새 helper 파일로 분리한다.
- C++ RPC, CARLA server, PythonAPI binding은 이 단계에서 수정하지 않는다.
- package 안정화는 별도 트랙이다. 이 문서 묶음에서는 Editor safe mode만 기준으로 한다.
- 실패 시 사용자 변경사항을 되돌리지 않는다.

## 권장 파일 구조

수정/추가 후보:

```text
~/sim/e2e_models/carla_garage/team_code/sensor_agent.py
~/sim/e2e_models/carla_garage/team_code/chrono_suspension_bridge.py
~/sim/e2e_models/carla_garage/team_code/analyze_tfpp_chrono_suspension.py
```

로그 위치:

```text
~/sim/e2e_models/logs/transfuserpp_chrono/
```

## 단계 순서

1. `01_log_only_planning_hook.md`
   - TransFuser++ planning/control output만 jsonl로 저장한다.
   - Chrono는 건드리지 않는다.

2. `02_chrono_enable_only.md`
   - leaderboard ego vehicle에 Chrono physics만 enable한다.
   - suspension command는 아직 적용하지 않는다.

3. `03_baseline_suspension_apply.md`
   - 매 tick 고정 baseline suspension command를 적용한다.
   - planning 기반 제어는 아직 하지 않는다.

4. `04_rule_based_suspension_controller.md`
   - TransFuser++ planning/control output 기반 rule controller를 연결한다.
   - baseline fallback을 반드시 유지한다.

5. `05_analysis_and_experiment_protocol.md`
   - 로그 분석 스크립트와 반복 실험 프로토콜을 정리한다.

## 단계별 성공 기준

각 단계는 최소한 아래를 만족해야 다음으로 넘어간다.

```text
1. TransFuser++ evaluator가 시작된다.
2. debug route가 crash 없이 진행된다.
3. 새 기능이 env var off 상태에서는 완전히 비활성화된다.
4. result.json 또는 jsonl 로그로 성공/실패 원인을 확인할 수 있다.
```

## 공통 TransFuser++ 실행 명령

각 단계별 env var를 붙여서 실행한다.

```bash
TEAM_CONFIG=$TFPP_CKPT_ROOT/pretrained_models/all_towns \
~/sim/e2e_models/scripts/run_tfpp_debug_route.sh
```

짧은 smoke test:

```bash
STOP_AFTER_METER=30 \
TEAM_CONFIG=$TFPP_CKPT_ROOT/pretrained_models/all_towns \
~/sim/e2e_models/scripts/run_tfpp_debug_route.sh
```

## Codex에게 지시할 때

각 단계 파일 전체를 Codex에게 주고 다음처럼 요청한다.

```text
이 markdown 파일의 지시를 따라 구현하고, 테스트 가능한 상태로 만들어줘.
수정한 파일, 실행 명령, 통과 기준, 남은 위험을 마지막에 요약해줘.
```

