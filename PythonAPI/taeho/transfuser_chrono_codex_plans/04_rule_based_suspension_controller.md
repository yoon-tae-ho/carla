# Phase 3-D - Rule-Based Suspension Controller From TransFuser++ Output

## 목표

TransFuser++ planning/control output을 이용해 real-time suspension command를 바꾼다.

이 단계는 최종 연구 controller가 아니라 integration skeleton이다. controller는 단순하고 해석 가능해야 하며, 언제든 baseline으로 fallback해야 한다.

## 선행 조건

`03_baseline_suspension_apply.md`가 통과되어야 한다.

## 환경변수

```bash
TFPP_CHRONO_ENABLE=1
TFPP_CHRONO_RULE_CONTROL=1
TFPP_CHRONO_LOG_DIR=$HOME/sim/e2e_models/logs/transfuserpp_chrono
```

`TFPP_CHRONO_RULE_CONTROL=1`일 때만 rule controller를 사용한다.

`TFPP_CHRONO_BASELINE_CONTROL=1`과 동시에 켜진 경우에는 rule controller가 우선한다. 단, rule 계산 실패 시 baseline으로 fallback한다.

## controller 입력

가능한 입력:

```text
speed
steer
throttle
brake
pred_target_speed_scalar
pred_checkpoints
pred_wp summary
```

최소 입력:

```text
speed
steer
brake
pred_target_speed_scalar
pred_checkpoints
```

## 초기 rule 설계

복잡한 최적제어 금지. 다음 정도의 rule로 시작한다.

```text
curvature_score = predicted checkpoints의 heading/curvature 요약값
brake_score = brake 또는 target speed 감소 정도
steer_score = abs(steer)
```

command mode:

```text
baseline:
  기본 상태

stiff:
  abs(steer)가 크거나 curvature_score가 큰 경우
  brake_score가 큰 경우

soft:
  저속, 직진, brake 없음, curvature 작음
```

값은 기존 probe의 soft/baseline/stiff command를 재사용한다.

```text
soft:     damping approx 2275, stiffness approx 52500
baseline: damping approx 3500, stiffness approx 70000
stiff:    damping approx 5075, stiffness approx 91000
```

정확한 vector structure는 `chrono_suspension_probe.py`를 따른다.

## 안전장치

반드시 구현한다.

```text
NaN/inf 입력이면 baseline fallback
speed가 비정상적으로 크거나 음수이면 baseline fallback
pred_checkpoints shape가 예상과 다르면 baseline fallback
연속 apply 실패가 많으면 baseline fallback 또는 apply 중지
```

controller는 차량을 teleport하거나 route/control을 바꾸면 안 된다. suspension command만 바꾼다.

## 로그 필드 추가

```text
suspension_mode = "rule_based"
controller_state = "soft|baseline|stiff|fallback"
curvature_score
steer_score
brake_score
fallback_reason
damping_command
stiffness_command
suspension_applied
```

## 테스트 명령

터미널 1: CARLA Editor safe mode 실행 후 Play.

터미널 2:

```bash
conda activate garage_2
source ~/sim/e2e_models/scripts/env_garage_2.sh

TFPP_CHRONO_LOG_ONLY=1 \
TFPP_CHRONO_ENABLE=1 \
TFPP_CHRONO_RULE_CONTROL=1 \
STOP_AFTER_METER=30 \
TEAM_CONFIG=$TFPP_CKPT_ROOT/pretrained_models/all_towns \
~/sim/e2e_models/scripts/run_tfpp_debug_route.sh
```

통과하면 전체 debug route:

```bash
TFPP_CHRONO_LOG_ONLY=1 \
TFPP_CHRONO_ENABLE=1 \
TFPP_CHRONO_RULE_CONTROL=1 \
TEAM_CONFIG=$TFPP_CKPT_ROOT/pretrained_models/all_towns \
~/sim/e2e_models/scripts/run_tfpp_debug_route.sh
```

## 통과 기준

- controller state가 로그에 기록된다.
- command vector가 soft/baseline/stiff 사이에서 실제로 변한다.
- suspension apply failure가 없거나 매우 적다.
- route가 crash 없이 진행된다.
- fallback이 작동한다.

## 실패 시 중단 조건

- command 계산 예외로 route 종료
- command 값이 NaN/inf
- 차량 자세가 즉시 불안정해짐
- 기존 TransFuser++ control 자체가 바뀌어 route가 망가짐

## Codex 최종 요약 요구

- rule 식
- 입력 feature
- fallback 조건
- command 분포
- apply success/failure 통계
- 다음 분석 단계 가능 여부

