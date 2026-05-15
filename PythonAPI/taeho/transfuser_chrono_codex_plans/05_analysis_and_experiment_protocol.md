# Phase 3-E - Analysis Script And Experiment Protocol

## 목표

TransFuser++ + Chrono suspension integration 로그를 분석하는 스크립트와 반복 실험 프로토콜을 만든다.

이 단계의 목적은 controller 성능을 결론내는 것이 아니라, 실험이 재현 가능하고 실패 원인을 구분할 수 있게 만드는 것이다.

## 선행 조건

`04_rule_based_suspension_controller.md`가 통과되어야 한다.

## 새 파일 권장

```text
~/sim/e2e_models/carla_garage/team_code/analyze_tfpp_chrono_suspension.py
```

또는 CARLA repo 안에 둘 경우:

```text
~/sim/carla-0.9.15/PythonAPI/taeho/analyze_tfpp_chrono_suspension.py
```

위치는 실제 로그 접근성과 실행 환경을 고려해 Codex가 결정한다.

## 분석 입력

jsonl 로그:

```text
~/sim/e2e_models/logs/transfuserpp_chrono/*.jsonl
```

선택적으로 result.json:

```text
~/sim/e2e_models/outputs/transfuserpp/debug_*/result.json
```

## 분석 출력

콘솔 요약:

```text
log path
mode
total ticks
chrono enable success
suspension apply attempts
suspension apply failures
distinct command states
soft/baseline/stiff counts
fallback count
avg/max roll
avg/max pitch
avg speed
avg steer
avg brake
route result if result.json is provided
```

선택적으로 CSV summary:

```text
~/sim/e2e_models/logs/transfuserpp_chrono/summary_YYYYMMDD_HHMMSS.csv
```

## 명령어 예시

최신 로그 자동 분석:

```bash
python ~/sim/e2e_models/carla_garage/team_code/analyze_tfpp_chrono_suspension.py
```

특정 로그:

```bash
python ~/sim/e2e_models/carla_garage/team_code/analyze_tfpp_chrono_suspension.py \
  --log ~/sim/e2e_models/logs/transfuserpp_chrono/<file>.jsonl
```

result.json 함께:

```bash
python ~/sim/e2e_models/carla_garage/team_code/analyze_tfpp_chrono_suspension.py \
  --log ~/sim/e2e_models/logs/transfuserpp_chrono/<file>.jsonl \
  --result ~/sim/e2e_models/outputs/transfuserpp/debug_YYYYMMDD_HHMMSS/result.json
```

## 반복 실험 프로토콜

기본 비교:

```text
Run A: TransFuser++ only
Run B: Chrono enable only
Run C: Chrono + baseline suspension
Run D: Chrono + rule-based suspension
```

각 run은 같은 route file과 같은 checkpoint를 사용한다.

권장 route:

```text
~/sim/e2e_models/carla_garage/leaderboard/data/debug.xml
```

실행은 Editor safe mode를 기준으로 한다.

## 판단 기준

이 단계에서 “controller가 성능을 향상시켰다”고 단정하지 않는다.

판단 가능한 것:

```text
TransFuser++ planning output이 기록되는가
Chrono enable이 성공하는가
suspension command가 매 tick 적용되는가
command state가 실제로 변하는가
차량 pose response가 로그에 남는가
route completion과 infractions가 같이 비교 가능한가
```

아직 판단하지 말 것:

```text
논문 수준의 승차감 개선
일반화 성능
Bench2Drive 전체 성능
package 기반 장시간 안정성
```

## 통과 기준

- 분석 스크립트가 최신 로그를 자동으로 찾는다.
- malformed line이 있어도 전체 분석이 중단되지 않는다.
- mode별 apply/fallback/pose 요약이 나온다.
- result.json이 주어지면 route status와 score를 함께 보여준다.

## Codex 최종 요약 요구

- 분석 스크립트 경로
- 실행 예시
- 출력 예시
- 반복 실험 프로토콜
- 아직 결론내릴 수 없는 항목

