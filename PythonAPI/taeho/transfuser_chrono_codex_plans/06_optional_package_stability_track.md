# Optional Track - Packaged CARLA Stability

이 단계는 Phase 3 mainline이 아니다. TransFuser++ suspension hook 개발은 Editor safe mode에서 진행한다.

package 안정화는 반복 실험 자동화가 필요해졌을 때 별도 트랙으로 수행한다.

## 현재 관찰

패치 전/후 모두 packaged Shipping 서버에서 Vulkan sensor rendering crash가 반복되었다.

대표 callstack:

```text
FSkeletalMeshSceneProxy::GetMeshElementsConditionallySelectable
FDeferredShadingSceneRenderer::Render_CARLA
UpdateSceneCaptureContent_RenderThread
RHI.RHIName = Vulkan
```

`-RenderOffScreen`을 꺼도 crash가 재현되었다.

OpenGL 전환 시도는 다음 경고와 함께 Vulkan fallback 되었다.

```text
Warning: OpenGL is no longer supported for desktop platforms. Vulkan will be used instead.
```

## package 트랙을 다시 시작할 조건

아래 중 하나가 필요할 때만 진행한다.

```text
반복 실험 자동화
장시간 route 평가
Editor 없이 headless-like 실행
다수 route batch 실행
```

## 후보 작업

1. Development package 생성

```bash
cd ~/sim/carla-0.9.15
make package ARGS="--config=Development --no-zip"
```

2. sensor 최소화 실험

```text
RGB only
RGB + IMU/GNSS/speed
RGB + LiDAR
full TransFuser++ sensors
```

3. route/map 최소화 실험

```text
server boot only
check_carla_server.py only
single RGB camera client only
TransFuser++ STOP_AFTER_METER=30
full debug route
```

4. package launch script는 최신 수정시간 기준 package를 선택한다.

```text
~/sim/e2e_models/scripts/launch_carla_package_tfpp_safe.sh
```

## 중단 기준

package 안정화가 suspension hook 개발을 막기 시작하면 중단한다.

연구 기능 개발의 기준 서버는 Editor safe mode다.

