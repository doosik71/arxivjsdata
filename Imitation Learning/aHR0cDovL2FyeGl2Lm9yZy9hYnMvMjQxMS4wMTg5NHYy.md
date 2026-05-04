# Efficient Active Imitation Learning with Random Network Distillation

Emilien Biré, Anthony Kobanda, Ludovic Denoyer, Rémy Portelas (2025)

## 🧩 Problem to Solve

본 논문은 복잡하고 목표 함수가 명확하지 않은 환경(예: 비디오 게임의 봇 구현)에서 에이전트를 학습시키기 위한 Imitation Learning(모방 학습)의 효율성 문제를 다룬다. 

일반적인 Behavioral Cloning(BC) 방식은 고정된 전문가 데이터셋으로 학습하기 때문에, 실제 배포 시 학습 데이터에 없었던 상태에 진입하게 되면 오류가 누적되어 성능이 급격히 저하되는 Covariate Shift(공변량 변화) 문제가 발생한다. 이를 해결하기 위해 데이터셋을 확장하는 방법이 있지만, 인간 전문가의 데이터를 계속해서 수집하는 것은 시간과 비용 측면에서 매우 비효율적이다.

따라서 본 연구의 목표는 전문가의 개입이 정말로 필요한 순간에만 선택적으로 개입을 요청하는 Active Imitation Learning 방법을 개발하여, 전문가의 부담(Expert Burden)을 최소화하면서도 높은 성능의 정책을 학습시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 state-based Out-of-Distribution(OOD) 측정을 통해 전문가의 개입 시점을 결정하는 **RND-DAgger** 알고리즘을 제안한 것이다.

1. **State-based OOD Trigger**: 기존의 Active IL 방법론들이 전문가의 행동(Action)과 에이전트의 행동 사이의 차이(Discrepancy)를 측정하여 개입 여부를 결정했던 것과 달리, RND(Random Network Distillation)를 활용하여 에이전트가 현재 방문한 상태(State)가 학습 데이터셋에 비해 얼마나 생소한지를 측정한다. 이를 통해 행동의 가변성이 큰 인간 전문가 환경에서도 안정적인 개입 트리거를 구축하였다.
2. **Minimal Demonstration Time (MET)**: 전문가가 한 번 개입했을 때 너무 짧은 시간만 제어하고 다시 에이전트에게 권한을 넘기는 현상을 방지하기 위해, 최소 $W$ 스텝 동안은 전문가가 제어권을 유지하도록 하는 메커니즘을 도입하여 학습의 안정성을 높이고 전문가의 인지적 부하를 줄였다.
3. **효율성 검증**: 로보틱스 및 3D 게임 환경에서 RND-DAgger가 기존 방법론 대비 전문가의 개입 횟수(Context Switches)와 총 소요 시간(Total Expert Time)을 획기적으로 줄이면서도 동등하거나 더 우수한 성능을 보임을 입증하였다.

## 📎 Related Works

### 기존 접근 방식 및 한계
- **Behavioral Cloning (BC)**: 전문가 데이터셋을 단순히 모방하지만, 앞서 언급한 Covariate Shift 문제에 취약하다.
- **DAgger**: 에이전트가 방문하는 모든 상태에 대해 전문가의 정답 행동을 쿼리하여 데이터셋을 확장한다. 하지만 전문가가 모든 타임스텝에서 개입해야 하므로 부담이 매우 크다.
- **Ensemble-DAgger**: 여러 정책의 불일치(Disagreement)와 전문가 행동과의 차이(Discrepancy)를 측정하여 개입한다. 그러나 여전히 매 스텝 전문가의 행동 값이 필요하며, 전문가가 지속적으로 게임을 플레이해야 하는 부자연스러운 환경을 만든다.
- **Lazy-DAgger**: 분류기(Classifier)를 통해 전문가 개입 필요성을 예측하여 개입 횟수를 줄이려 한다. 하지만 근본적으로 Action-based discrepancy에 의존하므로, 동일 상태에서 다른 행동을 할 수 있는 인간 전문가의 데이터에서는 노이즈가 심해 신뢰도가 떨어진다.

### 차별점
RND-DAgger는 전문가의 행동을 실시간으로 비교하는 대신, **상태의 희소성(State Novelty)**에 집중한다. 이는 전문가가 어떤 행동을 취하느냐와 관계없이 에이전트가 "모르는 영역"에 들어왔는지를 판단하는 것이므로, 인간 전문가의 행동 가변성에 영향을 받지 않는 더 강건한 트리거를 제공한다.

## 🛠️ Methodology

### 전체 파이프라인
RND-DAgger는 전문가 데이터셋으로 초기 정책 $\pi_0$를 학습시킨 후, 다음과 같은 반복적인 루프를 통해 정책을 업데이트한다.

1. **상태 탐색 및 OOD 측정**: 현재 정책 $\pi_i$로 에이전트를 제어하며, 매 스텝마다 해당 상태 $s_t$가 OOD인지 측정한다.
2. **전문가 개입 (Trigger)**: OOD 측정값이 임계값 $\lambda$를 초과하면 전문가 $\pi_{exp}$가 제어권을 넘겨받아 에이전트를 직접 조종하며 데이터를 수집한다.
3. **제어권 반환 (MET)**: OOD 측정값이 $\lambda$보다 낮아지더라도, 즉시 반환하지 않고 최소 $W$ 스텝 동안은 전문가가 제어를 유지한다.
4. **모델 업데이트**: 수집된 새로운 데이터를 기존 데이터셋 $D$에 추가하고, 정책 $\pi_{i+1}$와 RND 예측 네트워크 $f_{pred}$를 다시 학습시킨다.

### RND (Random Network Distillation) 기반 OOD 측정
상태의 생소함을 측정하기 위해 두 개의 신경망을 사용한다.
- **Target Network ($f_{targ}$)**: 랜덤하게 초기화된 후 고정(Freeze)된 네트워크이다.
- **Predictor Network ($f_{pred}$)**: $f_{targ}$의 출력을 예측하도록 학습되는 네트워크이다.

상태 $s_t$에 대한 OOD 측정값 $m_t$는 다음과 같은 평균 제곱 오차(MSE)로 정의된다.
$$m_t = \|f_{targ}(s_t) - f_{pred}(s_t)\|^2$$
에이전트가 자주 방문한 상태(In-distribution)에 대해서는 $f_{pred}$가 $f_{targ}$의 값을 잘 예측하므로 $m_t$가 낮아지지만, 처음 보는 상태(OOD)에서는 예측 오차가 크게 발생하여 $m_t$가 높아진다.

### Minimal Demonstration Time (MET)
단순히 $m_t \le \lambda$일 때 제어권을 반환하면, 전문가가 상태를 완전히 복구하기 전에 제어권이 넘어가 다시 즉시 개입이 요청되는 "잦은 스위칭" 문제가 발생한다. 이를 방지하기 위해 다음과 같은 조건을 적용한다.
- 전문가가 제어권을 유지하기 위한 조건: $m_t > \lambda$ 이거나, $m_t \le \lambda$인 상태가 연속적으로 $W$ 스텝 미만으로 지속되었을 때.
- 즉, 최소 $W$ 스텝 동안 안정적인 상태(In-distribution)에 머물러야만 정책 $\pi$에게 제어권을 반환한다.

## 📊 Results

### 실험 설정
- **환경**: 
    - `HalfCheetah`: 로봇 보행 제어 (보상 합계 측정)
    - `RaceCar`: 3D 레이싱 (완주 성공률 측정)
    - `3D Maze`: 목표 지점 도달 내비게이션 (성공률 측정)
- **비교 대상**: BC, DAgger, Ensemble-DAgger, Lazy-DAgger, HG-DAgger (인간 게이트 방식)
- **주요 지표**: Task Performance (성공률/보상), Dataset Size (전문가 데이터 양), Context Switches (개입 횟수), Total Expert Time (총 전문가 소요 시간)

### 주요 결과
- **정량적 성능**: Table 1에 따르면, RND-DAgger는 모든 환경에서 Ensemble-DAgger 및 Lazy-DAgger와 대등하거나 더 우수한 성능을 기록하였다. 특히 Maze 환경에서는 성공률 $0.717$로 가장 높은 성능을 보였다.
- **전문가 부담 감소**:
    - **Context Switches**: RND-DAgger는 Ensemble-DAgger 대비 압도적으로 적은 스위칭 횟수를 기록하였다.
    - **Total Expert Time**: Table 5에서 Maze 환경의 경우, Ensemble-DAgger는 성능 달성을 위해 약 $85.7$분이 소요된 반면, RND-DAgger는 약 $27.0$분만으로 유사한 성능을 달성하여 효율성을 입증하였다.
- **인간 전문가 실험**: 실제 인간을 대상으로 한 실험에서 RND-DAgger는 HG-DAgger와 유사한 성능을 보이면서도, 전문가가 화면을 계속 주시하며 개입 시점을 결정할 필요가 없어 인지적 부하를 획기적으로 줄였다.

## 🧠 Insights & Discussion

### 강점 및 분석
본 논문은 전문가의 행동 기반(Action-based) 측정 방식이 가진 근본적인 한계, 즉 "인간은 동일 상황에서도 다르게 행동할 수 있다"는 점을 정확히 짚어냈다. RND-DAgger는 상태 기반(State-based) OOD를 사용함으로써 전문가의 행동 가변성에 영향을 받지 않고, 에이전트가 정말로 "모르는 상황"에 처했을 때만 도움을 요청한다. 또한, MET 메커니즘을 통해 전문가가 충분한 복구 경로를 보여줄 수 있도록 보장함으로써 데이터의 질을 높이고 스위칭 횟수를 줄인 점이 매우 실용적인 설계라고 판단된다.

### 한계 및 향후 과제
- **실시간 제어의 위험성**: 전문가가 즉시 제어권을 넘겨받아야 한다는 가정은 실제 물리 환경에서는 위험하거나 불가능할 수 있다. 이를 위해 개입 전 미리 알림을 주는 Predictive approach가 필요하다.
- **관측 가능성 문제**: 본 연구는 상태 공간이 명확한 환경을 다루었으나, 픽셀 기반의 Partially Observable MDP(POMDP) 환경으로 확장할 경우 "Noisy-TV" 문제(의미 없는 무작위 변화를 새로운 상태로 오인하는 문제)가 발생할 수 있으며, 이를 해결하기 위한 표현 학습(Representation Learning) 연구가 필요하다.

## 📌 TL;DR

본 논문은 Random Network Distillation(RND)을 활용하여 에이전트가 방문한 상태의 생소함을 측정하고, 이를 기반으로 전문가의 개입 시점을 결정하는 **RND-DAgger**를 제안한다. 기존의 행동 기반 차이 측정 방식과 달리 상태 기반 OOD를 사용함으로써 전문가의 행동 가변성에 강건하며, Minimal Demonstration Time(MET)을 통해 개입의 안정성을 확보하였다. 실험 결과, 전문가의 시간적/인지적 부담을 획기적으로 줄이면서도 높은 성능의 정책을 학습시킬 수 있음을 보였으며, 이는 향후 실제 인간 전문가를 활용한 효율적인 로봇 및 게임 AI 학습에 중요한 기여를 할 것으로 기대된다.