# Smooth Imitation Learning for Online Sequence Prediction

Hoang M. Le, Andrew Kang, Yisong Yue, Peter Carr (2016)

## 🧩 Problem to Solve

본 논문은 동적이고 연속적인 환경에서 온라인 시퀀스 예측(online sequence prediction)을 위한 **Smooth Imitation Learning** 문제를 다룬다. 일반적인 Imitation Learning의 목표는 전문가의 행동을 모방하는 정책 $\pi$를 학습하는 것이지만, 실제 연속 제어 시스템(예: 자동 카메라 플래닝)에서는 단순히 정확한 예측뿐만 아니라 결과값이 시간에 따라 급격하게 변하지 않는 **Smoothness(부드러움)**가 필수적이다.

이 문제의 핵심적인 어려움은 학습된 정책의 예측값이 실제 실행 시 미래 상태의 분포에 영향을 미치기 때문에, 일반적인 지도 학습(Supervised Learning)의 기본 가정인 i.i.d.(independent and identically distributed) 가정이 깨진다는 점이다. 즉, 학습 데이터의 분포와 실제 실행 시의 상태 분포 사이에 불일치(distribution mismatch)가 발생하여 오차가 누적되는 현상이 나타난다.

따라서 본 논문의 목표는 상태 분포의 변화에 강건하면서도, 결과값이 부드럽게 유지되는 결정론적(deterministic) 정책을 효율적으로 학습할 수 있는 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Smooth Imitation Learning 문제를 **Smoothness가 보장된 복잡한 함수 클래스를 이용한 회귀 문제로 환원(Learning Reduction)**시키는 것이다. 이를 위해 **SIMILE(SmoothIMItationLEarning)** 알고리즘을 제안하며, 주요 기여 사항은 다음과 같다.

1. **Smooth Policy Class의 공식화**: 복잡한 모델 클래스 $F$와 부드러운 정규화 클래스 $H$를 결합하여, 예측값이 전문가의 행동과 정규화된 부드러운 함수 모두에 가깝도록 설계된 정책 클래스 $\Pi$를 정의하였다.
2. **SIMILE 알고리즘 제안**: 기존의 SEARN(확률적 혼합)이나 DAgger(데이터 집계)와 달리, **완전 결정론적인(fully deterministic) 정책**을 학습하며, 데이터 집계 없이도 안정적인 수렴을 보장하는 학습 환원 기법을 제시하였다.
3. **적응적 학습률(Adaptive Learning Rate) 도입**: 정책의 안정성(stability) 특성을 이용하여, 학습 단계에 따라 $\beta$ 값을 적응적으로 조절함으로써 기존 방식보다 훨씬 빠른 수렴 속도를 달성하였다.
4. **가상 피드백(Virtual Feedback) 메커니즘**: 전문가에게 매번 피드백을 요청하지 않고도, 현재 정책과 전문가의 행동을 보간한 가상 피드백을 생성하여 학습의 안정성을 높였다.
5. **이론적 보장**: 정책 개선(Policy Improvement) 보장이 시간 지평(time horizon $T$)에 의존하지 않음을 증명하여, 매우 긴 시퀀스에서도 빠르게 수렴할 수 있음을 보였다.

## 📎 Related Works

기존의 Imitation Learning 접근 방식은 크게 두 가지 방향으로 나뉜다.

- **Approximate Policy Iteration**: MDP 설정에서 정책 반복 기법을 사용하지만, 대부분 이산적이고 유한한 액션 공간에서 작동하며 연속적인 상태/액션 공간으로 확장하기 어렵다.
- **Iterative Learning Reductions**: Imitation Learning을 지도 학습 문제로 환원하는 방식이다.
  - **SEARN**: 새로운 정책을 학습한 후 기존 정책과 확률적으로 혼합($\pi_{n+1} = \beta \hat{\pi}_n + (1-\beta)\pi_n$)하여 분포 변화를 억제한다. 하지만 확률적 정책을 생성하므로 연속 제어에서 jitter(떨림)가 발생할 수 있고, 수렴 속도가 매우 느리다.
  - **DAgger**: 매 라운드마다 수집된 모든 데이터를 집계하여 학습한다. 하지만 데이터 양이 기하급수적으로 증가하여 계산 복잡도가 매우 높다.

**SIMILE와의 차별점**: SIMILE은 결정론적 보간(deterministic interpolation)을 사용하여 계산 효율성을 높이고, Smoothness 제약 조건을 이론적으로 활용하여 $T$에 무관한 수렴 속도를 보장한다.

## 🛠️ Methodology

### 1. 문제 정의 및 상태 공간

상태 공간 $S$는 외부 환경 입력 $x_t$와 이전 시점의 액션 $a_{t-1}$의 쌍으로 정의된다: $s_t = (x_t, a_{t-1})$. 정책 $\pi$는 상태 $S$를 액션 $A$로 매핑하는 함수 $\pi: S \rightarrow A$이다.

### 2. Smooth Policy Class $\Pi$

본 논문은 정책 $\pi$가 복잡한 모델 $f \in F$와 부드러운 정규화 함수 $h \in H$ 사이의 균형을 맞추도록 정의한다.
예를 들어, $\Pi_\lambda$ 클래스에서 정책 $\pi(x, a)$는 다음과 같이 정의된다:
$$\pi(x, a) = \text{argmin}_{a'} \left( \|f(x, a) - a'\|^2 + \lambda \|h(a) - a'\|^2 \right) = \frac{f(x, a) + \lambda h(a)}{1 + \lambda}$$
여기서 $\lambda$는 정확도와 부드러움 사이의 trade-off를 조절하는 하이퍼파라미터이다.

### 3. SIMILE 알고리즘 절차

SIMILE은 다음과 같은 반복적인 과정을 거친다 (Algorithm 1 참조).

1. **Roll-out**: 현재 정책 $\pi_{n-1}$을 사용하여 상태 시퀀스 $S_n$과 액션 시퀀스 $A_n$을 생성한다.
2. **Virtual Feedback 생성**: 학습 타겟 $\hat{a}^n_t$를 다음과 같이 생성한다:
    $$\hat{a}^n_t = \sigma a^n_t + (1 - \sigma) a^\circ_t$$
    여기서 $a^\circ_t$는 전문가의 액션이며, $\sigma$는 학습 초기에는 크게 설정하여 부드러운 수정을 유도하고 점차 0으로 줄여 전문가의 행동에 가깝게 만든다.
3. **Regularizer 업데이트**: 생성된 가상 피드백 $\hat{A}_n$에 가장 잘 맞는 부드러운 함수 $h_n \in H$를 학습한다.
4. **Policy 학습**: 고정된 $h_n$ 하에서 지도 학습을 통해 최적의 $\hat{\pi}_n$을 학습한다.
5. **Deterministic Interpolation**: 새로운 정책 $\pi_n$을 결정론적으로 업데이트한다:
    $$\pi_n = \beta \hat{\pi}_n + (1 - \beta) \pi_{n-1}$$
    이때 $\beta$는 $\hat{\pi}_n$과 $\pi_{n-1}$의 상대적 오차를 기반으로 적응적으로 결정한다:
    $$\hat{\beta} = \frac{\text{error}(\pi)}{\text{error}(\hat{\pi}) + \text{error}(\pi)}$$

### 4. 구체적인 구현: Smooth Regression Forests

논문에서는 $F$를 **Regression Tree Ensembles**로, $H$를 **Linear Auto-regressors**로 구현하였다. 결정 트리(Decision Tree)의 각 리프 노드에서 단순히 상수를 예측하는 대신, 이전 예측값들의 선형 조합인 $h_\pi(s)$와 결합된 값을 예측함으로써 시퀀스의 부드러움을 보장한다. 리프 노드의 값 $\bar{a}_{\text{node}}$는 다음의 joint loss를 최소화하도록 설정된다:
$$\text{argmin}_{\bar{a}} \sum_{(s, \hat{a}) \in D_{\text{node}}} ( \tilde{\pi}(s | \bar{a}) - \hat{a} )^2 + \lambda ( \tilde{\pi}(s | \bar{a}) - h_\pi(s) )^2$$

## 📊 Results

### 1. 실험 설정

- **작업**: 스포츠 중계용 자동 카메라 플래닝 (Automated Camera Planning).
- **입력**: 노이즈가 포함된 농구 선수들의 위치 데이터.
- **목표**: 전문가 운영자의 부드러운 카메라 팬(pan) 각도 궤적을 모방하는 정책 학습.
- **지표**: 전문가 궤적과의 평균 제곱 오차(MSE) 및 궤적의 1차 차분(first-order difference)을 통한 Smoothness 측정.

### 2. 주요 결과

- **Smoothness 향상**: $H$를 이용한 정규화를 적용한 경우, 적용하지 않은 경우보다 훨씬 부드러운 궤적을 생성함을 확인하였다 (Figure 2).
- **빠른 수렴 속도**: 적응적 학습률 $\beta$를 사용한 SIMILE이 SEARN의 보수적인 고정 학습률보다 훨씬 빠르게 수렴하였다 (Figure 3).
- **가상 피드백의 효과**: 학습 초기 단계에서 $\sigma$를 크게 설정하여 "부드러운 피드백"을 제공했을 때, 학습의 안정성이 크게 향상되었으며 이는 가상 피드백이 학습 가능한 smooth policy 영역 내로 정책을 유도하기 때문임을 보였다 (Figure 4, 5).
- **결정론적 보간의 우위**: 동일한 $\beta=0.5$ 조건에서 결정론적 정책 보간이 확률적 샘플링 방식보다 평균 오차가 더 낮음을 정량적으로 확인하였다 (Figure 7).

## 🧠 Insights & Discussion

### 강점 및 이론적 의의

본 논문은 Imitation Learning에서 발생하는 분포 불일치 문제를 해결하기 위해 "Smoothness"라는 도메인 지식을 이론적으로 통합하였다. 특히, Stability Condition(조건 1, 2)을 정의하여 정책 업데이트 시 발생하는 오차의 상한을 $T$에 무관하게 유도한 점이 매우 인상적이다. 이는 실제 환경에서 시퀀스 길이가 매우 길어질 때 기존의 환원 기법들이 겪는 느린 수렴 문제를 근본적으로 해결할 수 있는 가능성을 제시한다.

### 한계 및 논의사항

- **정규화 클래스 $H$의 선택**: 본 논문에서는 선형 오토레그레서(linear auto-regressor)를 사용하였으나, 시스템의 역학이 비선형성이 매우 강한 경우 단순한 선형 모델 $H$가 충분한 Smoothness를 제공하거나 정확도를 보장하지 못할 수 있다.
- **하이퍼파라미터 $\lambda, \sigma$**: $\lambda$와 $\sigma$의 스케줄링이 성능에 큰 영향을 미치는데, 이에 대한 자동화된 튜닝 방법론은 명시되지 않았다.

## 📌 TL;DR

본 논문은 연속적인 제어 환경에서 전문가의 행동을 부드럽게 모방하기 위한 **SIMILE** 알고리즘을 제안한다. 이 방법은 **결정론적인 정책 보간**과 **적응적 학습률**, 그리고 **가상 피드백** 메커니즘을 통해 기존의 SEARN이나 DAgger보다 계산 효율성이 높고 수렴 속도가 빠르며, 특히 시간 지평 $T$에 관계없이 안정적인 정책 개선을 보장한다. 실제 카메라 플래닝 작업에서 jitter가 적고 정확한 궤적 생성 능력을 입증하였으며, 이는 실시간 시스템의 부드러운 제어가 필요한 다양한 로보틱스 및 비전 제어 분야에 적용될 가능성이 높다.
