# VARIABLE-SHOT ADAPTATION FOR ONLINE META-LEARNING

Tianhe Yu, Xinyangang Geng, Chelsea Finn, Sergey Levine (2020)

## 🧩 Problem to Solve

본 논문은 새로운 작업을 학습하는 데 필요한 데이터의 양과 메타 학습(meta-learning)에 필요한 데이터의 양을 동시에 최소화하려는 문제, 즉 **온라인 증분 메타 학습(Online Incremental Meta-Learning)** 문제를 해결하고자 한다.

기존의 Few-shot 메타 학습 방법론들은 고정된 수의 예시(fixed number of examples)만을 사용하여 새로운 작업을 학습하는 성능에 집중해 왔다. 그러나 실제 환경에서는 데이터가 순차적으로 들어오며, 작업마다 가용한 데이터의 양이 다를 수 있다. 예를 들어, 어떤 작업은 데이터가 전혀 없는 Zero-shot 상태에서 시작하여 점진적으로 데이터가 쌓여 Many-shot 상태로 전환될 수 있다.

기존의 표준 경험적 위험 최소화(Empirical Risk Minimization, ERM) 방식은 데이터가 충분할 때 강력하지만, 데이터가 적은 초기 단계에서 적응 속도가 느리다는 단점이 있다. 반면, 기존 메타 학습 알고리즘들은 고정된 $K$-shot 설정에 최적화되어 있어, 가용한 데이터의 양이 변하는 가변 샷(variable-shot) 상황에서 최적의 성능을 내지 못한다. 따라서 본 논문의 목표는 가용한 데이터 양에 따라 유연하게 적응하며, 전체 시퀀스 작업에 대해 누적 후회(cumulative regret)를 최소화하는 메타 학습 알고리즘을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **가용한 데이터의 수($s$)에 따라 내부 학습률(inner learning rate)을 동적으로 조절하는 스케일링 규칙(scaling rule)**을 도입하는 것이다.

중심적인 직관은 데이터가 많아질수록 내부 그래디언트(inner gradient)의 분산이 줄어들기 때문에, 모델이 그래디언트를 더 신뢰하고 더 큰 보폭으로 업데이트할 수 있어야 한다는 것이다. 이를 위해 저자들은 이론적 유도를 통해 샷 수 $s$에 따라 최적의 학습률 $\alpha_s^*$가 결정됨을 보였으며, 이를 통해 소수의 파라미터만으로 다양한 샷 수에 대응할 수 있는 **MAML-VS(Model-Agnostic Meta-Learning with Variable-Shot and Scaling)** 알고리즘을 제안하였다. 또한 이를 온라인 설정으로 확장하여 **FTML-VS(Follow The Meta-Leader with Variable-Shot and Scaling)**를 구현함으로써, 순차적으로 들어오는 작업들에 대해 효율적인 전방 전이(forward transfer)를 달성하였다.

## 📎 Related Works

본 논문은 최적화 기반 메타 학습인 MAML을 기반으로 하며, 다음과 같은 관련 연구들과 차별점을 가진다.

1.  **MAML 및 FTML**: MAML은 빠른 적응을 위한 초기 파라미터를 찾고, FTML은 이를 온라인 설정으로 확장하여 순차적 작업에 대응한다. 하지만 두 방법 모두 내부 업데이트에 사용되는 데이터 수 $K$가 고정되어 있어, 실제 온라인 환경의 가변적인 데이터 양에 대응하기 어렵다.
2.  **비매개변수 메타 학습(Non-parametric Meta-learners)**: ProtoNet과 같은 방법들은 가변 샷에 자연스럽게 대응할 수 있지만, 일반적으로 클래스당 최소 한 개의 예시(one shot)가 필요하다. 본 논문은 클래스 간 경계가 배타적이지 않은(non-mutually exclusive) 설정까지 포함하여 Zero-shot 상황에서도 작동하는 모델을 지향한다.
3.  **지속 학습(Continual Learning)**: A-GEM과 같은 지속 학습 알고리즘은 주로 과거 지식의 망각(catastrophic forgetting)을 방지하는 후방 전이(backward transfer)에 집중한다. 반면, 본 연구는 과거 경험을 통해 새로운 작업을 더 빨리 배우는 전방 전이(forward transfer)의 가속화에 초점을 맞춘다.

## 🛠️ Methodology

### 1. MAML-VS: 가변 샷을 위한 학습률 스케일링

기존 MAML은 다음과 같은 목적 함수를 가진다.
$$\min_{\theta} \frac{1}{N} \sum_{i=1}^{N} f_i(U_i(\theta, \alpha, K))$$
여기서 $U_i$는 $\alpha$라는 고정된 학습률을 사용하는 내부 그래디언트 업데이트이다. 본 논문은 $\alpha$를 고정하지 않고 $s$에 따라 변하는 $\alpha_s(\beta, \eta)$로 대체한다.

**이론적 유도 및 방정식:**
저자들은 무한한 데이터가 있을 때의 최적 학습률 $\beta^*$와 유한한 $s$개의 데이터가 있을 때의 최적 학습률 $\alpha_s^*$ 사이의 평균 제곱 오차(MSE)를 최소화하는 식을 유도하였다. 그 결과, 다음과 같은 스케일링 규칙을 제안한다.
$$\alpha_s(\beta, \eta) = \left(1 - \frac{1}{1 + \eta s}\right) \beta$$
- $\beta$: 학습 가능한 기본 학습률 (s가 무한대로 갈 때의 최적 학습률에 해당)
- $\eta$: 학습 가능한 스케일링 인자 (데이터 수 $s$에 따른 신뢰도를 조절)
- $s$: 현재 사용 가능한 데이터(shot)의 수

이 규칙을 적용한 MAML-VS의 메타 학습 목적 함수는 다음과 같다.
$$\min_{\theta, \beta, \eta} \frac{1}{N} \sum_{i=1}^{N} \mathbb{E}_{s \sim \text{Unif}(0,M)} [f_i(\theta - \alpha_s(\beta, \eta) \nabla_\theta \hat{f}_i^s(\theta))]$$
여기서 $M$은 최대 샷 수이며, 모델은 $0$부터 $M$까지의 다양한 샷 상황에서 일반화되도록 학습된다.

### 2. FTML-VS: 온라인 증분 메타 학습

온라인 환경에서 위 알고리즘을 구현하기 위해 **Follow The Meta-Leader (FTML)** 구조를 채택한다.

**전체 파이프라인 및 절차:**
1.  **순차적 작업 수신**: 작업 $T_t$가 들어오면, 데이터 $\hat{D}_t(s)$를 하나씩(또는 작은 배치로) 수집한다.
2.  **가변 샷 적응**: 현재 수집된 $s$개의 데이터를 사용하여 $\alpha_s(\beta, \eta)$ 학습률로 모델을 업데이트한다.
3.  **능숙도 임계치(Proficiency Threshold) 기반 전이**: 모델의 성능 $f_t(U_t(\theta_t, \alpha, s))$가 미리 정의된 임계치 $C$보다 낮아지면(즉, 충분히 정확해지면), 즉시 다음 작업 $T_{t+1}$로 넘어간다.
4.  **메타 업데이트**: 리플레이 버퍼(replay buffer)에 저장된 이전 작업들을 샘플링하여 $\theta, \beta, \eta$를 동시에 업데이트함으로써, 앞으로 올 작업들에 더 빠르게 적응할 수 있도록 준비한다.

**최종 목표(Objective):**
전체 작업에 걸쳐 누적 후회(Cumulative Regret)를 최소화하는 것이다.
$$\text{Regret}_T = \sum_{t=1}^T \sum_{s=0}^{S_t} f_t(U_t(\theta_t, \alpha, \min\{s, M\})) - \min_{\theta} \sum_{t=1}^T \sum_{s=0}^{S_t} f_t(U_t(\theta, \alpha, \min\{s, M\}))$$

## 📊 Results

### 실험 설정
- **데이터셋**: Rainbow MNIST, Contextual MiniImageNet (비배타적 작업), Pose Prediction (비배타적 작업), Omniglot (배타적 작업)
- **비교 대상(Baselines)**:
    - **TOE (Train on Everything)**: 모든 데이터를 사용하여 표준 ERM으로 학습 (적응 과정 없음)
    - **FTML**: 고정 학습률을 사용하는 온라인 메타 학습
    - **FTML-VL**: 각 샷 수마다 개별 학습률을 학습하는 방식
    - **FTML + Meta-SGD**: 파라미터별 학습률을 학습하는 방식
    - **Incremental ProtoNet**: 비매개변수 기반의 프로토타입 네트워크 확장판
    - **A-GEM**: 대표적인 지속 학습 알고리즘
- **측정 지표**: 누적 후회(Cumulative Regret) 및 분류 정확도

### 주요 결과
1.  **누적 후회 감소**: Table 2에 따르면, FTML-VS는 세 가지 도메인(Rainbow MNIST, Contextual MiniImageNet, Pose Prediction) 모두에서 가장 낮은 누적 후회를 기록하였다. 특히 TOE(표준 ERM)보다 월등히 낮은 후회를 보여, 메타 학습이 단순 학습보다 훨씬 효율적으로 작업을 마스터함을 증명하였다.
2.  **가변 샷 적응 성능**: 오프라인 실험(Table 1)에서 MAML-VS는 0-shot부터 20-shot까지 전 구간에서 안정적이고 높은 성능을 보였다. 반면 기존 MAML은 0-shot 성능을 희생하고 Many-shot 성능을 높이거나, 그 반대의 경향을 보였다.
3.  **이론적 규칙의 효율성**: 모든 샷 수에 대해 개별 학습률을 학습하는 MAML-VL보다, 단순한 스케일링 규칙을 사용하는 MAML-VS가 유사하거나 더 나은 성능을 보였다. 이는 파라미터 효율성이 높으면서도 이론적으로 타당한 접근임을 시사한다.
4.  **지속 학습 및 비매개변수 방법론 대비 우위**: A-GEM이나 Incremental ProtoNet보다 낮은 후회를 기록하여, 전방 전이 가속화 측면에서 메타 학습의 강점을 확인하였다.

## 🧠 Insights & Discussion

본 논문의 강점은 메타 학습의 학습률 설정 문제를 단순한 휴리스틱이 아닌 **분산 분석을 통한 이론적 근거**로 해결했다는 점이다. 데이터 수가 적을 때의 높은 그래디언트 분산을 학습률 스케일링으로 제어함으로써, Zero-shot부터 Many-shot까지 하나의 모델이 유연하게 대응할 수 있게 하였다.

**비판적 해석 및 논의:**
- **작업의 성격**: 본 연구는 주로 '비배타적(non-exclusive) 작업' 설정에서 큰 이점을 보였다. 이는 모델이 사전 지식(prior)을 통해 Zero-shot 성능을 낼 수 있는 환경이기 때문이다. 실제로 배타적 작업인 Omniglot 실험(Table 5)에서는 MAML과 성능 차이가 거의 없었는데, 이는 데이터 수가 많아지면 그래디언트 분산이 자연스럽게 줄어들어 스케일링 규칙의 효과가 희석되기 때문으로 분석된다.
- **계산 복잡도**: 온라인 설정에서 리플레이 버퍼를 유지하고 지속적으로 메타 업데이트를 수행하는 것은 메모리와 계산 비용을 증가시킨다. 실시간 시스템에 적용하기 위해서는 이 비용과 성능 간의 트레이드-오프에 대한 추가 분석이 필요하다.
- **임계치 $C$의 의존성**: 작업 전환 기준이 되는 임계치 $C$가 성능에 영향을 미칠 수 있으며, 이 값의 설정 근거에 대한 논의가 더 보강될 필요가 있다.

## 📌 TL;DR

본 논문은 데이터가 순차적으로 들어오는 온라인 환경에서, 가용한 데이터 양에 따라 학습률을 동적으로 조절하는 **스케일링 규칙 $\alpha_s(\beta, \eta)$**를 제안하여 가변 샷 적응 문제를 해결하였다. 이를 통해 제안된 **FTML-VS** 알고리즘은 표준 ERM이나 기존 메타 학습 방법보다 훨씬 적은 데이터로 작업을 마스터하며 누적 후회를 크게 낮추었다. 이 연구는 향후 끊임없이 변화하는 환경에서 지속적으로 학습하고 개선되는 실제 AI 시스템을 구축하는 데 중요한 이론적/실천적 토대를 제공한다.