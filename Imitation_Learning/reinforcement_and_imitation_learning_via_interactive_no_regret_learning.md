# Reinforcement and Imitation Learning via Interactive No-Regret Learning

Stéphane Ross, J. Andrew Bagnell

## 🧩 Problem to Solve

본 논문은 모방 학습(imitation learning) 및 구조적 예측(structured prediction)에서 학습자의 예측이 테스트될 입력 분포에 영향을 미치는 문제에 주목합니다. 특히 기존 모방 학습 방법론(예: DAGGER)은 행동의 비용(cost) 정보를 활용하지 않아, 복합적인 의사결정 시 치명적인 오류를 초래할 수 있습니다 (예: 절벽 옆 운전 시 비용 불감으로 인한 추락). 또한, SEARN과 같은 방법은 비용 정보를 사용하지만, 현재 정책의 롤아웃이 필요하거나 확률적 정책을 강요하여 비실용적일 수 있습니다. 강화 학습(reinforcement learning) 영역에서는 온라인 근사 정책 반복(online approximate policy iteration)의 성공이 관찰됨에도 불구하고 이에 대한 강력한 이론적 기반이 부족하다는 문제도 제기됩니다.

## ✨ Key Contributions

- **비용 정보를 활용하는 대화형 모방 학습(AGGREVATE) 제안**: 전문가의 비용-잔여 가치(cost-to-go) 정보를 활용하여, 단순히 전문가 행동을 모방하는 것을 넘어 장기적인 비용을 최소화하는 정책 학습 방법을 제시합니다.
- **강화 학습으로의 확장 (NRPI)**: 모방 학습 접근 방식을 모델 없는(model-free) 강화 학습 문제로 확장하여, 정책 반복 알고리즘의 새로운 패밀리인 'No-Regret Policy Iteration (NRPI)'을 개발합니다.
- **이론적 보장 제공**: AGGREVATE와 NRPI가 온라인 no-regret 학습의 원리를 활용하여 강력한 통계적 regret 보장(statistical regret guarantee)을 달성함을 이론적으로 입증합니다. 이는 기존 방법의 통계적 오류 감소(statistical error reduction)보다 더 강력한 보장입니다.
- **기존 방법론 통합 및 정당화**: SEARN과 같은 기존 모방 학습 방법이 전문가의 cost-to-go를 휴리스틱하게 사용하는 것에 대한 이론적 정당성을 제공하며, 온라인 근사 정책 반복의 관찰된 성공에 대한 이론적 지원을 제공합니다.
- **새로운 알고리즘 패밀리 제시**: 모방 학습 및 강화 학습을 위한 광범위한 새 알고리즘 패밀리를 제안하고, 기존 기술에 대한 통일된 시각을 제공합니다.

## 📎 Related Works

- **DAGGER (Dataset Aggregation)**: 기존 모방 학습 방법론으로, 정책 실행과 학습을 번갈아 가며 오류 누적 효과를 수정하지만, 비용 정보를 사용하지 않습니다.
- **SEARN (Search-based Structured Prediction)**: 구조적 예측을 위한 모방 학습 방법론으로 비용-잔여 가치를 고려하지만, 현재 정책의 롤아웃이 필요하고 확률적 정책을 사용해야 하는 제약이 있습니다.
- **SMILE (Structured Output Learning with Inaccurate Loss Functions)**: DAGGER와 SEARN과 함께 언급되는 반복 학습 절차 중 하나입니다.
- **Approximate Policy Iteration (API)**: 강화 학습의 핵심 프레임워크로, 특히 온라인 API의 경험적 성공에 대한 이론적 기반을 탐구합니다.
- **Weighted Majority & Follow-The-Leader**: 유한 및 무한 정책 클래스에 적용되는 no-regret 온라인 학습 알고리즘의 예시로 언급됩니다.
- **Policy Search by Dynamic Programming (PSDP)**: NRPI와 유사한 정신을 가진 정책 탐색 방법론으로, NRPI는 PSDP의 일반화된 형태를 제시합니다.
- **Contextual Bandit Algorithms**: 부분 정보 설정에서의 탐색 문제와 관련하여 언급됩니다.

## 🛠️ Methodology

본 논문은 `AGGREVATE`와 `NRPI` 두 가지 핵심 알고리즘을 제안합니다.

### AGGREVATE (Imitation Learning with Cost-To-Go)

`DAGGER`의 확장으로, 전문가의 행동을 단순히 모방하는 대신 전문가의 비용-잔여 가치($Q$)를 최소화하도록 학습합니다.

1. **초기화**: 데이터셋 $D \leftarrow \emptyset$, 정책 $\hat{\pi}_1$은 임의의 정책으로 초기화합니다.
2. **반복 학습($N$회)**: 각 반복 $i$마다 다음을 수행합니다.
   - **정책 혼합(선택 사항)**: 현재 학습 정책 $\hat{\pi}_i$와 전문가 정책 $\pi^*$를 $\beta_i$ 비율로 혼합하여 실행 정책 $\pi_i = \beta_i \pi^* + (1-\beta_i) \hat{\pi}_i$를 구성합니다.
   - **데이터 수집($m$개)**:
     - 에피소드를 초기 상태에서 시작하여 시간 $t-1$까지 현재 정책 $\pi_i$를 실행합니다.
     - 무작위로 샘플링된 시간 $t$의 현재 상태 $s_t$에서 탐색 행동 $a_t$를 실행합니다.
     - 시간 $t+1$부터 $T$까지는 전문가가 제어를 넘겨받아 계속 실행하며, 이 시점에서 예상되는 전문가의 비용-잔여 가치 $\hat{Q}$를 관찰합니다.
     - 수집된 데이터 $(s, t, a, \hat{Q})$를 $D_i$에 추가합니다.
   - **데이터 집계**: $D \leftarrow D \cup D_i$.
   - **정책 훈련**: 집계된 데이터 $D$를 사용하여 비용-민감 분류기 $\hat{\pi}_{i+1}$를 훈련합니다 (또는 온라인 학습기를 사용하여 $\hat{\pi}_{i+1}$를 업데이트합니다).
3. **최고 정책 반환**: $N$회 반복 후 검증 성능이 가장 좋은 $\hat{\pi}_i$를 반환합니다.

### NRPI (No-Regret Policy Iteration for Reinforcement Learning)

`AGGREVATE`의 변형으로, 전문가가 없는 모델 없는 강화 학습 환경에서 정책 반복을 수행합니다.

1. **초기화**: 데이터셋 $D \leftarrow \emptyset$, 정책 $\hat{\pi}_1$은 임의의 정책으로 초기화합니다.
2. **반복 학습($N$회)**: 각 반복 $i$마다 다음을 수행합니다.
   - **데이터 수집($m$개)**:
     - 무작위로 샘플링된 시간 $t$와 탐색 분포 $\nu_t$로부터 상태 $s_t$를 샘플링합니다.
     - 상태 $s_t$에서 탐색 행동 $a_t$를 실행합니다.
     - 시간 $t+1$부터 $T$까지는 _현재 학습자의 정책_ $\hat{\pi}_i$를 실행하며, 이 시점에서 예상되는 현재 정책의 비용-잔여 가치 $\hat{Q}$를 관찰합니다.
     - 수집된 데이터 $(s, a, t, \hat{Q})$를 $D_i$에 추가합니다.
   - **데이터 집계**: $D \leftarrow D \cup D_i$.
   - **정책 훈련**: 집계된 데이터 $D$를 사용하여 비용-민감 분류기 $\hat{\pi}_{i+1}$를 훈련합니다 (또는 온라인 학습기를 사용하여 $\hat{\pi}_{i+1}$를 업데이트합니다).
3. **최고 정책 반환**: $N$회 반복 후 검증 성능이 가장 좋은 $\hat{\pi}_i$를 반환합니다.

## 📊 Results

본 논문은 `AGGREVATE` 및 `NRPI`의 이론적 성능 보장을 제시합니다.

### AGGREVATE (Imitation Learning)

$N$번의 반복 후, $\hat{\pi}$는 학습된 정책들 $\hat{\pi}_{1:N}$의 균일 혼합 정책이고, $\pi^*$는 전문가 정책이며, $Q^*_{\text{max}}$는 전문가의 비용-잔여 가치의 상한, $\epsilon_{\text{class}}$는 정책 클래스 $\Pi$ 내에서 달성 가능한 최소 기대 비용-민감 분류 regret, $\epsilon_{\text{regret}}$는 온라인 학습 평균 regret일 때:

$$
J(\hat{\pi}) \leq J(\pi^*) + T[\epsilon_{\text{class}} + \epsilon_{\text{regret}}] + O\left(\frac{Q^*_{\text{max}} T \log T}{\alpha N}\right)
$$

충분한 반복 횟수 $N \to \infty$에서, 온라인 no-regret 알고리즘이 사용될 경우:

$$
\lim_{N \to \infty} J(\hat{\pi}) \leq J(\pi^*) + T \epsilon_{\text{class}}
$$

이는 학습된 정책이 집계된 데이터셋에서 비용-민감 분류 regret이 작은 정책이 존재한다면 전문가에 준하는 성능을 달성할 수 있음을 나타냅니다.

### NRPI (Reinforcement Learning)

$N$번의 반복 후, $\hat{\pi}$는 학습된 정책들 $\hat{\pi}_{1:N}$의 균일 혼합 정책이고, $\pi' \in \Pi$는 임의의 정책이며, $Q_{\text{max}}$는 학습된 정책의 비용-잔여 가치의 상한, $\epsilon_{\text{regret}}$는 온라인 학습 평균 regret, $D(\nu, \pi') = \frac{1}{T}\sum_{t=1}^T ||\nu_t - d_t^{\pi'}||_1$는 탐색 분포 $\nu_t$와 정책 $\pi'$에 의해 유도되는 상태 분포 $d_t^{\pi'}$ 간의 평균 $L_1$ 거리일 때:

$$
J(\hat{\pi}) \leq J(\pi') + T \epsilon_{\text{regret}} + T Q_{\text{max}} D(\nu, \pi')
$$

충분한 반복 횟수 $N \to \infty$에서, 온라인 no-regret 알고리즘이 사용될 경우:

$$
\lim_{N \to \infty} J(\hat{\pi}) \leq J(\pi') + T Q_{\text{max}} D(\nu, \pi')
$$

이는 `NRPI`가 탐색 분포 $\nu_t$에 상태 분포가 가까운 정책 $\pi' \in \Pi$만큼 좋은 정책을 찾을 수 있음을 보장합니다.

## 🧠 Insights & Discussion

본 논문은 온라인 알고리즘과 no-regret 분석이 제어 및 의사결정 분야의 학습을 이해하는 데 중요하다는 점을 강조합니다.

- **이론적 정당화**: 온라인 근사 정책 반복의 경험적 성공과 `SEARN`에서 전문가의 cost-to-go를 휴리스틱하게 사용하던 방식에 대한 강력한 이론적 기반을 제공합니다. 이는 이전에 설명이 부족했던 현상에 대한 명확한 이해를 돕습니다.
- **안정성과 효율성**: 여러 반복에 걸쳐 데이터를 집계하고 학습하는 전략은 학습된 정책이 다양한 상태에서 일관되게 좋은 성능을 발휘하도록 보장하여, 배치 근사 동적 프로그래밍 알고리즘에서 나타날 수 있는 진동 및 발산을 방지합니다.
- **통일된 관점**: 모방 학습과 강화 학습이라는 두 가지 영역을 no-regret 온라인 학습이라는 단일 프레임워크로 통합하는 새로운 관점을 제시합니다.

**제한 사항**:

- **비용-잔여 가치 추정의 비실용성**: 각 상태-행동 쌍에 대한 비용-잔여 가치 추정은 전체 궤적을 실행해야 할 수 있어 비실용적일 수 있습니다. `DAGGER`와 같은 단순 모방 손실 최소화는 궤적당 더 많은 데이터를 수집할 수 있어 더 실용적일 수 있습니다.
- **정책 클래스의 한계**: 만약 전문가가 정책 클래스 $\Pi$ 내의 어떤 정책보다도 훨씬 뛰어나다면, `AGGREVATE`는 좋은 정책을 학습하지 못할 수 있습니다. 예를 들어, 전문가는 위험한 지름길을 빠르게 가지만, 학습 정책은 그 길에서 추락할 수밖에 없는 경우, `AGGREVATE`는 더 안전한 우회로를 찾지 못하고 위험한 지름길을 선호할 수 있습니다.
- **NRPI의 성능 보장**: `NRPI`의 보장은 $O(T^2)$로 감소할 수 있어, 탐색 분포 $\nu_t$가 최적 정책의 상태 분포와 매우 가깝지 않으면 의미 있는 보장이 어려울 수 있습니다.

**향후 연구**:
다양한 no-regret 학습 알고리즘의 실제적인 장단점을 탐색하고, `NRPI`에서 탐색 분포 $\nu_t$를 반복적으로 조정하는 메커니즘을 개발하는 것이 중요합니다.

## 📌 TL;DR

본 논문은 비용 정보를 활용하지 못하는 기존 모방 학습의 한계를 극복하고 강화 학습에 대한 이론적 기반을 강화하기 위해, 온라인 no-regret 학습 기반의 `AGGREVATE` 및 `NRPI` 알고리즘을 제안합니다. `AGGREVATE`는 전문가의 비용-잔여 가치를 최소화하여 모방 학습의 성능을 향상시키고, `NRPI`는 이를 모델 없는 강화 학습으로 확장합니다. 두 알고리즘 모두 통계적 regret 감소라는 강력한 이론적 보장을 제공하며, 온라인 정책 반복의 성공과 `SEARN`의 휴리스틱에 대한 이론적 정당성을 부여합니다. 이는 모방 및 강화 학습에 대한 통합적인 관점을 제시하고 새로운 알고리즘 패밀리를 제안하지만, 비용 추정의 비실용성 및 정책 클래스의 한계 등의 제약도 가집니다.
