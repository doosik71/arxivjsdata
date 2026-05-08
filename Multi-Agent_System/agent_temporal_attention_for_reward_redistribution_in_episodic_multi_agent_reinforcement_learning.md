# Agent-Temporal Attention for Reward Redistribution in Episodic Multi-Agent Reinforcement Learning

Baicen Xiao, Bhaskar Ramasubramanian, Radha Poovendran (2022)

## 🧩 Problem to Solve

본 논문은 에피소드 종료 시점에만 공유된 전역 보상(shared global reward)이 주어지는 에피소드형 다중 에이전트 강화학습(Episodic Multi-Agent Reinforcement Learning, MARL) 환경에서의 학습 효율성 문제를 다룬다.

에피소드 종료 시에만 보상이 제공되는 지연 보상(delayed reward) 구조에서는 에이전트가 에피소드 중간 단계에서 수행한 개별 행동의 품질을 평가하기 어렵다. 이는 강화학습의 핵심 과제인 두 가지 신용 할당(credit assignment) 문제로 이어진다.

1. **Temporal Credit Assignment (시간적 신용 할당):** 에피소드 전체 경로(trajectory) 중 어느 시점의 상태가 최종 보상에 결정적인 영향을 미쳤는지를 식별하는 문제이다.
2. **Multi-Agent Credit Assignment (다중 에이전트 신용 할당):** 특정 시점에서 여러 에이전트 중 누가 보상 획득에 얼마나 기여했는지를 식별하는 문제이다.

본 연구의 목표는 에피소드 보상을 각 시점과 에이전트별로 적절히 재배분하여, 학습을 가속화할 수 있는 밀집 보상 신호(dense reward signal)를 생성하는 방법론인 AREL(Agent-Temporal Attention for Reward Redistribution)을 제안하는 것이다.

## ✨ Key Contributions

AREL의 핵심 아이디어는 **Attention 메커니즘을 통해 시간적 차원과 에이전트 차원의 상관관계를 동시에 분석하여 보상을 재배분**하는 것이다.

- **Temporal Attention:** 에피소드 내의 상태 전이(state transition) 경로를 분석하여 시간적 중요도를 파악한다.
- **Agent Attention:** 각 시점에서 에이전트들이 서로 어떻게 영향을 주고받는지 분석하여 개별 에이전트의 기여도를 파악한다.
- **Permutation Invariance (치환 불변성):** 동일한 능력을 가진 동질적(homogeneous) 에이전트들의 경우, 에이전트의 순서가 바뀌어도 보상 결과가 동일해야 한다는 특성을 반영하여 샘플 효율성을 높였다.
- **Variance-based Regularization:** 재배분된 보상이 지나치게 희소(sparse)해지는 것을 방지하기 위해 분산 기반의 정규화 항을 손실 함수에 도입하였다.

## 📎 Related Works

기존의 보상 재배분 및 신용 할당 연구는 다음과 같은 한계점을 가지고 있다.

- **단일 에이전트 기반 방법론:** RUDDER나 Sequence Modeling과 같은 기법들은 시간적 신용 할당을 해결하지만, 다중 에이전트 환경으로 확장할 경우 관측 공간이 에이전트 수에 따라 지수적으로 증가하여 확장성(scalability) 문제가 발생한다.
- **다중 에이전트 신용 할당 방법론:** COMA, QMIX, QTRAN 등은 각 시점에서의 에이전트 간 기여도 분배(Multi-agent credit assignment)에 집중하지만, 보상이 에피소드 끝에만 주어지는 장기적인 시간적 신용 할당(Long-term temporal credit assignment) 문제는 해결하지 못한다.
- **기타 접근법:** IRCR과 같은 최신 기법은 대리 목적 함수(surrogate objective)를 사용하여 보상을 균일하게 재배분하지만, 실제 궤적(trajectory) 데이터를 통해 에이전트 간의 상대적 기여도를 정밀하게 특성화하지 못하는 한계가 있다.

AREL은 Temporal Attention과 Agent Attention을 결합함으로써 이 두 가지 차원의 신용 할당 문제를 동시에 해결하며, 에이전트별 관측치를 독립적으로 처리함으로써 확장성 문제를 극복한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

AREL은 에피소드 궤적 $E \in \mathbb{R}^{T \times N \times D}$ ($T$: 시간 길이, $N$: 에이전트 수, $D$: 임베딩 차원)를 입력으로 받아, 각 시점 $t$에 대한 재배분된 보상 $\hat{r}_t$를 예측하는 매핑 함수 $f_{redist}(E): \mathbb{R}^{T \times N \times D} \to \mathbb{R}^T$를 학습한다. 전체 구조는 **Agent-Temporal Attention Block**과 **Credit Assignment Block**으로 구성된다.

### 2. Agent-Temporal Attention Block

이 블록은 두 개의 어텐션 모듈이 직렬로 연결된 구조이다.

#### (1) Temporal-Attention Module

에이전트별로 시간 축을 따라 상태 간의 관계를 분석한다. 각 에이전트 $i$에 대해 Query($q$), Key($k$), Value($v$)를 생성하며, 다음과 같이 어텐션 가중치를 계산한다.
$$\alpha_{i,t} = \text{softmax}\left(\frac{q_{i,t} K_i^\top}{\sqrt{D}} \odot m_t\right)$$
여기서 $m_t$는 **Causality Mask**로, 시점 $t$에서 $t$ 이후의 미래 정보를 사용하지 못하도록 하여 인과성을 유지한다. 결과적으로 각 에이전트의 시간적 중요도가 반영된 특징 맵 $X \in \mathbb{R}^{N \times T \times D}$가 생성된다.

#### (2) Agent-Attention Module

앞선 모듈의 출력 $X$를 입력으로 받아, 동일 시점 내에서 에이전트 간의 상호작용을 분석한다.
$$\beta_{t,i} = \text{softmax}\left(\frac{q_{a,t,i} K_{a,t}^\top}{\sqrt{D}}\right)$$
이 모듈에서는 모든 에이전트의 정보를 참조하므로 마스킹을 사용하지 않는다. 최종적으로 모든 시점과 모든 에이전트의 정보가 통합된 특징 $Z \in \mathbb{R}^{T \times N \times D}$가 출력된다.

### 3. Credit Assignment Block

출력된 특징 $Z$를 기반으로 최종 보상을 예측한다. 이때 동질적 에이전트 간의 **Permutation Invariance**를 보장하기 위해 다음과 같은 구조를 사용한다.
$$\hat{r}_t = g_2 \left( \sum_{i=0}^{N-1} g_1(z_{t,i}) \right)$$

- $g_1, g_2$는 모든 에이전트가 공유하는 MLP(Multi-Layer Perceptron)이다.
- 개별 에이전트의 특징을 $g_1$으로 처리한 후 합산($\sum$)함으로써 에이전트의 순서와 상관없이 동일한 결과가 나오도록 설계되었다.

### 4. 학습 목표 및 손실 함수

재배분된 보상의 합이 실제 에피소드 보상 $R_T$와 일치하도록 회귀 손실(Regression Loss) $\mathcal{L}_r$을 최소화한다. 또한, 보상이 너무 희소하게 예측되어 학습이 정체되는 것을 막기 위해 재배분된 보상의 분산을 최소화하는 정규화 항 $\mathcal{L}_v$를 추가한다.
$$\mathcal{L}_{total}(\theta) = \mathcal{L}_r(\theta) + \omega \mathcal{L}_v(\theta)$$
$$\mathcal{L}_r(\theta) = \mathbb{E}_{E, R} \left[ \frac{1}{T} \left( \sum_t f_\theta(E_t) - R_T \right)^2 \right]$$
$$\mathcal{L}_v(\theta) = \mathbb{E}_E \left[ \frac{1}{T} \sum_t (f_\theta(E_t) - \bar{f}_\theta(E))^2 \right]$$
여기서 $\omega$는 정규화 강도를 조절하는 하이퍼파라미터이며, $\bar{f}_\theta(E)$는 재배분된 보상의 평균값이다.

### 5. 학습 절차 (Algorithm 1)

1. 에이전트들이 현재 정책 $\pi_\phi$에 따라 행동하며 궤적 $\tau$와 최종 보상 $R_T$를 수집하여 버퍼 $\mathcal{B}_e$에 저장한다.
2. 신용 할당 함수 $f_\theta$를 통해 궤적 $\tau$에 대한 재배분 보상 $\hat{r}_t$를 예측한다.
3. MARL 알고리즘(예: QMIX, MADDPG)의 업데이트 시, 실제 보상과 재배분 보상의 가중합 $\alpha \hat{r}_t + (1-\alpha)\mathbb{1}_{t=T}R_T$를 보상 신호로 사용하여 정책 $\pi_\phi$를 업데이트한다.
4. 일정 주기 $M$마다 수집된 궤적들을 사용하여 $\mathcal{L}_{total}$을 통해 신용 할당 함수 $f_\theta$를 업데이트한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋/환경:**
  - **Particle World:** Cooperative Push, Predator-Prey, Cooperative Navigation (에이전트 수 $N=3, 6, 15$ 설정).
  - **StarCraft (SMAC):** 2s3z, 3s_vs_5z, 1c3s5z 맵.
- **기준선 (Baselines):** RUDDER, Sequence Modeling, IRCR.
- **지표:** Particle World에서는 Agent Reward, StarCraft에서는 Test Win Rate를 측정하였다.

### 2. 주요 결과

- **성능 향상:** 모든 Particle World 태스크에서 AREL이 가장 높은 평균 보상을 기록하였다. StarCraft에서도 2s3z와 3s_vs_5z 맵에서 가장 높은 승률을 보였으며, 1c3s5z에서는 Sequence Modeling과 대등한 성능을 보였다.
- **기존 방법론 대비 우위:**
  - RUDDER와 PIC-baseline은 에피소드 보상 환경에서 정책 학습에 실패하는 경우가 많았다.
  - Sequence Modeling은 시간적 재배분은 수행하지만 에이전트 간 기여도를 고려하지 않아 AREL보다 성능이 낮았다.
  - IRCR은 일부 태스크에서 AREL과 유사한 성능을 보였으나, 보상 곡선의 분산이 매우 커서 불안정했다.

### 3. 절제 실험 (Ablation Study)

- **Agent Attention의 영향:** Agent Attention 모듈을 제거했을 때 보상이 유의미하게 하락하여, 에이전트 간 기여도 분석이 필수적임을 확인하였다.
- **정규화 파라미터 $\omega$:** $\omega$가 너무 작거나(0, 1) 너무 클 경우(10000) 성능이 저하되었으며, $\omega=20$에서 최적의 성능을 보였다.
- **보상 가중치 $\alpha$:** 재배분 보상만 사용하는 것($\alpha=1$)보다 실제 보상을 일부 섞어서 사용하는 것($\alpha=0.5$ 또는 $0.8$)이 더 높은 승률을 기록하였다.

## 🧠 Insights & Discussion

### 1. 강점 및 분석

본 논문은 단순히 보상을 쪼개는 것이 아니라, **Attention 메커니즘을 통해 "어떤 에이전트가 어떤 시점에 결정적인 행동을 했는가"를 데이터 기반으로 학습**했다는 점이 강력하다. 특히 정규화 항 $\mathcal{L}_v$를 통해 보상의 희소성 문제를 해결하여 RL 알고리즘이 더 안정적으로 수렴하게 만들었다.

### 2. 전략적 탐색 vs 신용 할당

MAVEN과 같은 전략적 탐색(strategic exploration) 기법과 비교했을 때, 보상이 지연되는 환경에서는 탐색 능력을 키우는 것보다 **효과적인 보상 재배분을 통해 신용 할당 문제를 해결하는 것이 승률 및 보상 향상에 더 직접적인 도움**이 됨을 입증하였다.

### 3. 해석 가능성 (Interpretability)

Cooperative Navigation 태스크의 시각화 결과, AREL이 예측한 보상 값이 에이전트들이 서로 다른 랜드마크를 향해 효율적으로 분산되는 시점에서 높게 나타남을 확인하였다. 이는 모델이 단순히 수치적으로 보상을 맞추는 것이 아니라, 실제 과제 달성에 중요한 '임계 상태(critical states)'를 정확히 포착하고 있음을 시사한다.

### 4. 한계점

에이전트 수가 증가함에 따라 어텐션 모듈을 학습시키기 위한 추가적인 계산 자원이 필요하며, 이는 에너지 소비 증가로 이어질 수 있다는 점이 명시되었다.

## 📌 TL;DR

이 논문은 에피소드 종료 시에만 보상이 주어지는 MARL 환경에서 학습 효율을 높이기 위해, **Temporal 및 Agent Attention을 결합하여 보상을 각 시점과 에이전트에게 밀집하게 재배분하는 AREL 프레임워크**를 제안한다. AREL은 Permutation Invariance와 분산 기반 정규화를 통해 확장성과 안정성을 확보했으며, Particle World와 StarCraft 실험을 통해 기존의 보상 재배분 기법들보다 우수한 성능을 입증하였다. 이 연구는 지연 보상 환경에서 단순한 탐색 강화보다 정밀한 신용 할당(Credit Assignment)이 더 중요할 수 있음을 시사하며, 향후 복잡한 협동 작업의 RL 학습에 중요한 기반이 될 가능성이 높다.
