# AOAD-MAT: Transformer-based Multi-Agent Deep Reinforcement Learning Model considering Agents’ Order of Action Decisions

Shota Takayama and Katsuhide Fujita (2025)

## 🧩 Problem to Solve

본 논문은 다중 에이전트 강화학습(Multi-Agent Reinforcement Learning, MARL)에서 발생하는 환경의 비정상성(non-stationarity)과 에이전트 수 증가에 따른 joint action space의 기하급수적 증가 문제를 해결하고자 한다. 최근 Multi-Agent Transformer(MAT)나 ACtion dEpendent deep Q-learning(ACE)과 같이 의사결정 과정을 순차적(sequential)으로 처리하여 성능을 향상시킨 모델들이 등장하였다.

그러나 기존의 순차적 의사결정 모델들은 에이전트들이 의사결정을 내리는 **순서(Order of Action Decisions)** 자체가 성능과 학습 안정성에 미치는 영향을 명시적으로 고려하지 않았다는 한계가 있다. 특히 에이전트마다 능력이 다르거나 환경의 역학이 특정 행동 순서에 유리한 경우, 결정 순서의 최적화가 필수적이다. 따라서 본 논문의 목표는 에이전트의 의사결정 순서를 명시적으로 학습하고 최적화하는 AOAD-MAT 모델을 제안하여 MARL의 전체적인 팀 성능과 학습 효율을 높이는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 다중 에이전트의 행동 결정 과정을 단순한 순차적 생성으로 보는 것을 넘어, **어떤 에이전트가 다음으로 행동해야 하는지를 결정하는 '순서 예측'을 하나의 학습 가능한 하위 작업(subtask)으로 통합**하는 것이다.

이를 위해 Transformer 기반의 Actor-Critic 구조를 채택하고, 다음 행동을 수행할 에이전트를 예측하는 메커니즘을 도입하였다. 특히, 행동 예측과 에이전트 순서 예측이라는 두 가지 작업을 단순한 가중치 합(weighted sum) 방식이 아닌, 확률 분포의 비율을 곱하는 시너지 기반의 손실 함수(synergistic loss function)로 설계하여 Proximal Policy Optimization(PPO) 프레임워크 내에서 최적화하였다.

## 📎 Related Works

기존의 MARL 접근 방식은 크게 독립적 학습, 중앙 집중식 학습, 그리고 중앙 집중식 학습-분산 실행(CTDE) 구조로 나뉜다.

- **Value Factorization (VDN, QMIX):** Joint value function을 개별 가치 함수의 조합으로 근사하지만, 학습 과정에서 개별 및 전체 가치 함수 간의 불일치 문제가 발생하여 샘플 효율성이 떨어진다.
- **Policy Gradient (MAPPO, HAPPO):** PPO 등을 다중 에이전트에 적용하였으며, 특히 HAPPO는 순차적 에이전트 업데이트의 중요성을 강조하였다.
- **Sequence Modeling (MAT, ACE):** MAT는 Transformer를 사용하여 다중 에이전트 의사결정을 시퀀스 생성 문제로 정의함으로써 에이전트 간의 의존성을 캡처하였다. ACE는 양방향 행동 의존성(bidirectional action-dependency)을 통해 비정상성 문제를 완화하였다.
- **Formation-aware Exploration (FoX):** 공간적 관계를 통한 포메이션(formation) 개념으로 탐색 공간을 줄이려 시도하였다.

AOAD-MAT는 이러한 선행 연구들과 달리, CTCE(Centralized Training Centralized Execution) 프레임워크 내에서 **행동 결정 순서의 동적 최적화**에 집중한다. 이는 단순한 탐색 공간의 축소가 아니라, 최적의 결정 순서를 학습함으로써 탐색 능력을 근본적으로 개선하려는 접근 방식이라는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 시스템 구조

AOAD-MAT는 Encoder-Decoder 구조의 Transformer 아키텍처를 기반으로 한다.

- **Encoder (Critic):** 모든 에이전트의 joint observation을 입력받아 상태 가치(state value)를 추정하는 표현형(representation)을 학습한다.
- **Decoder (Actor):** Encoder가 생성한 표현형을 바탕으로, 어떤 에이전트가 행동할지($\pi_i$)와 그 에이전트가 어떤 행동을 할지($\pi_a$)를 순차적으로 예측한다.

### 2. 순차적 행동 결정 순서 예측 (Sequential Action Decision Order Prediction)

에이전트의 관측치 시퀀스를 $(\hat{o}_{i_1}, \dots, \hat{o}_{i_n})$이라고 할 때, 이를 임의의 순서 $(\hat{i}_1, \dots, \hat{i}_n)$으로 재정렬하는 스왑 함수 $\gamma$를 정의한다.

$$\hat{o}_{\text{swap}} = \gamma(\hat{o}_{i_1}, \dots, \hat{o}_{i_n}) = (\hat{o}_{\hat{i}_1}, \dots, \hat{o}_{\hat{i}_n})$$

이후 Multi-Agent Advantage Decomposition Theorem에 따라, 재정렬된 순서에 맞춰 Advantage function을 다음과 같이 분해하여 계산한다.

$$A_{\hat{i}_{1:n}}^\pi(\hat{o}_{\text{swap}}, a_{\hat{i}_{1:n}}) = \sum_{m=1}^{n} A_{\hat{i}_m}^\pi(\hat{o}_{\text{swap}}, a_{\hat{i}_{1:m-1}}, a_{\hat{i}_m})$$

### 3. 훈련 목표 및 손실 함수

모델은 두 가지 확률 분포를 출력한다.

- $\pi_a^m(\theta)$: 현재 결정 순서의 $m$번째 에이전트 $\hat{i}_m$이 취할 행동 $a_{\hat{i}_m}$에 대한 예측 분포.
- $\pi_i^m(\theta)$: 다음 순서($m+1$번째)로 행동할 에이전트 $\sigma(i_{m+1})$에 대한 예측 분포.

**PPO 기반의 시너지 손실 함수:**
두 작업의 확률 비율 $r_a^m(\theta)$와 $r_i^m(\theta)$를 각각 계산한 뒤, 이들의 곱을 통해 전체 비율 $r_m(\theta)$를 구한다.

$$r_m(\theta) = r_a^m(\theta) \cdot r_i^m(\theta)$$

최종 Decoder 손실 함수는 다음과 같다.

$$L_{\text{Decoder}}(\theta) = -\frac{1}{Tn} \sum_{m=1}^{n} \sum_{t=1}^{T-1} \left( \min \{ r_m(\theta) \hat{A}_t, \text{clip}(r_m(\theta), 1 \pm \epsilon) \hat{A}_t \} \right) - \beta_1 H[\pi_a(\theta)] - \beta_2 H[\pi_i(\theta)]$$

여기서 $H[\cdot]$는 엔트로피 항으로 탐색을 촉진하며, $\epsilon$은 PPO의 clip 파라미터이다. 비율의 곱을 사용하는 이유는 두 작업이 동일한 Advantage function을 최적화하므로, 두 방향이 일치할 때 업데이트를 강하게 촉진하고 다를 때 억제하여 학습의 안정성을 높이기 위함이다.

## 📊 Results

### 1. 실험 설정

- **데이터셋 및 환경:** StarCraft Multi-Agent Challenge (SMAC)와 Multi-Agent MuJoCo (MA-MuJoCo) 벤치마크를 사용하였다.
- **비교 대상:** 기본 MAT 모델 및 PPO clip 값을 조정한 MAT-adjust 모델과 비교하였다.
- **측정 지표:** SMAC에서는 Median Win Rate 및 Top $n\%$ step 성능을, MA-MuJoCo에서는 Average Reward를 측정하였다.

### 2. 주요 결과

- **SMAC:** 5m\_vs\_6m, MMM2 등 고난도 작업에서 AOAD-MAT는 baseline 대비 가장 높은 승률을 기록하였다. 특히 모든 모델이 최종적으로 100% 승률에 도달한 경우, 상위 $n\%$ step 성능 분석에서 AOAD-MAT가 일관되게 우수한 성능을 보였으며, 특히 MMM2 작업에서 그 차이가 두드러졌다.
- **MA-MuJoCo (HalfCheetah):** AOAD-MAT는 기존 MAT 대비 median reward에서 약 10%의 향상을 보였으며, 최고 성능(peak performance) 또한 훨씬 높게 나타났다.
- **학습 역학 분석:** AOAD-MAT의 성능 향상은 '다음 에이전트 예측'에 대한 엔트로피가 수렴하는 시점(약 80M steps 이후)과 일치한다. 이는 최적의 결정 순서가 학습된 이후 정책 업데이트가 매우 안정적으로 이루어짐을 시사한다.

## 🧠 Insights & Discussion

### 1. 강점 및 분석

- **적응적 순서의 효용성:** 실험 결과, 무작위 순서(Random)나 고정 순서(Fixed)보다 모델이 스스로 학습한 적응적 순서(Adaptive Ordering)가 훨씬 뛰어난 성능을 보였다. 이는 단순히 순서의 다양성이 중요한 것이 아니라, 상황에 맞는 '고품질의 순서'를 예측하는 능력이 탐색 효율성을 결정짓는 핵심임을 의미한다.
- **시너지 손실 함수의 효과:** 기존의 멀티태스크 학습 방식(Weighted Sum)보다 비율의 곱을 이용한 손실 함수가 더 안정적인 성능 향상을 이끌어냈다. 이는 행동 예측과 순서 예측이 서로 충돌하지 않고 상호 보완적으로 작동하게 함으로써 가능하다.

### 2. 한계 및 비판적 해석

- **초기 수렴 속도:** 결과 그래프(Fig 5)를 보면 초기 수렴 속도는 기존 MAT가 더 빠를 수 있다. 이는 순서 예측이라는 추가적인 학습 단계가 필요하기 때문에 발생하는 오버헤드로 해석된다.
- **리드 에이전트의 영향:** 첫 번째 행동 에이전트의 선택이 성능에 영향을 미친다는 점이 확인되었다. 특히 동질적(homogeneous) 팀에서는 중앙 에이전트의 배치가 중요했다. 그러나 이 리드 에이전트를 어떻게 자동으로 최적 선택할 것인지에 대한 구체적인 메커니즘은 본 논문에서 완전히 해결되지 않은 과제로 남아있다.

## 📌 TL;DR

본 논문은 MARL에서 에이전트들이 행동을 결정하는 **'순서'**가 성능에 결정적인 영향을 미친다는 점에 착안하여, 이를 동적으로 학습하고 예측하는 **AOAD-MAT** 모델을 제안하였다. Transformer 구조에 '다음 에이전트 예측' 하위 작업을 통합하고 시너지 기반의 PPO 손실 함수를 적용함으로써, SMAC와 MA-MuJoCo 벤치마크에서 기존 MAT 모델을 상회하는 성능과 학습 안정성을 입증하였다. 이 연구는 다중 에이전트 협동 제어에서 의사결정 순서의 최적화가 탐색 효율성과 최종 성능을 높이는 중요한 요소임을 시사하며, 향후 부분 관측 환경(POMDP)으로의 확장 가능성을 제시한다.
