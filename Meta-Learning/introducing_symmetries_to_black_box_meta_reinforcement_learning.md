# Introducing Symmetries to Black Box Meta Reinforcement Learning

Louis Kirsch, Sebastian Flennerhag, Hado van Hasselt, Abram Friesen, Junhyuk Oh, Yutian Chen (2022)

## 🧩 Problem to Solve

본 논문은 메타 강화학습(Meta Reinforcement Learning)에서 '블랙박스(Black-box)' 접근 방식이 겪는 일반화(Generalisation) 성능 저하 문제를 해결하고자 한다. 블랙박스 메타 RL은 정책(Policy)과 학습 알고리즘(Learning Algorithm)을 단일 신경망(주로 RNN)으로 통합하여 표현하는 방식이다. 이러한 방식은 매우 유연하지만, 인간이 설계한 강화학습 알고리즘이나 역전파(Backpropagation) 기반의 메타 학습 방식에 비해 새로운 환경(Unseen environments)으로의 일반화 능력이 현저히 떨어진다는 한계가 있다.

문제의 핵심은 블랙박스 메타 학습자가 메타 훈련 환경의 특정 패턴에 과적합(Overfitting)되어, 범용적인 학습 원리를 깨우치는 것이 아니라 특정 환경에 특화된 고정된 정책을 생성하는 경향이 있다는 점이다. 예를 들어, 특정 팔(arm)의 보상이 항상 높았던 밴딧(Bandit) 환경에서 훈련된 블랙박스 모델은 보상을 관찰하여 학습하는 법을 배우는 대신, 단순히 첫 번째 팔을 당기도록 하는 편향된 해결책을 학습할 가능성이 크다. 따라서 본 논문의 목표는 블랙박스 메타 RL 시스템에 대칭성(Symmetry)을 도입하여, 훈련 데이터에 의존하지 않고 새로운 작업과 환경에서도 유연하게 작동하는 범용 학습 알고리즘을 발견하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 역전파 기반 학습 알고리즘이 본질적으로 가지고 있는 대칭성을 블랙박스 메타 학습 구조에 이식하는 것이다. 저자들은 역전파 기반 시스템이 다음과 같은 세 가지 주요 대칭성을 가지고 있음에 주목하였다: (1) 모든 파라미터에 동일한 업데이트 규칙이 적용되는 **대칭적 학습 규칙(Symmetric learning rule)**, (2) 입력, 출력 및 네트워크 크기에 제약을 받지 않는 **유연한 구조(Flexible architecture sizes)**, (3) 입력과 출력의 순서가 바뀌어도 동일한 정책을 생성하는 **순열 불변성(Permutation invariance)**이다.

이러한 직관을 바탕으로, 저자들은 **SymLA(Symmetric Learning Agents)**라는 새로운 아키텍처를 제안한다. SymLA는 신경망의 가중치(Weights)를 작은 파라미터 공유 RNN(LSTMs)으로 대체하고, 이들 간의 메시지 패싱(Message passing)을 통해 학습을 수행함으로써, 블랙박스 모델임에도 불구하고 위에서 언급한 역전파의 대칭적 특성을 그대로 유지하도록 설계되었다.

## 📎 Related Works

기존의 메타 강화학습은 크게 두 가지 방향으로 나뉜다.

1. **역전파 기반 방법(Backpropagation-based methods):** 학습 목표 함수(Objective function)를 메타 학습하거나 가중치 초기값을 최적화하는 방식이다. 이러한 방법들은 역전파 알고리즘 자체가 가진 대칭성 덕분에 일반화 능력이 뛰어나지만, 메모리 요구량이 많고 파괴적 망각(Catastrophic forgetting) 등의 문제와 미분 가능성 제약이 존재한다.
2. **블랙박스 방법(Black-box methods):** $\text{RL}^2$나 MetaRNN과 같이 단일 RNN이 에이전트의 상태와 학습 규칙을 동시에 표현하는 방식이다. 구현이 간단하고 표현력이 풍부하지만, 앞서 언급한 것처럼 메타 훈련 데이터에 과적합되어 새로운 환경에서의 일반화 성능이 낮다는 치명적인 단점이 있다.

본 논문은 특히 **Variable Shared Meta Learning (VSML)**의 개념을 강화학습 설정으로 확장하여, 블랙박스 모델의 유연성과 역전파 기반 모델의 일반화 능력을 동시에 확보하고자 하였다.

## 🛠️ Methodology

### 전체 구조 및 SymLA 아키텍처

SymLA는 기존 신경망의 고정된 가중치 $w_{ab}$를 동일한 파라미터 $\theta$를 공유하는 작은 RNN(LSTM)으로 대체한다. 각 RNN은 상태 $h_{ab}$를 가지며, 네트워크 내에서 전방향 메시지(Forward message, $\vec{m}$)와 후방향 메시지(Backward message, $\gets m$)를 주고받으며 상태를 업데이트한다.

### 주요 메커니즘 및 방정식

1. **RNN 상태 업데이트:** 각 레이어 $k$의 RNN 상태 $h_{ab}^{(k)}$는 다음과 같은 규칙으로 업데이트된다.
    $$h_{ab}^{(k)} \leftarrow f_{\text{RNN}}(h_{ab}^{(k)}, \vec{m}_a^{(k)}, \gets m_b^{(k)}, r_{t-1}, \vec{m}_b^{(k+1)}, \gets m_a^{(k-1)})$$
    여기서 $r_{t-1}$은 환경에서 받은 보상이며, $\vec{m}$과 $\gets m$은 각각 이전/다음 레이어로부터 오는 메시지이다.

2. **메시지 생성:**
    - **전방향 메시지:** $\vec{m}_b^{(k+1)} = \sum_a f_{\vec{m}}(h_{ab}^{(k)})$
    - **후방향 메시지:** $\gets m_a^{(k-1)} = \sum_b f_{\gets m}(h_{ab}^{(k)})$
    이러한 합산(Summation) 구조는 입력과 출력의 순서가 바뀌어도 결과가 동일하게 유지되는 순열 불변성을 보장한다.

3. **입출력 처리:**
    - 첫 번째 레이어의 전방향 메시지 $\vec{m}^{(1)}$에는 환경의 관측값(Observation) $o_t$가 입력된다.
    - 마지막 레이어의 후방향 메시지 $\gets m^{(K)}$에는 이전에 취한 행동 $a_{t-1}$이 입력된다.
    - 최종 출력은 $\vec{m}^{(K+1)}$의 첫 번째 차원을 액션 분포의 로짓(Logits)으로 사용하여 행동을 샘플링한다.

### 학습 절차

SymLA의 학습은 내측 루프(Inner loop)와 외측 루프(Outer loop)로 구성된다.

- **Inner Loop (Learning):** 고정된 메타 파라미터 $\theta$ 하에서 RNN 상태 $h_{ab}$가 업데이트된다. 이는 에이전트가 환경과 상호작용하며 실시간으로 학습하는 과정에 해당한다.
- **Outer Loop (Meta-Learning):** 에이전트의 생애 주기 동안 얻는 총 보상의 합 $\sum_{t=1}^L r_t(\theta)$를 최대화하도록 메타 파라미터 $\theta$를 최적화한다. 이때 미분 불가능한 영역이나 긴 시간 지평(Time horizon) 문제를 해결하기 위해 **진화 전략(Evolution Strategies, ES)**을 사용하여 $\theta$를 업데이트한다.
    $$\nabla_\theta \mathbb{E}_{\phi \sim \mathcal{N}(\phi|\theta, \Sigma)} \left[ \mathbb{E}_{e \sim p(e)} \left[ \sum_{t=1}^L r_t^{(e)}(\phi) \right] \right]$$

## 📊 Results

### 실험 설정

- **데이터셋 및 환경:** 밴딧(Bandits), 클래식 컨트롤(CartPole, Acrobot, MountainCar), 그리드 월드(GridWorld).
- **기준선(Baseline):** 표준 MetaRNN (대칭성이 없는 블랙박스 모델).
- **측정 지표:** 누적 후회(Cumulative Regret), 누적 보상(Cumulative Reward).

### 주요 결과

1. **밴딧 환경의 일반화:** MetaRNN은 훈련된 팔(arm)의 개수와 테스트 시의 팔 개수가 다르면 작동하지 않지만, SymLA는 팔의 개수가 변하더라도(Unseen action spaces) 안정적인 성능을 보였다.
2. **관측 및 행동 순열 불변성:** 클래식 컨트롤 작업에서 관측값과 행동의 순서를 섞었을 때(Shuffled), MetaRNN은 완전히 실패한 반면, SymLA는 성능 저하 없이 빠르게 새로운 순서에 적응하였다.
3. **미지의 작업으로의 일반화:** 그리드 월드에서 '하트(보상 +1)'와 '함정(보상 -1)'의 보상 체계를 서로 바꾼 실험에서, MetaRNN은 과거의 기억에 의존해 잘못된 아이템을 수집했으나, SymLA는 새로운 보상 관계를 실시간으로 추론하여 정답을 찾아냈다.
4. **환경 간 전이 학습:** 가장 놀라운 결과로, **그리드 월드에서 훈련된 SymLA가 한 번도 본 적 없는 CartPole 환경에서도 학습 능력을 발휘**했다. 이는 MetaRNN에서는 불가능했던 결과이며, 대칭성이 범용적인 학습 규칙을 발견하게 했음을 시사한다.

## 🧠 Insights & Discussion

### 강점 및 기여

본 논문은 블랙박스 메타 RL이 겪는 일반화 실패의 원인이 '대칭성의 부재'에 있음을 이론적/실험적으로 증명하였다. 단순한 모델 확장보다는 역전파가 가진 수학적 대칭성을 구조적으로 강제함으로써, 모델이 환경의 특정 지표를 외우는 것이 아니라 '보상을 통해 정책을 업데이트하는 방법' 그 자체를 배우도록 유도하였다.

### 한계 및 비판적 해석

- **계산 복잡도:** SymLA는 모든 가중치를 RNN으로 대체하므로, MetaRNN에 비해 계산 비용이 $O(N^2)$ 배 증가한다(여기서 $N$은 RNN의 hidden size). 실시간 추론 및 학습 속도 면에서 불리함이 있다.
- **초기화 의존성:** 순열 불변성이 성립하기 위해서는 RNN 상태 $h_{ab}$가 i.i.d.로 초기화되어야 한다는 가정이 필요하다.
- **ES의 효율성:** 외측 루프에서 ES를 사용하는데, 이는 파라미터 공간이 매우 클 경우 수렴 속도가 느려질 수 있다.

## 📌 TL;DR

이 논문은 블랙박스 메타 강화학습의 고질적인 문제인 과적합과 일반화 부족을 해결하기 위해, 역전파 알고리즘의 핵심 특성인 **대칭성(Symmetric learning rule, Permutation invariance)**을 도입한 **SymLA**를 제안한다. 가중치를 공유 RNN으로 대체한 이 구조는 입력/출력 크기의 변화나 순서 변경에 강건하며, 심지어 그리드 월드에서 배운 학습 능력을 CartPole과 같은 전혀 다른 환경으로 전이시키는 능력을 보여주었다. 이는 향후 블랙박스 기반의 범용 AI 학습 알고리즘 설계에 있어 구조적 대칭성이 필수적인 요소임을 시사한다.
