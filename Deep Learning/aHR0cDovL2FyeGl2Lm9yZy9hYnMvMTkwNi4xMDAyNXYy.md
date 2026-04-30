# Modern Deep Reinforcement Learning Algorithms

Sergey Ivanov (2019)

## 🧩 Problem to Solve

본 논문은 최근 딥러닝 패러다임과 고전적인 강화학습(Reinforcement Learning, RL) 이론의 결합으로 탄생한 Deep Reinforcement Learning(DRL) 알고리즘들을 체계적으로 분석하는 것을 목표로 한다. 

DRL은 고도의 복잡성을 가진 게임이나 로봇 제어 등에서 인간 수준의 성능을 보였으나, 다음과 같은 고질적인 문제점을 안고 있다. 첫째, 데이터 생성 비용이 높은 환경에서 샘플 효율성(Sample Efficiency)이 매우 떨어진다. 둘째, 무작위 초기화와 하이퍼파라미터에 대한 민감도가 높아 학습 과정이 불안정하며, 이는 결과의 재현성(Reproducibility) 저하로 이어진다. 셋째, 많은 알고리즘이 이론적 근거보다는 휴리스틱한 기법들의 조합으로 이루어져 있어, 각 구성 요소의 실질적인 기여도와 한계에 대한 명확한 분석이 필요하다.

따라서 본 보고서는 주요 DRL 알고리즘들의 이론적 정당성, 실무적 제한 사항, 그리고 관찰된 경험적 특성을 상세히 분석하여 제공하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 산재해 있는 최신 DRL 알고리즘들을 Value-based, Distributional, Policy Gradient라는 세 가지 주요 관점에서 통합적으로 리뷰하고, 이를 실제 실험을 통해 검증했다는 점이다.

특히, 단순한 이론 나열에 그치지 않고 다음과 같은 직관적 분석을 제공한다.
1. **Value-based 방법론의 고도화**: DQN에서 시작하여 Double, Dueling, Noisy, Prioritized Experience Replay, Multi-step 등 다양한 개선 기법들이 어떻게 상호작용하며 성능을 높이는지 분석한다.
2. **Distributional RL의 체계화**: 보상의 기댓값($Q$-value)만이 아닌 보상 분포($Z$-value) 전체를 학습하는 접근법(C51, QR-DQN)의 이론적 배경과 이점이 무엇인지 설명한다.
3. **Policy Gradient의 최적화 경로**: REINFORCE부터 PPO에 이르기까지, Gradient의 분산을 줄이고 안정적인 업데이트를 수행하기 위한 신뢰 영역(Trust Region) 및 클리핑(Clipping) 아이디어를 분석한다.
4. **실무적 Trade-off 분석**: 상호작용(Interaction) 횟수와 네트워크 업데이트(Training) 횟수 사이의 관계를 분석하여, Value-based 알고리즘의 계산 효율성을 높이는 방안을 제시한다.

## 📎 Related Works

논문은 강화학습을 Markov Decision Process(MDP)라는 일반적인 의사결정 프레임워크로 정의하며, 기존 연구들을 다음과 같이 분류하고 그 한계를 지적한다.

- **고전적 RL**: 상태 공간이 작을 때 테이블 형태로 저장하는 방식이다. 이론적 기초는 탄탄하나 고차원 입력(이미지 등)을 처리할 수 없다는 치명적인 한계가 있다.
- **Early DRL (DQN)**: 신경망을 통해 $Q$-함수를 근사함으로써 고차원 상태 공간 문제를 해결했다. 하지만 $Q$-value의 과대평가(Overestimation) 문제와 학습 불안정성이라는 문제가 제기되었다.
- **Policy Gradient**: $\pi(a|s)$를 직접 최적화하여 연속적인 행동 공간에서도 적용 가능하다. 그러나 Gradient 추정치의 분산이 매우 커서 학습 속도가 느리고 지역 최적점(Local Optima)에 빠지기 쉽다.

## 🛠️ Methodology

### 1. Value-based Algorithms
Value-based 방법론의 핵심은 Bellman 최적 방정식(Bellman Optimality Equation)을 푸는 것이다.

$$Q^*(s, a) = \mathbb{E}_{s' \sim p(s'|s,a)} [r(s') + \gamma \max_{a'} Q^*(s', a')]$$

- **DQN**: 신경망을 이용해 $Q$-함수를 근사하며, 학습 안정화를 위해 **Experience Replay**(과거 경험을 버퍼에 저장 후 무작위 샘플링)와 **Target Network**(타겟 값 계산용 네트워크를 분리하여 주기적으로 업데이트)를 도입한다.
- **Double DQN**: $\max$ 연산으로 인한 과대평가를 막기 위해, 행동 선택은 현재 네트워크 $\theta$가 하고, 가치 평가는 타겟 네트워크 $\theta^-$가 수행하도록 분리한다.
- **Dueling DQN**: $Q$-함수를 상태 가치 함수 $V(s)$와 이득 함수(Advantage function) $A(s, a)$의 합으로 분리하여, 특정 행동의 가치와 상관없이 상태 자체의 가치를 효율적으로 학습하게 한다.
- **Noisy DQN**: $\epsilon$-greedy 방식의 단순한 탐색 대신, 네트워크 가중치에 학습 가능한 노이즈를 추가하여 상태 의존적인 탐색을 수행한다.

### 2. Distributional Approach
기존 방법론이 보상의 기댓값 $Q(s, a) = \mathbb{E}[Z(s, a)]$를 학습했다면, Distributional RL은 확률 변수인 $Z(s, a)$의 분포 자체를 학습한다.

- **Categorical DQN (C51)**: 보상 범위를 일정한 간격의 원자(Atoms)들로 나누고, 각 원자에 속할 확률을 학습하는 Categorical 분포를 사용한다. 손실 함수로는 KL-divergence를 사용한다.
- **QR-DQN**: 고정된 원자 위치 대신, 확률 분포의 분위수(Quantiles) 위치 자체를 학습한다. 이는 Wasserstein metric을 최소화하는 것과 같으며, Quantile Regression 손실 함수를 통해 구현한다.
- **Rainbow DQN**: 위에서 언급한 DQN의 7가지 개선 사항(Double, Dueling, Noisy, PER, Multi-step, Distributional, DQN)을 모두 통합한 형태이다.

### 3. Policy Gradient Algorithms
정책 $\pi_\theta(a|s)$를 직접 최적화하여 목표 함수 $J(\theta)$를 최대화한다.

- **REINFORCE**: 몬테카를로 샘플링을 통해 Gradient를 직접 추정한다. 분산이 매우 크다는 단점이 있다.
- **A2C (Advantage Actor-Critic)**: 정책을 결정하는 Actor와 가치를 평가하는 Critic으로 구성된다. $\nabla_\theta \log \pi_\theta(a|s) A^\pi(s, a)$ 형태의 Gradient를 사용하여 분산을 줄인다.
- **PPO (Proximal Policy Optimization)**: 이전 정책과 현재 정책의 비율 $r(\theta)$가 너무 커지거나 작아지지 않도록 Clipping 함수를 적용하여 업데이트 보폭을 제한한다.

$$\text{Loss}^{CLIP}(\theta) = \mathbb{E}_t [\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$

## 📊 Results

### 실험 설정
- **환경**: Cartpole (기초 테스트), Atari Pong (복잡한 환경)
- **지표**: 샘플 효율성(Interaction steps), 벽시계 시간(Wall-clock time), 최종 스코어
- **비교 대상**: DQN, C51, QR-DQN, Rainbow, A2C, PPO

### 주요 결과
1. **Cartpole 결과**: PPO와 A2C가 매우 빠르게 수렴하며, Noisy 및 Prioritized 기법이 적용된 Value-based 모델들이 높은 성능을 보였다.
2. **Pong 결과**:
    - **샘플 효율성**: Value-based 알고리즘(특히 Rainbow, QR-DQN)이 Policy Gradient 방식보다 적은 상호작용으로도 더 빠르게 학습하는 경향을 보였다.
    - **계산 효율성 (Wall-clock time)**: PPO와 A2C가 압도적으로 빨랐다. 특히 PPO는 1시간 이내에 Pong을 해결했다. 반면 Rainbow는 Noisy Net의 연산 비용으로 인해 가장 느렸다.
3. **Interaction-Training Trade-off**: Value-based 알고리즘에서 네트워크 업데이트 횟수를 줄이고 배치 크기를 늘린 'Accelerated' 버전이 성능 저하 없이 학습 시간을 3.5배 단축시켰다. 이는 기존 DRL이 불필요하게 많은 업데이트를 수행하고 있음을 시사한다.

## 🧠 Insights & Discussion

### 강점 및 분석
본 논문은 DRL 알고리즘들이 단순히 성능 경쟁을 하는 것이 아니라, 서로 다른 계산적 비용-샘플 효율성 간의 Trade-off를 가지고 있음을 명확히 보여주었다. 특히 Value-based 방법론은 샘플 효율성이 좋지만 계산 비용이 높고, Policy Gradient 방법론은 샘플 효율성은 낮으나 계산 비용이 매우 낮다는 점을 실험적으로 입증하였다.

### 한계 및 비판적 해석
- **Rainbow DQN의 실체**: 저자는 Rainbow DQN이 이론적 일관성보다는 여러 기법을 "접착제와 테이프(glue and tape)"로 붙여놓은 형태라고 비판한다. 실제로 ablation study 결과, Dueling 구조는 성능 향상에 큰 영향이 없었음이 드러났다.
- **Distributional RL의 미스터리**: 보상의 분포를 학습하는 것이 왜 기댓값만 학습하는 것보다 성능이 좋은지에 대한 이론적 설명이 여전히 부족하다. 저자는 이를 '보조 작업(Auxiliary task)'으로서의 효과로 추측한다.
- **PPO의 효율성**: PPO가 Experience Replay 없이도 DQN 계열보다 더 나은 Gradient를 제공한다는 점은, 리플레이 버퍼에 저장된 데이터 중 상당수가 이미 학습되어 무의미한 정보일 수 있다는 점을 시사한다.

## 📌 TL;DR

본 논문은 현대 DRL의 핵심 알고리즘들을 이론적, 실무적 관점에서 종합 분석한 보고서이다. **Value-based(DQN 계열)**는 샘플 효율성이 뛰어나지만 계산 시간이 오래 걸리고, **Policy Gradient(PPO 계열)**는 샘플 효율성은 낮지만 학습 속도가 매우 빠르다는 특성을 확인하였다. 특히 Rainbow DQN과 같은 복합 알고리즘의 휴리스틱한 성격을 지적하며, 실제 적용 시에는 환경의 비용(데이터 생성 비용 vs 계산 비용)에 따라 적절한 알고리즘을 선택해야 함을 강조한다.