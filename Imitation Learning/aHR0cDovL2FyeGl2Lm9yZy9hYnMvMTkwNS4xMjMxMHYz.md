# Adversarial Imitation Learning from Incomplete Demonstrations

Mingfei Sun and Xiaojuan Ma (2019)

## 🧩 Problem to Solve

본 논문은 전문가의 시연(demonstrations)으로부터 상태-행동 매핑인 정책(policy)을 학습하는 Imitation Learning(모방 학습)에서 시연 데이터의 행동(action) 정보가 불완전한 경우를 해결하고자 한다.

일반적인 모방 학습 알고리즘인 Behavior Cloning(BC)이나 Inverse Reinforcement Learning(IRL)은 모든 상태에 대해 전문가의 행동이 완전히 관측 가능하다는 가정을 전제로 한다. 그러나 실제 환경에서는 다음과 같은 이유로 행동 정보가 누락될 수 있다. 첫째, 로봇이 인간의 동작을 관찰할 때 신체 움직임(state)은 보이지만, 관절에 가해지는 힘이나 토크(action)는 관측할 수 없는 경우가 많다. 둘째, 전문가의 숙련도나 개인적 선호도로 인해 일부 유효하지 않은 행동이 포함되어 학습에서 제외해야 할 필요가 있다.

기존 연구 중 행동 없이 상태 정보만으로 학습하는 방법들이 제안되었으나, 이는 행동 정보가 일부라도 존재할 때 얻을 수 있는 유용한 힌트를 완전히 무시한다는 한계가 있다. 따라서 본 논문의 목표는 행동 시퀀스가 불완전한 시연 데이터로부터 효율적으로 정책을 학습할 수 있는 알고리즘을 개발하는 것이다.

## ✨ Key Contributions

본 논문은 불완전한 시연 데이터로부터 정책을 학습하기 위한 **Action-Guided Adversarial Imitation Learning (AGAIL)** 알고리즘을 제안한다.

AGAIL의 핵심 아이디어는 시연 데이터를 상태 궤적(state trajectories)과 행동 궤적(action trajectories)으로 분리하여 처리하는 것이다. 상태 궤적은 전체적인 정책 학습의 기반으로 사용하고, 일부만 존재하는 행동 정보는 학습 과정을 안내하는 보조 정보(auxiliary information)로 활용하여 Generator의 탐색을 가이드한다. 이를 위해 AGAIL은 Generator, Discriminator, Guide라는 세 가지 네트워크 구성 요소를 도입하여 서로 상호작용하도록 설계하였다.

## 📎 Related Works

모방 학습의 대표적인 접근 방식으로는 상태에서 행동으로의 매핑을 직접 학습하는 Behavior Cloning(BC)과 전문가의 행동을 최적화하는 보상 함수를 찾는 Inverse Reinforcement Learning(IRL)이 있다. BC는 compounding error 문제로 인해 상태 분포가 달라지면 성능이 급격히 저하되며, IRL은 보상 함수 탐색 과정이 계산 집약적이고 제약 조건 설정이 까다롭다는 단점이 있다.

이를 해결하기 위해 Generative Adversarial Imitation Learning(GAIL)이 제안되었으며, 이는 판별자(Discriminator)와 생성자(Generator)의 적대적 학습을 통해 전문가의 상태-행동 점유 측도(occupancy measure)를 모방한다.

행동 정보가 없는 경우에 대한 기존 연구로는 상태 전이 모델을 통해 행동을 복원하거나, 상태 궤적만을 사용하여 GAIL을 확장한 방법들이 있다. 하지만 행동 복원 방식은 상태 전이의 노이즈에 취약하며, 상태 정보만 사용하는 방식은 행동 정보가 주는 유용한 가이드를 무시하므로 학습을 위해 막대한 양의 환경 상호작용이 필요하다는 한계가 있다.

## 🛠️ Methodology

### 전체 시스템 구조
AGAIL은 상태 기반의 적대적 모방 학습(State-Based Adversarial Imitation)과 행동 가이드 정규화(Action-Guided Regularization)라는 두 가지 핵심 메커니즘으로 구성된다. 전체 파이프라인은 Generator $\pi_\theta$, Discriminator $D_\omega$, Guide $Q_\psi$ 세 네트워크가 유기적으로 작동하며, TRPO(Trust Region Policy Optimization)를 통해 정책을 업데이트한다.

### 1. State-Based Adversarial Imitation
기존 GAIL은 상태-행동 쌍 $\rho(s, a)$의 분포를 일치시키려 하지만, AGAIL은 보상 함수 $r$이 주로 상태 $s$에 의존한다는 가정하에 상태 분포 $\nu(s)$만을 일치시키는 방식으로 단순화한다.

정책 $\pi$에 의한 상태 방문 빈도를 $\nu^\pi(s)$라고 할 때, 목적 함수는 다음과 같이 상태 분포 간의 차이를 줄이는 방향으로 정의된다.
$$\max_{D} \mathbb{E}_{s \sim \pi} [\log D(s)] + \mathbb{E}_{s \sim \pi^E} [\log(1 - D(s))]$$

여기서 $D_\omega(s)$는 입력된 상태가 전문가의 것인지 생성자의 것인지 판별하는 Discriminator이며, Generator는 이 Discriminator를 속이는 방향으로 학습되어 전문가의 상태 분포를 따라가게 된다.

### 2. Action-Guided Regularization
상태 정보만으로는 학습 속도가 느릴 수 있으므로, 가용한 행동 정보를 활용하여 정책을 가이드한다. 본 논문은 전문가의 행동 $a^E$와 생성된 행동 $a \sim \pi(s^E)$ 사이의 상호 정보량(Mutual Information) $I(a^E; a)$을 최대화하는 방식을 채택한다.

상호 정보량의 직접적인 계산이 어려우므로, InfoGAIL의 아이디어를 빌려 변분 하한(variational lower bound) $L_I$를 도입한다.
$$L_I(\pi, Q) = \mathbb{E}_{a^E \sim \{\tau^E_a\}} [\log Q(a^E | a, s^E)] + H(a^E)$$
여기서 $Q_\psi(a^E | a, s^E)$는 생성된 행동 $a$를 통해 전문가의 행동 $a^E$를 예측하는 Guide 네트워크이다. 행동 데이터가 존재하는 상태에 대해서만 $Q_\psi$를 업데이트하며, 이는 Generator에게 추가적인 보상을 제공하는 역할을 한다.

### 3. 학습 절차 및 최종 목적 함수
최종 목적 함수는 정책 엔트로피 $H(\pi_\theta)$, 상호 정보량 하한 $L_I$, 그리고 상태 기반 적대적 손실의 합으로 정의된다.
$$\min_{\pi \in \Pi} [ -\lambda_1 H(\pi_\theta) - \lambda_2 L_I(\pi_\theta, Q_\psi) + \max_{D} \mathbb{E}_{s \sim \pi_\theta} \log D_\omega + \mathbb{E}_{s \sim \pi^E} \log(1 - D_\omega) ]$$

Generator가 받는 최종 보상 함수 $r(s, a)$는 다음과 같이 구성된다.
$$r(s^E, a) = \alpha D_\omega(s^E) + \beta Q(a^E | s, a)$$
여기서 $\alpha$는 상태 일치에 대한 가중치, $\beta$는 행동 가이드에 대한 가중치이며, $\beta$는 행동 정보의 불완전성 비율 $\eta$에 따라 $\beta = 1 - \eta$로 설정된다.

## 📊 Results

### 실험 설정
- **데이터셋/환경**: CartPole, Hopper, Walker2d, Humanoid (이산/연속 공간 및 저/고차원 제어 포함).
- **비교 대상**: TRPO(정답 보상 사용), GAIL(완전한 시연), State-GAIL(상태만 사용).
- **측정 지표**: 누적 보상(Empirical Return).
- **변수**: 행동 불완전성 비율 $\eta \in \{0\%, 25\%, 50\%, 75\%\}$.

### 주요 결과
1. **불완전한 데이터에서의 성능**: AGAIL은 행동 정보가 75%까지 누락된 상황에서도 TRPO 및 GAIL과 대등한 성능을 보였다. 특히 Humanoid와 같은 고차원 작업에서 State-GAIL보다 월등한 성능 향상을 보여, Guide 네트워크의 중요성을 입증하였다.
2. **불완전성 비율의 영향**: 흥미롭게도 일부 환경(Hopper, Walker2d, Humanoid)에서는 행동 정보가 완전히 제공된 $\text{AGAIL}_{.00}$보다 일부가 누락된 $\text{AGAIL}_{.75}$의 성능이 더 높게 나타났다. 이는 전문가의 시연 데이터에 노이즈나 불안정한 행동이 포함되어 있을 경우, 이를 모두 학습하는 것보다 일부를 배제하는 것이 더 효과적일 수 있음을 시사한다.
3. **강건성(Robustness)**: 행동 누락 비율이 변하더라도 AGAIL은 GAIL보다 일관되게 높은 보상을 얻었으며, 특히 Hopper 환경에서는 GAIL을 압도하는 결과를 보였다.

## 🧠 Insights & Discussion

본 논문은 모방 학습에서 행동 정보가 반드시 완전할 필요가 없으며, 상태 분포 일치와 행동 가이드라는 두 가지 경로를 분리하여 학습하는 것이 효율적임을 보여주었다.

**강점 및 해석**:
- **정보의 효율적 활용**: 행동 정보를 직접적인 타겟(label)으로 쓰는 BC 방식과 달리, 상호 정보량 최대화를 통한 '가이드'로 활용함으로써 데이터 누락에 유연하게 대응하였다.
- **데이터 품질 문제 제기**: 실험 결과에서 나타난 $\text{AGAIL}_{.75}$의 높은 성능은, 시연 데이터의 '양'보다 '질'이 중요하며, 불완전한 데이터가 때로는 정규화(regularization) 효과를 주어 과적합을 방지하고 더 일반적인 정책을 학습하게 할 수 있다는 가능성을 제시한다.

**한계 및 논의사항**:
- **보상 함수 가정**: 본 방법론은 보상 함수 $r$이 주로 상태 $s$에 의존한다는 가정하에 성립한다. 만약 보상 함수가 상태-행동의 복잡한 상호작용에 강하게 의존하는 도메인이라면, 상태 분포 일치만으로는 한계가 있을 수 있다.
- **하이퍼파라미터 민감도**: $\alpha$와 $\beta$의 설정이 성능에 큰 영향을 미칠 수 있으며, 특히 불완전성 비율 $\eta$에 따른 $\beta$의 동적 설정 방식에 대한 추가적인 분석이 필요해 보인다.

## 📌 TL;DR

본 논문은 전문가의 행동 정보가 일부 누락된 '불완전한 시연' 상황에서도 정책을 학습할 수 있는 **AGAIL** 알고리즘을 제안한다. 상태 분포를 일치시키는 적대적 학습(Discriminator)과 가용한 행동 정보를 통해 탐색을 돕는 가이드 학습(Guide)을 결합하여, 행동 정보가 부족한 상황에서도 기존의 완전한 시연 기반 알고리즘(GAIL)과 대등하거나 오히려 더 우수한 성능을 달성하였다. 이는 실제 환경에서 관측 불가능한 행동 정보 문제를 해결하고, 데이터의 노이즈에 강건한 모방 학습을 가능하게 하여 로봇 제어 및 자율 주행 연구에 기여할 가능성이 크다.