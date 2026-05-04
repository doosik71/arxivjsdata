# Quantile Filtered Imitation Learning

David Brandfonbrener, William F. Whitney, Rajesh Ranganath, Joan Bruna (2021)

## 🧩 Problem to Solve

본 논문은 Offline Reinforcement Learning(Offline RL)에서 발생하는 근본적인 한계인 **Bias-Variance Tradeoff**와 **Safety** 문제를 해결하고자 한다. Offline RL에서는 에이전트가 환경과 직접 상호작용하지 않고 고정된 데이터셋만을 사용하여 학습해야 하므로, 데이터셋의 범위를 벗어난 상태-행동 공간으로 외삽(extrapolation)할 경우 예측 불가능하고 위험한 행동을 선택할 가능성이 크다.

이러한 제약 조건 하에서 정책 개선(policy improvement)을 수행할 때, 학습된 정책을 데이터 생성 정책인 behavior policy $\beta$에 가깝게 유지하면 분산(variance)을 줄일 수 있지만, 최적 정책으로부터는 멀어지는 편향(bias)이 발생한다. 반대로 최적 정책에 가까워지려고 시도하면 데이터가 부족한 영역으로 진입하게 되어 분산이 급격히 증가하게 된다. 따라서 본 논문의 목표는 안전성을 보장하면서도 이 편향과 분산 사이의 균형을 효과적으로 조절할 수 있는 새로운 정책 개선 연산자를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Quantile Filtered Imitation Learning (QFIL)**이라는 새로운 정책 개선 연산자의 도입이다. QFIL의 핵심 직관은 데이터셋 전체를 모방하는 것이 아니라, 추정된 Q-value가 특정 임계값보다 높은 '우수한' 샘플들만을 필터링하여 모방 학습(Imitation Learning)을 수행하는 것이다.

여기서 단순히 고정된 상수를 사용하는 것이 아니라, behavior policy $\beta$가 유도하는 Q-value의 **Pushforward distribution**과 그 분포의 **Quantile(분위수)**을 정의하여 필터링 기준을 설정한다. 이를 통해 사용자는 분위수 $\tau$라는 하이퍼파라미터를 통해 데이터의 얼마나 많은 부분을 유지할지를 결정함으로써, 편향과 분산 사이의 트레이드오프를 직관적으로 제어할 수 있다.

## 📎 Related Works

기존의 많은 Offline RL 알고리즘(MARWIL, CRR, AWR, AWAC, BAIL, ABM 등)은 지수 가중치 기반의 모방 학습(Exponentially weighted imitation learning)을 사용한다. 이들은 다음과 같은 형태의 목적 함수를 가진다:
$$\pi_{k+1} = \arg \max_{\pi} \sum_{s,a \in D} \exp[\alpha(Q(s,a) - V(s))] \log \pi(a|s)$$
하지만 이러한 방식은 가중치 $\alpha$가 커질수록 수치적 불안정성이 발생하며, 실제 구현 시 가중치를 클리핑(clipping)해야 하는 문제가 있다. 결과적으로 가중치 기반 방식은 하드 임계값(hard threshold) 방식과 유사하게 작동하게 된다.

또한, %BC와 같은 방식은 전체 에피소드의 리턴(return)을 기준으로 필터링하지만, 이는 상태에 의존하지 않는 전역적인 기준을 사용한다는 한계가 있다. 반면 QFIL은 학습된 Q-함수를 사용하여 상태별로 최적화된 분위수 임계값을 설정함으로써 더욱 세밀한 필터링이 가능하다.

## 🛠️ Methodology

### 1. Pushforward Q distribution 및 Value Function Quantile 정의

QFIL의 핵심은 특정 상태 $s$에서 behavior policy $\beta$를 통해 샘플링된 행동들이 $Q$ 함수를 통과했을 때 형성하는 값들의 분포, 즉 **Pushforward distribution** $\mathcal{Q}_{],\beta}(\cdot|s)$를 정의하는 것이다.

- **Pushforward distribution**: $\beta$로부터 행동 $a$를 샘플링하여 $Q(s, a)$ 값으로 매핑했을 때의 분포이다. 이 분포의 누적 분포 함수(CDF)는 다음과 같이 정의된다:
$$P(X \le v) = P_{a \sim \beta|s}(Q(s, a) \le v)$$
- **Value function quantile**: 위 분포에서 $\tau$ 분위수에 해당하는 값 $V_{\tau,\beta}(s)$를 다음과 같이 정의한다:
$$V_{\tau,\beta}(s) := \sup \{ v \in \mathbb{R} \text{ s.t. } P_{a \sim \beta|s}(Q(s, a) \le v) \le \tau \}$$

### 2. QFIL 정책 개선 연산자

QFIL은 위에서 정의한 분위수 $V_{\tau,\beta}(s)$를 기준으로 데이터셋을 필터링한다.

1. **데이터 필터링**: 추정된 $\hat{Q}_k$ 값이 분위수 $\hat{V}_{\tau,\beta}$보다 큰 샘플들만 포함하는 부분 집합 $D_\tau$를 생성한다.
$$D_\tau = \{ (s, a) \in D \text{ s.t. } \hat{Q}_k(s, a) > \hat{V}_{\tau,\beta}(s) \}$$
2. **정책 업데이트**: 필터링된 데이터 $D_\tau$에 대해서만 표준 지도 학습(Supervised Learning) 기반의 모방 학습을 수행한다.
$$\pi_{k+1} = \arg \max_{\pi} \sum_{s,a \in D_\tau} \log \pi(a|s)$$

### 3. 학습 파이프라인 (OAMPI)

본 논문은 **Offline Approximate Modified Policy Iteration (OAMPI)**라는 일반적인 템플릿 내에서 QFIL을 동작시킨다.

- **Policy Evaluation**: 고정된 데이터셋 $D$를 사용하여 $\hat{Q}$를 추정한다. (SARSA 또는 DDPG 스타일의 Q-estimation 사용)
- **Policy Improvement**: 위에서 설명한 QFIL 연산자를 통해 $\pi$를 업데이트한다.
- **Quantile 추정**: $\hat{\beta}$에서 $M$개의 샘플을 뽑아 $\hat{Q}$에 통과시킨 후, 경험적 분위수(empirical quantile)를 계산하는 샘플링 기반 방식을 사용한다.

### 4. 이론적 분석

논문은 Proposition 1을 통해 $\tau$가 편향-분산 트레이드오프를 어떻게 조절하는지 증명한다.

- **편향(Bias)**: $\tau$가 커질수록 더 우수한 정책을 모방하게 되어 Wasserstein-1 거리가 증가하며, 이는 이론적으로 성능 향상 가능성을 높인다.
- **분산(Variance)**: $\tau$가 커질수록 필터링 후 남는 데이터의 양 $(1-\tau)N$이 줄어들어, 함수 근사 오차와 샘플링 오차가 증가한다. 구체적으로 분산 항은 $1/(1-\tau)^{3/2} + \tau$의 형태로 스케일링되어 $\tau$가 커질수록 오차가 커짐을 보여준다.

## 📊 Results

### 1. 합성 실험 (Synthetic Experiment)

Contextual Bandit 환경에서 다양한 데이터셋 크기와 $\tau$ 값에 따른 성능을 측정하였다. 실험 결과, 데이터셋이 작을 때는 낮은 $\tau$ (더 많은 데이터 사용, 낮은 분산)가 유리하고, 데이터셋이 충분할 때는 높은 $\tau$ (더 정밀한 필터링, 낮은 편향)가 유리함을 확인하여 $\tau$가 실제로 Bias-Variance Tradeoff를 제어함을 입증하였다.

### 2. D4RL MuJoCo 실험 (One-step)

- **설정**: 단 한 번의 정책 개선 단계($K=1$)만 수행하여 알고리즘의 순수 개선 능력을 측정하였다.
- **결과**: `halfcheetah-med`, `walker2d-med` 등 대부분의 작업에서 QFIL이 기존의 지수 가중치 방식(Exp-adv)보다 우수한 성능을 보였다. 이는 하드 필터링이 수치적으로 더 안정적이며 효과적임을 시사한다.

### 3. D4RL AntMaze 실험 (One-step & Iterative)

- **One-step**: 희소 보상(sparse reward) 환경인 AntMaze에서도 QFIL이 베이스라인 대비 유의미한 성능 향상을 보였다.
- **Iterative**: QFIL을 반복적인 정책 평가 및 개선 루프에 적용하였다. 결과적으로 CQL, IQL과 같은 SOTA(State-of-the-art) 알고리즘과 경쟁 가능한 수준의 성능을 달성하였다. 특히 Iterative Exp-Adv보다 훨씬 높은 성능을 기록하였다.

## 🧠 Insights & Discussion

### 강점 및 기여

1. **안전성 보장**: QFIL은 오직 데이터셋에 존재하는 행동만을 모방하므로, 데이터 분포 밖의 행동을 선택하여 발생할 수 있는 위험을 원천적으로 차단한다.
2. **최적화의 단순성**: 복잡한 Q-value 최적화 대신 표준 지도 학습(Supervised Learning)을 사용하므로, 기존의 딥러닝 최적화 도구와 트릭을 그대로 사용할 수 있으며 수치적으로 매우 안정적이다.
3. **직관적인 하이퍼파라미터**: $\tau$는 "데이터의 상위 몇 %를 사용할 것인가"라는 직관적인 의미를 가지므로, 실무자가 데이터 양에 따라 적절한 값을 설정하기 용이하다.

### 한계 및 논의

- **이론적 범위**: 본 논문의 이론적 보장은 One-step variant에 국한되어 있으며, Iterative 방식에 대한 엄밀한 이론적 분석은 향후 과제로 남겨두었다.
- **분포 변화(Distribution Shift)**: 이론적 분석에서 분포 변화에 대한 바운드가 다소 비관적으로 설정되었을 가능성이 있으며, 실제 환경에서의 정밀한 영향 분석이 필요하다.

## 📌 TL;DR

본 논문은 Offline RL에서 안전하게 정책을 개선하기 위해, behavior policy의 Q-value 분포에서 특정 분위수($\tau$) 이상의 샘플만을 추출하여 모방 학습을 수행하는 **QFIL (Quantile Filtered Imitation Learning)**을 제안한다. 이 방법은 $\tau$를 통해 편향과 분산의 트레이드오프를 직관적으로 조절할 수 있게 하며, 단순한 지도 학습 구조를 통해 최적화 안정성과 안전성을 동시에 확보한다. 실험적으로 D4RL 벤치마크에서 SOTA 수준의 성능을 보였으며, 특히 하드 필터링 방식이 기존의 소프트 가중치 방식보다 우수함을 입증하였다.
