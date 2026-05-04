# Imitation learning with Sinkhorn Distances

Georgios Papagiannis and Yunpeng Li (2022)

## 🧩 Problem to Solve

본 논문은 전문가의 시연(Expert Demonstration)으로부터 에이전트의 행동 정책을 학습하는 Imitation Learning (IL)에서 전문가와 학습자의 상태-행동 분포(State-Action Distribution)를 어떻게 효과적으로 비교하고 일치시킬 것인가에 대한 문제를 다룬다.

강화학습에서 보상 함수(Reward Function)를 직접 설계하는 것은 매우 까다로운 작업이며, 이를 해결하기 위해 등장한 IL 방법론들은 주로 전문가와 학습자의 Occupancy Measure(점유 측정치) 간의 거리나 발산(Divergence)을 최소화하는 방향으로 발전해 왔다. 그러나 기존의 f-divergence 기반 방법론(예: GAIL)은 GAN 기반 학습의 고유한 불안정성과 Mode-covering 동작과 같은 한계가 있으며, Optimal Transport (OT) 기반 방법론(예: WAIL, PWIL)은 계산 복잡도가 높거나 Lipschitz 조건 강제와 같은 근사치에 의존해야 하므로 이론적 성질을 온전히 유지하기 어렵다는 문제가 있다.

따라서 본 연구의 목표는 Sinkhorn Distance를 활용하여 계산 효율성을 확보하면서도 OT의 이론적 이점을 유지하는 새로운 IL 프레임워크인 Sinkhorn Imitation Learning (SIL)을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 전문가와 학습자의 Occupancy Measure 간의 거리를 측정하기 위해 엔트로피 정규화가 적용된 Optimal Transport 메트릭인 Sinkhorn Distance를 도입하는 것이다.

가장 중심적인 설계는 단순히 고정된 거리 척도를 사용하는 것이 아니라, 적대적으로 학습되는 특징 공간(Adversarially learned feature space) 내에서 Cosine Distance를 기반으로 하는 Transport Cost를 정의한 점이다. 이를 통해 Critic 네트워크가 더욱 변별력 있는 신호를 제공하게 하며, 결과적으로 학습자가 전문가의 분포를 더 정확하게 추종하도록 유도한다. 또한, 제안된 방법론이 기존의 Regularized Maximum Entropy IRL 프레임워크의 변형으로 해석될 수 있음을 이론적으로 증명하였다.

## 📎 Related Works

논문에서는 다음과 같은 기존 접근 방식과 그 한계를 설명한다.

- **Behavioral Cloning (BC):** 전문가의 행동을 지도 학습(Supervised Learning) 문제로 변환하여 모방한다. 구현이 간단하지만, Covariate Shift 문제로 인해 샘플 효율성이 낮고 일반화 성능이 떨어진다.
- **Inverse Reinforcement Learning (IRL):** 전문가의 행동을 설명하는 보상 함수를 먼저 추론한 뒤 정책을 학습한다. 하지만 동일한 행동을 생성하는 보상 함수가 여러 개 존재할 수 있는 Ill-posed 문제이며, 보상 추론과 정책 학습의 반복적인 최적화로 인해 계산 비용이 높다.
- **Adversarial IL (GAIL):** 전문가와 학습자의 Occupancy Measure 사이의 Jensen-Shannon (JS) Divergence를 최소화한다. 하지만 f-divergence 기반의 학습은 훈련 불안정성과 특정 모드에 치우치는 경향이 있다.
- **OT-based IL (WAIL, PWIL):** Wasserstein Distance를 최소화하여 수치적 안정성과 불연속적인 분포에 대한 강건함을 꾀한다. 그러나 WAIL은 Dual form의 계산 불가능성으로 인해 Lipschitz 조건을 강제하는 근사법을 사용하며, PWIL은 탐욕적 결합(Greedy coupling) 전략을 사용하므로 최적 전송 맵(Optimal transport map)을 보장하지 못해 이론적 근거가 부족하다.

## 🛠️ Methodology

### 전체 시스템 구조
SIL은 전문가의 Occupancy Measure $\rho^E$와 학습자의 Occupancy Measure $\rho^\pi$ 사이의 Sinkhorn Distance를 최소화하는 것을 목표로 한다. 전체 파이프라인은 적대적으로 학습되는 Critic 네트워크와 이를 통해 유도된 보상 프록시(Reward Proxy)를 사용하여 정책을 업데이트하는 반복적인 과정으로 구성된다.

### 주요 구성 요소 및 상세 설명

**1. Sinkhorn Distance**
두 확률 분포 $p, q$ 사이의 Sinkhorn Distance $W_s^\beta$는 다음과 같이 정의된다.
$$W_s^\beta(p, q)_c = \inf_{\zeta \in \Omega_\beta(p, q)} \mathbb{E}_{x, y \sim \zeta} [c(x, y)]$$
여기서 $\Omega_\beta(p, q)$는 엔트로피 제약 조건이 추가된 결합 확률 분포의 집합이며, $c(x, y)$는 샘플 $x$를 $y$로 이동시키는 비용(Transport Cost)이다.

**2. Adversarial Reward Proxy**
학습자의 궤적 $\tau^\pi$에 포함된 각 샘플 $(s, a)^\pi$에 대해, Sinkhorn 알고리즘을 통해 계산된 최적 전송 계획(Optimal transport plan) $\zeta^\beta$를 사용하여 다음과 같은 보상 프록시 $v^c$를 정의한다.
$$v^c((s, a)^\pi) := -\sum_{(s, a)^{\pi_E} \in \tau^E} c((s, a)^\pi, (s, a)^{\pi_E}) \zeta^\beta((s, a)^\pi, (s, a)^{\pi_E})$$

단순한 거리 척도는 변별력이 떨어지므로, 본 논문에서는 신경망 $f_w$를 이용해 상태-행동 쌍을 특징 공간으로 매핑하고, 그 공간에서의 Cosine Distance를 비용 함수 $c^w$로 사용한다.
$$c^w((s, a)^\pi, (s, a)^{\pi_E}) = 1 - \frac{f_w((s, a)^\pi) \cdot f_w((s, a)^{\pi_E})}{\|f_w((s, a)^\pi)\|_2 \|f_w((s, a)^{\pi_E})\|_2}$$

**3. 학습 절차 (Algorithm 1)**
SIL은 다음과 같은 Minimax 최적화 문제를 해결한다.
$$\arg \min_\pi \max_w W_s^\beta(\rho^\pi, \rho^E)_{c^w}$$

학습 과정은 다음과 같은 순서로 진행된다.
1. 학습자 정책 $\pi_{\theta_k}$를 통해 궤적 $\tau^{\pi_{\theta_k}}$를 샘플링한다.
2. 학습자의 궤적과 전문가의 궤적을 무작위로 짝지어 Sinkhorn 알고리즘을 통해 Sinkhorn Distance를 계산한다.
3. **Critic 업데이트:** Sinkhorn Distance $W_s^\beta$를 최대화하도록 파라미터 $w$를 경사 상승법(Gradient Ascent)으로 업데이트한다.
4. **Policy 업데이트:** 계산된 보상 프록시 $v_{c^w}$를 보상으로 사용하여 Trust Region Policy Optimization (TRPO) 알고리즘을 통해 정책 $\theta$를 업데이트한다.

## 📊 Results

### 실험 설정
- **환경:** MuJoCo 시뮬레이터의 5가지 환경 (Hopper-v2, HalfCheetah-v2, Walker2d-v2, Ant-v2, Humanoid-v2).
- **비교 대상 (Baselines):** Behavioral Cloning (BC), GAIL, AIRL.
- **측정 지표:**
    - **True Reward:** 환경에서 제공하는 실제 보상 값.
    - **Sinkhorn Distance:** 평가 시에는 고정된 Cosine Distance 비용 함수를 사용하여 전문가 분포와의 유사도를 측정한다 (값이 작을수록 우수).
- **데이터셋:** 전문가 궤적의 수를 $\{2, 4, 8, 16, 32\}$개로 다양하게 설정하여 샘플 효율성을 테스트하였다.

### 주요 결과
- **Sinkhorn Metric 관점:** SIL은 대부분의 환경에서 GAIL, AIRL과 대등하거나 더 우수한 성능을 보였다. 특히 HalfCheetah-v2와 Ant-v2에서 강점을 보였으며, Humanoid-v2 환경에서는 적은 수의 전문가 궤적(8, 16개)만으로도 타 방법론보다 월등히 높은 샘플 효율성을 입증하였다.
- **Reward Metric 관점:** 모든 적대적 IL 알고리즘이 전문가 성능에 근접한 결과를 냈으나, 일부 환경(Ant-v2)에서는 AIRL이 SIL보다 높은 보상을 기록하기도 했다. 그러나 저자들은 보상 값보다 Sinkhorn Distance가 분포 일치라는 IL의 본질적 목표를 더 정확히 반영하는 지표라고 주장한다.
- **Ablation Study:** 적대적 Critic 없이 고정된 Cosine Distance 비용 함수만을 사용하여 학습했을 때, 훈련 과정은 더 안정적이었으나 최종 성능(보상 및 Sinkhorn Distance 모두)은 크게 저하되었다. 이는 Adversarial training을 통한 변별력 있는 비용 함수 학습이 SIL의 핵심 요소임을 시사한다.

## 🧠 Insights & Discussion

본 논문은 Sinkhorn Distance를 IL에 도입함으로써 OT의 이론적 장점과 계산적 효율성을 동시에 잡았다. 특히, 적대적 학습을 통해 특징 공간을 구축하고 그 안에서 거리 척도를 정의함으로써, 단순한 분포 매칭을 넘어 매우 강력한 변별 신호를 학습자에게 제공할 수 있음을 보여주었다.

하지만 다음과 같은 한계점과 논의 사항이 존재한다.
첫째, Sinkhorn Distance를 계산하기 위해서는 전체 궤적(Complete trajectory)이 필요하므로, 본 방법론은 본질적으로 On-policy 방식에 국한된다. 이는 Off-policy 방법론들에 비해 환경 상호작용 횟수가 더 많이 필요함을 의미하며, 향후 Off-policy RL 알고리즘과의 결합이 필요하다.
둘째, Critic 네트워크의 구조가 성능에 미치는 영향에 대한 분석이 부족하다. 상태-행동 공간의 차원에 따라 최적의 네트워크 아키텍처가 다를 수 있으며, 이에 대한 추가 연구가 요구된다.
셋째, 현재의 전송 비용 계산은 시간적 순서를 고려하지 않은 단순 샘플 간의 매칭이다. 궤적의 시간적 의존성(Temporal dependence)을 반영한 OT 결합 방식을 도입한다면 샘플 효율성과 일반화 성능을 더욱 향상시킬 수 있을 것으로 보인다.

## 📌 TL;DR

본 논문은 전문가와 학습자의 상태-행동 분포를 일치시키기 위해 **Sinkhorn Distance**를 최소화하는 **SIL (Sinkhorn Imitation Learning)** 프레임워크를 제안한다. 적대적으로 학습되는 특징 공간상의 Cosine Distance를 Transport Cost로 활용하여 변별력 있는 보상 신호를 생성하며, MuJoCo 실험을 통해 특히 복잡한 환경(Humanoid)에서 높은 샘플 효율성과 분포 일치 성능을 입증하였다. 이 연구는 OT 기반의 IL이 실질적으로 계산 가능하며 강력한 도구가 될 수 있음을 보여주었으며, 향후 Off-policy 확장 및 시간적 의존성 반영 연구의 토대를 마련하였다.