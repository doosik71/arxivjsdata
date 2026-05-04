# RAIL: Risk-Averse Imitation Learning

Anirban Santara, Abhishek Naik, Balaraman Ravindran, Dipankar Das, Dheevatsa Mudigere, Sasikanth Avancha, Bharat Kaul (2017)

## 🧩 Problem to Solve

본 논문은 모방 학습(Imitation Learning), 특히 Generative Adversarial Imitation Learning (GAIL) 알고리즘이 학습한 정책의 신뢰성 문제를 해결하고자 한다. GAIL은 전문가의 궤적(trajectory)을 모방하여 최적의 정책을 학습하는 최신 알고리즘이지만, 평균적인 성능 향상에만 집중하는 경향이 있다.

연구진은 GAIL 에이전트가 생성하는 궤적 비용(trajectory-cost)의 분포를 분석한 결과, 전문가에 비해 분포의 꼬리 부분(tail)이 더 두꺼운 'heavy-tailed' 특성을 보인다는 것을 발견하였다. 이는 확률은 낮지만 발생 시 치명적인 실패로 이어질 수 있는 고비용 궤적, 즉 '꼬리 위험(tail risk)'이 GAIL 에이전트에게서 더 빈번하게 나타남을 의미한다. 이러한 특성은 자율 주행이나 로봇 수술과 같이 안전성과 신뢰성이 필수적인 위험 민감형(risk-sensitive) 응용 분야에 GAIL을 그대로 적용하기 어렵게 만든다. 따라서 본 논문의 목표는 GAIL 프레임워크 내에서 꼬리 위험을 최소화하여 더욱 신뢰할 수 있는 정책을 학습하는 RAIL(Risk-Averse Imitation Learning) 알고리즘을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 GAIL의 목적 함수에 꼬리 위험을 정량화할 수 있는 지표인 Conditional-Value-at-Risk (CVaR) 최적화 항을 추가하는 것이다. 단순한 평균 비용 최소화가 아니라, 최악의 경우에 해당하는 비용들의 기댓값을 직접적으로 최소화함으로써 정책의 안정성을 높이고 치명적인 실패 가능성을 줄이는 설계 구조를 가진다.

## 📎 Related Works

논문에서는 모방 학습을 크게 두 가지 범주로 구분하여 설명한다.

1. **Behavioral Cloning (BC):** 전문가의 상태-행동 쌍을 지도 학습 방식으로 학습하는 방법이다. 단순하지만 데이터가 제한적일 때 성능이 떨어지며, 특히 예측된 행동이 미래 관측치에 영향을 주는 순차적 의사결정 문제에서 covariate shift로 인한 복합 오류(compounding error) 문제가 발생한다.
2. **Inverse Reinforcement Learning (IRL):** 전문가의 행동을 통해 보상 함수(reward function)를 먼저 추론하고, 이를 바탕으로 강화학습을 통해 정책을 학습하는 방법이다. 복합 오류 문제는 해결하지만, 보상 함수를 학습한 후 다시 정책을 학습해야 하는 간접적인 과정으로 인해 계산 비용이 매우 높고 대규모 환경에서의 확장성(scalability) 문제가 존재한다.

**GAIL**은 이러한 IRL의 과정을 단일 최적화 문제로 통합하여 확장성을 확보하고 복합 오류 문제를 해결한 state-of-the-art 알고리즘이다. 그러나 기존의 GAIL을 포함한 대부분의 모방 학습 연구들은 평균적인 성능에만 집중하여 최악의 경우 발생하는 꼬리 위험을 간과했다는 한계가 있다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

RAIL은 GAIL의 적대적 학습 구조를 유지하면서, 학습 목표에 CVaR를 통합한다. 생성자(Generator)인 정책 $\pi$는 전문가와 유사한 행동을 생성하려 하고, 판별자(Discriminator) $D$는 전문가와 에이전트의 궤적을 구분하려 한다. 여기에 더해, 궤적 비용의 최악의 경우를 최소화하는 CVaR 항이 추가되어 정책의 안정성을 강제한다.

### 2. Conditional-Value-at-Risk (CVaR)

꼬리 위험을 측정하기 위해 사용된 $\text{CVaR}_\alpha$는 특정 신뢰 수준 $\alpha$에서 $\text{VaR}_\alpha$(Value-at-Risk)보다 높은 비용을 가진 궤적들의 조건부 기댓값이다. 수학적으로 다음과 같이 정의된다.

먼저, $\text{VaR}_\alpha(Z)$는 확률 $\alpha$로 $Z$가 초과하지 않는 최소값 $z$이다.
$$\text{VaR}_\alpha(Z) = \min(z | P(Z \le z) \ge \alpha)$$

$\text{CVaR}_\alpha(Z)$는 $\text{VaR}_\alpha(Z)$ 이상의 손실에 대한 기댓값이며, 다음과 같은 최적화 형태로 표현 가능하다.
$$\text{CVaR}_\alpha(Z) = \min_{\nu \in \mathbb{R}} H_\alpha(Z, \nu)$$
여기서 $H_\alpha(Z, \nu)$는 다음과 같다.
$$H_\alpha(Z, \nu) = \nu + \frac{1}{1-\alpha} E[(Z - \nu)^+]$$
($\text{단, } (x)^+ = \max(x, 0)$)

### 3. RAIL 목적 함수

RAIL은 GAIL의 목적 함수에 $\text{CVaR}$ 항을 가중치 $\lambda_{\text{CVaR}}$와 함께 추가하여 다음과 같이 정의한다.
$$\min_{\pi, \nu} \max_{D \in (0,1)^{S \times A}} J = \min_{\pi, \nu} \max_{D \in (0,1)^{S \times A}} \left\{ -H(\pi) + E_\pi[\log(D(s,a))] + E_{\pi_E}[\log(1-D(s,a))] + \lambda_{\text{CVaR}} H_\alpha(R^\pi(\xi|c(D)), \nu) \right\}$$

- $H(\pi)$: 정책의 엔트로피로, 탐색을 촉진하고 편향을 방지하는 정규화 항이다.
- $R^\pi(\xi|c(D))$: 판별자 $D$에 의해 정의된 비용 함수 $c(D)$에 따른 궤적 $\xi$의 총 비용이다.
- $\nu$: $\text{CVaR}$ 최적화를 위해 함께 학습되는 파라미터이다.

### 4. 학습 절차

RAIL은 다음과 같은 반복적인 업데이트 과정을 거친다.

1. **판별자 업데이트:** Adam 알고리즘을 사용하여 목적 함수 $J$에 대해 경사 상승법(gradient ascent)을 수행한다.
2. **정책 업데이트:** Trust Region Policy Optimization (TRPO)를 사용하여 KL-제약 조건 하에서 정책 파라미터 $\theta$에 대해 경사 하강법을 수행한다.
3. **$\nu$ 업데이트:** 배치 경사 하강법(batch gradient descent)을 통해 $\text{CVaR}$ 파라미터 $\nu$를 최적화한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋 및 작업:** MuJoCo 물리 시뮬레이터 기반의 OpenAI Gym 연속 제어 작업 5종(Reacher-v1, HalfCheetah-v1, Hopper-v1, Walker-v1, Humanoid-v1)을 사용하였다.
- **기준선:** 기존의 GAIL 및 전문가(Expert) 정책과 비교하였다.
- **지표:** $\text{VaR}_{0.9}$, $\text{CVaR}_{0.9}$, 그리고 전문가 대비 상대적 꼬리 위험을 나타내는 Percentage Relative Tail Risk와 Gain in Reliability (GR)를 사용하였다.

### 2. 주요 결과

- **꼬리 위험 감소:** 모든 실험 작업에서 RAIL은 GAIL보다 낮은 $\text{VaR}$와 $\text{CVaR}$ 값을 기록하였다. 특히 Table 2에서 확인할 수 있듯이, 많은 경우에서 GAIL보다 훨씬 낮은 비용을 보였으며 일부 작업에서는 전문가의 수준에 근접하거나 오히려 능가하는 결과를 보였다.
- **신뢰성 이득(GR):** Table 3의 GR-CVaR 지표를 통해 RAIL이 GAIL 대비 꼬리 위험을 유의미하게 낮추었음을 확인하였다.
- **수렴 속도:** Figure 2의 평균 궤적 비용 수렴 그래프를 보면, RAIL은 GAIL과 거의 동일하거나 때로는 더 빠른 수렴 속도를 보였다. 이는 위험 회피 항을 추가했음에도 불구하고 학습 효율성이 저하되지 않았음을 의미한다.

## 🧠 Insights & Discussion

본 연구의 결과는 다음과 같은 통찰을 제공한다.

첫째, RAIL은 GAIL이 생성하는 궤적 비용 분포가 반드시 heavy-tailed가 아닌 환경(예: Reacher-v1)에서도 효과적이다. $\text{CVaR}$ 최소화는 분포가 정규 분포에 가까울 때 평균과 표준편차를 동시에 낮추는 효과를 주기 때문에, 결과적으로 분포를 평균 주변으로 더 응집시켜 전반적인 안정성을 높인다.

둘째, RAIL은 GAIL의 핵심 장점인 확장성(scalability)을 그대로 유지한다. 고차원 상태-행동 공간을 가진 Humanoid-v1 작업에서도 성공적으로 동작함을 통해, 복잡한 환경에서도 위험 민감형 모방 학습이 가능함을 입증하였다.

셋째, 한계점으로 본 알고리즘은 학습 과정에서 에이전트가 환경과 상호작용하여 궤적을 샘플링해야 한다는 점이 있다. 이를 해결하기 위해서는 시뮬레이터에서 학습하고 실제 환경에 적용하는 '제3자 모방 학습(third person imitation learning)' 프레임워크로의 확장이 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 GAIL의 치명적인 단점인 '꼬리 위험(tail risk)', 즉 낮은 확률로 발생하는 대형 사고의 가능성을 줄이기 위해 **Conditional-Value-at-Risk (CVaR)** 최적화를 통합한 **RAIL** 알고리즘을 제안하였다. 실험 결과, RAIL은 기존 GAIL의 확장성과 수렴 속도를 유지하면서도 최악의 경우의 비용을 유의미하게 낮추어 정책의 신뢰성을 크게 향상시켰다. 이 연구는 자율 주행이나 의료 로봇과 같이 안전이 최우선인 분야에서 모방 학습을 적용하기 위한 중요한 방법론적 토대를 제공한다.
