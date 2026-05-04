# Risk-Sensitive Generative Adversarial Imitation Learning

Jonathan Lacotte, Mohammad Ghavamzadeh, Yinlam Chow, Marco Pavone (2018)

## 🧩 Problem to Solve

본 논문은 모방 학습(Imitation Learning) 환경에서 에이전트가 단순히 전문가의 평균적인 성능을 따라가는 것을 넘어, **위험 프로필(Risk Profile)** 관점에서도 전문가만큼 혹은 그 이상으로 수행하도록 만드는 문제를 다룬다.

일반적인 모방 학습, 특히 GAIL(Generative Adversarial Imitation Learning)과 같은 방식은 위험 중립적(Risk-neutral)인 관점에서 기대 비용(Expected cost)을 최소화하는 데 집중한다. 하지만 실제 시스템에서는 확률적 특성으로 인해 발생하는 비용의 변동성이나 최악의 상황(Tail events)에 대한 제어가 필수적이다. 따라서 본 연구의 목표는 전문가의 $\text{CVaR}$(Conditional Value-at-Risk)를 고려한 위험 민감적 모방 학습 프레임워크를 구축하여, 에이전트가 평균 성능과 위험 지표 모두에서 전문가의 수준에 도달하게 하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 에이전트와 전문가의 단순한 점유 측정치(Occupancy measure)를 일치시키는 것이 아니라, **위험 프로필을 인코딩하는 '왜곡된 점유 측정치(Distorted occupancy measures)의 집합'을 일치시키는 것**이다.

이를 위해 저자들은 $\text{Mean} + \lambda \text{CVaR}_\alpha$ 형태의 일관된 위험 척도(Coherent risk measure)를 정의하고, 이를 기반으로 한 RS-GAIL(Risk-Sensitive GAIL) 최적화 문제를 제안한다. 특히, 거리 측정 방식에 따라 Jensen-Shannon (JS) 발산과 Wasserstein 거리를 사용하는 두 가지 변형 알고리즘(JS-RS-GAIL, W-RS-GAIL)을 도출하여 위험 민감도에 따른 정책 학습 가능성을 제시하였다.

## 📎 Related Works

기존의 모방 학습 접근 방식은 크게 세 가지로 나뉜다. 첫째, Behavioral Cloning은 단순하지만 많은 데이터가 필요하며, 둘째, Inverse Reinforcement Learning(IRL)은 비용 함수를 먼저 찾고 RL을 수행하므로 계산 비용이 매우 높다. 셋째, GAIL은 IRL 단계를 생략하고 점유 측정치 매칭을 통해 직접 정책을 학습하여 효율성을 높였다.

위험 민감도 측면에서는 RAIL(Risk-Averse Imitation Learning)이 제안된 바 있다. 그러나 본 논문의 분석에 따르면, RAIL은 에이전트의 왜곡된 점유 측정치를 전문가의 **평균** 점유 측정치와 매칭할 뿐, 전문가가 가진 위험 프로필(위험 성향) 자체를 고려하지 않는다는 한계가 있다. 본 연구는 전문가의 위험 프로필까지 함께 매칭함으로써 이 차별점을 갖는다.

## 🛠️ Methodology

### 1. 위험 척도 정의

본 논문에서는 위험을 측정하기 위해 $\text{VaR}$(Value-at-Risk)와 $\text{CVaR}$를 사용한다. 신뢰 수준 $\alpha \in (0,1]$에 대해 $\text{CVaR}$는 다음과 같이 정의된다.
$$\rho_\alpha[C^\pi] = \inf_{\nu \in \mathbb{R}} \left\{ \nu + \frac{1}{\alpha} \mathbb{E}[(C^\pi - \nu)^+] \right\}$$
여기서 $C^\pi$는 정책 $\pi$ 하에서의 손실 랜덤 변수이며, $x^+ = \max(x, 0)$이다. 또한, 평균 성능과 위험의 절충안을 위해 다음과 같은 통합 위험 척도 $\rho_\lambda^\alpha$를 도입한다.
$$\rho_\lambda^\alpha[C^\pi] = \frac{\mathbb{E}[C^\pi] + \lambda \rho_\alpha[C^\pi]}{1 + \lambda}$$

### 2. RS-GAIL 최적화 문제

에이전트의 목표는 전문가의 비용 함수 $c$를 모르는 상태에서 다음 문제를 해결하는 것이다.
$$\min_\pi \mathbb{E}[C^\pi], \quad \text{s.t. } \rho_\alpha[C^\pi] \le \rho_\alpha[C^\pi_E]$$
이를 라그랑주 완화(Lagrangian relaxation)와 최대 인과 엔트로피(Maximum Causal Entropy) 프레임워크에 적용하여 다음과 같은 RS-GAIL 목적 함수를 도출한다.
$$\sup_{\lambda \ge 0} \min_\pi -H(\pi) + L_\lambda(\pi, \pi^E)$$
여기서 $L_\lambda(\pi, \pi^E)$는 다음과 같이 정의된다.
$$L_\lambda(\pi, \pi^E) = \sup_{f \in \mathcal{C}} (1 + \lambda) (\rho_\lambda^\alpha[C^\pi_f] - \rho_\lambda^\alpha[C^{\pi_E}_f]) - \psi(f)$$
$\psi(f)$는 비용 함수에 대한 볼록 정규화 항(Convex regularizer)이다.

### 3. 변형 알고리즘: JS-RS-GAIL 및 W-RS-GAIL

저자들은 정규화 항 $\psi(f)$의 선택에 따라 두 가지 버전을 제안한다.

* **JS-RS-GAIL**: JS 발산을 사용하여 왜곡된 점유 측정치 집합 $D_\pi^\xi$와 $D_{\pi_E}^\xi$를 매칭한다.
    $$(1 + \lambda) \sup_{f: S \times A \to (0,1)} \rho_\lambda^\alpha[F^\pi_{1,f}] - \rho_\lambda^\alpha[-F^{\pi_E}_{2,f}]$$
* **W-RS-GAIL**: Wasserstein 거리를 사용하여 두 집합을 매칭한다.
    $$(1 + \lambda) \sup_{f \in \mathcal{F}_1} \rho_\lambda^\alpha[C^\pi_f] - \rho_\lambda^\alpha[C^{\pi_E}_f]$$
    여기서 $\mathcal{F}_1$은 1-Lipschitz 함수 집합이다.

### 4. 학습 절차

알고리즘은 TRPO(Trust Region Policy Optimization)를 기반으로 하며, 다음 과정을 반복한다.

1. 현재 정책 $\pi_\theta$를 통해 궤적을 생성한다.
2. $\text{VaR}$ 값을 추정하고, Adam 옵티마이저를 사용하여 비용 함수(Discriminator) 파라미터 $w$를 업데이트(Gradient Ascent)한다.
3. KL-제한(KL-constrained) 하에서 정책 파라미터 $\theta$를 업데이트(Gradient Descent)하여 목적 함수를 최소화한다.

## 📊 Results

### 1. 실험 환경 및 설정

* **데이터셋/태스크**: OpenAI classical control (CartPole, Pendulum) 및 MuJoCo (Hopper, Walker).
* **스토캐스틱성 주입**: 원래 결정론적인 환경에 액션 노이즈나 비용 함수 노이즈를 추가하여 위험 민감도 평가가 가능하도록 설정하였다.
* **기준선(Baselines)**: GAIL, RAIL.
* **지표**: 평균(Mean), $\text{VaR}_\alpha$, $\text{CVaR}_\alpha$, 그리고 최종 목표인 $\rho_\lambda^\alpha$.

### 2. 정량적 결과

* **JS-RS-GAIL의 우수성**: 모든 태스크에서 JS-RS-GAIL이 $\rho_\lambda^\alpha$ 지표에서 GAIL과 RAIL보다 뛰어난 성능을 보였다. 특히 상위 10개 정책을 평균 낸 결과에서 통계적으로 유의미한 우위가 나타났다.
* **RAIL과의 비교**: RAIL은 평균 성능은 어느 정도 따라가지만, 위험 지표($\text{CVaR}$) 관점에서는 RS-GAIL보다 성능이 낮았다.
* **W-RS-GAIL의 특성**: W-RS-GAIL은 W-GAIL보다는 우수했으나, JS 기반 알고리즘보다는 전반적으로 낮은 성능을 보였다. 이는 작은 네트워크 구조에서 Lipschitz 연속성을 유지하기 위한 Weight clipping이 표현력을 제한했기 때문으로 분석된다.

## 🧠 Insights & Discussion

본 논문은 위험 민감적 모방 학습에서 단순히 에이전트의 위험을 줄이는 것이 아니라, **전문가의 위험 프로필 자체를 학습하는 것**이 얼마나 중요한지를 이론적, 실험적으로 증명하였다.

**강점 및 해석**:

* **이론적 기여**: 위험 척도를 점유 측정치 매칭 문제로 변환하여 GAIL의 프레임워크 내에서 해결할 수 있음을 수학적으로 보였다. 특히 Theorem 3를 통해 단순 점유 측정치 매칭만으로는 $\text{CVaR}$ 차이를 줄일 수 없음을 증명하여 본 연구의 필요성을 뒷받침하였다.
* **비판적 분석**: RAIL이 전문가의 위험 성향을 무시하고 평균 점유 측정치와 매칭한다는 점을 명확히 짚어내어, 진정한 의미의 '위험 민감적 모방'이 무엇인지 정의하였다.

**한계점**:

* **샘플 효율성**: $\text{CVaR}$와 같은 꼬리 부분(Tail)의 위험 척도를 학습하기 위해서는 일반적인 RL보다 훨씬 많은 샘플(전문가 궤적 100개 사용)이 필요하며, 이는 여전히 해결해야 할 과제로 남아있다.
* **네트워크 민감도**: Wasserstein 기반 방식이 복잡한 문제에서는 더 나을 수 있다는 가설을 제시했으나, 본 실험의 소규모 네트워크에서는 JS 방식보다 성능이 낮게 나왔다.

## 📌 TL;DR

본 논문은 에이전트가 전문가의 평균 성능뿐만 아니라 위험 성향($\text{CVaR}$)까지 모방하도록 하는 **RS-GAIL** 프레임워크를 제안한다. 핵심은 위험 프로필이 인코딩된 '왜곡된 점유 측정치'의 집합을 매칭하는 것이며, 이를 JS 발산과 Wasserstein 거리 기반의 두 알고리즘으로 구현하였다. 실험 결과, 제안된 방식이 기존 GAIL 및 RAIL보다 위험 민감적 지표($\rho_\lambda^\alpha$)에서 월등한 성능을 보였으며, 이는 고위험 상황 제어가 중요한 실제 로봇 제어나 자율 주행 시스템의 모방 학습에 중요한 기여를 할 수 있다.
