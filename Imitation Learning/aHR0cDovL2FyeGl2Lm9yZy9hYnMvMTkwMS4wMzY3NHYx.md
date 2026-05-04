# On the Global Convergence of Imitation Learning: A Case for Linear Quadratic Regulator

Qi Cai, Mingyi Hong, Yongxin Chen, Zhaoran Wang (2019)

## 🧩 Problem to Solve

본 논문은 모방 학습(Imitation Learning), 특히 생성적 적대 모방 학습(Generative Adversarial Imitation Learning, GAIL)에서 발생하는 학습의 불안정성 문제를 해결하고자 한다.

기존의 Behavioral Cloning은 시간에 따른 예측 오차의 누적 문제로 인해 장기적으로 전문가의 궤적에서 벗어나는 한계가 있다. 이를 해결하기 위해 Inverse Reinforcement Learning(IRL)과 GAIL은 보상 함수(Reward Function)와 최적 정책(Optimal Policy)을 동시에 학습하는 Minimax 최적화 방식을 취한다. 그러나 GAIL은 비볼록-비오목(non-convex-concave) 기하학적 구조로 인해 최적화 과정이 매우 불안정하며, 특히 정책 최적화와 보상 함수 최적화를 완전히 해결하지 않고 교차 업데이트하는 방식은 수렴성을 보장하기 어렵다.

따라서 본 연구의 목표는 선형 이차 조절기(Linear Quadratic Regulator, LQR)라는 기초적인 설정 하에서 GAIL의 전역 수렴성(Global Convergence)을 이론적으로 입증하고, 이를 통해 모방 학습의 불안정성을 제어할 수 있는 토대를 마련하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 LQR 설정에서의 GAIL이 유일한 안장점(Saddle Point)으로 전역 수렴함을 이론적으로 증명한 것이다. 주요 아이디어와 기여 사항은 다음과 같다.

1. **전역 수렴성 입증**: GAIL을 Minimax 최적화 문제로 정의하고, 교차 경사 하강-상승(Alternating Gradient Descent-Ascent) 알고리즘이 전역적으로 수렴함을 보였다.
2. **수렴 속도 분석**: 전역적으로는 서브리니어(Sublinear) 수렴 속도를 가지며, 안장점 근처의 국소 영역에서는 선형(Linear) 수렴 속도를 가짐을 증명하여, 최종적으로 Q-선형(Q-linear) 수렴 속도를 가짐을 입증하였다.
3. **자기 강화 안정화 메커니즘(Self-enforcing Stabilizing Mechanism) 발견**: 명시적인 정규화 없이도 적절한 스텝 사이즈(Stepsize) 설정만으로 시스템이 불안정 영역으로 진입하기 전에 감속하는 암묵적 장벽이 존재하여, 중간 단계의 정책들이 항상 안정적(Stabilizing)으로 유지됨을 밝혔다.
4. **새로운 잠재 함수(Potential Function) 제안**: 비볼록-비오목 Minimax 최적화의 교차 업데이트 분석을 위해 경사도(Gradient)를 포함하는 새로운 잠재 함수를 설계하여 수렴성을 분석하였다.

## 📎 Related Works

논문은 모방 학습의 발전 과정을 다음과 같이 설명하며 기존 연구의 한계를 지적한다.

- **Behavioral Cloning**: 전문가의 행동을 단순 예측하지만, 오차 누적으로 인해 궤적이 발산하는 문제가 있다.
- **Inverse Reinforcement Learning (IRL)**: 보상 함수를 추론하고 이를 통해 정책을 학습하여 오차 누적 문제를 해결하려 하지만, 계산 복잡도가 높고 학습이 불안정하다.
- **GAIL**: IRL을 GAN의 구조로 풀어내어 생성자와 판별자(여기서는 정책과 보상 함수)가 서로 경쟁하며 학습하게 한다. 하지만 실제 구현 시 신경망의 비선형성과 RL의 불안정성이 결합되어 수렴성이 보장되지 않는다.
- **LQR 기반 RL 연구**: 최근 LQR 설정이 RL의 이론적 분석을 위한 렌즈로 사용되고 있다. 본 논문은 이러한 LQR 분석 프레임워크를 모방 학습 영역으로 확장하여, Minimax 최적화, 비볼록 기하학, 교차 업데이트, 시스템 안정성이라는 네 가지 핵심 난제를 동시에 다룬다.

## 🛠️ Methodology

### 1. LQR 설정 및 문제 정의

시스템의 상태 $x_t \in \mathbb{R}^d$와 행동 $u_t \in \mathbb{R}^k$에 대하여, 동역학은 다음과 같이 선형적으로 정의된다.
$$x_{t+1} = Ax_t + Bu_t$$
비용 함수(Cost Function)는 상태와 행동의 이차 형식으로 정의된다.
$$c(x_t, u_t) = x_t^\top Q x_t + u_t^\top R u_t$$
여기서 $Q, R$은 양의 정부호(Positive Definite) 행렬이다. 목표는 누적 비용을 최소화하는 선형 피드백 정책 $u_t = -Kx_t$를 찾는 것이다.

### 2. GAIL의 Minimax 정식화

본 논문은 GAIL을 다음과 같은 Minimax 최적화 문제로 정의한다.
$$\min_{K \in \mathcal{K}} \max_{\theta \in \Theta} m(K, \theta)$$
여기서 목적 함수 $m(K, \theta)$는 다음과 같다.
$$m(K, \theta) = C(K; \theta) - C(K_E; \theta) - \psi(\theta)$$

- $C(K; \theta)$: 정책 $K$ 하에서의 기대 누적 비용.
- $C(K_E; \theta)$: 전문가 정책 $K_E$ 하에서의 기대 누적 비용.
- $\psi(\theta)$: 비용 매개변수 $\theta = (Q, R)$에 대한 강볼록(Strongly Convex) 정규화 항.

이 식의 직관은 정책 $K$가 전문가 $K_E$와 다를 때 비용의 차이를 극대화하는 $\theta$를 찾고(판별자), 다시 그 비용을 최소화하는 $K$를 찾는(생성자) 구조이다.

### 3. 알고리즘: Alternating Gradient Update

정책 $K$와 비용 매개변수 $\theta$를 다음과 같이 순차적으로 업데이트한다.

**정책 업데이트 (Descent):**
$$K_{i+1} = K_i - \eta \nabla_K C(K_i; \theta_i)$$

**비용 매개변수 업데이트 (Ascent):**
$$\theta_{i+1} = \Pi_\Theta [\theta_i + \lambda \nabla_\theta m(K_{i+1}, \theta_i)]$$
여기서 $\Pi_\Theta$는 볼록 집합 $\Theta$로의 투영(Projection) 연산자이며, $\eta$와 $\lambda$는 각각의 스텝 사이즈이다.

### 4. 주요 수식 설명

- **상태 공분산 행렬**: $\Sigma_K = \mathbb{E}[\sum_{t=0}^\infty x_t x_t^\top]$는 정책 $K$에 의해 유도된 시스템의 상태 분포를 나타내며, 시스템의 안정성을 판별하는 핵심 지표가 된다.
- **비용 함수의 선형성**: $C(K; \theta) = \langle \Sigma_K, Q \rangle + \langle K \Sigma_K K^\top, R \rangle$로 표현되며, 이는 $\theta = (Q, R)$에 대해 선형적이다.

## 📊 Results

본 논문은 수치적 실험보다는 수학적 증명을 통한 이론적 결과 제시-에 집중하고 있다.

### 1. 안장점의 유일성 및 회복

- 목적 함수 $m(K, \theta)$의 근사 정지점(Proximal Stationary Point)이 유일하게 존재하며, 이 점이 유일한 안장점 $(K^*, \theta^*)$임을 증명하였다.
- 특히, 이 안장점에서 회복된 정책 $K^*$는 전문가의 정책 $K_E$와 일치하며, $\theta^*$는 $K_E$가 최적이 되게 하는 비용 매개변수 중 하나가 됨을 보였다.

### 2. 전역 수렴성 및 속도

- **안정성 보장**: Condition 4.1에 명시된 스텝 사이즈 조건을 만족할 경우, 모든 반복 단계 $i$에서 정책 $K_i$가 시스템을 안정적으로 유지(Stabilizing)함을 입증하였다.
- **수렴 속도**:
  - **Global Convergence**: 임의의 안정적인 초기 정책 $K_0$에서 시작하여 $\lim_{i \to \infty} \|L(K_i, \theta_i)\|_F = 0$ 임을 보였으며, 오차 $\epsilon$에 대해 필요한 반복 횟수 $\Gamma(\epsilon) \le \zeta/\epsilon$인 서브리니어 수렴성을 가짐을 증명하였다.
  - **Q-Linear Convergence**: 안장점 근처에서는 잠재 함수 $Z_i$가 $Z_{i+1} \le \upsilon Z_i$ ($\upsilon \in (0, 1)$) 형태로 감소하는 선형 수렴성을 가짐을 입증하였다.

## 🧠 Insights & Discussion

### 1. 강점 및 통찰

- **불안정성의 이론적 규명**: GAIL의 불안정성을 단순한 하이퍼파라미터 문제가 아닌, 비볼록-비오목 최적화의 기하학적 구조와 시스템 동역학의 안정성 문제로 연결하여 분석하였다.
- **암묵적 안정화 메커니즘**: 시스템이 불안정해질수록(즉, 비용 함수가 가팔라질수록) 경사도 업데이트가 자연스럽게 조절되는 '자기 강화' 메커니즘을 발견한 점이 매우 흥미롭다. 이는 복잡한 제약 조건 없이 스텝 사이즈 조절만으로 안정성을 확보할 수 있음을 시사한다.

### 2. 한계 및 논의사항

- **LQR 설정의 한계**: 본 연구는 선형 시스템과 이차 비용 함수라는 매우 특수한 설정(LQR)에서 이루어졌다. 실제 GAIL이 적용되는 심층 신경망 기반의 비선형 시스템에서도 이러한 전역 수렴성이 유지될지는 미지수이다.
- **스텝 사이즈 조건의 복잡성**: 이론적으로 제시된 스텝 사이즈 조건(Condition 4.1, 4.5, 4.6)이 매우 까다롭고 상수가 많아, 실제 구현 시 이를 정확히 맞추는 것이 어려울 수 있다.

## 📌 TL;DR

본 논문은 LQR 설정 하에서 GAIL의 전역 수렴성을 이론적으로 최초로 입증하였다. 특히, 교차 경사 하강-상승 알고리즘이 유일한 안장점으로 수렴하여 전문가의 정책과 보상 함수를 정확히 회복함을 보였으며, 그 속도는 전역적으로 서브리니어, 국소적으로 선형(Q-linear)임을 증명하였다. 이 연구는 모방 학습의 고질적인 불안정성 문제를 해결하기 위한 이론적 토대를 제공하며, 향후 비선형 시스템으로의 확장 연구에 중요한 지침이 될 가능성이 높다.
