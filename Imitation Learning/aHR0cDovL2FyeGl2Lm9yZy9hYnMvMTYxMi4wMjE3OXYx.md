# Model-based Adversarial Imitation Learning

Nir Baram, Oron Anschel, Shie Mannor (2016)

## 🧩 Problem to Solve

본 논문은 보상 신호(reward signal)에 대한 접근 권한이 없는 상태에서, 전문가의 시연 데이터(expert demonstrations)만을 이용하여 정책을 학습하는 **Imitation Learning(모방 학습)** 문제를 다룬다. 

전통적인 모방 학습 방식인 Behavioral Cloning(BC)은 환경의 역학(dynamics)을 고려하지 않아 작은 오차가 누적되는 **Covariate Shift** 문제와 높은 샘플 복잡도 문제를 가진다. Inverse Reinforcement Learning(IRL)은 보상 함수를 추론하려 하지만, 문제 자체가 매우 불량하게 정의되어(ill-posed) 해결이 어렵다는 한계가 있다.

최근 제안된 GAIL(Generative Adversarial Imitation Learning)과 같은 Adversarial approach는 GAN의 구조를 도입하여 이러한 문제를 완화하였으나, **Model-free** 방식이기 때문에 정책을 업데이트할 때 high-variance gradient estimation(예: REINFORCE)에 의존해야 한다. 이는 학습 효율을 떨어뜨리고 매우 많은 환경 상호작용을 요구하게 된다. 따라서 본 논문의 목표는 **Forward Model을 도입하여 시스템을 완전히 미분 가능(fully differentiable)하게 만듦으로써, 고분산의 기울기 추정치 대신 정확한 기울기를 사용하여 효율적으로 정책을 학습하는 MAIL(Model-based Adversarial Imitation Learning) 알고리즘을 제안**하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **환경의 전이 모델인 Forward Model을 학습하여, Discriminator의 출력을 정책 파라미터까지 직접 역전파(backpropagation)할 수 있는 경로를 생성**하는 것이다.

기존의 Model-free 방식은 상태(state)를 고정된 값으로 취급하여 행동(action)의 변화가 미래 상태 분포에 미치는 영향을 직접적으로 계산할 수 없었다. 그러나 MAIL은 $s' = f(s, a)$ 형태의 Forward Model을 통해 상태 $s$를 정책의 함수로 표현함으로써, Discriminator가 제공하는 $\nabla_s D$와 $\nabla_a D$ 정보를 모두 활용하여 정책을 최적화할 수 있게 한다.

## 📎 Related Works

1.  **Behavioral Cloning (BC):** 전문가의 상태-행동 쌍을 지도 학습(Supervised Learning) 방식으로 학습한다. 하지만 환경의 역학을 무시하므로, 학습 시 보지 못한 상태에 진입했을 때 오차가 누적되는 compounding errors 문제가 발생한다.
2.  **Inverse Reinforcement Learning (IRL):** 전문가가 최적화했을 것으로 추정되는 보상 함수 $\hat{r}$을 먼저 찾고, 이를 통해 RL로 정책을 학습한다. 하지만 적절한 보상 함수를 찾는 과정이 매우 어렵고 도메인 지식이 많이 필요하다.
3.  **Generative Adversarial Imitation Learning (GAIL):** GAN의 적대적 학습 구조를 모방 학습에 적용하여, 전문가의 분포와 생성 모델(정책)의 분포를 구분하는 Discriminator를 통해 학습한다. 하지만 Model-free 구조로 인해 $\nabla_\theta E_\pi [\log D(s, a)]$를 계산할 때 likelihood-ratio estimator(REINFORCE 등)를 사용해야 하며, 이는 분산이 매우 커 학습이 불안정하고 샘플 효율성이 낮다.

## 🛠️ Methodology

### 1. Discriminator의 해석
Discriminator $D(s, a)$는 주어진 상태-행동 쌍이 전문가 $\pi^E$에 의해 생성되었을 확률과 학습 정책 $\pi$에 의해 생성되었을 확률을 구분한다. 본 논문은 이를 다음과 같이 수식화하여 해석한다.

$$D(s, a) = \frac{1}{1 + \phi(s, a) \cdot \psi(s)}$$

여기서 $\phi(s, a) = \frac{p(a|s, \pi^E)}{p(a|s, \pi)}$는 정책의 가능도 비율(policy likelihood ratio)이며, $\psi(s) = \frac{p(s|\pi^E)}{p(s|\pi)}$는 상태 분포의 가능도 비율(state distribution likelihood ratio)이다. 이는 Discriminator가 "현재 행동이 전문가와 유사한가"와 "현재 도달한 상태가 전문가의 분포 내에 있는가"라는 두 가지 질문을 통해 판단함을 의미한다.

### 2. Re-parametrization Trick ($\nabla_a D$ 활용)
stochastic policy의 미분 가능성을 확보하기 위해 re-parametrization trick을 사용한다. 정책을 $\pi_\theta(a|s) = \mu_\theta(s) + \xi \sigma_\theta(s)$ ($\xi \sim \mathcal{N}(0, 1)$)로 정의하면, 기댓값의 기울기를 다음과 같이 몬테카를로 추정치로 계산할 수 있다.

$$\nabla_\theta E_{\pi(a|s)} [D(s, a)] \approx \frac{1}{M} \sum_{i=1}^M \nabla_a D(s, a) \nabla_\theta \pi_\theta(a|s) \Big|_{\xi=\xi_i}$$

### 3. Forward Model ($\nabla_s D$ 활용)
상태 $s$를 정책의 함수로 만들기 위해 $s' = f(s, a)$라는 Forward Model을 도입한다. 이를 통해 미래 상태 $s$에 대한 Discriminator의 기울기 $\nabla_s D$가 이전 시점의 정책 결정 $\theta$까지 전달될 수 있다. 전체 미분 법칙(law of total derivative)에 의해 다음과 같은 전파 경로가 형성된다.

$$\nabla_\theta D(s_t, a_t) = \frac{\partial D}{\partial a} \frac{\partial a}{\partial \theta} + \frac{\partial D}{\partial s} \left( \frac{\partial f}{\partial s} \frac{\partial s}{\partial \theta} + \frac{\partial f}{\partial a} \frac{\partial a}{\partial \theta} \right)$$

### 4. MAIL 알고리즘 및 목표 함수
MAIL의 목적은 궤적(trajectory)을 따라 Discriminator가 예측하는 확률의 할인된 합계를 최소화하는 것이다.

$$J(\theta) = E \left[ \sum_{t=0}^T \gamma^t D(s_t, a_t) \mid \theta \right]$$

학습 과정은 다음과 같다:
1. 환경과 상호작용하여 경험 버퍼 $B$에 $(s, a, s')$를 저장한다.
2. 버퍼 $B$를 사용하여 **Forward Model $f$**와 **Discriminator $D$**를 지도 학습 방식으로 학습시킨다.
3. $t=T$부터 $t=0$까지 역순으로 recursive하게 기울기 $J_s$와 $J_\theta$를 계산한다.
4. 계산된 $J_\theta$를 사용하여 정책 $\pi_\theta$를 업데이트한다.

## 📊 Results

### 실험 설정
- **환경:** MuJoCo 물리 시뮬레이터의 Hopper 및 Walker 작업.
- **전문가:** TRPO(Trust Region Policy Optimization)로 학습된 정책.
- **데이터셋:** 전문가 궤적의 수를 $\{4, 11, 18, 25\}$개로 다르게 설정하여 데이터 효율성을 측정.
- **비교 대상:** Behavioral Cloning(BC), GAIL.
- **지표:** 누적 보상(Total cumulative reward).

### 정량적 결과
| Task | Dataset Size | BC | GAIL | MAIL (Ours) |
| :--- | :--- | :--- | :--- | :--- |
| **Hopper** | 4 | $450.57 \pm 0.95$ | $3614.22 \pm 7.17$ | $\mathbf{3669.53 \pm 6.09}$ |
| | 25 | $3383.96 \pm 657.61$ | $3560.85 \pm 3.09$ | $\mathbf{3673.41 \pm 7.73}$ |
| **Walker** | 4 | $432.18 \pm 1.25$ | $4877.98 \pm 2848.37$ | $\mathbf{6916.34 \pm 115.20}$ |
| | 25 | $1599.36 \pm 1456.59$ | $6832.01 \pm 254.64$ | $\mathbf{7070.45 \pm 30.68}$ |

- MAIL은 모든 데이터셋 크기에서 BC와 GAIL보다 높은 보상을 달성했으며, 특히 전문가 정책의 성능에 가장 근접한 결과를 보였다.
- 데이터셋의 크기가 매우 작은 경우(4 trajectories)에도 GAIL 대비 월등한 성능 향상을 보여, 샘플 효율성이 높음을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 이점
- **기울기의 정확성:** Model-free 방식의 고분산 추정치 대신, Discriminator의 partial derivatives($\nabla_a D, \nabla_s D$)를 직접 사용하여 더 안정적이고 빠른 수렴이 가능하다.
- **효율성:** 적은 수의 환경 상호작용만으로도 학습이 가능하며, 튜닝해야 할 하이퍼파라미터가 상대적으로 적다.

### 한계 및 가정
- **Forward Model 의존성:** 시스템의 핵심이 Forward Model의 정확도에 달려 있다. 모델이 부정확할 경우 역전파되는 기울기에 노이즈가 섞여 수렴을 방해할 수 있다.
- **분포 변화 문제(Changing Distribution):** 정책이 업데이트됨에 따라 생성되는 데이터 분포가 계속 변하므로, Discriminator와 Forward Model이 이에 지속적으로 적응해야 하는 어려움이 있다.

### 비판적 해석 및 제언
- 저자는 Discriminator의 크기를 정책 네트워크보다 약 2배 크게 설정하고, 학습률을 완만하게 감소시키는 것이 효과적이었다고 언급한다. 또한 전문가 데이터에 노이즈를 추가하는 것이 Discriminator가 너무 쉽게 구분하는 것을 막아 학습 성능을 높였다는 점은 주목할 만하다.
- 다만, 복잡한 고차원 환경에서는 Forward Model을 정확하게 학습시키는 것이 매우 어려울 수 있으므로, 모델의 오차를 보정하거나 불확실성을 처리하는 메커니즘이 추가된다면 더 강건한 알고리즘이 될 것으로 보인다.

## 📌 TL;DR

본 논문은 Adversarial Imitation Learning에 **Forward Model**과 **Re-parametrization trick**을 도입하여, Model-free 방식의 고분산 기울기 문제를 해결한 **MAIL** 알고리즘을 제안한다. 이를 통해 시스템 전체를 미분 가능하게 만들어 Discriminator의 기울기를 정책에 직접 전달함으로써, **더 적은 데이터와 환경 상호작용만으로도 전문가의 성능에 근접한 정책을 학습**할 수 있음을 MuJoCo 시뮬레이션을 통해 입증하였다. 이 연구는 모델 기반 강화학습과 적대적 학습을 결합하여 모방 학습의 샘플 효율성을 극대화하는 방향성을 제시한다.