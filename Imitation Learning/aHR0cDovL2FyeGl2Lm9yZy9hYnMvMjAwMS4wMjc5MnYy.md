# On Computation and Generalization of Generative Adversarial Imitation Learning

Minshuo Chen, Yizhou Wang, Tianyi Liu, Zhuoran Yang, Xingguo Li, Zhaoran Wang, and Tuo Zhao (2020)

## 🧩 Problem to Solve

본 논문은 전문가의 시연 데이터로부터 순차적 의사결정 정책을 학습하는 Generative Adversarial Imitation Learning (GAIL)의 이론적 토대를 분석하는 것을 목표로 한다. 

기존의 모방 학습(Imitation Learning, IL) 방식들은 다음과 같은 한계가 있었다. 첫째, Behavioral Cloning (BC)은 훈련 데이터와 테스트 데이터의 분포가 달라지는 Covariate Shift 문제로 인해 복리 오차(compounding errors)가 발생하며 일반화 성능이 떨어진다. 둘째, Inverse Reinforcement Learning (IRL)은 보상 함수를 찾기 위해 내부적으로 고비용의 강화학습 문제를 반복해서 풀어야 하므로 고차원 환경에서 계산 효율성이 매우 낮다.

GAIL은 이를 해결하기 위해 모방 학습 문제를 Minimax 최적화 문제로 정식화하여 계산 효율성을 높였으며, 실무적으로 뛰어난 성과를 거두었다. 그러나 GAIL의 이론적 배경, 특히 시연 데이터의 시간적 의존성(temporal dependency)과 비볼록-비오목(non-convex-concave) 구조를 가진 최적화 문제의 수렴성 및 일반화 특성에 대해서는 그동안 명확히 규명되지 않았다. 따라서 본 논문은 GAIL의 통계적 일반화(Generalization)와 계산적 수렴성(Computation)에 대한 이론적 보장을 제공하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 GAIL의 일반화 성능과 최적화 알고리즘의 수렴성에 대한 최초의 이론적 분석을 제공했다는 점이다.

1.  **일반화 보장 (Generalization Guarantee):** $R$-reward distance라는 개념을 통해 GAIL의 일반화를 정의하고, 보상 함수 클래스 $R$의 복잡도(complexity)가 적절히 제어된다면 학습된 정책이 전문가의 정책으로 일반화될 수 있음을 증명하였다.
2.  **계산적 수렴성 증명 (Computational Convergence):** 보상 함수가 Reproducing Kernel Hilbert Space (RKHS) 상의 함수로 매개변수화될 때, 교차 미니배치 확률적 경사 하강법(alternating mini-batch stochastic gradient algorithm)이 정지점(stationary solution)으로 수렴함을 보였다. 특히, 일반적인 Minimax 문제와 달리 볼록-오목 구조가 아님에도 불구하고 수렴성을 입증하였다.

## 📎 Related Works

본 논문은 다음과 같은 기존 연구들과의 차별점을 가진다.

*   **Syed et al. (2008):** Apprenticeship learning의 일반화와 계산 특성을 연구했으나, 상태 공간(state space)이 유한하다고 가정하여 함수 근사(function approximation)를 고려하지 않았다. 반면 본 논문은 신경망이나 커널 함수를 이용한 함수 근사 환경에서의 이론을 다룬다.
*   **Cai et al. (2019):** 모방 학습의 계산 특성을 연구했지만, 선형 정책(linear policy)과 이차 보상(quadratic reward)이라는 매우 제한적인 가정 하에 분석을 진행했다. 본 논문은 더 일반적인 함수 클래스를 다룬다.
*   **GANs (Generative Adversarial Networks):** GAIL은 구조적으로 GAN과 유사하며, GAN의 일반화는 i.i.d. 데이터 분포 간의 Integral Probability Metric (IPM)으로 분석된다. 하지만 GAIL은 전문가의 궤적(trajectory)이라는 시간적 의존성이 있는 데이터를 다루므로, 본 논문은 이를 해결하기 위해 Independent Block Technique 등을 도입하여 분석을 수행했다.

## 🛠️ Methodology

### 1. GAIL의 정식화
GAIL은 전문가 정책 $\pi^*$와 학습 정책 $\pi$ 사이의 차이를 최소화하기 위해 다음과 같은 Minimax 최적화 문제를 해결한다.

$$\min_{\pi} \max_{r \in R} \mathbb{E}_{\pi}[r(s,a)] - \mathbb{E}_{\pi^*_n}[r(s,a)]$$

여기서 $\mathbb{E}_{\pi}[r(s,a)]$는 정책 $\pi$ 하에서의 평균 보상이며, $\mathbb{E}_{\pi^*_n}[r(s,a)]$는 전문가 시연 데이터로부터 계산된 경험적 평균 보상이다. 실제 구현에서는 정책 $\pi$와 보상 함수 $r$을 각각 $\omega$와 $\theta$라는 파라미터를 가진 함수 $\tilde{\pi}_\omega, \tilde{r}_\theta$로 근사하여 최적화한다.

### 2. 일반화 분석 (Generalization)
논문은 두 정책 간의 거리를 측정하기 위해 다음과 같이 $R$-distance를 정의한다.

$$d_R(\pi, \pi') = \sup_{r \in R} [\mathbb{E}_{\pi} r(s,a) - \mathbb{E}_{\pi'} r(s,a)]$$

이 거리는 본질적으로 정지 분포(stationary distribution) 상의 IPM에 해당한다. 저자들은 전문가 궤적이 $\beta$-mixing 마르코프 체인이라는 가정 하에, 보상 함수 클래스 $R$의 Covering Number가 제어될 때 일반화 오차가 유계(bounded)임을 증명하였다. 특히 RKHS와 ReLU 신경망 보상 함수에 대해 구체적인 일반화 상한(generalization bound)을 제시하였다.

### 3. 계산적 수렴성 분석 (Computation)
최적화의 안정성을 위해 논문은 다음과 같이 정규화 항이 추가된 수정된 목적 함수 $F(\omega, \theta)$를 제안한다.

$$\min_{\omega} \max_{\|\theta\| \le \kappa} \mathbb{E}_{\tilde{\pi}_\omega}[\tilde{r}_\theta(s,a)] - \mathbb{E}_{\pi^*}[\tilde{r}_\theta(s,a)] - \lambda H(\tilde{\pi}_\omega) - \frac{\mu}{2}\|\theta\|^2$$

여기서 $H(\tilde{\pi}_\omega)$는 정책의 엔트로피 정규화 항이며, $\frac{\mu}{2}\|\theta\|^2$는 보상 함수의 가중치 규제 항이다. 이 문제를 풀기 위해 두 가지 알고리즘을 분석한다.

*   **Alternating Mini-batch SGD:** 보상 파라미터 $\theta$를 업데이트한 후, 그 결과물을 이용하여 정책 파라미터 $\omega$를 업데이트하는 교차 업데이트 방식이다.
*   **Greedy SGD:** 보상 함수의 최적해 $\theta^*(\omega)$에 대한 비편향 추정치 $\hat{\theta}(\omega)$를 사용하여 $\omega$를 직접 업데이트하는 방식이다.

저자들은 Potential Function을 구축하여, 이 알고리즘들이 비볼록-비오목 구조임에도 불구하고 $L$-stationary point로 수렴함을 수학적으로 증명하였다.

## 📊 Results

### 실험 설정
*   **환경:** Acrobot-v1, MountainCar-v0, Hopper-v2 (RL 표준 벤치마크).
*   **전문가 데이터:** PPO 알고리즘으로 500회 반복 학습시킨 정책으로 생성.
*   **네트워크 구조:** 
    *   정책 $\pi$: 2개의 은닉층(각 128 뉴런), tanh 활성화 함수.
    *   보상 $r$: 2개의 은닉층(1024, 512 뉴런), ReLU 활성화 함수.
*   **비교 대상:** Neural Network(NN) 보상, Kernel 보상(특성 매핑 방식), Greedy SGD, Alternating Mini-batch SGD.

### 주요 결과
*   **알고리즘 성능:** Greedy SGD와 Alternating Mini-batch SGD가 거의 유사한 성능을 보였다.
*   **보상 함수 형태:** NN 보상이 Kernel 보상보다 약간 더 높은 성능을 기록하였다. 그러나 NN 보상은 학습 과정이 더 불안정하며 수렴하는 데 더 많은 시간이 소요되는 경향을 보였다.
*   **결론:** 이론적으로 제시한 수렴성 분석이 실제 환경에서도 유효함을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 이론적 가치
본 논문은 실무적으로 널리 사용되던 GAIL에 대해 처음으로 통계적 일반화와 계산적 수렴성을 엄밀하게 증명하였다. 특히 시간적 의존성이 있는 마르코프 체인 데이터에 대해 Independent Block Technique을 적용하여 일반화 상한을 도출한 점과, 복잡한 Minimax 최적화의 수렴성을 Potential Function으로 풀어낸 점이 돋보인다.

### 한계 및 논의사항
*   **일반화와 표현력의 트레이드오프:** 분석 결과, 보상 함수 클래스 $R$의 복잡도가 낮아야 일반화 성능이 보장된다. 하지만 $R$이 너무 단순하면 전문가 정책을 충분히 표현하지 못해 최적의 정책을 회복할 수 없다. 따라서 일반화 성능과 표현력 사이의 최적의 균형점을 찾는 것이 중요하다.
*   **정책 업데이트의 제한:** 본 연구의 계산 이론은 Policy Gradient 업데이트만을 고려하였다. 실제로는 PPO, TRPO, Natural Policy Gradient와 같은 더 발전된 업데이트 방식을 사용하는데, 이를 이론적으로 확장하는 것이 향후 중요한 연구 방향이 될 것이다.

## 📌 TL;DR

본 논문은 GAIL의 일반화 특성과 최적화 수렴성을 이론적으로 규명하였다. 보상 함수 클래스의 복잡도가 제어될 때 일반화가 보장됨을 증명하였으며, RKHS 기반의 보상 함수를 사용할 때 교차 SGD 알고리즘이 정지점으로 수렴함을 보였다. 이 연구는 GAIL의 블랙박스적인 성격을 제거하고 이론적 근거를 제시함으로써, 향후 더 안정적이고 효율적인 모방 학습 알고리즘 설계에 기여할 가능성이 크다.