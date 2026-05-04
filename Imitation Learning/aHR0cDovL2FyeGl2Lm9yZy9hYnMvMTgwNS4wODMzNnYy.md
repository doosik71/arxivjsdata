# Maximum Causal Tsallis Entropy Imitation Learning

Kyungjae Lee, Sungjoon Choi, Songhwai Oh (2018)

## 🧩 Problem to Solve

본 논문은 전문가의 비결정론적(non-deterministic) 행동, 즉 동일한 상황에서 여러 가지 서로 다른 최적 행동을 취하는 다중 모드(multi-modal) 정책을 효율적으로 학습하는 것을 목표로 한다. 복잡한 작업에서는 동일한 목표를 달성하기 위한 여러 경로가 존재할 수 있으며, 이러한 다중 모드 정책을 학습하는 것은 에이전트가 예상치 못한 사건으로 인해 실패했을 때 복구 능력을 높이는 등 강건성(robustness) 측면에서 매우 중요하다.

기존의 Maximum Causal Entropy (MCE) 프레임워크나 GAIL과 같은 방법론들은 주로 Softmax 분포나 단일 가우시안(single Gaussian) 분포를 사용하여 전문가의 정책을 모델링한다. 그러나 Softmax 분포는 행동 공간이 넓어질 때 전문가가 선택하지 않은 불필요한 행동에도 무시할 수 없는 확률 질량을 할당하는 경향이 있으며, 단일 가우시안 분포는 본질적으로 단일 모드(uni-modal) 특성을 가지므로 전문가의 다중 모드 행동을 적절히 표현하지 못한다는 한계가 있다. 따라서 본 연구는 전문가의 다중 모드 정책을 효율적으로 학습하고 불필요한 행동을 배제할 수 있는 새로운 모방 학습 프레임워크를 제안한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Maximum Causal Tsallis Entropy (MCTE) 프레임워크를 제안하여 전문가의 희소(sparse)하고 다중 모드인 정책 분포를 학습할 수 있게 한 점이다.

중심적인 설계 아이디어는 정책 분포를 모델링할 때 Softmax 대신 Sparsemax 분포를 도입하는 것이다. Sparsemax는 지지 집합(supporting set)을 스스로 조절하여 불필요한 행동에 정확히 0의 확률을 할당할 수 있어, 전문가의 행동을 더 정확하게 모방할 수 있다. 또한, 연속적인 행동 공간으로 확장하기 위해 Sparsemax 기반의 가중치를 사용하는 Sparse Mixture Density Network (Sparse MDN)를 제안하였다. 특히, MDN의 Tsallis entropy가 해석 가능한 분석적 형태(analytic form)를 가짐을 증명하였으며, 이를 통해 혼합 성분들의 평균이 서로 멀어지도록 유도하여 탐색 효율을 높이고 모드 붕괴(mode collapse)를 방지하는 메커니즘을 구축하였다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 언급하며 본 연구와의 차별점을 제시한다.

첫째, Maximum Causal Entropy (MCE) 기반의 Inverse Reinforcement Learning (IRL) 연구들은 전문가의 확률적 행동을 모델링하기 위해 Softmax 분포를 사용한다. 하지만 앞서 언급했듯이, 이는 행동 공간이 커질수록 비전문가 행동에 확률을 할당하는 문제가 발생한다.

둘째, Generative Adversarial Imitation Learning (GAIL)은 모델-프리(model-free) 환경에서 IRL을 해결하기 위해 생성적 적대 신경망 구조를 도입하였다. GAIL은 매우 강력한 성능을 보이지만, 기본적으로 단일 가우시안 정책을 사용하여 다중 모드 행동을 학습하는 데 한계가 있다.

셋째, Sparse Markov Decision Processes (Sparse MDP) 연구는 Causal Sparse Tsallis Entropy를 사용하여 최적 정책이 희소하고 다중 모드인 분포를 가짐을 보였다. 본 논문은 이러한 Sparse MDP의 개념을 모방 학습 프레임워크로 확장하여 MCTEIL을 제안함으로써, 기존의 MCE 기반 방법론들이 가진 한계를 극복하고자 한다.

## 🛠️ Methodology

### 1. Maximum Causal Tsallis Entropy (MCTE) 원리

MCTE 프레임워크는 다음과 같이 정책 $\pi$의 Causal Tsallis Entropy $W(\pi)$를 최대화하는 문제로 정의된다.

$$\text{maximize}_{\pi \in \Pi} \quad \alpha W(\pi)$$
$$\text{subject to} \quad E_\pi[\phi(s, a)] = E_{\pi_E}[\phi(s, a)]$$

여기서 $W(\pi) = \frac{1}{2} E_\pi[1 - \pi(a|s)]$이며, $\phi(s, a)$는 특징 맵(feature map)이다. 본 논문은 이 문제의 최적해 $\pi^*$가 Sparsemax 분포를 따른다는 것을 수학적으로 증명하였다. Sparsemax 정책은 특정 임계값 $\tau$보다 낮은 가치를 가진 행동에 0의 확률을 할당함으로써 지지 집합을 최적화한다.

### 2. MCTEIL 알고리즘 및 적대적 학습 구조

연속적인 행동 공간과 모델-프리 환경에서 MCTE를 구현하기 위해, 본 논문은 GAIL과 유사한 생성적 적대 신경망 구조를 채택한다. 전체 최적화 문제는 다음과 같은 minimax 게임으로 정식화된다.

$$\min_{\pi \in \Pi} \max_{D} E_\pi[\log(D(s, a))] + E_{\pi_E}[\log(1 - D(s, a))] - \alpha W(\pi)$$

여기서 $D$는 판별자(discriminator)이며, $\pi$는 학습 정책이다. 학습 과정은 판별자 $D$가 전문가와 학습자의 궤적을 구분하도록 업데이트하고, 정책 $\pi$는 판별자를 속이는 동시에 Tsallis entropy 보너스 $\alpha W(\pi)$를 최대화하는 방향으로 업데이트되는 반복적인 구조를 가진다.

### 3. Sparse Mixture Density Network (Sparse MDN)

연속 공간에서 다중 모드를 표현하기 위해, 본 논문은 가우시안 혼합 모델(GMM)의 가중치 $w_i(s)$를 Sparsemax 분포로 모델링하는 Sparse MDN을 제안한다. 정책 $\pi(a|s)$는 다음과 같이 정의된다.

$$\pi(a|s) = \sum_{i=1}^K w_i(s) \mathcal{N}(a; \mu_i(s), \Sigma_i(s))$$

기존 MDN은 $w_i(s)$를 Softmax로 계산하지만, Sparse MDN은 Sparsemax를 사용하여 불필요한 혼합 성분을 완전히 제거함으로써 더 효율적으로 전문가의 정책을 모방한다.

### 4. MDN의 Tsallis Entropy 분석적 형태

본 논문의 핵심적인 이론적 기여 중 하나는 MDN의 Tsallis entropy가 다음과 같은 닫힌 형태(closed-form)로 계산될 수 있음을 증명한 것이다.

$$W(\pi) = \frac{1}{2} \sum_s \rho_\pi(s) \left( 1 - \sum_{i,j} w_i(s) w_j(s) \mathcal{N}(\mu_i(s); \mu_j(s), \Sigma_i(s) + \Sigma_j(s)) \right)$$

이 식은 Tsallis entropy가 혼합 성분들의 평균 $\mu_i, \mu_j$ 사이의 거리에 비례함을 보여준다. 따라서 $W(\pi)$를 최대화하면 각 모드의 평균들이 서로 멀리 떨어지도록 유도되어, 탐색(exploration)이 촉진되고 여러 모드가 하나로 뭉치는 모드 붕괴 현상을 방지할 수 있다.

## 📊 Results

### 1. Multi-Goal Environment 실험

네 개의 attractor(목표 지점)와 네 개의 repulsor(장애물)가 존재하는 환경에서 다중 모드 행동 학습 능력을 검증하였다.

- **측정 지표**: 평균 리턴(Average Return) 및 도달 가능성(Reachability, 얼마나 많은 목표 지점에 도달했는가).
- **결과**: MCTEIL은 Soft GAIL보다 훨씬 높은 도달 가능성을 보였다. 이는 Soft GAIL이 모드 붕괴로 인해 일부 목표만 학습하는 반면, MCTEIL은 Tsallis entropy의 특성 덕분에 전문가의 모든 모드를 성공적으로 학습했음을 의미한다.

### 2. MuJoCo Continuous Control 실험

Halfcheetah, Walker2d, Reacher, Ant의 네 가지 환경에서 BC, GAIL, Soft GAIL과 성능을 비교하였다.

- **설정**: 전문가 궤적 데이터를 4, 11, 18, 25개로 다양하게 설정하여 실험하였다.
- **결과**: Walker2d를 제외한 대부분의 환경에서 MCTEIL이 가장 높은 평균 리턴을 기록하였다. 특히 Reacher 환경에서는 기존 GAIL이 실패하거나 BC보다 성능이 낮았던 것과 달리, MCTEIL이 모든 데이터 수량 조건에서 최적의 성능을 보였다. 이는 MDN이 다양한 방향으로 탐색을 수행하고, Sparsemax가 불필요한 성분을 제거함으로써 학습 효율을 극대화했기 때문이다.

## 🧠 Insights & Discussion

본 논문은 수학적 분석과 실험을 통해 Tsallis entropy가 모방 학습, 특히 다중 모드 정책 학습에 있어 매우 강력한 도구가 될 수 있음을 보여주었다.

가장 큰 강점은 **해석 가능한 분석적 형태의 엔트로피 식**을 도출했다는 점이다. 기존의 Gibbs-Shannon entropy는 MDN 구조에서 계산이 불가능하여 근사치에 의존해야 했으나, Tsallis entropy는 정확한 계산이 가능하여 모드 간 거리를 직접적으로 제어할 수 있다. 이는 강화학습의 고질적인 문제인 탐색과 활용의 균형을 맞추는 데 기여하며, 특히 전문가 데이터가 부족한 상황에서도 강건한 정책을 학습할 수 있게 한다.

다만, 본 논문에서 제안한 방법은 혼합 성분의 개수 $K$를 브루트 포스(brute force) 방식으로 탐색하여 결정했다는 한계가 있다. 실제 매우 복잡한 환경에서는 최적의 $K$를 찾는 것이 어려울 수 있으며, 이를 동적으로 결정하는 메커니즘에 대한 논의가 추가된다면 더 발전된 연구가 될 것이다. 또한, Robust Bayes 추정과의 연결성을 통해 Brier score 관점에서 최악의 상황을 최소화한다는 이론적 근거를 제시한 점은 매우 인상적이다.

## 📌 TL;DR

본 논문은 전문가의 다중 모드 행동을 모방하기 위해 **Maximum Causal Tsallis Entropy (MCTE)** 프레임워크를 제안하였다. 최적해인 **Sparsemax 분포**와 이를 연속 공간으로 확장한 **Sparse MDN**을 통해 불필요한 행동 확률을 제거하고 다중 모드를 효율적으로 학습한다. 특히 MDN의 Tsallis entropy에 대한 분석적 식을 통해 모드 붕괴를 막고 탐색을 촉진함으로써, MuJoCo 등 복잡한 제어 환경에서 기존 GAIL 및 BC보다 뛰어난 성능을 입증하였다. 이 연구는 다중 모드 정책 학습이 필수적인 로보틱스 및 자율 주행 분야의 모방 학습에 중요한 이론적/실무적 토대를 제공한다.
