# Energy-Based Imitation Learning

Minghuan Liu, Tairan He, Minkai Xu, Weinan Zhang (2021)

## 🧩 Problem to Solve

본 논문은 전문가의 시연(Expert Demonstrations)으로부터 최적의 정책(Optimal Policy)을 복원하고자 하는 모방 학습(Imitation Learning, IL)의 고질적인 문제들을 해결하려 한다.

기존의 모방 학습 방법론들은 크게 두 가지 방향으로 나뉘지만, 각각 뚜렷한 한계를 가진다. 첫째, 행동 복제(Behavior Cloning, BC)는 지도 학습 방식을 채택하여 단순하지만, 훈련 데이터에 없는 상태에 진입했을 때 오차가 누적되는 compounding error 및 covariate shift 문제에 취약하다. 둘째, 역강화학습(Inverse Reinforcement Learning, IRL)이나 최근의 생성적 적대 신경망(GAN) 기반 방법론(예: GAIL)은 보상 함수와 정책을 번갈아 업데이트하는 bi-level 또는 alternating optimization 구조를 가진다. 이러한 방식은 계산 비용이 매우 높고, GAN 특유의 훈련 불안정성으로 인해 수렴이 어렵다는 단점이 있다.

따라서 본 논문의 목표는 보상 함수 추정과 정책 학습을 완전히 분리하여 계산 효율성을 높이고 훈련 안정성을 확보한 새로운 모방 학습 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 에너지 기반 모델(Energy-Based Model, EBM)을 활용하여 모방 학습을 **'에너지 복원(Energy Recovery)'**과 **'정책 학습(Policy Learning)'**의 두 단계로 분리하는 것이다.

중심적인 직관은 전문가의 상태-행동 분포(Occupancy Measure)를 에너지 함수로 모델링하고, 이 에너지를 대리 보상 함수(Surrogate Reward Function)로 사용하는 것이다. 이를 통해 기존의 적대적 학습(Adversarial Training) 방식에서 벗어나, 먼저 전문가의 에너지 분포를 고정적으로 추정한 뒤 이를 바탕으로 강화학습을 수행하는 단순하고 유연한 구조를 제안하였다. 또한, 이론적 분석을 통해 이 방식이 Maximum Entropy IRL(MaxEnt IRL)과 본질적으로 동일한 원리를 공유하며, 적대적 IRL 방법론의 단순화된 대안이 될 수 있음을 입증하였다.

## 📎 Related Works

논문에서는 다음과 같은 기존 연구들의 한계를 지적한다.

1. **Behavior Cloning (BC):** 단순한 지도 학습으로 정책을 학습하지만, 전문가의 궤적에서 벗어났을 때 발생하는 오차 누적 문제로 인해 장기적인 작업 수행 능력이 떨어진다.
2. **Inverse Reinforcement Learning (IRL):** 전문가의 보상 함수를 먼저 찾고 그에 맞는 정책을 학습하는 구조이나, 보상 함수 업데이트와 정책 업데이트가 상호 의존적이어서 계산 복잡도가 매우 높다.
3. **Generative Adversarial Imitation Learning (GAIL):** 에이전트의 점유 측정치(Occupancy Measure)와 전문가의 점유 측정치 사이의 발산(Divergence)을 최소화하려 한다. 하지만 GAN의 훈련 불안정성을 그대로 계승하여 하이퍼파라미터에 민감하고 수렴이 불안정하다.

EBIL은 이러한 적대적/교차적 최적화 과정 없이, Score Matching이라는 통계적 기법을 통해 에너지 함수를 독립적으로 먼저 학습함으로써 기존 방식들과 차별화된다.

## 🛠️ Methodology

### 1. 전체 파이프라인

EBIL은 다음과 같은 2단계 구조로 작동한다.

- **Stage 1 (Energy Recovery):** 전문가의 시연 데이터를 사용하여 상태-행동 쌍의 에너지 함수 $V_{\pi_E}(s, a)$를 추정한다. 이때 보상 함수를 직접 찾는 것이 아니라, 데이터의 밀도를 나타내는 에너지 기반 모델을 학습한다.
- **Stage 2 (Policy Learning):** 학습된 에너지 함수를 기반으로 대리 보상 함수 $\hat{r}(s, a)$를 정의하고, 이를 사용하여 일반적인 강화학습(RL) 알고리즘을 통해 정책을 학습한다.

### 2. 이론적 배경 및 목표 함수

본 논문은 에이전트의 점유 측정치 $\rho_\pi$와 전문가의 점유 측정치 $\rho_{\pi_E}$ 사이의 Reverse KL Divergence를 최소화하는 문제를 다룬다. 전문가의 점유 측정치를 볼츠만 분포(Boltzmann distribution) 형태의 EBM으로 모델링하면 다음과 같이 정의된다.
$$(1-\gamma)\rho_{\pi_E}(s, a) = \frac{1}{Z} \exp(-V_{\pi_E}(s, a))$$
여기서 $Z$는 파티션 함수(Partition function)이다. 분석 결과, KL Divergence를 최소화하는 문제는 다음과 같은 MaxEnt RL의 목적 함수를 최대화하는 것과 동일함이 유도된다.
$$\arg \max_\pi \mathbb{E}_\pi [-V_{\pi_E}(s, a)] + \mathcal{H}(\pi)$$
여기서 $\mathcal{H}(\pi)$는 정책의 엔트로피이며, 이는 에너지 함수 $-V_{\pi_E}$가 곧 보상 함수 $r$의 역할을 수행할 수 있음을 의미한다.

### 3. 에너지 추정 방법: Score Matching

파티션 함수 $Z$를 직접 계산하는 것은 고차원 공간에서 불가능에 가깝다. 이를 해결하기 위해 본 논문은 DEEN(Deep Energy Estimator Networks) 프레임워크의 **Denoising Score Matching** 기법을 사용한다.

데이터 $x = (s, a)$에 가우시안 노이즈를 섞은 $y = x + \epsilon$를 생성하고, 다음의 목적 함수를 최소화하여 에너지 함수 $V_\theta$를 학습한다.
$$\arg \min_\theta \sum_{x_i \in \mathcal{D}, y_i \in \mathcal{Y}} \| x_i - y_i + \sigma^2 \nabla_y V_\theta(y) \|^2$$
이 식은 에너지 함수의 기울기(Score)를 직접 학습함으로써 $Z$를 계산하지 않고도 에너지 값 자체를 추정할 수 있게 한다.

### 4. 보상 함수 구성 및 정책 학습

학습된 에너지 함수를 사용하여 대리 보상 함수 $\hat{r}(s, a)$를 다음과 같이 정의한다.
$$\hat{r}(s, a) = h(-V_{\pi_E}(s, a))$$
여기서 $h(\cdot)$는 단조 증가하는 선형 함수로, 환경에 맞게 보상의 스케일을 조정하는 역할을 한다. 이후 SAC(Soft Actor-Critic)나 TRPO와 같은 강화학습 알고리즘을 사용하여 $\hat{r}$을 최대화하는 정책을 학습한다.

## 📊 Results

### 1. 실험 설정

- **Synthetic Task:** 1차원 공간에서 이동하는 단순 환경. 전문가의 정책이 상태에 따라 두 가지 가우시안 분포를 따르도록 설정하였다.
- **MuJoCo Benchmarks:** Humanoid, Hopper, Walker2d, Swimmer, InvertedDoublePendulum 등 고차원 연속 제어 작업.
- **비교 대상:** GAIL, AIRL, RED, GMMIL, BC.
- **지표:** 에피소드당 실제 보상(Episodic true rewards) 및 전문가 궤적과의 KL Divergence.

### 2. 주요 결과

- **정성적 분석 (Synthetic Task):** EBIL은 전문가의 궤적 밀도를 정확하게 캡처한 보상 지도를 생성하였으며, 학습 과정이 매우 안정적이었다. 반면 GAIL은 보상 신호가 무의미하게 학습되는 경향이 있었고, AIRL은 에너지가 아닌 엔트로피가 포함된 보상을 복원하는 특이점을 보였다.
- **정량적 분석 (Sub-optimal Experts):** 하위 최적 전문가 데이터를 사용했을 때, EBIL은 대부분의 환경에서 최상위권 혹은 경쟁력 있는 성능을 보였다. 특히 GAN 기반의 GAIL/AIRL보다 학습 곡선이 훨씬 안정적이었다.
- **정량적 분석 (Optimal Experts):** 고차원 작업(Hopper, Walker2d)에서는 적대적 방법론들이 여전히 우세했으나, 저차원 작업(LunarLander)에서는 EBIL이 더 우수한 성능을 보였다.
- **State-only Imitation:** 전문가의 행동 없이 상태 정보만 주어진 상황에서도 타겟 상태 분포의 에너지를 복원하여 성공적으로 정책을 학습함을 입증하였다.

## 🧠 Insights & Discussion

### 1. 강점 및 이론적 의의

EBIL은 복잡한 bi-level 최적화를 단순한 2단계(에너지 학습 $\rightarrow$ 정책 학습) 과정으로 치환함으로써 훈련 안정성을 극적으로 향상시켰다. 또한, MaxEnt IRL과 EBIL이 본질적으로 같은 문제의 다른 표현임을 이론적으로 증명하여, 에너지 기반 모델링이 모방 학습의 강력한 도구가 될 수 있음을 보여주었다.

### 2. 한계 및 비판적 해석

- **Sparse Reward 문제:** 에너지 모델이 너무 완벽하게 학습되면 전문가가 방문하지 않은 영역의 보상이 급격히 낮아지는 '희소 보상(Sparse Reward)' 문제가 발생한다. 흥미롭게도 논문의 ablation study 결과, **'절반쯤 학습된(half-trained)'** 모델이 오히려 더 부드러운 보상 신호를 제공하여 정책 학습에 유리한 경우가 있었다. 이는 에너지 모델의 정밀도와 강화학습의 탐색(Exploration) 사이의 트레이드오프가 존재함을 시사한다.
- **고차원 성능 격차:** 고차원 작업에서 적대적 방법론에 비해 성능이 다소 낮은 이유는 DEEN과 같은 단순 MLP 구조의 에너지 모델이 고차원 데이터의 복잡한 분포를 완전히 캡처하는 데 한계가 있기 때문으로 분석된다.

## 📌 TL;DR

본 논문은 모방 학습의 불안정한 적대적 학습 구조를 탈피하여, **에너지 기반 모델(EBM)을 이용한 2단계 학습 프레임워크(EBIL)**를 제안한다. 먼저 전문가의 데이터 분포를 에너지 함수로 추정하고, 이를 고정된 보상 함수로 사용하여 정책을 학습하는 방식이다. 실험 결과, EBIL은 기존 GAIL/AIRL 대비 매우 안정적인 학습 곡선을 보였으며, 특히 하위 최적 전문가 데이터나 상태 전용 모방 학습에서 강점을 나타냈다. 이 연구는 향후 복잡한 환경에서 보상 함수를 설계하기 어려운 작업에 대해, 데이터 기반의 에너지 복원을 통한 효율적인 정책 학습 가능성을 제시한다.
