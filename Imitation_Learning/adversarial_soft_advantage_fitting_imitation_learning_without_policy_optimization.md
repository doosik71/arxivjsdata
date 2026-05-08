# Adversarial Soft Advantage Fitting: Imitation Learning without Policy Optimization

Paul Barde, Julien Roy, Wonseok Jeon, Joelle Pineau, Christopher Pal, Derek Nowrouzezahrai (2020)

## 🧩 Problem to Solve

본 논문은 모방 학습(Imitation Learning, IL) 분야에서 Adversarial Imitation Learning(AIL) 알고리즘들이 겪고 있는 학습의 불안정성과 높은 계산 복잡도 문제를 해결하고자 한다. 기존의 AIL 방식은 전문가의 데이터와 생성된 데이터를 구분하는 Discriminator와, 이 Discriminator를 속이기 위해 정책을 최적화하는 Generator(Policy)를 교대로 학습시키는 구조를 가진다.

이러한 교대 최적화 과정은 매우 섬세한 튜닝을 요구하며, 특히 Generator의 업데이트 단계에서 Reinforcement Learning(RL) 알고리즘을 사용해야 한다는 점이 큰 부담이 된다. RL은 본질적으로 샘플 효율성이 낮고 학습 과정이 취약(brittle)하여, 전체적인 학습 루프의 불안정성을 가중시킨다. 따라서 본 연구의 목표는 AIL 프레임워크에서 RL을 통한 정책 최적화(Policy Optimization) 단계를 완전히 제거하면서도 전문가의 행동을 효과적으로 모방할 수 있는 단순하고 효율적인 알고리즘을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Discriminator의 구조를 특수하게 설계하여, Discriminator를 학습시키는 것만으로도 최적의 정책(Policy)을 동시에 학습하도록 만드는 것이다.

구체적으로, Discriminator를 단순히 전문가와 생성기의 데이터를 구분하는 이진 분류기로 보는 것이 아니라, 학습 가능한 정책 $\tilde{\pi}$에 의해 매개변수화된 structured discriminator로 정의한다. 이 구조 덕분에 Discriminator의 파라미터를 최적화하는 과정이 곧 전문가의 정책을 찾아가는 과정이 되며, 별도의 RL 루프 없이도 Discriminator 학습 후 그 파라미터를 그대로 Generator의 정책으로 사용할 수 있게 된다. 결과적으로 RL 단계의 제거를 통해 구현의 단순함과 계산 효율성을 동시에 달성하였다.

## 📎 Related Works

기존의 모방 학습은 크게 Behavioral Cloning(BC)과 Inverse Reinforcement Learning(IRL)으로 나뉜다. BC는 지도 학습 방식으로 단순하지만, 훈련 데이터 분포를 벗어날 때 발생하는 Compounding Error 문제가 심각하다. 이를 해결하기 위해 등장한 IRL은 보상 함수(Reward Function)를 학습하여 에이전트를 훈련시키는데, 초기 방법들은 매 단계마다 RL을 수렴시켜야 하므로 비용이 매우 컸다.

최근의 GAIL(Generative Adversarial Imitation Learning)이나 AIRL(Adversarial Inverse Reinforcement Learning)과 같은 AIL 방식들은 GAN의 구조를 도입하여 이 비용을 줄였으나, 여전히 내부적으로 PPO나 TRPO 같은 RL 알고리즘을 사용하여 정책을 업데이트해야 한다. 또한 SQIL과 같은 최신 연구는 RL만을 사용하여 단순화를 시도했으나, 학습이 진행됨에 따라 보상 신호가 감쇠하여 학습이 불안정해지는 문제가 보고되었다. 본 논문은 이러한 기존 AIL의 RL 의존성과 SQIL의 불안정성을 극복하고자 한다.

## 🛠️ Methodology

### 전체 파이프라인

ASAF(Adversarial Soft Advantage Fitting)는 RL 최적화 단계 없이 Discriminator 학습과 정책 업데이트를 교대로 수행한다.

1. 현재 정책 $\pi_G$를 사용하여 궤적(Trajectory) 데이터를 수집한다.
2. 수집된 데이터와 전문가 데이터를 사용하여 structured discriminator $D_{\tilde{\pi}, \pi_G}$를 학습시킨다.
3. 학습된 Discriminator의 파라미터 $\tilde{\pi}$를 그대로 다음 세대의 Generator 정책 $\pi_G$로 업데이트한다 ($\pi_G \leftarrow \tilde{\pi}$).

### Structured Discriminator 및 주요 방정식

본 논문은 Discriminator를 다음과 같은 형태로 정의한다:
$$D_{\tilde{\pi}, \pi_G}(\tau) = \frac{P_{\tilde{\pi}}(\tau)}{P_{\tilde{\pi}}(\tau) + P_{\pi_G}(\tau)}$$
여기서 $P_{\pi}(\tau)$는 정책 $\pi$ 하에서 궤적 $\tau$가 생성될 확률이다. 궤적 확률은 환경의 전이 확률 $\xi(\tau)$와 정책 확률 $q_\pi(\tau)$의 곱으로 분해되며, Discriminator 식의 분자와 분모에서 $\xi(\tau)$가 상쇄되므로 결과적으로 정책 확률의 곱만으로 계산이 가능하다:
$$D_{\tilde{\pi}, \pi_G}(\tau) = \frac{\prod_{t=0}^{T-1} \tilde{\pi}(a_t|s_t)}{\prod_{t=0}^{T-1} \tilde{\pi}(a_t|s_t) + \prod_{t=0}^{T-1} \pi_G(a_t|s_t)}$$

### 학습 절차 및 손실 함수

Discriminator $\tilde{\pi}$는 전문가 궤적 $\tau^{(E)}$와 생성된 궤적 $\tau^{(G)}$를 구분하기 위해 Binary Cross Entropy(BCE) 손실 함수를 최소화하도록 학습된다:
$$L_{BCE} \approx -\frac{1}{n_E} \sum_{i=1}^{n_E} \log D_{\tilde{\pi}, \pi_G}(\tau^{(E)}_i) - \frac{1}{n_G} \sum_{i=1}^{n_G} \log(1 - D_{\tilde{\pi}, \pi_G}(\tau^{(G)}_i))$$
이 최적화가 완료되면 $\tilde{\pi}$는 전문가의 궤적 분포와 일치하게 되며, 이를 통해 RL 과정 없이도 정책이 업데이트된다.

### 정책 클래스 및 변형 (ASAF-w, ASAF-1)

- **정책 파라미터화**: 이산 동작 공간에서는 Categorical 분포를, 연속 동작 공간에서는 정규 분포(Normal distribution)를 사용하여 밀도 함수를 평가하고 샘플링할 수 있도록 한다.
- **ASAF-w**: 전체 궤적 대신 크기가 $w$인 윈도우(sub-trajectories) 단위로 학습하여 궤적 길이가 매우 길거나 가변적인 환경에 대응한다.
- **ASAF-1**: 윈도우 크기를 1로 설정한 경우로, 궤적이 아닌 개별 전이(transition) 단위로 학습하는 방식이다.

## 📊 Results

### 실험 설정

- **데이터셋 및 환경**: Classic Control, Box2D, MuJoCo(연속 제어), Pommerman(이산 제어) 등 다양한 환경에서 평가하였다.
- **비교 대상**: GAIL+PPO, AIRL+PPO, SQIL 등 기존의 대표적인 IL 알고리즘과 비교하였다.
- **평가 지표**: Evaluation Return(평가 보상)을 통해 전문가의 성능에 얼마나 빠르게, 그리고 안정적으로 도달하는지 측정하였다.

### 주요 결과

- **안정성**: ASAF 계열 알고리즘은 GAIL, AIRL, SQIL과 달리 학습 도중 성능이 급격히 떨어지는 현상이 거의 없었으며, 전문가 성능에 도달한 후 이를 안정적으로 유지하는 모습을 보였다.
- **효율성**: RL 루프가 없기 때문에 학습 속도가 매우 빠르며, 하이퍼파라미터 변화에 대해서도 훨씬 강건(robust)하였다.
- **환경별 특성**:
  - Classic Control 및 Box2D에서는 ASAF, ASAF-w, ASAF-1 모두 우수한 성능을 보였다.
  - MuJoCo와 같이 궤적이 긴 환경에서는 전체 궤적을 사용하는 ASAF보다 ASAF-w나 ASAF-1이 훨씬 더 샘플 효율적이고 성능이 좋았다.
  - Pommerman과 같은 고차원 이산 제어 환경에서도 BC를 포함한 ASAF 계열이 다른 베이스라인보다 높은 승률을 기록하였다.

## 🧠 Insights & Discussion

### 강점 및 이론적 근거

본 논문은 structured discriminator를 도입함으로써 "Discriminator 학습 = 정책 학습"이라는 등식을 성립시켰다. 이론적으로 최적의 Discriminator 파라미터가 곧 전문가의 정책 분포와 일치함을 증명함으로써, RL이라는 복잡한 최적화 과정을 생략하고도 동일한 목적 함수를 달성할 수 있음을 보여주었다.

### 한계 및 비판적 해석

ASAF-1(전이 단위 학습)의 경우, 이론적으로는 전문가와 생성기의 상태 방문 분포(state occupancy measure)가 동일하다고 가정한다. 하지만 실제 학습 초기에는 이 가정이 성립하지 않으므로 이론과 실제 사이에 간극이 존재한다. 그럼에도 불구하고 실험적으로 ASAF-1이 매우 잘 작동한다는 점은, 상태 분포의 불일치보다 정책 자체의 유사성을 학습하는 것이 더 중요할 수 있음을 시사한다.

또한, 연속 동작 공간에서 $\tilde{\pi}$를 정규 분포로 근사하는 과정에서 발생하는 오차가 성능에 영향을 줄 수 있다. 하지만 이는 기존의 MaxEnt RL 프레임워크에서도 공통적으로 나타나는 한계이다.

## 📌 TL;DR

본 논문은 Adversarial Imitation Learning에서 가장 까다로운 부분인 **RL 기반의 정책 최적화 단계를 완전히 제거**한 **ASAF** 알고리즘을 제안한다. 특수하게 설계된 Structured Discriminator를 통해 Discriminator를 학습시키는 것만으로 전문가의 정책을 직접 복원하며, 이를 통해 학습의 안정성을 극대화하고 계산 비용을 절반 가까이 줄였다. 이 연구는 RL 없이도 효과적인 모방 학습이 가능하다는 것을 입증하였으며, 특히 ASAF-1과 같은 변형을 통해 실무적으로 적용 가능한 매우 단순하고 강력한 IL 프레임워크를 제공한다.
