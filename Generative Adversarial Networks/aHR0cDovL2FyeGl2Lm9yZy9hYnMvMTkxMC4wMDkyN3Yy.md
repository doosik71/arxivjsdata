# Stabilizing GANs: A Survey

Maciej Wiatrak, Stefano V. Albrecht, Andrew Nystrom (2020)

## 🧩 Problem to Solve

본 논문은 Generative Adversarial Networks (GANs) 학습 과정에서 발생하는 고질적인 불안정성 문제를 해결하기 위해 제안된 다양한 방법론들을 체계적으로 분석하고 분류하는 것을 목표로 한다. GAN은 복잡한 실세계 데이터를 모델링하는 능력이 뛰어나지만, 실제 학습 과정에서는 다음과 같은 심각한 문제들이 발생한다.

- **비수렴 (Non-convergence):** Generator($G$)와 Discriminator($D$)가 서로 대립하며 최적화되는 과정에서 진동(Oscillatory)하거나 발산(Diverging)하는 동작이 나타나며, 전역 최적점인 Nash Equilibrium (NE)에 도달하지 못하고 지역 최적점(Local NE)에 빠지는 경우가 많다.
- **기울기 소실 및 폭주 (Vanishing or Exploding Gradients):** $D$가 너무 빠르게 최적화되어 완벽하게 실제 데이터와 가짜 데이터를 구분하게 되면, $G$에게 전달되는 기울기가 0에 수렴하여 학습이 완전히 멈추거나 매우 느려지는 현상이 발생한다.
- **모드 붕괴 (Mode Collapse):** $G$가 다양한 샘플을 생성하지 못하고, $D$를 속이기 가장 쉬운 특정 몇 가지 출력값(mode)만을 반복적으로 생성하는 현상이다.

이러한 불안정성은 GAN을 범용적인 도구로 사용하는 데 큰 장애물이 되며, 모델을 작동시키기 위해 과도한 수동 하이퍼파라미터 튜닝이 필요하게 만든다. 따라서 본 논문은 기존의 안정화 기법들을 종합적으로 검토하여 연구자들에게 가이드를 제공하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 분산되어 있던 GAN 안정화 방법론들을 다섯 가지 주요 카테고리로 체계화하여 통합적인 택소노미(Taxonomy)를 제시한 점이다.

1. **구조적 변경 (Modified Architectures):** 네트워크 설계 및 계층 구조의 변경을 통한 안정화.
2. **손실 함수 수정 (Modified Loss Functions):** 거리 측정 방식 및 목적 함수 변경을 통한 기울기 문제 해결.
3. **게임 이론 적용 (Game Theory):** 2인 제로섬 게임의 이론적 접근을 통한 수렴성 개선.
4. **다중 에이전트 활용 (Multi-Agent):** 여러 개의 $G$ 또는 $D$를 사용하여 모드 붕괴 방지 및 학습 효율 증대.
5. **기울기 최적화 수정 (Modified Gradient Optimization):** 표준 Gradient Descent Ascent (GDA)의 한계를 극복하는 최적화 알고리즘 도입.

또한, 각 방법론의 장단점을 비교 분석하고, 현재까지 해결되지 않은 개방형 문제(Open Problems)를 제시함으로써 향후 연구 방향을 명시하였다.

## 📎 Related Works

기존의 GAN 관련 서베이 논문들은 주로 GAN의 전반적인 개념, 다양한 응용 분야 및 일반적인 구조를 소개하는 데 집중하였으며, 학습 절차의 세부적인 불안정성 문제와 그 해결책을 깊이 있게 다루지 않았다. 예를 들어, 일부 연구는 특정 아키텍처나 손실 함수에만 집중하거나, 단순히 성능 지표 위주의 실험적 결과만을 보고하였다.

반면, 본 논문은 단순한 성능 나열이 아니라 '학습 안정화'라는 구체적인 목적에 초점을 맞춘다. 특히 기존 서베이에서 간과되었던 게임 이론적 접근과 기울기 기반 최적화 방법을 포함하여, 이론적 배경과 실무적 휴리스틱을 동시에 다룬다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### GAN의 기본 원리와 불안정성의 수학적 배경
GAN은 $G$와 $D$ 사이의 경쟁 게임으로, 다음과 같은 목적 함수 $L(D, G)$를 최적화한다.
$$L(D, G) = \mathbb{E}_{x \sim p_r} \log[D(x)] + \mathbb{E}_{z \sim p_z} \log[1 - D(G(z))]$$
이론적으로는 $D = \frac{1}{2}$일 때 유일한 해가 존재하며 이는 Nash Equilibrium (NE)에 해당한다. 그러나 실제 학습에서는 $D$가 최적화될 때 Jensen-Shannon Divergence (JSD)를 최소화하게 되는데, $p_r$과 $p_g$의 서포트(support)가 겹치지 않는 저차원 매니폴드 상에 존재할 경우 $D$가 너무 완벽해져 $G$로 전달되는 기울기가 사라지는 문제가 발생한다.

### 안정화 방법론의 5대 분류

#### 1. Modified Architectures
- **Convolutional Architectures:** DCGAN은 CNN 구조를 도입하여 이미지 생성 품질을 높였으며, SAGAN은 Self-Attention 메커니즘을 통해 전역적 의존성을 학습한다. BigGAN은 배치 크기와 파라미터 수를 대폭 늘려 SOTA 성능을 달성했다.
- **Hierarchical Architectures:** PROGAN은 저해상도에서 고해상도로 네트워크를 점진적으로 확장하며 학습하여 초기 학습 불안정성을 줄인다. SGAN은 다수의 $G-D$ 쌍을 스택 구조로 쌓아 조건부 손실(Conditional Loss)을 통해 학습한다.
- **Autoencoder Architectures:** BEGAN과 같이 $D$를 오토인코더 구조로 설계하여 재구성 비용(Reconstruction Cost)을 에너지 값으로 사용함으로써 $D$의 탐욕적 최적화를 방지한다.
- **Latent Space Decomposition:** InfoGAN은 잠재 변수 $z$ 외에 의미 있는 특징을 포착하는 변수 $c$를 추가하여 상호 정보량(Mutual Information)을 최대화함으로써 특징을 분리(Disentangle)한다.

#### 2. Modified Loss Functions
- **$f$-Divergence:** 표준 GAN의 JSD 외에 다양한 $f$-divergence를 사용하는 $f$-GAN이나, 시그모이드 교차 엔트로피 대신 최소제곱법을 사용하는 LSGAN 등이 있다.
- **Integral Probability Metric (IPM):** 데이터 차원에 영향을 덜 받는 거리 측정 방식이다. WGAN은 Wasserstein 거리(Earth Mover's distance)를 도입하여 기울기 소실 문제를 획기적으로 개선하였으며, WGAN-GP는 여기에 Gradient Penalty를 추가하여 립시츠(Lipschitz) 연속성 조건을 강제한다.
- **Auxiliary Loss:** Unrolled GAN은 $G$가 $D$의 미래 반응을 예측하도록 손실 함수에 추가 항을 넣어 모드 붕괴를 방지한다.

#### 3. Game Theory for GANs
표준 GAN의 Pure Strategy NE 대신 Mixed Strategy Nash Equilibrium (MNE)을 찾는 접근법이다. ChekhovGAN은 Regret Minimization 알고리즘을 사용하여 수렴성을 보장하려 했으나, $D$가 단일 레이어여야 한다는 강한 제약이 있다.

#### 4. Multi-Agent GANs
- **Multiple Generators:** MAD-GAN이나 MGAN은 여러 $G$를 두어 각 $G$가 서로 다른 모드를 생성하도록 유도함으로써 모드 붕괴를 방지한다.
- **Multiple Discriminators:** 여러 $D$를 사용하여 $G$에게 더 풍부한 피드백을 제공한다.

#### 5. Modified Gradient Optimization
표준 GDA의 진동 문제를 해결하기 위해 Optimistic Mirror Descent (OMD)나, 잠재적 게임(Potential Game)과 해밀토니안 게임(Hamiltonian Game)으로 분해하여 해결하는 ConOpt 등의 알고리즘이 제안되었다.

## 📊 Results

본 논문은 특정 실험 데이터셋을 통한 정량적 결과보다는, 기존 문헌들의 결과를 종합한 **비교 요약(Comparative Summary)**을 제시한다.

- **아키텍처 기반 방법:** 현재 실무적으로 가장 높은 성능 향상을 보이며(예: BigGAN), 진입 장벽이 낮다. 하지만 이론적 근거가 부족한 경우가 많고 특정 문제(예: 모드 붕괴)만 부분적으로 해결하는 경향이 있다.
- **손실 함수 기반 방법:** WGAN과 같이 이론적 정당성이 높고 일반화 능력이 뛰어나다. 하지만 하이퍼파라미터 튜닝이 잘 된 아키텍처 기반 모델(DCGAN 등)보다 항상 우월하지는 않다는 결과가 보고되었다.
- **게임 이론 및 기울기 기반 방법:** 이론적으로는 가장 유망하지만, 제약 조건이 너무 많거나 아직 초기 연구 단계여서 실무 적용 사례가 적다.
- **다중 에이전트 방법:** 모드 붕괴 방지에 효과적이지만, 계산 비용이 급격히 증가하는 단점이 있다.

## 🧠 Insights & Discussion

### 강점 및 분석
본 논문은 GAN의 불안정성을 단순한 '현상'이 아니라, 수학적 목적 함수와 최적화 알고리즘(GDA)의 상호작용 관점에서 분석하였다. 특히, 단순한 모델 구조 변경보다는 손실 함수와 최적화 알고리즘의 근본적인 수정이 병행되어야 진정한 안정화가 가능하다는 통찰을 제공한다.

### 한계 및 비판적 해석
- **하이퍼파라미터의 영향:** Lucic et al. (2018)의 연구를 인용하며, 많은 신규 방법론들이 사실은 정교한 하이퍼파라미터 튜닝만으로도 달성 가능한 성능 향상을 '방법론의 기여'로 포장하고 있을 가능성을 지적한다.
- **이론과 실제의 괴리:** 많은 논문이 이론적 수렴성을 증명하지만, 실제 딥러닝 네트워크의 비볼록-비오목(non-convex-concave) 특성 때문에 이론이 실제 학습 결과와 일치하지 않는 경우가 많다.

### 향후 연구 방향 (Open Problems)
1. **방법론의 결합:** 아키텍처, 손실 함수, 최적화 알고리즘을 전략적으로 결합한 통합 프레임워크의 필요성.
2. **Actor-Critic 구조 도입:** 강화학습의 Replay Buffer나 Prioritized Experience Replay 개념을 GAN에 도입하여 아웃라이어 샘플을 효율적으로 학습하는 방안.
3. **$D$의 일반화 능력:** $D$가 단순히 몇 가지 특징만으로 가짜를 구분하는 것이 아니라, 진정한 데이터 분포를 학습하여 $G$에게 유의미한 피드백을 주도록 강제하는 메커니즘 연구.

## 📌 TL;DR

이 논문은 GAN 학습의 3대 난제인 **비수렴, 기울기 소실/폭주, 모드 붕괴**를 해결하기 위한 기존 연구들을 **구조, 손실 함수, 게임 이론, 다중 에이전트, 기울기 최적화**라는 5가지 관점에서 체계적으로 정리한 서베이 논문이다. 

단순히 성능이 좋은 모델을 나열하는 것이 아니라, 각 접근법의 이론적 배경과 한계를 분석함으로써, 향후 GAN 연구가 단순한 구조 변경을 넘어 **최적화 알고리즘의 근본적 개선과 일반화 능력 향상**으로 나아가야 함을 시사한다. 특히 실무자들에게는 하이퍼파라미터 튜닝의 중요성과 함께, 각 안정화 기법의 상호 보완적 적용 가능성을 제시한다는 점에서 가치가 높다.