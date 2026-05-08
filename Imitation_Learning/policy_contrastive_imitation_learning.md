# Policy Contrastive Imitation Learning

Jialei Huang, Zhaoheng Yin, Yingdong Hu, Yang Gao (2023)

## 🧩 Problem to Solve

본 논문은 Adversarial Imitation Learning (AIL) 방식이 복잡한 작업에서 여전히 만족스럽지 못한 성능을 보이는 문제에 집중한다. 저자들은 그 주요 원인이 AIL에서 사용되는 Discriminator의 표현(Representation) 품질이 낮기 때문이라고 분석한다.

기존의 AIL Discriminator는 전문가(Expert)의 데이터와 에이전트(Agent)의 데이터를 구분하는 이진 분류(Binary Classification) 방식으로 학습된다. 그러나 이진 분류의 목적은 단지 두 클래스를 분리하는 것일 뿐, 표현 공간 내에서 전문가와 에이전트의 행동을 의미 있게 비교할 수 있는 매끄럽고(Smooth) 구조적인 공간을 생성하도록 강제하지 않는다. 결과적으로, 이러한 표현 공간을 기반으로 생성된 보상(Reward) 신호 역시 품질이 낮아지며, 이는 에이전트의 학습 효율과 성능 저하로 이어진다. 따라서 본 논문의 목표는 전문가와 에이전트의 행동을 의미 있게 비교할 수 있는 안정적인 표현 공간을 학습하여 모방 학습의 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 이진 분류 대신 **Contrastive Learning(대조 학습)**을 도입하여 **Policy-Contrastive Representation Space**를 구축하는 것이다.

중심적인 설계 직관은 전문가의 상태-행동 쌍(state-action pair)들은 표현 공간 상에서 서로 가깝게 뭉치도록(Compactness) 만들고, 에이전트의 샘플들은 전문가의 샘플들로부터 멀어지게 밀어내는 것이다. 이를 통해 전문가의 행동 특성을 대표하는 강건한 특징(Robust features)을 캡처할 수 있으며, 단순히 클래스를 나누는 경계선을 찾는 것을 넘어 전문가의 분포와 에이전트의 분포 사이의 거리를 의미 있게 측정할 수 있는 메트릭 공간을 형성할 수 있다.

## 📎 Related Works

본 논문은 다음과 같은 기존 연구들의 한계를 지적한다.

1. **Adversarial Imitation Learning (AIL/GAIL):** GAIL과 같은 방식은 전문가와 에이전트의 분포를 일치시키려 하지만, Discriminator의 학습이 불안정하고 하이퍼파라미터 튜닝에 매우 민감하다. 특히 Discriminator가 학습하는 latent space의 구조적 제약이 없어 보상 신호가 부정확할 수 있다.
2. **Trajectory-matching IRL (PWIL, SIL):** Wasserstein 거리나 Sinkhorn 거리 등을 사용하여 분포 일치 문제를 해결하려 했으나, 여전히 표현 학습(Representation Learning)의 관점보다는 거리 측정 방식의 개선에 치중해 있다.
3. **기존의 Contrastive Learning 적용 시도:** 일부 연구에서 self-supervised representation learning을 AIL에 접목했으나 큰 효과를 거두지 못했다. 저자들은 그 이유가 데이터 증강(Augmentation) 기반의 일반적인 대조 학습은 '좋은 행동'과 '나쁜 행동' 사이의 미세한 차이를 구분해내지 못하기 때문이라고 분석한다.

## 🛠️ Methodology

### 전체 시스템 구조

PCIL의 전체 파이프라인은 **Contrastive Encoder $\Phi$ 학습 $\to$ 유사도 기반 보상 생성 $\to$ RL 알고리즘을 통한 정책 $\pi$ 최적화** 순으로 진행된다. 에이전트는 학습 과정에서 생성한 궤적을 리플레이 버퍼에 저장하며, 이를 기반으로 인코더와 정책을 동시에 업데이트한다.

### Contrastive Policy Representation (인코더 학습)

인코더 $\Phi: S \to S$는 상태-행동 쌍을 고차원 구(Sphere) 공간의 벡터로 매핑한다. 본 논문에서는 전문가 샘플을 긍정 샘플(Positive)로, 에이전트 샘플을 부정 샘플(Negative)로 정의하는 InfoNCE 손실 함수를 사용한다.

$$L = \mathbb{E} \left[ \Phi(x^0)^T \Phi(x^p) + \log \left( \exp \Phi(x^0)^T \Phi(x^p) + \sum_{i=1}^n \exp \Phi(x^0)^T \Phi(\tilde{x}_i) \right) \right]$$

여기서 $x^0$는 앵커(Anchor)가 되는 전문가 샘플, $x^p$는 다른 전문가 샘플(Positive), $\tilde{x}_i$는 에이전트의 샘플(Negative)이다. 이 목적 함수는 전문가 데이터들끼리는 서로 당기고, 에이전트 데이터는 전문가로부터 밀어내어 전문가의 특성을 응축한 표현 공간을 만든다.

### Similarity-Based Imitation Reward (보상 설계)

학습된 표현 공간의 메트릭 특성을 활용하여, 코사인 유사도(Cosine Similarity) 기반의 보상을 정의한다.

$$r(x) = \Phi(x)^T \mathbb{E}_{x_E \sim D} [\Phi(x_E)]$$

에이전트의 현재 상태-행동 표현 $\Phi(x)$가 전문가 샘플들의 평균 표현 $\mathbb{E}[\Phi(x_E)]$와 유사할수록 더 높은 보상을 받게 된다. 실제 구현에서는 계산 효율을 위해 전문가 샘플 하나를 무작위로 추출하여 유사도를 계산한다.

### 이론적 분석 및 학습 절차

저자들은 PCIL의 목적 함수가 Taylor 전개를 통해 Apprenticeship Learning (AL)의 형태로 환원될 수 있음을 수학적으로 증명하였다. 또한, PCIL의 목적 함수가 전문가 분포 $\rho_E$와 에이전트 분포 $\rho_\pi$ 사이의 **Total Variation (TV) Divergence**를 최소화하는 것과 이론적으로 등가임을 Theorem 1을 통해 제시하였다.

학습 시에는 RL 알고리즘으로 DrQ-v2를 사용하며, 학습 안정성을 위해 Wasserstein-GAN에서 사용되는 Gradient Penalty 기법을 보상 함수에 적용하였다.

## 📊 Results

### 실험 설정

- **데이터셋:** DeepMind Control Suite의 10가지 MuJoCo 작업 (단순한 Cart-pole부터 복잡한 Quadruped Run까지 포함).
- **기준선(Baselines):** Behavioral Cloning (BC), DAC (최신 AIL 방식), PWIL 및 SIL (Trajectory-matching 방식).
- **지표:** Episode Return (에피소드당 총 보상).
- **환경 상호작용 제한:** 모든 실험은 $2 \times 10^6$ steps 이내에서 수행되었다.

### 주요 결과

- **성능 향상:** PCIL은 거의 모든 작업에서 기존 baseline들을 압도하는 성능을 보였다. 특히 Cheetah Run, Quadruped Run과 같은 복잡한 작업에서 성능 향상 폭이 매우 컸다.
- **샘플 효율성:** 동일한 환경 상호작용 횟수 대비 PCIL이 전문가 수준의 성능에 더 빠르게 도달함을 확인하였다.
- **표현 공간 분석:** t-SNE 시각화 결과, DAC와 달리 PCIL은 전문가 샘플들이 하나의 조밀한 클러스터를 형성하고 있었다. 또한, 에이전트 샘플과 전문가 클러스터 사이의 거리가 실제 환경 보상(Ground truth reward)과 높은 상관관계를 보임을 확인하여, 학습된 표현 공간이 의미 있게 구조화되었음을 입증하였다.

### Ablation Study

1. **TCN 표현 학습 vs PCIL:** 시간적 연속성을 이용하는 TCN 방식은 전문가와 비전문가를 구분하는 목적이 없으므로 PCIL보다 성능이 현저히 낮았다.
2. **GAIL-like 보상 vs 유사도 보상:** PCIL 표현 공간을 사용하더라도 이진 분류기 기반의 GAIL 보상을 사용하면 성능이 급격히 하락했다. 이는 PCIL이 구축한 메트릭 공간의 특성상 거리 기반의 유사도 보상이 필수적임을 시사한다.

## 🧠 Insights & Discussion

본 논문은 AIL의 성능 저하 원인을 'Discriminator의 표현 학습 부족'이라는 관점에서 정확히 짚어냈으며, 이를 Contrastive Learning으로 해결함으로써 이론적 기반과 실무적 성능을 모두 확보하였다. 특히 단순히 손실 함수를 바꾼 것이 아니라, 표현 공간의 구조적 제약이 어떻게 보상 신호의 품질을 높이고 결과적으로 정책 최적화에 기여하는지를 t-SNE 시각화와 이론적 증명을 통해 명확히 설명한 점이 돋보인다.

다만, 본 연구는 전문가 데이터셋 $D$가 충분히 주어졌다는 가정하에 진행되었으며, 전문가 데이터가 극히 적은 상황(Few-shot)에서도 이와 같은 Contrastive Representation이 유효하게 작동할지에 대해서는 명시적으로 다루지 않았다. 또한, 에이전트의 정책이 진화함에 따라 Negative 샘플의 분포가 계속 변하는데, 이에 대한 동적 적응 전략보다는 단순 리플레이 버퍼 샘플링에 의존했다는 점이 향후 개선 가능성으로 보인다.

## 📌 TL;DR

본 논문은 AIL의 고질적인 문제인 낮은 Discriminator 표현 품질을 해결하기 위해, 전문가와 에이전트의 정책을 대조하여 학습하는 **Policy Contrastive Imitation Learning (PCIL)**을 제안한다. InfoNCE 손실 함수를 통해 전문가-에이전트를 구분하는 의미 있는 메트릭 공간을 학습하고, 이를 코사인 유사도 기반의 보상으로 연결하여 DeepMind Control Suite의 복잡한 작업들에서 SOTA 성능을 달성하였다. 이 연구는 모방 학습에서 단순한 분류를 넘어 '의미 있는 표현 학습'이 얼마나 중요한지를 시사하며, 향후 더 복잡한 로봇 제어 및 모방 학습 연구에 중요한 기반을 제공할 것으로 기대된다.
