# Actor-Critic Instance Segmentation

Nikita Araslanov, Constantin A. Rothkopf, Stefan Roth (2019)

## 🧩 Problem to Solve

본 논문은 이미지 내에서 서로 유사하거나 부분적으로 가려진(occluded) 여러 객체를 분리해내는 Instance Segmentation 문제를 해결하고자 한다. 기존의 많은 접근 방식은 이미지 요소들을 병렬적으로 처리하는 방식에 집중했으나, 저자들은 인간의 시각 시스템처럼 인스턴스 분할을 순차적인 작업으로 처리하는 방식에 주목한다.

특히, 기존의 순차적(recurrent) 모델들은 예측된 결과와 Ground-truth(GT) 세그먼트를 매칭하기 위해 Kuhn-Munkres 알고리즘 기반의 global max-matching assignment 방식을 사용했다. 하지만 이러한 방식은 다음과 같은 치명적인 한계가 있다. 첫째, 초기 할당(initial assignment) 결과에 따라 최종 예측 순서가 결정되는 편향이 발생한다. 둘째, 각 타임스텝의 손실 함수가 미래의 예측에 미치는 영향(long-term effect)을 고려하지 못한다. 따라서 본 논문의 목표는 Reinforcement Learning(RL)의 Actor-Critic 구조를 도입하여, 예측 순서에 대한 탐색을 가능하게 하고 미래의 보상을 고려한 최적의 순차적 분할 모델을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Instance Segmentation을 마르코프 결정 과정(MDP)으로 정의하고, 이를 해결하기 위해 Actor-Critic 프레임워크를 적용하는 것이다.

1. **Actor-Critic 구조의 도입**: Actor는 순차적으로 인스턴스 마스크를 예측하고, Critic은 해당 예측이 미래에 미칠 영향을 점수화하여 Actor에게 그래디언트를 제공함으로써 예측 순서의 최적화를 돕는다.
2. **cVAE를 통한 Action Space 압축**: 픽셀 단위의 마스크 예측은 차원이 너무 높아 RL의 탐색(exploration)이 불가능하다는 점을 해결하기 위해, Conditional Variational Auto-Encoder(cVAE)를 사용하여 고차원 마스크를 저차원의 연속적인 잠재 표현(latent representation)으로 압축하여 Action Space로 정의했다.
3. **State Pyramid 설계**: 디코더가 해상도 손실을 보완하고 객체의 경계 및 중심점을 더 잘 파악할 수 있도록, foreground 예측과 angle quantisation 정보를 포함한 다중 스케일의 보조 채널을 제공하는 State Pyramid를 제안했다.

## 📎 Related Works

기존의 Instance Segmentation 연구는 크게 두 가지 방향으로 나뉜다. 하나는 픽셀 단위의 인코딩을 학습한 뒤 후처리를 통해 클러스터링하는 방식이며, 다른 하나는 Mask R-CNN과 같이 Bounding Box를 먼저 예측하고 그 내부에서 마스크를 생성하는 병렬 처리 방식이다. 후자는 현재 가장 성능이 좋지만, 검출 단계의 성능에 의존하며 픽셀 수준의 컨텍스트 활용에 한계가 있다.

순차적 예측 방식의 경우, Convolutional LSTM이나 Spatial Softmax를 이용한 연구들이 존재했다. 그러나 기존의 순차적 모델들은 주로 Bounding Box를 먼저 예측하거나, 앞서 언급한 max-matching 기반의 손실 함수를 사용하여 학습되었기에 예측 순서의 편향 문제와 미래 보상 결여 문제를 안고 있었다. 본 논문은 이러한 한계를 극복하기 위해 RL의 Actor-Critic 구조를 통해 직접적으로 마스크를 생성하는 방식을 취하며, 이는 GAN의 Generator-Discriminator 구조와 유사한 원리로 작동한다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조

시스템은 크게 cVAE 기반의 Actor 네트워크와 Q-value를 예측하는 Critic 네트워크로 구성된다. Actor는 현재 상태 $s_t$를 입력받아 잠재 변수 $a_t$를 샘플링하고, 이를 디코더를 통해 고해상도 마스크 $m_t$로 변환한다. Critic은 상태 $s_t$와 액션 $a_t$를 입력받아 기대 보상인 Q-value를 계산하여 Actor의 학습을 가이드한다.

### MDP 정의

- **State ($s_t$)**: 입력 이미지 $I$와 이전 타임스텝까지 예측된 마스크들의 합집합인 aggregated mask $M_t$의 튜플 $(I, M_t)$로 정의된다.
- **Action ($a_t$)**: cVAE의 잠재 공간(latent space) 내의 연속적인 벡터 $a_t \in \mathbb{R}^l$이다. 디코더 $D$를 통해 $D(a_t)$라는 바이너리 마스크로 확장된다.
- **State Transition**: 새로운 상태는 이전 상태의 마스크와 현재 예측된 마스크의 픽셀 단위 최대값으로 업데이트된다.
  $$T((I, M_t), a_t) = (I, \max(M_t, D(a_t)))$$
- **Reward ($r_t$)**: 상태 잠재력(state potential) $\phi_t$의 차이로 정의된다. $\phi_t$는 현재까지의 예측 결과와 GT 세그먼트 간의 max-matching 기반 거리(Dice score 또는 IoU)의 합이다.
  $$\phi_t := \max_{k \in P(N)} \sum_{i=1}^t F(S_i, T_{k_i})$$
  $$r_t := \phi(s_{t+1}) - \phi(s_t)$$

### 학습 절차 및 손실 함수

학습은 두 단계로 진행된다.

1. **Pre-training**: Actor의 cVAE 부분을 먼저 학습시켜, 임의의 GT 마스크를 입력받아 이를 잠재 변수로 인코딩하고 다시 복원하는 능력을 갖추게 한다. 이때 Binary Cross-Entropy(BCE)와 KL-divergence 손실을 사용한다.
2. **Joint Training (Actor-Critic)**:
   - **Critic Update**: 실제 보상의 할인 합(discounted sum of rewards)과 Critic이 예측한 $Q_\phi$ 값 사이의 L2 거리를 최소화한다.
     $$L_{Critic,t} = \| Q_\phi(s_t, a_t) - \sum_{i=t}^N \gamma^{i-t} r_i \|^2$$
   - **Actor Update**: Critic이 제공하는 $Q$ 값을 최대화하는 방향으로 그래디언트를 업데이트하며, 동시에 잠재 공간의 분포를 유지하기 위해 KL-divergence 손실 $L_{KL}$을 함께 사용한다.

## 📊 Results

### 실험 설정

- **데이터셋**: CVPPP(식물 잎 분할) 및 KITTI(차량 분할) 벤치마크를 사용했다.
- **지표**: Symmetric Best Dice (SBD), absolute Difference in Counting (|DiC|), MWCov, MUCov 등을 사용했다.
- **비교 대상**: truncated BPTT를 사용하는 Baseline(BL-Trunc) 및 기존 SOTA 모델들(E2E, Mask R-CNN 계열 등)과 비교했다.

### 주요 결과

1. **Ablation Study**: Actor-Critic 모델(AC-Dice)이 단순 recurrent baseline(BL)보다 Dice score와 Counting 정확도 모두에서 우수한 성능을 보였다. 특히 KL-divergence를 통한 액션 공간 탐색과 State Pyramid의 존재가 성능 향상에 핵심적임을 확인했다.
2. **타임스텝별 분석**: 그림 3에서 확인되듯, AC 모델은 예측 후반부(later timesteps)로 갈수록 Baseline 대비 Dice score가 크게 향상되는 경향을 보였다. 이는 Critic이 미래의 보상을 고려하여 초기 단계부터 최적의 순서를 학습했기 때문으로 분석된다.
3. **벤치마크 성능**:
   - **CVPPP**: SOTA 수준의 Counting 성능을 보이며, 경쟁력 있는 세그멘테이션 정확도를 달성했다.
   - **KITTI**: Bounding box를 사용하지 않음에도 불구하고 경쟁력 있는 성능을 보였으며, 고해상도 이미지와 더 큰 액션 공간에서도 잘 작동함을 입증했다.

## 🧠 Insights & Discussion

본 논문은 Instance Segmentation을 순차적 의사결정 문제로 정의함으로써, 기존 recurrent 모델들이 가졌던 할당 편향(assignment bias) 문제를 RL의 탐색 메커니즘으로 해결했다.

특히 주목할 점은 모델이 학습한 **예측 순서(Prediction Order)**이다. 시각화 결과, 모델은 별도의 제약 조건 없이도 "쉬운 것부터 어려운 것 순으로" 예측하는 전략을 학습했다. 구체적으로는 크고 가려지지 않은 객체를 먼저 예측하고, 작거나 가려진 객체를 나중에 예측하는 패턴을 보였다. 이는 이전 예측 결과가 컨텍스트가 되어 어려운 객체를 분리하는 데 도움을 주는 인간의 시각 처리 방식과 유사하다.

한계점으로는, Critic 네트워크가 매우 복잡한 reward function을 완벽하게 근사하는 데 한계가 있어, 일부 미세한 디테일(예: 잎의 줄기 부분)을 놓치는 경우가 발생한다는 점이 언급되었다. 또한, 현재의 Actor 네트워크 구조가 최종 정확도의 상한선을 제한하고 있어 향후 아키텍처 개선이 필요함을 시사한다.

## 📌 TL;DR

본 논문은 Instance Segmentation을 순차적 작업으로 모델링하고, 고차원 마스크 공간의 한계를 cVAE로 해결한 Actor-Critic 프레임워크를 제안한다. Critic이 미래의 보상을 예측함으로써 예측 순서의 편향을 줄이고 후반부 예측 정확도를 높였으며, 결과적으로 '쉬운 객체 $\rightarrow$ 어려운 객체' 순으로 분할하는 직관적인 전략을 학습함을 보였다. 이 연구는 RL이 복잡한 픽셀 단위의 구조적 예측 문제에도 효과적으로 적용될 수 있음을 입증하여, 향후 순차적 비전 작업 연구에 중요한 기여를 할 가능성이 높다.
