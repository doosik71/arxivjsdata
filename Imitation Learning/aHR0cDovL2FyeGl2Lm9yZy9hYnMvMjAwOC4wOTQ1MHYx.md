# Adversarial Imitation Learning via Random Search

MyungJae Shin, Joongheon Kim (2020)

## 🧩 Problem to Solve

본 논문은 복잡한 동적 작업(Dynamic tasks)을 수행하는 에이전트를 학습시키기 위한 강화학습(Reinforcement Learning, RL)의 복잡성과 재현성 문제를 해결하고자 한다. 구체적으로는 다음과 같은 문제점들에 집중한다.

첫째, Model-free RL은 합리적인 성능을 내기 위해 방대한 양의 데이터가 필요하며, 이를 해결하기 위해 제안된 최신 기법들이 지나치게 복잡해짐에 따라 알고리즘의 재현성 위기(Reproducibility crisis)가 발생하고 있다.

둘째, 보상 함수 설계에 대한 의존성 문제이다. 환경에서 제공되는 보상 신호가 매우 희소(Sparse)하거나 아예 없는 경우, 보상 함수를 수동으로 설계하는 Reward shaping은 매우 까다로우며 이는 모델이 최적해에 도달하지 못하고 Local optima에 빠지는 원인이 된다.

셋째, 기존의 모방 학습(Imitation Learning) 방법론인 GAIL(Generative Adversarial Imitation Learning) 등이 TRPO나 PPO와 같은 복잡한 Deep RL 알고리즘을 내부 루프에서 사용하기 때문에, 모방 학습 역시 동일한 재현성 문제와 복잡성 문제를 안고 있다.

따라서 본 논문의 목표는 단순한 Linear Policy와 Derivative-free optimization 기법을 결합하여, 재현성이 높으면서도 계산 효율적인 새로운 모방 학습 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **GAIL의 적대적 네트워크(Adversarial Network) 구조**와 **ARS(Augmented Random Search)의 무작위 탐색 기반 최적화**를 결합하는 것이다.

중심적인 설계 직관은 복잡한 신경망 기반의 정책과 경사 하강법(Gradient Descent) 기반의 최적화 대신, 단순한 선형 정책(Linear Policy)과 파라미터 공간에서의 무작위 탐색(Random Search)을 사용함으로써 모델의 복잡도를 획기적으로 줄이고 학습의 안정성과 재현성을 확보하는 것이다. 이를 위해 보상 함수 대신 전문가의 시연(Demonstration)과 에이전트의 궤적을 구분하는 Discriminator를 사용하여 에이전트를 학습시킨다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 소개하며 본 제안 방법과의 차이점을 설명한다.

1.  **Behavioral Cloning (BC):** 전문가의 상태-행동 쌍을 직접 학습하는 지도 학습 방식이다. 단순하지만 전문가 데이터가 매우 많이 필요하며, 학습 시와 테스트 시의 상태 분포가 달라지는 Distribution mismatch 문제가 발생할 경우 에이전트가 매우 취약해진다.
2.  **Inverse Reinforcement Learning (IRL):** 전문가의 행동을 설명할 수 있는 숨겨진 보상 함수 $R^*$를 찾아내고 이를 통해 정책을 최적화한다. 궤적 전체를 고려하므로 BC보다 견고하지만, 보상 함수 최적화와 정책 최적화를 동시에 수행해야 하므로 계산 비용이 매우 높다.
3.  **Generative Adversarial Imitation Learning (GAIL):** GAN의 구조를 차용하여 Discriminator가 보상 함수 역할을 하도록 설계되었다. 하지만 TRPO와 같은 복잡한 DRL 알고리즘을 사용하므로 하이퍼파라미터 설정이 어렵고 재현성이 낮다는 한계가 있다.
4.  **Augmented Random Search (ARS):** 파라미터 공간에서 무작위 방향으로 샘플링하여 보상을 최대화하는 Derivative-free 최적화 방법이다. 단순한 Linear Policy만으로도 MuJoCo 벤치마크에서 경쟁력 있는 성능을 보였으며, 본 논문은 이 최적화 방식을 모방 학습에 도입하였다.

## 🛠️ Methodology

본 논문에서 제안하는 방법론의 명칭은 **AILSRS (Adversarial Imitation Learning through Simple Random Search)**이다. 전체 시스템은 전문가의 궤적을 학습하는 Discriminator($D_\phi$)와 이를 통해 보상을 얻어 최적화되는 Linear Policy($\pi_\theta$)로 구성된다.

### 1. Discriminator ($\text{D}_\phi$) 업데이트
에이전트의 궤적이 전문가의 궤적과 유사해지도록 유도하기 위해 Discriminator를 학습시킨다. 본 논문에서는 학습의 안정성을 높이기 위해 Sigmoid cross-entropy loss 대신 **LS-GAN(Least Squares GAN)**의 손실 함수를 사용한다.

Discriminator의 목적 함수는 다음과 같다:
$$\text{argmin}_\phi \mathcal{L}_{LS}(D) = \frac{1}{2} \mathbb{E}_{\pi_E}[(D_\phi(s,a) - b)^2] + \frac{1}{2} \mathbb{E}_{\pi_\theta}[(D_\phi(s,a) - a)^2]$$
여기서 $a=0$은 에이전트의 궤적에 대한 타겟 레이블, $b=1$은 전문가 궤적에 대한 타겟 레이블이다. LS-GAN 손실 함수는 결정 경계에서 멀리 떨어진 샘플에 대해서도 패널티를 부여하므로, 일반적인 GAN보다 더 정확한 보상 신호를 정책 네트워크에 전달할 수 있다.

### 2. Policy ($\pi_\theta$) 업데이트
정책 $\pi_\theta$는 단순한 선형 정책을 사용하며, ARS-V2 알고리즘을 통해 최적화된다. 에이전트는 환경의 직접적인 보상 대신 Discriminator가 제공하는 다음의 보상 신호를 최대화하도록 학습된다:
$$r^{\pi_\theta}(s,a) = -\log(1 - D_\phi(s,a))$$
즉, Discriminator가 해당 상태-행동 쌍이 전문가의 것일 확률이 높다고 판단할수록 에이전트는 더 높은 보상을 받게 된다.

### 3. 학습 절차 (Algorithm)
AILSRS의 상세 학습 과정은 다음과 같다:
1.  **파라미터 샘플링:** 현재 정책 파라미터 $\theta_t$ 주변에서 $N$개의 무작위 방향 $\delta_i$를 가우시안 분포에서 샘플링한다.
2.  **롤아웃 수집:** $\theta_t + \nu\delta_i$와 $\theta_t - \nu\delta_i$ 두 가지 변형된 정책을 사용하여 각각 궤적(Rollout)을 수집한다. 이때 상태 정규화(State Normalization)를 적용하여 서로 다른 범위의 상태 성분들이 정책에 동일한 영향을 주도록 한다.
3.  **Discriminator 학습:** 현재 수집된 에이전트의 궤적과 전문가의 궤적을 이용하여 LS-GAN 손실 함수를 통해 $\phi$를 업데이트한다.
4.  **Policy 업데이트:** 수집된 각 방향의 보상 차이를 이용하여 $\theta$를 업데이트한다:
    $$\theta_{t+1} = \theta_t + \frac{\alpha}{N\sigma_R} \sum_{i=1}^N [r(\pi_{t,i,+}) - r(\pi_{t,i,-})] \delta_i$$
    여기서 $\sigma_R$은 보상의 표준편차로, Adaptive step size를 조절하여 학습의 안정성을 높인다.

## 📊 Results

### 실험 설정
- **환경:** MuJoCo Locomotion tasks (HalfCheetah-v2, Hopper-v2, Walker-v2, Swimmer-v2, Ant-v2, Humanoid-v2)
- **비교 대상:** Behavior Cloning (BC), GAIL
- **평가 지표:** 전문가 궤적 대비 획득 보상 및 학습 곡선의 안정성
- **구현:** Python/TensorFlow, NVIDIA Titan XP GPU 사용

### 주요 결과
1.  **전문가 성능 도달:** AILSRS는 HalfCheetah, Hopper, Walker, Swimmer 작업에서 전문가의 보상 수준에 도달하거나 이를 상회하는 성능을 보였다. 특히 적은 수의 전문가 시연 데이터만으로도 합리적인 성능을 낼 수 있음을 확인하였다.
2.  **BC와의 비교:** BC는 전문가 데이터가 매우 많을 때만 성능이 향상되었으며, 데이터가 적을 때는 성능이 매우 낮고 불안정했다. 반면 AILSRS는 환경과의 상호작용을 통해 학습하므로 BC보다 훨씬 안정적이고 높은 성능을 보였다.
3.  **GAIL과의 비교:** AILSRS는 복잡한 TRPO/PPO 기반의 GAIL과 비교했을 때 경쟁력 있는 성능을 보여주었다. 특히 단순한 선형 정책과 Random Search만으로도 딥러닝 기반의 복잡한 모델과 유사한 성능을 낼 수 있음을 증명하였다.
4.  **학습 안정성:** 여러 개의 랜덤 시드(1, 3, 5개)를 사용한 실험에서 대부분의 작업(Ant, Humanoid 제외)이 빠르게 전문가 수준의 보상에 도달하는 안정적인 학습 곡선을 그렸다.

## 🧠 Insights & Discussion

본 논문은 복잡한 Deep RL 기반의 모방 학습이 반드시 필요한 것은 아니며, 적절한 적대적 구조와 단순한 최적화 기법의 결합만으로도 높은 성능을 낼 수 있음을 보여주었다.

**강점:**
- **재현성 확보:** 경사 하강법이나 복잡한 신경망 구조를 배제하고 Random Search와 Linear Policy를 사용함으로써, 하이퍼파라미터 민감도를 낮추고 결과의 재현성을 크게 높였다.
- **계산 효율성:** Backpropagation 과정이 필요 없는 Derivative-free 최적화를 사용하여 계산 복잡도를 줄였다.

**한계 및 논의사항:**
- **작업 난이도에 따른 성능 차이:** 상대적으로 단순한 Locomotion 작업에서는 성공적이었으나, Ant-v2나 Humanoid-v2와 같이 매우 복잡한 제어가 필요한 작업에서는 학습 속도가 매우 느려 경쟁력 있는 성능을 얻지 못했다. 이는 Linear Policy의 표현력 한계일 가능성이 크며, 향후 연구에서 정책 네트워크의 복잡도를 어떻게 조절할 것인지가 과제로 남는다.
- **On-policy 특성:** 본 방법론은 On-policy 알고리즘이므로 샘플 효율성 측면에서 Off-policy 방법론보다 불리할 수 있다.

## 📌 TL;DR

본 논문은 복잡한 Deep RL 기반 모방 학습의 재현성 문제를 해결하기 위해, **LS-GAN 기반의 Discriminator**와 **ARS(Augmented Random Search) 기반의 선형 정책 최적화**를 결합한 **AILSRS**를 제안한다. 실험 결과, MuJoCo의 주요 locomotion 작업에서 GAIL과 경쟁 가능한 성능을 보였으며, 단순한 구조 덕분에 높은 재현성과 계산 효율성을 확보하였다. 이는 향후 복잡한 제어 시스템에서 단순하면서도 강력한 모방 학습 가이드라인을 제공할 수 있을 것으로 기대된다.