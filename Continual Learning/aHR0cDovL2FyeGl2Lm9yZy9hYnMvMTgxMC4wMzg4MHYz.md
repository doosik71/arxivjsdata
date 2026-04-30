# Continual State Representation Learning for Reinforcement Learning using Generative Replay

Hugo Caselles-Dupré, Michael Garcia-Ortiz, David Filliat (2018)

## 🧩 Problem to Solve

본 논문은 강화학습(Reinforcement Learning, RL)을 위한 상태 표현 모델(State Representation model)을 지속적인 학습(Continual Learning) 방식으로 구축하는 문제를 다룬다. 에이전트가 실세계와 같이 시간이 지남에 따라 환경이 변하는 상황에서 동작할 때, 새로운 환경의 정보를 효율적으로 압축하여 학습하면서도 과거에 습득한 지식을 잊지 않고 유지하는 것이 핵심이다.

일반적으로 신경망을 이용한 상태 표현 학습(State Representation Learning, SRL) 모델은 확률적 경사 하강법(SGD)을 통해 학습되는데, 학습 데이터의 분포가 변경될 때 과거의 지식을 급격히 망각하는 Catastrophic Forgetting 문제가 발생한다. 따라서 본 연구의 목표는 과거 데이터에 직접 접근하지 않고도 시스템 크기를 일정하게 유지하며, 환경 변화를 자동으로 감지하고 과거의 지식을 보존하는 지속적인 상태 표현 학습 체계를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 아이디어는 Variational Auto-Encoders(VAEs)의 인코딩 능력과 생성 능력을 결합하여 Catastrophic Forgetting을 방지하는 것이다. 구체적인 기여 사항은 다음과 같다.

1.  **Generative Replay의 도입**: VAE가 학습한 과거 환경의 잠재 공간(Latent Space)에서 샘플을 생성하고, 이를 새로운 환경의 데이터와 함께 학습함으로써 과거의 지식을 복구하고 유지하는 Generative Replay 기법을 SRL에 적용하였다.
2.  **자동 환경 변화 감지**: VAE의 재구성 오차(Reconstruction Error) 분포에 대해 통계적 검정을 수행함으로써, 사용자의 개입 없이 환경의 변화를 자동으로 감지하는 일반적이고 통계적으로 견고한 방법을 제안하였다.
3.  **효율적인 상태 표현 및 전이 학습**: 제안 방법이 Catastrophic Forgetting을 방지할 뿐만 아니라, 과거의 지식을 이용해 새로운 작업을 더 잘 수행하게 하는 Forward Transfer 효과가 있음을 입증하였다.

## 📎 Related Works

논문에서는 상태 표현 학습(SRL)과 지속적 학습(CL)의 관련 연구를 언급한다. VAE는 입력을 연속적인 잠재 표현으로 매핑하는 능력이 있어 RL의 상태 표현 모델로 유망하며, 최근 generative 모델을 위한 CL 접근 방식들이 제안되고 있다. 특히 생성된 샘플을 사용하여 망각을 방지하는 Generative Replay 기술이 주목받고 있다.

기존 연구인 DARLA(Higgins et al., 2017)는 특정 VAE 아키텍처를 통해 얽힘이 해제된 표현(Disentangled Representations)을 학습함으로써 환경의 미세한 변화에 강건한 특징을 추출하여 Catastrophic Forgetting을 우회하려 했다. 반면, 본 논문의 접근 방식은 환경 변화가 감지될 때마다 특징 표현을 지속적으로 업데이트하며, 생성 모델을 통해 과거 지식을 명시적으로 유지한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조
제안하는 시스템은 크게 **환경 변화 감지 $\rightarrow$ 데이터 생성 및 수집 $\rightarrow$ VAE 지속 학습 $\rightarrow$ RL 정책 학습**의 순서로 동작한다. 에이전트는 VAE를 통해 고차원의 감각 상태(Sensory State)를 저차원의 벡터 표현으로 압축하며, 이 특징값이 RL 알고리즘(PPO)의 입력으로 사용된다.

### 주요 구성 요소 및 상세 설명

**1. Generative Replay를 이용한 지속 학습**
환경이 변화하면, 이전 환경에서 학습된 VAE의 잠재 공간에서 샘플을 생성한다. 이 생성된 샘플들을 새로운 환경에서 수집된 데이터와 결합하여 VAE를 다시 학습시킨다. 이 방식은 과거의 원본 데이터를 저장할 필요가 없으므로 메모리 사용량이 제한된 Bounded System Size를 유지할 수 있다.

**2. 자동 환경 변화 감지 (Automatic Environment Change Detection)**
VAE의 재구성 오차 분포를 기반으로 환경 변화를 감지한다. 구체적으로, 서로 다른 두 샘플 집단의 평균 재구성 오차가 동일하다는 귀무가설($H_0$)을 검정하기 위해 Welch's t-test를 사용한다. 두 집단의 분산이 동일하다고 가정할 수 없기에 표준 t-test 대신 Welch's t-test를 선택하였다. 통계량 $t$와 자유도 $\nu$는 다음과 같이 계산된다.

$$t = \frac{x_1 - x_2}{\sqrt{(s_1^2 + s_2^2)/N}}$$
$$\nu \approx \frac{(N-1)(s_1^2 + s_2^2)^2}{s_1^4 + s_2^4}$$

여기서 $x_1, x_2$는 각 샘플의 평균 재구성 오차, $s_1, s_2$는 표준편차, $N$은 샘플 수이다. p-value가 기준치(0.01)보다 작으면 환경이 변화한 것으로 판단한다.

**3. 훈련 절차 및 손실 함수**
- **VAE 구조**: 3개의 1-D Convolutional layers와 1개의 Fully Connected layer로 구성된 Encoder와 Decoder를 사용하며, 잠재 표현의 크기는 64로 설정하였다.
- **Inverse KL Annealing**: 표준적인 KL Annealing(0에서 1로 증가) 대신, KL 항의 가중치를 1에서 시작하여 0으로 서서히 감소시키는 Inverse Annealing 방식을 적용하여 재구성 성능을 높였다.
- **RL 학습**: 상태 표현으로 추출된 VAE 특징값을 입력으로 하여 Proximal Policy Optimization (PPO) 알고리즘을 통해 정책을 학습하였다.

## 📊 Results

### 실험 설정
- **환경**: Flatland 2-D 환경을 사용하였으며, 환경 1(빨간색 먹이)에서 환경 2(초록색 먹이)로 전이되는 시나리오를 구성하였다.
- **비교 대상**: Raw pixels(압축 없음), Fine-tuning(단순 전이 학습), Generative Replay(제안 방법).
- **지표**: 재구성 품질을 측정하는 Mean Squared Error(MSE)와 RL 에이전트의 누적 보상(Mean Reward)을 사용하였다.

### 주요 결과
1.  **환경 변화 감지**: Welch's t-test 기반의 감지 방법은 환경 변화가 있을 때 100%, 없을 때 99.5%의 정확도를 보여 매우 효율적임을 확인하였다.
2.  **재구성 성능 (MSE)**: Fine-tuning 방식은 환경 2를 학습한 후 환경 1에 대한 재구성 능력을 상실(MSE 급증)하였으나, Generative Replay는 환경 1과 2 모두에서 낮은 MSE를 유지하며 Catastrophic Forgetting을 성공적으로 방지하였다.
3.  **RL 성능**:
    - **상태 표현의 효율성**: Raw pixels를 사용하는 것보다 VAE 특징값을 사용하는 것이 최종 성능과 샘플 효율성 면에서 우수하였다.
    - **Zero-shot Transfer**: 환경 1에서 학습된 VAE 특징만을 사용하여 환경 2의 작업을 학습시키는 것이 매우 효율적이었다.
    - **Forward Transfer**: Generative Replay를 통해 두 환경을 모두 학습한 모델이 환경 2에서 가장 높은 성능을 보였는데, 이는 과거의 학습 경험이 새로운 환경의 학습을 돕는 Forward Transfer가 발생했음을 시사한다.

## 🧠 Insights & Discussion

본 연구는 생성 모델의 특성을 이용하여 과거 데이터를 직접 저장하지 않고도 지식을 유지할 수 있음을 보여주었다. 특히, 단순한 Fine-tuning이 초래하는 급격한 망각 문제를 Generative Replay로 해결함으로써, RL 에이전트가 지속적으로 변화하는 환경에서도 안정적인 상태 표현을 유지할 수 있게 하였다.

**강점 및 의의**:
- 과거 데이터 저장 공간이 필요 없는 Bounded system size를 달성하였다.
- 통계적 검정을 통해 하이퍼파라미터 설정 없이 일반적인 환경 변화 감지가 가능하다.
- 단순한 망각 방지를 넘어, 이전 지식이 이후 학습에 긍정적인 영향을 주는 Forward Transfer 효과를 확인하였다.

**한계 및 향후 과제**:
- 실험이 비교적 단순한 2-D 환경에서 수행되었으며, 환경 변화가 이산적(Discrete)으로 발생한다는 가정하에 진행되었다.
- 향후 연구에서는 랜덤하게 생성되는 미로(Maze)와 같이 더 복잡한 설정이나, 연속적으로 변화하는 환경(Non-discrete changes)으로 확장할 필요가 있다.

## 📌 TL;DR

본 논문은 RL의 상태 표현 학습에서 발생하는 Catastrophic Forgetting 문제를 해결하기 위해 **VAE와 Generative Replay**를 결합한 지속적 학습 프레임워크를 제안한다. VAE의 생성 능력을 이용해 과거 데이터를 복원함으로써 메모리 효율적으로 지식을 유지하며, **Welch's t-test**를 통해 환경 변화를 자동으로 감지한다. 실험 결과, 제안 방법은 망각을 방지할 뿐만 아니라 이전 지식을 활용해 새로운 환경을 더 빠르게 학습하는 **Forward Transfer** 능력을 보였으며, 이는 지속적으로 변화하는 환경에서 동작하는 자율 에이전트 구현에 중요한 기여를 할 수 있다.