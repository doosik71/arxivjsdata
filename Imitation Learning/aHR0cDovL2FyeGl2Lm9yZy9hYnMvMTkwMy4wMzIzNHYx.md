# Dyna-AIL : Adversarial Imitation Learning by Planning

Vaibhav Saxena, Srinivasan Sivanandan, Pulkit Mathur (2019)

## 🧩 Problem to Solve

본 논문은 Adversarial Imitation Learning(적대적 모방 학습) 방법론이 가진 낮은 샘플 효율성 문제를 해결하고자 한다. GAIL과 같은 기존의 적대적 모방 학습 방식들은 최적의 정책에 수렴하기 위해 환경과의 방대한 상호작용(environment interactions)을 필요로 한다. 그러나 실제 많은 제어 시스템이나 물리적 환경에서 환경과의 상호작용은 비용이 매우 높거나 시간이 오래 걸리는 작업이다. 따라서 본 연구의 목표는 환경과의 상호작용 횟수를 획기적으로 줄이면서도 전문가의 정책을 효과적으로 모방할 수 있는 효율적인 학습 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Model-Based Planning(모델 기반 계획)과 Model-Free Learning(모델 프리 학습)을 교차하여 수행하는 Dyna-like 프레임워크를 적대적 모방 학습에 도입한 것이다. 구체적으로, 환경의 동역학을 예측하는 Forward-model을 학습시키고, 이를 통해 가상의 궤적(imaginary trajectories)을 생성하여 정책을 업데이트하는 Planning 단계를 추가함으로써 실제 환경에서의 데이터 수집 부담을 줄였다. 즉, 실제 환경에서 얻은 데이터로 학습하는 Model-Free 방식과 학습된 모델을 통해 시뮬레이션하며 학습하는 Model-Based 방식을 결합하여 수렴 속도를 높인 것이 핵심 기여점이다.

## 📎 Related Works

본 논문은 다음과 같은 기존 연구들을 배경으로 한다.

- **GAIL (Generative Adversarial Imitation Learning):** GAN의 구조를 차용하여 생성자(Generator)는 정책 $\pi$가 되고, 판별자(Discriminator) $D$는 주어진 상태-행동 쌍이 전문가의 데이터인지 학습자의 데이터인지 판별한다. 하지만 Model-Free 방식이므로 환경 상호작용 비용이 매우 크다는 한계가 있다.
- **MGAIL (End-to-end differentiable Adversarial Imitation Learning):** Forward-model을 도입하여 판별자의 그래디언트를 상태(state)를 통해 정책 네트워크까지 전달함으로써 분산을 줄이고 학습 효율을 높이려 했다. 그러나 MGAIL 역시 손실 함수 계산을 위해 여전히 실제 환경에서 샘플링한 궤적에 의존한다는 점에서 완전한 Model-Based 방식은 아니라고 분석한다.

Dyna-AIL은 이러한 기존 방식들과 달리, 모델을 통해 생성한 가상 데이터를 직접적으로 정책 업데이트에 활용하는 Planning 단계를 명시적으로 포함함으로써 환경 상호작용 의존도를 더욱 낮췄다.

## 🛠️ Methodology

### 전체 시스템 구조 및 파이프라인

Dyna-AIL은 학습 단계(Learning phase)와 계획 단계(Planning phase)를 반복적으로 수행하는 구조를 가진다.

1. **Learning Phase (Model-Free):** 실제 환경에서 정책 $\pi$를 통해 궤적을 샘플링하고, 이를 이용해 판별자 $D$와 정책 $\pi$를 업데이트한다.
2. **Forward-model Training:** 실제 환경에서 수집된 데이터를 Experience Replay Buffer $B$에 저장하고, 이를 사용하여 환경의 상태 전이 모델 $f$를 학습시킨다.
3. **Planning Phase (Model-Based):** 학습된 Forward-model $f$를 사용하여 가상의 궤적을 생성하고, 이 가상 데이터를 이용해 정책 $\pi$를 추가로 업데이트한다.

### 주요 구성 요소 및 수식 설명

**1. 목표 함수 (Objective Function)**
판별자 $D$와 정책 $\pi$는 다음과 같은 적대적 목적 함수를 기반으로 학습된다.
$$\min_{\pi} \max_{D \in (0,1)} \mathbb{E}_{\pi} [\log(D(s,a))] + \mathbb{E}_{\pi_E} [\log(1-D(s,a))]$$
여기서 $\mathbb{E}_{\pi}$는 학습자의 정책에서 생성된 기대치이며, $\mathbb{E}_{\pi_E}$는 전문가의 데이터에서 생성된 기대치이다.

**2. Forward-model 및 손실 함수**
모델 $f$는 현재 상태 $s$와 행동 $a$를 입력받아 다음 상태의 변화량(delta)을 예측하도록 설계되었다.
$$s'_{pred} = s + f(s,a)$$
모델의 학습을 위한 손실 함수 $L_f$는 예측된 상태 변화량과 실제 관찰된 상태 변화량 사이의 평균 제곱 오차(MSE)를 최소화하는 방향으로 정의된다.
$$L_f = \sum_{(s,a,s') \in B} \frac{1}{2} ||f(s,a) - (s' - s)||^2$$

**3. 학습 및 계획 절차 (Algorithm 1)**

- **판별자 업데이트:** 실제 환경에서 샘플링한 궤적 $\tau_i$와 전문가 궤적 $\tau_E$를 사용하여 $D$의 파라미터 $\theta_d$를 업데이트한다.
- **정책 업데이트 (Model-Free):** 실제 환경 데이터 $\tau_i$에 대해 판별자 $D$의 출력을 보상으로 간주하여 $\pi$의 파라미터 $\theta_g$를 업데이트한다.
- **정책 업데이트 (Model-Based Planning):** 학습된 모델 $f$를 통해 생성한 가상 궤적 $\tau_j$를 사용하여 $\pi$를 업데이트한다. 이때, 안정성을 위해 가상 궤적의 길이를 $T_p$로 제한한다.

### 네트워크 구현 세부사항

- **아키텍처:** 판별자는 2개의 은닉층(200, 100 units), 정책 네트워크는 2개의 은닉층(100, 50 units)과 ReLU 활성화 함수를 사용한다.
- **상태-행동 임베딩:** 상태와 행동의 분포가 다르므로 각각의 인코더를 통해 공유 공간으로 투영한 뒤, Hadamard product를 통해 결합하여 모델의 입력으로 사용한다.
- **GRU 적용:** 환경을 $n$차 MDP로 모델링하기 위해 상태 인코더에 GRU 레이어를 추가하여 이전 상태들의 정보를 유지한다.

## 📊 Results

### 실험 설정

- **데이터셋 및 작업:** 이산 제어 작업인 CartPole과 연속 제어 작업인 MuJoCo 시뮬레이터의 Hopper, HalfCheetah를 사용하였다.
- **전문가 데이터:** TRPO 알고리즘으로 학습된 전문가의 궤적 50개를 사용하였으며, 각 궤적의 길이는 $N=1000$이다.
- **평가 지표:** 환경과의 상호작용 횟수(number of model-free trajectories) 대비 획득한 보상을 측정하였다.

### 주요 결과

- **수렴 속도 향상:** CartPole, Hopper, HalfCheetah 모든 작업에서 Dyna-AIL이 MGAIL보다 훨씬 적은 수의 실제 환경 상호작용만으로도 최적 정책에 도달함을 확인하였다.
- **작업별 특성:**
  - **HalfCheetah:** 낮은 분산과 함께 매우 빠르게 수렴하였다.
  - **Hopper:** MGAIL 대비 빠른 수렴을 보였으나, 성능의 분산(variance)이 크게 나타났다. 이는 Forward-model이 복잡한 Hopper 환경의 상태 공간을 완전히 학습하지 못해 발생하는 모델 편향(bias) 때문으로 분석된다.

## 🧠 Insights & Discussion

### 강점 및 한계

- **강점:** Forward-model을 통한 Planning 단계를 도입함으로써 샘플 효율성을 극대화하였으며, 이는 환경 상호작용 비용이 높은 실제 시스템에 적용 가능성이 높음을 시사한다.
- **한계:** 모델의 정확도에 따라 정책 성능이 크게 좌우된다. 특히 Hopper와 같이 상태 공간이 복잡한 경우, 모델의 편향이 정책의 불안정성으로 이어진다.

### 비판적 해석 및 논의

- **Planning 길이 ($T_p$)의 영향:** 실험 결과 $T_p$가 커질수록(예: $T_p=50$) Hopper 환경에서의 분산이 증가하였다. 이는 모델의 오차가 시간이 지남에 따라 누적되어 가상 궤적이 실제와 동떨어지기 때문이며, 이를 해결하기 위해 모델의 용량(capacity)을 키우거나 더 정교한 학습 방법이 필요하다.
- **최적화 알고리즘의 선택:** 논문은 TRPO 업데이트 룰을 적용했을 때 Hopper 환경에서의 분산이 감소함을 보여주었다. 이는 KL-divergence 제약 조건이 노이즈가 섞인 그래디언트 상황에서도 안정적인 업데이트를 가능하게 하기 때문이다.
- **적응형 스위칭의 필요성:** 현재는 학습과 계획을 단순히 교차 수행하고 있다. 하지만 모델의 불확실성(uncertainty)을 측정하여, 모델이 확신하는 상태에서는 Planning을 수행하고 불확실한 상태에서만 실제 환경을 쿼리하는 'Adaptive Switching' 메커니즘을 도입한다면 효율성을 더 높일 수 있을 것이다.

## 📌 TL;DR

본 논문은 적대적 모방 학습(Adversarial Imitation Learning)의 고질적인 문제인 낮은 샘플 효율성을 해결하기 위해, Model-Based Planning과 Model-Free Learning을 결합한 **Dyna-AIL** 프레임워크를 제안한다. 학습된 Forward-model을 통해 가상 궤적을 생성하고 이를 정책 업데이트에 활용함으로써, 실제 환경과의 상호작용 횟수를 획기적으로 줄이면서도 전문가의 성능에 빠르게 수렴함을 입증하였다. 이 연구는 향후 모델 기반 강화학습과 모방 학습의 결합 및 적응형 샘플링 전략 연구에 중요한 기반이 될 수 있다.
