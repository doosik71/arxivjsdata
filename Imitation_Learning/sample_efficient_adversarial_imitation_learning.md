# Sample-efficient Adversarial Imitation Learning

Dahuin Jung, Hyungyu Lee, Sungroh Yoon (2024)

## 🧩 Problem to Solve

본 논문은 보상 함수(reward function)를 사전에 정의하기 어려운 순차적 의사결정 작업에서 전문가의 시연(demonstration)을 통해 정책을 학습하는 Imitation Learning (IL), 특히 Adversarial Imitation Learning (AIL)의 샘플 효율성 문제를 해결하고자 한다.

기존의 IL 방법론들은 전문가의 행동을 성공적으로 모방하기 위해 여전히 매우 많은 양의 전문가 시연 데이터가 필요하다는 한계가 있다. 이를 해결하기 위해 이미지 도메인에서는 Self-supervised Representation Learning (SSL)을 통해 데이터 효율성을 높이는 연구가 진행되었으나, 이를 비-이미지 제어 작업(non-image control tasks, tabular data)에 그대로 적용할 경우 데이터의 의미론적/공간적 특성이 달라 Out-of-distribution (OOD) 샘플이 생성되거나 유용한 학습 신호를 얻지 못하는 문제가 발생한다. 따라서 본 연구의 목표는 비-이미지 제어 벤치마크에서도 적용 가능하며, 매우 적은 양의 전문가 데이터만으로도 효율적으로 학습할 수 있는 SSL 기반의 AIL 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 상태(state)와 행동(action)의 표현(representation)이 **다양한 왜곡(distortion)에 강건(robust)**하면서 동시에 **시간적으로 예측 가능(temporally predictive)**하도록 학습시키는 것이다. 이를 위해 다음과 같은 설계를 제안한다.

1. **시간적 예측 가능성 학습**: 현재의 상태 및 행동 표현으로부터 다음 상태의 표현을 예측하는 Forward Dynamics 모델을 도입하여, RL 작업에 필수적인 시간적 정보가 담긴 특징을 추출한다.
2. **Swapping Corruption 방법 제안**: 기존의 tabular 데이터 대상 SSL 방법론들이 OOD 데이터를 생성하는 문제를 해결하기 위해, 배치 내의 다른 샘플과 특징 인덱스를 교환하는 단순하면서도 효과적인 corruption 방법을 제안하여 In-distribution 내의 다양한 뷰를 생성한다.
3. **상보적 손실 함수 설계**: 시간적 정보를 극대화하는 InfoNCE 손실 함수와, 불필요한 잡음(nuisance variables)을 제거하여 강건성을 높이는 MSE 및 Barlow Twins 손실 함수를 결합하여 일반화 성능을 극대화한다.

## 📎 Related Works

### 관련 연구 및 한계

- **Adversarial Imitation Learning (AIL)**: GAIL, AIRL, VAIL 등은 전문가와 에이전트의 분포를 맞추는 적대적 학습을 통해 보상 함수를 추론한다. 하지만 학습 과정이 불안정하며, 여전히 많은 양의 전문가 궤적(trajectory)이 필요하다는 단점이 있다.
- **Self-supervised Learning (SSL)**: 이미지 도메인에서는 대조 학습(Contrastive Learning) 등이 성공적이었으나, tabular 데이터에 적용 가능한 VIME나 SCARF 같은 방법론들은 변형된 샘플이 실제 데이터 분포에서 크게 벗어나 성능 저하를 초래하는 경우가 많다.
- **Inverse Reinforcement Learning (IRL)**: 전문가의 비용 함수를 추론하는 방식이지만, 연속 제어 벤치마크에서 전문가 정책을 복구하기 위해 최소 5개 이상의 전체 궤적이 필요하다는 한계가 있다.

### 기존 접근 방식과의 차별점

본 논문은 기존 AIL 방법론들이 간과했던 전문가 데이터의 샘플 효율성 문제를 SSL을 통해 직접적으로 해결하려 한다. 특히, 단순한 데이터 증강이 아니라 시간적 역학(temporal dynamics)을 학습하는 보조 작업(auxiliary task)을 추가하고, tabular 데이터의 특성에 최적화된 Swapping Corruption을 통해 OOD 문제 없이 표현 학습의 효율을 높였다는 점에서 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조

본 모델은 다음과 같은 6개의 네트워크로 구성된다.

1. **Policy ($\pi_\theta$)**: 상태 $s$를 입력받아 행동 $a$를 생성한다.
2. **Value Function ($V$)**: 현재 정책 $\pi_\theta$를 평가하며, 판별자(Discriminator)가 제공하는 보상 $r$을 통해 학습된다.
3. **State Encoder ($SE$)**: 원시 상태 $s$를 특징 표현 $z_s$로 변환한다.
4. **Action Encoder ($AE$)**: 행동 $a$를 특징 표현 $z_a$로 변환한다.
5. **Forward Dynamics Model ($F$)**: 현재 상태 표현 $z_t^s$, 행동 표현 $z_t^a$, 그리고 가우시안 노이즈 $N$을 입력받아 왜곡된 다음 상태 표현 $\hat{z}_{t+1}^s$를 예측한다.
6. **Discriminator ($D_\omega$)**: 전문가 데이터와 에이전트 데이터의 표현 $(z_s \oplus z_a)$를 입력받아 이를 구분하며, 이는 RL 단계에서 비용 함수(cost function)로 사용된다.

### 학습 목표 및 손실 함수

전체 SSL 손실 함수 $\mathcal{L}_{SS}$는 다음과 같이 세 가지 손실의 가중 합으로 정의된다.
$$\mathcal{L}_{SS} = \lambda_F \mathcal{L}_F + \lambda_S \mathcal{L}_{SC} + \lambda_A \mathcal{L}_{AC}$$

#### 1. Forward Dynamics Loss ($\mathcal{L}_F$)

시간적으로 유의미한 정보를 학습하기 위해 InfoNCE 손실 함수를 사용한다. 예측된 $\hat{z}_{t+1}^s$와 실제 관측된 $z_{t+1}^s$ 사이의 일치성을 최대화하고, 배치 내의 다른 샘플들과의 일치성은 최소화한다.
$$\mathcal{L}_F = -\mathbb{E} \left[ \log \frac{e^{cs(\hat{z}_{i,t+1}^s, z_{i,t+1}^s)/\tau}}{\sum_{j \neq i} e^{cs(\hat{z}_{i,t+1}^s, z_{j,t}^s)/\tau} + \sum_{j \neq i} e^{cs(\hat{z}_{i,t+1}^s, z_{j,t+1}^s)/\tau}} \right]$$
여기서 $cs(u, v)$는 코사인 유사도를 의미한다.

#### 2. State Corruption Loss ($\mathcal{L}_{SC}$)

상태 표현의 강건성을 위해 MSE 손실 함수를 사용하여, 왜곡된 상태 $s'$의 표현 $z_s'$와 원본 상태 $s$의 표현 $z_s$가 유사해지도록 학습한다.
$$\mathcal{L}_{SC} = \mathbb{E} [ \| z_s - z_s' \|_2^2 ]$$

#### 3. Action Corruption Loss ($\mathcal{L}_{AC}$)

행동 표현의 경우 단순 MSE는 collapse(표현이 상수로 수렴하는 문제)가 발생할 수 있으므로, Barlow Twins 손실 함수를 사용하여 상호 상관 행렬(cross-correlation matrix)을 단위 행렬(identity matrix)에 가깝게 만들어 중복성을 줄이고 강건성을 확보한다.

### Swapping Corruption 방법

데이터를 왜곡하는 방식인 Swapping Corruption은 다음과 같이 동작한다.

1. 배치 내의 상태-행동 쌍들의 복사본을 만들고 무작위로 순서를 섞는다 ($\text{perm}(X_b^c)$).
2. 특정 비율 $c$만큼의 특징 인덱스 $I$를 무작위로 선택한다.
3. 원본 데이터의 해당 인덱스 값을 섞인 데이터의 값으로 교체하여 왜곡된 샘플 $x'$를 생성한다.
이 방식은 완전히 무작위인 값으로 채우는 대신 실제 존재하는 다른 샘플의 값을 사용하므로, 데이터 분포를 유지하면서도 다양한 뷰를 생성할 수 있다.

## 📊 Results

### 실험 설정

- **데이터셋**: MuJoCo (Ant-v2, HalfCheetah-v2, Hopper-v2, Swimmer-v2, Walker2d-v2) 및 Atari RAM (BeamRider, SpaceInvaders).
- **제한 조건**: 전문가 데이터셋을 100개의 상태-행동 쌍(state-action pairs) 이하로 매우 제한적으로 설정하여 샘플 효율성을 측정한다.
- **지표**: 누적 보상(Cumulative Reward) 및 전문가 정책과의 유사도.

### 주요 결과

1. **정량적 성과**: MuJoCo 벤치마크에서 기존 AIL 방법론(GAIL, AIRL, VAIL 등) 대비 평균적으로 약 39%의 성능 향상을 보였다. 특히 HalfCheetah에서는 단 100개의 샘플만으로 전문가 정책을 거의 완벽하게 모방하는 데 성공했다.
2. **Corruption 방법 비교**: 제안한 Swapping 방법이 Random, Mean, Each dim 방법보다 높은 보상을 기록했다. 또한, Local Outlier Factor (LOF) 분석을 통해 Swapping 방법이 다른 방법들보다 OOD 샘플을 적게 생성하면서도 충분한 다양성을 제공함을 입증했다.
3. **불완전한 시연(Imperfect Demonstrations)**: 전문가 데이터가 섞여 있는 상황에서 2IWIL 알고리즘과 결합했을 때, 기존의 CAIL이나 2IWIL 단독 사용보다 월등한 성능 향상을 보였다. 특히 Ant 환경에서는 2IWIL 대비 평균 288%의 성능 향상을 기록했다.
4. **이산 제어 작업**: Atari RAM 환경에서도 GAIL 대비 훨씬 높은 누적 보상을 달성하여, 제안 방법론의 확장성을 확인했다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 논문은 SSL의 두 가지 상충하는 목표인 '최대 정보 추출(InfoMax)'과 '불필요한 정보 제거(Nuisance removal)'를 적절히 결합했다. $\mathcal{L}_F$는 시간적 예측력을 통해 필요한 정보를 최대화하고, Corruption 기반의 $\mathcal{L}_{SC}, \mathcal{L}_{AC}$는 불필요한 변동성을 억제하여 일반화 성능을 높였다. 실험을 통해 이 두 요소가 상보적(complementary)으로 작용함을 확인하였으며, 특히 행동 표현의 경우 Barlow Twins 손실이 collapse 방지에 결정적임을 밝혔다.

### 한계 및 비판적 해석

1. **모델 복잡도 증가**: 표현 학습을 위해 상태/행동 인코더, Forward Dynamics 모델 등 3개의 추가 네트워크가 필요하며, 이로 인해 학습 시 계산 비용과 메모리 사용량이 증가한다.
2. **하이퍼파라미터 민감도**: Corruption rate $c$에 따라 성능 차이가 크게 나타나며, 특히 행동(action) 표현이 상태 표현보다 왜곡 비율에 훨씬 민감하게 반응한다. 이는 최적의 $c$ 값을 찾기 위한 추가적인 탐색 과정이 필요함을 의미한다.
3. **데이터 의존성**: 불완전한 시연 데이터의 비율이 너무 높을 경우(예: optimality 25%), SSL이 학습할 수 있는 유효 신호 자체가 부족해져 성능 향상 폭이 제한될 수 있다.

## 📌 TL;DR

본 논문은 비-이미지 제어 작업에서 전문가 데이터가 극도로 부족할 때의 학습 효율을 높이기 위해 **시간적 예측 가능성**과 **왜곡 강건성**을 동시에 학습하는 SSL 기반의 Adversarial Imitation Learning 방법론을 제안한다. 특히 In-distribution 데이터를 유지하는 **Swapping Corruption** 기법과 **InfoNCE, Barlow Twins** 손실 함수를 통해, 단 100개의 샘플(1개 미만의 전체 궤적)만으로도 기존 AIL 방법론들을 압도하는 성능을 달성했다. 이 연구는 데이터 수집 비용이 높은 실제 로봇 제어 등의 분야에서 전문가 시연 데이터를 최소화하며 고성능 정책을 학습시키는 데 중요한 기여를 할 가능성이 크다.
