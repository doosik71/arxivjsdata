# Out-of-Dynamics Imitation Learning from Multimodal Demonstrations

Yiwen Qiu, Jialong Wu, Zhangjie Cao, Mingsheng Long (2022)

## 🧩 Problem to Solve

일반적인 모방 학습(Imitation Learning)은 시연자(Demonstrator)와 모방자(Imitator)가 동일한 다이내믹스(Dynamics), 즉 동일한 상태 공간(State space), 액션 공간(Action space), 그리고 상태 전이 모델(Transition model)을 공유한다는 가정을 전제로 한다. 하지만 이러한 엄격한 가정은 실제 환경에서 시연 데이터를 수집하기 어려운 경우 모방 학습의 활용도를 크게 제한한다.

본 논문은 시연자와 모방자가 상태 공간은 공유하지만, 액션 공간과 다이내믹스는 서로 다를 수 있다는 가정을 가진 Out-of-Dynamics Imitation Learning (OOD-IL) 문제를 다룬다. OOD-IL 환경에서는 다양한 시연자의 데이터를 활용할 수 있다는 장점이 있으나, 다이내믹스의 차이로 인해 모방자가 물리적으로 수행 불가능한 '비전이성 시연(Non-transferable demonstrations)'이 포함된다는 새로운 도전 과제가 발생한다. 특히, 서로 다른 다이내믹스를 가진 여러 시연자가 섞여 있을 경우 시연 데이터가 다봉 분포(Multimodal distribution)를 띠게 되며, 이는 기존의 단봉(Unimodal) 정책 기반 전이성 측정 방식으로는 정확한 필터링이 불가능하게 만든다. 따라서 본 연구의 목표는 다봉 분포를 갖는 시연 데이터에서 모방자가 수행 가능한 데이터만을 정확히 식별하여 학습 효율을 높이는 전이성 측정(Transferability measurement) 방법을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 시연 데이터의 다봉 분포를 먼저 분해하고, 분해된 각 모드 내에서 전이성을 개별적으로 측정하는 것이다. 이를 위해 다음과 같은 두 가지 핵심 설계를 제안한다.

1. **Sequence-based Contrastive Clustering**: 시연 궤적(Trajectory)들의 내재된 다봉 분포를 처리하기 위해, 대조 학습(Contrastive Learning)과 클러스터링을 결합하여 동일한 모드에 속하는 궤적들을 하나의 클러스터로 묶는 알고리즘을 설계하였다.
2. **Adversarial Transferability Measurement**: 각 클러스터별로 GAIL(Generative Adversarial Imitation Learning)의 판별자(Discriminator)를 학습시켜, 해당 시연 데이터가 모방자의 다이내믹스로 재현 가능한지를 측정하는 전이성 지표를 도출하였다.

## 📎 Related Works

기존의 모방 학습은 주로 Behavior Cloning (BC), Inverse Reinforcement Learning (IRL), GAIL 등으로 나뉘며, 모두 동일 다이내믹스 가정을 전제로 한다. OOD-IL 문제를 해결하기 위해 일부 연구는 시연자와 모방자 사이의 대응 모델(Correspondence model)을 학습시키려 했으나, 이는 상태 공간과 액션 공간 사이에 일대일 대응 관계가 존재해야 한다는 제약이 있어 7-DoF 로봇과 3-DoF 로봇 간의 모방과 같은 상황에서는 적용이 불가능하다.

최근에는 상태 공간만을 공유하는 방식의 연구가 진행되었으며, 특히 f-MDP를 통해 전이성 측정치를 학습하여 비전이성 데이터를 가중치 낮게 처리하는 방식이 제안되었다. 그러나 f-MDP는 다봉 분포의 데이터를 하나의 단봉 정책으로 모델링하려 하기 때문에 정확도가 떨어지며, 단계별 최적화 과정으로 인해 학습 속도가 느리고 최적화가 어렵다는 한계가 있다. 본 논문은 이러한 한계를 극복하기 위해 클러스터링과 적대적 학습 기반의 측정 방식을 도입하여 차별성을 갖는다.

## 🛠️ Methodology

본 논문은 전체 파이프라인을 두 단계(클러스터링 $\rightarrow$ 전이성 측정)로 구성하며, 최종적으로 가중치가 적용된 데이터를 사용하여 정책을 학습한다.

### 1. Sequence-based Contrastive Clustering

시연 데이터 $\Xi$의 다봉 분포를 제거하기 위해, RNN 기반의 특징 추출기 $F$와 거리 측정법을 학습한다. 동일한 궤적에서 무작위로 추출된 두 개의 부분 궤적(Sub-trajectories)을 양성 쌍(Positive pair)으로, 서로 다른 궤적의 부분 궤적들을 음성 쌍(Negative pair)으로 설정하여 대조 학습을 수행한다.

대조 학습 손실 함수 $L_{contrast}$는 다음과 같다.
$$L_{contrast} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\langle F(\xi_{sub_{2i-1}}), F(\xi_{sub_{2i}}) \rangle)}{\sum_{i' \neq 2i-1, i' \neq 2i} \exp(\langle F(\xi_{sub_{2i-1}}), F(\xi_{sub_{i'}}) \rangle) + \exp(\langle F(\xi_{sub_{2i-1}}), F(\xi_{sub_{2i}}) \rangle)}$$
여기서 $\langle \cdot, \cdot \rangle$은 코사인 유사도를 의미한다. 여기에 클러스터 중심 $c_k$와의 거리를 최소화하는 클러스터링 목적 함수를 결합하여 최종 손실 함수 $L_{cluster}$를 정의한다.
$$L_{cluster} = L_{contrast} + \frac{\lambda}{2} \| F(\xi_{sub_n}) - Cy_n \|^2_2$$
이를 통해 궤적들을 $K$개의 클러스터 $\Xi_k$로 분리하여 각 클러스터가 단봉 분포를 갖도록 유도한다.

### 2. Adversarial Transferability Measurement

분리된 각 클러스터 $\Xi_k$에 대해 GAIL 기반의 적대적 학습을 수행한다. GAIL의 판별자 $D_k$는 입력된 상태 전이가 시연 데이터에서 왔는지, 아니면 모방자의 정책 $\pi_k$에서 왔는지를 구분한다.

판별자의 손실 함수 $L_{tran}$은 다음과 같다.
$$L_{tran} = -\sum_{k=1}^{K} \left( \mathbb{E}_{(s^d_t, s^d_{t+1}) \sim \Xi_k} \log(1 - D_k(s^d_t, s^d_{t+1})) + \mathbb{E}_{(s^{\pi_k}_t, s^{\pi_k}_{t+1}) \sim \pi_k} \log(D_k(s^{\pi_k}_t, s^{\pi_k}_{t+1})) \right)$$
학습이 완료된 후, 판별자 $D_k$의 출력값이 1에 가까울수록 해당 상태 전이가 모방자의 정책에 의해 생성되었을 가능성이 높으며, 이는 곧 모방자가 수행 가능한 '전이 가능한(Transferable)' 데이터임을 의미한다. 특정 상태 전이의 전이성 가중치 $w$는 다음과 같이 계산된다.
$$w(s^d_t, s^d_{t+1}) = \sum_{k=1}^{K} I[(s^d_t, s^d_{t+1}) \in \Xi_k] D_k(s^d_t, s^d_{t+1})$$

### 3. Transferability-sampling Imitation Learning

계산된 가중치 $w$를 정규화하여 샘플링 분포 $p_w$를 생성한다.
$$p_w(s^d_t, s^d_{t+1}) = \frac{w(s^d_t, s^d_{t+1})}{\sum_{(s^{d'}_t, s^{d'}_{t+1}) \in \Xi} w(s^{d'}_t, s^{d'}_{t+1})}$$
최종적으로 이 샘플링 분포 $p_w$를 사용하여 가중치가 적용된 GAIL 학습을 수행함으로써, 전이 가능한 데이터에 더 집중하여 정책 $\pi$를 학습한다.
$$L_{GAIL} = -\mathbb{E}_{(s^d_t, s^d_{t+1}) \sim p_w} \log(1 - D(s^d_t, s^d_{t+1})) - \mathbb{E}_{(s^\pi_t, s^\pi_{t+1}) \sim \pi} \log(D(s^\pi_t, s^\pi_{t+1}))$$

## 📊 Results

### 실험 환경 및 설정

- **MuJoCo**: HalfCheetah(다리 힘 조절), Hopper(중력 가속도 변경), Walker2d(발 마찰력 변경) 환경에서 각기 다른 다이내믹스를 가진 4명의 시연자를 설정하였다.
- **Driving**: 차량의 속도와 장애물 너비를 다르게 설정하여 현실적인 다봉 분포를 모사하였다.
- **Simulated Franka Panda Arm**: 로봇 팔의 특정 조인트(Joint)를 비활성화하여 서로 다른 물리적 능력을 가진 시연자와 모방자를 설정하였다.
- **비교 대상**: Naive GAIL, ID-Random, ID-GAIL, f-MDP.

### 주요 결과

- **정량적 성과**: 모든 환경에서 제안 방법이 가장 높은 기대 수익(Expected Return)을 달성하였다. 특히 다봉 분포가 심한 Driving과 Franka Panda 환경에서 타 베이스라인 대비 압도적인 성능 향상을 보였다.
- **베이스라인 분석**: f-MDP는 다봉 분포 데이터에서 단봉 정책을 학습하려다 실패하였으며, ID(Inverse Dynamics) 기반 방식은 랜덤 궤적을 사용한 경우(ID-Random) 전이성 측정의 정확도가 매우 낮았다.
- **Ablation Study**: 클러스터링을 제거한 경우($Ours\ w/o\ Cluster$)와 전이성 측정까지 제거한 경우($Ours\ w/o\ Cluster, Tran$) 모두 성능이 하락하였다. 이는 다봉 분포를 분해하는 클러스터링과 비전이성 데이터를 걸러내는 전이성 측정 모두가 필수적임을 입증한다.

## 🧠 Insights & Discussion

본 논문은 OOD-IL 문제에서 가장 큰 걸림돌인 **'데이터의 다봉성'**과 **'물리적 불가능성'**을 체계적으로 해결하였다. 특히 단순한 필터링이 아니라, 대조 학습을 통한 잠재 공간(Latent space) 구성 $\rightarrow$ 클러스터링 $\rightarrow$ 적대적 전이성 측정으로 이어지는 파이프라인을 통해 데이터의 질을 높인 점이 강점이다.

**한계 및 논의사항:**

1. **계산 비용**: 클러스터 수 $K$가 매우 많아질 경우, 각 클러스터마다 GAIL 모델을 학습시켜야 하므로 계산 비용이 증가한다. 저자는 이를 위해 메타 학습(Meta-learning)이나 사전 학습(Pre-training)의 가능성을 제시하였다.
2. **안전성 문제**: 전이성을 측정하는 과정에서 모방자가 환경과 상호작용하며 탐색(Exploration)을 수행해야 하므로, 실제 물리 시스템에 적용할 때 안전 제약 조건을 준수하지 못할 위험이 있다.
3. **최적성 가정**: 본 연구는 수집된 시연 데이터가 타겟 환경에서 최적(Optimal)이라는 가정을 하고 있으나, 실제로는 서브-옵티멀(Sub-optimal)한 데이터가 섞여 있을 가능성이 크다. 이는 향후 연구 과제로 남겨져 있다.

## 📌 TL;DR

본 논문은 시연자와 모방자의 다이내믹스가 다른 상황(OOD-IL)에서, 다봉 분포를 갖는 시연 데이터 중 모방자가 실제로 수행 가능한 데이터만을 골라내어 학습하는 방법을 제안한다. **대조 학습 기반의 궤적 클러스터링**으로 데이터를 모드별로 분리하고, **GAIL 판별자를 통해 전이성을 측정**하여 데이터에 가중치를 부여하는 방식이다. 이를 통해 다양한 소스에서 수집된 불균일한 데이터셋을 효과적으로 정제하여 고성능의 정책을 학습할 수 있음을 입증하였으며, 이는 향후 대규모 이종 데이터셋을 활용한 로봇 학습에 중요한 기여를 할 것으로 보인다.
