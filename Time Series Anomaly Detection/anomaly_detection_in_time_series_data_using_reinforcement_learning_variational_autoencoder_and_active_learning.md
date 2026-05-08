# Anomaly Detection in Time Series Data Using Reinforcement Learning, Variational Autoencoder, and Active Learning

Bahareh Golchin, Banafsheh Rekabdar (2025)

## 🧩 Problem to Solve

본 논문은 시계열 데이터에서 이상치(Anomaly)를 탐지하는 과정에서 발생하는 기존 방법론들의 한계점을 해결하고자 한다. 시계열 이상치 탐지는 데이터 센터, 센서 네트워크, 금융 등 다양한 도메인에서 매우 중요하지만, 다음과 같은 핵심적인 문제점들이 존재한다.

첫째, 기존의 많은 방법론들이 수동적인 파라미터 튜닝과 특징 추출에 의존하며, 이는 데이터의 특성에 따라 성능 편차가 크고 많은 노동력을 요구한다. 둘째, 이상치는 본질적으로 희소(rare)하기 때문에, 모델을 효과적으로 학습시키기 위한 레이블링된 데이터가 부족하다. 셋째, 실제 환경의 데이터 분포는 시간이 지남에 따라 변화하므로, 정적인 모델은 새로운 유형의 이상치를 탐지하는 데 한계가 있다.

따라서 본 연구의 목표는 매우 적은 양의 레이블링된 데이터만으로도 새로운 유형의 이상치를 효과적으로 탐지하고, 데이터의 변화에 적응할 수 있는 자동화된 약지도 학습(Weakly-supervised learning) 기반의 이상치 탐지 시스템을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Deep Reinforcement Learning (DRL), Variational Autoencoder (VAE), 그리고 Active Learning을 통합한 **RLVAL** 프레임워크를 제안한 것이다. 주요 설계 아이디어는 다음과 같다.

1. **DRL-VAE 통합**: VAE를 통해 정상 데이터의 분포를 학습하고, 여기서 발생하는 Reconstruction Error를 DRL 에이전트의 내재적 보상(Intrinsic Reward)으로 사용하여 레이블이 없는 데이터에서도 새로운 이상치 패턴을 탐색하게 한다.
2. **Active Learning의 도입**: 데이터 레이블링 비용을 최소화하기 위해, 모델이 가장 불확실해하는 샘플만을 선택적으로 전문가에게 질의하는 Margin Sampling 기법을 적용하여 학습 효율을 극대화한다.
3. **LSTM 기반 시퀀스 모델링**: 시계열 데이터의 장기 의존성(Long-term dependencies)을 효과적으로 캡처하기 위해 LSTM 네트워크를 DRL 에이전트의 핵심 구조로 사용한다.

## 📎 Related Works

논문에서는 기존의 이상치 탐지 방법을 통계 기반 방법과 머신러닝 기반 방법으로 구분하여 설명한다.

- **통계 기반 방법**: Gaussian 모델이나 커널 함수 기반 방법들이 있으며, 주로 데이터가 특정 분포를 따른다는 가정하에 작동한다. 그러나 실제 데이터는 이러한 가정을 벗어나는 경우가 많아 오탐률(False-positive rate)이 높다는 한계가 있다.
- **머신러닝 기반 방법**: SVM, Bayesian networks, k-means 클러스터링 등이 사용된다. 지도 학습 방식은 성능이 좋지만 충분한 레이블 데이터가 필요하며, 분포 변화 시 재학습이 강제된다는 단점이 있다.
- **딥러닝 및 RL 기반 방법**: RNN, LSTM, Autoencoder 등이 시퀀스 데이터 예측 및 재구성 오차를 이용해 이상치를 탐지한다. 최근에는 DQN과 같은 DRL 기반의 접근법이 등장하여 자기 주도적 학습 가능성을 보여주었으나, 본 논문은 여기에 VAE의 생성 능력과 Active Learning의 효율성을 결합하여 기존 RLAD 등의 모델보다 발전된 성능을 구현하고자 한다.

## 🛠️ Methodology

### 전체 시스템 구조

RLVAL 프레임워크는 VAE를 통한 이상치 점수 생성, DQN 기반의 의사결정 에이전트, 그리고 전문가의 피드백을 받는 Active Learning 모듈로 구성된다.

### 1. Variational Autoencoder (VAE)를 이용한 탐지

VAE는 정상 데이터만을 사용하여 학습된다. 인코더는 입력 데이터를 잠재 공간(Latent space)의 가우시안 분포로 변환하고, 디코더는 이를 다시 복원한다. VAE의 학습 목표는 다음과 같은 ELBO(Evidence Lower Bound)를 최대화하는 것이다.

$$L(\theta, \phi; x) = \langle \log p(x|z; \theta) \rangle_{q(z|x; \phi)} - KL[q(z|x; \phi) \| p(z)]$$

여기서 정상 데이터는 복원 오차가 낮게 나타나지만, 이상치 샘플은 학습된 정상 패턴에서 벗어나므로 높은 복원 오차를 보인다. 이 오차는 다음과 같이 계산되며, DRL 에이전트의 보상 신호로 활용된다.

$$\text{Reconstruction Error} = \|x - x'\|^2$$

### 2. Deep Reinforcement Learning (DQN) 프레임워크

에이전트는 현재 상태 $s$에서 이상치 여부를 판단하는 액션 $a$ (이상치 $a_1$ 또는 정상 $a_0$)를 선택한다. 최적의 정책 $\pi^*$는 누적 기대 보상을 최대화하는 방향으로 학습된다.

$$\pi^* = \arg \max_{\pi} E_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]$$

에이전트는 Bellman 방정식을 통해 Q-value를 업데이트하며, 학습의 안정성을 위해 Experience Replay와 Target Network를 사용한다.

$$Q^{\pi}(s, a) \leftarrow Q^{\pi}(s, a) + \alpha \left( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right)$$

### 3. 보상 함수 (Reward Function)

본 모델의 핵심은 외재적 보상($r_1$)과 내재적 보상($r_2$)의 결합이다.

- **외재적 보상 ($r_1$)**: 레이블링된 데이터($D_{la}$)를 올바르게 분류했을 때 부여된다.
    $$r_1 = \begin{cases} 1 & \text{if } a=a_1 \text{ and } s \in D_{la} \\ 0 & \text{if } a=a_0 \text{ and } s \in D_u \\ -1 & \text{otherwise} \end{cases}$$
- **내재적 보상 ($r_2$)**: VAE의 복원 오차를 기반으로 하며, 레이블이 없는 데이터($D_u$) 내에서 새로운 이상치 후보를 탐색하도록 유도한다.

최종 보상은 $r = r_1 + r_2$로 정의되어, 알려진 이상치의 활용(Exploitation)과 새로운 이상치의 탐색(Exploration) 사이의 균형을 맞춘다.

### 4. Active Learning (Margin Sampling)

레이블링 비용을 줄이기 위해 Margin Sampling을 적용한다. 에이전트가 예측한 두 액션($a_0, a_1$)의 Q-value 차이를 계산하여, 그 차이가 가장 작은(즉, 가장 불확실한) 샘플을 전문가에게 질의한다.

$$\text{min margin} = \min |q_0 - q_1|$$

전문가가 레이블링한 데이터는 다시 학습 풀에 추가되어 모델의 성능을 점진적으로 향상시킨다.

## 📊 Results

### 실험 설정

- **데이터셋**: Yahoo A1Benchmark (실제 트래픽 데이터 및 합성 데이터) 및 KPI 데이터셋 (IT 운영 데이터)을 사용하였다.
- **비교 대상**: Unsupervised 모델(SPOT, SR-CNN, Autoencoder) 및 Semi-supervised 모델(RLAD)과 비교하였다.
- **평가 지표**: Precision, Recall, F1-score를 사용하였다.

### 주요 결과

실험 결과, RLVAL은 모든 데이터셋에서 기존 모델들보다 우수한 성능을 보였다.

1. **Yahoo A1Benchmark**: 레이블링된 데이터의 비율을 1%, 5%, 10%로 조정하며 실험한 결과, RLVAL은 10% 데이터 사용 시 **F1-score 0.921**을 기록하여, RLAD의 최고 성능(0.797)을 크게 상회하였다.
2. **KPI 데이터셋**: 극소량의 레이블(0.05% $\sim$ 0.1%)만 사용했음에도 불구하고, 0.1% 레이블 사용 시 **F1-score 0.908**을 달성하였다. 특히 Unsupervised 방법들이 0.170 이하의 낮은 F1-score를 보인 것과 대조적으로 압도적인 성능 향상을 보였다.

이 결과는 RLVAL이 최소한의 레이블링 데이터만으로도 VAE의 내재적 보상과 Active Learning의 효율적 질의를 통해 매우 정밀하게 이상치를 탐지할 수 있음을 입증한다.

## 🧠 Insights & Discussion

본 논문은 VAE의 생성 모델 특성과 DRL의 의사결정 능력을 결합하여, 약지도 학습 환경에서의 이상치 탐지 문제를 효과적으로 해결하였다. 특히 단순히 데이터의 재구성 오차에 의존하는 Autoencoder 방식과 달리, 이를 RL의 보상 체계로 편입시킴으로써 에이전트가 능동적으로 이상치 패턴을 학습하게 만든 점이 인상적이다.

또한, Active Learning의 Margin Sampling을 통해 무작위 레이블링이 아닌 '가장 정보 가치가 높은' 샘플을 선택함으로써, 인간 전문가의 개입 비용을 획기적으로 줄이면서도 모델의 강건성을 확보하였다.

다만, 본 연구는 전문가의 레이블링이 항상 정확하다는 가정하에 진행되었으므로, 실제 환경에서 전문가의 실수(Human error)가 발생했을 때의 대응 방안은 명시되지 않았다. 또한, LSTM을 사용한 시퀀스 모델링이 구체적으로 어떤 하이퍼파라미터 설정 하에 최적의 성능을 냈는지에 대한 상세한 아키텍처 수치는 부족한 편이다.

## 📌 TL;DR

본 논문은 적은 레이블 데이터로 시계열 이상치를 탐지하기 위해 **VAE-DRL-Active Learning**을 결합한 **RLVAL** 프레임워크를 제안하였다. VAE의 복원 오차를 DRL의 내재적 보상으로 활용해 새로운 이상치를 탐색하고, Margin Sampling 기반의 Active Learning으로 효율적인 레이블 확장을 구현하였다. 실험 결과, Yahoo 및 KPI 데이터셋에서 기존 SOTA 모델인 RLAD 및 다양한 비지도 학습 모델들을 압도하는 F1-score를 기록하며 그 실효성을 증명하였다. 이 연구는 데이터 레이블링 비용이 높은 산업 현장의 시계열 모니터링 시스템에 즉시 적용 가능한 높은 잠재력을 가진다.
