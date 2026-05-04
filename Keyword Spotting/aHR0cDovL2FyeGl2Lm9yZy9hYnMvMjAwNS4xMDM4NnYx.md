# End-to-End Multi-Look Keyword Spotting

Meng Yu, Xuan Ji, Bo Wu, Dan Su, Dong Yu (2020)

## 🧩 Problem to Solve

본 논문은 원거리(Far-field) 및 소음 환경에서 Keyword Spotting (KWS) 시스템의 성능이 크게 저하되는 문제를 해결하고자 한다. 특히 다중 화자(Multi-talker) 환경에서는 간섭 신호로 인해 오경보(False Alarm)와 미검출(False Reject) 비율이 증가하는 경향이 있다.

기존의 다채널 처리 방식은 타겟 화자의 도착 각도(Direction of Arrival, DOA)를 정확히 추정하여 해당 방향의 신호를 추출하는 방식을 사용한다. 그러나 실제 환경, 특히 소음이 심한 다중 화자 환경에서는 타겟 화자의 정확한 DOA를 추정하는 것이 매우 어렵다. 따라서 본 연구의 목표는 타겟 화자의 위치 정보에 의존하지 않고도, 여러 후보 방향(Look directions)을 동시에 탐색하고 최적의 신호를 통합하여 KWS 성능을 높이는 End-to-End 모델을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Multi-look Neural Network Modeling**을 통한 음성 향상과 KWS 모델의 **Joint Training**이다. 

핵심 설계 방향은 다음과 같다.
1. **Multi-look Enhancement**: 특정 한 방향만을 추적하는 대신, 수평 평면상의 여러 샘플링된 방향(Look directions)을 동시에 처리하는 MLENet을 제안한다. 이를 통해 타겟 화자가 어느 방향에 있더라도 최소한 하나의 Look direction 채널에서 타겟 음성이 보존될 가능성을 높인다.
2. **Supervised Multi-look Learning**: 각 Look direction에 대해 가장 가까운 위치에 있는 음원(Source)을 학습 타겟으로 할당하는 지도 학습 방식을 도입하여, 신경망이 각 방향의 음원을 효과적으로 분리하도록 한다.
3. **Attention-based Fusion**: MLENet에서 출력된 여러 채널의 신호와 참조 마이크로폰의 신호를 Soft self-attention 메커니즘을 통해 동적으로 통합함으로써, 가장 신뢰할 수 있는 소스에 집중하여 KWS 입력으로 사용한다.
4. **End-to-End Optimization**: 음성 향상 네트워크(MLENet)와 KWS 모델을 통합하여 전체 시스템을 동시에 최적화함으로써, KWS 인식 정확도 향상에 최적화된 음성 향상이 이루어지도록 한다.

## 📎 Related Works

기존 연구들은 크게 단일 채널 음성 향상과 다채널 빔포밍(Beamforming) 방식으로 나뉜다. 단일 채널 방식은 다중 화자 환경에서 공간적 정보를 활용할 수 없다는 한계가 있으며, 전통적인 빔포밍 방식은 타겟 화자의 DOA를 미리 알고 있거나 정확히 추정해야 한다는 전제가 필요하다.

최근에는 고정 빔포머(Fixed Beamformer, FBF)를 사용하여 여러 고정 빔 채널을 생성하고 이를 KWS 입력으로 사용하는 연구가 진행되었다. 하지만 이러한 방식은 선형 필터의 한계로 인해 간섭 화자를 완전히 억제하기 어렵고, 각 채널을 개별적으로 평가하거나 단순히 결합하는 수준에 그쳐 최적의 신호 통합이 어렵다는 한계가 있다. 본 논문은 이를 개선하기 위해 학습 가능한 신경망 기반의 Multi-look 향상 모델을 제안하고, 이를 KWS 모델과 End-to-End로 연결하여 차별점을 둔다.

## 🛠️ Methodology

### 1. Direction-Aware Enhancement (DAE) 기반 구조
MLENet의 기초가 되는 DAE 구조는 다채널 입력 파형을 STFT를 통해 복소 스펙트로그램으로 변환한 후 다음의 특징들을 추출한다.
- **Logarithm Power Spectrum (LPS)**: 참조 채널의 에너지 정보를 나타내며 $\text{LPS} = \log(|Y_1|^2)$로 계산된다.
- **Inter-channel Phase Difference (IPD)**: 마이크로폰 쌍 사이의 위상 차이를 이용하여 공간 정보를 제공한다.
  $$\text{IPD}^{(m)}(t,f) = \angle Y_1^{(m)}(t,f) - \angle Y_2^{(m)}(t,f)$$
- **Directional Feature (DF)**: 특정 방향 $\theta$에 대한 타겟 화자의 바이어스(Bias)를 제공하며, 타겟 스티어링 벡터와 IPD 간의 코사인 유사도 평균으로 계산된다.
  $$d_\theta(t,f) = \sum_{m=1}^{M} \langle e^{\angle v^{(m)}_\theta(f)}, e^{\text{IPD}^{(m)}(t,f)} \rangle$$
여기서 $e(\cdot) = [\cos(\cdot), \sin(\cdot)]^T$이며, $d_\theta(t,f)$ 값이 1에 가까울수록 해당 T-F bin이 방향 $\theta$의 음원에 의해 지배됨을 의미한다.

### 2. Multi-Look Enhancement Network (MLENet)
MLENet은 수평 평면에서 $K$개의 샘플링된 방향 $\Theta_1, \dots, \Theta_K$에 대해 각각의 DF 벡터를 입력으로 받는다. 네트워크는 각 방향 $\Theta_k$에 대해 가장 가까운 음원 $x_k$를 예측하도록 학습된다. 학습 타겟 할당 방식은 다음과 같다.
$$\tilde{k} = \arg \min_j |\Theta_k - \theta_j|$$
여기서 $\theta_j$는 혼합 신호 내 실제 음원의 DOA이다. 즉, 각 Look direction 채널 $\hat{x}_k$는 그 방향과 가장 인접한 음원을 복원하도록 최적화된다. 손실 함수로는 Scale-Invariant Signal-to-Noise Ratio (SI-SNR)를 사용하여 각 채널의 복원 성능을 극대화한다.
$$L = \sum_{k=1}^{K} \text{SI-SNR}(\hat{x}_k, x_k)$$

### 3. End-to-End KWS 및 Attention Mechanism
MLENet의 $K$개 출력 채널과 1개의 참조 마이크로폰 채널(총 $K+1$개)의 fbank 특징 벡터 $z = [z_1, \dots, z_{K+1}]$를 입력으로 하여, Soft self-attention을 통해 단일 채널 특징 $\hat{z}$로 통합한다.
- **Attention Weight 계산**:
  $$e_i = v^T \tanh(W z_i + b)$$
  $$\alpha_i = \frac{\exp(e_i)}{\sum_{k=1}^{K+1} \exp(e_k)}$$
- **최종 통합 특징**:
  $$\hat{z} = \sum_{i=1}^{K+1} \alpha_i z_i$$
최종적으로 통합된 $\hat{z}$는 LWS-CNN 기반의 KWS 네트워크로 전달되며, 전체 시스템은 키워드 인식 정확도를 높이는 방향으로 공동 학습(Joint Training)된다.

## 📊 Results

### 실험 설정
- **데이터셋**: AISHELL-2 코퍼스를 사용하여 최대 3명의 화자가 포함된 가상 가청 환경을 시뮬레이션하였다. 6개의 마이크로폰 원형 배열(UCA)을 가정하였으며, 평균 $T_{60}$은 300ms, SIR은 $-12\text{dB}$에서 $12\text{dB}$ 사이로 설정하였다.
- **KWS 타겟**: 중국어 키워드 "ni-hao-wei-ling" 검출.
- **평가 지표**: 음성 향상 성능은 SI-SNR(dB)로 측정하였고, KWS 성능은 12시간 노출 시 오경보 1회 이하 조건에서의 Wake-up accuracy로 측정하였다.

### 주요 결과
1. **음성 향상 성능**: Table 1에 따르면, MLENet을 단독으로 사전 학습(Pre-train)했을 때보다 KWS 모델과 함께 Joint training 했을 때 SI-SNR 성능이 향상되었다. 이는 KWS 작업에 최적화된 음성 복원이 이루어졌음을 시사한다.
2. **KWS 정확도**: 
   - Baseline(raw+KWS) 대비 MLENet 기반 시스템은 매우 큰 성능 향상을 보였다.
   - 특히 저 SIR(SIR < 6dB) 환경에서 고정 빔포머(FBF) 기반 시스템보다 월등한 성능을 보였으며, 오라클(Oracle) DOA를 사용한 DAE 시스템에 근접하는 정확도를 달성하였다.
   - 참조 마이크로폰 채널을 추가한 `MLENet&mic KWS`가 가장 높은 성능(최대 94.5%)을 기록하였는데, 이는 Multi-look 채널에서 발생할 수 있는 음성 왜곡이나 타겟 누락(off-target) 문제를 참조 채널이 보완해주기 때문이다.

## 🧠 Insights & Discussion

본 논문은 타겟 화자의 위치를 모르는 상황에서도 다수의 후보 방향을 동시에 처리하고 이를 신경망으로 통합함으로써 원거리 KWS의 강건성을 확보하였다. 특히 고정 빔포머와 달리 학습 가능한 MLENet을 통해 비선형적인 잡음 제거 및 신호 복원이 가능함을 입증하였다.

**강점 및 한계**:
- **강점**: DOA 추정 단계 없이도 다중 화자 환경에서 효과적으로 타겟 키워드를 추출할 수 있으며, Attention 메커니즘을 통해 가변적인 환경에서 최적의 채널을 선택하는 능력을 갖추었다.
- **한계**: 실험 결과, 타겟 화자가 어떤 Look direction과도 충분히 가깝지 않아 출력이 누락되는 'off-target' 케이스가 약 9% 발생하였다. 이를 해결하기 위해 참조 마이크로폰 채널을 추가하여 보완하였으나, 샘플링 방향 $K$의 수와 성능 간의 트레이드-오프에 대한 더 깊은 분석이 필요해 보인다.

**비판적 해석**:
본 연구는 시뮬레이션 데이터에 크게 의존하고 있다. 실제 환경에서는 마이크로폰 배열의 불완전성이나 벽면 반사로 인한 복잡한 잔향이 존재하므로, 시뮬레이션 기반의 SI-SNR 향상이 실제 KWS 정확도로 얼마나 직접적으로 전이되는지에 대한 실데이터 검증이 추가된다면 더 설득력 있는 연구가 되었을 것이다.

## 📌 TL;DR

본 논문은 원거리 및 다중 화자 환경에서 KWS 성능을 높이기 위해, 여러 방향의 음원을 동시에 추출하는 **Multi-look Enhancement Network (MLENet)**와 **Attention 기반 통합 메커니즘**을 제안하고 이를 KWS 모델과 **End-to-End**로 학습시켰다. 실험 결과, 기존 고정 빔포머 방식보다 특히 소음 및 간섭이 심한 환경에서 우수한 Wake-up accuracy를 보였으며, 이는 향후 스마트 홈이나 차량용 음성 인식 시스템의 전처리 단계에 효과적으로 적용될 가능성이 높다.