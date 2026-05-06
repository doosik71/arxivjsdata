# MRNet: a Multi-scale Residual Network for EEG-based Sleep Staging

Xue Jiang (2021)

## 🧩 Problem to Solve

본 논문은 뇌파(Electroencephalogram, EEG) 신호를 이용한 자동 수면 단계 분류(Sleep Staging) 시스템의 성능 향상을 목표로 한다. 수면 단계 분류는 수면 장애의 임상 진단과 치료에 필수적이지만, 숙련된 전문가가 24시간 분량의 EEG 데이터를 분석하는 데 약 5시간이 소요될 정도로 노동 집약적인 작업이다.

기존의 딥러닝 기반 자동 분류 시스템은 다음과 같은 두 가지 핵심적인 문제를 가지고 있다.
첫째, 네트워크의 층이 깊어짐에 따라 발생하는 **정보 손실(Information Loss)** 문제이다. 일반적인 심층 신경망은 마지막 컨볼루션 층의 특징(feature)만을 분류에 사용하는데, 이 과정에서 이전 층들이 보유했던 세부적인 신호 정보가 사라져 표현력이 저하된다.
둘째, 분류 결과에서 나타나는 **출력 지터(Output Jitters)** 현상이다. 전문가마다 라벨링 기준이 주관적이기 때문에 발생하는 라벨 노이즈로 인해, 실제 생리학적 수면 주기(Sleep Cycle)와는 다르게 수면 단계가 매우 빈번하고 불규칙하게 변하는 예측 결과가 도출되는 문제가 있다.

## ✨ Key Contributions

본 논문은 위 문제들을 해결하기 위해 **MRNet**이라는 새로운 프레임워크를 제안하며, 핵심 기여 사항은 다음과 같다.

1. **Multi-scale Feature Fusion (MFF) 모델**: 네트워크의 서로 다른 깊이에서 추출된 특징들을 결합하여 특징 피라미드(Feature Pyramid)를 구축함으로써, 다양한 스케일의 신호 정보를 보존하고 활용한다.
2. **Adaptive Channel Fusion (ACF) 모듈**: MFF 과정에서 발생하는 정보 중복을 제거하고, 각 채널의 고유한 특징을 강화하기 위해 채널 주의(Channel Attention) 메커니즘을 도입하였다.
3. **Markov-based Sequential Correction (MSC) 알고리즘**: 수면 단계 전이 규칙(Sleep Stage Transition Rule)과 마르코프 체인(Markov Chain)을 결합하여, 생리학적으로 불가능하거나 희박한 단계 전이를 필터링함으로써 예측 결과의 일관성을 높였다.

## 📎 Related Works

자동 수면 단계 분류 연구는 크게 두 가지 방향으로 진행되어 왔다.
첫째는 **수작업 특징 기반 모델(Handcrafted feature-based models)**으로, STFT(Short-Time Fourier Transform)나 DWT(Discrete Wavelet Transform)를 통해 특징을 추출한 후 SVM과 같은 분류기에 입력하는 방식이다. 이는 전문가의 도메인 지식이 필요하며 튜닝 과정이 복잡하다는 한계가 있다.
둘째는 **딥러닝 기반 모델(Deep learning-based models)**로, CNN이나 LSTM 등을 이용하여 EEG 신호에서 자동으로 특징을 추출하는 방식이다. 예를 들어, U-Time은 Encoder-Decoder 구조의 fully convolutional network를 사용하였고, DeepSleepNet과 IITNet은 CNN-RNN 구조를 통해 시간적 관계를 학습하였다.

본 논문은 이러한 기존 딥러닝 모델들이 네트워크 전파 과정에서의 정보 손실을 간과하고, 수면 단계 간의 생리학적 전이 규칙을 고려하지 않은 채 각 에포크(epoch)를 독립적인 샘플로 처리한다는 점을 차별점으로 삼는다.

## 🛠️ Methodology

### 1. Backbone Architecture

MRNet의 백본은 1D 잔차 블록(Residual Block) 기반 네트워크로 구성된다. 총 19개의 컨볼루션 층과 9개의 잔차 블록을 포함하며, 필터 길이는 32로 설정하여 넓은 수용 영역(reception field)을 확보함으로써 생리학적 신호의 시간적 정보를 효과적으로 포착한다.

정보 이론 관점에서 입력 신호 $x_n$의 엔트로피를 $H(\cdot)$라고 할 때, 잔차 블록 전후의 정보량은 다음과 같이 정의된다.

- 입력 정보량: $I_0 = H(x_n)$
- 컨볼루션 층 통과 후 정보량: $I_1 = H(F(x_n))$
- 잔차 블록 출력 정보량: $I_2 = H(F(x_n) + x_n)$

정보 처리 정리(Information Processing Theorem)에 의해 $I_0 \ge I_1$이 성립하므로, 컨볼루션 층을 거치며 정보 손실이 발생한다. MRNet은 skip connection을 통해 $I_0$의 상한선을 회복함으로써 이러한 정보 손실을 완화한다.

### 2. Multi-scale Feature Fusion (MFF)

네트워크가 깊어질수록 신호의 추상화 수준은 높아지지만 세부 정보는 손실된다. 이를 해결하기 위해 본 논문은 FPN(Feature Pyramid Network)에서 영감을 얻은 MFF 모델을 제안한다.

- **구조**: 백본의 5번째, 7번째, 9번째 잔차 블록에서 추출된 특징 $\{C_1, C_2, C_3\}$를 사용한다.
- **Adaptive Channel Fusion (ACF)**: 각 특징 맵이 ACF 모듈을 통과하여 중요도가 조정된 $\{W_1, W_2, W_3\}$가 된다. ACF는 $\text{Conv1D} \to \text{Global Average Pooling} \to \text{FC(ReLU)} \to \text{FC(Sigmoid)}$ 순으로 구성되어 채널별 가중치를 학습한다.
- **융합 방식**: 업샘플링 후 채널 결합(Channel Concatenation)을 통해 최종 특징 $P_3$를 생성하고 이를 분류기에 입력한다.

### 3. Markov-based Sequential Correction (MSC)

분류기의 raw 예측 결과에서 발생하는 지터를 제거하기 위해 수면 주기 규칙을 적용한 후처리 알고리즘이다.

**1-order forward checking**
수면 단계 전이 확률 행렬 $M$을 기반으로 하며, 전이 확률의 변동성을 조절하기 위해 다음과 같이 전처리된 행렬 $M'$를 사용한다.
$$M' = \text{norm}(G(M)^r), r \ge 1$$
여기서 $G(\cdot)$는 압축 함수(compression function)이며, $r$은 행렬 요소 간의 상대적 크기를 조절하는 상수이다.

**n-order backward checking**
현재 단계의 결정이 이후 $n$개의 단계에 영향을 받는다는 가정하에 관성 계수(Inertia factor) $w_i$를 계산한다.
$$w_i = \sum_{j=1}^{n} \delta_j a^{-j}, (a \ge 1)$$
$$\delta_j = \begin{cases} 1 & \text{if } C_{i+j} = C_{i-1} \\ -1 & \text{if } C_{i+j} = C_i \end{cases}$$
여기서 $a$는 감쇠 상수이며, $\delta_j$는 변화 방향을 나타낸다.

**최종 보정 절차**
예측 단계가 변경되는 시점($C_i \neq C_{i-1}$)에서, 기존의 분류 확률 $P_i$에 전이 확률 $R$(행렬 $M'$의 행)과 관성 계수 $w_i$를 곱하여 보정된 확률 $P'_i$를 구하고, 이를 통해 최종 단계 $C'_i$를 결정한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Sleep-EDF (건강한 피험자 20명), Sleep-EDFx (건강한 사람 및 경미한 수면 장애 환자 포함 100명).
- **지표**: 정확도(Acc), Macro-F1 score(MF1), 클래스별 F1 score.
- **검증**: 10-fold 교차 검증을 수행하였다.

### 주요 결과

MRNet은 기존 SOTA 모델들과 비교하여 우수한 성능을 보였다.

- **Sleep-EDF (Fpz-Cz 채널)**: Acc 87.59%, MF1 79.62%를 기록하여 DeepSleepNet(82.00% Acc)과 IITNet(83.60% Acc)을 상회하였다.
- **Sleep-EDFx (Fpz-Cz 채널)**: Acc 85.14%, MF1 78.91%를 달성하여 U-Time(83.16% Acc)보다 높은 성능을 보였다.

### 분석 결과

1. **채널 간 비교**: Fpz-Cz 채널의 성능이 Pz-Oz 채널보다 높게 나타났는데, 이는 데이터 분포의 편향 때문으로 분석된다.
2. **데이터셋 간 비교**: Sleep-EDFx의 성능이 Sleep-EDF보다 낮은 이유는 수면 장애 환자의 데이터가 포함되어 분포가 더 복잡하기 때문이다.
3. **Ablation Study**:
    - MFF 모델 적용 시 모든 클래스에서 성능이 향상되었다.
    - MSC 알고리즘은 N3 단계의 성능을 일부 희생시키는 대신, REM 단계의 예측 결과를 대폭 개선하여 전체적인 일관성을 높였다.
    - 하이퍼파라미터 실험 결과, 압축 함수 $G(\cdot)$로는 $\log$ 함수가 가장 적합했으며, 관성 계수의 차수 $n=4$일 때 가장 높은 정확도를 보였다.

## 🧠 Insights & Discussion

본 논문의 강점은 단순한 모델 깊이의 증가가 아닌, **신호의 다중 스케일 특성(MFF)**과 **생리학적 전이 규칙(MSC)**이라는 도메인 지식을 딥러닝 구조에 성공적으로 통합했다는 점이다. 특히, 딥러닝 모델이 흔히 간과하는 '시계열 데이터의 연속성'과 '전이 가능성'을 마르코프 체인으로 해결하여 실용적인 예측 결과를 도출하였다.

다만, 한계점으로는 MSC 알고리즘이 특정 단계(N3)의 성능을 희생시켜 다른 단계(REM)를 개선한다는 점이 언급된다. 이는 마르코프 전이 확률이 통계적 평균에 의존하기 때문에, 개인별 수면 패턴의 특수성을 완전히 반영하지 못할 가능성이 있음을 시사한다. 또한, 단일 채널 EEG만을 사용하였는데, 다채널 EEG를 활용했을 때 MFF가 어떻게 확장될 수 있을지에 대한 논의가 부족하다.

## 📌 TL;DR

MRNet은 EEG 기반 수면 단계 분류에서 발생하는 **정보 손실**과 **예측 지터** 문제를 해결하기 위해 **다중 스케일 특징 융합(MFF)**과 **마르코프 기반 순차 보정(MSC)** 알고리즘을 제안한 연구이다. 실험 결과 Sleep-EDF 및 Sleep-EDFx 데이터셋에서 SOTA 성능을 달성하였으며, 이는 딥러닝의 특징 추출 능력과 생리학적 전이 규칙의 결합이 자동 수면 분석의 신뢰성을 높이는 데 매우 효과적임을 입증한다.
