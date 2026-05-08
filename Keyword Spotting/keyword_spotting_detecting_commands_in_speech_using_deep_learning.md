# Keyword spotting - Detecting commands in speech using deep learning

Sumedha Rai, Tong Li, Bella Lyu (2023)

## 🧩 Problem to Solve

본 연구가 해결하고자 하는 문제는 음성 신호 내에서 특정 명령어인 '키워드(keyword)'를 식별하는 Keyword Spotting(KWS) 작업이다. 음성 인식 기술은 인간과 컴퓨터의 상호작용(Human-Computer Interaction)을 가능하게 하는 핵심 기술이며, 특히 로봇 공학, 가상 비서, 차량 탑재 전자 장치와 같은 환경에서 특정 명령어를 통해 시스템을 제어하는 것이 매우 중요하다.

KWS 시스템은 효율적인 작동을 위해 높은 인식 정확도를 유지하면서도, 지연 시간(latency)이 낮아야 하며, 특히 모바일 기기와 같이 계산 자원이 제한된 환경에서도 구동 가능해야 한다는 제약 조건이 있다. 따라서 본 논문의 목표는 공개 데이터셋인 Google Speech Commands Dataset을 활용하여 전통적인 통계 모델과 최신 딥러닝 모델들의 성능을 비교 분석하고, 음성의 순차적 특성을 가장 잘 반영하는 최적의 아키텍처를 찾는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 다양한 머신러닝 및 딥러닝 알고리즘을 동일한 KWS 작업에 적용하여 그 효용성을 정량적으로 비교한 점이다. 특히 다음과 같은 설계 아이디어를 통해 접근하였다.

1. **특성 공학(Feature Engineering)의 적용**: 원시 파형(raw waveform)을 인간의 청각 특성을 반영한 Mel Frequency Cepstral Coefficients(MFCCs)로 변환하여 모델의 입력값으로 사용하였다.
2. **모델 계층적 비교**: 전통적인 Hidden Markov Model with Gaussian Mixture(HMMGMM)를 베이스라인으로 설정하고, Convolutional Neural Networks(CNN), 그리고 Recurrent Neural Networks(RNN)의 다양한 변형(LSTM, BiLSTM, Attention)을 순차적으로 적용하여 성능 향상을 검증하였다.
3. **순차적 특성 반영**: 음성 데이터의 시계열적 특성을 처리하기 위해 단방향 LSTM에서 양방향 LSTM(BiLSTM) 및 Attention 메커니즘으로 확장하며 성능 변화를 분석하였다.

## 📎 Related Works

논문에서는 음성 인식 분야의 전통적 방법론과 현대적 방법론을 다음과 같이 소개한다.

- **전통적 접근 방식**: Hidden Markov Models(HMM)는 음성 신호를 시계열 모델로 처리하는 전통적인 방식으로 널리 사용되어 왔다. 하지만 고정된 통계적 분포에 의존하는 한계가 있다.
- **딥러닝 접근 방식**: 최근에는 CNN이 특징 추출기로 빈번하게 사용되며, Graves et al. (2013)은 LSTM RNN을 음성 인식에 도입하여 긴 의존성을 처리하였다. 또한, Convolutional Recurrent Network와 Attention을 결합한 구조가 KWS 작업에서 제안된 바 있으며, 영어와 중국어의 소음 및 억양 문제를 해결하기 위한 복잡한 아키텍처들이 개발되었다.

본 연구는 이러한 기존 연구들을 바탕으로, 특히 단순한 키워드 분류 작업에서 각 모델 구조(CNN vs RNN)가 주는 이점을 명확히 비교하는 데 초점을 맞추고 있다.

## 🛠️ Methodology

### 전체 파이프라인 및 데이터 전처리

본 연구는 35개의 단일 단어 명령어로 구성된 Google Speech Commands Dataset v2를 사용한다. 데이터는 80:10:10의 비율로 학습, 검증, 테스트 세트로 분할된다.

입력 데이터 전처리는 모델에 따라 다르게 적용된다. HMMGMM과 RNN 계열 모델은 원시 오디오를 12개 성분의 MFCCs로 변환하여 사용한다. MFCCs는 인간의 귀가 저주파수 대역의 변화에는 민감하고 고주파수 대역에는 둔감하다는 특성을 반영하여, 저주파에서는 선형적으로, 고주파에서는 로그 스케일로 필터를 배치해 계산한다. 반면 CNN 모델은 자체적인 특징 추출 능력을 활용하기 위해 MFCCs 대신 원시 오디오를 $16,000\text{Hz}$에서 $8,000\text{Hz}$로 리샘플링하여 입력으로 사용한다.

### 사용된 알고리즘 및 아키텍처

#### 1. Hidden Markov Model with Gaussian Mixture (HMMGMM)

베이스라인 모델로 사용되며, $N$개의 은닉 상태(hidden states)와 상태당 $M$개의 관측치(observations)를 정의한다.

- **구성 요소**: 상태 전이 확률 행렬 $A$($A_{ij}$는 상태 $i$에서 $j$로 전이할 확률), 방출 확률 행렬 $B$($B_i(k)$는 상태 $j$에서 관측치 $k$가 생성될 확률), 초기 상태 분포 $\pi$로 구성된다. 이때 방출 확률은 Gaussian Mixture 분포를 따른다.
- **학습 절차**: Expectation-Maximization(EM) 알고리즘을 사용하여 다음의 확률을 최대화한다.
  $$P(x|\theta) = \int_{z} P(x, z|\theta)dz$$
  - **E-step**: 고정된 파라미터 $\theta^{old}$ 하에서 $q(z) = P(z|x, \theta^{old})$를 최대화한다.
  - **M-step**: $Q(\theta, \theta^{old}) = \int_{z} P(z|x, \theta^{old}) \log P(x, z|\theta)$를 최대화하여 파라미터를 갱신한다.

#### 2. Convolutional Neural Networks (CNN)

M5 CNN 구조를 채택하였으며, 입력 데이터를 Bag-of-words 스타일로 처리한다.

- **구조**: `Conv(kernel 80) $\rightarrow$ BatchNorm $\rightarrow$ MaxPool(4)` 과정을 시작으로, 이후 `Conv(kernel 3) $\rightarrow$ BatchNorm $\rightarrow$ MaxPool(4)` 과정을 3회 반복한다. 마지막으로 Fully-Connected 레이어와 Softmax를 통해 35개 클래스 중 하나를 예측한다.
- **학습 설정**: Optimizer는 Adam, Learning rate는 $0.01$, 손실 함수는 Cross-entropy를 사용하며 총 10 epoch 동안 학습한다.

#### 3. Recurrent Neural Networks (RNN)

음성의 순차적 특성을 반영하기 위해 LSTM 및 그 변형을 사용한다.

- **LSTM**: Vanishing Gradient 문제를 해결하기 위해 Forget, Input, Output 게이트를 사용한다.
  - Forget gate: $f_t = \sigma(U_f h_{t-1} + W_f x_t)$
  - Cell state update: $c_t = \sigma(U_i h_{t-1} + W_i x_t) \odot \tanh(U_g h_{t-1} + W_g x_t) + f_t \odot c_{t-1}$
  - Hidden state: $h_t = \sigma(U_o h_{t-1} + W_o x_t) \odot \tanh(c_t)$
- **BiLSTM with Attention**: 단방향 LSTM을 넘어 양방향으로 정보를 처리하는 BiLSTM을 적용하고, Attention 메커니즘을 추가하였다. Attention은 고정된 마지막 은닉 상태만을 사용하는 대신, 입력 시퀀스의 모든 은닉 상태를 유지하고 중요한 영역(명령어가 실제로 존재하는 구간)에 가중치를 두어 정보를 추출한다. 본 연구에서는 Convolution과 BiLSTM 뒤에 Dense 레이어와 Softmax로 구성된 Attention 레이어를 배치한 구조를 사용하였다.

## 📊 Results

### 실험 설정

- **데이터셋**: Google Speech Commands Dataset v2 (35개 단어, 105,829개 발화)
- **평가 지표**: Accuracy (정확도)
- **분할**: Train/Val/Test = 80% / 10% / 10%

### 정량적 결과

실험 결과, 딥러닝 모델들이 전통적인 통계 모델인 HMMGMM보다 압도적인 성능 향상을 보였다.

- **HMMGMM**: 가장 낮은 성능을 기록한 베이스라인 모델이다.
- **CNN**: HMMGMM 대비 절대 정확도 기준 $38.7\%$의 성능 향상을 보였다.
- **RNN**: CNN보다 더 높은 성능을 기록하였으며, 특히 순차적 특성을 고려하는 RNN 구조가 KWS 작업에 더 적합함을 입증하였다 (CNN 대비 약 $20.3\% \sim 22.3\%$ 향상).
- **최종 성능**: RNN with BiLSTM and Attention 모델이 **$93.9\%$**의 정확도로 가장 높은 성능을 달성하였다.

## 🧠 Insights & Discussion

### 모델별 성능 분석

CNN이 HMMGMM보다 월등한 성능을 보인 이유는 컨볼루션 레이어가 음성 신호에서 중요한 특징을 효과적으로 추출했기 때문으로 분석된다. 또한, RNN 계열이 CNN보다 더 높은 성능을 기록한 것은 음성 데이터가 본질적으로 시계열(sequential) 데이터이기 때문에 시간적 의존성을 학습하는 RNN의 구조가 더 유리했음을 시사한다.

### Attention 메커니즘의 실효성

BiLSTM과 Attention을 추가했을 때 성능 향상이 있었으나, 단방향 LSTM 대비 그 상승폭은 약 $2\%$로 매우 미미하였다. 이에 대해 저자들은 두 가지 가능성을 제시한다.

1. LSTM의 Forget/Add 게이트가 이미 이전 상태의 정보를 충분히 잘 유지하고 있어 Attention의 기여도가 낮았을 가능성이 있다.
2. Attention 메커니즘은 본 연구와 같은 단일 출력(Single-output) 분류 모델보다는 Encoder-Decoder 구조의 Sequence-to-Sequence 모델에서 더 효과적일 가능성이 있다.

### 오류 분석 (Confusion Matrix)

Attention RNN의 혼동 행렬을 분석한 결과, 발음이 유사한 단어들 사이에서 오분류가 빈번하게 발생함을 확인하였다.

- 예: 'tree' $\leftrightarrow$ 'three', 'forward' $\leftrightarrow$ 'four'
이러한 문제는 단어의 음소(phoneme)가 유사하여 발생하는 것으로, 향후 모델에 문맥(context) 정보를 추가함으로써 해결할 수 있을 것이라고 논의한다.

## 📌 TL;DR

본 논문은 35개의 음성 명령어를 분류하는 KWS 작업에서 HMMGMM, CNN, RNN(LSTM, BiLSTM, Attention) 모델의 성능을 비교 분석하였다. 실험 결과, 음성의 순차적 특성을 반영한 **BiLSTM + Attention 모델이 $93.9\%$의 최고 정확도**를 기록하며 딥러닝 모델의 우위성을 입증하였다. 다만, Attention의 성능 향상 폭이 적어 단순한 분류 작업에서의 효율성에 대해서는 의문을 제기하였으며, 발음이 유사한 단어 간의 오분류 문제를 해결하기 위한 문맥 정보 도입의 필요성을 제시하였다. 이 연구는 향후 BERT와 같은 사전 학습 모델의 적용이나 복잡한 데이터 증강 기법 연구의 기초 자료로 활용될 가능성이 크다.
