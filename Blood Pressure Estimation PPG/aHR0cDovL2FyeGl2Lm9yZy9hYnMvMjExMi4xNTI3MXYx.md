# BP-Net: Cuff-less, Calibration-free, and Non-invasive Blood Pressure Estimation via a Generic Deep Convolutional Architecture

Soheil Zabihi, Elahe Rahimian, Fatemeh Marefat, Amir Asif, Pedram Mohseni, and Arash Mohammadi (2021)

## 🧩 Problem to Solve

본 논문은 비침습적이고 커프가 필요 없는(cuff-less) 방식의 연속적인 혈압(Blood Pressure, BP) 모니터링 솔루션을 개발하는 것을 목표로 한다. 기존의 커프 기반 측정 방식은 사용자가 불편함을 느끼며, 실시간 연속 측정이 불가능하고, 병원 환경과 가정 환경 간의 측정값 차이가 발생하는 등의 한계가 있다.

또한, 기존의 커프리스(cuff-less) 추정 방식들은 대부분 Pulse Arrival Time(PAT)과 같은 수작업으로 설계된 특징량(hand-crafted features) 추출에 의존한다. 이러한 방식은 개인별 특성이나 시간적 변화에 따른 비선형적 관계를 충분히 반영하지 못해 강건성(robustness)이 떨어지며, 빈번한 보정(calibration)이 필요하다는 치명적인 단점이 있다. 따라서 본 연구의 목표는 수작업 특징 추출 없이 raw 신호로부터 직접 혈압을 추정하여 보정이 필요 없고 강건성이 높은 딥러닝 기반의 BP-Net 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 설계 아이디어는 raw ECG(심전도)와 PPG(광전용적맥파) 신호를 입력으로 사용하여, 딥러닝 모델이 내재적 특징(intrinsic deep features)을 스스로 학습하게 함으로써 수작업 특징 추출의 한계를 극복하는 것이다.

주요 기여 사항은 다음과 같다:

- **End-to-End 구조의 BP-Net 제안**: PAT와 같은 수작업 특징 없이 raw ECG 및 PPG 파형을 직접 입력으로 사용하는 합성곱 신경망(CNN) 아키텍처를 설계하였다.
- **효율적인 시퀀스 모델링**: Causal Dilated Convolutions와 Residual Connections를 결합하여, recurrent neural networks(RNN)보다 학습 속도가 빠르면서도 더 넓은 수용 영역(receptive field)과 긴 유효 메모리를 확보하였다.
- **일반화 성능 검증**: 다양한 유형의 ECG 리드(I, II, III, IV)를 사용하여 모델의 범용성을 확인하였다.
- **벤치마크 데이터셋 구축**: MIMIC-I 및 MIMIC-III 데이터베이스를 통합하여 총 104명의 피험자 데이터를 포함하는 표준화된 벤치마크 데이터셋을 구축함으로써, 향후 연구의 공정한 비교 기반을 마련하였다.

## 📎 Related Works

기존의 혈압 측정 및 추정 방식은 크게 두 가지로 분류된다:

1. **수작업 회귀 기반 모델 (Hand-crafted Regression-based Models)**:
   - PPG와 ECG 신호에서 PAT를 추출하고 이를 SVR(Support Vector Regression), 결정 트리(Decision Tree) 등의 전통적인 머신러닝 알고리즘에 입력하는 방식이다.
   - **한계**: BP 역학의 중요한 시간적 의존성(temporal dependencies)을 무시하며, 보정 파라미터와 선택된 특징량에 지나치게 의존하여 장기적인 정확도가 낮고 강건성이 부족하다.

2. **딥러닝 기반 모델 (Deep Learning-based Models)**:
   - LSTM, RNN, CNN 등을 활용하여 혈압을 예측하는 방식이다.
   - **한계**: 최신 연구들조차 딥러닝 모델에 입력하기 전, 여전히 PAT와 같은 수작업 특징을 먼저 추출하여 입력하는 경우가 많다. 이는 딥러닝의 내재적 특징 추출 능력을 완전히 활용하지 못하는 것이다. 또한, 사용되는 데이터셋이 통일되지 않아 객관적인 성능 비교가 어렵다는 문제가 있다.

## 🛠️ Methodology

### 1. 데이터 전처리 (Preprocessing)

raw 신호의 노이즈를 제거하기 위해 Discrete Wavelet Transform(DWT) 기반의 파이프라인을 사용한다.

- **업샘플링**: 125 Hz의 신호를 1,000 Hz로 업샘플링하여 샘플링 주파수 변화에 대응한다.
- **DWT 분해**: Biorthogonal 6.8 mother wavelet을 사용하여 10단계로 분해한다.
- **노이즈 제거**: 고주파 노이즈(근육 수축 등)를 제거하기 위해 상세 계수(Detail coefficients) $D_1, D_2, D_3$를 제거하고, 저주파 노이즈(호흡, 신체 움직임)를 제거하기 위해 근사 계수(Approximation coefficient) $A_{10}$을 제거한다.
- **전원선 잡음 제거**: 60 Hz 전원선 간섭을 제거하기 위해 bandstop 필터(59.5 - 61.5 Hz)를 적용한 후 Inverse DWT(IDWT)를 통해 신호를 재구성한다.

### 2. BP-Net 아키텍처

본 모델은 혈압 추정 문제를 시퀀스 모델링 작업으로 접근한다.

#### 데이터 정규화

입력 데이터의 정규화를 위해 $\mu$-law transformation을 적용한다:
$$F(X) = \text{sign}(X) \frac{\ln (1 + \mu |X|)}{\ln (1 + \mu)}$$

#### 핵심 구성 요소

- **Causal Convolutions**: 미래의 데이터가 과거의 예측에 영향을 주는 정보 누설(information leakage)을 방지하기 위해, 출력 $\hat{B}(t)$가 $t$ 시점 이전의 입력 샘플에만 의존하도록 설계되었다.
- **Dilated Convolutions**: 해상도 손실 없이 수용 영역을 확장하기 위해 사용한다. dilation rate $L$인 1차원 dilated convolution 연산은 다음과 같이 정의된다:
$$D(p), (x *_L K)(p) = \sum_{i=0}^{R-1} K(i) \times x(p - L \times i)$$
- **Residual Connections**: 층이 깊어짐에 따라 발생하는 기울기 소실(vanishing gradient) 및 성능 저하(degradation) 문제를 해결하기 위해 Identity block과 Convolutional block을 사용한다.

#### 네트워크 구조 및 학습 절차

- **입력층**: PPG와 ECG 신호를 각각 32개의 커널을 가진 $1 \times 1$ convolution 층에 통과시킨 후 채널 방향으로 결합(concatenate)한다.
- **Residual Blocks**: 총 6개의 잔차 블록이 쌓여 있으며, 각 블록은 2개의 dilated causal convolutions와 ELU 활성화 함수, Dropout, Weight Normalization으로 구성된다.
  - **Dilation Factor ($L$)**: 층마다 2배씩 증가 ($1, 2, 4, 8, 16, 32$).
  - **커널 수**: $32, 32, 64, 64, 128, 256$ 순으로 증가.
  - **커널 크기**: 모든 dilated causal convolutions의 커널 크기는 5이다.
- **출력층**: 6번째 블록의 출력을 $\to 1 \times 1$ Conv (256 kernels) $\to$ ELU $\to 1 \times 1$ Conv (2 kernels) $\to$ ELU 순으로 통과시켜 최종적으로 SBP와 DBP 값을 출력한다.
- **학습 설정**: Adam optimizer를 사용하며, 학습률은 0.001에서 시작하여 100 epoch 주기로 변하는 cyclic learning rate 방식을 적용하여 지역 최솟값(local minimum) 탈출을 돕는다. 배치 크기는 64이다.

## 📊 Results

### 1. 실험 환경 및 지표

- **데이터셋**: MIMIC-I 및 MIMIC-III에서 수집한 104명의 피험자 데이터.
- **평가 지표**: Root Mean Square Error (RMSE) 및 Mean Absolute Error (MAE).
  $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{j=1}^{n} |B_j - \hat{B}_j|^2}, \quad \text{MAE} = \frac{1}{n} \sum_{j=1}^{n} |B_j - \hat{B}_j|$$

### 2. 정량적 결과

전체 104명 피험자에 대한 평균 오차 결과는 다음과 같다.

- **SBP (수축기 혈압)**: $\text{Average RMSE} = 3.23 \pm 2.11 \text{ mmHg}$, $\text{Average MAE} = 2.76 \pm 1.92 \text{ mmHg}$
- **DBP (이완기 혈압)**: $\text{Average RMSE} = 1.57 \pm 1.12 \text{ mmHg}$, $\text{Average MAE} = 1.30 \pm 0.93 \text{ mmHg}$

### 3. 표준 가이드라인 준수 여부

- **AAMI 표준**: 평균 오차(ME) $\le 5 \text{ mmHg}$ 및 표준편차(SDE) $\le 8 \text{ mmHg}$ 요건을 모두 충족하였다. (SBP: ME 0.27, SDE 3.77 / DBP: ME 0.07, SDE 1.86)
- **BHS 표준**: 누적 오차 비율 기준 SBP와 DBP 모두에서 최고 등급인 **Grade A**를 획득하였다. 특히 SBP의 94.01%, DBP의 98.78%가 오차 $5 \text{ mmHg}$ 이내로 측정되었다.
- **통계적 분석**: Pearson 상관계수가 DBP $r=0.9872$, SBP $r=0.986$으로 나타나 실제 값과 예측 값 사이에 매우 높은 선형성이 있음을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 기존 혈압 추정 모델들의 고질적인 문제였던 '수작업 특징 추출 의존성'을 완전히 제거하고 raw 신호를 직접 처리하는 end-to-end 딥러닝 구조를 제안했다는 점에서 큰 강점이 있다. 특히, Causal Dilated Convolution을 통해 RNN의 시간적 의존성 학습 능력을 유지하면서도 연산 효율성을 높인 설계가 돋보인다. 또한, 파편화되어 있던 혈압 추정 데이터셋 문제를 해결하기 위해 104명 규모의 벤치마크 데이터셋을 공개하여 학계의 기여도를 높였다.

다만, 논문에서 제시된 결과가 매우 우수함에도 불구하고, 다른 최신 raw-signal 기반 딥러닝 모델과의 직접적인 정량적 비교 분석이 부족하다. "기존 recurrent networks보다 정확하다"고 언급하고 있으나, 구체적인 베이스라인 모델과의 수치 비교 표가 제시되지 않아 실제 어느 정도의 성능 향상이 이루어졌는지 판단하기 어렵다. 또한, 104명의 데이터셋을 사용했음에도 불구하고, 연령대별/성별/기저질환별 세부 분석이 누락되어 있어 실제 임상 환경에서의 완전한 일반화 가능성에 대해서는 추가적인 검증이 필요해 보인다.

## 📌 TL;DR

본 연구는 수작업 특징 추출(PAT 등)이나 별도의 보정 과정 없이, raw ECG 및 PPG 신호만을 이용하여 혈압을 추정하는 **BP-Net** 아키텍처를 제안하였다. Causal Dilated Convolutions와 Residual Connections를 통해 시퀀스 데이터의 특징을 효과적으로 추출하였으며, MIMIC 데이터셋 기반의 벤치마크를 통해 AAMI 및 BHS 표준의 최고 등급 성능을 달성하였다. 이 연구는 커프리스 혈압 모니터링의 실용성을 높였으며, 제공된 표준 데이터셋은 향후 딥러닝 기반 혈압 추정 연구의 중요한 기준점이 될 것으로 기대된다.
