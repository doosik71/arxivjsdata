# TransfoRhythm: A Transformer Architecture Conductive to Blood Pressure Estimation through Solo PPG Signal Capturing

Amir Arjomand, Amin Boudesh, Farnoush Bayatmakou, Georgiy Krylov, Kenneth B. Kent, Arash Mohammadi (2025)

## 🧩 Problem to Solve

본 논문은 고혈압 진단 및 관리에 필수적인 혈압(Blood Pressure, BP)을 비침습적이고 연속적으로 측정하는 방법을 다룬다. 전통적인 커프(cuff) 기반 측정 방식은 다음과 같은 한계가 존재한다. 첫째, 자원이 제한된 환경에서 접근성이 낮다. 둘째, 커프의 팽창과 수축 과정이 환자에게 불편함을 주어 빈번한 측정에 부적합하다. 셋째, 실시간 연속 측정이 불가능하여 야간 혈압 패턴과 같은 중요한 임상 정보를 놓칠 수 있다.

최근 AI 기반의 커프리스(cuff-less) 혈압 추정 연구가 활발하지만, 대부분의 기존 방식은 심전도(ECG)와 광전용적맥파(PPG) 센서를 동시에 사용하는 결합 방식에 의존한다. 이는 다중 센서 사용으로 인한 전력 소모 증가, 센서 간 고정 거리 유지 필요성, 잦은 재교정 등의 제약이 따른다. 따라서 단일 PPG 신호(solo PPG)만을 이용하여 혈압을 정확하게 추정하는 것이 핵심 과제이다. 하지만 단일 PPG 신호는 보조 센서(ECG)의 부재로 인해 형태학적 특징(morphological features)에 의존해야 하며, 움직임에 의한 잡음(motion artifacts)과 고주파 노이즈에 취약하다는 문제가 있다.

본 논문의 목표는 Transformer 기반의 딥러닝 아키텍처인 **TransfoRhythm**을 통해, 단일 PPG 신호만으로 수축기 혈압(SBP)과 이완기 혈압(DBP)을 정밀하게 추정하는 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 연구의 중심적인 아이디어는 Transformer의 **Multi-Head Attention (MHA)** 메커니즘을 활용하여 PPG 신호의 시계열 데이터 내에서 복잡한 의존성과 유사성을 포착하는 것이다. 주요 기여 사항은 다음과 같다.

1.  **단일 PPG 기반 고정밀 추정**: ECG와 같은 보조 센서 없이 PPG 신호의 형태학적 특징과 시계열적 특성만을 활용하여 혈압을 추정하는 TransfoRhythm 프레임워크를 설계하였다.
2.  **MIMIC-IV 데이터셋의 최초 적용**: 최신 생체 신호 데이터베이스인 MIMIC-IV v2.0을 커프리스 혈압 추정 연구에 최초로 적용하여 모델의 신뢰성과 일반화 성능을 입증하였다.
3.  **효율적인 아키텍처 설계**: 1D Convolution을 통한 특징 차원 확장, Positional Encoding을 통한 순서 정보 보존, 그리고 계산 부하를 줄이기 위한 Time Frame Compressor를 도입하여 실용적인 딥러닝 구조를 제안하였다.

## 📎 Related Works

기존의 커프리스 혈압 추정 연구는 주로 맥파 전달 시간(Pulse Transit Time, PTT)이나 맥파 속도(Pulse Wave Velocity, PWV) 측정에서 시작되었다. 최근에는 딥러닝 모델을 통해 PPG와 ECG 신호에서 변별력 있는 특징을 추출하는 방식이 주를 이루었다.

-   **결합 센서 방식**: CNN-LSTM 하이브리드 모델이나 TCN(Temporal Convolutional Network) 등을 사용하여 PPG와 ECG를 함께 입력으로 사용해 높은 정확도를 달성하였다. 그러나 이는 센서 구성의 복잡성과 전력 소모라는 한계가 있다.
-   **단일 PPG 방식**: raw PPG 시계열 데이터나 특징 기반 벡터를 입력으로 하는 SVR, Random Forest, LSTM, GRU 등의 모델이 연구되었다. 최근에는 Transformer 기반의 Attention 메커니즘을 도입한 AriaNet 등이 등장하였으나, 여전히 노이즈 제거와 형태학적 특징 추출의 어려움이 존재한다.

본 논문은 기존의 CNN이나 RNN 기반 모델들이 가진 장기 의존성(long-term dependencies) 포착의 한계와 시간적 이동에 대한 불변성 부족 문제를 Transformer의 Attention 메커니즘으로 해결함으로써 차별성을 갖는다.

## 🛠️ Methodology

### 1. 전체 파이프라인
전체 시스템은 **데이터 전처리 $\rightarrow$ 특징 추출 $\rightarrow$ Transformer 모델 입력 $\rightarrow$ 혈압 추정**의 순서로 구성된다.

### 2. 데이터셋 및 전처리
-   **데이터셋**: MIMIC-IV Waveform Dataset을 사용하며, 샘플링 레이트는 $62.4\text{Hz}$이다.
-   **데이터 정제**: 진폭이 범위를 벗어난 신호, 15분 미만의 짧은 기록, 평탄화된 라인(flattened-line) 또는 극심한 노이즈 구간을 제거하였다.
-   **신호 필터링**: 
    -   $0.7 \sim 10\text{Hz}$ 범위의 5차 Butterworth 밴드패스 필터를 적용하여 베이스라인 표류(wandering baseline)와 전원선 간섭($50\text{-}60\text{Hz}$)을 제거하였다.
    -   5차 이동 평균 필터(Moving Average Filter, MAF)를 적용하여 신호를 평활화하였다.

### 3. 특징 추출 (Feature Extraction)
PPG 신호와 그 2차 미분 신호(SDPPG)에서 피크(peak)와 밸리(valley)를 탐색하여 개별 PPG 사이클을 분리한다. 각 사이클에서 혈역학적 특징을 반영하는 총 12가지 특징을 추출한다.
-   **추출 특징**: 사이클 지속 시간(TD1), 상승 시간(Trhp), 피크-노치 간격(TD2), PPG 적분값(PPGI), SDPPG 최대 진폭(AMP), SDPPG 풋-피크 간격(TD4) 등.

### 4. 네트워크 아키텍처
#### (1) 입력 및 임베딩
입력 데이터는 셔플링된 프레임으로 구성된 Stacked Frames(SF) 형태이다. 입력 특징 길이 $L_{in} = 12$, 시간 프레임 시퀀스 $T = 48$이다. 1D Convolution 레이어를 통해 특징 차원을 $12$에서 $128(L_{out})$로 확장한다.

$$ST_{Extended} = \text{Bias}(L_{out}) + \sum_{k=0}^{L_{in}} \text{weight}(L_{out}, k) \cdot \text{input}(N_i, k)$$

#### (2) Positional Encoding (PE)
Transformer는 기본적으로 순서 정보가 없으므로, 사인-코사인 함수 기반의 Positional Encoding을 임베딩 벡터에 더해 시간적 위치 정보를 부여한다.

#### (3) Multi-Head Attention (MHA)
14개의 Head를 가진 MHA 메커니즘을 사용하여 각 프레임 간의 의존성을 학습한다. 입력 임베딩 행렬에 학습 가능한 가중치 $W^K, W^Q, W^V$를 곱하여 Key, Query, Value 행렬을 생성하고, 이를 통해 Attention Score를 계산하여 중요한 시퀀스 부분에 집중한다.

#### (4) Position-Wise Feed-Forward 및 출력
-   **Feed-Forward**: 1D Convolution과 ReLU 활성화 함수를 사용하여 위치별 표현력을 높인다.
-   **Time Frame Compressor**: 고차원 출력을 1차원으로 변환할 때 발생하는 계산 부하를 줄이기 위해, 시간축 차원을 압축하는 비학습형 유닛을 도입하였다.
-   **최종 출력**: Flattening 레이어와 ReLU 함수를 거쳐 최종적으로 SBP와 DBP 값을 예측한다.

### 5. 학습 및 평가 설정
-   **검증 전략**: 5-fold cross-validation을 수행하여 과적합을 방지하였다.
-   **손실 함수**: Mean Squared Error (MSE)를 사용하였다.
-   **최적화**: Adam Optimizer, 학습률 $1\text{E-4}$, 배치 사이즈 $128$, 총 400 에포크 동안 학습하였다.
-   **평가 지표**:
    -   $R^2$ (결정계수)
    -   $\text{ME} = \frac{1}{n} \sum_{i=1}^n (T_i - P_i)$
    -   $\text{MAE} = \frac{1}{n} \sum_{i=1}^n |T_i - P_i|$
    -   $\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (T_i - P_i)^2}$

## 📊 Results

### 1. 정량적 결과
TransfoRhythm은 매우 높은 예측 정확도를 보였다.
-   **SBP**: $\text{MAE} = 1.37\text{ mmHg}$, $\text{RMSE} = 2.21\text{ mmHg}$, $R^2 = 0.993$
-   **DBP**: $\text{MAE} = 1.06\text{ mmHg}$, $\text{RMSE} = 1.84\text{ mmHg}$, $R^2 = 0.994$

### 2. 의료 표준 검증
-   **AAMI 표준**: 평균 오차(ME) $5\text{ mmHg}$ 미만, 표준편차(SD) $8\text{ mmHg}$ 미만 요건을 충족하였다.
-   **BHS 표준**: 오차 범위별 누적 백분율을 분석한 결과, 최고 등급인 **Grade A**를 획득하였다.
-   **Bland-Altman 분석**: SBP의 평균 차이는 $0.26\text{ mmHg}$, DBP는 $0.07\text{ mmHg}$로 체계적 편향(systematic bias)이 거의 없음을 확인하였다.

### 3. 벤치마크 비교
MIMIC-IV 데이터셋에서 ResNet1D, AlexNet1D, U-Net, Bi-LSTM 및 Hybrid CNN-RNN 모델과 비교 실험을 수행하였다.
-   **결과**: 모든 지표에서 TransfoRhythm이 압도적인 성능을 보였다. 특히 $\text{MAE}_{\text{SBP}}$ 기준, 차순위 모델인 ResNet1D($2.28\text{ mmHg}$)보다 훨씬 낮은 $1.37\text{ mmHg}$를 기록하였다. 이는 MHA 메커니즘이 PPG 신호의 시간적 의존성과 공간적 관계를 더 효과적으로 포착했기 때문으로 분석된다.

## 🧠 Insights & Discussion

### 강점
본 논문의 가장 큰 강점은 단일 PPG 신호만을 사용함에도 불구하고, Transformer의 Attention 메커니즘을 통해 기존의 RNN/CNN 기반 모델들이 해결하지 못한 장기 의존성 문제를 해결했다는 점이다. 또한, 의료 기기 표준인 AAMI 및 BHS 기준을 모두 충족함으로써 실제 임상 적용 가능성을 높였다.

### 한계 및 비판적 해석
1.  **데이터셋의 단일성**: 모든 실험이 MIMIC-IV 데이터셋 하나에 의존하고 있다. 다양한 인종, 연령대, 기저 질환을 가진 환자들이 포함된 외부 데이터셋에서의 일반화 성능 검증이 필요하다.
2.  **특징 추출의 의존성**: Raw 데이터가 아닌 12가지의 정교하게 설계된 특징(feature)을 입력으로 사용한다. 이는 모델의 성능 향상에 기여했지만, 실제 실시간 시스템에서는 이러한 특징 추출 알고리즘의 연산 비용과 강건성(robustness)이 변수가 될 수 있다.
3.  **해석 가능성 언급 부족**: 논문에서는 Attention Score를 통해 모델의 해석 가능성을 높일 수 있다고 주장하였으나, 실제로 어떤 특징이나 시퀀스 구간이 혈압 추정에 결정적인 영향을 미쳤는지에 대한 구체적인 시각화 분석이나 정성적 결과는 제시되지 않았다.

## 📌 TL;DR

본 논문은 단일 PPG 신호만을 이용해 수축기 및 이완기 혈압을 추정하는 Transformer 기반의 **TransfoRhythm** 프레임워크를 제안한다. MIMIC-IV 데이터셋을 최초로 적용하여 검증하였으며, Multi-Head Attention을 통해 PPG의 복잡한 시계열 특징을 효과적으로 학습함으로써 기존 CNN, RNN 기반 모델들을 능가하는 성능($\text{SBP MAE } 1.37, \text{ DBP MAE } 1.06$)을 달성하고 의료 표준(AAMI, BHS Grade A)을 충족하였다. 이 연구는 향후 웨어러블 기기를 통한 연속적이고 비침습적인 혈압 모니터링 기술 발전에 중요한 기초가 될 것으로 보인다.