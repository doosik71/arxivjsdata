# AdaWaveNet: Adaptive Wavelet Network for Time Series Analysis

Han Yu, Peikun Guo, Akane Sano (2024)

## 🧩 Problem to Solve

시계열 데이터 분석은 금융, 의료, 기상학 등 다양한 도메인에서 필수적이다. 그러나 실제 세계의 시계열 데이터는 통계적 특성이 시간에 따라 변하는 비정상성(non-stationary)이라는 특성을 가진다. 기존의 딥러닝 모델들은 대부분 데이터의 통계적 성질이 일정하다는 정상성(stationarity) 가정을 기반으로 설계되었기 때문에, 시간에 따라 역동적으로 변화하는 패턴을 포착하는 데 한계가 있으며, 이는 결국 분석의 편향(bias)과 오차로 이어진다.

본 논문의 목표는 이러한 비정상성 문제를 해결하기 위해 적응형 웨이브렛 변환(Adaptive Wavelet Transformation)을 도입하여 다중 스케일 분석(multi-scale analysis)이 가능한 모델을 구축하는 것이다. 이를 통해 다양한 주파수 대역의 특성을 유연하게 포착하고, 비정상 시계열 데이터에서도 강건한 성능을 내는 AdaWaveNet을 제안한다.

## ✨ Key Contributions

AdaWaveNet의 핵심 아이디어는 고정된 파라미터를 사용하는 기존 웨이브렛 방식에서 벗어나, 리프팅 스킴(lifting scheme)을 통해 학습 가능한 적응형 웨이브렛 변환을 구현하는 것이다. 주요 기여 사항은 다음과 같다.

1. **학습 가능한 웨이브렛 변환:** 리프팅 스킴을 CNN 구조에 통합하여 데이터의 통계적 특성 변화에 동적으로 적응하는 적응형 웨이브렛 변환을 구현하였다.
2. **다중 스케일 분석 프레임워크:** 시계열 분해(decomposition) 모듈과 적응형 웨이브렛 블록(AdaWave block)을 결합하여, 저주파의 전역적 추세와 고주파의 국소적 변동을 동시에 효과적으로 모델링한다.
3. **채널 간 의존성 강화:** 그룹화된 선형 모듈(Grouped Linear Module)을 도입하여, 서로 다른 특성을 가진 채널들을 클러스터링하고 각 그룹에 최적화된 선형 투영을 적용함으로써 추세(trend) 성분의 예측 품질을 높였다.
4. **시계열 초해상도(Super-resolution) 벤치마크 구축:** 기존의 예측 및 결측치 보간 외에, 저해상도 신호를 고해상도로 복원하는 초해상도 작업에 대한 새로운 벤치마크를 제시하였다.

## 📎 Related Works

기존의 시계열 분석 모델들은 RNN, MLP, CNN, Transformer 기반으로 발전해 왔으나, 앞서 언급한 비정상성 문제에 취약하다. 이를 해결하기 위해 다음과 같은 접근 방식들이 시도되었다.

- **분해 기반 방법(Decomposition-Based Methods):** DRCNN, MICN, TEMPO 등은 데이터를 추세(trend)와 잔차(residual) 또는 계절성(seasonal) 성분으로 분리하여 분석한다. 하지만 이들은 고정된 커널이나 미리 정의된 분해 방식을 사용하여 동적으로 변하는 신호에 대한 적응력이 부족하다.
- **비정상성 강화 모델(Non-Stationarity-Enhanced Models):** Non-stationary Transformer는 인스턴스 정규화(instance normalization)를 통해 통계적 특성을 표준화하며, FEDformer는 푸리에 변환 및 웨이브렛 변환을 사용하여 주파수 도메인에서 분석을 수행한다. 그러나 FEDformer와 같은 모델은 수동으로 조정된 웨이브렛 파라미터에 의존하며, 계산 복잡도가 높은 경우가 많다.

AdaWaveNet은 리프팅 스킴을 통해 웨이브렛 계수를 엔드-투-엔드(end-to-end)로 학습함으로써, 수동 설정의 한계를 극복하고 데이터 주도적인 적응력을 확보했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

AdaWaveNet의 전체 파이프라인은 입력 시퀀스를 추세와 계절성 성분으로 분리한 후, 각각 특화된 모듈로 처리하여 최종적으로 합산하는 구조이다.

### 1. 시계열 분해 (Time Series Decomposition)

입력 시퀀스 $x_{input}$을 다음과 같이 가산적 분해(additive decomposition)를 통해 계절성 성분 $x_s$와 추세 성분 $x_{trend}$로 나눈다.
$$x_{input} = x_s + x_{trend}$$
추세 성분은 이동 평균(moving average)을 통해 추출하며, 원본에서 이를 뺀 나머지가 계절성 성분이 된다.

### 2. 적응형 웨이브렛 블록 (AdaWave Block)

계절성 성분 $x_s$를 다중 스케일로 분석하기 위해 리프팅 스킴 기반의 AdaWave 블록을 사용한다. 이 과정은 크게 세 단계로 나뉜다.

- **분할(Split):** 입력 $x_{l-1}^s$를 짝수 인덱스($e_l$)와 홀수 인덱스($o_l$) 성분으로 분리한다.
- **예측(Predict):** 짝수 성분을 이용해 홀수 성분을 예측하고, 그 차이를 통해 세부 계수(detail coefficients) $c_l$을 계산한다.
    $$c_l = o_l - \sigma(W_p^l * e_l + b_p^l)$$
- **업데이트(Update):** 계산된 세부 계수 $c_l$을 사용하여 짝수 성분을 정밀화하여 근사치(approximation) $e'_l$을 생성한다.
    $$e'_l = e_l + \sigma(W_u^l * c_l + b_u^l)$$
여기서 $\sigma$는 활성화 함수이며, $W$와 $b$는 학습 가능한 1D CNN 커널과 편향이다.

### 3. 채널별 어텐션 (Channel-wise Attention, CWA)

최종 단계의 분해 결과인 저차원 근사치 $x_N^s$에 대해 self-attention 메커니즘을 적용한다. 이는 채널 간의 전역적 맥락 정보를 포착하여 $\hat{x}_N^s$로 정밀하게 투영하는 역할을 한다.

### 4. 역 적응형 웨이브렛 블록 (InvAdaWave Block)

정밀화된 근사치 $\hat{x}_N^s$와 저장해둔 세부 계수 $c_l$들을 이용하여 전치 합성곱(Transposed Convolution)을 통해 원래의 계절성 성분 $\hat{x}_s$를 복원한다.

### 5. 그룹화된 선형 모듈 (Grouped Linear Module)

추세 성분 $x_{trend}$는 채널마다 특성이 다를 수 있으므로, K-means 클러스터링을 통해 유사한 채널들을 그룹화한다. 각 그룹에는 서로 다른 선형 투영 헤드(linear heads)를 적용하여 $\hat{x}_{trend}$를 예측한다.

최종 출력은 $\hat{x}_{pred} = \hat{x}_s + \hat{x}_{trend}$로 계산된다.

## 📊 Results

### 실험 설정

- **작업:** 시계열 예측(Forecasting), 결측치 보간(Imputation), 초해상도(Super-resolution).
- **데이터셋:** ETT, ECL, Traffic, Weather, Exchange, Solar, PTB-XL, Sleep-EDFE, CLAS 등 총 10개.
- **지표:** Mean Squared Error (MSE), Mean Absolute Error (MAE).
- **비교 모델:** iTransformer, FreTS, TimesNet, FEDformer, DLinear, PatchTST, Stationary Transformer 등.

### 주요 결과

1. **예측 작업:** 대부분의 데이터셋에서 SOTA 성능을 달성하였다. 특히 주파수 도메인 강화 방법들보다 MSE 기준 약 7.7%, MAE 기준 약 6.2% 성능이 향상되었다.
2. **보간 작업:** 랜덤 마스킹과 확장 마스킹(contiguous segments) 모두에서 경쟁력 있는 성능을 보였으며, 특히 시퀀스 길이가 긴 PTB-XL 및 Sleep-EDFE 데이터셋에서 강점을 보였다.
3. **초해상도 작업:** 저해상도 입력에서 고해상도 신호를 복원하는 작업에서 다중 스케일 분석 능력을 바탕으로 타 모델 대비 우수한 복원력을 보여주었다.
4. **효율성:** iTransformer와 유사한 학습 속도와 메모리 사용량을 유지하면서도, PatchTST보다 훨씬 효율적인 자원 사용량을 기록하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

AdaWaveNet은 고정된 웨이브렛 기반이 아닌 학습 가능한 리프팅 스킴을 도입함으로써, 데이터의 비정상적 특성에 유연하게 대응할 수 있음을 입증하였다. 특히 합성 데이터(Synthetic data) 실험을 통해 저주파 사인파와 고주파 과도 응답(transients)이 섞인 복잡한 신호에서도 지상참값(ground truth)에 매우 근접한 예측을 수행함을 확인하였다. 이는 모델이 다중 스케일의 특징을 효과적으로 분리하고 학습하고 있음을 시사한다.

### 한계 및 비판적 해석

- **모델 복잡도:** 단순한 MLP 기반 모델인 DLinear보다는 계산 비용이 높다. 따라서 매우 제한된 자원 환경이나 실시간성이 극도로 요구되는 환경에서는 적용에 제약이 있을 수 있다.
- **일반화 성능:** 전반적으로 우수하지만, 특정 작업(예: 전력 및 기상 데이터의 랜덤 보간)에서는 TimesNet이 더 나은 성능을 보이는 경우가 존재한다. 이는 데이터의 주기성이나 특성에 따라 적응형 웨이브렛보다 다른 구조가 더 유리할 수 있음을 의미한다.
- **가정:** 본 모델은 시계열을 가산적으로 분해($x_s + x_{trend}$)한다는 가정을 사용하는데, 곱셈적 특성을 가진 시계열의 경우 추가적인 전처리가 필요할 수 있다.

## 📌 TL;DR

AdaWaveNet은 시계열 데이터의 **비정상성(non-stationarity)** 문제를 해결하기 위해 **학습 가능한 리프팅 스킴 기반의 적응형 웨이브렛 변환**을 도입한 모델이다. 다중 스케일 분석과 그룹화된 선형 모듈을 통해 예측, 보간, 초해상도라는 세 가지 핵심 작업에서 기존 SOTA 모델들을 능가하는 성능을 보였다. 특히 시계열 초해상도라는 새로운 벤치마크를 제시하여 향후 고정밀 신호 복원 연구에 기여할 가능성이 크다.
