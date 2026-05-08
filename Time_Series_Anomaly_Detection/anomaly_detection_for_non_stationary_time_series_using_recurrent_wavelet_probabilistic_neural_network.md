# Anomaly Detection for Non-stationary Time Series using Recurrent Wavelet Probabilistic Neural Network

Pu Yang, J. A. Barria (2025)

## 🧩 Problem to Solve

본 논문은 비정상성(Non-stationary) 환경에서의 시계열 이상치 탐지(Time Series Anomaly Detection, TSAD) 문제를 해결하고자 한다. 실시간 서비스 데이터 분석에서 발생하는 주요 도전 과제는 다음과 같다. 첫째, 개인정보 보호 규정이나 이상치 샘플의 희소성으로 인해 모델 학습에 사용할 수 있는 데이터의 양이 제한적이라는 점이다. 둘째, 학습 시 보지 못한(unseen) 데이터가 기존과 다른 통계적 특성을 보이는 Concept Drift 현상이 발생할 수 있다는 점이다.

기존의 재구성 기반(Reconstruction-based) 또는 예측 기반 방법론들은 비정상성 데이터에서 클래스 간의 차이를 정확하게 묘사하는 데 한계가 있다. 또한, 생성 모델 기반의 방법론들은 주로 가우시안 혼합 모델(GMM)과 같은 모수적 밀도 추정(Parametric density estimation) 방식을 사용하는데, 이는 데이터 분포에 대한 강한 가정을 전제로 하므로 비정상성 환경이나 불연속적이고 국소적인 특성을 가진 밀도 함수를 처리하는 데 효율적이지 않다. 따라서 본 논문의 목표는 데이터 분포에 대한 제약이 적은 비모수적(Nonparametric) 방식을 활용하여 비정상성 환경에서도 강건하게 이상치를 탐지할 수 있는 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 비모수적 Wavelet Density Estimator(WDE)를 딥러닝 구조에 통합한 **Recurrent Wavelet Probabilistic Neural Network (RWPNN)**를 제안한 것이다. 주요 설계 아이디어는 다음과 같다.

1. **잠재 공간의 비모수적 모델링**: Stacked Recurrent Encoder-Decoder(SREnc-Dec)를 통해 고차원 시계열 데이터를 저차원의 latent space로 압축하고, 이 공간의 확률 밀도 함수(PDF)를 WDE 기반의 확률 네트워크로 모델링함으로써 모수적 모델의 한계를 극복하였다.
2. **Multi-Receptive-field Wavelet Probabilistic Network (MRWPN)**: 단일한 데이터 변동률만 감시하던 기존 WPNN을 확장하여, 서로 다른 망각 계수(Forgetting factor, $\alpha$)를 가진 여러 PDF들의 앙상블 모델을 생성하였다. 이를 통해 추가적인 네트워크 구성 없이도 다양한 속도의 데이터 변동을 동시에 포착할 수 있는 '다중 수용 영역(Multi-receptive-field)' 관점을 제공한다.
3. **차원의 저주 해결**: WDE가 고차원 데이터에서 겪는 계산 복잡도와 차원의 저주 문제를 SREnc-Dec를 통한 차원 축소로 해결하여, 고차원 시계열 데이터에도 적용 가능하게 하였다.

## 📎 Related Works

기존의 시계열 이상치 탐지 접근 방식은 크게 세 가지로 분류된다.

- **통계적 접근 방식**: Autoregressive 모델이나 ARIMA 등이 있으며, 단순한 통계 특성에 의존한다.
- **일반 머신러닝 방식**: Isolation Forest, One-Class SVM, Local Outlier Factor(LOF) 등이 활용된다.
- **딥러닝 기반 방식**: CNN, RNN, Transformer 등을 사용하여 표현 학습을 수행한다. 특히 Encoder-Decoder 구조를 통해 데이터를 재구성하고, 그 재구성 오차(Reconstruction Error)를 이상치 점수로 사용하는 방식이 널리 사용되었다.

최근에는 VAE나 Autoencoder를 사용하여 latent space를 학습하고 GMM 등으로 분포를 모델링하는 생성 모델 기반 방식이 제안되었다. 그러나 이러한 방식들은 데이터 분포에 대한 강한 가정을 필요로 하며, 비정상성 환경에서는 성능이 저하된다. 본 논문은 이러한 한계를 극복하기 위해 데이터 분포에 대한 가정이 거의 없는 Wavelet Density Estimators(WDE)를 도입하였다. WDE는 다해상도 분석(Multiresolution analysis)을 통해 시간-주파수 영역의 국소적 특성과 일시적 현상을 효과적으로 포착할 수 있다는 장점이 있다.

## 🛠️ Methodology

RWPNN은 크게 두 가지 모듈인 **SREnc-Dec**와 **MRWPN**으로 구성된다.

### 1. Stacked Recurrent Encoder-Decoder (SREnc-Dec)

입력 시계열 데이터 $x$로부터 유의미한 temporal feature를 추출하는 모듈이다.

- **구조**: 여러 층의 RNN(본 논문에서는 LSTM 사용) 셀을 쌓아 올린 형태이다. Encoder는 입력 데이터를 압축하여 최종 은닉 상태 $h_E$를 생성하고, Decoder는 이 $h_E$를 다시 원래의 입력 공간으로 복원한다.
- **학습 목표**: 입력 $x$와 재구성된 $\hat{x}$ 사이의 평균 절대 오차(Mean Absolute Error, MAE)를 최소화하는 방향으로 학습된다. 이 과정에서 모델은 데이터의 일반적인 패턴을 latent space $h_E$에 저장하게 된다.

### 2. Multi-Receptive-field Wavelet Probabilistic Network (MRWPN)

SREnc-Dec가 추출한 latent space $h_E$의 확률 밀도 함수를 추정하는 비모수적 네트워크이다.

**A. Radial B-Spline Scaling Function**
계산 복잡도를 줄이기 위해 해석적 폐형식(Analytic closed-form)을 가진 Radial B-spline scaling function $\Phi_{j_0,k}(x)$를 사용한다.
$$\Phi_{j_0,k}(x) = 2^{nj_0} 2^N \phi\left(\left\|(2^{j_0}x - k)\right\| + \frac{m}{2}\right)$$
여기서 $n$은 차원, $j_0$는 팽창(dilation) 파라미터, $k$는 이동(translation) 파라미터, $m$은 B-spline의 차수이다.

**B. 밀도 추정 및 계수 업데이트**
추정된 밀도 함수 $\hat{p}(x)$는 다음과 같이 scaling function들의 선형 결합으로 표현된다.
$$\hat{p}(x) = \sum_k \hat{w}_{j_0,k} \Phi_{j_0,k}(x)$$
비정상성 환경에 대응하기 위해 망각 계수 $\alpha$를 도입한 재귀적 업데이트 식을 사용한다.
$$\hat{w}_t = (1-\alpha) \hat{w}_{t-1} + \alpha \Phi_{j_0,k}(x_t)$$
$\alpha$가 클수록 최신 데이터에 더 많은 가중치를 두어 Concept Drift에 빠르게 적응한다.

**C. 앙상블 뷰(Ensemble View) 구성**
MRWPN은 단일 $\alpha$가 아닌 $\Gamma = [\alpha_1, \alpha_2, \dots, \alpha_\gamma]$라는 집합을 사용하여 여러 개의 PDF를 동시에 계산한다. 이는 서로 다른 변동 속도를 가진 데이터 특성을 동시에 포착하는 효과를 준다.

### 3. 이상치 탐지 절차

학습 단계에서는 정상 데이터($C_{normal}$)만을 사용하여 MRWPN의 계수 $\hat{w}$를 업데이트한다. 테스트 단계에서는 테스트 데이터 $x_{test}$의 latent representation $h_{E_{test}}$에 대한 PDF $\hat{p}(h_{E_{test}})$를 계산한다. 만약 추정된 확률 밀도 값이 임계값 $\beta$보다 낮으면 이상치로 판정한다.
$$\text{Anomaly if } \hat{p}(h_{E_{test}}) < \beta$$
최적의 $\beta$와 최적의 앙상블 뷰 인덱스 $i$는 검증 세트를 통해 F1-score를 최대화하는 방향으로 결정된다.

## 📊 Results

### 실험 설정

- **데이터셋**: UCR/UEA 저장소의 45개 실세계 시계열 데이터셋과 서버 머신 데이터셋(SMD)을 사용하였다.
- **비교 대상**: AML, LAD, LED, GE, DIF, TED 등 최신 비지도 학습 기반 이상치 탐지 알고리즘과 비교하였다.
- **평가 지표**: Precision, Recall, F1-score를 사용하였다.
- **시나리오**: 데이터 가용성(학습 데이터 비율 $P=0.8$ vs $P=0.2$)과 Concept Drift(테스트 세트에 가우시안 노이즈 추가) 유무에 따라 성능을 측정하였다.

### 주요 결과

1. **데이터 희소성 환경 ($P=0.8$)**: 학습 데이터가 매우 적은 상황에서 RWPNN은 다른 벤치마크 모델들보다 높은 Recall과 F1-score를 기록하였다. 특히 Concept Drift가 발생했을 때 RWPNN의 F1-score는 최선의 벤치마크 모델 대비 약 6% 향상되었다.
2. **충분한 데이터 환경 ($P=0.2$)**: 데이터가 충분할 때 모든 모델의 성능이 향상되었으나, RWPNN은 여전히 Precision, Recall, F1-score 모든 지표에서 우위를 점했다.
3. **SMD 데이터셋**: 가장 도전적인 과제인 SMD 데이터셋에서 RWPNN은 특히 강력한 성능을 보였다. Concept Drift가 있는 환경에서 DIF가 F1-score 0.10을 기록한 반면, RWPNN은 이보다 22% 높은 성능을 보였다.
4. **조기 경보(Early Warning)**: ECG5000 데이터셋 분석 결과, RWPNN은 실제 이상치가 명확히 드러나기 전인 초기 단계에서 PDF의 미세한 변화를 포착함으로써 조기 경보 시스템으로서의 가능성을 입증하였다.
5. **Ablation Study**: MRWPN을 OCSVM, IForest, DIF 등으로 교체했을 때 성능이 크게 저하됨을 확인하여, 제안한 비모수적 밀도 추정 모듈의 중요성을 증명하였다.

## 🧠 Insights & Discussion

본 연구는 비모수적 WDE를 딥러닝의 latent space에 결합함으로써 비정상성 시계열 데이터 탐지의 새로운 방향을 제시하였다.

**강점 및 해석**:
RWPNN이 Concept Drift 상황에서 강건한 이유는 PDF를 모델링할 때 특정 분포를 가정하지 않는 비모수적 방식을 사용하고, 망각 계수를 통한 재귀적 업데이트를 통해 최신 데이터의 통계적 특성을 실시간으로 반영하기 때문이다. 또한, SREnc-Dec를 통해 고차원 데이터를 저차원으로 압축함으로써 WDE의 고질적인 문제였던 차원의 저주와 계산 복잡도 문제를 효율적으로 해결하였다.

**한계 및 논의**:
본 모델은 SREnc-Dec의 백본으로 RNN(LSTM)만을 사용하므로, 시간적 의존성은 잘 포착하지만 공간적(spatial) 특성이나 전역적(global) 특징을 추출하는 능력은 부족할 수 있다. 실제로 실험 결과에서 공간적 특징이 중요한 데이터셋에서는 상대적으로 낮은 성능을 보였다. 이를 해결하기 위해 향후 연구에서는 CNN이나 Transformer와 같은 구조를 feature extraction 단계에 통합하는 방안이 고려될 필요가 있다.

## 📌 TL;DR

본 논문은 비정상성 환경의 시계열 이상치 탐지를 위해 **SREnc-Dec(특징 추출)**와 **MRWPN(비모수적 밀도 추정)**을 결합한 **RWPNN** 프레임워크를 제안한다. 특히 MRWPN은 다중 망각 계수를 이용한 앙상블 뷰를 통해 Concept Drift에 강건하게 대응하며, 학습 데이터가 부족한 상황에서도 우수한 탐지 성능을 보인다. 이 연구는 고차원 비정상성 시계열 데이터의 실시간 모니터링 및 조기 경보 시스템 구축에 중요한 기여를 할 수 있을 것으로 기대된다.
