# KEDformer: Knowledge Extraction Seasonal Trend Decomposition for Long-term Sequence Prediction

Zhenkai Qin, Baozhong Wei, Caifeng Gao, and Jianyuan Ni (2024)

## 🧩 Problem to Solve

본 논문은 에너지, 금융, 기상학 등 다양한 도메인에서 필수적인 장기 시계열 예측(Long-term Time Series Forecasting, LSTF) 문제를 해결하고자 한다. 기존의 Transformer 기반 모델들은 시계열 데이터의 시간적 의존성을 포착하는 데 유망한 성능을 보였으나, 두 가지 핵심적인 한계점이 존재한다.

첫째는 계산 효율성 문제이다. 전통적인 Self-attention 메커니즘은 시퀀스 길이 $L$에 대해 $O(L^2)$의 시간 및 공간 복잡도를 가지므로, 예측 범위가 길어질수록 메모리 요구량과 연산 비용이 기하급수적으로 증가한다. 둘째는 일반화 및 노이즈 문제이다. 긴 시퀀스 내에 존재하는 무관한 정보(noise)가 Attention 분포를 약화시켜, 정작 중요한 장기 의존성을 효과적으로 모델링하는 것을 방해한다.

따라서 본 연구의 목표는 계산 복잡도를 획기적으로 낮추면서도, 시계열 데이터의 핵심적인 패턴을 효과적으로 추출하여 예측 정확도를 높이는 KEDformer 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

KEDformer의 핵심 아이디어는 **지식 추출(Knowledge Extraction)**과 **계절성-추세 분해(Seasonal-Trend Decomposition)**를 결합하여 효율성과 정확성을 동시에 잡는 것이다.

1. **KEDA(Knowledge Extraction Attention) 모듈**: 모든 Query-Key 쌍을 계산하는 대신, KL Divergence를 통해 정보량이 많은 가중치만을 선택적으로 추출하고, Autocorrelation(자기상관) 메커니즘을 결합하여 계산 복잡도를 $O(L^2)$에서 $O(L \log L)$로 낮추었다.
2. **동적 시계열 분해**: 시계열 데이터를 계절성(Seasonal) 성분과 추세-순환(Trend-Cyclical) 성분으로 분해하여, 단기적인 변동과 장기적인 패턴을 동시에 포착할 수 있도록 설계하였다.
3. **효율적인 아키텍처**: Encoder-Decoder 구조 내에 MSTWDecomp(분해 블록)와 KEDA를 통합하여, 장기 예측에서도 안정적인 성능을 유지하도록 하였다.

## 📎 Related Works

### Transformer-based LSTF

기존의 Informer는 ProbSparse attention을 통해 복잡도를 $O(L \log L)$로 낮추었으며, Autoformer는 분해 블록과 Autocorrelation 메커니즘을 도입하였다. FEDformer는 주파수 도메인 강화(frequency-domain enhancements)를 통해 효율성을 높였다. 그러나 이러한 모델들은 복잡한 장기 의존성이나 비주기적인 패턴을 모델링할 때 여전히 불안정한 모습을 보인다.

### Time Series Decomposition

ARIMA나 Prophet 같은 전통적 방법은 선형 가정에 의존하여 복잡한 다변량 시나리오나 비정상성(non-stationary) 데이터 처리에 한계가 있다. 최근의 N-BEATS나 Autoformer 등은 딥러닝 기반 분해를 시도하였으나, KEDformer는 이를 보다 동적으로 통합하고 지식 추출 메커니즘과 결합하여 차별점을 둔다.

## 🛠️ Methodology

### 1. 데이터 분해 (Data Decomposition)

KEDformer는 입력 시퀀스를 추세 성분과 계절성 성분으로 분리하는 MSTWDecomp 블록을 사용한다.

- **추세 추출**: Moving Average(이동 평균)를 위해 `AvgPool` 연산을 적용하여 주기적인 변동을 제거하고 장기적인 흐름을 포착한다.
  $$x_t = \text{AvgPool}(\text{Padding}(X))$$
- **계절성 추출**: 원본 데이터에서 추출된 추세 성분을 빼서 구한다.
  $$x_s = x - x_t$$

### 2. 지식 추출 프로세스 (Knowledge Extraction Process)

#### Knowledge Selection (KL Divergence)

모든 attention 가중치를 사용하는 대신, 균등 분포(uniform distribution) $q$와 실제 attention 확률 분포 $p$ 사이의 KL Divergence를 계산하여 정보량이 많은 Query를 선별한다.
$$KL(q||p) = \ln \frac{1}{L_K} \sum_{j=1}^{L_K} e^{q_i k_j^T / \sqrt{d}} - \dots \text{(상세 수식 생략)}$$
이를 통해 정의된 지표 $M(q_i, K)$ 값이 클수록 해당 Query가 지배적인 패턴을 가지고 있다고 판단하여 우선적으로 처리한다.

#### Decoupled Knowledge Extraction (Autocorrelation)

시계열의 주기적 특성을 파악하기 위해 자기상관 함수(Autocorrelation function)를 사용한다.
$$R_{XX}(\tau) = \lim_{L \to \infty} \frac{1}{L} \sum_{t=1}^{L} X_t X_{t-\tau}$$
여기서 $\tau$는 시간 지연(time lag)을 의미하며, $R_{XX}(\tau)$의 피크(peak) 지점을 통해 가장 확률이 높은 주기 $\tau_1, \dots, \tau_k$를 결정한다.

#### KEDattention 연산

최종적으로 선택된 주기 $\tau_i$에 대해 Value 행렬 $V$를 해당 주기만큼 시프트(Shift)하는 `Roll` 연산을 수행하고, 추출된 가중치 $\hat{R}_{Q,K}(\tau_i)$를 곱하여 합산한다.
$$\text{KEDattention}(\hat{Q}, K, V) = \sum_{i=1}^{k} \text{Roll}(V, \tau_i) \hat{R}_{Q,K}(\tau_i)$$
이 과정은 전체 행렬 곱셈을 피하고 필요한 주기성 정보만 활용하므로 계산 효율성이 매우 높다.

### 3. 전체 시스템 구조

- **Encoder**: KEDA 모듈과 MSTWDecomp가 교대로 배치된 다층 구조로, 과거 데이터의 핵심 특징을 추출한다.
- **Decoder**: Encoder의 출력과 초기값(Seasonal Init, Trend Init)을 입력으로 받는다. 추세 성분은 누적(accumulation)하고, 계절성 성분은 KEDA를 통해 지식을 스택(stacking)하며 미래 값을 예측한다.

## 📊 Results

### 실험 설정

- **데이터셋**: ETT (ETTh1, ETTh2, ETTm1, ETTm2), Electricity, Exchange, Traffic, Weather 등 5개 공개 데이터셋 사용.
- **비교 모델**: Autoformer, Informer, Reformer, LSTNet, LSTM, TCN, LogTrans, DeepAR, Prophet, ARIMA 등.
- **평가 지표**: Mean Squared Error (MSE), Mean Absolute Error (MAE).

### 주요 결과

1. **다변량 예측 (Multivariate)**: 모든 벤치마크에서 KEDformer가 일관되게 우수한 성능을 보였다. 특히 예측 길이가 336일 때, Exchange 데이터셋에서 MSE를 10.4% (0.509 $\rightarrow$ 0.456) 감소시키는 등 괄목할 성과를 냈다. 전체적으로 MSE를 평균 2.48% 감소시켰다.
2. **단변량 예측 (Univariate)**: ETTm2 데이터셋(predict-336)에서 MAE를 2.6% 감소시켰으며, 주기성이 낮은 Exchange 데이터셋에서도 다른 모델 대비 7.7% 높은 성능을 보였다.
3. **계산 효율성**: KEDA 모듈의 도입으로 Epoch당 소요 시간이 크게 단축되었다. 특히 ETTm1 데이터셋에서는 연산 시간이 794.0초에서 467.2초로 감소하였다.
4. **Ablation Study**: Self-attention과 Cross-attention을 모두 KEDA로 교체했을 때 성능 향상이 가장 뚜렷했다.

## 🧠 Insights & Discussion

### 강점 및 분석

KEDformer는 단순한 연산량 감소를 넘어, 시계열 데이터의 **주기성(Periodicity)**과 **추세(Trend)**를 명시적으로 분리하여 처리함으로써 장기 예측의 안정성을 확보하였다. 특히, 주기성이 뚜렷하지 않은 Exchange 데이터셋에서도 좋은 성능을 보인 점은 KL Divergence 기반의 지식 선택 메커니즘이 유연하게 작동하고 있음을 시사한다.

### 한계 및 비판적 해석

논문에서도 언급되었듯이, **완전한 비주기적(non-periodic) 데이터**에 대해서는 계절성-추세 분해와 자기상관 메커니즘의 이점이 줄어들어 성능이 저하될 가능성이 있다. 또한, KEDformer 메커니즘의 개수(Hyperparameter) 설정에 따라 효율성과 정확도가 민감하게 변하는 경향이 있으며, 최적의 가중치 밸런싱을 찾는 문제가 여전히 과제로 남아 있다.

### 구조적 특징

실험 결과, Encoder보다 Decoder에 더 많은 KEDformer 메커니즘을 배치했을 때 성능이 향상되는 경향이 발견되었다. 이는 과거 데이터를 인코딩하는 것보다, 추출된 지식을 바탕으로 미래의 복잡한 의존성을 재구성하는 디코딩 과정에서 더 많은 정보 처리 능력이 필요함을 의미한다.

## 📌 TL;DR

KEDformer는 기존 Transformer의 $O(L^2)$ 복잡도 문제를 해결하기 위해 **KL Divergence 기반의 지식 추출(KEDA)**과 **자기상관(Autocorrelation)**을 결합하여 복잡도를 $O(L \log L)$로 낮춘 모델이다. 동시에 **계절성-추세 분해**를 통해 장단기 패턴을 정밀하게 포착함으로써, 다양한 벤치마크 데이터셋에서 기존 SOTA 모델들보다 높은 예측 정확도와 계산 효율성을 입증하였다. 이 연구는 특히 대규모 시계열 데이터의 실시간 분석 및 장기 예측 시스템 구축에 중요한 기여를 할 것으로 보인다.
