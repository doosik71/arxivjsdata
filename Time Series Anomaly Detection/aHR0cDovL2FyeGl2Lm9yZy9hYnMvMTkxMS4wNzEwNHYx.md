# RSM-GAN: A Convolutional Recurrent GAN for Anomaly Detection in Contaminated Seasonal Multivariate Time Series

Farzaneh Khoshnevisan, Zhewen Fan (2019)

## 🧩 Problem to Solve

본 논문은 다변량 시계열(Multivariate Time Series, MTS) 데이터에서 이상치 탐지(Anomaly Detection, AD)를 수행할 때 발생하는 세 가지 핵심 문제를 해결하고자 한다.

첫째, **복잡한 계절성(Seasonality)의 처리**이다. 실제 시스템에서 발생하는 시계열 데이터는 단순한 주기성이 아니라 여러 개의 복잡하고 불규칙한 계절성 패턴이 동시에 나타나는 경우가 많으며, 이를 정확히 모델링하지 못할 경우 오탐지율(False Positive Rate, FPR)이 높아진다.

둘째, **훈련 데이터의 오염(Contamination)** 문제이다. 기존의 딥러닝 기반 AD 모델들은 훈련 데이터가 오직 '정상' 데이터로만 구성되어 있다는 가정을 전제로 한다. 그러나 실제 환경에서 수집된 데이터에는 이미 탐지되지 않은 이상치나 노이즈가 포함되어 있을 가능성이 높으며, 이러한 오염된 데이터는 모델의 성능을 저하시킨다.

셋째, **근본 원인 분석(Root Cause Analysis)의 필요성**이다. 단순히 이상 시점을 찾아내는 것을 넘어, 어떤 변수가 해당 이상치에 기여했는지 식별하는 것이 실제 시스템 운영자에게는 필수적이다.

결과적으로 본 연구의 목표는 오염된 훈련 데이터와 복잡한 계절성이 존재하는 MTS 환경에서도 강건하게 동작하며, 근본 원인을 식별할 수 있는 비지도 학습 기반의 딥러닝 아키텍처인 RSM-GAN을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같이 요약할 수 있다.

1. **RSM-GAN 아키텍처 제안**: Convolutional-LSTM 레이어를 결합한 Wasserstein GAN 구조를 통해 MTS의 공간적(Spatial) 상관관계와 시간적(Temporal) 의존성을 동시에 캡처한다.
2. **평활화된 어텐션 메커니즘(Smoothed Attention Mechanism)**: 다중 계절성 패턴을 모델링하기 위해 과거의 계절적 시점을 참조하고, 이를 어텐션 메커니즘을 통해 가중치를 부여함으로써 계절성 변화에 유연하게 대응한다.
3. **오염된 데이터에 대한 강건성 확보**: GAN의 생성기 외에 추가적인 Encoder를 동시에 학습시켜, 훈련 데이터 내에 이상치가 포함되어 있더라도 잠재 공간(Latent Space)에서 정상 데이터의 분포를 더 잘 학습하도록 설계하였다.
4. **새로운 이상치 점수 할당 및 근본 원인 추론 프레임워크**: 단순한 재구성 오차 기반의 점수 산출 방식에서 벗어나, 근본 원인 기반의 카운팅 절차($context_h$)와 Elbow Method를 이용한 최적의 원인 변수 개수($k$) 식별 방법을 제안한다.

## 📎 Related Works

논문에서는 MTS 이상치 탐지 방법을 세 가지 범주로 나누어 설명하며 기존 연구의 한계를 지적한다.

1. **전통적 시계열 분석(TSA) 및 머신러닝 방법**: Vector Autoregression(VAR), Kalman Filter와 같은 TSA 방법이나 k-Nearest Neighbor(kNN), One-Class SVM, Isolation Forest 같은 ML 방법들이 존재한다. 하지만 이들은 변수 간의 상호 의존성이나 시간적 의존성, 특히 계절성 패턴을 적절히 처리하지 못한다는 한계가 있다.
2. **딥러닝 기반 방법 (Autoencoder-based)**: RNN과 결합된 Autoencoder 모델들은 시간적/공간적 의존성을 잘 캡처하지만, 여전히 계절성 문제를 해결하지 못했으며 훈련 데이터가 오염되지 않았다는 가정을 유지한다.
3. **GAN 기반 방법**: 이미지 분야에서는 GAN을 이용해 데이터 분포를 학습하고 재구성 오차를 통해 이상치를 탐지하는 연구가 성과를 거두었으나, 이를 MTS 구조에 적용한 사례는 부족하며 대부분의 모델이 오염되지 않은 훈련 데이터를 가정한다.

RSM-GAN은 이러한 기존 연구들과 달리 MTS를 이미지 형태의 상관 행렬로 변환하여 GAN에 적용하고, 어텐션과 추가 인코더를 통해 계절성과 데이터 오염 문제를 동시에 해결함으로써 차별점을 갖는다.

## 🛠️ Methodology

### 1. MTS to Image Conversion (MCM)

MTS의 변수 간 상관관계를 캡처하기 위해, 원본 데이터를 **Multi-Channel Correlation Matrix (MCM)**라는 이미지 유사 구조로 변환한다. 윈도우 크기 $W = \{5, 10, 30\}$를 설정하여 각 윈도우 내에서 시계열 간의 내적(Inner Product)을 계산한다. 시점 $t$에서 윈도우 크기 $w$에 대한 행렬 $S_t^w$의 요소 $s_{ij}$는 다음과 같이 계산된다.

$$s_{ij} = \sum_{\delta=0}^{w} x_{t-\delta}^i \cdot x_{t-\delta}^j$$

이 과정을 통해 $c$개의 채널을 가진 $n \times n$ 크기의 행렬들이 생성되며, 시간적 의존성을 위해 $h=4$개의 이전 스텝을 스택하여 모델의 입력으로 사용한다.

### 2. RSM-GAN Architecture

RSM-GAN은 **Encoder-Decoder-Encoder** 구조를 가진다.

- **Generator ($G$)**: Autoencoder 구조로 구성되어 있으며, 인코더($G_E$)와 디코더($G_D$)가 입력 $x$와 재구성된 $x'$ 사이의 거리($l_2$ distance)를 최소화하는 Reconstruction Loss를 학습한다.
- **Additional Encoder ($E$)**: 생성기와 동시에 학습되며, 잠재 벡터 $z$와 재구성된 잠재 벡터 $z'$ 사이의 거리를 최소화하는 Latent Loss를 학습한다. 이는 오염된 데이터에 대해 모델의 강건성을 높이는 역할을 한다.
- **Discriminator ($D$)**: 실제 입력 $x$와 생성된 입력 $G(x)$를 구분하며, Feature Matching Loss를 사용하여 학습을 최적화한다.

학습의 안정성과 수렴 속도를 높이기 위해 **Wasserstein GAN with Gradient Penalty (WGAN-GP)**를 채택하였으며, 목적 함수는 다음과 같다.

$$L_D = \max_{w \in W} \mathbb{E}_{x \sim p_x}[f_w(x)] - \mathbb{E}_{x \sim p_x}[f_w(G(x))]$$

$$L_G = \min_G \min_E \left( w_1 \mathbb{E}_{x \sim p_x} \|x - G(x)\|^2 + w_2 \mathbb{E}_{x \sim p_x} \|G_E(x) - E(G(x))\|^2 + w_3 \mathbb{E}_{x \sim p_x} [f_w(G(x))] \right)$$

여기서 $w_1, w_2, w_3$는 각 손실 함수의 가중치이다.

### 3. Internal Structure & Seasonality Adjustment

- **Conv-LSTM**: 공간적 패턴과 시간적 흐름을 동시에 잡기 위해 모든 컨볼루션 레이어에 Conv-LSTM 게이트를 적용하였다.
- **Smoothed Attention Mechanism**: 계절성 처리를 위해 현재 시점 $t$ 외에 과거의 계절적 시점(예: 하루 전, 일주일 전 동일 시간)의 MCM 데이터를 추가로 입력한다. 이때 30분 윈도우 내에서 평균을 내어 평활화(Smoothing)를 수행한다.
- **Attention 가중치**: 최근성보다 유사성에 기반하여 중요도를 결정하도록 어텐션 메커니즘을 적용하며, 가중치 $\alpha_i$는 다음과 같이 계산된다.

$$\alpha_i = \text{softmax} \left( \frac{Vec(H_t)^T Vec(H_i)}{X} \right), \quad H'_t = \sum_{i \in (t-N, t)} \alpha_i H_i$$

또한, 공휴일과 같은 특이 시점의 영향을 제거하기 위해 바이너리 비트 $b_i \in \{0, 1\}$를 곱하여 불필요한 시점의 정보를 차단한다.

### 4. Testing Phase: Scoring & Causal Inference

- **Anomaly Score ($context_h$)**: 재구성 오차가 임계값 $\theta_b$보다 큰 요소를 'Broken Tile'이라 정의한다. 기존 방식($context_b$)은 단순 개수를 세었으나, 제안된 $context_h$는 행/열의 절반 이상이 Broken Tile인 경우에만 개수를 세는 방식을 사용하여 오탐지를 줄인다.
- **Root Cause Framework**: 각 변수별로 Absolute Error의 합 등을 계산하여 점수를 매기고, **Elbow Method**를 통해 급격하게 점수가 변하는 지점을 찾아 최적의 근본 원인 변수 개수 $k$를 결정한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: 합성 데이터(다중 계절성 및 오염 수준 조절)와 실제 암호화 키(Encryption Key) 데이터셋을 사용하였다.
- **비교 모델**: OC-SVM, Isolation Forest, MSCRED.
- **지표**: Precision, Recall, F1, FPR 및 실시간 탐지 성능을 측정하는 NAB(Numenta Anomaly Benchmark) Score를 사용하였다.

### 2. 주요 결과

- **점수 산출 방식 비교**: 제안된 $context_h$ 방식이 $context_b$나 Latent-based 방식보다 Precision과 NAB Score 면에서 우수함을 확인하였다.
- **오염 내성 평가**: 훈련 데이터의 오염 수준이 심해질수록 MSCRED의 성능은 급격히 하락하였으나, RSM-GAN은 모든 오염 수준에서 가장 낮은 FPR과 높은 F1-score를 유지하였다.
- **계절성 처리 평가**: 계절성 패턴(Daily, Weekly, Monthly)이 복잡해질수록 타 모델들은 Precision이 급락하고 FPR이 상승하였다. 특히 공휴일 데이터 처리에서 RSM-GAN은 바이너리 비트를 이용한 조정 덕분에 압도적인 성능을 보였다.
- **실제 데이터 적용**: 암호화 키 데이터셋에서도 RSM-GAN은 타 모델 대비 높은 Precision과 NAB Score를 기록하며 실용성을 입증하였다.

## 🧠 Insights & Discussion

본 논문의 분석 결과, RSM-GAN은 다음과 같은 강점을 가진다.

첫째, **추가 인코더의 도입이 실질적인 강건성을 제공**한다. 기존의 GAN 기반 AD 모델들이 훈련 데이터의 순수성을 가정하는 것과 달리, 잠재 공간에서의 일관성을 강제하는 추가 인코더 구조가 오염된 데이터 환경에서도 정상 분포를 효과적으로 학습하게 함을 보여주었다.

둘째, **어텐션 메커니즘을 통한 도메인 지식(공휴일 등)의 결합**이 효과적이다. 단순히 데이터만으로 학습하는 것이 아니라, 공휴일과 같은 외부 정보를 바이너리 마스크 형태로 어텐션 가중치에 결합함으로써, 모델이 계절적 특이사항을 이상치로 오인하는 문제를 효과적으로 해결하였다.

셋째, **시계열 데이터를 이미지 형태로 변환(MCM)하여 처리**함으로써, 1D 시계열 모델들이 놓치기 쉬운 변수 간의 고차원적 상관관계를 CNN의 공간적 특성으로 캡처할 수 있었다.

다만, 실제 암호화 키 데이터셋에서의 성능이 합성 데이터보다 낮게 나타난 점은 실제 데이터의 극심한 노이즈와 전문가 라벨링의 불확실성에서 기인한 것으로 보이며, 이는 향후 더 정교한 노이즈 제거 기법이나 반지도 학습으로 해결해야 할 과제로 보인다.

## 📌 TL;DR

RSM-GAN은 **MCM 변환, Conv-LSTM, 추가 인코더, 그리고 평활화된 어텐션 메커니즘**을 결합하여, 복잡한 계절성과 오염된 훈련 데이터가 존재하는 다변량 시계열 환경에서 매우 강건한 이상치 탐지 성능을 보이는 모델이다. 특히 **낮은 오탐지율(FPR)과 빠른 탐지 속도(NAB score)**를 달성하였으며, Elbow Method 기반의 근본 원인 식별 프레임워크를 통해 실무적인 진단 가능성을 제시하였다. 이 연구는 GAN을 시계열 도메인으로 확장하고 실제 산업 현장의 제약 사항(데이터 오염, 계절성)을 아키텍처 수준에서 해결했다는 점에서 향후 실시간 시스템 모니터링 및 보안 관제 분야에 중요한 기여를 할 것으로 평가된다.
