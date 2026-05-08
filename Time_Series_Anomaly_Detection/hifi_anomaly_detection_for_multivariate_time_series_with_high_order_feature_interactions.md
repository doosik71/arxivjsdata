# HIFI: Anomaly Detection for Multivariate Time Series with High-order Feature Interactions

Liwei Deng, Xuanhao Chen, Yan Zhao, and Kai Zheng (2021)

## 🧩 Problem to Solve

본 논문은 서버나 항공기 같은 복잡한 시스템에서 생성되는 다변량 시계열(Multivariate Time Series, MTS) 데이터의 이상치 탐지(Anomaly Detection) 문제를 다룬다. 복잡한 시스템의 정상 작동을 유지하기 위해서는 이상 징후를 적시에 정확하게 탐지하는 것이 매우 중요하지만, 다음과 같은 한계점들이 존재한다.

첫째, 기존의 많은 다변량 시계열 이상 탐지 알고리즘들이 변수 간의 상관관계(Correlation) 모델링을 무시한다. 변수 간의 상호작용은 복잡한 시계열 정보를 모델링하는 데 중요한 단서가 되지만, 이를 간과할 경우 탐지 성능이 저하된다. 둘째, RNN 및 그 변형 모델들은 장기 의존성(Long-term temporal dependencies)을 포착하는 데 한계가 있다. 셋째, 결정론적(Deterministic) 모델들은 표현 능력이 부족하고 강건성(Robustness)이 떨어진다는 문제가 있다.

따라서 본 논문의 목표는 변수 간의 고차원 특징 상호작용(High-order Feature Interactions)을 자동으로 모델링하고, 장기 의존성을 효과적으로 포착하며, 강건성을 높인 비지도 학습 기반의 이상 탐지 모델인 HIFI를 제안하는 것이다.

## ✨ Key Contributions

HIFI의 핵심 아이디어는 다변량 시계열 데이터의 각 변수를 그래프의 노드로 간주하고, 이들 사이의 상호작용을 그래프 신경망(Graph Neural Network, GNN)을 통해 학습하는 것이다.

1. **다변량 특징 상호작용 모듈(Multivariate Feature Interaction Module)**: 변수 간의 관계 그래프를 자동으로 구축하고, GCN을 사용하여 고차원 특징 상호작용을 추출한다.
2. **어텐션 기반 시계열 모델링(Attention-based Time Series Modeling)**: RNN의 한계를 극복하기 위해 Multi-head Attention 메커니즘을 도입하여 장기 의존성을 모델링한다.
3. **변분 인코딩(Variational Encoding)**: 결정론적 인코딩 대신 변분 인코딩 기법을 적용하여 모델의 표현 능력을 높이고 이상 탐지의 강건성을 향상시킨다.

## 📎 Related Works

본 논문은 기존의 비지도 시계열 이상 탐지 방식들의 한계를 지적한다.

- **RNN/LSTM 기반 모델**: EncDec-AD, LSTM-NDT 등은 시퀀스-투-시퀀스 구조나 LSTM을 사용하여 시간적 정보를 모델링하지만, 변수 간의 명시적인 관계를 모델링하지 않는다.
- **변분 오토인코더(VAE) 기반 모델**: OmniAnomaly 등은 확률적 변수 연결과 Normalizing Flow를 통해 정상 패턴을 캡처하여 성능을 높였으나, 여전히 특징 간의 상호작용 모델링이 부족하다.
- **GCN 및 Attention 모델**: GCN은 비정형 그래프 표현에 강점이 있고, Transformer의 Attention 메커니즘은 장기 의존성 포착에 탁월하다. 본 논문은 이러한 최신 기법들을 이상 탐지 분야에 통합하여 기존 접근 방식과 차별화를 둔다.

## 🛠️ Methodology

HIFI 모델은 전체적으로 Encoder-Decoder 구조를 따르며, 크게 세 가지 모듈로 구성된다.

### 1. Multivariate Feature Interaction Module

이 모듈은 변수 간의 관계를 그래프로 구축하고 고차원 특징을 추출한다.

- **그래프 구축**: 원본 특징 $X \in \mathbb{R}^{w \times d}$를 은닉 공간 $X_h \in \mathbb{R}^{w \times d_1}$로 투영한다.
  $$x^h_i = W_h x_i + b_h$$
  이후 비대칭 구조의 임베딩 행렬 $E_1, E_2$를 정의하고, 다음과 같이 특징 상호작용 그래프 $A$를 자동으로 생성한다.
  $$M_1 = \tanh(E_1 \Theta_1), \quad M_2 = \tanh(E_2 \Theta_2)$$
  $$A = \text{ReLU}(\tanh(M_1 M_2^T - M_2^T M_1))$$
  계산 효율성을 위해 `topk` 함수를 사용하여 상위 $k$개의 연결만 남긴 희소 그래프(Sparse Graph)로 변환한다.

- **GCN 모듈**: 구축된 그래프 $A$와 은닉 특징 $X_h$를 GCN에 입력하여 고차원 특징을 얻는다.
  $$\hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} \quad (\text{where } \tilde{A} = A + I_N)$$
  $$H_{k+1} = (1-\alpha) \hat{A} H_k + \alpha H_0$$
  여기서 $\alpha$는 원본 특징을 유지하는 하이퍼파라미터이다. 최종 고차원 특징 $X_{ho}$는 각 층의 출력을 연결(Concatenate)한 후 선형 변환을 통해 얻는다.
  $$X_{ho} = \text{Concat}(H_0^T, H_1^T, \dots, H_K^T) W_{ho}$$

### 2. Attention-based Time Series Modeling Module

장기 의존성을 포착하기 위해 Scaled-dot product attention을 사용한다.
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$
위의 Attention 레이어와 ReLU 활성화 함수를 포함한 비선형 레이어를 교대로 쌓아 Encoder와 Decoder를 구성한다. 또한, 어텐션의 위치 불변성 문제를 해결하기 위해 Sinusoidal Positional Encoding $P$를 추가하여 입력값 $X_{in} = P + X_{ho} + X$를 생성한다.

### 3. Variational Encoding Module

모델의 강건성을 위해 Encoder의 출력 $X_{eo}$를 정규분포로 모델링한다.
$$\mu_i = W_\mu x^{eo}_i + b_\mu, \quad \log \sigma^2_i = W_\sigma x^{eo}_i + b_\sigma$$
Reparameterization trick을 사용하여 샘플 $z_i$를 추출하고 이를 Decoder의 $K$와 $V$로 입력한다.
$$z_i = \mu_i + \epsilon * \sigma_i \quad (\epsilon \sim \mathcal{N}(0, 1))$$

### 4. Model Training 및 이상 탐지

손실 함수는 재구성 오차(Reconstruction Error)와 KL 발산(KL Divergence)의 합으로 정의된다.
$$\text{loss} = \sum_{i=1}^{w} \|x_i - x^{rec}_i\|^2 + \beta \sum_{i=1}^{w} \text{KL}(\mathcal{N}(\mu_i, \sigma^2_i) \| \mathcal{N}(0, 1))$$
이상치 점수(Anomaly Score)는 현재 윈도우의 마지막 시점에서의 재구성 오차 $\|x_t - x^{rec}_t\|^2$로 산출한다.

## 📊 Results

### 실험 설정

- **데이터셋**: SMD(Server Machine Dataset), SMAP(Soil Moisture Active Passive), MSL(Mars Science Laboratory) 세 가지 공개 데이터셋을 사용하였다.
- **지표**: Precision, Recall, $F1_{best}$를 측정하였으며, 시계열 이상 탐지의 특성을 고려하여 Point-adjust 기법을 적용하였다.
- **비교 모델**: LSTM-NDT, EncDec-AD, OED-IF, OmniAnomaly.

### 주요 결과

정량적 분석 결과, HIFI는 모든 데이터셋에서 가장 높은 $F1_{best}$ 성능을 기록하였다. 특히 MSL, SMAP, SMD 데이터셋에서 기존 SOTA 모델 대비 각각 $2.89\%$, $7.42\%$, $0.43\%$의 성능 향상을 보였다.

| Methods | MSL ($F1_{best}$) | SMAP ($F1_{best}$) | SMD ($F1_{best}$) |
| :--- | :---: | :---: | :---: |
| LSTM-NDT | 0.8623 | 0.7852 | 0.7942 |
| EncDec-AD | 0.9039 | 0.8707 | 0.9317 |
| OED-IF | 0.9185 | 0.8458 | 0.9685 |
| OmniAnomaly | 0.9257 | 0.8966 | 0.9337 |
| **HIFI** | **0.9546** | **0.9708** | **0.9737** |

### Ablation Study

각 구성 요소의 효과를 검증한 결과, $\text{w/o FI}$(특징 상호작용 제거) 및 $\text{w/o VE}$(변분 인코딩 제거) 시 성능이 하락함을 확인하였다. 특히 MSL과 SMAP 데이터셋에서 FI와 VE 모듈의 영향이 뚜렷하게 나타났으며, 이는 복잡한 변수 관계 모델링과 강건한 인코딩이 이상 탐지 성능 향상에 필수적임을 시사한다.

## 🧠 Insights & Discussion

본 논문의 강점은 다변량 시계열 데이터에서 간과되기 쉬운 **변수 간 상호작용(Feature Interaction)**을 그래프 구조로 명시화하고, 이를 고차원적으로 확장하여 모델링했다는 점이다. 또한, 단순한 RNN 구조에서 벗어나 Attention 메커니즘을 통해 장기 의존성을 해결하고 VAE 구조를 통해 확률적 강건성을 확보한 통합적 설계가 돋보인다.

다만, 실험 결과에서 SMD 데이터셋의 경우 다른 모듈의 제거 여부에 따른 성능 차이가 크지 않았는데, 이는 SMD 데이터셋 자체가 시계열 정보만으로도 이상 탐지가 충분히 가능한 특성을 가졌기 때문으로 분석된다. 이는 모든 데이터셋에 대해 복잡한 고차원 특징 모델링이 항상 절대적인 성능 향상을 보장하는 것은 아니며, 데이터의 특성에 따라 필요한 모델 복잡도가 다를 수 있음을 시사한다.

## 📌 TL;DR

HIFI는 다변량 시계열 이상 탐지를 위해 **GCN 기반의 변수 상호작용 모델링**, **Attention 기반의 장기 의존성 포착**, **변분 인코딩을 통한 강건성 확보**를 결합한 비지도 학습 모델이다. 실험을 통해 기존 SOTA 모델들을 능가하는 성능을 증명하였으며, 특히 변수 간의 고차원 관계를 자동으로 학습하는 구조가 복잡한 시스템의 이상 탐지에 매우 효과적임을 보여주었다. 향후 연구에서 이 구조는 다변량 센서 데이터 기반의 예지 보전(Predictive Maintenance) 시스템 등에 광범위하게 적용될 가능성이 높다.
