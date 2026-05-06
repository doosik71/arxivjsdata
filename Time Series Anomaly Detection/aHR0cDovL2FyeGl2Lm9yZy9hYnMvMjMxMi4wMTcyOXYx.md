# EdgeConvFormer: Dynamic Graph CNN and Transformer based Anomaly Detection in Multivariate Time Series

Jie Liu, Qilin Li, Senjian An, Bradley Ezard and Ling Li (2023)

## 🧩 Problem to Solve

본 논문은 다변량 시계열(Multivariate Time Series, MTS) 데이터에서 이상치 탐지(Anomaly Detection)를 수행하는 문제를 다룬다. 현대의 제조 산업이나 엔지니어링 서비스에서는 수많은 센서가 복잡한 시스템의 상태를 모니터링하며 방대한 양의 다변량 시계열 데이터를 생성한다. 이러한 시스템에서 이상 징후를 조기에 발견하고 근본 원인을 파악하는 것은 매우 중요하다.

기존의 Transformer 기반 이상 탐지 모델들은 시계열 데이터의 장기 의존성(Long-term dependencies)을 모델링하는 데 강점이 있으나, 다음과 같은 세 가지 주요 한계점을 가진다.

1. **부적절한 위치 인코딩(Positional Encoding):** 바닐라 Transformer의 코사인/사인 함수 기반 위치 인코딩은 고정된 주파수와 위상 이동을 가지므로, 시계열 데이터 특유의 주기성(Periodicity)과 복잡한 시간적 패턴을 충분히 반영하지 못한다.
2. **귀납적 편향(Inductive Bias)의 부족:** Transformer는 CNN과 같은 평행 이동 불변성(Translational Invariance)이나 지역성(Locality)과 같은 귀납적 편향이 부족하여, 학습을 위해 막대한 양의 데이터가 필요하다.
3. **센서 간 위상 구조(Topology) 무시:** 기존 모델들은 다수의 센서 간의 상호 의존성이나 동적인 위상 관계를 고려하지 않아, 여러 센서에 걸쳐 발생하는 이상 징후를 탐지하고 설명하는 능력이 제한적이다.

따라서 본 논문의 목표는 시간적 주기성, 지역적 시공간 특징, 그리고 센서 간의 동적 관계를 모두 포착할 수 있는 새로운 모델인 **EdgeConvFormer**를 제안하여 다변량 시계열 이상 탐지 성능을 향상시키는 것이다.

## ✨ Key Contributions

EdgeConvFormer의 핵심 아이디어는 **Time2vec 임베딩, 동적 그래프 CNN(EdgeConv), 그리고 Transformer를 계층적으로 통합**하여 전역적 및 지역적 시공간 정보를 동시에 추출하는 것이다.

- **Time2Vec의 도입:** 학습 가능한 벡터 표현을 통해 시계열의 주기적 패턴과 비주기적 패턴을 동시에 캡처하여 Transformer의 위치 인코딩 문제를 해결한다.
- **EdgeConv를 통한 지역성 보완:** Dynamic Graph CNN의 일종인 EdgeConv를 사용하여 시공간 임베딩 공간에서 가장 관련성이 높은 인접 노드들을 동적으로 탐색함으로써, Transformer의 지역성 부족 문제를 해결하고 센서 간의 시공간적 위상 관계를 학습한다.
- **계층적-다중 스케일 구조:** EdgeConv와 Transformer를 다층으로 쌓아(Stacked) 지역적 특징과 전역적 특징을 반복적으로 정제하고 결합함으로써 임베딩의 표현 능력을 극대화한다.

## 📎 Related Works

### 1. 딥러닝 기반 이상 탐지

기존 연구들은 CNN, RNN/LSTM, Autoencoder(AE), VAE 등을 활용하여 이상 탐지를 수행해 왔다. 특히 재구성 기반(Reconstruction-based) 모델은 정상 데이터의 재구성 오차를 최소화하도록 학습하여, 이상치가 입력되었을 때 발생하는 큰 재구성 오차를 통해 이상을 판별한다. 하지만 이러한 모델들은 노이즈에 취약하거나 복잡한 상호 의존성을 캡처하는 데 한계가 있다.

### 2. Transformer 기반 이상 탐지

Self-attention 메커니즘을 통해 장기 의존성을 모델링하는 Transformer 계열 모델들이 등장하였으나, 앞서 언급한 위치 인코딩 문제와 메모리 병목 현상, 그리고 다변량 시계열의 센서 간 관계 고려 부족이라는 문제가 여전히 존재한다.

### 3. GNN 기반 이상 탐지

Graph Neural Networks(GNN)는 센서 간의 복잡한 관계를 모델링하는 데 성공적인 접근 방식을 보여주었다. 하지만 대부분의 기존 GNN 기반 모델들은 정적인 공간 위상(Static Spatial Topology)을 가정하며, 실제 시스템에서 발생하는 동적인 시공간적 관계(Spatiotemporal relationship)를 반영하지 못하는 한계가 있다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

EdgeConvFormer는 **Encoder-Decoder** 구조의 재구성 기반 모델이다. 전체 파이프라인은 다음과 같다:
`입력 데이터` $\rightarrow$ `Time2Vec 임베딩` $\rightarrow$ `EdgeConv-Transformer Encoder (4개 층)` $\rightarrow$ `MLP Decoder` $\rightarrow$ `재구성된 데이터`

### 2. 주요 구성 요소 및 상세 설명

#### (1) Time2Vec Embedding

각 센서의 시계열 데이터를 주기적 패턴과 비주기적 패턴으로 분해하여 $(m+1)$ 차원의 임베딩으로 변환한다.
$$ \text{t2v}(\mathbf{x}_i)[j] = \begin{cases} \omega_j \mathbf{x}_i + \phi_j, & j = 0 \\ \sin(\omega_j \mathbf{x}_i + \phi_j), & 1 \le j \le m \end{cases} $$
여기서 $\omega$와 $\phi$는 학습 가능한 파라미터이다. $j=0$일 때의 선형 항은 비주기적 패턴을, $j \ge 1$일 때의 사인 함수는 다양한 주파수의 주기적 패턴을 캡처한다.

#### (2) Encoder (EdgeConv + Transformer)

인코더는 EdgeConv 모듈과 Transformer 모듈이 한 쌍을 이루는 4개의 층으로 구성된다.

- **EdgeConv Module:** 시공간 2차원 공간$(\text{sensors} \times \text{timestamps})$에서 $k$-최근접 이웃($k$-NN) 그래프를 동적으로 생성한다. 중심점 $i$와 이웃 $j$의 특징을 결합하여 새로운 표현을 얻는다.
  $$ h^{(l+1)}_i = \max_{j \in \mathcal{N}(i)} \left( \text{ReLU}(\Theta \cdot (h^{(l)}_j - h^{(l)}_i) + \Phi \cdot h^{(l)}_i) \right) $$
  여기서 $\Theta, \Phi$는 선형 레이어이며, $\max$ 풀링을 통해 가장 상관관계가 높은 엣지 특징을 추출한다.

- **Transformer Module:** EdgeConv가 추출한 지역적 특징을 바탕으로, 시간 차원에 대해 Self-attention을 수행하여 장거리 시간 의존성을 캡처한다. 바닐라 Transformer의 인코더 구조를 사용하되, 이미 Time2Vec에서 위치 정보가 반영되었으므로 별도의 위치 임베딩은 제거하였다.

#### (3) Decoder 및 손실 함수

디코더는 인코더의 각 층에서 나온 다중 스케일 특징들을 연결(Concat)하고 MLP를 통해 집계한다. 최종적으로 입력 시계열과 동일한 차원으로 선형 투영하여 재구성된 시계열 $\hat{\mathbf{x}}$를 생성한다.
학습 시에는 원본 신호와 재구성 신호 사이의 평균 제곱 오차(MSE)를 손실 함수로 사용한다.
$$ \mathcal{L}_{MSE} = \frac{1}{W} \sum_{t=0}^{W-1} \|\hat{\mathbf{x}}^t - \mathbf{x}^t\|_2^2 $$

### 3. 이상치 판별 절차 (Anomaly Scoring)

재구성 오차 $\text{Er}_{it} = \|\hat{x}_{it} - x_{it}\|_1$를 기반으로 **동적 가우시안 스코어링 함수(Dynamic Gaussian Scoring)**를 적용한다. 롤링 윈도우를 통해 평균 $\mu_{it}$와 분산 $\sigma_{it}$를 지속적으로 업데이트하며, 누적 분포 함수(cdf)를 이용하여 센서별 이상 점수 $a_{it}$를 계산하고, 모든 센서의 점수를 합산하여 최종 점수 $S_t$를 산출한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋:** SMD, MSL, SMAP, SWaT, PSM (일반 MTS 데이터셋) 및 Exathlon (고차원 범위 기반 데이터셋).
- **비교 대상(Baselines):** UAE, LSTM-VAE, MSCRED, OmniAnomaly, BeatGAN, TCN-AE, Anomaly Transformer, TranAD.
- **측정 지표:** $F_1$, $F_{pa1}$ (Point-adjusted $F_1$), $F_{c1}$ (Composite $F_1$), AU-ROC, AU-PRC 및 Range-based $F_1$ (Exathlon용).

### 2. 주요 결과

- **정량적 성능:** EdgeConvFormer는 MSL, SMAP, SWaT, PSM 데이터셋에서 거의 모든 지표에 대해 최상위(Best) 또는 이에 준하는 성능을 보였으며, 5개 데이터셋 평균 랭킹에서 1위를 기록하였다.
- **범위 기반 탐지 성능:** Exathlon 데이터셋 실험 결과, 특히 까다로운 AD4 레벨(정확한 범위 및 조기 탐지 요구)에서도 다른 모델들이 거의 탐지에 실패한 반면, EdgeConvFormer는 $0.7309$의 $F_1$-Range 점수를 기록하며 압도적인 강건성을 보였다.
- **효율성:** 데이터 포인트당 평균 실행 시간은 약 $26\text{ms}$로, MSCRED($60\text{ms}$)보다는 훨씬 빠르며 정확도와 효율성 사이의 적절한 균형을 맞춘 것으로 나타났다.

## 🧠 Insights & Discussion

### 1. 강점 및 분석

본 논문은 Transformer의 고질적인 문제인 '지역성 부족'과 '위상 구조 무시'를 **EdgeConv**라는 동적 그래프 구조를 통해 효과적으로 해결하였다. 특히, Time2Vec를 통해 시계열의 주기성을 먼저 학습하고 이를 EdgeConv와 Transformer가 순차적으로 정제하는 구조가 다변량 시계열의 복잡한 시공간적 상관관계를 포착하는 데 매우 핵심적이었음을 확인하였다.

### 2. Ablation Study 결과

- **Time2Vec 제거 시:** $F_1$ 및 AU-PRC 점수가 크게 하락하여, 주기적/비주기적 패턴 학습의 중요성이 입증되었다.
- **EdgeConv 제거 시:** 성능 하락폭이 가장 컸다. 이는 시공간적 지역 특징 추출이 모델 성능의 가장 중추적인 역할을 수행함을 의미한다.
- **Transformer 제거 시:** 성능이 하락하였으나 EdgeConv보다는 영향이 적었다. 이는 EdgeConv가 일차적으로 유의미한 정보를 제공하고, Transformer는 이를 보완하여 전역적 의존성을 강화하는 관계임을 시사한다.

### 3. 비판적 해석 및 한계

EdgeConvFormer는 매우 뛰어난 성능을 보이지만, 여전히 모든 시점(point-wise)에 대해 그래프를 생성하고 어텐션을 수행하므로 연산 복잡도가 존재한다. 저자들 또한 결론에서 언급했듯이, 포인트 단위 표현을 서브-시리즈(sub-series) 레벨로 확장하여 메모리 소비와 계산 복잡도를 줄이는 방향의 후속 연구가 필요할 것으로 보인다.

## 📌 TL;DR

EdgeConvFormer는 **Time2Vec(시간 주기성) $\rightarrow$ EdgeConv(동적 시공간 위상) $\rightarrow$ Transformer(전역 시간 의존성)**를 계층적으로 결합한 다변량 시계열 이상 탐지 모델이다. 기존 Transformer 기반 모델의 한계인 지역성 부족과 센서 간 상관관계 무시 문제를 해결하였으며, 특히 실제 산업 데이터셋과 고차원 범위 기반 이상 탐지(Exathlon)에서 SOTA(State-of-the-art) 성능을 달성하였다. 이 연구는 복잡한 시스템의 다변량 데이터에서 지역적 특징과 전역적 특징을 동시에 고려하는 것이 이상 탐지 성능 향상에 결정적임을 보여주었다.
