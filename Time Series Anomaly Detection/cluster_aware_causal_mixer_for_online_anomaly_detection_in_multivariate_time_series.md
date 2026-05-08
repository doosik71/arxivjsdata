# Cluster-Aware Causal Mixer for Online Anomaly Detection in Multivariate Time Series

Md Mahmuddun Nabi Murad, Yasin Yilmaz (2025)

## 🧩 Problem to Solve

본 논문은 다변량 시계열(Multivariate Time Series, MTS) 데이터에서 이상치를 조기에 정확하게 탐지하는 문제를 다룬다. 특히 기존의 MLP-Mixer 기반 모델들이 가진 두 가지 핵심적인 한계점을 해결하고자 한다.

첫째, 기존 MLP-Mixer 모델은 시스템 고유의 시간적 의존성을 보존할 수 있는 인과성(Causality) 메커니즘이 부족하다. 실제 시계열 시스템은 현재의 상태가 과거의 데이터에 의해서만 영향을 받는 인과적 특성을 가지지만, 일반적인 Mixer 모델은 룩백 윈도우(Look-back window) 내의 모든 데이터 포인트가 서로 상호작용하게 되어 미래의 정보가 현재에 영향을 주는 데이터 누수 문제가 발생할 수 있다.

둘째, 다변량 시계열 데이터의 수많은 채널들은 서로 매우 다양한 상관관계를 가진다. 모든 채널에 대해 단일 임베딩 메커니즘을 적용할 경우, 서로 관계가 없는 채널들 사이에 가짜 상관관계(Spurious correlation)가 생성되어 모델의 일반화 성능이 저하될 위험이 있다.

마지막으로, 기존의 일부 고성능 모델(예: SensitiveHUE)은 테스트 세트 전체의 통계량을 미리 알아야 하는 오프라인 정규화에 의존하고 있어, 실시간(Online) 이상 탐지 적용에 한계가 있다. 따라서 본 논문의 목표는 인과성을 보존하고 채널 간 상관관계를 효율적으로 반영하며 실시간 탐지가 가능한 **CCM-TAD** 모델을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 크게 세 가지 설계 아이디어로 요약된다.

1. **Causal MLP-Mixer 설계**: MLP-Mixer 구조에 마스킹 메커니즘을 도입하여, 특정 시점의 표현(Representation)이 오직 현재와 과거의 정보에 의해서만 결정되도록 보장하는 Causal Mixer를 제안한다.
2. **Cluster-Aware Multi-Embedding**: 모든 채널을 하나의 임베딩 층으로 처리하는 대신, Spectral Clustering을 통해 상관관계가 유사한 채널끼리 그룹화하고 각 클러스터마다 전용 임베딩 층을 할당함으로써 가짜 상관관계를 억제하고 표현력을 높였다.
3. **Sequential Anomaly Scoring**: 단순한 포인트 기반 탐지에서 벗어나, 이상 징후(Anomaly evidence)를 시간에 따라 누적하여 판단하는 방식을 제안한다. 이를 통해 일시적인 노이즈로 인한 오탐(False Positive)을 줄이고 이상 구간의 시작과 끝을 더 정확하게 식별한다.

## 📎 Related Works

기존의 시계열 이상 탐지 연구는 크게 비지도 학습 기반의 예측(Prediction) 또는 재구성(Reconstruction) 모델로 나뉜다.

- **Autoencoder 및 RNN 기반**: LSTM-VAE, USAD 등은 정상 데이터의 저차원 표현을 학습하여 재구성 오차를 통해 이상치를 탐지한다.
- **Graph 기반**: GDN, MTAD-GAT 등은 채널 간의 관계를 그래프 구조로 모델링하여 공간적 의존성을 캡처한다.
- **Transformer 기반**: 최근 NPSR, SensitiveHUE 등이 높은 성능을 보였으나, Transformer 구조의 복잡성과 더불어 일부 모델의 경우 테스트 세트 전체 데이터가 필요한 정규화 과정 때문에 실시간 탐지가 어렵다는 한계가 있다.
- **MLP-Mixer 기반**: 최근 연구들에 따르면 단순한 MLP 구조가 Transformer보다 시계열 예측에서 더 효율적일 수 있음이 밝혀졌으나, 시계열 데이터 처리에 필수적인 인과성(Causality)을 보장하는 Mixer 구조는 아직 제시되지 않았다.

## 🛠️ Methodology

### 1. 전체 파이프라인

CCM-TAD는 재구성 기반(Reconstruction-based) 모델이다. 입력 데이터 $X^L \in \mathbb{R}^{L \times C}$ (윈도우 길이 $L$, 채널 수 $C$)가 들어오면 다음과 같은 과정을 거친다.
$$\text{Input} \rightarrow \text{Cluster-Aware Multi-Embedding} \rightarrow \text{Causal Mixer Layers} \rightarrow \text{Head Layer} \rightarrow \text{Reconstructed Output } \hat{x}_t$$
최종적으로 원본 데이터 $x_t$와 재구성된 $\hat{x}_t$ 사이의 재구성 오차를 기반으로 이상 여부를 판단한다.

### 2. Cluster-Aware Multi-Embedding

채널 간의 복잡한 관계를 처리하기 위해 다음과 같은 절차로 임베딩을 수행한다.

- **채널 클러스터링**: Pearson 상관계수 행렬 $\Phi$를 구하고, 각 채널의 상관관계 프로필(행 벡터) 간의 코사인 유사도를 기반으로 가중 인접 행렬 $W$를 구성한다. 이후 Normalized Graph Laplacian $L_n = I - D^{-1/2}WD^{-1/2}$의 고유벡터를 추출하는 Spectral Embedding을 수행하고, K-Means를 통해 채널들을 $M$개의 클러스터로 나눈다.
- **다중 임베딩 층**: 각 클러스터 $i$에 대해 전용 임베딩 층을 할당한다. 각 클러스터의 임베딩 차원 $d_i$는 해당 클러스터에 속한 채널 수 $C_i$에 비례하여 설정하여 전체 차원의 합이 $d$가 되도록 한다.
  $$d_i = \lfloor \frac{C_i}{C} d \rfloor \quad (\text{for } i=1, \dots, M-1)$$
  이 방식은 단일 임베딩 층을 사용할 때보다 학습 파라미터 수를 약 $1/M$ 수준으로 줄이는 효과가 있다.

### 3. Causal Mixer

Causal Mixer는 **Causal Temporal Mixer**와 **Embedding Mixer**의 결합으로 구성된다.

- **Causal Temporal Mixer**: 시간 축을 따라 정보를 섞되 인과성을 유지한다. 이를 위해 가중치 행렬 $\Theta_c \in \mathbb{R}^{L \times L}$에 상삼각 마스크(Upper-triangular mask) $\Gamma$를 적용하여 하삼각 성분만 남긴 $\Theta_u = \Gamma \odot \Theta_c$를 사용한다.
  - 마스크 $\gamma_{i,j}$는 $i \le j$일 때 1, 그 외에는 0이다. 이를 통해 $j$번째 출력 뉴런은 $i \le j$인 입력 뉴런에만 의존하게 되어 미래 정보의 유입을 차단한다.
- **Embedding Mixer**: 임베딩 차원 방향으로 정보를 섞으며, 일반적인 Linear Layer와 GELU 활성화 함수를 사용한다.

### 4. Sequential Anomaly Detection

포인트 기반의 한계를 극복하기 위해 이상 증거를 누적하는 방식을 사용한다.

1. **이상 증거($\beta_t$) 계산**: 재구성 오차의 p-value($p_t$)를 이용하여 통계적 유의성 파라미터 $\alpha$와 함께 다음과 같이 계산한다.
    $$\beta_t = \log \left( \frac{\alpha}{p_t + \epsilon} \right)$$
2. **증거 누적($s_t$)**: $\beta_t$를 시간에 따라 누적하며, 특정 조건(윈도우 $\delta$ 내 모든 $\beta$가 0보다 작은 경우 등)에서 리셋된다.
    $$s_t = \max(s_{t-1} + \beta_t, 0)$$
3. **판단 및 구간 확장**: $s_t$가 임계값 $h$를 초과하면 이상치로 판정한다. 이후 backward search를 통해 증거가 상승하기 시작한 시점($t^*_a$)과 하락하기 시작한 시점($t^*_b$)을 찾아 이상 구간을 정교하게 확정한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: SWaT, WADI, PSM, SMD, MSL, SMAP 등 6개의 공개 벤치마크 데이터셋을 사용하였다.
- **비교 대상**: Autoencoder 기반(USAD 등), Graph 기반(GDN 등), Transformer 기반(TranAD, NPSR, SensitiveHUE 등) 총 15개 모델과 비교하였다.
- **지표**: F1 Score를 주 지표로 사용하였으며, Point Adjustment를 적용하지 않은 엄격한 기준으로 평가하였다.

### 2. 주요 결과

- **정량적 성능**: CCM-TAD는 거의 모든 데이터셋에서 기존 모델들보다 우수한 F1 Score를 기록하였다. 특히 PSM(+10.32%), MSL(+12.34%), SMD(+11.96%), WADI(+4.71%)에서 상당한 성능 향상을 보였다.
- **실시간 탐지 능력**: SWaT 데이터셋에서 SensitiveHUE(Offline)가 약간 더 높았으나, 실시간 탐지가 가능한 SensitiveHUE(Online) 버전과는 CCM-TAD가 더 우수한 성능을 보였다.
- **Sequential 탐지의 효과**: 포인트 기반 탐지 방식보다 제안된 누적 탐지 방식이 모든 데이터셋에서 F1 Score와 Recall을 향상시켰으며, 이는 특히 이상 구간 내의 누락(False Negative)을 줄이는 데 효과적임을 확인하였다.

### 3. 절제 연구 (Ablation Study)

- **모듈 기여도**: Causal Temporal Mixer와 Channel Clustering을 모두 적용했을 때 최적의 성능(Case-10)이 나타났다. 특히 Non-causal mixer보다 Causal mixer가 시간적 정보를 훨씬 더 효과적으로 캡처함을 입증하였다.
- **클러스터링 변형**: 단순 코사인 유사도나 K-Means보다 본 논문에서 제안한 상관관계 프로필 기반의 Spectral Clustering이 가장 높은 성능을 보였다.
- **가짜 상관관계 억제**: 채널 클러스터링을 적용했을 때, 재구성된 데이터의 상관관계 행렬과 실제 데이터의 상관관계 행렬 간의 차이(Spurious Correlation)가 PSM 데이터셋 기준 평균 27.98% 감소하였다.

## 🧠 Insights & Discussion

본 논문은 MLP-Mixer의 효율성을 유지하면서도 시계열 데이터의 핵심인 '인과성'과 '채널 간 복잡한 상관관계'를 성공적으로 통합하였다.

**강점:**

- **효율적인 파라미터 관리**: 채널을 클러스터링하여 다중 임베딩을 적용함으로써, 성능을 높이면서도 임베딩 층의 파라미터 수를 $1/M$로 줄인 점이 인상적이다.
- **실무적 탐지 메커니즘**: 단순히 오차 값 하나로 판단하지 않고, 통계적 p-value와 증거 누적 방식을 도입하여 실제 산업 현장에서 발생할 수 있는 일시적 노이즈에 강건한 탐지 체계를 구축하였다.

**한계 및 논의사항:**

- **하이퍼파라미터 민감도**: 저자들은 이상 탐지 성능이 $\alpha$ 값에 민감하며, 이는 모델이 정상 데이터로 얼마나 최적으로 학습되었느냐에 의존한다고 명시하였다. 데이터셋의 크기가 작을 경우 결과가 불안정할 수 있다는 점이 한계로 지적된다.
- **Temporal Mixer의 확장성**: 현재 Causal Temporal Mixer의 확장 계수(expansion factor)가 1로 고정되어 있다. 이를 1보다 크게 설정하면 인과성이 깨지는 문제가 발생하는데, 이에 대한 대안적인 구조 탐색이 향후 과제로 남아있다.
- **공통 모듈의 한계**: 클러스터별 임베딩을 사용함에도 불구하고 이후의 Embedding Mixer와 Head Layer는 공유된다. 이 과정에서 다시 채널들이 섞이므로, 완전히 분리된 Mixer 구조를 사용하는 것이 가짜 상관관계를 더 완벽히 제거할 수 있을 것이라는 분석이 가능하다.

## 📌 TL;DR

본 논문은 다변량 시계열 이상 탐지를 위해 **인과성을 보존하는 MLP-Mixer(CCM-TAD)**를 제안한다. 핵심은 **(1) Spectral Clustering을 통한 채널별 맞춤형 임베딩으로 가짜 상관관계 제거**, **(2) 마스킹 기법을 적용한 Causal Mixer로 시간적 의존성 보존**, **(3) 이상 증거 누적 방식을 통한 정교한 이상 구간 탐지**이다. 실험 결과, 6개 벤치마크 데이터셋에서 기존 SOTA 모델들을 뛰어넘는 성능을 보였으며, 특히 실시간 탐지 환경에서 매우 강력한 효율성과 정확도를 입증하였다. 이는 향후 실시간 산업 시스템의 모니터링 및 이상 탐지 연구에 중요한 기초가 될 것으로 보인다.
