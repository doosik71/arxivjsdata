# DDMT: Denoising Diffusion Mask Transformer Models for Multivariate Time Series Anomaly Detection

Yang Chaocheng(2023)

## 🧩 Problem to Solve

본 논문은 다변량 시계열 데이터(Multivariate Time Series)에서 이상치 탐지(Anomaly Detection)를 수행하는 문제를 다룬다. 다변량 시계열 이상 탐지는 금융 사기 탐지, 시스템 고장 진단, 상태 추정 등 다양한 산업 분야에서 매우 중요한 과제이다.

최근 재구성 기반(Reconstruction-based) 모델들이 우수한 성능을 보이고 있으나, 데이터의 규모와 차원이 급격히 증가함에 따라 두 가지 주요 문제점이 대두되었다. 첫째는 데이터에 포함된 노이즈 문제이며, 둘째는 **Weak Identity Mapping (WIM)** 현상이다. WIM이란 모델이 입력 데이터를 깊이 있게 학습하여 재구성하는 것이 아니라, 단순히 입력을 출력으로 복제하는 단순 매핑만을 학습하여 이상치까지 너무 완벽하게 재구성해버림으로써, 결과적으로 원본과 재구성 데이터 간의 차이가 사라져 이상치를 구분하지 못하게 되는 현상을 의미한다.

따라서 본 연구의 목표는 Denoising Diffusion Model과 Transformer를 결합하여 시계열의 복잡한 패턴을 학습하고, ADNM이라는 새로운 마스킹 메커니즘을 통해 WIM 문제를 해결함으로써 다변량 시계열 이상 탐지의 성능을 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 다음과 같다.

1. **Denoising Diffusion Model과 Transformer의 최초 결합**: 다변량 시계열 이상 탐지를 위해 Diffusion 모델의 내부 네트워크로 Transformer를 채택한 첫 번째 프레임워크인 DDMT를 제안하였다. 이를 통해 시계열 데이터의 전역적 의존성(Global Dependencies)과 장기 의존성(Long-term Relationships)을 효과적으로 캡처하고, 점진적인 노이즈 제거 과정을 통해 정상 데이터의 분포를 정밀하게 모델링한다.
2. **Adaptive Dynamic Neighbor Mask (ADNM) 제안**: 재구성 과정에서 입력과 출력 간의 정보 누설을 막기 위해, 재구성 오차와 피어슨 상관계수(Pearson Correlation Coefficient)를 기반으로 동적으로 마스크를 설정하는 ADNM 메커니즘을 도입하였다. 이는 모델이 자기 자신과 인접한 노드에 과도하게 의존하는 WIM 문제를 완화한다.
3. **SOTA 성능 입증**: 5개의 공개 다변량 시계열 데이터셋에 대해 광범위한 실험을 진행하여, 기존의 최신 베이스라인 모델들보다 우수한 F1-score를 달성하며 State-of-the-art 성능을 기록하였다.

## 📎 Related Works

### 1. 머신러닝 기반 방법론

- **선형 모델**: PCA 등이 있으며, 차원 축소를 통해 변동성을 추출하지만 가우시안 분포 가정 및 노이즈에 민감하다는 한계가 있다.
- **거리/밀도 기반 모델**: K-means, KNN, LOF, DBSCAN 등이 있다. 이들은 클러스터 밀도나 거리를 이용하지만, 시간적 상관관계를 캡처하지 못하며 이상치의 지속 시간이나 개수에 대한 사전 지식이 필요할 때가 많다.
- **분류기 기반 모델**: Bayesian Network, SVM 등이 있으며, 특히 SVM은 하이퍼플레인을 통해 이상치를 구분하지만 고차원 데이터에서 계산 비용이 매우 높다.

### 2. 딥러닝 기반 방법론

- **Autoencoder (AE) 및 VAE**: 재구성 오차를 이용해 이상치를 탐지한다. VAE는 잠재 공간을 연속적으로 가정하므로 실제 데이터의 이산적 특성을 반영하지 못해 생성된 샘플의 현실성이 떨어질 수 있다.
- **GAN**: 적대적 학습을 통해 정밀한 샘플을 생성하지만, 학습 과정이 불안정하고 Mode Collapse(모드 붕괴) 문제가 발생하기 쉽다.
- **Transformer**: Self-attention 메커니즘을 통해 장기 의존성을 잘 캡처하며 병렬 계산이 가능하지만, 시계열 재구성 시 자기 자신과 인접 노드 정보에 지나치게 의존하는 WIM 문제가 발생한다.
- **Denoising Diffusion Model**: 최근 이미지 생성 등에서 SOTA 성능을 보이며, GAN보다 학습이 안정적이고 VAE보다 샘플 품질이 높다. 기존에는 U-Net 구조를 주로 사용했으나, 이는 이미지의 국소적 특징 추출에는 유리하지만 시계열의 복잡한 시간적 의존성을 다루기에는 부족함이 있다.

## 🛠️ Methodology

### 1. 전체 파이프라인

DDMT의 전체 구조는 **ADNM $\rightarrow$ DDT $\rightarrow$ Anomaly Detection**의 세 단계로 구성된다.

### 2. Adaptive Dynamic Neighbor Mask (ADNM)

WIM 문제를 해결하기 위해 Transformer의 Self-attention 이전에 적용되는 모듈이다.

- **동작 원리**: 단순한 Autoencoder를 사용하여 1차 재구성을 수행한다. 이때 재구성 오차가 큰 노드와 그 주변 노드들을 마스킹 처리하여, 모델이 자기 자신과 인접한 정보만으로 재구성하는 '지름길(shortcut)'을 차단한다.
- **마스크 결정**: 각 노드 $i$에 대해 마스크 규모 $m_i$를 결정한다. 재구성 오차와 실제 이상치 비율을 고려하여 마스킹 대상을 선정하고, 해당 노드와 주변 노드 간의 피어슨 상관계수가 특정 임계값보다 높으면 마스크 윈도우를 확장한다.
- **효과**: 모델이 강제로 멀리 떨어진 노드들의 정보를 활용하게 함으로써, 정상 패턴에 대한 이해도를 높이고 일반화 능력을 향상시킨다.

### 3. Denoising Diffusion Transformer (DDT)

Diffusion 모델의 역과정(Reverse Process)을 학습하는 신경망으로 U-Net 대신 Transformer Encoder를 사용한다.

- **순방향 과정 (Forward Process)**: 데이터 $\mathbf{x}_0$에 가우시안 노이즈를 단계적으로 추가하여 완전한 노이즈 상태 $\mathbf{x}_T$로 만드는 마르코프 체인 과정이다.
  $$q(\mathbf{x}_1, \dots, \mathbf{x}_T | \mathbf{x}_0) = \prod_{t=1}^T q(\mathbf{x}_t | \mathbf{x}_{t-1})$$
  여기서 각 단계의 전이 확률은 다음과 같다.
  $$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t \mathbf{I})$$
  $\beta_t$는 $10^{-4}$에서 $0.02$까지 선형적으로 증가하는 분산 스케줄이다.

- **역방향 과정 (Reverse Process)**: 학습 가능한 파라미터 $\theta$를 통해 노이즈를 제거하며 원본 데이터를 복원하는 과정이다.
  $$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \tilde{\beta}_t \mathbf{I})$$

- **손실 함수 (Loss Function)**: 모델은 실제 추가된 노이즈 $\epsilon$과 모델이 예측한 노이즈 $\epsilon_\theta$ 사이의 평균 제곱 오차(MSE)를 최소화하도록 학습된다.
  $$L = \min_\theta \mathbb{E}_{\mathbf{x}_0 \sim \mathcal{D}, \epsilon \sim \mathcal{N}(0, \mathbf{I}), t \sim \mathcal{U}(1, T)} \|\epsilon - \epsilon_\theta(\sqrt{\alpha_t}\mathbf{x}_0 + \sqrt{1-\alpha_t}\epsilon, t)\|^2$$
  여기서 $\alpha_t = 1 - \beta_t$이다.

- **Transformer 구조**: Multi-head Self-attention, Feed-forward Network, Layer Normalization 및 Residual Connection을 포함하는 표준 Transformer Encoder 블록을 사용하여 시계열의 전역적/국소적 관계를 학습한다.

### 4. 이상 탐지 절차

최종적으로 DDT를 통해 생성된 재구성 시퀀스와 원본 시퀀스 사이의 차이를 계산하여 이상치 점수(Anomaly Score)를 부여하고, 임계값을 기준으로 이상 여부를 판별한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: SMD(서버), PSM(서버), MSL(우주선), SMAP(위성), SWaT(수처리 시설) 총 5개 데이터셋 사용.
- **평가 지표**: Precision, Recall, F1-score.
- **구현 상세**: 윈도우 크기 100, SGD 옵티마이저(LR=1e-4), Diffusion 단계 $S=500$, 마스크 규모 5.

### 2. 정량적 결과

- **비교 모델**: Isolation Forest, LOF, OC-SVM 등 ML 모델 5종 및 LSTM-VAE, OmniAnomaly, TranAD, MEGA 등 최신 DL 모델 10종 등 총 15개 모델과 비교하였다.
- **결과**: DDMT는 MSL, SMAP, SWaT, PSM 데이터셋에서 가장 높은 F1-score를 기록하였다. SMD 데이터셋에서는 MEGA 모델이 약간 더 높았으나, 전반적으로 DDMT가 가장 우수한 성능을 보였다.
- **통계적 유의성**: t-test 결과, LOF, DAGMM, THOC, LSTM-VAE, OmniAnomaly, USAD 모델과 비교했을 때 통계적으로 유의미한 성능 향상이 있음이 확인되었다 ($p < 0.05$).

### 3. 절제 실험 (Ablation Study)

- **W/O DDT**: Diffusion 과정을 제거했을 때 평균 F1-score가 $94.37\% \rightarrow 87.19\%$로 하락하여, DDT 모듈이 성능 향상에 핵심적임을 확인하였다.
- **W/O ADNM**: ADNM을 일반 Self-attention으로 대체했을 때 평균 F1-score가 $94.37\% \rightarrow 90.38\%$로 하락하여, WIM 문제를 억제하는 마스킹 기법의 중요성이 입증되었다.
- **vs Transformer**: 단순 Transformer 베이스라인($77.42\%$) 대비 DDMT($94.37\%$)는 매우 압도적인 성능 향상을 보였다.

### 4. 하이퍼파라미터 분석

- **윈도우 크기**: 윈도우 크기가 커질수록 모든 모델의 성능이 점진적으로 향상되는 경향을 보였다.
- **마스크 스케일**: 피어슨 상관계수 임계값이 0.7일 때까지는 성능이 향상되나, 그 이상으로 마스킹이 과도해지면 오히려 깊은 특징 학습을 방해하여 성능이 하락하였다.
- **Diffusion 단계 ($S$)**: $S$가 증가할수록 성능이 좋아지지만, $S=500$ 이후에는 성능 향상이 미미한 반면 계산 시간은 급격히 증가하였다.

## 🧠 Insights & Discussion

### 강점

본 논문은 생성 모델의 최신 트렌드인 Diffusion Model을 시계열 이상 탐지에 성공적으로 접목하였다. 특히 단순히 모델을 결합한 것에 그치지 않고, 시계열 재구성 모델의 고질적인 문제인 WIM(Weak Identity Mapping)을 해결하기 위해 ADNM이라는 동적 마스킹 전략을 제안한 점이 학술적으로 매우 가치 있다.

### 한계 및 미래 연구 방향

1. **사전 지식 부족**: 현재 모델은 데이터만으로 학습하는 end-to-end 방식이다. 실제 환경에서는 공휴일이나 특정 이벤트 같은 도메인 지식(Prior Knowledge)을 결합한다면 오탐지(False Positive)를 더 줄일 수 있을 것이다.
2. **계산 효율성**: Diffusion 모델의 특성상 반복적인 디노이징 과정이 필요하므로 계산 자원 소모가 크고 추론 시간이 길다. 실시간 탐지가 중요한 환경에서는 모델 경량화나 샘플링 단계 최적화가 필수적이다.

## 📌 TL;DR

본 논문은 다변량 시계열 이상 탐지를 위해 **Denoising Diffusion Model**과 **Transformer**를 결합한 **DDMT** 프레임워크를 제안한다. 특히 재구성 모델의 단순 복제 문제(WIM)를 해결하기 위해 **ADNM(Adaptive Dynamic Neighbor Mask)** 메커니즘을 도입하여, 모델이 인접 정보가 아닌 전역적 패턴을 학습하도록 강제하였다. 실험 결과, 5개의 벤치마크 데이터셋에서 기존 SOTA 모델들을 상회하는 성능을 달성하였으며, 이는 Diffusion 모델의 강력한 분포 모델링 능력과 Transformer의 의존성 캡처 능력이 시계열 이상 탐지에 매우 효과적임을 시사한다.
