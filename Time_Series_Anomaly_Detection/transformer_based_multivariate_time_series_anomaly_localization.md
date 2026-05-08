# Transformer-based Multivariate Time Series Anomaly Localization

Charalampos Shimillas, Kleanthis Malialis, Konstantinos Fokianos, Marios M. Polycarpou (2025)

## 🧩 Problem to Solve

본 논문은 다변량 시계열(Multivariate Time Series, MTS) 데이터에서 이상치 탐지(Anomaly Detection)를 넘어, 어떤 변수가 이상치 발생의 원인이 되었는지를 찾아내는 이상치 국소화(Anomaly Localization) 문제를 해결하고자 한다.

사이버 물리 시스템(Cyber-Physical Systems, CPS)과 사물인터넷(IoT)의 복잡성이 증가함에 따라, 수많은 센서에서 생성되는 고차원 MTS 데이터를 모니터링하는 것이 필수적이다. 기존의 연구들은 "이상치가 발생했는가"를 판별하는 탐지(Detection) 영역에서는 큰 발전을 이루었으나, "어떤 센서가 문제인가"를 식별하는 국소화(Localization) 영역은 상대적으로 덜 연구되었다. 시스템의 신뢰성과 안전성을 유지하기 위한 지능적 의사결정을 위해서는 정확한 원인 변수를 식별하는 국소화 기술이 매우 중요하다. 따라서 본 논문의 목표는 복잡한 시스템의 동적 특성을 학습하면서도, 높은 성능의 국소화를 수행할 수 있는 비지도 학습 기반의 Transformer 모델을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Transformer의 Self-attention 메커니즘이 학습하는 잠재 표현(Latent Representation)이 통계적 시공간 모델과 밀접한 관련이 있다는 통찰에서 출발한다.

1. **시공간적 관점의 분석**: Transformer의 인코딩 과정이 Space-Time Autoregressive (STAR) 모델과 유사하게 작동하며, 특정 변수의 이상치가 잠재 공간을 통해 상관관계가 없는 다른 변수의 재구성(Reconstruction) 결과에까지 영향을 미칠 수 있음을 이론적 및 실험적으로 분석하였다.
2. **STAS (Space-Time Anomaly Score) 제안**: STAR 모델의 특성과 LISA(Local Indicators of Spatial Association) 개념에서 영감을 받아, 변수 마스킹과 변수 간 상관관계를 결합한 새로운 국소화 지표인 STAS를 설계하였다.
3. **SFAS (Statistical Feature Anomaly Score) 도입**: STAS의 오탐(False Alarm)을 줄이기 위해, 이상치 전후의 통계적 특징 변화를 분석하는 SFAS를 보정 항목으로 추가하여 국소화 정밀도를 높였다.
4. **3단계 국소화 프레임워크**: 실제 의사결정 과정을 반영하여 타임스텝(Time-step), 윈도우(Window), 세그먼트(Segment) 기반의 3단계 국소화 접근 방식을 정의하고 검증하였다.

## 📎 Related Works

기존의 MTS 이상치 국소화 연구들은 주로 다음과 같은 접근 방식을 취해왔다.

- **재구성 기반 모델**: MSCRED, OmniAnomaly, InterFusion 등은 Autoencoder나 VAE, GRU 기반의 모델을 사용하여 재구성 오차(Reconstruction Error)의 크기를 통해 이상치 변수를 식별한다. 하지만 이러한 모델들은 고차원 데이터의 복잡한 동적 특성을 완전히 포착하는 데 한계가 있으며, 특히 InterFusion과 같은 모델은 계산 복잡도로 인해 개별 타임스텝이 아닌 세그먼트 단위의 해석만 가능하다는 단점이 있다.
- **해석 가능성 중심 모델**: ARCANA와 같은 최적화 기반 모델이나 SHAP와 같은 XAI 기법이 시도되었으나, 전자는 복잡한 동적 특성 포착 능력이 떨어지고, 후자는 계산 비용이 매우 높으며 MTS의 특성인 변수 간 의존성을 무시하는 가정을 사용하는 경우가 많다.
- **Transformer 기반 모델**: 최근 Transformer가 MTS 탐지 분야에서 우수한 성과를 보이고 있으나, 이를 국소화 작업에 정밀하게 적용한 연구는 여전히 부족한 상태이다.

본 논문은 Transformer의 강력한 표현 학습 능력과 통계적 시공간 분석 기법을 결합함으로써, 기존 블랙박스 모델의 한계를 극복하고 해석 가능성과 성능을 동시에 확보하고자 한다.

## 🛠️ Methodology

### 1. Representation Learning Module (RLM)

먼저 입력된 MTS 데이터를 Transformer 인코더를 통해 학습한다. 데이터는 길이 $T$의 비중첩 윈도우로 처리되며, $\mathcal{L}$개의 레이어를 거친다. 각 레이어는 Multi-head Attention (MHA)와 Feed-Forward Network로 구성되며, 최종 출력은 MLP를 통해 원래의 MTS로 재구성된다.

학습을 위한 전체 손실 함수 $\mathcal{L}_{\text{Total}}$은 다음과 같다.
$$\mathcal{L}_{\text{Total}} = \|\mathbf{X} - \mathbf{\hat{X}}\|_{F}^{2} - \lambda \|\mathbf{d}_{\text{div}}(\mathbf{P}, \mathbf{A})\|_{1}$$
여기서 첫 번째 항은 재구성 오차를 최소화하는 Frobenius norm이며, 두 번째 항은 self-attention 행렬 $\mathbf{A}$와 Laplace 커널 기반의 prior-attention 행렬 $\mathbf{P}$ 사이의 KL-divergence를 측정하는 Association Discrepancy이다. 이는 모델이 단순히 인접한 타임스텝에만 과도하게 집중(Overfitting)하는 것을 방지한다.

### 2. STAS (Space-Time Anomaly Score)

STAS는 특정 변수를 마스킹했을 때의 재구성 오차 변화를 통해 해당 변수의 이상 여부를 판단한다.

- **개별 기여도**: $i$번째 변수를 마스킹했을 때의 전체 재구성 오차 $\mathbf{E}^{(i)}$와 마스킹하지 않았을 때의 오차 $\mathbf{E}$의 차이의 제곱 $(\mathbf{E}^{(i)} - \mathbf{E})^2$을 계산한다.
- **상관 변수 기여도**: $i$번째 변수와 상관관계가 높은 다른 변수 $j$들의 기여도를 합산한다. 이때 상관계수 $|\rho_{ij}|$ (Spearman correlation)를 가중치로 사용한다.

최종 STAS 점수 $\text{AS}_{\text{STAS}_i}$는 다음과 같이 정의된다.
$$\text{AS}_{\text{STAS}_i} = \frac{(\mathbf{E}^{(i)} - \mathbf{E})^2 + \sum_{j \neq i} |\rho_{ij}| (\mathbf{E}^{(j)} - \mathbf{E})^2}{\sum_{n=1}^{d} (\mathbf{E}^{(n)} - \mathbf{E})^2}$$
이 수식은 이상치의 근원이 되는 변수뿐만 아니라, 그와 밀접하게 연결된 변수들에서도 높은 점수가 나오게 함으로써 시스템적인 이상 확산을 포착한다.

### 3. SFAS (Statistical Feature Anomaly Score)

STAS에서 발생할 수 있는 오탐을 보정하기 위해 통계적 특징을 분석한다. 이상치 발생 전($W_{\text{before}}$)과 발생 시점 주변($W_{\text{around}}$)의 윈도우에서 분산, 추세 강도, 선형성, 곡률, 계절성 등의 통계적 특징을 추출한다.

추출된 고차원 특징은 PCA를 통해 2차원으로 투영되어 $\mathbf{P}_{\text{before}}$와 $\mathbf{P}_{\text{around}}$가 된다. SFAS 점수는 이 두 벡터의 $L_1$ norm 차이로 계산된다.
$$\text{AS}_{\text{SFAS}_i} = \|\mathbf{P}_{i, \text{before}} - \mathbf{P}_{i, \text{around}}\|_{1}$$

### 4. 최종 국소화 결정 절차 (Algorithm 1)

1. 먼저 STAS를 통해 1차 국소화 맵 $\mathbf{C}_1$을 생성한다 (임계값 $\tilde{h}_1$ 적용).
2. STAS에서 정상으로 판별되었으나 SFAS 점수가 임계값 $\tilde{h}_2$를 초과하는 경우, 이를 이상치로 재분류하여 $\mathbf{C}_{\text{combined}}$를 업데이트한다.
3. SFAS로 인해 추가된 이상치 수만큼, STAS 점수가 가장 낮은(가장 덜 이상한) 변수들을 다시 정상으로 되돌림으로써 전체 이상치 규모를 유지하며 정밀도를 최적화한다.

## 📊 Results

### 실험 설정

- **데이터셋**: 합성 데이터셋(Waves), 실제 데이터셋(Server Machine Dataset - SMD, Application Server Dataset - ASD).
- **비교 대상**: DAEMON (Adversarial AE), OmniAnomaly (GRU-VAE), InterFusion (Hierarchical GRU-VAE).
- **지표**: Precision, Recall, F1-Score, AUC, 그리고 세그먼트 국소화 성능을 측정하는 Interpretation Score (IPS).

### 주요 결과

1. **타임스텝 기반 국소화 (Time-step-wise)**:
    - ASD 데이터셋에서 기존 방법들 대비 F1-score가 약 55% 향상되는 압도적인 성능을 보였다.
    - SMD 데이터셋에서도 약 9%의 성능 향상을 달성하였다.
2. **윈도우 기반 국소화 (Window-based)**:
    - 윈도우 크기가 커질수록 SFAS의 보정 효과가 증가하며 성능이 향상되는 경향을 보였다.
    - 최고 성능 기준 SMD 14%, ASD 58%, WVS 59%의 F1-score 향상을 기록하였다.
3. **세그먼트 기반 국소화 (Segment-based)**:
    - F1-score와 AUC에서 22%~35%의 향상을 보였으며, 특히 IPS 지표에서 9%~45%의 개선을 이루어 실제 원인 변수 식별 능력이 매우 뛰어남을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 이론적 배경

본 연구는 단순한 모델 제안을 넘어, Transformer의 Attention mechanism이 내부적으로 어떻게 MTS의 시공간적 관계를 학습하는지를 수식적으로 증명하였다. 특히, 한 변수의 이상치가 잠재 공간(Latent Space)에서 다른 변수의 표현에 영향을 주는 '이상치 전이' 현상을 밝혀냈으며, 이를 STAS라는 지표를 통해 역으로 이용하여 국소화 성능을 높인 점이 매우 독창적이다.

### 한계 및 비판적 해석

1. **상관관계의 정적 가정**: STAS에서 사용된 상관계수 $\rho_{ij}$가 시간에 따라 일정하다고 가정한다. 그러나 실제 CPS 환경에서는 시스템 상태에 따라 변수 간의 상관관계가 동적으로 변할 수 있으므로, 정적 상관계수 사용은 장기적인 모니터링에서 한계가 있을 수 있다.
2. **윈도우 크기 의존성**: SFAS의 성능이 윈도우 크기에 의존적이라는 결과가 나왔다. 최적의 윈도우 크기를 결정하는 명확한 기준이 제시되지 않았으며, 이는 사용자 환경마다 수동 튜닝이 필요함을 시사한다.
3. **계산 복잡도**: 모든 변수를 순차적으로 마스킹하여 RLM을 통과시키는 과정은 변수의 개수 $d$가 매우 많은 대규모 시스템에서 상당한 계산 오버헤드를 발생시킬 가능성이 크다.

## 📌 TL;DR

본 논문은 Transformer의 잠재 표현이 시공간 통계 모델(STAR)과 유사하게 동작한다는 통찰을 바탕으로, MTS 이상치 국소화를 위한 **STAS(시공간 이상 점수)**와 **SFAS(통계적 특징 이상 점수)** 기반의 프레임워크를 제안한다. 제안 방법은 변수 마스킹과 상관관계 분석을 통해 이상치의 근원 변수를 정확히 찾아내며, 특히 기존 SOTA 모델들보다 타임스텝, 윈도우, 세그먼트 모든 수준의 국소화 작업에서 월등한 성능 향상을 보였다. 이 연구는 향후 고차원 센서 데이터의 실시간 진단 및 유지보수 시스템의 핵심 기술로 적용될 가능성이 높다.
