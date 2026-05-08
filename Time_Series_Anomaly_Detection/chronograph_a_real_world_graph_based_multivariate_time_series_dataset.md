# CHRONOGRAPH: A Real-World Graph-Based Multivariate Time Series Dataset

Adrian Catalin Lutu, Ioana Pintilie, Elena Burceanu, Andrei Manolache (2025)

## 🧩 Problem to Solve

본 논문은 대규모 자동화 시스템, 특히 마이크로서비스 아키텍처(Microservice Architecture) 환경에서 시스템 성능 지표의 단기 및 중기 변화를 예측하고 이상 징후를 탐지하는 문제에 집중한다. 마이크로서비스 환경에서는 수백 개의 서비스가 복잡한 의존성 그래프(Dependency Graph)를 형성하며 연결되어 있으며, 특정 서비스에서 발생한 장애나 리소스 경합 등의 문제가 호출 체인을 통해 다른 서비스로 전파되는 특성이 있다. 따라서 정확한 예측을 위해서는 개별 서비스의 시간적 역학(Temporal Dynamics)뿐만 아니라 서비스 간의 상호 영향력을 함께 고려해야 한다.

그러나 기존의 그래프 기반 시계열 벤치마크 데이터셋들은 다음과 같은 한계를 가지고 있다. 교통량이나 공기 질 데이터셋은 단변량(Univariate) 데이터이며 장애 주석(Incident Annotation)이 부족하고, SWaT나 WADI와 같은 산업 제어 시스템 데이터셋은 다변량(Multivariate) 데이터와 이상치 레이블을 제공하지만, 실제 인접 행렬(Adjacency Matrix) 형태의 명시적인 그래프 구조를 제공하지 않고 단순한 프로세스 다이어그램만을 제공한다. 결과적으로 다변량 시계열, 명시적인 의존성 그래프, 그리고 실제 장애 레이블이 모두 결합된 벤치마크의 부재로 인해, 현재의 예측 및 이상 탐지 모델들은 대부분 그래프 구조를 활용하지 못하는 Topology-agnostic 방식으로 개발되고 있다.

## ✨ Key Contributions

본 논문의 핵심 기여는 실제 운영 환경의 마이크로서비스 텔레메트리 데이터를 기반으로 구축된 **CHRONOGRAPH** 데이터셋을 제안하는 것이다. 이 데이터셋의 중심 설계 아이디어는 다음과 같다.

1. **다변량 시계열과 명시적 그래프 구조의 결합**: 각 노드는 마이크로서비스를 나타내며 5차원의 시스템 성능 지표를 가지며, 에지는 서비스 간의 호출 의존성을 나타내는 3차원의 지표를 가진다. 이를 통해 모델이 시간적 변화와 구조적 상관관계를 동시에 학습할 수 있는 환경을 제공한다.
2. **전문가 주석 기반의 장애 레이블 제공**: 내부 장애 보고서를 통해 실제 발생한 사고 구간을 레이블링하여 제공함으로써, 단순한 통계적 이상치 탐지를 넘어 실제 운영상의 중단(Operational Disruptions)에 대한 예측 강건성과 탐지 성능을 평가할 수 있게 한다.
3. **현행 모델의 한계 분석**: 최신 시계열 파운데이션 모델(Foundation Models)을 포함한 다양한 베이스라인 모델을 벤치마킹하여, 현재의 Topology-agnostic 방식이 장기 예측과 구조적 전파 효과 포착에 한계가 있음을 정량적으로 입증하였다.

## 📎 Related Works

논문에서는 기존의 시계열 예측 및 이상 탐지 접근 방식과 데이터셋의 한계를 설명한다.

- **기존 데이터셋의 한계**: 앞서 언급한 바와 같이, 기존 데이터셋들은 단변량 데이터이거나($\text{Traffic, Air-quality}$), 그래프 구조가 명시적이지 않아($\text{SWaT, WADI}$) 실제 마이크로서비스의 복잡한 의존성 전파를 연구하기에 부적합하다.
- **Topology-agnostic 모델**: 대부분의 고전적/신경망 기반 예측 모델과 시계열 파운데이션 모델(예: $\text{Chronos, TabPFN-TS}$)은 각 시계열을 독립적으로 처리하거나 구조적 맥락 없이 특징을 집계한다.
- **기존 Graph-aware 방법론의 한계**: 일부 연구에서는 그래프 구조를 활용하려 하지만, 모든 변수가 연결되었다고 가정하는 완전 그래프(Complete Graph)를 사용하거나, $\text{Gumbel-Softmax}$ 샘플링 등을 통해 잠재적인 그래프 구조(Latent Graph)를 학습하는 방식을 취한다. 이러한 데이터 기반의 유도된 구조는 시스템의 실제 물리적/논리적 의존성 토폴로지와 일치하지 않을 가능성이 크다.

## 🛠️ Methodology

### 데이터셋 구성 (CHRONOGRAPH)

CHRONOGRAPH는 대규모 기업의 실제 운영 마이크로서비스 플랫폼에서 6개월간 수집된 텔레메트리 데이터로 구성된다.

- **노드(Node)**: 총 708개의 서비스가 노드로 정의된다. 각 노드는 30분 간격으로 수집된 8,005개의 타임스텝을 가지며, 다음과 같은 5차원 시스템 레벨 지표를 포함한다: $\text{CPU usage, Memory usage, Memory working set, Network traffic rate (Incoming), Network traffic rate (Outgoing)}$.
- **에지(Edge)**: 서비스 간의 통신을 기반으로 하는 방향성 그래프(Directed Graph)이다. 에지는 다음과 같은 3차원의 통신 지표를 가진다: $\text{Number of requests, Return codes, Latency}$.
- **레이블(Label)**: 내부 장애 보고서를 파싱하여 추출한 17개의 전문가 주석 기반 이상 구간(Anomaly segments)을 제공한다.

### 실험 파이프라인 및 모델

연구진은 예측(Forecasting)과 이상 탐지(Anomaly Detection) 두 가지 작업을 수행하였다.

1. **예측 모델 (Forecasting Models)**:
    - $\text{Prophet}$: 추세, 계절성, 휴일 효과를 분해하여 모델링하는 통계적 모델이다.
    - $\text{Chronos}$: 시계열 데이터를 언어처럼 학습한 파운데이션 모델로, 64개 타임스텝씩 예측하고 이를 다시 컨텍스트에 추가하는 Rolling prediction 전략을 사용한다.
    - $\text{TabPFN-TS}$: 트랜스포머 기반의 Prior-data-fitted 네트워크를 시계열에 적응시킨 모델로, 다변량 시계열 전체를 공동으로 학습하여 시리즈 간 의존성을 모델링한다.

2. **이상 탐지 모델 (Anomaly Detection Models)**:
    - **잔차 기반 탐지**: 위 예측 모델들의 예측값과 실제값의 차이(Residuals)를 계산하고 $\text{z-score}$를 통해 이상치를 판별한다.
    - $\text{Autoencoder}$: 재구성 오차(Reconstruction error)를 이용해 이상을 탐지한다.
    - $\text{Isolation Forest}$ 및 $\text{OC-SVM}$: 고차원 데이터의 아웃라이어를 탐지하는 전통적인 머신러닝 기법이다.
    - $\text{Ensemble}$: $\text{Prophet, Isolation Forest, Autoencoder}$를 결합하여 탐지 성능을 보완한다.

### 평가지표

- **예측 성능**: $\text{MAE (Mean Absolute Error), MSE (Mean Squared Error), MASE (Mean Absolute Scaled Error)}$를 사용한다.
- **이상 탐지 성능**: 단순 $\text{F1-score}$는 세그먼트 내의 부분적 탐지에 너무 관대하거나 엄격할 수 있으므로, 다양한 $K$ 비율(정확히 예측된 포인트의 비율)을 통합하여 평가하는 $F1_K\text{-AUC}$와 $ROC_K\text{-AUC}$를 주 지표로 사용한다.

## 📊 Results

### 예측 성능 (Forecasting)

실험 결과, 모든 모델이 단기 예측(500 steps)보다 장기 예측(3202 steps)에서 성능이 현저히 저하되는 양상을 보였다.

- **단기 예측**: $\text{Chronos}$가 가장 낮은 오차를 기록하며 우수한 성능을 보였다.
- **장기 예측**: $\text{Chronos}$의 성능 하락폭이 가장 컸으며, $\text{TabPFN-TS}$와 $\text{Prophet}$이 상대적으로 더 안정적인 성능을 유지하였다. 이는 현재의 모델들이 복잡한 장기 시간적 역학을 모델링하는 데 한계가 있음을 시사한다.

### 이상 탐지 성능 (Anomaly Detection)

전반적인 탐지 성능은 낮게 나타났으며, 모델별 특징은 다음과 같다.

- $\text{Prophet}$은 개별 모델 중 가장 높은 성능을 보였으나, 미탐지율(False Negative Rate)이 매우 높았다.
- $\text{Isolation Forest}$와 $\text{OC-SVM}$은 더 많은 이상을 탐지하지만, 오탐지율(False Positive Rate)이 매우 높았다.
- **Ensemble 모델**은 $\text{Prophet, Isolation Forest, Autoencoder}$를 결합했을 때 가장 균형 잡힌 성능을 보였으나, 여전히 실제 장애 구간을 완벽하게 식별하는 데는 어려움이 있었다.

## 🧠 Insights & Discussion

본 연구는 실험 결과를 통해 다음과 같은 중요한 통찰을 제시한다.

첫째, **장기 예측의 불안정성**이다. 정성적 분석 결과, 시계열의 변동성이 커지거나 실제 장애가 발생했을 때 예측값이 급격히 발산하는 경향이 확인되었다. 이는 단순한 시계열 패턴 학습만으로는 실제 운영 환경의 동적인 변화를 따라잡기 어렵다는 것을 의미한다.

둘째, **구조적 전파 효과(Propagation Effects)**이다. 이상 탐지 결과, 예측된 이상 징후들이 그래프 상에서 서로 연결된 서비스들에 군집(Cluster)을 이루며 나타나는 경향이 발견되었다. 이는 시스템의 장애가 토폴로지를 따라 전파된다는 가설을 뒷받침하며, 단순히 개별 시계열을 분석하는 $\text{Topology-agnostic}$ 방식이 아닌, 그래프 구조를 명시적으로 통합한 $\text{Structure-aware}$ 모델의 필요성을 강력하게 시사한다.

셋째, **데이터셋의 실용적 가치**이다. 레이블링되지 않은 이상 징후(False Positives로 집계된 것들) 중 일부는 실제로는 발생했지만 보고되지 않은 일시적 장애일 가능성이 크다. 이러한 패턴을 찾아내는 것은 서비스에 치명적인 영향을 주기 전 예방 정비를 가능하게 하므로 운영 관점에서 매우 가치 있는 작업이 될 수 있다.

## 📌 TL;DR

본 논문은 실제 기업의 마이크로서비스 환경에서 수집한 다변량 시계열, 명시적인 서비스 의존성 그래프, 그리고 실제 장애 레이블이 결합된 **CHRONOGRAPH** 데이터셋을 제안한다. 다양한 최신 예측 및 탐지 모델을 벤치마킹한 결과, 현재의 모델들은 그래프 구조를 활용하지 못해 장기 예측 성능이 낮고 구조적 장애 전파를 포착하지 못한다는 한계를 드러냈다. 이 연구는 향후 시계열 분석 연구가 단순한 시간적 패턴 학습을 넘어, 시스템의 물리적/논리적 구조(Topology)를 통합하는 방향으로 나아가야 함을 제시하며, 이를 위한 표준 벤치마크를 제공했다는 점에서 중요한 의미를 갖는다.
