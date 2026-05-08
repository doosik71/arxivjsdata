# Understanding Time Series Anomaly State Detection through One-Class Classification

Hanxu Zhou, Yuan Zhang, Guangjie Leng, Ruofan Wang, Zhi-Qin John Xu (2024)

## 🧩 Problem to Solve

본 논문은 기존의 시계열 이상치 탐지(Time Series Anomaly Detection, TSAD) 연구가 주로 주어진 시계열 내에서 개별적인 이상치(Outliers)를 찾는 것에 집중해 왔다는 점을 지적한다. 하지만 실제 산업 현장에서는 표준 시계열(Standard Time Series)이 주어졌을 때, 새로운 테스트 시계열이 이 표준에서 벗어났는지를 판단하는 문제가 더 중요하다. 이는 단순히 데이터셋 내의 희귀한 점을 찾는 것이 아니라, 전체적인 상태가 정상 범주에 속하는지를 판별하는 One-Class Classification(OCC) 문제에 더 가깝다.

따라서 본 연구의 목표는 '시계열 이상 상태 탐지(Time Series Anomaly State Detection)'라는 개념을 정의하고, 이를 수학적으로 정립하며, 다양한 알고리즘의 성능을 공정하게 비교 평가할 수 있는 프레임워크와 데이터셋을 제공하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 시계열 이상 탐지를 '상태 탐지' 관점에서 재정의하고 이를 검증하기 위한 체계적인 방법론을 제시한 것이다. 주요 기여 사항은 다음과 같다.

- **개념적 프레임워크 도입**: Stochastic Process(확률 과정)와 가설 검정(Hypothesis Testing) 도구를 사용하여 '시계열 이상 상태 탐지 문제'와 그에 따른 이상 상태를 수학적으로 엄격하게 정의하였다.
- **합성 데이터셋 구축**: 전통적인 이상치 탐지 데이터셋은 본 연구가 정의한 문제 해결에 적합하지 않으므로, 시계열 분류(Time Series Classification) 데이터셋을 활용하여 표준 시계열과 이상 상태가 포함된 테스트 시계열을 생성하는 데이터 구축 방법을 제안하였다.
- **광범위한 실증 평가 및 비교 분석**: 총 38가지의 이상 탐지 알고리즘을 본 문제에 맞게 수정하여 적용하고, 정량적 지표와 시간 효율성, 안정성 등을 종합적으로 비교 분석하여 향후 연구 방향을 위한 통찰을 제공하였다.

## 📎 Related Works

논문에서는 시계열 이상 탐지 알고리즘을 작동 메커니즘에 따라 다음과 같이 네 가지 범주로 분류한다.

1. **Forecast-based**: 미래 값을 예측하고 실제 값과의 차이를 통해 이상치를 판별한다. (예: Linear Regression, FCNN, CNN, RNN, GRU, LSTM)
2. **Reconstitution-based**: 데이터의 잠재 구조를 학습하여 재구성하고, 재구성 오차가 큰 데이터를 이상치로 간주한다. (예: KPCA, Auto-Encoder, VAE, Transformer, TadGan)
3. **Statistical-model-based**: 데이터가 특정 통계 분포를 따른다고 가정하고 가설 검정을 수행한다. (예: COPOD, ECOD, GMM, KDE, MAD)
4. **Proximity-based**: 거리, 밀도, 각도 등의 유사도 측정 방식을 사용하여 주변 데이터와 거리가 먼 데이터를 이상치로 판별한다. (예: LOF, KNN, IForest, OCSVM, DeepSVDD)

기존 연구들은 대부분 데이터셋 내에서 일반적인 분포를 벗어나는 개별 관측치(Outlier)를 찾는 'Outlier Detection'에 집중되어 있었으며, 본 논문이 제시하는 '표준 시계열 대비 상태 변화'를 탐지하는 문제에 대한 연구는 상대적으로 부족했다는 점이 차별점이다.

## 🛠️ Methodology

### 1. 수학적 정의

본 논문은 시계열을 이산 시간 지점에서의 Stochastic Process $\{X(t), t \in \mathbb{N}\}$로 정의한다. '시계열 이상 상태 탐지' 문제는 다음과 같이 정의된다.

- **입력**: 표준 확률 과정 $\{X^0(t), t \in \mathbb{N}\}$에서 추출된 표준 시계열 데이터셋 $D$와, 보지 못한 테스트 시계열 데이터셋 $U$ 및 그 세그먼트 부분 집합 $U^S$가 주어진다.
- **목표**: 테스트 세그먼트 $S^k_j$가 표준 확률 과정 $\{X^0(t), t \in \mathbb{N}\}$로부터 생성되지 않았음을 식별하는 것이다. 즉, $S^k_j \not\sim \{X^0(t), t \in \mathbb{N}\}$인 이상 세그먼트 집합 $O$를 찾는 문제이다.

### 2. 시스템 파이프라인

전체 프로세스는 전처리, 학습, 추론의 단계로 구성된다.

1. **전처리 ($P$)**: 표준 데이터 $D$와 테스트 세그먼트 $U^S$에서 시간 정보 특징을 추출하여 $F_{\text{standard}}$와 $F_{\text{test}}$를 생성한다.
2. **탐지 알고리즘 ($M$)**: $F_{\text{standard}}$를 통해 이상치 점수 함수 $A_M(\cdot)$를 학습한다.
3. **판별**: 테스트 세그먼트의 점수 $A_M(P(S^k_j))$가 특정 임계값 $L_M$보다 크면 이상 상태로 판별한다.

### 3. 결정 함수 (Decision Function)

본 논문은 검증 데이터의 변형된 이상치 점수 $\hat{A}_v$의 평균 $\mu$와 분산 $\sigma$를 이용하여 다음과 같은 결정 함수 $D$를 정의한다.

$$D(\hat{a}) = \max\left(\left(\text{erf}\left(\frac{\hat{a} - \mu - \sigma}{\sqrt{2}\sigma}\right) - 0.5\right) \times 2, 0\right)$$

여기서 $\text{erf}$는 오차 함수(Error Function)이며, 임계값 $T$는 다음과 같이 설정된다.
$$T = 1 - \frac{1 - \text{erf}(2/\sqrt{2})}{2}$$
이는 $\hat{a}$가 $\mu$로부터 일정 거리($2\sigma$) 이상 떨어져 있을 때만 이상치로 간주하겠다는 통계적 유의 수준을 계산하는 방식이다.

### 4. 데이터셋 구축 방법

전통적인 이상치 탐지 데이터셋을 사용할 수 없으므로, 시계열 분류 데이터셋을 이용해 다음과 같이 합성 데이터를 생성한다.

- 특정 카테고리의 데이터를 선택하고 k-means 클러스터링을 통해 일관된 베이스라인을 확보한다.
- 데이터의 절반을 표준 시계열($X_{\text{standard}}$)로 사용한다.
- 나머지 절반에는 다른 카테고리의 데이터를 무작위로 삽입하여 테스트 시계열($X_{\text{test}}$)을 구성함으로써 '상태'의 변화를 유도한다.

## 📊 Results

### 1. 실험 설정

- **대상**: 38가지의 이상 탐지 알고리즘.
- **측정 지표**: Precision, Recall, F1-score, Range-F1 및 임계값 독립 지표인 AUC-ROC, AUC-PR, VUS-ROC, VUS-PR을 사용하였다.
- **데이터 난이도**: K-Nearest-Neighbors Normalized Clusteredness (KNC) 지표를 제안하여 데이터셋의 난이도를 측정하였다.

$$KNC = \frac{\mathbb{E}_{s \in S_{\text{ano}}} [D^{\text{kmean}}(s, S_{\text{std}})]}{\mathbb{E}_{s \in S_{\text{nor}}} [D^{\text{kmean}}(s, S_{\text{std}})]}$$

### 2. 주요 결과

- **최상위 알고리즘**: 실험 결과 **Sampling, LOF, KNN** 알고리즘이 가장 우수한 성능을 보였다.
- **범주별 비교**: Proximity-based 알고리즘이 전반적으로 가장 높은 성능을 기록했으며, 그 뒤를 Reconstitution-based, Forecast-based, Statistical-model-based 순으로 이었다.
- **딥러닝 vs 전통적 방식**: 흥미롭게도 신경망 기반 모델보다 전통적인 거리 기반 알고리즘(Distance-based)의 성능이 더 높게 나타났다. 예측 기반 신경망 중에서는 복잡한 구조보다 단순한 FCNN이 가장 좋은 성능을 보였다.
- **시간 효율성**: AUC-ROC가 0.95 이상인 알고리즘 중에서는 LOF가 가장 빠른 처리 속도를 보였다.
- **강건성(Robustness)**: KNC 값이 낮아질수록(난이도가 높아질수록) 모든 알고리즘의 성능이 하락했으나, LOF와 SOS가 가장 적은 성능 하락폭을 보이며 강건함을 입증하였다.

## 🧠 Insights & Discussion

본 연구는 시계열 이상 탐지를 '개별 점의 이상'이 아닌 '상태의 이상'으로 접근했을 때, 기존의 딥러닝 기반 모델들이 반드시 최선의 선택은 아니라는 점을 시사한다. 특히 Proximity-based 알고리즘들이 높은 성능을 보인 이유는, 표준 시계열의 특징 공간 내에서 테스트 세그먼트의 거리를 측정하는 방식이 OCC(One-Class Classification) 문제의 본질과 잘 부합하기 때문으로 해석된다.

다만, 본 연구에서 사용한 데이터셋이 시계열 분류 데이터를 기반으로 생성된 합성 데이터셋이라는 점은 한계로 작용할 수 있다. 실제 산업 현장의 데이터는 더 복잡한 노이즈와 동적인 변화를 포함하고 있으므로, 제안된 프레임워크를 실제 데이터에 적용하여 검증하는 후속 연구가 필요하다.

## 📌 TL;DR

본 논문은 시계열 이상 탐지를 단순한 Outlier Detection이 아닌 **One-Class Classification 관점의 '상태 탐지(State Detection)' 문제로 재정의**하였다. 이를 위해 수학적 정의와 합성 데이터셋 구축 방법론을 제시하였으며, 38종의 알고리즘을 벤치마킹한 결과 **Proximity-based 알고리즘(특히 Sampling, LOF, KNN)이 딥러닝 모델보다 우수한 성능과 강건성**을 보임을 확인하였다. 이 연구는 실무적인 시계열 상태 모니터링 시스템 구축 시 단순한 거리 기반 모델이 효율적인 대안이 될 수 있음을 시사한다.
