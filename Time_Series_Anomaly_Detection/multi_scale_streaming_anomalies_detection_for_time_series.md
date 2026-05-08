# Multi-scale streaming anomalies detection for time series

B Ravi Kiran (2024)

## 🧩 Problem to Solve

본 논문은 단변량 시계열(univariate time series) 데이터에서 발생하는 스트리밍 이상치 탐지(streaming anomaly detection) 문제를 다룬다. 시계열 데이터에서 이상치는 특정 모델에서 크게 벗어난 지점으로 정의되며, 이는 산업 공정 제어나 생체 의료 응용 분야에서 매우 중요한 문제이다.

기존의 스트리밍 이상치 탐지 알고리즘들은 통계량을 계산하기 위한 슬라이딩 윈도우(sliding window)의 크기를 설정해야 하는데, 이 윈도우 크기는 성능에 결정적인 영향을 미치는 중요한 파라미터이다. 그러나 실제 시계열 데이터에서는 의사 주기성(pseudo-periodicity)의 척도(scale)가 가변적으로 변하는 특성이 있어, 고정된 단일 윈도우 크기만으로는 다양한 스케일에서 발생하는 이상 패턴을 효과적으로 포착하기 어렵다. 따라서 본 논문의 목표는 다양한 시간 척도를 동시에 고려할 수 있는 **Multi-scale streaming anomaly detector**를 설계하고 그 효용성을 검증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 다양한 크기의 윈도우를 가진 **Multi-scale lag-matrix**를 구성하고, 이에 대해 **Streaming PCA**를 적용하여 각 스케일별 이상치 점수를 산출하는 것이다. 주요 기여 사항은 다음과 같다.

1. **Multi-scale Streaming PCA 제안**: 여러 개의 기하학적 윈도우 크기($p_j = 2^j$)를 동시에 추적하여 다양한 시간 척도에서의 이상치를 탐지한다.
2. **Hierarchical Streaming PCA 도입**: Multi-scale 접근 방식의 높은 시간 복잡도 문제를 해결하기 위해, 이전 스케일의 투영 결과를 다음 스케일의 입력으로 사용하는 계층적 구조를 제안하여 연산 효율성을 높였다.
3. **Haar Wavelet Transform 활용**: Lag-matrix를 Haar 기저(Haar basis)로 변환하여 표현함으로써, 이상치 발생 시 재구성 오차(reconstruction error)의 스파이크를 더욱 명확하게 만들어 탐지 성능을 향상시켰다.
4. **다양한 점수 집계(Aggregation) 방법 제시**: 각 스케일에서 도출된 여러 개의 이상치 점수를 하나의 최종 점수로 통합하기 위한 세 가지 방법(Norm, 2nd iteration PCA, Least correlated scale)을 제안하고 비교 분석하였다.

## 📎 Related Works

논문에서는 다음과 같은 기존 연구들을 언급하며 차별점을 제시한다.

- **기존 탐지 방법**: $k$-최근접 이웃($k$-nearest neighbour)을 통한 밀도 측정, 자기회귀 모델(auto-regressive models)의 예측 오차 활용, 그리고 주 공간 추적(principal subspace tracking)을 통한 재구성 오차 활용 방식이 있다.
- **Streaming Anomaly Detection**: 데이터가 실시간으로 들어오는 상황에서 파라미터를 지속적으로 업데이트해야 하며, 저장 공간의 제한으로 인해 모델의 전체 저장보다는 실시간 업데이트가 중요하다.
- **차별점**:
  - [PY06]은 Multi-scale streaming PCA를 사용해 가장 대표적인 윈도우를 추출했으나, 이상치 탐지에 집중하지 않았다.
  - [CZ08]은 Multi-scale 이상치 탐지를 수행했으나, 오프라인(off-line) 방식으로 동작하며 실시간 스트리밍 환경을 고려하지 않았다.
  - 본 논문은 실시간 스트리밍 환경에서 다중 척도를 체계적으로 연구한 첫 번째 시도임을 강조한다.

## 🛠️ Methodology

### 1. Lag-matrix 구성 및 Streaming PCA

단변량 시계열 $x(t)$가 주어졌을 때, 윈도우 크기 $p$에 대해 현재 시점 $t$에서의 $p$-차원 샘플 $X^p_t$를 다음과 같이 정의한다.
$$X^p_t = [x_t, x_{t-1}, \dots, x_{t-p+1}]^T \in \mathbb{R}^p$$
이 벡터들을 쌓아 만든 행렬을 Lag-matrix라고 한다. 본 연구에서는 이 Lag-matrix의 첫 번째 주성분 방향(first principal direction) $w^p$를 실시간으로 추적한다. $w^p$는 데이터의 에너지를 가장 많이 포착하는 1차원 투영 방향으로 정의된다.
$$w^p = \arg \min_{\|w\|=1} \sum_{t=1}^{T} \|X^p_t - (ww^T)X^p_t\|^2$$
시점 $t$에서 $X^p_t$를 $w^p$에 투영한 값을 $\tilde{X}^p_t = w^T_p X^p_t$라고 할 때, 해당 스케일 $p$에서의 이상치 점수 $\alpha^p_t$는 **재구성 오차(reconstruction error)**로 계산된다.
$$\alpha^p_t = \|\tilde{X}^p_t - X^p_t\|^2$$

### 2. Multi-scale 및 Hierarchical 접근법

- **Multi-scale Streaming PCA**: $p_j = 2^j$ ($j=1, \dots, J$)와 같이 기하급수적으로 증가하는 윈도우 크기들에 대해 각각 PCA를 수행한다. 하지만 이 방식은 시간 복잡도가 $O(TP)$로 매우 높다.
- **Hierarchical Streaming PCA**: 연산량을 줄이기 위해 계층적 구조를 사용한다. 스케일 $p_j$에서 $p_{j+1}$로 넘어갈 때, 전체 Lag-matrix를 다시 만드는 대신 이전 스케일의 주성분 방향으로 투영된 값들을 사용하여 축소된 Lag-matrix $Z^{j+1}_t$를 구성한다.
$$Z^{j+1}_t = [w^T_j Z^j_t, w^T_j Z^j_{t-2^j}]^T$$
이 방식을 통해 시간 복잡도를 $O(T \log P)$까지 낮출 수 있다.

### 3. Haar Transform의 적용

단순한 Lag-matrix 대신, 유니타리 Haar 기저 $H$를 사용하여 기저 변환을 수행한 $\text{H}^T X^p_t$를 사용한다. Haar 변환은 신호를 다중 척도로 근사하는 특성이 있으며, 실험적으로 이상치 발생 시 재구성 오차의 피크(peak)를 더 날카롭게 만들어 탐지력을 높이는 효과가 확인되었다.

### 4. 이상치 점수 집계 방법 (Aggregation Methods)

각 스케일 $j$에서 얻은 점수 집합 $\vec{\alpha}_t = \{\alpha^j_t\}_{j \le J}$를 최종 점수로 통합하는 세 가지 방법을 제안한다.

1. **Norm**: 모든 스케일 점수의 L2 노름을 계산한다. $\|\vec{\alpha}_t\|^2$
2. **2nd Iteration PCA**: $\vec{\alpha}_t$ 자체를 다시 Streaming PCA의 입력으로 넣어, 그 재구성 오차를 최종 점수로 사용한다. $\|\tilde{\vec{\alpha}}_t - \vec{\alpha}_t\|^2$
3. **Least Correlated Scale**: 다른 스케일들과 상관관계가 가장 낮은 스케일의 점수를 선택한다. $\alpha^{j^*}_t$ (단, $j^* = \arg \min_j \sum_i \text{corr}(\alpha^j, \alpha^i)$)

## 📊 Results

### 실험 설정

- **데이터셋**: Yahoo! 시계열 데이터셋(Benchmark 1~4) 및 Numenta Anomaly Benchmark(NAB).
- **평가 지표**: AUC (Area Under the ROC Curve). 0.5는 무작위 추측, 1.0은 완벽한 탐지를 의미한다.
- **비교 대상**: 단일 척도(fixed-scale) 방법 vs 다중 척도(multi-scale) 방법.

### 주요 결과

- **Multi-scale의 우월성**: 모든 벤치마크에서 다중 척도 접근 방식이 단일 척도 방식보다 높은 AUC를 기록하였다. 이는 다양한 시간 주기에서 발생하는 이상치를 포착하는 것이 필수적임을 보여준다.
- **집계 방법의 영향**: 단순한 Norm 방식보다 **Least Correlated Scale** 방식이나 **2nd Iteration PCA** 방식이 전반적으로 더 좋은 성능을 보였다. 특히 스케일 간의 디커릴레이션(de-correlation)을 이용하는 것이 강건한 탐지에 효과적이었다.
- **Haar 변환 효과**: Haar 변환을 적용한 경우 이상치 지점에서 오차의 피크가 더 뚜렷하게 나타나 탐지 성능이 향상되는 경향을 보였다.
- **주성분 개수**: 주성분 개수를 1개($PC=1$)에서 2개($PC=2$)로 늘려도 탐지 성능의 유의미한 향상은 관찰되지 않았다.

## 🧠 Insights & Discussion

### 강점 및 한계

- **강점**: 실시간 스트리밍 환경에서 시간 복잡도를 효율적으로 관리하면서($O(T \log P)$), 다양한 시간 척도의 이상치를 동시에 탐지할 수 있는 프레임워크를 구축하였다.
- **한계 및 실패 사례**: 값이 급격히 떨어지는(drop) 형태의 이상치의 경우, 재구성 오차가 오히려 감소하거나 변화가 적어 탐지하지 못하는 경우가 발생하였다. 이러한 사례는 이동 윈도우의 최솟값(minimum) 같은 비선형 함수를 사용해야 해결 가능하다.

### 비판적 해석

본 논문은 Streaming PCA의 적응성(adaptation) 문제를 언급한다. 새로운 데이터가 들어올 때마다 주성분 방향 $w^p$를 업데이트하는데, 이상치 자체가 $w^p$의 업데이트에 영향을 주어 재구성 오차를 낮추게 되면 위양성(False Positive)이나 미탐지가 발생할 수 있다. 저자는 이를 해결하기 위해 재구성 오차가 임계치를 넘을 경우 업데이트를 중단하는 기법을 향후 과제로 제시하였는데, 이는 시스템의 강건성(robustness)을 위해 반드시 해결되어야 할 지점이다.

## 📌 TL;DR

본 연구는 단변량 시계열의 스트리밍 이상치 탐지를 위해 **다중 척도 Lag-matrix와 Streaming PCA를 결합한 방법론**을 제안한다. 특히 연산 효율을 위한 **계층적 PCA 구조**와 탐지 정밀도를 높이기 위한 **Haar Wavelet 변환**을 도입하였으며, 스케일 간 상관관계를 이용한 점수 집계 방식이 단일 척도 방식보다 우수함을 입증하였다. 이 연구는 실시간으로 변화하는 시계열의 주기성을 고려해야 하는 산업 모니터링 시스템 등의 실무 적용 가능성이 높다.
