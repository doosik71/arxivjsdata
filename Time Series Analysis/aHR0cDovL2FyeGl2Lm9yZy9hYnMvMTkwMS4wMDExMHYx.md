# Time series classification based on triadic time series motifs

Wen-Jie Xie, Rui-Qi Han, Wei-Xing Zhou (2019)

## 🧩 Problem to Solve

본 논문은 시계열 데이터의 유사성을 정량화하고 이를 통해 서로 다른 시계열을 분류하는 문제를 해결하고자 한다. 시계열 분석에서 두 시계열 간의 유사성을 측정하는 것은 매우 중요한 기초 작업이며, 이는 금융, 의료 등 다양한 분야에서 활용된다.

기존의 대표적인 거리 측정 방식인 Euclidean distance는 계산이 단순하지만 시계열의 특성을 충분히 반영하지 못해 성능이 떨어지는 경우가 많다. 반면, Dynamic Time Warping (DTW)은 매우 경쟁력 있는 성능을 보이지만 계산 복잡도가 높고 윈도우 너비와 같은 파라미터 최적화가 필요하다는 단점이 있다. 따라서 본 연구의 목표는 시계열의 동역학적 특성을 효과적으로 추출할 수 있는 새로운 특징량인 Triadic time series motifs를 정의하고, 이를 이용한 Motif occurrence profiles를 통해 시계열 분류 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 시계열 내에서 세 개의 데이터 포인트로 구성된 특정 패턴인 'Triadic time series motifs'를 정의하고, 이들의 발생 빈도를 벡터화한 'Motif occurrence profile'을 특징량(feature)으로 사용하는 것이다.

전통적인 시계열 분석이 값의 절대적인 크기나 전체적인 형상에 집중했다면, 제안된 방법은 시계열 내의 국소적인 서열 관계와 가시성(visibility)을 결합하여 시스템의 동역학적 특성을 파악한다. 이는 긴 시계열 데이터를 매우 낮은 차원의 특징 공간(6개 모티프의 빈도 분포)으로 축소함으로써, 효율적이면서도 강력한 분류 기준을 제공한다.

## 📎 Related Works

본 연구는 시계열을 네트워크 형태로 변환하는 Visibility Graph (VG) 및 Horizontal Visibility Graph (HVG) 연구에서 영감을 받았다. 특히 HVG에서 나타나는 네트워크 모티프(Network motifs) 개념을 시계열 영역으로 확장하였다.

기존의 HVG 모티프는 단순히 데이터 포인트 간의 가시성(연결 여부)만을 고려하므로, 무방향 HVG에서는 체인(chain)과 삼각형(triangle) 두 가지 형태만 존재한다. 그러나 본 논문에서 제안하는 Triadic time series motifs는 가시성뿐만 아니라 데이터 포인트 간의 상대적 크기와 순서(ordinal order)를 동시에 고려한다. 이를 통해 기존 HVG 모티프보다 더 세밀한 구조(finer structures)를 탐색할 수 있으며, 이는 Bandt와 Pompe가 제안한 Ordinal patterns 기반의 Permutation entropy와도 유사한 맥락에서 동역학적 특성을 추출한다.

## 🛠️ Methodology

### 1. Triadic Time Series Motifs 정의

시계열 $\{x_i\}_{i=1,\dots,L}$에서 임의로 선택된 세 개의 데이터 포인트 $\{x_i, x_j, x_k\}$ ($i < j < k$)가 다음 조건을 만족할 때 하나의 모티프가 형성된다고 정의한다.

$$x_i > x_n \text{ and } x_j > x_n, \forall n \in (i, j)$$
$$x_j > x_m \text{ and } x_k > x_m, \forall m \in (j, k)$$

위 조건은 포인트 $x_i$와 $x_j$ 사이의 모든 값들이 두 값보다 작아야 하며, 마찬가지로 $x_j$와 $x_k$ 사이의 모든 값들이 두 값보다 작아야 함을 의미한다. 이 조건을 만족하는 세 포인트의 상대적 크기 관계에 따라 총 6가지 유형의 모티프 $M_1, M_2, M_3, M_4, M_5, M_6$가 정의된다.

### 2. Motif Occurrence Profile 추출

각 시계열에서 각 모티프 $M_i$가 나타난 횟수를 $O(M_i)$라고 할 때, 전체 모티프 발생 횟수 대비 개별 모티프의 발생 빈도 $f_i$를 다음과 같이 계산한다.

$$f_i = \lim_{L \to \infty} \frac{O(M_i)}{\sum_{j=1}^{6} O(M_j)}$$

이렇게 계산된 $\mathbf{f} = [f_1, f_2, f_3, f_4, f_5, f_6]$ 벡터를 Motif occurrence profile이라고 하며, 이는 시계열의 고유한 특성을 나타내는 지문과 같은 역할을 한다.

### 3. 분류 절차

추출된 Motif occurrence profile을 특징 벡터로 사용하여 1-Nearest Neighbor (1NN) 분류기를 적용한다. 유사도 측정 방식으로는 제안된 Motif profile의 거리, DTW 거리, 그리고 Euclidean distance를 각각 사용하여 분류 정확도를 비교한다.

## 📊 Results

### 1. 합성 데이터 실험 (Chaotic Maps)

Logistic map 및 다양한 카오스 맵(Henon, Ikeda, Generalized Henon, Folded-tower map)을 통해 생성된 시계열로 실험을 진행하였다.

- **Logistic Map:** 제어 파라미터 $r$의 값에 따라(3.2, 3.5, 3.6, 3.8, 4) 모티프 분포가 뚜렷하게 다르게 나타났다. 특히 $r=3.2$인 경우 분석적으로 $\mathbf{f} = [0.4, 0.2, 0.2, 0.2, 0, 0]$임이 증명되었으며 실험 결과와 일치하였다.
- **분류 성능:** 시계열 길이가 길어질수록 Motif profile과 DTW 기반의 분류 정확도는 100%에 수렴하였으나, Euclidean distance는 길이가 길어질수록 노이즈의 영향으로 오히려 정확도가 떨어지는 경향을 보였다.
- **강건성(Robustness):** 데이터 삭제율($E$)을 높였을 때, Logistic map 데이터셋에서는 Motif profile 방법이 DTW보다 더 높은 강건성을 보였다. 반면, 카오스 맵 데이터셋에서는 DTW가 더 우수한 강건성을 나타냈다.

### 2. 실세계 데이터 실험 (UCR Archive)

UCR Time Series Classification Archive의 128개 데이터셋을 대상으로 성능을 평가하였다.

- **정량적 결과:** 128개 데이터셋 중 11개 데이터셋에서 제안 방법이 DTW보다 높은 정확도를 보였으며, 18개 데이터셋에서 Euclidean distance보다 우수한 성능을 기록하였다.
- **정성적 결과:** 레이더 차트를 통해 클래스별 평균 모티프 프로파일을 시각화한 결과, 클래스 간 프로파일 라인의 분리 정도가 분류 정확도와 밀접한 관련이 있음을 확인하였다.

## 🧠 Insights & Discussion

본 연구는 시계열 데이터를 복잡 네트워크의 관점에서 해석하여, 단순한 수치적 거리 측정에서 벗어나 시스템의 내재된 동역학적 패턴을 추출하는 새로운 접근법을 제시하였다.

**강점:**

- 시계열의 길이에 관계없이 항상 6차원(실질적으로는 합이 1이므로 5차원)의 고정된 특징 벡터로 변환하므로 차원 축소 효과가 매우 뛰어나다.
- 특정 데이터셋(Logistic map 등)에서 데이터 손실이 발생하더라도 시스템의 구조적 특성을 유지하므로 DTW보다 강건한 분류가 가능하다.

**한계 및 비판적 해석:**

- 전반적인 성능 면에서는 여전히 DTW가 우세한 경우가 많다. 이는 DTW가 시계열의 시간적 왜곡을 매우 유연하게 처리하기 때문이다.
- 본 논문에서 제안한 모티프 정의는 세 포인트의 상대적 크기만을 고려하므로, 값의 절대적인 변화 폭이나 진폭이 중요한 도메인에서는 정보 손실이 발생할 수 있다.
- 11개의 데이터셋에서 DTW를 이겼다는 점은 고무적이나, 대다수의 데이터셋에서는 여전히 DTW가 더 나은 성능을 보인다는 점은 본 방법이 보편적인 대체제보다는 특정 유형의 시계열(동역학적 특성이 강한 데이터)에 특화된 도구임을 시사한다.

## 📌 TL;DR

이 논문은 시계열에서 세 데이터 포인트의 상대적 크기와 가시성을 기반으로 한 **Triadic time series motifs**를 정의하고, 그 발생 빈도 분포(**Motif occurrence profiles**)를 이용해 시계열을 분류하는 방법을 제안한다. 실험 결과, 이 방법은 시계열을 매우 낮은 차원의 특징 공간으로 효율적으로 축소하며, 일부 데이터셋에서는 DTW보다 높은 정확도와 데이터 손실에 대한 강건성을 보였다. 향후 복잡한 동역학 시스템의 시계열 특성을 분석하고 분류하는 데 유용한 도구가 될 가능성이 크다.
