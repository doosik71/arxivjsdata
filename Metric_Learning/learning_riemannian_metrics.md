# Learning Riemannian Metrics

Guy Lebanon (2003)

## 🧩 Problem to Solve

본 논문은 주어진 미분 가능 다양체(differentiable manifold)와 데이터 포인트 집합이 있을 때, 이에 적합한 Riemannian metric을 학습하는 문제를 다룬다.

일반적인 머신러닝 알고리즘(k-NN, Neural Networks, SVM 등)은 데이터가 유클리드 공간($\mathbb{R}^n$)이나 힐베르트 공간(Hilbert space)에 임베딩되어 있다고 가정하며, 기본적으로 유클리드 거리 측정 방식을 사용한다. 그러나 이러한 가정은 데이터의 실제 특성을 반영하지 못하는 경우가 많으며, 데이터 자체로부터 거리 구조(metric structure)를 추론하는 것이 더 타당하다.

특히 기존의 비선형 차원 축소 방식(LLE, PCA 등)은 데이터를 저차원 부분 다양체(submanifold)로 투영시키는데, 이는 다음과 같은 한계가 있다:

1. 실제 고차원 희소 데이터(sparse data)에서는 부분 다양체의 차원을 추정하기 매우 어렵다.
2. 데이터가 여러 개의 분리된 부분 다양체에 존재하거나 위치에 따라 차원이 다를 수 있다.
3. 학습 데이터에 과적합되어 새로운 데이터 포인트가 부분 다양체 밖에 존재할 경우, 이를 다시 투영하는 과정에서 유클리드 가정을 사용할 수밖에 없는 모순이 발생한다.

따라서 본 논문의 목표는 전체 임베딩 공간에서 정의되면서도 국소적인 변화를 포착할 수 있는 Riemannian metric을 학습하여, 분류 및 클러스터링 작업의 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **데이터셋의 역-부피(inverse volume)를 최대화하는 파라미터 기반의 metric 가문을 선택**하는 것이다.

중심적인 직관은 다음과 같다:

- 데이터가 밀집된 지역의 부피(volume)를 작게 만들면(즉, 역-부피를 크게 만들면), 해당 지역을 통과하는 경로의 길이가 상대적으로 짧아진다.
- 결과적으로 최단 경로인 측지선(geodesic)이 데이터가 밀집된 영역을 통과하게 되며, 이는 데이터의 내재적인 기하학적 구조를 효과적으로 포착하는 결과로 이어진다.
- 이를 위해 구체적으로 **Pull-back metric** 개념을 도입하여, 변환(transformation)의 파라미터 집합을 통해 metric을 정의함으로써 학습 가능한 형태로 구현하였다.

## 📎 Related Works

논문에서는 다음과 같은 기존 접근 방식들을 언급하며 차별점을 제시한다:

- **Saul and Jordan [12] / Xing et al. [13]:** 최적의 경로를 학습하거나 전역적(global) metric 구조를 학습한다. 하지만 Xing et al.의 방식은 공간 전체에서 metric이 일정하므로 비유클리드 기하학을 제한적으로만 포착할 수 있다.
- **Lanckriet et al. [7]:** 커널 행렬을 통해 유사도를 학습하지만, 결과로 얻은 Gram 행렬은 학습되지 않은 새로운 포인트(unseen points)로 일반화되지 않는다.
- **Manifold Learning (LLE, PCA, Spherical subfamilies):** 데이터를 저차원 부분 다양체로 표현하려 하지만, 앞서 언급한 대로 차원 추정의 어려움과 'off-manifold' 포인트 처리 문제(유클리드 투영 사용)라는 한계가 있다.

본 제안 방식은 metric이 공간 전체에서 정의되면서도 국소적(local)인 특성을 가지므로, 위 방법론들이 가진 일반화 및 투영 문제에서 자유롭다.

## 🛠️ Methodology

### 1. Riemannian Metric 학습 프레임워크

미분 가능 다양체 $M$과 데이터셋 $D = \{x_1, \dots, x_N\}$이 주어졌을 때, metric 후보 집합 $\mathcal{Q}$에서 최적의 Riemannian metric $g$를 선택한다. 이때 목적 함수 $O(g, D)$는 다음과 같이 정의된다:

$$O(g, D) = \sum_{i=1}^N \frac{1}{\text{dvol}_g(x_i)} / \int_M \frac{1}{\text{dvol}_g(x)} dx$$

여기서 $\text{dvol}_g(x) = \sqrt{\det g(x)}$는 metric $g$에 따른 미분 부피 요소(differential volume element)이다. 이 식은 정규화된 역-부피 요소의 합을 최대화하는 문제이며, 통계학적으로는 확률 밀도가 부피 요소에 반비례하는 모델에서의 최대 가능도 추정(Maximum Likelihood Estimation, MLE) 문제로 해석될 수 있다.

### 2. Pull-back Metrics와 변환

Metric $g$를 직접 정의하는 것은 어렵기 때문에, 본 논문은 **Pull-back metric**을 사용한다.

- 미분 동형 사상(diffeomorphism) $F: M \to N$이 있고 $N$에 metric $h$가 정의되어 있을 때, $M$에서의 pull-back metric $F^*h$는 다음과 같다:
  $$F^*h_x(u, v) = h_{F(x)}(F_*u, F_*v)$$
  (여기서 $F_*$는 push-forward map이다.)
- 이를 통해 $M$과 $N$ 사이의 거리 관계가 $d_{F^*h}(x, y) = d_h(F(x), F(y))$가 되는 등거리 사상(isometry)을 구현할 수 있다.

### 3. 다항 심플렉스(Multinomial Simplex)에의 적용

본 논문은 구체적인 구현으로 $n$-심플렉스 $P_n$ 위에서의 metric 학습을 다룬다.

- **기본 Metric:** Fisher information metric $J$를 기본으로 사용한다.
- **변환 가문:** 파라미터 $\lambda$에 의해 정의되는 Lie 군 변환 $F_\lambda$를 도입하여 $\mathcal{Q} = \{F_\lambda^* J : \lambda \in \text{int} P_n\}$를 후보군으로 설정한다.
- **측지선 거리:** 이 구조에서 두 점 $x, y$ 사이의 거리는 다음과 같은 폐쇄형(closed form)으로 계산된다:
  $$d(x, y) = \text{acos} \left( \frac{\sum x_i A_i \sum y_i A_i}{\sqrt{\sum x_i^2 A_i^2} \sqrt{\sum y_i^2 A_i^2}} \right)$$
  (여기서 $A$는 파라미터 $\lambda$와 관련된 값이다.)

### 4. 학습 및 계산 효율성

목적 함수를 최대화하기 위해 파라미터 $\lambda$를 추정해야 하며, 이때 정규화 항 $Z = \int_{P_n} \frac{1}{\sqrt{\det F_\lambda^* J(x)}} dx$를 계산해야 한다.

- **효율적 계산:** 동적 계획법(Dynamic Programming)과 고속 푸리에 변환(FFT)을 결합하여 $Z$를 $O(n^2 \log n)$ 시간 복잡도로 계산할 수 있음을 보였다.
- **최적화:** 경사 하강법(Gradient Descent)을 사용하여 최적의 $A$ 파라미터를 학습한다.

## 📊 Results

### 실험 설정

- **데이터셋:** WebKB 데이터셋의 '교수(faculty) vs 학생(student)' 웹페이지 분류 작업.
- **표현 방식:** 문서를 다항 MLE 또는 MAP 추정을 통해 심플렉스 상의 점으로 매핑(TF 표현).
- **비교 대상 (Baselines):**
    1. TFIDF cosine similarity
    2. TF 표현의 $L_2$ 거리
    3. 제안 방법(Learned Riemannian Metric의 측지선 거리)
- **평가 지표:** 테스트 세트의 오류율(Test-set error rate).

### 주요 결과

- **정량적 결과:** 그림 5의 결과에 따르면, 학습된 Riemannian metric 기반의 거리를 사용한 k-NN 분류기가 TFIDF와 $L_2$ 거리보다 일관되게 낮은 오류율을 보였다. 특히 훈련 데이터의 크기가 커질수록 성능 향상 폭이 뚜렷하게 나타났다.
- **정성적 결과:** 학습된 파라미터 $A$와 TFIDF의 IDF 값을 비교했을 때, 두 방법 모두 흔한 단어(stopwords)에는 낮은 가중치를 부여했다. 그러나 IDF는 희귀한 고유 명사에 높은 점수를 주는 반면, 제안 방법은 희귀한 일반 명사에 더 높은 가중치를 두는 경향을 보였다. 이는 IDF는 단어의 출현 여부(binary)를 보지만, 제안 방법은 단어의 출현 빈도(count)를 고려하기 때문이다.

## 🧠 Insights & Discussion

### 강점

- **기하학적 타당성:** 단순히 휴리스틱한 가중치를 주는 것이 아니라, 다양체 상의 부피 개념을 통해 데이터의 내재적 구조를 학습하는 수학적 프레임워크를 제시하였다.
- **일반화 능력:** 부분 다양체 투영 방식의 고질적인 문제인 'off-manifold' 포인트 문제를 해결하고, 전체 공간에서 정의되는 국소적 metric을 학습함으로써 강건함을 확보하였다.
- **계산 효율성:** 고차원 데이터에서도 적용 가능하도록 FFT와 DP를 이용해 정규화 항 계산 복잡도를 획기적으로 낮추었다.

### 한계 및 논의사항

- **파라미터 가문의 제한:** 본 논문에서는 $F_\lambda$라는 특정 변환 군을 가정하였다. 만약 실제 데이터의 기하학적 구조가 이 변환 군으로 표현할 수 없는 형태라면 성능에 한계가 있을 수 있다.
- **계산 비용:** $O(n^2 \log n)$으로 낮추었음에도 불구하고, 매우 거대한 어휘 사전(vocabulary size $n$이 매우 큰 경우)에서는 여전히 계산 부담이 존재할 수 있다.

## 📌 TL;DR

본 논문은 유클리드 거리 가정의 한계를 극복하기 위해, 데이터 밀집 지역의 부피를 최소화하는 방향으로 Riemannian metric을 학습하는 프레임워크를 제안한다. 특히 다항 심플렉스 상에서 Fisher information의 Pull-back metric을 사용하여 텍스트 데이터를 효과적으로 임베딩하였으며, 이를 통해 TFIDF보다 우수한 문서 분류 성능을 입증하였다. 이 연구는 고차원 희소 데이터의 내재적 기하 구조를 학습하는 새로운 방법론을 제시했다는 점에서 향후 메트릭 학습(Metric Learning) 연구에 중요한 기여를 한다.
