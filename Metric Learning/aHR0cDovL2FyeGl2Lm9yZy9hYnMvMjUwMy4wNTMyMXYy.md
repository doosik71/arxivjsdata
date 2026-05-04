# A Review on Riemannian Metric Learning: Closer to You than You Imagine

Samuel Gruffaz and Josua Sassen (2025)

## 🧩 Problem to Solve

본 논문은 전통적인 거리 학습(Distance Metric Learning)이 가지는 한계를 극복하기 위해 **Riemannian Metric Learning (RML)**이라는 프레임워크를 체계적으로 분석하고 리뷰하는 것을 목표로 한다.

전통적인 거리 학습 방식은 주로 유클리드 공간(Euclidean space)에서의 전역적 거리(global distance)에 의존한다. 그러나 실제 복잡한 데이터 구조는 단순한 선형 변환이나 유클리드 거리만으로는 그 내재적인 기하학적 구조(intrinsic data geometry)를 충분히 캡처할 수 없다는 문제가 있다. 데이터가 곡률을 가진 다양체(manifold) 위에 존재할 때, 전역적인 거리 척도는 데이터의 국소적 특성과 위상적 구조를 무시하게 되어 분석의 정확도를 떨어뜨린다.

따라서 본 논문은 미분 기하학의 Riemannian manifold 개념을 도입하여, 데이터의 위치에 따라 변화하는 거리 척도인 Riemannian metric을 학습함으로써 데이터의 분포, 곡률, 거리를 보다 정확하게 모델링하는 방법론을 제시하고, 이를 다양한 응용 분야에 적용하는 방안을 논의한다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 파편화되어 있던 Riemannian metric learning 관련 연구들을 통합하여 학술적인 체계를 세운 것에 있다. 구체적인 기여 사항은 다음과 같다.

1.  **RML의 일반적 프레임워크 정의**: 거리 학습을 Riemannian geometry의 관점에서 일반화하여, 단순한 임베딩 학습을 넘어 Riemannian metric $g$ 자체를 추정하는 문제로 정의하였다.
2.  **방대한 응용 사례의 체계화**: 인과 추론(Causal Inference), 최적 운송(Optimal Transport), 생성 모델링(Generative Modeling), 표현 학습(Representation Learning) 등 다양한 분야에서 RML이 어떻게 활용되는지를 분석하였다.
3.  **방법론의 분류 체계(Taxonomy) 제시**: Riemannian metric의 파라미터화 방법(Parametrization), 계산 방법(Computational methods), 목적 함수(Objectives), 최적화 전략(Optimization)에 따라 기존 연구들을 정교하게 분류하였다.
4.  **이론적/실무적 연구 방향 제시**: 비선형 거리 학습의 통계적 보장, 기하학적 특성을 고려한 최적화 방법, 소프트웨어 도구의 필요성 등 향후 연구 방향을 제안하였다.

## 📎 Related Works

본 논문은 기존의 거리 학습(Distance Metric Learning) 리뷰들과 RML의 차별점을 명확히 한다.

-   **기존 거리 학습 리뷰**: Yang and Jin [205], Bellet et al. [21], Kulis et al. [111] 등의 연구들은 주로 유클리드 공간에서의 Mahalanobis 거리나 딥러닝 기반의 임베딩 학습에 집중하였다. 이들은 주로 전역적인 거리 척도 $G$를 학습하거나 비선형 변환 $\phi(x)$를 통한 거리 $d(\phi(x), \phi(y))$를 다루었으나, 공간의 각 지점마다 서로 다른 metric이 존재하는 Riemannian structure는 다루지 않았다.
-   **차별점**: RML은 단순한 비선형 변환을 넘어, 매끄러운 기하학적 형태를 가진 다양체 위에서 국소적으로 길이를 측정하는 metric field $g$를 학습한다. 이는 갈릴레이 상대성 이론이 일반 상대성 이론으로 확장된 것과 유사한 개념적 도약을 의미하며, Geodesics(측지선), Parallel Transport(평행 이동), Ricci Curvature(리치 곡률)와 같은 강력한 기하학적 도구들을 머신러닝에 도입할 수 있게 한다.

## 🛠️ Methodology

본 논문은 Riemannian metric learning을 다음과 같은 수학적 최적화 문제로 정의한다.

### 1. 전체 시스템 구조 및 목표 함수
학습하고자 하는 Riemannian metric $g$의 집합을 $G_\theta \subset G_M$이라 할 때, 최적화 문제는 다음과 같이 표현된다.
$$\text{argmin}_{g \in G_\theta \subset G_M} L(F(g), X)$$
여기서 $F(g) = (d_g, \text{Exp}_g, T_g, \text{vol}_g)$는 학습된 metric $g$로부터 유도되는 기하학적 객체들(측지선 거리, 지수 맵, 평행 이동, 체적 형식)의 집합이며, $L$은 목적 함수, $X$는 데이터셋이다.

### 2. 핵심 기하학적 구성 요소
-   **Riemannian Metric ($g$):** 다양체 $M$의 각 점 $p$의 접공간(Tangent space) $T_p M$에서 정의되는 매끄러운 내적(inner product)의 가족이다.
-   **Riemannian Distance ($d_g$):** 두 점 $x, y$를 잇는 모든 가능한 매끄러운 곡선 $\gamma$ 중 길이가 최소인 경로의 길이다.
    $$d_g(x, y) = \inf_{\gamma} \int_{0}^{1} \sqrt{\dot{\gamma}(t)^\top g(\gamma(t)) \dot{\gamma}(t)} dt$$
-   **Geodesics (측지선):** 유클리드 공간의 직선을 일반화한 개념으로, 가속도가 0인 곡선($\nabla_{\dot{\gamma}} \dot{\gamma} = 0$)을 의미한다.
-   **Exponential Map ($\text{Exp}_p$):** 접공간의 벡터 $V$를 따라 측지선을 그려 다양체 위의 한 점으로 매핑하는 함수이다.
-   **Parallel Transport ($T$):** 곡선을 따라 벡터를 이동시킬 때, 그 벡터의 길이와 각도를 보존하며 이동시키는 방법이다.

### 3. Metric 파라미터화 방법 (Parametrization)
논문은 metric $g$를 어떻게 구현할 것인가에 대해 두 가지 접근법을 제시한다.
-   **명시적 방법 (Explicit Methods):** $M \to \mathbb{R}^{d \times d}$ 맵을 직접 학습한다.
    -   Constant Metric: 모든 공간에서 동일한 $G$를 사용 (Mahalanobis 거리).
    -   Piecewise Constant: 보로노이 셀(Voronoi cell) 단위로 상수를 할당.
    -   Kernel Metric: RBF 커널 등을 이용해 주요 지점의 metric을 보간(interpolation).
    -   Neural Metric: 신경망을 통해 $S_{++}^d$(양정치 행렬 공간) 값을 출력하도록 설계.
-   **암시적 방법 (Implicit Methods):** 다른 구조를 통해 metric을 유도한다.
    -   Pullback Metric: 매끄러운 맵 $f: M \to N$을 학습하고, 타겟 공간 $N$의 metric을 $M$으로 끌어온다.
    -   Graph-based: Isomap과 같이 그래프의 최단 경로를 통해 거리를 근사한다.

### 4. 학습 절차 및 최적화
-   **목적 함수 (Objectives):**
    -   Classification: Contrastive loss나 Triplet loss를 사용하여 유사한 데이터는 가깝게, 다른 데이터는 멀게 배치한다.
    -   Regression: 관측된 거리값과의 MSE를 최소화하거나, 관측된 궤적(trajectory)을 측지선으로 피팅한다.
    -   Distribution: 데이터 밀도에 맞춰 체적 형식 $\text{vol}_g$를 학습한다.
-   **최적화 알고리즘:** Adam, L-BFGS와 같은 표준 경사 하강법부터, SPD 행렬의 기하학적 구조를 직접 이용하는 Riemannian Optimization까지 사용된다.

## 📊 Results

본 논문은 리뷰 논문이므로 새로운 실험 결과보다는 기존 문헌들의 성과를 종합하여 분석하였다. RML이 실제 적용된 주요 사례와 그 결과는 다음과 같다.

-   **질병 진행 모델링 (Alzheimer's Disease):** 알츠하이머 환자의 뇌 영상 및 인지 점수 변화를 측지선으로 모델링하였다. 특히 $\text{exp-parallelization}$을 통해 환자 개개인의 건강 상태 차이를 유지하면서 질병의 공통적 진행 방향을 학습하는 데 성공하였다.
-   **단일 세포 RNA 시퀀싱 (scRNA-seq):** 세포의 분화 과정을 추론하는 Trajectory Inference 문제에서, 유클리드 보간법은 데이터가 없는 빈 공간으로 궤적이 이탈하는 문제가 있었다. RML을 적용하여 데이터 다양체 상의 측지선을 따라 보간함으로써 보다 생물학적으로 타당한 전이 경로를 생성하였다.
-   **생성 모델 및 샘플링:** latent space에 Riemannian metric을 도입하여, 데이터 밀도가 높은 지역에서는 샘플링 속도를 조절하고, $\text{vol}_g$를 이용해 선택 편향(selection bias)이 없는 균일한 샘플링을 구현하였다.
-   **인과 추론 (Causal Inference):** 공변량(covariates) 간의 Riemannian 거리를 학습하여 매칭(matching) 성능을 향상시켰으며, 그래프의 Ricci 곡률을 이용해 처리 효과(treatment effect) 추정의 신뢰도를 높였다.

## 🧠 Insights & Discussion

### 강점 및 가능성
-   **표현력의 확장**: RML은 데이터를 단순한 점의 집합이 아닌 기하학적 구조로 파악하게 함으로써, 데이터 희소성(data sparsity) 문제에서도 강건한 일반화 성능을 보일 가능성이 크다.
-   **해석 가능성**: 측지선이나 평행 이동과 같은 개념은 물리적 궤적이나 상태 전이를 설명하는 직관적인 도구가 된다.

### 한계 및 미해결 과제
-   **계산 복잡도**: 측지선 거리 $d_g$를 계산하기 위해서는 미분 방정식(ODE)을 수치적으로 풀어야 하므로, 고차원 데이터에서 계산 비용이 매우 높다.
-   **이론적 보장 부족**: Mahalanobis 거리 학습에 비해 비선형 Riemannian metric 학습에 대한 통계적 수렴성이나 표본 복잡도(sample complexity)에 대한 이론적 분석이 매우 부족하다.
-   **도구의 부재**: Riemannian geometry를 머신러닝에 쉽게 적용할 수 있는 통합 라이브러리가 부족하여 진입 장벽이 높다.

### 비판적 해석
본 논문은 RML의 광범위한 가능성을 제시하지만, 실제 구현 단계에서의 'Tractability(다루기 쉬움)' 문제를 완전히 해결하지는 못했다. 특히 신경망을 이용한 explicit metric 학습은 $S_{++}^d$ 제약 조건을 만족시켜야 하며, 고차원에서의 ODE 수치 적분은 실시간 시스템에 적용하기 어렵다. 따라서 향후에는 효율적인 근사 알고리즘(예: Neural ODE의 최적화)과의 결합이 필수적일 것으로 보인다.

## 📌 TL;DR

본 논문은 유클리드 거리 학습의 한계를 넘어 데이터의 내재적 기하학 구조를 학습하는 **Riemannian Metric Learning (RML)**의 이론과 응용을 집대성한 리뷰 보고서이다. RML은 **측지선, 지수 맵, 평행 이동**과 같은 미분 기하학적 도구를 통해 복잡한 데이터 궤적 추론, 생성 모델링, 인과 추론 등에서 탁월한 성능을 보임을 입증하였다. 비록 계산 복잡도와 이론적 기반이라는 숙제가 남아있으나, 표현 학습(Representation Learning)의 패러다임을 '평면'에서 '곡면'으로 확장했다는 점에서 향후 고차원 데이터 분석 및 생성 AI 분야에 핵심적인 역할을 할 것으로 기대된다.