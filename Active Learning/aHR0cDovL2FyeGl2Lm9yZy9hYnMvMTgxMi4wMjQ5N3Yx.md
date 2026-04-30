# Active Learning Methods based on Statistical Leverage Scores

Cem Orhan, Oznur Tastan (2018)

## 🧩 Problem to Solve

본 논문은 지도 학습(Supervised Learning)에서 발생하는 데이터 라벨링 비용의 문제, 특히 라벨이 없는 데이터는 풍부하지만 정답 라벨을 획득하는 데에는 많은 비용과 전문가의 시간이 소요되는 상황을 해결하고자 한다. 모든 데이터를 라벨링하는 것은 정보의 중복성으로 인해 비효율적이며, 따라서 모델 성능을 최대한 유지하면서 라벨링할 예제를 최소화하는 Active Learning(능동 학습)이 필요하다.

기존의 Active Learning 방법론들은 크게 정보성(Informativeness) 기반의 불확실성 샘플링(Uncertainty Sampling)과 데이터 분포의 대표성(Representativeness) 기반의 샘플링으로 나뉜다. 하지만 불확실성 샘플링은 학습 초기 단계에서 결정 경계가 불안정하여 신뢰도가 낮고, 샘플링 편향(Sampling Bias)이 발생하여 데이터의 전체적인 분포를 반영하지 못한다는 한계가 있다. 본 논문의 목표는 Statistical Leverage Scores를 활용하여 데이터의 대표성을 효과적으로 측정하고, 이를 통해 효율적인 샘플링 기준을 제시하는 ALEVS와 DBALEVS 알고리즘을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 행렬의 구조적 특성을 분석하여 가장 영향력 있는 행(또는 열)을 찾아내는 Statistical Leverage Scores를 Active Learning의 쿼리 기준으로 도입한 것이다.

1.  **ALEVS (Active Learning by Statistical Leverage Scores)**: 순차적(Sequential) 모드에서 각 클래스별 커널 행렬을 구축하고, Leverage Score가 가장 높은 예제를 선택하여 데이터의 대표성을 확보한다.
2.  **DBALEVS (Diverse Batch-mode Active Learning by Statistical Leverage Scores)**: 배치(Batch) 모드에서 대표성뿐만 아니라 예제 간의 다양성(Diversity)을 동시에 고려한다. 이를 위해 submodular set function을 정의하고 탐욕적(Greedy) 최적화를 통해 최적에 가까운 배치를 효율적으로 선택한다.
3.  **대표성과 다양성의 통합**: 단순히 leverage score가 높은 샘플을 뽑는 것이 아니라, 이미 선택된 샘플 및 라벨링된 데이터와의 유사도를 페널티로 부여함으로써 중복 정보를 배제하고 정보 효율성을 극대화하였다.

## 📎 Related Works

논문에서는 기존 Active Learning 접근 방식을 두 가지 주요 흐름으로 설명한다.

-   **정보성 기반 접근법**: Uncertainty sampling이 대표적이며, 결정 경계와의 거리, 엔트로피, 앙상블 모델 간의 불일치 등을 측정한다. 그러나 초기 반복 단계에서 분류기가 불확실한 점이 너무 많아 신뢰할 수 없는 샘플을 선택할 위험이 있으며, 데이터 분포를 무시하는 경향이 있다.
-   **대표성 기반 접근법**: 데이터 분포를 잘 나타내는 인스턴스를 선택하며, 주로 클러스터링 알고리즘에 의존한다. 이 경우 클러스터링 성능에 따라 전체 시스템의 성능이 결정된다는 한계가 있다.
-   **하이브리드 접근법**: QUIRE와 같이 정보성과 대표성을 동시에 최적화하려는 시도가 있었으나, 본 논문은 Leverage Score라는 선형 대수적 도구를 통해 더 효율적이고 이론적인 근거가 명확한 대표성 측정 방식을 제안하며 차별화를 둔다.
-   **배치 모드 학습**: 단순 상위 $b$개 선택 방식은 중복 정보 문제를 야기한다. 기존 연구들은 Mutual Information 최대화나 Fisher Information 최소화 등을 통해 다양성을 확보하려 했으며, 본 논문은 이를 submodular optimization 프레임워크 내에서 해결하고자 한다.

## 🛠️ Methodology

### Statistical Leverage Scores의 정의

SPSD(Symmetric Positive Semi-Definite) 행렬 $A$의 고유값 분해가 $A = U \Sigma U^T$일 때, 상위 $k$개의 고유 공간을 생성하는 $m \times k$ 행렬 $U_1$을 정의한다. 이때 $A$에 대한 Statistical Leverage Score $\ell_i$는 $U_1$의 각 행의 제곱 유클리드 노름(squared Euclidean norm)으로 정의된다.

$$\ell_i := \|(U_1)_{(i)}\|_2^2, \quad i \in \{1, \dots, m\}$$

여기서 $\ell_i \in [0, 1]$이며, $\sum_{i=1}^m \ell_i = k$가 성립한다. 직관적으로 Leverage Score는 해당 행(예제)이 행렬의 저차원 근사(low-rank approximation)에 얼마나 큰 영향을 미치는지를 나타내며, 값이 클수록 데이터 분포 내에서 대표성이 높음을 의미한다.

### ALEVS (Sequential Mode)

ALEVS는 매 반복마다 하나의 예제를 쿼리하며, 절차는 다음과 같다.

1.  **클래스 분리**: 현재 학습된 분류기 $h_t$를 사용하여 라벨이 없는 데이터를 긍정($X_t^+$) 및 부정($X_t^-$) 클래스로 예측 분리한다.
2.  **커널 행렬 생성**: 각 클래스 내부의 유사도를 반영하는 커널 행렬 $K_t^+$와 $K_t^-$를 각각 계산한다.
3.  **Leverage Score 계산**: 각 행렬에 대해 Leverage Score를 계산한다. 이때 서로 다른 $m, k$ 값에서도 비교가 가능하도록 평균 Leverage Score가 1이 되도록 스케일링된 값을 사용한다.
    $$\ell_i = \frac{m}{k} \|(U_1)_{(i)}\|_2^2$$
4.  **샘플 선택**: 계산된 $\ell_i$ 중 최대값을 가진 예제를 쿼리한다.
    $$q = \arg \max_{i \in D_u^t} \ell_i$$

### DBALEVS (Batch Mode)

배치 모드에서는 대표성과 다양성을 동시에 최적화하기 위해 다음과 같은 set scoring function $F(S)$를 정의한다.

$$F(S) = \sum_{i \in S} (\ell_i + 1) - \frac{\alpha}{M} \sum_{i,j \in S, i \neq j} K(i,j)$$

-   **첫 번째 항**: 집합 $S$에 포함된 예제들의 개별적 대표성(Leverage Score)의 합이다.
-   **두 번째 항**: 선택된 예제들 간의 커널 유사도 $K(i,j)$를 합산하여 페널티를 부여함으로써 다양성을 강제한다. $\alpha$는 다양성 반영 비중을 조절하는 파라미터이며, $M$은 선택 가능한 집합 크기에 대한 제약 조건이다.

이 함수 $F(S)$는 **Submodular** 성질(한계 효용 체감 법칙)을 가지므로, 탐욕적(Greedy) 알고리즘을 통해 최적해의 $(1 - 1/e)$ 배 이상의 성능을 보장하는 근사해를 효율적으로 찾을 수 있다. DBALEVS는 긍정/부정 클래스에서 각각 $b/2$개씩 샘플을 추출하며, 기존에 라벨링된 데이터 $L_t$를 초기 집합으로 설정하여 이미 확보된 데이터와도 중복되지 않는 배치를 구성한다.

## 📊 Results

### 실험 설정
-   **데이터셋**: digit1, USPS, spambase, MNIST(3vs5), ringnorm 등 8개 이상의 데이터셋을 사용하였다.
-   **비교 대상**: Random sampling, Uncertainty sampling, LevOnAll(전체 데이터 leverage score), Transductive experimental design, QUIRE 등이 포함되었다.
-   **평가 지표**: 분류 정확도(Accuracy) 및 클래스 불균형 데이터의 경우 F1 Score를 사용하였다.

### 주요 결과
1.  **ALEVS 성능**: 대부분의 데이터셋에서 Random 및 Uncertainty sampling보다 높은 정확도를 보였다. 특히 학습 초기 단계에서 매우 효과적이었으며, 상태가 가장 좋은(State-of-the-art) QUIRE와 비교했을 때 비슷하거나 더 나은 성능을 보였으며 실행 속도는 훨씬 빨랐다.
2.  **클래스별 Leverage Score의 중요성**: 전체 풀에서 leverage score를 계산하는 `LevOnAll`보다 클래스별로 나누어 계산하는 ALEVS의 성능이 일관되게 높았다.
3.  **클래스 불균형 강건성**: 클래스 비율이 5:1, 10:1인 상황에서도 ALEVS는 두 클래스에서 균형 있게 샘플링하는 경향을 보였으며, 이에 따라 F1 Score에서 다른 방법론들을 압도하였다.
4.  **DBALEVS 성능**: 배치 모드에서 DBALEVS는 NearOpt, Random, Uncertainty sampling보다 우수한 성능을 기록하였다. 특히 Uncertainty sampling은 배치 모드에서 초기 가설에 과하게 의존하여 성능이 급격히 떨어지는 모습을 보였으나, DBALEVS는 이를 극복하였다.

## 🧠 Insights & Discussion

본 연구는 Statistical Leverage Scores가 데이터의 분포적 특징을 포착하는 강력한 도구임을 입증하였다.

-   **강점**: ALEVS와 DBALEVS는 특히 학습 초기 단계(Early iterations)에서 매우 강력하다. 이는 불확실성 기반 방법론들이 초기에 겪는 '신뢰할 수 없는 결정 경계' 문제를 데이터의 대표성 기반 샘플링으로 보완할 수 있음을 시사한다.
-   **한계 및 분석**: Leverage Score는 오직 데이터의 분포(X)만을 고려하며, 클래스 라벨의 불확실성(Y|X)은 고려하지 않는다. 따라서 논문에서는 ALEVS로 초반에 데이터 분포를 잡고, 후반에 Uncertainty sampling으로 정교하게 경계를 다듬는 하이브리드 전략의 가능성을 제시한다.
-   **비판적 해석**: DBALEVS에서 다양성 항을 추가한 것은 배치 모드의 고질적인 문제인 중복성(Redundancy)을 해결하기 위한 적절한 선택이었다. 다만, $\alpha$나 $\tau$ 같은 하이퍼파라미터가 데이터셋의 고유 특성(고유값 분포 등)에 따라 영향을 미치므로, 이에 대한 자동 최적화 방법이 추가된다면 더 범용적인 도구가 될 것이다.

## 📌 TL;DR

본 논문은 행렬의 구조적 중요도를 측정하는 **Statistical Leverage Scores**를 Active Learning에 도입하여, 데이터의 대표성을 극대화하는 **ALEVS**(순차 모드)와 다양성을 함께 고려하는 **DBALEVS**(배치 모드)를 제안하였다. 제안 방법론은 특히 **학습 초기 단계에서 매우 효율적**이며, **클래스 불균형 상황에서도 강건한 성능**을 보인다. 이는 향후 불확실성 기반 샘플링과의 하이브리드 구조를 통해 더욱 강력한 능동 학습 프레임워크를 구축하는 데 기여할 수 있을 것으로 기대된다.