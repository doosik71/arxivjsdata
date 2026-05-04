# POSITIVE SEMIDEFINITE SUPPORT VECTOR REGRESSION METRIC LEARNING

Lifeng Gu (2020)

## 🧩 Problem to Solve

본 논문은 거리 행렬 학습(Metric Learning)에서 기존 방식들이 가지는 한계점을 해결하고자 한다. 일반적인 거리 학습 방법들은 샘플 쌍(sample pair)을 단순히 '유사함(similar)' 또는 '유사하지 않음(dissimilar)'의 이분법적 관계로 정의하여 학습한다. 그러나 멀티 라벨 학습(Multi-label learning)이나 라벨 분포 학습(Label distribution learning)과 같은 복잡한 실제 응용 분야에서는 샘플 간의 관계를 단순히 이분법적으로 정의하기 어렵다는 문제가 존재한다.

이를 해결하기 위해 제안된 Relation Alignment Metric Learning (RAML) 프레임워크는 회귀(Regression) 방식을 도입하여 샘플 간의 관계를 보다 유연하게 모델링한다. 하지만 기존 RAML은 Support Vector Regression (SVR) 솔버를 사용하여 최적화를 수행하는데, 이 과정에서 학습된 거리 행렬 $M$이 Positive Semidefinite (PSD) 조건을 만족한다는 보장이 없다. 거리 행렬이 PSD가 아닐 경우, 수학적으로 유효한 거리 척도로서의 성질을 잃게 되며, 기존 RAML은 이를 해결하기 위해 SVD(Singular Value Decomposition)를 통한 근사치 계산법을 사용하였으나 이는 거리 행렬의 변별력을 저하시키는 결과를 초래한다. 따라서 본 논문의 목표는 SVR 기반의 프레임워크 내에서 PSD 조건을 직접적으로 만족하는 거리 행렬을 학습하는 새로운 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 SVR 프레임워크를 확장하여 거리 행렬 $M$이 PSD 성질을 갖도록 강제하는 두 가지 새로운 정식화(formulation)인 **RAML-PCSVR**과 **RAML-NCSVR**을 제안한 것이다.

중심 아이디어는 거리 행렬 $M$을 샘플 쌍의 외적(outer product)들의 선형 결합으로 표현하고, 결합 계수가 특정 조건을 만족하게 함으로써 행렬의 고윳값이 항상 0 이상이 되도록 보장하는 것이다. 이를 통해 SVD를 통한 사후 처리 없이도 최적화 과정에서 직접 PSD 거리 행렬을 획득할 수 있으며, 이는 결과적으로 다양한 학습 작업(단일 라벨, 멀티 라벨, 라벨 분포 학습)에서 더 높은 성능의 변별력을 제공한다.

## 📎 Related Works

기존의 거리 학습 연구들은 주로 마할라노비스 거리(Mahalanobis distance)를 학습하는 데 집중해 왔으며, 라벨 정보를 바탕으로 Doublet, Triplet, Quadruplet 제약 조건을 생성하여 유사한 샘플은 가깝게, 유사하지 않은 샘플은 멀게 배치하는 방식을 취한다.

본 논문에서 기반으로 하는 RAML 프레임워크는 이러한 이분법적 제약 대신, 특성 공간(feature space)에서의 샘플 관계와 결정 공간(decision space)에서의 샘플 관계를 일치시키는 '관계 정렬(Relation Alignment)' 관점을 도입하였다. 결정 공간에서의 관계 함수 $g(y_i, y_j)$를 정의하고, 이를 특성 공간의 거리 함수 $f(x_i, x_j, M, b)$가 모사하도록 하는 회귀 문제로 정의한 것이 특징이다. 그러나 앞서 언급했듯이 기존 RAML의 SVR 솔버는 PSD 행렬을 보장하지 못하며, SVD를 이용한 강제 PSD 변환은 행렬의 정보를 손실시킨다는 한계가 있다.

## 🛠️ Methodology

### 전체 시스템 구조 및 목적 함수
본 논문은 특성 공간에서의 샘플 관계를 다음과 같이 정의한다.
$$f(x_i, x_j, M, b) = (x_i - x_j)^T M (x_i - x_j) + b = \langle M, T_{ij} \rangle + b$$
여기서 $T_{ij} = (x_i - x_j)(x_i - x_j)^T$는 샘플 쌍의 scaled second sample moment이며, $\langle \cdot, \cdot \rangle$는 Frobenius 내적을 의미한다. 학습의 목표는 이 함수값이 결정 공간의 관계 함수 $g(y_i, y_j)$와 최대한 일치하도록 하는 것이다.

### 1. RAML-PCSVR
RAML-PCSVR은 기존 RAML의 듀얼 최적화 문제에 직접적인 제약 조건을 추가하는 방식이다. 거리 행렬 $M$의 해가 다음과 같이 표현될 때:
$$M = \sum_{i=1}^n (a_i - a_i^*) T_i$$
여기서 $a_i, a_i^*$는 라그랑주 승수이다. 본 논문은 $a_i \ge a_i^*$라는 제약 조건을 추가함으로써 $M$이 항상 PSD가 됨을 증명하였다. 임의의 벡터 $\mu$에 대해 $\mu^T M \mu = \sum (a_i - a_i^*) (\mu^T (x_{i1} - x_{i2}))^2 \ge 0$이 성립하기 때문이다.

### 2. RAML-NCSVR
RAML-NCSVR은 듀얼 문제의 수정이 아닌, primal formulation 단계에서 $M$을 다음과 같이 정의하여 PSD를 보장한다.
$$M = \sum_{i=1}^n \mu_i T_i, \quad \text{where } \mu_i \ge 0$$
이를 위해 다음과 같은 최적화 문제를 푼다.
$$\min_{\mu, \xi, \xi^*} \frac{1}{2} \sum_{i,j=1}^n \mu_i \mu_j \langle T_i, T_j \rangle + \lambda \sum_{i=1}^n (\xi_i + \xi_i^*)$$
제약 조건은 $\epsilon$-insensitive loss 함수를 따르며, $\mu_i \ge 0$ 조건을 포함한다. 이 문제는 직접 풀기 어렵기 때문에 보조 변수 $\rho$를 도입하고, $\alpha, \alpha^*$와 $\rho$를 번갈아 업데이트하는 교대 최적화(Alternative Optimization) 방식을 통해 해결한다.

### 학습 절차 (Algorithm 1)
1. 학습 데이터에서 샘플 쌍 $(x_{i1}, x_{i2})$을 생성하고 결정 공간의 관계 $g(x_{i1}, x_{i2})$를 계산한다.
2. **RAML-PCSVR**의 경우: QP(Quadratic Programming)를 통해 제약 조건 $a_i \ge a_i^*$를 만족하는 해를 구한다.
3. **RAML-NCSVR**의 경우: $\alpha, \alpha^*$를 업데이트한 후 $\rho$를 업데이트하고, 최종적으로 $\mu$를 결정할 때까지 반복한다.
4. 최종적으로 거리 행렬 $M$을 산출한다.

## 📊 Results

### 실험 설정
- **작업 및 데이터셋**: 
    - 단일 라벨 분류: Binalpha, Caltech101, MnistDat, Mpeg7, News20, TDT20, USPST
    - 멀티 라벨 분류: Emotion, Flags, Corel800
    - 라벨 분포 학습: Nature Scene (2,000장 이미지, 9개 라벨)
- **비교 대상**: ITML, LMNN, DML, DSVM, GMML 및 기존 RAML-SVR, MLKNN, AAKNN 등
- **평가 지표**: 정확도(Accuracy), Hamming loss, Ranking loss, Average Precision, 그리고 라벨 분포 학습을 위한 Chebyshev, Clark distance 등

### 주요 결과
- **단일 라벨 분류**: 대부분의 데이터셋에서 RAML-PCSVR와 RAML-NCSVR가 기존 RAML-SVR 및 다른 baseline들보다 우수한 정확도를 보였다. 특히 SVD를 통한 근사치를 사용한 RAML-SVR보다 성능이 향상되었는데, 이는 PSD 조건을 직접 학습함으로써 거리 행렬의 변별력이 유지되었기 때문이다.
- **멀티 라벨 분류**: MLKNN 대비 모든 지표(Hamming Loss $\downarrow$, Average Precision $\uparrow$ 등)에서 우수한 성능을 보였으며, 특히 RAML-NCSVR가 가장 좋은 성능을 기록하였다.
- **라벨 분포 학습**: AAKNN 대비 예측 분포와 실제 분포 사이의 거리가 더 가깝게 학습됨을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 거리 행렬 학습에서 PSD 조건이 갖는 중요성을 다시 한번 확인시켜 주었다. 기존 RAML이 가졌던 '사후 SVD 변환' 방식은 최적화된 해를 강제로 변형하는 것이기에 정보 손실이 발생할 수밖에 없다. 반면, 본 논문이 제안한 방식은 최적화 목적 함수와 제약 조건 자체에 PSD 성질을 내재화함으로써 수학적 정밀도와 실용적 성능을 동시에 잡았다.

다만, 몇 가지 한계점과 논의 사항이 존재한다. 첫째, 결정 공간의 관계 함수 $g(y_i, y_j) = \|y_i - y_j\|_1$을 모든 작업에 동일하게 사용하였는데, 저자 스스로도 각 작업에 최적화된 관계 함수를 찾는 것은 여전히 열린 문제(open problem)임을 언급하고 있다. 둘째, RAML-PCSVR의 경우 QP 솔버를 사용하므로 대규모 데이터셋에서의 계산 복잡도 문제가 발생할 수 있다.

## 📌 TL;DR

이 논문은 SVR 기반의 거리 학습 프레임워크에서 거리 행렬 $M$이 Positive Semidefinite (PSD) 조건을 만족하지 못해 발생하는 성능 저하 문제를 해결하였다. 이를 위해 $M$의 결합 계수에 제약을 두는 RAML-PCSVR와 RAML-NCSVR 두 가지 방법론을 제안하였으며, 이를 통해 단일 라벨, 멀티 라벨, 라벨 분포 학습 등 다양한 작업에서 기존 방식보다 우수한 변별력을 가진 거리 행렬을 직접 학습할 수 있음을 입증하였다. 이 연구는 특히 정교한 거리 척도가 필요한 복잡한 라벨링 환경의 분류 문제에 중요한 기여를 할 것으로 보인다.