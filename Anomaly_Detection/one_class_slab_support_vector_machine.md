# One-Class Slab Support Vector Machine

Victor Fragoso, Walter Scheirer, Joao Hespanha, Matthew Turk (2016)

## 🧩 Problem to Solve

본 논문은 **Open-set recognition(개방형 집합 인식)** 문제, 즉 학습 단계에서 보지 못한 미지의 클래스(novel classes)가 테스트 단계에서 나타났을 때 이를 정확하게 식별하는 문제를 해결하고자 한다.

일반적인 인식 시스템은 모든 클래스의 데이터를 충분히 확보했을 때는 잘 작동하지만, 미지의 클래스가 포함된 환경에서는 성능이 크게 저하된다. 특히, 타겟 클래스의 데이터는 구하기 쉽지만 부정 클래스(negative classes)의 데이터를 모두 수집하는 것은 현실적으로 불가능한 경우가 많다. 이를 위해 One-class classifier가 유용한 대안이 될 수 있으나, 기존의 **One-class SVM(OCSVM)**은 결정 경계를 단일 하이퍼플레인(hyperplane)으로 설정하여 데이터의 하한선만을 제한한다. 이로 인해 결정 점수(decision score)의 우측 꼬리 부분(right tail)에 위치한 이상치(outlier)나 미지의 클래스 샘플들이 타겟 클래스로 오분류되는 **높은 거짓 양성률(False Positive Rate)** 문제가 발생한다.

따라서 본 논문의 목표는 타겟 클래스의 정상 영역을 더 정밀하게 정의함으로써 거짓 양성률을 낮추고 미지의 클래스 탐지 정확도를 높인 **One-class Slab SVM(OC-SSVM)**을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 특징 공간(feature space)에서 타겟 클래스를 단순히 하나의 하이퍼플레인으로 구분하는 것이 아니라, **동일한 법선 벡터(normal vector)를 가진 두 개의 평행한 하이퍼플레인**을 사용하여 타겟 클래스의 정상 영역을 감싸는 **'Slab(슬랩)'** 영역을 정의하는 것이다.

기존 OCSVM이 "특정 임계값 $\rho$보다 큰 모든 영역"을 양성으로 판단했다면, OCSSVM은 "두 임계값 $\rho_1$과 $\rho_2$ 사이의 영역"에 해당하는 샘플만을 양성으로 판단한다. 이를 통해 결정 점수의 상한과 하한을 모두 제어함으로써, 타겟 클래스의 분포를 더 타이트하게 포괄하고 그 외의 영역(너무 작거나 너무 큰 점수)을 모두 미지의 클래스로 처리하여 탐지 성능을 개선한다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구와 그 한계를 언급한다.

1. **One-class SVM (OCSVM):** 원점과 데이터 사이의 마진을 최대화하는 하이퍼플레인을 학습한다. 하지만 상한선이 없기 때문에 타겟 클래스와 무관하게 매우 높은 점수를 가진 미지의 클래스 샘플을 양성으로 오분류하는 한계가 있다.
2. **SVDD (Support Vector Data Description) & One-class Kernel PCA (KPCA):** 타겟 클래스를 구형(sphere)이나 특정 분포로 캡처하려는 시도이다.
3. **Parallel Hyperplanes 접근법:** Cevikalp and Triggs의 캐스케이드 분류기나 1-vs-Set SVM 등이 평행 하이퍼플레인을 사용한 사례가 있다. 그러나 이들은 학습 단계에서 이미 알려진 부정 클래스(known negative classes)의 샘플을 사용하므로 엄격한 의미의 One-class classifier가 아니다.

본 연구의 차별점은 **오직 타겟 클래스의 샘플만을 사용**하면서도, 최적화 문제를 통해 슬랩(slab)의 크기를 자동으로 계산하여 Open-set recognition 문제를 직접적으로 해결한다는 점이다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

OCSSVM은 특징 공간에서 동일한 법선 벡터 $w$를 공유하고 서로 다른 오프셋 $\rho_1, \rho_2$를 가진 두 개의 평행한 하이퍼플레인을 학습한다. 샘플 $x$가 투영된 점수(SVM score)가 $\rho_1 \le \langle w, \Phi(x) \rangle \le \rho_2$ 범위 내에 있으면 타겟 클래스로, 그렇지 않으면 미지의 클래스로 분류한다.

### 2. 최적화 문제 (Primal Problem)

OCSSVM은 다음과 같은 convex optimization 문제를 풀어 하이퍼플레인 파라미터 $(w, \rho_1, \rho_2)$를 찾는다.

$$
\begin{aligned}
\text{minimize}_{w, \rho_1, \rho_2, \xi, \bar{\xi}} \quad & \frac{1}{2} \|w\|^2 + \frac{1}{\nu_1 m} \sum_{i=1}^m \xi_i - \rho_1 + \frac{\epsilon}{\nu_2 m} \sum_{i=1}^m \bar{\xi}_i + \epsilon \rho_2 \\
\text{subject to} \quad & \langle w, \Phi(x_i) \rangle \ge \rho_1 - \xi_i, \quad \xi_i \ge 0 \\
& \langle w, \Phi(x_i) \rangle \le \rho_2 + \bar{\xi}_i, \quad \bar{\xi}_i \ge 0, \quad i = 1, \dots, m
\end{aligned}
$$

여기서 각 변수의 역할은 다음과 같다.

- $w$: 하이퍼플레인의 법선 벡터.
- $\rho_1, \rho_2$: 하단 및 상단 하이퍼플레인의 오프셋(임계값).
- $\xi, \bar{\xi}$: 각각 하단 및 상단 경계를 벗어나는 샘플에 대한 슬랙 변수(slack variables).
- $\nu_1, \nu_2$: 슬랩의 크기를 조절하는 파라미터.
- $\epsilon$: 상단 하이퍼플레인의 페널티와 오프셋 $\rho_2$가 목적 함수에 기여하는 정도를 조절하는 파라미터.

### 3. 결정 함수 (Decision Function)

학습된 파라미터를 이용한 결정 함수 $f(x)$는 다음과 같이 정의된다.

$$
f(x) = \text{sgn} \{ (\langle w, \Phi(x) \rangle - \rho_1) (\rho_2 - \langle w, \Phi(x) \rangle) \}
$$

이 식에서 $\langle w, \Phi(x) \rangle$가 $\rho_1$보다 크고 $\rho_2$보다 작을 때만 결과값이 양수가 되어 타겟 클래스로 판정된다.

### 4. 듀얼 문제 (Dual Problem) 및 파라미터 복원

비선형 커널을 적용하기 위해 듀얼 문제로 변환하여 해결하며, 최적화된 듀얼 변수 $\alpha, \bar{\alpha}$를 통해 다음과 같이 SVM 점수와 오프셋을 계산한다.

- **SVM Score:** $s_w = \langle w, \Phi(x) \rangle = \sum_{i=1}^m (\alpha_i - \bar{\alpha}_i) k(x, x_i)$
- **오프셋 $\rho_1, \rho_2$:** KKT 조건을 통해, $0 < \alpha_i < \frac{1}{\nu_1 m}$를 만족하는 서포트 벡터들의 평균 점수로 $\rho_1$을, $0 < \bar{\alpha}_i < \frac{\epsilon}{\nu_2 m}$를 만족하는 서포트 벡터들의 평균 점수로 $\rho_2$를 계산한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋:** Letter dataset (16차원 특성 벡터), PascalVOC 2012 (HOG 특성 사용).
- **비교 대상:** OCSVM (Main Baseline), SVDD, One-class KPCA, KDE.
- **지표:** 데이터 불균형에 강건한 **Matthews Correlation Coefficient (MCC)**를 사용한다. MCC는 $-1$에서 $+1$ 사이의 값을 가지며, $+1$에 가까울수록 완벽한 예측을 의미한다.
- **커널:** Linear, RBF, Intersection, Hellinger, $\chi^2$ 커널을 사용하였다.

### 2. 주요 결과

- **Letter Dataset:** OCSSVM은 모든 커널에서 OCSVM보다 일관되게 높은 성능을 보였으며, SVDD, KPCA, KDE와 비교해서도 비슷하거나 더 우수한 성능을 기록하였다. 특히 RBF 커널에서 매우 높은 median MCC를 달성하였다.
- **PascalVOC 2012:** 고차원 데이터셋에서도 OCSSVM은 대부분의 커널(특히 additive kernels인 Hellinger, Intersection, $\chi^2$)에서 OCSVM을 능가하였으며, KPCA 및 SVDD와 경쟁력 있는 성능을 보였다. (단, RBF 커널의 경우 OCSVM이 소폭 높게 나타났다.)
- **KDE의 한계:** KDE 방식은 데이터 양이 많은 PascalVOC 데이터셋에서 메모리 및 계산 문제로 인해 실행되지 않았다.

## 🧠 Insights & Discussion

### 1. 강점 및 해석

본 연구는 타겟 클래스의 분포를 상하한으로 묶어 '슬랩' 영역을 정의함으로써, OCSVM이 가졌던 "너무 높은 점수를 가진 샘플을 양성으로 오분류하는 문제"를 효과적으로 해결하였다. 특히 additive kernels와 결합했을 때 높은 성능 향상을 보였다.

### 2. RBF 커널의 특이성 (Doughnut-like slab)

Toy dataset 실험을 통해 RBF 커널을 사용한 OCSSVM의 결정 영역을 분석한 결과, 마치 **'도넛'**과 같은 형태의 슬랩이 형성됨을 발견하였다. 이는 RBF 커널이 두 가지 극단적인 경우(정상 범위를 크게 벗어난 샘플 $\text{AND}$ 평균에 너무 가까워 밀도가 낮은 샘플)를 모두 이상치로 판단하기 때문이다. 고차원 공간으로 갈수록 데이터가 평균에서 멀어지는 경향이 있으므로, 이는 이론적으로 타당한 결과이다.

### 3. 한계 및 향후 과제

현재 구현된 Newton-based QP solver는 계산 효율성이 떨어지는 문제가 있다. 저자들은 이를 해결하기 위해 **Sequential Minimal Optimization (SMO)** 알고리즘을 OCSSVM의 추가 제약 조건에 맞게 수정하여 적용하는 것을 향후 과제로 제시하고 있다.

## 📌 TL;DR

본 논문은 기존 One-class SVM의 높은 거짓 양성률 문제를 해결하기 위해, 두 개의 평행한 하이퍼플레인으로 타겟 클래스를 감싸는 **One-class Slab SVM (OCSSVM)**을 제안하였다. 이 방법은 타겟 클래스의 결정 점수 상하한을 모두 제한함으로써 미지의 클래스 탐지 능력을 향상시킨다. Letter 및 PascalVOC 데이터셋 실험을 통해 OCSVM 대비 일관된 성능 향상을 입증하였으며, 특히 고차원 특징 공간에서도 강건한 성능을 보였다. 이 연구는 오직 양성 샘플만을 이용해 Open-set recognition 시스템을 구축하는 데 중요한 기여를 한다.
