# Metric Learning in an RKHS

Gokcan Tatli, Yi Chen, Blake Mason, Robert Nowak, Ramya Korlakai Vinayak (2025)

## 🧩 Problem to Solve

본 논문은 "항목 $h$가 항목 $i$와 $j$ 중 어느 것과 더 유사한가?"라는 형태의 triplet comparison(삼조 비교) 데이터를 통해 Reproducing Kernel Hilbert Space(RKHS) 상에서의 거리 함수(metric)를 학습하는 문제를 다룬다. 

전통적인 거리 학습은 유클리드 공간에서의 선형 Mahalanobis 거리 학습에 집중해 왔으나, 실제 인간의 지각 능력이나 복잡한 데이터 간의 유사성은 고차원적인 상호작용을 포함하므로 비선형적인 거리 표현이 필수적이다. 따라서 본 연구의 목표는 커널 방법을 이용하여 비선형 거리 학습을 수행하는 일반적인 RKHS 프레임워크를 구축하고, 이에 대한 이론적 기초인 일반화 오차(generalization error) 및 샘플 복잡도(sample complexity)의 상한선을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 비선형 거리 학습, 특히 커널화된 거리 학습에 대한 이론적 분석을 최초로 제공했다는 점에 있다. 주요 기여 사항은 다음과 같다.

- **이론적 보장:** triplet comparison으로부터 학습된 커널 거리 학습의 일반화 오차와 샘플 복잡도에 대한 수학적 보장(guarantee)을 수립하였다.
- **정규화 분석:** Schatten $p$-norm을 이용한 정규화가 샘플 복잡도와 일반화 경계에 어떠한 영향을 미치는지 분석하여, 모델의 유연성과 편향-분산 트레이드오프를 조절하는 방법을 제시하였다.
- **선형 프레임워크 확장:** 기존의 선형 거리 학습 연구들이 가졌던 제약 조건(예: 데이터 개수 $n$이 차원 $d$보다 커야 함)을 극복하고, 이를 무한 차원 RKHS 환경으로 확장하여 더 일반적인 이론적 틀을 완성하였다.

## 📎 Related Works

거리 학습(Metric Learning)은 이미지 검색, 얼굴 인식, 추천 시스템 등 다양한 분야에서 활용되어 왔다.

- **선형 거리 학습:** Mahalanobis 거리 학습에 관한 연구들이 다수 존재하며, 특히 triplet comparison을 이용한 선형 거리 학습의 일반화 오차 경계가 연구된 바 있다. 그러나 이러한 접근 방식은 데이터가 선형적으로 분리 가능하거나 저차원일 때만 유효하다는 한계가 있다.
- **비선형 및 딥러닝 접근법:** 커널 방법론이나 Siamese Network와 같은 딥러닝 기반의 비선형 거리 학습이 실무적으로는 널리 사용되고 있으나, 이에 대한 엄밀한 이론적 분석(특히 샘플 복잡도 측면)은 매우 부족한 상태였다. 
- **차별점:** 본 논문은 단순한 성능 향상이 아니라, RKHS라는 수학적 구조 내에서 거리 학습의 수렴성과 일반화 성능을 이론적으로 증명함으로써, 경험적(empirical)으로만 알려졌던 비선형 거리 학습에 학술적 근거를 제공한다.

## 🛠️ Methodology

### 1. 시스템 구조 및 거리 정의
본 논문에서는 데이터를 $\mathbb{R}^d$에서 RKHS $\mathcal{H}$로 매핑하는 특징 맵(feature map) $\phi$를 가정한다. 이때 커널 함수 $k(x, y) = \langle \phi(x), \phi(y) \rangle_{\mathcal{H}}$가 정의된다. 학습하고자 하는 비선형 Mahalanobis 거리는 유계 선형 연산자(bounded linear operator) $L: \mathcal{H} \to \mathcal{H}$를 이용하여 다음과 같이 정의된다.

$$d_L^2(x_i, x_j) = \|L\phi(x_i) - L\phi(x_j)\|^2_{\mathcal{H}}$$

### 2. 학습 목표 및 손실 함수
학습의 목표는 주어진 triplet $\{x_h, x_i, x_j\}$에 대하여 인간의 판단 결과인 레이블 $y_t \in \{\pm 1\}$를 가장 잘 예측하는 $L$을 찾는 것이다. 0/1 손실(misclassification probability)을 직접 최소화하는 것은 불가능하므로, 이를 $\alpha$-Lipschitz 연속인 볼록 손실 함수 $\ell$로 완화한 경험적 위험(empirical risk) $\hat{R}_S(L)$를 최소화한다.

$$\hat{R}_S(L) = \frac{1}{|S|} \sum_{(t, y_t) \in S} \ell(y_t( \|L\phi_h - L\phi_i\|^2_{\mathcal{H}} - \|L\phi_h - L\phi_j\|^2_{\mathcal{H}} ))$$

### 3. 정규화 및 모델 클래스
모델의 복잡도를 제어하기 위해 $L$의 Schatten $p$-norm을 제한하는 정규화를 도입한다.
- **Schatten 2-norm (Hilbert-Schmidt norm):** $\|L^\dagger L\|_{S_2} \leq \lambda_F$로 제한하여 유효 차원을 제어한다.
- **Schatten 1-norm (Nuclear norm):** $\|L^\dagger L\|_{S_1} \leq \lambda^*$로 제한하여 $L$이 저차원(low-rank) 구조를 갖도록 유도한다.

### 4. 구현 방법: KPCA를 통한 유한 차원 최적화
무한 차원 공간 $\mathcal{H}$에서의 최적화 문제를 해결하기 위해, 본 논문은 Kernel Principal Component Analysis(KPCA)를 활용하여 문제를 유한 차원의 양의 준정부호(PSD) 행렬 $M \in \mathbb{R}^{n \times n}$을 찾는 문제로 변환한다.

- **KPCA 매핑:** Gram 행렬 $K$를 생성하고 중심화(centering)한 뒤, 고유벡터를 통해 데이터를 $\mathbb{R}^n$ 공간의 벡터 $\phi_i$로 변환한다.
- **최적화 문제:** 다음과 같은 볼록 최적화 문제(Convex Program)를 푼다.
$$\min_{M \succeq 0} \hat{R}_S(M) \quad \text{s.t. } \|M\| \leq \lambda$$
여기서 $\|M\|$은 선택한 Schatten norm(Frobenius 또는 Nuclear norm)이다. 학습된 $M$의 Cholesky 분해를 통해 최종적으로 RKHS 상의 연산자 $\hat{L}_0$를 복원할 수 있다.

## 📊 Results

### 1. 실험 설정
- **시뮬레이션:** 2D Spiral 데이터(측지선 거리 학습) 및 Gaussian Kernel Map 데이터를 사용하였다.
- **실제 데이터셋:** Food-100 데이터셋(100가지 음식, 약 19만 개의 triplet)을 사용하였으며, AlexNet의 전층(antepenultimate layer) 임베딩을 입력으로 사용하였다.
- **지표:** Train/Test Accuracy를 통해 일반화 성능을 측정하였다.

### 2. 주요 결과
- **커널 함수 비교:** Food-100 데이터셋 실험 결과, **Gaussian 커널**이 가장 우수한 성능을 보였으며, 그 뒤를 Polynomial, Laplacian 커널이 이었다. Linear 및 Sigmoid 커널은 상대적으로 낮은 성능을 기록하였다.
- **샘플 복잡도 검증:** 시뮬레이션 결과, 학습 triplet의 수가 증가함에 따라 Train accuracy와 Test accuracy의 간격이 줄어드는 것이 확인되었다. 이는 본 논문의 Theorem 1, 2에서 제시한 일반화 오차 상한선이 실제 데이터에서도 유효함을 시사한다.
- **차원 및 랭크의 영향:** 저차원 매니폴드에 존재하는 거리 함수를 학습할 때, Schatten 1-norm(Nuclear norm) 제약 조건이 Schatten 2-norm보다 적은 샘플 수로도 높은 정확도에 도달하게 함을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 이론적 의의
본 논문은 그동안 경험적으로만 사용되었던 비선형 거리 학습에 대해 **최초로 일반화 오차 및 샘플 복잡도 보장**을 제공하였다. 특히, 기존 선형 거리 학습 이론의 한계였던 $n > d$ 조건을 제거함으로써 고차원 데이터나 무한 차원 RKHS에서도 이론적 분석이 가능함을 증명한 점이 매우 고무적이다.

### 한계 및 비판적 해석
- **계산 복잡도:** KPCA 과정의 시간 복잡도가 $O(n^3)$으로 매우 높다. 저자들은 Nyström method를 통해 이를 $O(nm^2)$으로 완화하여 해결하였으나, 여전히 대규모 데이터셋에 적용하기에는 계산 비용이 크다.
- **커널 선택의 의존성:** 실험 결과에서 보이듯 성능이 커널 함수의 종류와 파라미터에 크게 의존한다. 하지만 최적의 커널을 찾는 체계적인 방법론보다는 교차 검증(cross-validation)에 의존하고 있다는 점이 아쉽다.
- **가정의 단순함:** 데이터가 특정 분포에서 i.i.d.로 샘플링되었다는 가정이 실제 복잡한 데이터 분포에서 얼마나 유지될지는 추가적인 연구가 필요하다.

## 📌 TL;DR

본 논문은 **비선형 거리 학습(Kernelized Metric Learning)을 위한 RKHS 기반의 이론적 프레임워크를 제안**하고, triplet comparison 데이터를 통한 학습의 **일반화 오차와 샘플 복잡도의 수학적 상한선을 도출**하였다. 무한 차원의 문제를 KPCA를 통해 유한 차원의 볼록 최적화 문제로 변환하여 실용적인 구현 방법을 제시하였으며, 이를 통해 데이터의 유효 차원이 낮을수록 적은 샘플로도 효율적인 학습이 가능함을 입증하였다. 이 연구는 향후 딥러닝 기반 거리 학습의 이론적 분석을 위한 기초 토대가 될 가능성이 높다.