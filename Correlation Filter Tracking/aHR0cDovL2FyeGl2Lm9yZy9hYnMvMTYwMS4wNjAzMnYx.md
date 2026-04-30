# Learning Support Correlation Filters for Visual Tracking

Wangmeng Zuo, Xiaohe Wu, Liang Lin, Lei Zhang, and Ming-Hsuan Yang (2016)

## 🧩 Problem to Solve

본 논문은 비주얼 트래킹(Visual Tracking) 분야에서 **정확도(Accuracy)와 효율성(Efficiency) 사이의 트레이드오프(Trade-off)** 문제를 해결하고자 한다. 

기존의 Support Vector Machine (SVM) 기반 트래커들은 변별력이 뛰어나 정확도가 높지만, 학습 샘플을 생성하는 샘플링(Sampling) 과정과 계산량을 줄이기 위한 버젯팅(Budgeting) 메커니즘으로 인해 실시간(Real-time) 성능을 확보하는 데 어려움이 있었다. 반면, Correlation Filter (CF) 기반 방법들은 Circulant Matrix와 Fast Fourier Transform (FFT)을 사용하여 매우 빠른 속도를 자랑하지만, 주로 Ridge Regression 기반의 예측기를 사용하여 SVM만큼의 변별력을 갖지 못하는 한계가 있었다.

따라서 본 논문의 목표는 **SVM의 강력한 변별력과 Correlation Filter의 계산 효율성을 결합하여, 정확하면서도 실시간 동작이 가능한 새로운 트래킹 알고리즘을 개발**하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **SVM 모델을 Circulant Matrix 표현식으로 재구성하여, DFT(Discrete Fourier Transform)를 이용한 효율적인 교대 최적화(Alternating Optimization) 방법으로 풀어내는 것**이다.

- **Support Correlation Filters (SCF) 제안**: SVM의 최적화 문제를 Circulant 구조로 정식화하여, 기존 SVM 기반 접근 방식의 계산 복잡도를 $O(n^4)$에서 $O(n^2 \log n)$으로 획기적으로 낮추었다.
- **효율적인 학습 알고리즘**: DFT와 IDFT(Inverse DFT)를 포함하는 교대 최적화 과정을 통해 글로벌 최적해(Global Optimal Solution)를 빠르게 찾을 수 있는 학습 절차를 제시하였다.
- **확장 모델 개발**: 기본 SCF를 넘어 다중 채널 특징을 사용하는 **MSCF(Multi-channel SCF)**, 비선형 커널을 적용한 **KSCF(Kernelized SCF)**, 그리고 스케일 변화에 대응하는 **SKSCF(Scale-adaptive KSCF)**를 제안하여 트래킹 성능을 극대화하였다.

## 📎 Related Works

논문에서는 비주얼 트래킹을 위한 외형 모델(Appearance Models)과 Correlation Filter에 대해 다음과 같이 설명한다.

1.  **Discriminative Appearance Models**: SVM, Boosting, Random Forest 등을 이용해 대상과 배경을 구분하는 방법이다. 특히 SVM 기반 트래커(예: Struck, MEEM)는 성능이 뛰어나지만 계산 비용이 높아 실시간 적용이 어렵다는 한계가 있다.
2.  **Correlation Filters (CF)**: MOSSE, KCF 등이 대표적이며, 이미지 패치의 dense sampling을 통해 생성된 Circulant Matrix의 성질을 이용하여 FFT 기반의 빠른 연산을 수행한다. 그러나 대부분 Ridge Regression 방식을 사용하여 Max-margin 기반의 SVM보다 변별력이 떨어진다.
3.  **차별점**: 기존의 Max-Margin CF (MMCF) 방식이 존재하지만, 이는 오프라인 학습 기반이며 Circulant 구조를 충분히 활용하지 않아 실시간 트래킹에 부적합하다. 본 논문은 온라인 학습 환경에서 Circulant 구조를 직접적으로 이용하여 SVM 학습 속도를 비약적으로 향상시켰다는 점에서 차별화된다.

## 🛠️ Methodology

### 1. 문제 정의 및 정식화 (Problem Formulation)
이미지 $x$의 모든 이동 버전(translated versions)으로 구성된 Circulant Matrix $\mathbf{X}$는 다음과 같이 DFT를 통해 대각화될 수 있다.
$$\mathbf{X} = \mathbf{F}^H \text{Diag}(\hat{x})\mathbf{F}$$
여기서 $\mathbf{F}$는 DFT 기저 벡터이며, $\hat{x}$는 $x$의 푸리에 변환 결과이다. 목표는 필터 $w$와 편향(bias) $b$를 학습하여 $y_i = \text{sgn}(w^T x_i + b)$로 분류하는 것이다.

본 논문은 **Squared Hinge Loss**를 사용하여 다음과 같은 SVM 모델을 정의한다.
$$\min_{w, b} \|w\|^2 + C \sum_i \xi_i^2 \quad \text{s.t. } y_i(w^T x_i + b) \geq 1 - \xi_i, \forall i$$
이 식을 Circulant 성질을 이용하여 다음과 같이 변형한다.
$$\min_{w, b} \|w\|^2 + C\|\xi\|^2_2 \quad \text{s.t. } y \circ (\mathcal{F}^{-1}(\hat{x}^* \circ \hat{w}) + b\mathbf{1}) \geq 1 - \xi$$
여기서 $\circ$는 원소별 곱셈(element-wise multiplication)을 의미한다.

### 2. 교대 최적화 알고리즘 (Alternating Optimization)
문제의 복잡도를 낮추기 위해 $\xi = e + 1 - y \circ (\mathcal{F}^{-1}(\hat{x}^* \circ \hat{w}) + b\mathbf{1})$로 치환하고, $e$와 $\{w, b\}$를 번갈아 가며 업데이트하는 방식을 제안한다.

1.  **$e$ 업데이트**: $\{w, b\}$가 고정되었을 때, $e$에 대한 최적해는 닫힌 형태(closed-form)로 존재한다.
    $$e = \max\{y \circ (\mathcal{F}^{-1}(\hat{x}^* \circ \hat{w}) + b\mathbf{1}) - 1, 0\}$$
2.  **$\{w, b\}$ 업데이트**: $e$가 고정되었을 때, $q = y + y \circ e$로 정의하면 $b$와 $w$를 다음과 같이 갱신한다.
    - $b = \text{mean}(q)$
    - $\hat{w} = \frac{\hat{x}^* \circ (\hat{q} - b\hat{1})}{\hat{x}^* \circ \hat{x} + 1/C}$

이 과정은 수렴할 때까지 반복되며, 각 반복 회차당 계산 복잡도는 $O(n^2 \log n)$이다.

### 3. 모델 확장
- **MSCF (Multi-channel SCF)**: HOG, Color Name 등 여러 채널의 특징을 사용한다. Sherman-Morrison 공식을 적용하여 다중 채널 환경에서도 $O(n^2 \log n)$의 복잡도를 유지하며 필터를 학습한다.
- **KSCF (Kernelized SCF)**: 비선형 매핑 $\psi(x)$를 도입한다. 커널 행렬 $\mathbf{K}$가 Circulant 성질을 갖는 커널(Gaussian RBF 등)을 사용하여, 커널 공간에서도 DFT를 통해 효율적으로 최적화를 수행한다.
- **SKSCF (Scale-adaptive KSCF)**: 다양한 스케일의 이미지 패치를 포함하는 Scaling Pool을 유지하고 Bilinear Interpolation을 통해 스케일 변화에 대응한다.

## 📊 Results

### 실험 설정
- **데이터셋**: 50개의 챌린징한 이미지 시퀀스로 구성된 벤치마크 데이터셋 사용.
- **평가 지표**: 20픽셀 임계값에서의 거리 정밀도(Distance Precision, DP)와 Success Plot의 곡선 아래 면적(AUC), 그리고 초당 프레임 수(FPS)를 측정한다.
- **비교 대상**: MOSSE, KCF, DCF, Struck, MEEM, TGPR 등 최신 CF 및 SVM 기반 트래커들.

### 주요 결과
- **특징 표현의 영향**: Raw Pixel보다 HOG 및 Color Name 특징을 결합했을 때 성능이 크게 향상되었다. 특히 KSCF에서 HOG+CN 조합을 사용했을 때 DP가 85.0%까지 상승하였다.
- **커널 함수의 영향**: Linear 커널보다 Polynomial, Gaussian RBF 커널을 사용했을 때 더 높은 정밀도와 AUC를 기록하였다.
- **종합 성능 (Table IV, VI)**: 
    - **SKSCF**는 DP 87.4%, AUC 62.3%로 비교 대상 중 가장 높은 정확도를 보였다.
    - **속도 면**에서 KSCF(약 51 FPS)와 SKSCF(약 83 FPS) 모두 실시간 동작이 가능함을 입증하였다.
    - 기존 SVM 기반 트래커인 MEEM(10 FPS)이나 Struck(10 FPS)보다 압도적으로 빠르면서도 더 높은 정확도를 기록하였다.
- **속성별 분석 (Table VII, VIII)**: Scale Variation(SV)과 같은 까다로운 조건에서도 SKSCF가 매우 강력한 성능을 보였다.

## 🧠 Insights & Discussion

본 논문은 **SVM의 Max-Margin 원리와 CF의 계산 효율성을 성공적으로 결합**하였다. 

- **강점**: 기존 CF 기반 방법들이 단순히 MSE(Mean Squared Error)를 최소화하는 Regression 방식이었던 것과 달리, SCF는 분류(Classification) 문제로 접근하여 대상과 배경 사이의 마진을 최대화함으로써 더 강력한 변별력을 확보하였다. 또한, 수학적 증명을 통해 제안된 교대 최적화 알고리즘이 글로벌 최적해로 수렴하며 $q$-linear 수렴 속도를 가짐을 보였다.
- **한계 및 논의**: 논문에서는 주로 단일 타겟 트래킹에 집중하고 있으며, 매우 극심한 가림(Occlusion)이나 완전히 타겟이 사라지는 상황(Out-of-view)에서의 복구 메커니즘에 대한 상세한 논의는 부족하다. 다만, SKSCF를 통해 스케일 변화 문제는 효과적으로 해결하였다.
- **비판적 해석**: SVM의 복잡한 최적화 문제를 DFT 도메인으로 옮겨 단순한 원소별 연산으로 바꾼 점이 매우 영리한 설계이다. 특히 다중 채널 확장 시 발생할 수 있는 연산량 증가 문제를 Sherman-Morrison 공식을 통해 해결하여 실용성을 높인 점이 돋보인다.

## 📌 TL;DR

이 논문은 SVM의 높은 정확도와 Correlation Filter의 빠른 속도를 결합한 **Support Correlation Filters (SCF)**를 제안한다. SVM 모델을 Circulant Matrix 구조로 재정의하고 DFT 기반의 교대 최적화 알고리즘을 적용함으로써, 계산 복잡도를 $O(n^4)$에서 $O(n^2 \log n)$으로 줄여 **실시간 SVM 트래킹**을 구현하였다. 특히 커널화(KSCF)와 스케일 적응(SKSCF)을 통해 기존의 최신 트래커들보다 뛰어난 정확도와 속도를 동시에 달성하였으며, 이는 향후 실시간 고정밀 객체 추적 시스템 구축에 중요한 기여를 할 것으로 보인다.