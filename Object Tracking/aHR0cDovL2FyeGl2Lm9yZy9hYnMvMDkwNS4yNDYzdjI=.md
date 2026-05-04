# Generalized Kernel-based Visual Tracking

Chunhua Shen, Junae Kim, and Hanzi Wang (2009)

## 🧩 Problem to Solve

본 논문은 기존의 Kernel-based Mean Shift (MS) 트래커가 가진 두 가지 근본적인 한계점을 해결하고자 한다. 첫째, 템플릿 모델을 단일 이미지(single image)로부터만 구축할 수 있다는 점이며, 둘째, 템플릿 모델을 적응적으로 업데이트(adaptively update)하는 것이 매우 어렵다는 점이다. 이러한 한계는 조명 변화, 외형 변화, 부분 가려짐(partial occlusion)과 같은 실제 환경의 변수에 취약하게 만들어, 긴 영상 시퀀스에서 트래커가 쉽게 실패하는 원인이 된다. 따라서 본 연구의 목표는 MS 트래커를 일반화하여 보다 견고한 객체 표현 모델을 구축하고, 이를 효율적으로 업데이트할 수 있는 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 시각적 추적 문제를 이진 분류(binary classification) 문제로 재정의하고, Support Vector Machine (SVM)을 통해 학습된 판별 규칙을 MS 트래킹 프레임워크에 통합하는 것이다. 주요 기여 사항은 다음과 같다.

1.  **Probabilistic Kernel-based SVM의 통합**: Probability Product Kernels (PPK), 특히 Bhattacharyya kernel을 사용하여 SVM을 학습시키고, 이를 MS 트래킹의 비용 함수로 활용하였다. 이를 통해 기존의 단일 뷰 MS 트래킹과 비교하여 추가적인 계산 비용 없이 더 강력한 표현력을 갖게 하였다.
2.  **On-line SVM을 통한 적응적 업데이트**: On-line SVM 학습 기법을 도입하여 추적 과정 중에 타겟 모델을 실시간으로 업데이트함으로써 외형 변화에 대응할 수 있도록 하였다.
3.  **Global Mode Seeking의 이론적 확장**: 기존의 Annealed MS 알고리즘을 Continuation method의 특수한 사례로 재해석함으로써, 이를 밀도 함수뿐만 아니라 더 일반적인 비용 함수로 확장할 수 있는 이론적 근거를 제시하였다.
4.  **Scale 추정을 위한 Cascade 구조**: SVM 분류기를 활용해 타겟의 크기(scale)를 결정할 수 있는 Cascade 아키텍처 기반의 개선된 Annealed MS 알고리즘을 제안하였다.

## 📎 Related Works

기존의 시각적 추적 방식은 주로 Particle Filtering 기반의 트래커와 Kernel-based MS 트래커로 나뉜다. MS 트래커는 계산 효율성이 높지만, 대개 단일 정적 템플릿 이미지의 밀도 모델(density model)에 의존하므로 모델의 취약성이 높고 업데이트가 어렵다는 한계가 있다. 

일부 연구에서는 증분적 고유벡터 업데이트(incremental eigenvector update)나 AdaBoost, GMM(Gaussian Mixture Models) 등을 이용해 모델을 업데이트하려 시도하였으나, 영역 기반 밀도 모델(region-wise density models)을 우아하게 업데이트하는 방법은 부족했다. 또한, Avidan의 SVM 기반 트래킹 방식이 존재하지만, 이는 닫힌 형태의 해(closed-form solution)를 얻기 위해 커널의 종류를 이차 다항식 커널(quadratic polynomial kernel) 등으로 제한해야 한다는 제약이 있다. 반면, 본 논문은 L-BFGS와 같은 최적화 기법을 사용하여 커널 선택의 자유도를 높이고, MS 트래킹의 패러다임을 유지하면서 SVM의 판별 능력을 결합하였다는 점에서 차별점을 갖는다.

## 🛠️ Methodology

### 1. Probability Product Kernels (PPK) 및 SVM
본 연구에서는 확률 분포 공간에서 정의되는 Probability Product Kernels를 사용한다. 일반적인 PPK는 다음과 같이 정의된다.
$$\text{K}_{\rho}(q(x), p(x)) = \int_{X} q(x)^{\rho} p(x)^{\rho} dx$$
여기서 $\rho = 1/2$인 경우, 이는 Bhattacharyya kernel이 되며 이산 히스토그램에서는 다음과 같이 계산된다.
$$\text{K}_{1/2}(q(x), p(x)) = \sqrt{q(x)^T p(x)} = \sum_{u=1}^{m} \sqrt{q_u p_u}$$
SVM의 결정 함수 $f(x)$는 다음과 같이 표현된다.
$$f(x) = \sum_{i=1}^{N_S} \beta_i K(\hat{x}_i, x) + b$$
여기서 $\hat{x}_i$는 Support Vector, $\beta_i$는 가중치, $b$는 편향(bias)이다. PPK 기반 SVM의 특징은 테스트 단계에서 계산 복잡도가 Support Vector의 수와 무관하다는 점이다. 이는 $\sum \beta_i q(x_i)^{\rho}$ 부분을 미리 계산할 수 있기 때문이다.

### 2. Decision Score Maximization
트래킹의 목표는 SVM 결정 점수 $f(c)$의 국소 최댓값(local maximum)을 찾는 것이다. 이미지 영역의 히스토그램 $p(c)$를 적용했을 때의 비용 함수는 다음과 같다.
$$f(c) = \sum_{i=1}^{N_S} \beta_i \sum_{u=1}^{m} \sqrt{q_{i,u} p_u(c)} + b$$
이를 전개하면 다음과 같은 가중 커널 밀도 추정(weighted KDE) 형태로 변환된다.
$$f(c) = \frac{\lambda}{2} \sum_{\ell=1}^{n} \hat{w}_{\ell} k\left(\left\| \frac{c - I_{\ell}}{h} \right\|^2\right) + \Delta$$
여기서 $\hat{w}_{\ell}$은 각 픽셀의 가중치이다. 기존 MS는 $\hat{w}_{\ell}$이 항상 양수임을 가정하여 Fixed-point iteration을 사용하지만, SVM에서는 $\beta_i$가 음수일 수 있어 $\hat{w}_{\ell}$이 음수가 될 수 있다. 이 경우 Fixed-point iteration은 국소 최솟값으로 수렴할 위험이 있다. 따라서 본 논문은 Quasi-Newton 방법인 **L-BFGS 알고리즘**을 사용하여 $f(c)$를 최대화한다.

### 3. Global Optimum Seeking 및 Scale 추정
국소 최적해에 빠지는 문제를 해결하기 위해 Continuation method를 도입한다. 이는 비용 함수 $f$를 스무딩 함수 $k$와 컨볼루션하여 점진적으로 변형된 함수 $\langle f \rangle_h$를 최적화하는 방식이다.
$$\langle f \rangle_h(x) = C_h \int f(x') k\left(\left\| \frac{x - x'}{h} \right\|^2\right) dx'$$
대역폭 $h$가 클수록 함수가 더 매끄러워져 전역 최적해(global optimum)를 찾기 쉬워진다. 또한, SVM 분류기의 출력값 $\text{sign}(f(I))$를 이용하여 타겟의 크기를 결정하는 Cascade 구조를 제안한다. 큰 $h$에서 시작하여 수렴 후 점수가 음수이면 $h$를 줄여가며 다시 검색하는 방식을 취한다.

### 4. On-line Adaptation
추적 중 발생하는 외형 변화에 대응하기 위해 NORMA 알고리즘 기반의 On-line SVM을 사용한다. 추적된 영역을 양성 샘플(positive example)로, 주변 영역을 음성 샘플(negative example)로 간주하여 실시간으로 SVM 모델을 업데이트한다.

## 📊 Results

### 1. Localization 실험
CalTech-101 데이터셋의 얼굴 이미지(양성 404개, 음성 1400개)로 모델을 학습시킨 후 위치 추적 성능을 평가하였다. 실험 결과, 제안 방법은 조명 변화와 외형 차이가 큰 이미지에서도 성공적으로 얼굴을 검출하였으며, 단일 템플릿 기반의 표준 MS 트래커가 실패하는 상황에서도 강건하게 작동함을 보였다.

### 2. Tracking 실험
다양한 비디오 시퀀스(Face, Walker, Cubicle 등)에서 표준 MS 트래커 및 Particle Filter와 성능을 비교하였다.
- **정성적 결과**: 빠른 움직임, 모션 블러, 조명 변화가 심한 환경에서 제안 방법이 가장 정확한 추적 성능을 보였으며, 특히 타겟이 화면 밖으로 나갔다 돌아오는 상황에서 회복 능력이 뛰어났다.
- **정량적 결과 (Cubicle sequence 1)**:
    - **Average Tracking Error**: 제안 방법(update 적용 시)이 $6.5 \pm 2.8$ 픽셀로, MS($9.6 \pm 5.7$)와 Particle Filter($10.5 \pm 5.8$)보다 낮았다.
    - **Failure Rate (FR)**: $\text{FR}_{0.20}$ 기준, 제안 방법(update)은 $6.0\%$로 가장 낮았으며, 업데이트를 적용하지 않은 경우($28.0\%$)보다 월등히 우수한 성능을 보였다.
- **속도**: 제안 방법은 약 65 FPS로 작동하며, 이는 표준 MS와 유사한 수준이고 Particle Filter보다 훨씬 빠르다.

## 🧠 Insights & Discussion

본 논문은 단순한 밀도 거리 최소화 문제였던 MS 트래킹을 SVM 결정 점수 최대화 문제로 확장함으로써, 통계적 학습 이론을 트래킹에 성공적으로 접목하였다. 특히 PPK의 특성을 이용하여 SVM의 강력한 판별력을 확보하면서도, 추론 단계의 계산 복잡도를 MS 수준으로 유지한 점이 매우 효율적이다.

또한, $\hat{w}_{\ell}$이 음수일 수 있다는 점을 이론적으로 분석하여 L-BFGS 최적화의 필요성을 제시한 점은 학술적으로 가치가 높다. 다만, 색상(color) 특징만을 사용했을 때 scale 추정 과정에서 판별력이 부족하여 조기에 검색이 종료되는 문제가 발생할 수 있음을 언급하였다. 이는 향후 공간-특징 공간(spatial-feature space)의 확장이나 추가적인 특징 추출을 통해 해결할 수 있을 것으로 보인다.

## 📌 TL;DR

본 연구는 기존 Mean Shift 트래커의 단일 템플릿 의존성과 업데이트의 어려움을 해결하기 위해 **SVM 기반의 일반화된 커널 트래킹 프레임워크**를 제안하였다. Probability Product Kernels를 통해 효율적인 판별 모델을 구축하고, L-BFGS 최적화와 On-line SVM 업데이트를 통해 강건성과 적응성을 확보하였다. 실험 결과, 계산 효율성을 유지하면서도 기존 MS 및 Particle Filter 대비 낮은 추적 오차와 실패율을 기록하였으며, 이는 향후 실시간 객체 추적 시스템의 성능 향상에 기여할 가능성이 크다.