# Wasserstein Distance Guided Representation Learning for Domain Adaptation

Jian Shen, Yanru Qu, Weinan Zhang, Yong Yu

## 🧩 Problem to Solve

도메인 적응(Domain Adaptation)은 레이블이 풍부한 소스 도메인의 지식을 활용하여, 데이터 분포가 다르지만 관련성 있는 레이블 부족 타겟 도메인에서 고성능 학습기를 훈련하는 것이 목표입니다. 특히 비지도 도메인 적응(Unsupervised Domain Adaptation) 환경에서는 타겟 데이터의 레이블 부재와 공변량 이동(Covariate Shift)으로 인해 모델 성능 저하가 발생합니다. 기존의 적대적 도메인 적응 방법(예: DANN)은 도메인 분류기(domain classifier)가 소스/타겟 특징을 완벽하게 구별하게 되면, 특징 추출기(feature extractor)로의 기울기가 소실되는(gradient vanishing) 문제가 발생하여 효과적인 도메인 불변 특징 학습이 어려울 수 있습니다.

## ✨ Key Contributions

- **Wasserstein Distance Guided Representation Learning (WDGRL) 제안:** Wasserstein 거리를 사용하여 소스 및 타겟 도메인 특징 분포 간의 불일치를 측정하고 줄이는 새로운 적대적 표현 학습 방법을 제시합니다.
- **안정적인 기울기 특성 활용:** Wasserstein 거리가 분포 간 거리가 멀거나 겹치지 않아도 안정적인 기울기를 제공하는 특성을 활용하여, 기존 적대적 방법의 기울기 소실 문제를 효과적으로 해결합니다.
- **이론적 일반화 경계 증명:** Wasserstein 거리에 기반한 도메인 적응의 일반화 경계(generalization bound)를 이론적으로 증명하여 방법론의 타당성을 뒷받침합니다.
- **최첨단 성능 달성:** 감성 분석 및 이미지 분류 도메인 적응 벤치마크 데이터셋에서 기존의 도메인 불변 표현 학습(domain invariant representation learning) 방법들보다 뛰어난 성능을 보였습니다.

## 📎 Related Works

- **인스턴스 기반(Instance-based) 방법:** 소스 샘플에 가중치를 부여하거나 재샘플링하여 타겟 분포에 맞춥니다.
- **파라미터 기반(Parameter-based) 방법:** 공유 또는 정규화된 모델 파라미터를 통해 지식을 전이합니다.
- **특징 기반(Feature-based) 방법:** 데이터를 공통의 잠재 공간으로 매핑하여 도메인 불변 특징을 학습하며, 가장 널리 연구됩니다.
  - **MMD (Maximum Mean Discrepancy):** RKHS(Reproducing Kernel Hilbert Space)에서 두 분포의 평균 임베딩 간 거리를 최소화합니다 (예: DDC, DAN).
  - **CORAL (Correlation Alignment):** 소스 및 타겟 분포의 2차 통계량(공분산)을 정렬합니다 (예: Deep CORAL).
  - **적대적 도메인 적응(Adversarial Domain Adaptation):** 도메인 분류기와 특징 추출기 간의 미니맥스 게임을 통해 도메인 불변 특징을 학습합니다 (예: DANN). WDGRL도 이 범주에 속하지만 Wasserstein 거리를 사용합니다.
- **최적 수송(Optimal Transport):** Wasserstein 거리와 등가 관계이며, 도메인 적응 문제에 활용됩니다.
- **도메인 분리 네트워크 (DSN):** 각 도메인에 대한 사설 및 공유 표현을 명시적으로 분리하여 학습합니다.

## 🛠️ Methodology

WDGRL은 특징 추출기(Feature Extractor), 도메인 비평가(Domain Critic), 분류기(Discriminator)로 구성된 적대적 학습 프레임워크입니다.

1. **도메인 불변 표현 학습:**

   - **특징 추출기($f_g$):** 입력 $x$를 $d$차원 특징 표현 $h = f_g(x)$으로 매핑하여 도메인 불변 특징을 학습합니다.
   - **도메인 비평가($f_w$):** 특징 추출기가 생성한 소스 표현 $h^s = f_g(x^s)$와 타겟 표현 $h^t = f_g(x^t)$ 사이의 경험적 Wasserstein 거리($W_1(P_{h}^{s}, P_{h}^{t})$)를 추정합니다.
   - **Wasserstein 거리 추정:** 듀얼 형식(dual formulation)을 사용하여 비평가 손실 $L_{wd}$를 최대화함으로써 Wasserstein 거리를 추정합니다:
     $$ L*{wd}(x^s, x^t) = \frac{1}{n_s}\sum*{x^s \in X^s} f*w(f_g(x^s)) - \frac{1}{n_t}\sum*{x^t \in X^t} f_w(f_g(x^t)) $$
   - **기울기 페널티($L_{grad}$):** 비평가 $f_w$의 1-Lipschitz 제약 조건을 강제하기 위해, 소스 및 타겟 표현 사이의 임의의 점 $\hat{h}$에 대해 기울기 페널티를 적용합니다:
     $$ L*{grad}(\hat{h}) = (\left\| \nabla*{\hat{h}} f*w(\hat{h}) \right\|\_2 - 1)^2 $$
     비평가의 최적화 목표는 $\max*{\theta*w} \{L*{wd} - \gamma L\_{grad}\}$ 입니다.

2. **분류기 결합:**

   - 학습된 도메인 불변 표현이 분류에도 효과적이도록, 특징 추출기 뒤에 분류기($f_c$)를 추가합니다.
   - 분류기 손실($L_c$)은 레이블된 소스 데이터를 사용하여 교차 엔트로피로 정의됩니다:
     $$ L*c(x^s, y^s) = -\frac{1}{n_s} \sum*{i=1}^{n*s} \sum*{k=1}^l \mathbf{1}(y_i^s=k) \cdot \log f_c(f_g(x_i^s))\_k $$
   - 최종 목표 함수는 분류 손실과 Wasserstein 거리 추정의 조합입니다:
     $$ \min*{\theta_g, \theta_c} \left\{ L_c + \lambda \max*{\theta*w} [L*{wd} - \gamma L\_{grad}] \right\} $$
        여기서 $\lambda$는 판별적(discriminative) 및 전이 가능(transferable) 특징 학습 간의 균형을 조절합니다.

3. **학습 알고리즘:**
   - 표준 역전파(back-propagation)와 두 단계의 반복 학습을 사용합니다.
   - **1단계:** 도메인 비평가 네트워크를 최적 상태로 훈련합니다(기울기 상승을 통해 $\max$ 연산 최적화).
   - **2단계:** 레이블된 소스 데이터로 계산된 분류 손실과 추정된 Wasserstein 거리를 동시에 최소화하여 특징 추출기 및 분류기를 업데이트합니다. 이 과정을 통해 특징 추출기($\theta_g$)는 도메인 비평가와 분류기 손실 모두로부터 기울기를 받게 됩니다.

## 📊 Results

- **Amazon Review 및 Office-Caltech 데이터셋:** WDGRL은 감성 분석 및 이미지 분류 벤치마크에서 S-only, MMD, DANN, CORAL 등 기존 최첨단 도메인 불변 표현 학습 방법들보다 평균적으로 더 나은 성능을 달성했습니다.
  - Amazon Review 데이터셋의 12개 태스크 중 10개에서 WDGRL이 최고 성능을 보였고, 나머지 2개에서도 두 번째로 높은 점수를 기록했습니다.
  - Office-Caltech 데이터셋에서도 대부분의 태스크에서 우수한 성능을 보이며, 소규모 데이터셋에도 효과적임을 입증했습니다.
- **DANN 대비 우위:** 적대적 적응 방법인 DANN에 비해 WDGRL이 더 나은 성능을 보였는데, 이는 Wasserstein 거리의 안정적인 기울기 특성에 대한 이론적 분석과 일치합니다.
- **특징 시각화:** t-SNE를 이용한 특징 시각화 결과, WDGRL은 소스 및 타겟 도메인 클래스 간의 정렬이 더 좋고, 클래스 간 혼합 영역이 더 작아 더 전이 가능하면서도 판별적인 특징을 생성함을 보여주었습니다.

## 🧠 Insights & Discussion

- **기울기 우월성:** Wasserstein 거리는 두 특징 분포가 멀리 떨어져 있거나 저차원 매니폴드에 놓여 있는 상황에서도 안정적인 기울기를 제공하여, 기존 적대적 방법에서 발생하는 기울기 소실 문제를 극복합니다. 이는 WDGRL의 주요 장점 중 하나입니다.
- **이론적 기반:** K-Lipschitz 연속 가설 클래스에 대한 Wasserstein 거리 기반의 일반화 경계를 이론적으로 증명하여 WDGRL의 전이 가능성을 뒷받침합니다. 이는 신경망 모델이 Lipschitz 연속 함수로 구현될 수 있다는 점에서 실용적인 의미가 있습니다.
- **높은 통합성:** WDGRL은 MMD 또는 DANN을 사용하는 기존의 특징 기반 도메인 적응 프레임워크에 쉽게 통합되어 전이 가능성을 향상시킬 수 있습니다.
- **효율성 고려:** Wasserstein 거리를 추정하는 데 DANN보다 더 많은 시간이 소요될 수 있다는 점이 잠재적인 한계로 언급됩니다.

## 📌 TL;DR

본 논문은 도메인 적응을 위한 Wasserstein Distance Guided Representation Learning (WDGRL)이라는 새로운 적대적 접근 방식을 제안합니다. WDGRL은 Wasserstein 거리의 안정적인 기울기 특성을 활용하여 기존 적대적 도메인 적응 방법의 기울기 소실 문제를 해결하고, 소스 및 타겟 도메인 간의 특징 분포 차이를 효과적으로 최소화합니다. 이론적으로 일반화 경계를 보장하며, 감성 분석 및 이미지 분류 벤치마크 데이터셋에서 최첨단 성능을 달성하고, 시각화를 통해 도메인 불변적이면서 판별적인 특징을 성공적으로 학습함을 입증했습니다.
