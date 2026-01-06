# Multi-Representation Adaptation Network for Cross-domain Image Classification

Yongchun Zhu, Fuzhen Zhuang, Jindong Wang, Jingwu Chen, Zhiping Shi, Wenjuan Wu, and Qing He

## 🧩 Problem to Solve

이미지 분류는 종종 충분한 레이블 데이터를 확보하는 데 많은 비용과 시간이 소요됩니다. 이를 해결하기 위해 도메인 적응(Domain Adaptation)은 풍부한 레이블이 있는 소스 도메인에서 레이블이 부족하거나 없는 타겟 도메인으로 지식을 전이하는 매력적인 방법을 제공합니다. 그러나 기존의 도메인 적응 접근 방식은 단일 네트워크 구조로 추출된 *단일 표현(single representation)*의 분포만을 정렬하는 데 중점을 둡니다. 이러한 단일 표현은 이미지의 채도, 밝기, 색조 정보 중 일부만 포함하는 등 *부분적인 정보*만을 담을 수 있어, 다양한 시나리오에서 만족스럽지 못한 전이 학습 성능을 초래할 수 있습니다.

## ✨ Key Contributions

- **다중 도메인 불변 표현 학습:** 교차 도메인 이미지 분류를 위해 Inception Adaptation Module (IAM)을 사용하여 여러 가지 다른 도메인 불변 표현을 학습한 최초의 연구입니다.
- **새로운 MRAN(Multi-Representation Adaptation Network) 제안:** 이미지에 대한 더 많은 정보를 포함할 수 있는 여러 가지 다른 표현들의 분포를 정렬하는 네트워크를 제안했습니다.
- **MMD를 CMMD(Conditional MMD)로 확장:** 깊은 신경망에서 도메인 간 조건부 분포의 불일치를 측정하기 위해 Maximum Mean Discrepancy (MMD)를 조건부 MMD로 확장했습니다.
- **광범위한 실험:** 제안된 MRAN의 효과를 세 가지 벤치마크 이미지 데이터셋에서 검증했습니다.

## 📎 Related Works

- **이미지 분류:** 파라메트릭 및 비파라메트릭 분류기, 그리고 ResNet과 같은 최신 딥러닝 기법들이 소개되지만, 이러한 방법들은 훈련 및 테스트 세트가 동일한 분포를 따른다고 가정하여 교차 도메인 문제 해결에는 부적합합니다.
- **도메인 적응:**
  - **얕은(Shallow) 방법:** 훈련 데이터 재가중치 부여, 저차원 매니폴드 변환 등이 있습니다.
  - **깊은(Deep) 방법:** 도메인 불변 표현을 학습하며, 주로 분포 임베딩 일치(예: DDC, DAN, JAN의 MMD 기반) 또는 적대적 학습(예: RevGrad, MADA, CAN의 도메인 판별기 기반)으로 나뉩니다. 기존의 모든 딥 도메인 적응 방법은 단일 구조로 추출된 표현의 분포를 정렬하는 데 초점을 맞춥니다.
- **다중 뷰 학습(Multi-view learning):** 데이터가 여러 개의 별개의 특징 세트(‘뷰’)로 표현될 때 학습하는 문제에 중점을 둡니다. 이는 하이브리드 구조를 통해 단일 데이터 뷰에서 여러 표현을 추출하는 본 논문의 다중 표현 학습(Multi-Representation learning)과는 다릅니다.

## 🛠️ Methodology

1. **다중 표현 적응(MRA) 개념:** 기존의 단일 표현 적응 방식의 한계를 극복하기 위해, 하이브리드 신경 구조를 통해 추출된 *다중 표현*의 분포를 소스 및 타겟 도메인 간에 정렬합니다.
2. **Inception Adaptation Module (IAM):**
   - 전형적인 CNN의 전역 평균 풀링(global average pooling) 레이어를 대체하는 하이브리드 신경 구조입니다.
   - $n_r$개의 서로 다른 하위 구조(예: conv1x1 + conv5x5, conv1x1 + conv3x3 + conv3x3, conv1x1, pool + conv1x1)를 포함하여 이미지로부터 여러 표현 ($h_1 \circ g(X), \dots, h_{n_r} \circ g(X)$)을 추출합니다.
   - 추출된 다중 표현들은 서로 연결되어 분류기 $s(\cdot)$의 입력으로 사용됩니다.
3. **조건부 MMD(Conditional Maximum Mean Discrepancy, CMMD):**
   - MMD를 확장하여 _조건부 분포_ $P(x^s | y^s=c)$와 $Q(x^t | y^t=c)$ 간의 불일치를 측정합니다.
   - 타겟 도메인 데이터의 레이블이 없으므로, 네트워크의 예측 $\hat{y}_t = f(x_t)$를 가짜 레이블(pseudo-label)로 사용하여 조건부 분포를 추정합니다.
   - CMMD 공식은 다음과 같습니다:
     $$ \hat{d}_{\mathcal{H}}(X_s,X_t) = \frac{1}{C}\sum_{c=1}^{C} \left\| \frac{1}{n*s^{(c)}} \sum*{x*i^{s(c)} \in D_X^{s(c)}} \varphi(x_i^{s(c)}) - \frac{1}{n_t^{(c)}} \sum*{x*j^{t(c)} \in D_X^{t(c)}} \varphi(x_j^{t(c)}) \right\|*{\mathcal{H}}^2 $$
        여기서 $C$는 클래스 수, $n_s^{(c)}$와 $n_t^{(c)}$는 각각 소스 및 타겟 도메인의 클래스 $c$에 속하는 샘플 수입니다.
4. **MRAN(Multi-Representation Adaptation Network):**
   - IAM과 CMMD를 ResNet 기반의 종단 간(end-to-end) 딥러닝 모델에 통합합니다.
   - 전체 손실 함수는 분류 손실(cross-entropy loss)과 다중 표현에 대한 CMMD 적응 손실의 합으로 구성됩니다:
     $$ \min*f \frac{1}{n_s} \sum*{i=1}^{n_s} J(f(x_i^s), y_i^s) + \lambda \sum_i^{n_r} \hat{d}((h_i \circ g)(X_s), (h_i \circ g)(X_t)) $$
        여기서 $\lambda$는 트레이드오프 파라미터입니다.
   - ImageNet2012로 사전 학습된 모델을 사용하며, SGD(Stochastic Gradient Descent)를 통해 미세 조정(fine-tune)합니다. 학습률 및 $\lambda$는 점진적 스케줄에 따라 조정됩니다.

## 📊 Results

- **최첨단 성능 달성:** MRAN (CMMD+IAM)은 ImageCLEF-DA, Office-31, Office-Home 데이터셋에서 대부분의 전이 태스크에서 모든 비교 방법들을 능가하는 분류 정확도를 보였습니다. 특히 ImageCLEF-DA에서 상당한 성능 향상을 달성했습니다.
- **조건부 분포 적응의 중요성:** MRAN (CMMD)은 marginal 분포를 정렬하는 DAN보다 우수한 성능을 보여, 조건부 분포 적응의 효과를 입증했습니다.
- **다중 표현의 기여:** MRAN (CMMD+IAM)이 MRAN (CMMD)보다 훨씬 뛰어난 성능을 보이며, 다중 표현 정렬의 중요성을 강조했습니다.
- **특징 시각화:** t-SNE 임베딩 결과, MRAN의 결합된 표현은 DAN이나 개별 IAM 하위 구조의 표현보다 타겟 카테고리를 훨씬 명확하게 구별하는 것을 보여주었습니다.
- **분포 불일치(A-distance):** MRAN의 결합된 표현은 CNN, DAN 및 개별 IAM 하위 구조보다 더 작은 A-distance 값을 보여, 더 나은 전이성을 입증했습니다.
- **시간 복잡도:** IAM 및 CMMD 사용으로 반복당 계산 시간이 약간 증가하지만(IAM +0.025s, CMMD +0.014s), 얻어지는 성능 향상에 비하면 합리적인 수준입니다.

## 🧠 Insights & Discussion

- **다중 표현의 포괄성:** 단일 표현이 놓칠 수 있는 중요한 정보를 IAM을 통해 다중 표현으로 추출하고 정렬함으로써, 이미지에 대한 보다 포괄적인 이해를 가능하게 하여 교차 도메인 분류 성능을 크게 향상시켰습니다.
- **조건부 정렬의 효율성:** 단순히 주변 분포(marginal distribution)를 맞추는 것보다 클래스 조건부 분포를 정렬하는 것이 도메인 간 같은 클래스의 데이터 샘플이 동일한 잠재 공간에 놓이도록 유도하여 더욱 견고한 적응을 이끌어냅니다.
- **모듈화 및 확장성:** IAM은 대부분의 피드포워드 모델에 쉽게 통합될 수 있도록 설계되어, 기존 네트워크의 마지막 평균 풀링 레이어를 대체하는 방식으로 적용 가능합니다. 이는 MRAN의 일반성과 확장성을 시사합니다.
- **하이퍼파라미터 민감도:** 트레이드오프 파라미터 $\lambda$의 점진적 조정 전략은 훈련의 안정성과 모델 선택의 용이성을 제공합니다.

## 📌 TL;DR

**문제:** 기존의 도메인 적응 방법론들은 단일 네트워크 구조에서 추출된 *단일 표현(single representation)*만을 정렬하여, 이미지의 부분적인 정보만을 담아 교차 도메인 이미지 분류 성능을 제한한다.
**방법:** 본 논문은 이러한 한계를 극복하기 위해 *다중 표현(multiple representations)*을 활용하는 **Multi-Representation Adaptation Network (MRAN)**를 제안한다. MRAN은 **Inception Adaptation Module (IAM)**이라는 하이브리드 신경 구조를 도입하여 이미지로부터 다양한 측면의 다중 표현을 추출하고, 표준 MMD를 확장한 **Conditional MMD (CMMD)**를 사용하여 이 다중 표현들의 *조건부 분포*를 도메인 간에 효과적으로 정렬한다.
**발견:** MRAN (CMMD+IAM)은 ImageCLEF-DA, Office-31, Office-Home 세 가지 벤치마크 데이터셋에서 최신 도메인 적응 방법들을 능가하는 우수한 성능을 달성했다. 이는 다중 표현의 정렬과 조건부 분포 적응이 교차 도메인 이미지 분류 성능 향상에 결정적인 역할을 함을 입증한다.
