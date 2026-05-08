# Joint Domain Alignment and Discriminative Feature Learning for Unsupervised Deep Domain Adaptation

Chao Chen, Zhihong Chen, Boyuan Jiang, Xinyu Jin

## 🧩 Problem to Solve

기존의 심층 도메인 적응(Deep Domain Adaptation) 연구들은 주로 서로 다른 도메인 간의 분포 불일치(distribution discrepancy)를 최소화하여 공유 특징 표현을 학습하는 데 중점을 둡니다. 그러나 모든 도메인 정렬(domain alignment) 방법은 도메인 이동(domain shift)을 줄일 수는 있지만 완전히 제거할 수는 없습니다. 이로 인해 타겟 도메인의 샘플 중 클러스터 가장자리에 있거나 해당 클래스 중심에서 멀리 떨어진 샘플들은 소스 도메인에서 학습된 초평면(hyperplane)에 의해 쉽게 오분류될 수 있습니다. 이 문제를 해결하여 분류 성능을 향상시키는 것이 목표입니다.

## ✨ Key Contributions

- 심층 도메인 적응을 위해 도메인 정렬과 판별적인 심층 특징 학습을 **공동으로** 수행하려는 첫 시도입니다.
- 판별적인 심층 특징을 학습하기 위한 **인스턴스 기반(Instance-Based)** 및 **센터 기반(Center-Based)** 판별 학습 전략을 제안합니다.
- 판별적인 공유 표현의 통합이 도메인 이동을 더욱 완화하고 최종 분류에 기여하여 전이(transfer) 성능을 크게 향상시킴을 분석적이고 실험적으로 입증했습니다.

## 📎 Related Works

- **특징 기반 도메인 적응:**
  - **불일치 기반(Discrepancy-based) 방법:** MMD(Maximum Mean Discrepancy, Long et al. 2014; Tzeng et al. 2014; Long et al. 2017), CORAL(Correlation Alignment, Sun, Feng, and Saenko 2016; Sun and Saenko 2016), CMD(Center Moment Discrepancy, Zellinger et al. 2017) 등을 사용하여 도메인 불일치를 최소화합니다.
  - **재구성 기반(Reconstruction-based) 방법:** 공유 인코딩 표현을 학습하고 타겟 샘플 재구성을 수행합니다 (Ghifary et al. 2016). DSN(Domain Separation Networks, Bousmalis et al. 2016)은 각 도메인에 대한 private subspace를 도입합니다.
  - **적대적 적응(Adversarial adaptation) 방법:** 표준 minimax objective(Ganin and Lempitsky 2015; Ganin et al. 2016) 또는 대칭적 혼란(symmetric confusion) objective(Tzeng et al. 2015)를 사용합니다.
- **판별적인 특징 학습:** 대조 손실(contrastive loss, Sun et al. 2014) 및 센터 손실(center loss, Wen et al. 2016)과 같이 더욱 판별적인 특징을 학습하여 CNN 성능을 향상시키는 기법들이 있습니다.

## 🛠️ Methodology

본 논문은 도메인 정렬(CORAL)과 판별적인 특징 학습을 결합한 JDDA(Joint Domain Alignment and Discriminative Feature Learning) 방법을 제안합니다. 두 개의 스트림 CNN 아키텍처(공유 가중치)를 사용하여 소스 데이터와 타겟 데이터를 처리합니다. 전체 손실 함수는 다음과 같습니다:

$$L(\Theta|X_s,Y_s,X_t) = L_s + \lambda_1 L_c + \lambda_2 L_d$$

여기서:

- $L_s$: 소스 데이터에 대한 표준 분류 손실($c(\Theta|x_s_i, y_s_i)$)입니다.
- $L_c$: CORAL(Correlation Alignment)로 측정된 도메인 불일치 손실입니다.
  $$L_c = \text{CORAL}(H_s,H_t) = \frac{1}{4L^2} \| \text{Cov}(H_s) - \text{Cov}(H_t) \|^2_F$$
  여기서 $H_s, H_t$는 병목 계층(bottleneck layer)의 특징 벡터입니다.
- $L_d$: 제안하는 판별 손실로, 클래스 내 압축성(intra-class compactness)을 높이고 클래스 간 분리성(inter-class separability)을 향상시킵니다.

$L_d$를 위한 두 가지 방법:

1. **인스턴스 기반(Instance-Based) 판별 손실 ($L^I_d$)**:

   - 동일 클래스 샘플은 가깝게, 다른 클래스 샘플은 멀리 떨어뜨립니다.
   - 미니 배치 내의 각 소스 특징 쌍 간의 거리를 조절합니다.
   - $L^I_d = \alpha \| \text{max}(0, D_H - m_1)^2 \circ L \|_{\text{sum}} + \| \text{max}(0, m_2 - D_H)^2 \circ (1-L) \|_{\text{sum}}$
   - $D_H$는 특징 간의 쌍별 거리 행렬, $L$은 동일 클래스 여부를 나타내는 지시 행렬입니다. $m_1, m_2$는 마진 매개변수입니다.

2. **센터 기반(Center-Based) 판별 손실 ($L^C_d$)**:
   - 각 샘플과 해당 클래스 중심 간의 거리를 최소화하고, 다른 클래스 중심 간의 거리는 최대화합니다.
   - 글로벌 클래스 센터($c_y_i$)를 지속적으로 업데이트하여 클래스 내 압축성을 측정하고, 배치 클래스 센터($c_i, c_j$)를 사용하여 클래스 간 분리성을 측정합니다.
   - $L^C_d = \beta \sum_{i=1}^{n_s} \text{max}(0, \| h^s_i - c_{y_i} \|^2_2 - m_1) + \sum_{i,j=1, i \ne j}^c \text{max}(0, m_2 - \| c_i - c_j \|^2_2)$
   - 이 방법은 인스턴스 기반보다 계산 효율적이며, 각 반복에서 글로벌 정보를 고려하여 더 빠른 수렴을 유도합니다.

**학습 과정:**

- 두 방법 모두 미니 배치 SGD(Stochastic Gradient Descent)를 통해 구현됩니다.
- $L^I_d$는 모든 구성 요소가 입력에 대해 미분 가능하므로 표준 역전파로 $\Theta$를 직접 업데이트합니다.
- $L^C_d$는 $\Theta$와 "글로벌 클래스 센터"를 동시에 업데이트합니다.

## 📊 Results

- **Office-31 데이터셋 (ResNet 기반):**
  - 제안하는 JDDA는 대부분의 전이 태스크에서 모든 비교 방법(DDC, DAN, DANN, CMD, CORAL 등)보다 뛰어난 성능을 보였습니다.
  - 특히 A→W, W→A와 같은 어려운 전이 태스크에서 분류 정확도를 크게 향상시켰습니다.
  - JDDA-C(80.2%)가 JDDA-I(79.2%)보다 평균 정확도에서 약간 더 높았습니다.
- **디지털 인식 데이터셋 (수정된 LeNet 기반):**
  - JDDA는 SVHN→MNIST, MNIST→MNIST-M과 같은 대규모의 어려운 전이 태스크를 포함한 모든 태스크에서 비교 방법들을 능가했습니다.
  - JDDA-C(94.3%)가 JDDA-I(93.8%)보다 평균 정확도에서 약간 더 높았습니다.
- **특징 시각화:** 제안된 판별 손실($L_d$)을 적용한 방법들은 특징들을 더 조밀하게 클러스터링하고 더 잘 분리시켰으며, 클래스 간 간격이 더 커졌음을 t-SNE 시각화를 통해 보여주었습니다.
- **수렴 성능:** JDDA-C는 훈련 중 도메인 불변 특징의 글로벌 클러스터 정보를 고려하기 때문에 가장 빠르게 수렴하고 가장 낮은 테스트 오류에 도달했습니다.

## 🧠 Insights & Discussion

- **판별적 특징의 중요성:** 제안된 판별 손실은 공유 특징 공간에서 클래스 내 압축성과 클래스 간 분리성을 강화하여 도메인 정렬과 최종 분류 모두에 긍정적인 영향을 미칩니다.
  - 특징이 더 잘 클러스터링되면 도메인 정렬이 더 쉬워집니다.
  - 클래스 간 분리성이 향상되면 초평면과 각 클러스터 사이에 큰 마진이 생겨, 타겟 도메인의 가장자리 또는 중심에서 멀리 떨어진 샘플이 오분류될 가능성이 줄어듭니다.
- **효율성:** 센터 기반 방법(JDDA-C)이 인스턴스 기반 방법(JDDA-I)보다 계산 효율적이며 더 빠른 수렴을 보였습니다. 이는 센터 기반 방법이 미니 배치 SGD에서 "글로벌 클래스 센터"를 고려하기 때문입니다.
- **매개변수 민감도:** 판별 손실의 균형 매개변수 $\lambda_2$는 적절한 값에서 최적의 성능을 보였으며, 너무 높거나 낮으면 성능이 저하됩니다. 적절한 $\lambda_2$ 값은 도메인 정렬과 판별 특징 학습 간의 균형이 중요함을 시사합니다.
- **한계 및 미래 연구:** 현재 연구는 주로 도메인 불변 특징에 대한 판별성을 높이는 데 초점을 맞추었으며, 향후 연구에서는 정렬된 특징 공간에서 도메인 이동을 추가적으로 완화하기 위한 다른 제약 조건을 탐색할 수 있습니다.

## 📌 TL;DR

이 논문은 기존 도메인 적응 방법이 도메인 이동을 완전히 제거하지 못해 타겟 도메인 샘플의 오분류가 발생하는 문제에 주목하여, **도메인 정렬과 판별적 특징 학습을 공동으로 수행하는 JDDA(Joint Domain Alignment and Discriminative Feature Learning)를 제안합니다.** 특히 **인스턴스 기반**과 **센터 기반**의 두 가지 판별 손실을 도입하여 공유 특징의 클래스 내 압축성과 클래스 간 분리성을 향상시킵니다. 실험 결과, JDDA는 Office-31 및 디지털 인식 데이터셋에서 기존 최첨단 방법들을 능가하는 성능을 보였으며, 특히 **센터 기반 방법(JDDA-C)이 더 효율적이고 빠르게 수렴함**을 입증하여 심층 도메인 적응의 전이 성능을 크게 향상시켰습니다.
