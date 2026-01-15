# Domain Separation Networks

Konstantinos Bousmalis, George Trigeorgis, Nathan Silberman, Dilip Krishnan, Dumitru Erhan

## 🧩 Problem to Solve

머신러닝 알고리즘을 새로운 작업이나 데이터셋에 적용할 때 대규모 데이터 수집 및 주석(annotation) 비용이 과도하게 드는 문제가 있습니다. 합성 데이터(synthetic data)는 이러한 비용 문제를 해결할 수 있지만, 합성 데이터로 훈련된 모델은 실제 이미지에 잘 일반화되지 못하여 도메인 적응(domain adaptation) 알고리즘이 필요합니다. 기존 도메인 적응 방법들은 한 도메인에서 다른 도메인으로의 표현을 매핑하거나, 도메인 불변(domain-invariant) 특징을 추출하는 데 중점을 두었으나, 각 도메인에 고유한 특성을 명시적으로 모델링하지 못했습니다. 이로 인해 공유 표현이 도메인 관련 노이즈에 오염될 수 있습니다.

## ✨ Key Contributions

- **명시적인 도메인 분리:** 각 도메인에 고유한(private) 특징과 도메인 간에 공유되는(shared) 특징을 명시적으로 분리하여 학습하는 새로운 딥러닝 아키텍처인 Domain Separation Networks (DSN)를 제안합니다.
- **하위 공간 직교성 제약:** 사적인 특징과 공유 특징이 서로 다른 측면을 인코딩하도록 강제하기 위해 하위 공간 직교성(soft subspace orthogonality) 제약인 $L_{difference}$ 손실 함수를 도입했습니다.
- **재구성 손실:** 사적인 표현이 유용함을 유지하고 모델의 일반화 능력을 향상시키기 위해 이미지 재구성 손실($L_{recon}$)을 포함했습니다.
- **최첨단 성능 달성:** 제안된 DSN 모델은 다양한 비지도 도메인 적응 시나리오(객체 분류 및 자세 추정)에서 기존의 최첨단 방법을 능가하는 성능을 보였습니다.
- **시각화를 통한 해석 가능성:** 사적 및 공유 표현의 시각화를 제공하여 도메인 적응 과정을 해석할 수 있는 이점을 제공합니다.

## 📎 Related Works

- **도메인 적응 이론:** Ben-David et al. [4]의 도메인 적응 분류기 상한 이론 및 Mansour et al. [19]의 다중 소스 도메인 확장.
- **적대적 훈련 기반 방법:** Ganin et al. [7, 8]의 Domain-Adversarial Neural Networks (DANN)는 기울기 반전 계층(Gradient Reversal Layer, GRL)을 사용하여 도메인 불변 표현을 학습합니다.
- **MMD (Maximum Mean Discrepancy) 기반 방법:** Tzeng et al. [30]의 Deep Domain Confusion Network와 Long et al. [18]의 Deep Adaptation Network는 MMD 메트릭을 사용하여 도메인 간 특징 분포의 유사성을 최소화합니다.
- **표현 매핑 기반 방법:** Correlation Alignment (CORAL) [27]과 같은 방법은 한 도메인의 특징을 다른 도메인의 특징 공분산에 맞춰 재조정합니다.
- **사적-공유 구성 요소 분석 (Private-Shared Component Analysis):** Salzmann et al. [25] 및 Jia et al. [15]의 연구에서 영감을 받아 DSN은 표현을 사적 및 공유 하위 공간으로 분할합니다.

## 🛠️ Methodology

DSN은 공유 인코더($E_c$), 사적 소스/타겟 인코더($E_{p_s}, E_{p_t}$), 공유 디코더($D$), 분류기($G$)로 구성됩니다. 모델은 다음의 총 손실 함수를 최소화하도록 훈련됩니다:
$$L = L_{task} + \alpha L_{recon} + \beta L_{difference} + \gamma L_{similarity}$$

- **$L_{task}$ (분류 손실):** 소스 도메인에 대해서만 적용되며, 분류 작업의 정확도를 최대화합니다. 주로 교차 엔트로피 손실을 사용합니다.
  $$L_{task} = - \sum_{i=0}^{N_s} y_{s_i} \cdot \log \hat{y}_{s_i}$$
- **$L_{recon}$ (재구성 손실):** 공유 및 사적 인코더의 출력을 사용하여 원본 이미지를 재구성하며, 양쪽 도메인에 적용됩니다. 스케일 불변 평균 제곱 오차(scale-invariant mean squared error)를 사용하여 이미지의 전반적인 모양 재구성에 집중합니다.
  $$L_{recon} = \sum_{i=1}^{N_s} L_{si\_mse}(x_{s_i}, \hat{x}_{s_i}) + \sum_{i=1}^{N_t} L_{si\_mse}(x_{t_i}, \hat{x}_{t_i})$$
  $$L_{si\_mse}(x, \hat{x}) = \frac{1}{k} \|x - \hat{x}\|_2^2 - \frac{1}{k^2} ([x - \hat{x}] \cdot 1_k)^2$$
- **$L_{difference}$ (차이 손실):** 각 도메인의 공유 표현($H_c$)과 사적 표현($H_p$) 간에 소프트 하위 공간 직교성(soft subspace orthogonality)을 강제하여 서로 다른 정보를 인코딩하도록 합니다.
  $$L_{difference} = \|H_s^c{}^\top H_s^p\|_F^2 + \|H_t^c{}^\top H_t^p\|_F^2$$
- **$L_{similarity}$ (유사성 손실):** 공유 인코더에서 추출된 소스 도메인 표현($h_s^c$)과 타겟 도메인 표현($h_t^c$)을 최대한 유사하게 만듭니다. 두 가지 유형의 유사성 손실이 실험되었습니다:
  - **DANN 기반 적대적 손실:** GRL (Gradient Reversal Layer)과 도메인 분류기를 사용하여 공유 표현이 도메인을 구분하기 어렵게 만듭니다.
    $$L_{DANN}^{similarity} = \sum_{i=0}^{N_s+N_t} \{ d_i \log \hat{d}_i + (1 - d_i) \log(1 - \hat{d}_i) \}$$
  - **MMD 기반 손실:** 소스 및 타겟 도메인의 공유 특징 분포 간의 MMD를 최소화합니다. 다중 RBF 커널의 선형 조합을 사용합니다.
    $$L_{MMD}^{similarity} = \frac{1}{(N_s)^2} \sum_{i,j=0}^{N_s} \kappa(h_{s_i}^c, h_{s_j}^c) - \frac{2}{N_s N_t} \sum_{i,j=0}^{N_s,N_t} \kappa(h_{s_i}^c, h_{t_j}^c) + \frac{1}{(N_t)^2} \sum_{i,j=0}^{N_t} \kappa(h_{t_i}^c, h_{t_j}^c)$$

## 📊 Results

- **최첨단 성능 달성:** DSN w/ DANN 모델은 MNIST to MNIST-M, Synthetic Digits to SVHN, SVHN to MNIST, Synthetic Signs to GTSRB, Synthetic Objects to LINEMOD 등 모든 비지도 도메인 적응 시나리오에서 CORAL, MMD, DANN을 포함한 기존의 모든 방법을 능가했습니다. 특히, DANN을 유사성 손실로 사용하는 DSN이 MMD를 사용하는 DSN보다 더 나은 성능을 보였습니다.
- **차이 손실의 중요성:** $L_{difference}$ 손실을 제거했을 때 모든 시나리오에서 모델 성능이 일관되게 저하되어, 사적-공유 분리가 모델의 강점임을 입증했습니다.
- **재구성 손실의 효과:** 스케일 불변 평균 제곱 오차($L_{si\_mse}$) 대신 일반 평균 제곱 오차($L_{L2}^{recon}$)를 사용했을 때도 성능이 일관되게 저하되어, 스케일 불변 재구성 손실의 선택이 유효함을 확인했습니다.
- **시각화를 통한 해석:** MNIST-M 및 LINEMOD 시나리오에서 공유 및 사적 표현의 재구성 이미지를 시각화하여, 모델이 배경이나 저수준 통계와 같은 도메인 고유 특징을 사적 공간으로 효과적으로 분리하고, 객체의 모양과 같은 공유 특징을 공유 공간에 보존함을 보여주었습니다.

## 🧠 Insights & Discussion

- **명시적 분리의 이점:** DSN은 도메인 고유 특징과 공유 특징을 명시적으로 분리함으로써, 공유 표현이 도메인 불변 정보를 더 깨끗하게 학습하고 도메인 특유의 노이즈나 변화에 오염되지 않도록 합니다. 이는 기존 방법론이 공유 표현을 학습하면서 겪을 수 있는 오염 문제를 해결합니다.
- **모델 해석 가능성:** 사적 및 공유 표현의 시각화는 모델이 도메인 적응 과정에서 어떤 정보를 분리하고 어떤 정보를 보존하는지 직관적으로 이해할 수 있게 하여, 딥러닝 모델의 블랙박스 특성을 일부 해소합니다.
- **적대적 학습의 효율성:** DANN 기반 유사성 손실이 MMD 기반 손실보다 더 나은 성능을 보인 것은, 도메인 분류기에 대한 적대적 훈련이 도메인 불변 특징을 학습하는 데 더 효과적일 수 있음을 시사합니다.
- **하이퍼파라미터 튜닝의 도전:** 비지도 도메인 적응에서 하이퍼파라미터 튜닝은 여전히 어려운 문제이며, 이 연구에서는 소량의 레이블링된 타겟 도메인 데이터를 검증 세트로 사용하여 이를 해결했습니다. 이는 실제 비지도 시나리오에서의 한계점으로 볼 수 있습니다.
- **데이터셋 선택의 중요성:** Office 및 Caltech-256과 같은 기존 도메인 적응 데이터셋의 "고수준" 변이(예: 클래스 오염, 다양한 자세)에 대한 비판은, DSN이 "저수준" 이미지 통계의 차이에 초점을 맞춘 시나리오에서 특히 강점을 가짐을 시사합니다.

## 📌 TL;DR

DSN은 **도메인 적응** 문제를 해결하기 위해 **도메인 고유(private) 특징과 도메인 공유(shared) 특징을 명시적으로 분리**하는 딥러닝 모델을 제안합니다. 이 모델은 **재구성 손실**, **사적-공유 표현 간의 직교성 손실**, 그리고 **도메인 적대적 또는 MMD 기반 유사성 손실**을 조합하여 학습됩니다. 결과적으로 DSN은 **다양한 합성-실제 도메인 적응 시나리오에서 최첨단 성능을 달성**했으며, 표현의 시각화를 통해 도메인 적응 과정을 해석하는 이점을 제공합니다.
