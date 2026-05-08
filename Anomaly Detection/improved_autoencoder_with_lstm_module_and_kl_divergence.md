# Improved AutoEncoder with LSTM Module and KL Divergence for Anomaly Detection

Wei Huang, Bingyang Zhang, Kaituo Zhang, Hua Gao and Rongchun Wan

## 🧩 Problem to Solve

본 논문은 이상 탐지 분야에서 널리 사용되는 두 가지 모델의 주요 문제점을 해결하고자 합니다.

- **CAE (Convolutional AutoEncoder)**: 이상 데이터에 대한 과잉 재구성(over-reconstruction) 능력이 있어, 이상 데이터의 재구성 오류가 정상 데이터와 유사하게 낮아질 수 있습니다. 이는 높은 오탐(false negative) 비율로 이어져 이상 데이터를 식별하기 어렵게 만듭니다.
- **Deep SVDD (Support Vector Data Description)**: 특징 붕괴(feature collapse) 현상이 발생할 수 있습니다. 이는 모든 데이터 포인트가 특징 공간에서 단일 지점으로 매핑되어 이상 데이터와 정상 데이터를 구별하는 능력을 상실하게 만들어 탐지 정확도를 저하시킵니다.

## ✨ Key Contributions

- **IAE-LSTM-KL(Improved AutoEncoder with LSTM module and KL divergence)**이라는 새로운 이상 탐지 모델을 제안했습니다. 이 모델은 CAE의 과잉 재구성 문제와 Deep SVDD의 특징 붕괴 문제를 동시에 해결합니다.
  - 오토인코더, SVDD 모듈, LSTM 모듈 및 KL 다이버전스 페널티의 조합을 통해 핵심적인 정상 특징을 보존하면서 이상 데이터를 걸러냅니다.
- 합성 및 실제 데이터셋(CIFAR10, Fashion MNIST, WTBI, MVTec AD)에 대한 광범위한 실험을 통해 제안된 IAE-LSTM-KL 모델이 다른 최첨단 이상 탐지 모델에 비해 **우수한 정확도와 낮은 오류율**을 보임을 입증했습니다.
- IAE-LSTM-KL 모델이 데이터셋 내의 **오염된 이상치(contaminated outliers)에 대해 향상된 견고성(robustness)**을 보여주었습니다. 훈련 세트에 이상치가 포함되어 있어도 높은 이상 탐지 능력을 유지합니다.

## 📎 Related Works

- **Convolutional AutoEncoder (CAE)** [6]: 인코더와 디코더로 구성된 재구성 기반 신경망으로, 정상 데이터를 훈련하여 이상 데이터가 더 큰 재구성 오류를 생성할 것이라고 가정합니다. VAE [7], DAE [8], VQ-VAE [9], MemAE [10]와 같은 여러 변형이 제안되었습니다.
- **Support Vector Data Description (SVDD)** [11]: 1-클래스 분류를 위한 비지도 학습 방법으로, 데이터 포인트의 대부분을 고차원 공간의 초구(hypersphere) 안에 가둡니다. 고차원 데이터셋에는 적합하지 않습니다.
- **Deep Support Vector Data Description (Deep SVDD)** [12]: SVDD를 심층 학습에 적용한 것으로, 인코더를 통해 데이터를 초구 공간으로 매핑합니다. 하지만 특징 붕괴 문제가 발생할 수 있습니다.
- **Improved AutoEncoder for Anomaly Detection (IAEAD)** [13]: CAE와 Deep SVDD를 결합하여 두 모델의 단점을 보완하고, 입력 데이터의 지역 구조를 보존하는 특징 학습을 목표로 합니다.
- 그 외 비교 모델: Gaussian AD [21], LSA (Latent Space Autoregression) [22].

## 🛠️ Methodology

제안하는 IAE-LSTM-KL 모델은 CAE, SVDD, LSTM 및 KL 다이버전스를 결합한 아키텍처입니다.

1. **모델 프레임워크**:
   - **CAE 인코더**: 입력 데이터 $x$를 잠재 특징 벡터 $z$로 압축합니다.
   - **LSTM 모듈**: 인코더와 디코더 사이에 위치합니다.
     - $z$를 입력으로 받아 정상 데이터의 특징 표현을 기억하고 저장합니다.
     - 입력 게이트($s_{i,t}$)와 출력 게이트($o_{i,t}$)를 통해 비정상적인 정보의 흐름을 걸러내 정상 정보의 순수성을 보장합니다.
     - 단일 타임 스텝 데이터($z_{i,t}$)에 대해 $h_{i,t-1}$ 및 $c_{i,t-1}$은 0으로 초기화됩니다.
     - LSTM의 출력은 $\hat{z}$입니다.
   - **SVDD 모듈**: LSTM 모듈의 출력 $\hat{z}$에 연결됩니다.
     - 정상 데이터의 잠재 특징 $\hat{z}$ 대부분을 포함하는 최소 볼륨의 초구를 구성하여 정상 샘플이 서로 가깝게 유지되도록 합니다.
   - **CAE 디코더**: LSTM 모듈의 출력 $\hat{z}$로부터 원본 데이터를 재구성합니다.
2. **KL 다이버전스 페널티**:
   - LSTM 모듈을 통과한 잠재 특징 $\hat{z}$에 적용됩니다.
   - $\hat{z}$가 표준 가우시안 분포 $N(0, I)$를 따르도록 강제합니다.
   - SVDD 모듈에서 특징 붕괴 현상을 완화하고, 잠재 특징 분포의 다양성을 확보하는 데 기여합니다.
3. **훈련 목표**: 정상 데이터만을 사용하여 전체 손실 함수 $L_{total}$을 최소화합니다.
   $$L_{total} = L_{svdd} + \lambda_1 L_{kl} + \lambda_2 L_{rec}$$
   - $L_{svdd}$: Deep SVDD 손실 (soft-boundary $L_{soft}$ 또는 hard-boundary $L_{hard}$ 버전).
     - $L_{soft} = R^2 + \frac{1}{\nu n} \sum_{i=1}^{n} \max(0, \|\hat{z}_i - c\|^2 - R^2) + \frac{\lambda_3}{2} \sum_{l=1}^{L} \|W_l\|^2_F$
     - $L_{hard} = \frac{1}{n} \sum_{i=1}^{n} \|\hat{z}_i - c\|^2 + \frac{\lambda_3}{2} \sum_{l=1}^{L} \|W_l\|^2_F$
   - $L_{kl}$: $\hat{z}$와 표준 가우시안 분포 사이의 KL 다이버전스 손실.
     - $L_{kl} = \frac{1}{n} \sum_{i=1}^{n} \text{KL}(\hat{z}_i \| N(0, I))$
   - $L_{rec}$: CAE의 재구성 손실.
     - $L_{rec} = \frac{1}{n} \sum_{i=1}^{n} \|x_i - \hat{x}_i\|^2$
   - $\lambda_1, \lambda_2, \lambda_3$는 하이퍼파라미터입니다.
4. **이상 점수 (테스트 단계)**:
   - 테스트 샘플의 특징 벡터 $\hat{z}$와 초구 중심 $c$ 간의 거리를 측정하여 이상 점수를 계산합니다.
   - $\text{SCORE} = \|\hat{z} - c\|^2$

## 📊 Results

- **데이터셋**: CIFAR10, Fashion MNIST (이미지), WTBI (시간-시계열), MVTec AD (실제 이미지).
- **평가 지표**: AUROC (Area Under the Receiver Operating Characteristic).
- **성능 비교**: IAE-LSTM-KL 모델은 모든 데이터셋에서 다른 최첨단 모델(CAE, MemAE, VAE, Gaussian AD, LSA, Deep SVDD, IAEAD)에 비해 **가장 높은 평균 AUROC 값**을 달성했습니다.
  - 예: CIFAR10에서 IAE-LSTM-KL(h)는 0.708, Fashion MNIST에서 IAE-LSTM-KL(h)는 0.931, WTBI에서 IAE-LSTM-KL(s)는 0.963, MVTec AD에서 IAE-LSTM-KL(s)는 0.828을 기록했습니다.
- **오염된 이상치에 대한 견고성**: 훈련 세트에 이상치 비율을 5%에서 25%까지 변화시키는 실험에서, IAE-LSTM-KL 모델은 다른 모델들에 비해 **일관되게 더 높은 AUROC 값**을 유지하여 훈련 데이터의 노이즈에 대한 뛰어난 견고성을 입증했습니다.
- **수렴 속도**: IAE-LSTM-KL 모델은 훈련 초기에 이미 높은 AUROC 값에 도달하고 빠르게 안정화되어, 다른 모델들보다 빠른 수렴 속도를 보였습니다.
- **제거 연구 (Ablation Studies)**: IAE-LSTM-KL 모델에서 LSTM 모듈, KL 다이버전스 또는 디코더와 같은 개별 구성 요소를 제거하면 대부분의 경우 성능이 저하되었습니다. 특히, LSTM 모듈의 입력 및/또는 출력 게이트를 제거했을 때 이상 탐지 성능이 크게 감소하여 이들 게이트의 중요성이 확인되었습니다.

## 🧠 Insights & Discussion

- LSTM 모듈을 통해 정상 데이터의 특징 표현을 기억하고, KL 다이버전스를 사용하여 특징 붕괴를 완화하는 방식이 이상 탐지 성능 향상에 매우 효과적입니다.
- LSTM의 입력 및 출력 게이트는 비정상적인 정보의 유입을 효과적으로 걸러내어, 저장된 "정상 패턴" 메모리의 순수성을 유지하는 데 결정적인 역할을 합니다. 이는 훈련 세트에 이상 데이터가 오염될 수 있는 실제 환경에서 특히 중요합니다.
- KL 다이버전스는 잠재 특징의 분포를 넓게 유지하여 SVDD 모듈이 정상과 이상 데이터를 더 잘 구별할 수 있도록 특징 붕괴를 방지합니다.
- 모델이 훈련 데이터 내의 오염된 이상치에 대해 높은 견고성을 보인다는 점은 실제 응용 분야에서 큰 장점입니다.
- 빠른 수렴 속도는 모델의 실용적인 가치를 높입니다.
- 본 연구에서는 LSTM을 시계열 데이터가 아닌 인코더 출력의 단일 타임 스텝으로 활용하여 게이팅 메커니즘을 정보 필터링 및 비정상 데이터 학습 방지에 사용한 것이 효과적인 전략이었습니다.

## 📌 TL;DR

본 논문은 오토인코더의 이상 데이터 과잉 재구성 및 Deep SVDD의 특징 붕괴 문제에 대응하기 위해 **IAE-LSTM-KL** 모델을 제안합니다. 이 모델은 인코더와 디코더 사이에 LSTM 모듈을 추가하여 정상 특징을 기억하고, LSTM 출력에 KL 다이버전스를 적용하여 잠재 특징의 분포를 다양하게 유지하며 SVDD의 특징 붕괴를 방지합니다. 다양한 데이터셋에 대한 실험 결과, IAE-LSTM-KL은 뛰어난 이상 탐지 정확도와 훈련 데이터 오염에 대한 강력한 견고성을 입증했으며, 이는 LSTM의 효과적인 정보 필터링과 KL 다이버전스의 특징 다양성 보존 역할 덕분입니다.
