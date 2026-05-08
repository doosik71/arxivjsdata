# A Generalized Zero-Shot Quantization of Deep Convolutional Neural Networks via Learned Weights Statistics

Prasen Kumar Sharma, Arun Abraham, and Vikram Nelvoy Rajendiran (2021)

## 🧩 Problem to Solve

딥 컨볼루션 신경망(Deep CNNs)을 리소스가 제한된 엣지 디바이스에 배포하기 위해서는 모델의 크기를 줄이고 추론 속도를 높이는 양자화(Quantization) 과정이 필수적이다. 특히, 가중치와 활성화 함수(Activation)의 부동 소수점(FP32) 표현을 고정 소수점(Fixed-point) 표현으로 변환하는 Post-Training Quantization(PTQ)은 학습 이후에 적용할 수 있어 효율적이다.

하지만 PTQ를 수행하기 위해서는 활성화 값의 범위를 결정하는 Range Calibration 과정이 필요하며, 이를 위해 일반적으로는 원본 학습 데이터셋의 일부가 필요하다. 데이터 접근이 불가능한 상황에서 이를 해결하려는 Zero-Shot Quantization(데이터 프리 양자화) 방식들이 제안되었으나, 기존의 최신 기법들은 대부분 Batch Normalization(BN) 레이어의 학습된 통계치(평균과 표준편차)에 전적으로 의존하여 활성화 범위를 추론한다.

이러한 기존 방식은 다음과 같은 치명적인 한계가 있다. 첫째, BN 레이어가 없는 네트워크(Unnormalized Networks)에서는 적용이 불가능하다. 둘째, 추론 효율성을 위해 BN 파라미터를 가중치와 편향으로 통합하는 BN Folding이 수행된 경우, 가중치의 분산이 커져 기존의 제로샷 기법들의 성능이 심각하게 저하된다. 따라서 본 논문은 BN 레이어의 존재 여부와 상관없이, 오직 사전 학습된 가중치(Pre-trained Weights)의 통계만을 이용하여 활성화 범위를 추정하는 일반화된 제로샷 양자화 프레임워크를 구축하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 BN 레이어의 통계치 대신, 모델의 사전 학습된 가중치 분포 자체가 원본 학습 데이터의 특성을 반영하고 있다는 점에 착안하여 이를 통해 활성화 값의 대체 통계치(Substitutes)를 생성하는 것이다.

주요 기여 사항은 다음과 같다.

1. **GZSQ(Generalized Zero-Shot Quantization) 프레임워크 제안**: 원본 데이터나 BN 레이어의 통계치 없이, 오직 사전 학습된 가중치만을 활용하여 활성화 범위 캘리브레이션을 위한 풍부한 데이터를 생성(Data Distillation)하는 기법을 제안하였다.
2. **Z-score 기반 손실 함수 도입**: 데이터 증류 과정에서 기존의 $L_1, L_2$ 노름이나 KL-Divergence 대신, 분포의 차이를 효율적으로 최소화할 수 있는 절대 Z-score 기반의 손실 함수를 제안하였다.
3. **범용적 적용 가능성 증명**: BN 레이어가 있는 모델뿐만 아니라, BN이 없거나 Folding된 모델에서도 높은 성능을 보임을 입증하였으며, 이미지 분류, 객체 탐지, 의료 영상 분석, 이미지 디레이닝(De-raining) 등 다양한 도메인에서 효용성을 검증하였다.

## 📎 Related Works

기존의 모델 압축 연구는 Pruning, Knowledge Distillation, Matrix Factorization 등으로 나뉜다. 특히 양자화 분야에서는 학습 중에 양자화 오차를 줄이는 Quantization-Aware Training(QAT)이 가장 성능이 좋으나, 전체 학습 데이터셋이 필요하고 시간과 자원 소모가 크다는 단점이 있다.

데이터 의존성을 줄이기 위한 연구로는 제한된 데이터를 사용하는 방식과 완전히 데이터가 필요 없는 Zero-Shot 방식이 있다. DFQ는 가중치 균등화(Weight Equalization)와 편향 보정(Bias Correction)을 사용하지만, 활성화 범위 추정을 위해 BN의 $\beta$와 $\gamma$ 파라미터에 의존한다. ZEROQ는 BN 통계치와 일치하는 가짜 데이터를 생성하는 Data Distillation 방식을 사용한다.

그러나 이러한 방식들은 모두 BN 레이어가 없거나 Folding된 모델에서는 작동하지 않는다. 최근 연구 흐름은 BN을 대체하는 Unnormalized 네트워크로 이동하고 있으므로, BN에 의존하지 않는 GZSQ의 접근 방식은 미래의 딥러닝 모델 양자화에 있어 중요한 차별점을 가진다.

## 🛠️ Methodology

GZSQ 프레임워크는 크게 세 단계의 파이프라인으로 구성된다.

### 1. 통계치 추정 (Statistics Estimation, SE)

원본 데이터 없이 각 레이어의 활성화 값에 대한 대체 통계치 $\mathcal{A} = \{(\mu_{a_1}, \sigma_{a_1}), \dots, (\mu_{a_N}, \sigma_{a_N})\}$를 계산한다. 중심극한정리(Central Limit Theorem)에 의해 가중치가 가우시안 분포 $N(\mu, \sigma)$를 따른다고 가정하며, 입력 데이터 $J$ 역시 $N(0, 1)$을 따른다고 가정한다.

가중치 $W_{n+1}$과 이전 레이어의 활성화 $f_n$이 모두 가우시안 분포를 따를 때, 두 가우시안 분포의 컨볼루션 결과 역시 가우시안 분포를 따른다는 성질을 이용한다. 편향 $b=0$일 때, $n+1$번째 레이어의 대체 평균 $\mu_{a_{n+1}}$과 표준편차 $\sigma_{a_{n+1}}$은 다음과 같이 재귀적으로 계산된다.

$$\mu_{a_{n+1}} = \mu_{W_{n+1}} + \mu_{a_n}$$
$$\sigma_{a_{n+1}} = \sqrt{\sigma_{W_{n+1}}^2 + \sigma_{a_n}^2}$$

### 2. 경험적 통계 조정 (Empirical Statistics Adjustment, ESA)

실제 네트워크에서는 레이어 간 채널 수($C$)가 동일하지 않은 확장(Expansion) 또는 축소(Contraction) 구조가 빈번하다. GZSQ는 채널 수가 불일치할 때, $\{\min, \text{mean} \pm \min, \text{mean}, \max \pm \text{mean}, \max\}$와 같은 경험적 값들의 집합을 사용하여 부족한 채널의 통계치를 보간한다. 예를 들어, MobileNetV2의 경우 축소 구조에서 표준편차 값을 $\text{mean}(\sigma_{a_n}) + \min(\sigma_{a_n})$으로 설정하는 것이 효과적임을 발견하였다.

### 3. 데이터 증류 (Data Distillation, DD)

추정된 대체 통계치 $\mathcal{A}$를 타겟으로 하여, 이와 가장 유사한 분포를 갖는 가짜 데이터 $\hat{y}$를 생성한다. 이때 단순한 거리 기반의 손실 함수보다 분포의 특성을 더 잘 반영하는 절대 Z-score 기반 손실 함수 $L_Z$를 사용한다.

$$L_Z(u, v) = \frac{\|\mu_u - \mu_v\|}{\sqrt{(\sigma_u + s)^2 + (\sigma_v + s)^2}}$$

여기서 $s = 1e-6$은 0으로 나누는 것을 방지하기 위한 상수이다. 최종 손실 함수 $L_D$는 모든 레이어의 통계치 차이와 입력 데이터의 가우시안 분포 유지 조건을 합산하여 정의한다.

$$L_D = \left[ \sum_{n=1}^{N} L_Z(f_n, a_n) \right] + L_Z(y, N(0, 1))$$

최종적으로 $\arg \min_{\hat{y}} L_D$를 통해 최적화된 $\hat{y}$를 생성하며, 이 데이터를 모델에 통과시켜 활성화 값의 실제 범위를 측정하고 양자화 파라미터 $\Delta$와 $z$를 결정한다.

### BN Folding 처리

BN Folding이 수행되면 가중치와 편향에 추가적인 바이어스가 도입된다. GZSQ는 이를 반영하여 통계 추정 단계에서 다음과 같이 수정된 평균을 계산함으로써 성능 저하를 막는다.

$$\mu_{a_n} = \mu_{W_{n,fold}} + \mu_{a_{n-1}} + b_{n,fold}$$

## 📊 Results

### 실험 설정

- **데이터셋**: CIFAR-10, ImageNet, MS COCO (객체 탐지), 흉부 X-ray (폐렴 분류).
- **모델**: ResNet, MobileNetV2, ShuffleNet, InceptionV3, SqueezeNet 및 BN이 없는 Fixup-ResNet, ISONet.
- **비교 대상**: RVQuant, DFQ, OCS, ACIQ, GDFQ, DFC, ZEROQ.
- **지표**: Top-1 Accuracy, mAP (객체 탐지), NIQE/BRISQUE (이미지 품질).

### 주요 결과

1. **BN 없는 모델 (Fixup-ResNet, ISONet)**: 기존 ZEROQ는 BN이 없을 때 일반 가우시안 입력과 유사하게 동작하여 성능이 낮았으나, GZSQ는 FP32 정밀도에 매우 근접한 정확도를 달성하였다.
2. **BN 있는 모델 (ImageNet)**: MobileNetV2, ResNet18 등 다양한 모델에서 ZEROQ 및 OCS보다 우수한 성능을 보였으며, 특히 SqueezeNetV5에서는 약 1.25%의 정확도 향상을 보였다.
3. **객체 탐지 (RetinaNet)**: MS COCO 데이터셋에서 36.3 mAP를 기록하여 ZEROQ와 대등한 수준의 견고함을 보였으며, 가우시안 입력(35.2 mAP)보다 훨씬 뛰어난 결과를 얻었다.
4. **의료 영상 및 디레이닝**: 폐렴 분류 작업에서 ZEROQ와 가우시안 입력을 유의미하게 상회하였으며, 이미지 디레이닝 작업에서도 NIQE 및 BRISQUE 지표에서 FP32 모델에 가장 근접한 성능을 기록하였다.

## 🧠 Insights & Discussion

### 강점 및 해석

GZSQ의 가장 큰 강점은 모델의 '구조적 정보(Structural Information)'보다는 '분포적 특성(Distributional Properties)'을 복원하는 데 집중했다는 점이다. 실험 결과, $L_2$ 노름 기반의 손실 함수는 데이터의 시각적 구조를 어느 정도 유지하지만, 제안된 $L_Z$ 손실 함수는 시각적 형태는 덜 복원하더라도 활성화 값의 통계적 분포를 더 정확하게 일치시켜 최종 양자화 정확도를 극대화한다.

### 한계 및 비판적 논의

논문에서는 KL-Divergence를 사용한 베이스라인($GZSQ-L^*_{KL}$)을 비교하였으나, 대체 통계치의 평균값이 음수가 될 수 있어 KL-Divergence를 완전히 활용하지 못하고 표준편차만을 사용했다는 점을 언급한다. 이는 방법론적 제약으로 작용하며, 추후 평균값을 포함하여 KL-Divergence를 온전히 활용했을 때의 결과에 대한 비교가 추가될 필요가 있다. 또한, 데이터 증류 과정에서의 최적화 수렴 속도나 하이퍼파라미터 민감도에 대한 상세 분석이 부족하다.

## 📌 TL;DR

본 논문은 BN 레이어의 통계치 없이 **사전 학습된 가중치의 통계만을 이용해 활성화 범위를 추정**하는 **GZSQ** 프레임워크를 제안한다. 가중치 분포로부터 대체 통계치를 계산하고, **Z-score 기반의 새로운 손실 함수**를 통해 최적의 가짜 데이터를 증류하여 양자화에 활용한다. 이를 통해 BN이 없거나 Folding된 모델에서도 기존 제로샷 기법보다 월등한 성능을 보였으며, 이는 향후 BN을 사용하지 않는 차세대 신경망의 효율적인 배포를 가능하게 하는 중요한 기반 연구가 될 것이다.
