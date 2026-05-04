# Mutual Consistency Learning for Semi-supervised Medical Image Segmentation

Yicheng Wu, Zongyuan Ge, Donghao Zhang, Minfeng Xu, Lei Zhang, Yong Xia, Jianfei Cai (2022)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation)에서 딥러닝 모델의 성능을 최적화하기 위해서는 대량의 정밀한 픽셀/복셀 단위 어노테이션 데이터가 필요하다. 그러나 의료 영상의 어노테이션은 전문 지식을 갖춘 인력이 많은 시간을 투자해야 하므로 비용이 매우 높으며, 이로 인해 학습 데이터 부족으로 인한 과적합(Over-fitting) 문제가 빈번하게 발생한다.

특히, 제한된 데이터로 학습된 모델은 접착된 경계(Adhesive edges)나 얇은 가지(Thin branches)와 같은 모호한 영역(Ambiguous regions)에서 매우 불확실하고 오분류되기 쉬운 예측 결과를 출력하는 경향이 있다. 본 논문의 목표는 이러한 **불확실한 영역(Hard regions)을 효과적으로 활용하여, 레이블이 없는 데이터를 통해 모델의 일반화 능력을 향상시키는 반지도 학습(Semi-supervised Learning) 프레임워크를 제안하는 것**이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 모델의 **에피스테믹 불확실성(Epistemic Uncertainty)**을 추정하고, 이를 통해 식별된 어려운 영역에 대해 여러 디코더 간의 일관성을 강제하는 것이다. 이를 위해 다음과 같은 설계를 제안한다.

1.  **다양한 디코더 구조의 도입**: 하나의 공유 인코더(Shared Encoder)와 서로 약간 다른 업샘플링 전략을 가진 여러 개의 디코더를 구성하여 모델 내부의 다양성(Intra-model diversity)을 확보한다.
2.  **상호 일관성 제약(Mutual Consistency Constraint)**: 특정 디코더의 확률 출력값과 다른 디코더들의 Soft Pseudo Label 사이의 일관성을 강제함으로써, 불확실한 영역에서도 모델이 일관되고 낮은 엔트로피의 예측을 생성하도록 유도한다.
3.  **효율적인 불확실성 추정**: 기존의 MC-Dropout 방식이 많은 수의 Forward pass를 요구하는 것과 달리, 미리 정의된 다중 디코더를 통해 한 번의 pass만으로 불확실성을 추정할 수 있는 구조를 설계하였다.

## 📎 Related Works

반지도 학습의 기존 접근 방식은 크게 두 가지로 나뉜다.
- **Consistency-based models**: 입력 데이터에 작은 섭동(Perturbation)을 주어도 출력값이 변하지 않아야 한다는 평활성 가정(Smoothness assumption)에 기반한다.
- **Entropy-minimization methods**: 각 클래스의 클러스터가 조밀해야 하며 엔트로피가 낮아야 한다는 클러스터 가정(Cluster assumption)에 기반하며, 주로 Pseudo Labeling을 사용한다.

기존의 의료 영상 분할 연구(예: UA-MT, SASSNet)들은 좋은 성과를 거두었으나, 레이블이 없는 데이터 중 특히 학습이 어려운 'Challenging regions'의 효과를 충분히 활용하지 못했다는 한계가 있다. 또한, CPS 모델과 같이 사이클 일관성을 이용한 연구가 있었으나, 이는 모델 아키텍처가 동일하고 초기화 파라미터나 입력 노이즈에 의존하는 방식이다. 반면, 본 논문의 MC-Net+는 **아키텍처 수준의 차이(업샘플링 전략)**를 통해 모델 다양성을 확보했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 모델 아키텍처 (Model Architecture)
MC-Net+는 하나의 공유 인코더 $f_{\theta}^e$와 $n$개의 약간씩 다른 디코더 $f_{\theta}^i$로 구성된다. 각 서브 모델 $f_{\theta i}^{\text{sub}}$는 다음과 같이 정의된다.
$$f_{\theta i}^{\text{sub}} = f_{\theta}^e \boxplus f_{\theta i}^d, \quad i \in \{1, \dots, n\}$$
여기서 $\boxplus$는 공유 인코더와 개별 디코더의 결합을 의미한다. 모델의 다양성을 높이기 위해 서로 다른 업샘플링 전략인 **Transposed Convolutional layer, Linear Interpolation layer, Nearest Interpolation layer**를 사용하여 3개의 디코더($n=3$)를 구성한다.

### 2. 불확실성 추정 및 Sharpening
모델의 불확실성 $\mu_x$는 $n$개 디코더 출력값들의 통계적 불일치(Statistical discrepancy)로 계산된다. 또한, 엔트로피 최소화 제약을 적용하기 위해 확률 맵 $p(y_{\text{pred}}|x; \theta)$를 다음과 같은 Sharpening 함수를 통해 Soft Pseudo Label $p^*$로 변환한다.
$$p^*(y^*_{\text{pred}}|x; \theta) = \frac{p(y_{\text{pred}}|x; \theta)^{1/T}}{p(y_{\text{pred}}|x; \theta)^{1/T} + (1-p(y_{\text{pred}}|x; \theta))^{1/T}}$$
여기서 $T$는 Sharpening의 온도를 조절하는 하이퍼파라미터이다.

### 3. 학습 절차 및 손실 함수
본 모델은 지도 학습 손실(Supervised Loss)과 상호 일관성 손실(Mutual Consistency Loss)의 가중 합을 통해 학습된다.

- **상호 일관성 손실 ($L_{\text{mc}}$)**: 한 디코더의 확률 출력과 다른 디코더들의 Soft Pseudo Label 간의 MSE(Mean Squared Error)를 계산한다.
$$L_{\text{mc}} = \sum_{i,j=1, i \neq j}^n D[p^*(y^*_{\text{pred}}|x; \theta_i^{\text{sub}}), p(y_{\text{pred}}|x; \theta_j^{\text{sub}})]$$
- **전체 손실 함수**:
$$\text{Loss} = \lambda \times \sum_{i=1}^n L_{\text{seg}}(p(y_{\text{pred}}|x^l; \theta_i^{\text{sub}}), y^l) + \beta \times L_{\text{mc}}$$
여기서 $L_{\text{seg}}$는 Dice Loss이며, $\lambda$와 $\beta$는 각 손실의 균형을 맞추는 하이퍼파라미터이다. $L_{\text{mc}}$는 레이블이 있는 데이터와 없는 데이터 모두에 적용된다.

**추론(Inference) 단계**에서는 추가적인 연산 비용을 없애기 위해, 앙상블 결과가 아닌 첫 번째 디코더의 출력값만을 최종 결과로 사용한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: LA (Left Atrium, 3D MR), Pancreas-CT (3D CT), ACDC (2D MR)
- **비교 대상**: UA-MT, SASSNet, DTC, URPC, MC-Net(이전 버전)
- **지표**: Dice, Jaccard, 95HD (Hausdorff Distance), ASD (Average Surface Distance)
- **설정**: 레이블 데이터 10% 및 20% 사용 환경에서 테스트

### 2. 주요 결과
- **LA 데이터셋**: 10% 레이블 데이터만 사용했을 때, 기존 SOTA 방법들보다 높은 Dice 성능을 보였으며, 특히 20% 레이블 사용 시에는 전체 데이터를 사용한 V-Net(Upper bound)에 근접하는 성능(91.07% vs 91.62%)을 달성하였다.
- **Pancreas-CT 데이터셋**: 단일 스케일에서는 대부분의 방법보다 우수했으며, 특히 URPC의 멀티스케일 구조를 결합한 'Multi-scale MC-Net+' 버전은 모든 설정에서 최적의 성능을 기록하였다.
- **ACDC 데이터셋**: 2D 멀티 클래스 분할 작업에서도 다른 반지도 학습 모델들보다 높은 Dice 및 Jaccard 지표를 기록하였으며, Fully-supervised U-Net 대비 10% 및 20% 레이블 설정에서 각각 약 10%와 3%의 성능 향상을 보였다.

## 🧠 Insights & Discussion

### 1. 모델 설계의 효과
Ablation Study를 통해 분석한 결과, 단순히 디코더의 수를 늘리는 것보다 **다중 디코더가 유사한 결과를 생성하도록 강제하는 상호 일관성 제약(MC)**이 성능 향상에 가장 결정적인 역할을 함을 확인하였다. 또한, 서로 다른 업샘플링 전략을 사용하는 것이 모델 내부의 다양성을 높여 불확실성을 더 정확하게 추정하게 함을 입증하였다.

### 2. 하이퍼파라미터 강건성
온도 파라미터 $T$의 변화에 대해 모델 성능이 비교적 일정하게 유지되어 하이퍼파라미터 설정에 강건함을 보였다. 손실 가중치 $\lambda$의 경우, 너무 작으면 레이블 데이터 학습이 부족하고 너무 크면 일관성 제약이 약해지므로 0.5 정도의 적절한 균형점이 필요함을 확인하였다.

### 3. 한계 및 비판적 해석
본 연구는 모델 수준의 섭동(Model-level perturbation)에 집중하였으나, 데이터 수준의 섭동(Data-level perturbation)을 함께 고려하지 않았다. 저자들은 의료 영상의 특성상 일반적인 Color-Jitter 같은 기법이 부적절할 수 있다고 언급하였으나, 의료 영상에 특화된 데이터 증강 기법을 결합한다면 더 높은 성능 향상을 기대할 수 있을 것이다. 또한, 업샘플링 전략의 선택지가 제한적이라는 점이 향후 개선 과제로 남아있다.

## 📌 TL;DR

본 논문은 의료 영상 분할에서 레이블 부족 문제를 해결하기 위해 **공유 인코더와 서로 다른 업샘플링 전략을 가진 3개의 디코더를 사용하는 MC-Net+**를 제안한다. 핵심은 디코더 간의 출력 불일치로 불확실한 'Hard region'을 찾아내고, **상호 일관성 제약(Mutual Consistency)**을 통해 이 영역에서도 일관되고 확신 있는 예측을 하도록 학습시키는 것이다. 실험 결과, 2D 및 3D 의료 영상 데이터셋 모두에서 기존 SOTA 모델들을 능가하는 성능을 보였으며, 추론 시 추가 비용 없이 효율적으로 적용 가능하다는 점이 큰 강점이다.