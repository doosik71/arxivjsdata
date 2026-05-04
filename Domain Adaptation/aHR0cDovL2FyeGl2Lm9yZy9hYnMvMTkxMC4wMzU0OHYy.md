# Multi-Source Domain Adaptation and Semi-Supervised Domain Adaptation with Focus on Visual Domain Adaptation Challenge 2019

Yingwei Pan, Yehao Li, Qi Cai, Yang Chen, and Ting Yao (2019)

## 🧩 Problem to Solve

본 논문은 컴퓨터 비전 분야에서 소스 도메인(Source Domain)에서 학습된 모델을 타겟 도메인(Target Domain)으로 일반화할 때 발생하는 '도메인 간극(Domain Gap)' 문제를 해결하고자 한다. 소스와 타겟 데이터의 분포가 크게 다를 경우 모델의 성능이 급격히 저하되는 문제가 발생하며, 이를 해결하기 위해 다음 두 가지 구체적인 과제에 집중한다.

첫째, **Multi-Source Domain Adaptation (MSDA)**이다. 이는 단일 소스 도메인에서 지식을 전이하는 기존의 Unsupervised Domain Adaptation (UDA)보다 더 어렵고 실용적인 문제로, 여러 개의 소스 도메인으로부터 하나의 라벨이 없는 타겟 도메인으로 지식을 전이하는 것을 목표로 한다.

둘째, **Semi-Supervised Domain Adaptation (SSDA)**이다. 타겟 도메인에서 매우 적은 양의 라벨링된 데이터만 사용할 수 있는 상황에서, 이를 활용해 도메인 간극을 줄이고 분류 성능을 높이는 것을 목표로 한다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 픽셀 수준(Pixel-level)의 적응과 특징 수준(Feature-level)의 적응을 결합하고, 다양한 백본(Backbone) 네트워크의 앙상블 및 특징 융합을 통해 도메인 불변(Domain-invariant) 특성을 학습하는 것이다.

주요 기여 사항은 다음과 같다.

- **MSDA를 위한 하이브리드 접근법**: CycleGAN을 이용한 픽셀 수준의 데이터 생성과, End-to-End Adaptation (EEA) 및 Feature Fusion based Adaptation (FFA) 모듈을 통한 특징 수준의 적응을 순차적으로 적용하였다.
- **노이즈에 강건한 학습**: 의사 라벨(Pseudo label)의 부정확성을 해결하기 위해 일반적인 Cross Entropy 대신 Generalized Cross Entropy Loss를 도입하여 self-learning의 안정성을 높였다.
- **특징 융합 전략**: Bilinear Pooling을 통해 서로 다른 백본에서 추출된 특징들을 융합함으로써, 단일 모델보다 더 강력한 도메인 불변 분류기를 구축하였다.
- **SSDA를 위한 프로토타입 분류**: EEA 모듈에 더해 각 클래스의 대표 벡터를 정의하는 Prototype-based Classification (PC) 모듈을 추가하여 예측 성능을 보완하였다.

## 📎 Related Works

논문에서는 기존의 Unsupervised Domain Adaptation (UDA) 연구들을 언급하며, 단일 소스 도메인을 사용하는 방식의 한계를 지적한다. 실제 환경에서는 여러 소스 도메인이 존재할 가능성이 높으므로 MSDA가 더 실용적임을 강조한다.

비교 대상으로 사용된 기존 접근 방식은 다음과 같다.

- **SWD (Sliced Wasserstein Discrepancy)**, **MCD (Maximum Classifier Discrepancy)**: 도메인 간 분포 차이를 줄이려는 시도이다.
- **BSP+CDAN**, **CAN (Contrastive Adaptation Network)**: 특징 공간에서의 정렬을 통해 적응을 수행하는 최신 기법들이다.
- **TPN (Transferrable Prototypical Networks)**: 프로토타입 기반의 적응 방식을 제안한 연구로, 본 논문의 SSDA 모듈 설계에 영감을 주었다.

본 논문의 방식은 단순히 분포를 맞추는 것에 그치지 않고, CycleGAN을 통한 데이터 증강, 다중 백본 앙상블, 그리고 반복적인 의사 라벨 업데이트(Self-learning)를 결합했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. Multi-Source Domain Adaptation (MSDA)

MSDA의 전체 파이프라인은 픽셀 수준의 적응 후 특징 수준의 적응으로 이어진다.

**픽셀 수준 적응 (Pixel-level Adaptation)**:
CycleGAN을 사용하여 소스 도메인(sketch, real)의 이미지를 타겟 도메인(clipart, painting) 스타일로 변환한다. 이를 통해 $\text{sketch} \rightarrow \text{sketch}^*$, $\text{real} \rightarrow \text{real}^*$와 같이 타겟 도메인과 유사한 가상 소스 데이터를 생성하여 학습에 활용한다.

**특징 수준 적응 (Feature-level Adaptation)**:
8개의 서로 다른 백본(EfficientNet-B4~B7, SENet-154, Inception-ResNet-v2, Inception-v4, PNASNet-5)을 사용하여 초기 모델을 학습시킨 후, 다음 두 모듈을 교대로 적용한다.

- **End-to-End Adaptation (EEA) Module**:
  의사 라벨을 이용하여 백본 전체를 미세 조정(Fine-tuning)한다. 이때 타겟 데이터의 의사 라벨에 포함된 노이즈를 억제하기 위해 Generalized Cross Entropy Loss를 사용하여 학습한다. 학습 후에는 8개 모델의 예측값을 평균 내어 의사 라벨을 업데이트한다.
  
- **Feature Fusion based Adaptation (FFA) Module**:
  EEA에서 학습된 백본들로부터 특징을 추출하고, 서로 다른 두 백본의 특징을 $\text{Bilinear Pooling}$으로 융합한다.
  $$\text{Fused Feature} = \text{Feature}_A \otimes \text{Feature}_B$$
  이렇게 융합된 특징과 단일 특징을 입력으로 하는 총 36개의 분류기를 처음부터 학습시킨다. 여기서도 소스 데이터에는 Cross Entropy Loss를, 타겟 데이터에는 Generalized Cross Entropy Loss를 적용한다.

### 2. Semi-Supervised Domain Adaptation (SSDA)

SSDA는 타겟 도메인의 일부 라벨링된 데이터를 최대한 활용하는 전략을 취한다.

- **분류기 사전 학습**: 라벨링된 타겟 샘플을 10배 오버샘플링(Over-sampling)하여 소스 데이터와 함께 학습시킨다.
- **EEA 모듈**: MSDA와 유사하게 의사 라벨을 생성하고 이를 다시 학습에 사용하는 self-learning 과정을 3회 반복하여 도메인 간극을 줄인다.
- **Prototype-based Classification (PC) 모듈**:
  각 클래스별로 라벨링된 타겟 샘플들의 특징 평균을 계산하여 '프로토타입(Prototype)'을 정의한다. 타겟 샘플의 분류는 이 프로토타입들과의 거리를 측정하여 결정하는 비매개변수(Non-parametric) 방식으로 수행된다. 최종 예측은 EEA 모델들의 결과와 PC 모듈의 결과를 평균하여 산출한다.

## 📊 Results

### 1. MSDA 실험 결과

- **픽셀 수준 적응의 효과**: Table 1에서 소스 도메인의 수를 늘릴수록, 특히 CycleGAN으로 생성한 합성 도메인($\text{real}^*$)을 추가했을 때 성능이 일관되게 향상됨을 확인하였다. 최적의 백본은 EfficientNet-B7이었다.
- **EEA 모듈의 성능**: Table 2에서 EEA 모듈에 Generalized Cross Entropy를 적용했을 때, 기존 SOTA 기법들(CAN, TPN 등)보다 우수한 성능을 보였다. 특히 일반 Cross Entropy를 썼을 때보다 성능이 높게 나타나, 의사 라벨의 노이즈 제어 능력을 입증하였다.
- **FFA 모듈의 성능**: Table 3에서 특징 융합(FFA)을 적용했을 때 EEA 단독 모델보다 훨씬 높은 정확도를 기록하였다.
- **최종 테스트 셋 결과**: Table 4에 따르면, (EEA+FFA) 과정을 4회 반복하고 입력 이미지의 해상도를 높였을 때 가장 높은 성능(Mean Acc: 81.61%)을 달성하였다.

### 2. SSDA 실험 결과

- Table 5의 결과에 따르면, EEA 반복 횟수가 증가할수록 성능이 향상되었으며, 최종적으로 EEA 결과에 프로토타입 분류(PC)를 융합했을 때 가장 높은 성능(Mean Acc: 71.41%)을 보였다.

## 🧠 Insights & Discussion

본 논문은 다중 소스 도메인 적응 문제에서 단순히 모델 하나를 잘 학습시키는 것보다, **다양한 구조의 백본을 앙상블하고 이들의 특징을 융합하는 전략**이 훨씬 효과적임을 보여주었다. 특히, 특징 수준의 융합(Bilinear Pooling)이 단일 모델의 한계를 극복하고 도메인 불변 특징을 더 잘 포착하게 함을 알 수 있다.

또한, self-learning 과정에서 발생하는 '확증 편향(Confirmation Bias)' 또는 '라벨 노이즈' 문제를 해결하기 위해 **Generalized Cross Entropy Loss**를 사용한 점이 주효했다. 이는 의사 라벨이 완벽하지 않은 상황에서도 모델이 과적합되지 않고 강건하게 학습될 수 있도록 돕는다.

다만, 8개의 무거운 백본을 사용하고 이를 여러 차례 반복 학습시키며 36개의 분류기를 운용하는 방식은 **계산 비용과 메모리 소모가 매우 크다**는 한계가 있다. 실시간 추론이 필요한 환경에서는 이러한 앙상블 구조를 경량화하는 추가 연구가 필요할 것으로 보인다.

## 📌 TL;DR

본 연구는 VisDA-2019 챌린지를 위해 MSDA와 SSDA를 해결하는 시스템을 제안하였다. MSDA에서는 CycleGAN을 통한 이미지 변환과 다중 백본 앙상블 및 특징 융합(Bilinear Pooling)을, SSDA에서는 반복적 self-learning과 프로토타입 기반 분류를 적용하였다. 특히 Generalized Cross Entropy Loss를 통해 의사 라벨의 노이즈 문제를 해결함으로써 높은 성능을 달성하였으며, 이는 향후 대규모 다중 도메인 전이 학습 및 데이터 부족 상황의 도메인 적응 연구에 중요한 기초 자료가 될 것으로 평가된다.
