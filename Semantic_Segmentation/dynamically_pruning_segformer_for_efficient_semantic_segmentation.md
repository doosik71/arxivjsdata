# DYNAMICALLY PRUNING SEGFORMER FOR EFFICIENT SEMANTIC SEGMENTATION

Haoli Bai, Hongda Mao, Dinesh Nair

## 🧩 Problem to Solve

SegFormer는 컴퓨터 비전 태스크, 특히 의미론적 분할(semantic segmentation)에서 뛰어난 성능을 보이지만, Transformer 기반 모델의 높은 연산 비용으로 인해 모바일 기기와 같은 엣지 디바이스에 배포하기 어렵다는 문제가 있습니다. Mix Transformer (MiT) 인코더 내의 넓고 조밀한 선형 레이어가 이러한 높은 연산 부담의 주원인입니다. 본 논문은 SegFormer의 연산 효율성을 높여 경량화하는 것을 목표로 합니다.

## ✨ Key Contributions

* **동적 게이팅 선형 레이어(Dynamic Gated Linear Layer, DGL) 제안**: 입력 인스턴스에 따라 정보 가치가 낮은 뉴런을 식별하고 가지치기(pruning)하여 동적으로 연산을 줄이는 새로운 모듈을 제안했습니다.
* **2단계 지식 증류(Two-stage Knowledge Distillation) 도입**: 원본 SegFormer(교사 모델)의 지식을 가지치기된 학생 네트워크로 효과적으로 전이하기 위해 2단계 지식 증류 전략을 적용했습니다.
* **상당한 연산량 감소 및 성능 유지**: 제안된 방법은 SegFormer의 연산 오버헤드를 60% 이상 줄이면서도 mIoU 성능 하락을 0.5% 이내로 최소화했습니다.
* **뉴런 유형별 분석 및 선택적 가지치기 입증**: 뉴런의 중요도를 유형별로 분석하고, 동적 게이팅 모듈이 중요 뉴런은 유지하고 비정보성 뉴런을 선택적으로 가지치기 함을 시각적으로 입증했습니다.

## 📎 Related Works

* **Vision Transformers (ViT) [1]**: 이미지 인식 분야에서 Transformer 아키텍처의 성공을 가져온 초기 연구.
* **SegFormer [6]**: 계층적 특징 표현 추출과 효율적인 MLP 디코더를 사용한 의미론적 분할 Transformer.
* **CNN 기반 의미론적 분할 모델 [7, 8, 9, 10, 11]**: FCN, DeepLabV3+, PSPNet 등 기존의 컨볼루션 기반 모델들.
* **지식 증류(Knowledge Distillation) [15, 18]**: BERT 양자화(quantization) 등 Transformer 기반 모델의 압축에 효과적임이 입증된 기법.
* **동적 가지치기(Dynamic Pruning) [17]**: CNN에서 입력에 따라 채널을 동적으로 가지치기하는 이전 연구. 본 논문은 $W_1$ 파라미터 정보를 게이트 예측에 활용하여 차별점을 둠.

## 🛠️ Methodology

본 논문은 SegFormer의 MiT 인코더 내 중복 뉴런을 동적으로 가지치기하기 위해 다음과 같은 방법을 제안합니다:

1. **동적 게이팅 선형 레이어 (Dynamic Gated Linear Layer, DGL)**:
    * **게이트 예측(Gate Prediction)**: 입력 $X \in \mathbb{R}^{N \times C}$가 주어졌을 때, DGL 레이어는 경량 게이트 예측기 $g_\phi(\cdot)$를 통해 인스턴스별 게이트 $M = g_\phi(X) \in \mathbb{R}^{N \times \bar{C}}$를 계산합니다.
    * **입력 요약**: 게이트 예측기 $g_\phi(\cdot)$의 입력은 $X$의 $N$개 이미지 패치에 대해 평균 풀링(AvgPool)을 수행한 후, 레이어 정규화(Layer-Normalization, LN)를 적용하여 얻은 $\hat{X} = \text{LN}(\text{AvgPool}(X)) \in \mathbb{R}^{C}$입니다.
    * **마스크 생성**: $\hat{X}$와 현재 선형 레이어의 파라미터 $W_1$을 2계층 MLP에 입력하여 로짓 $G = \text{MLP}(\hat{X}W_1) \in \mathbb{R}^{\bar{C}}$를 얻습니다.
    * **상위 $r\%$ 선택**: 로짓 $G$에서 상위 $r\%$의 가장 큰 요소만 유지하고 나머지는 0으로 만드는 $\text{Top-r}(\cdot)$ 연산을 통해 마스크 $M$을 최종 결정합니다.
    * **연산 감소**: 이 마스크 $M$은 현재 레이어의 출력 차원과 다음 레이어의 입력 차원에 적용되어 연산량을 줄입니다: $Y = X(W_1 \circ M)$, $Z = Y(W_2 \circ M^>)$.
    * **점진적 희소성 증가(Sparsity Annealing)**: 희소성 비율 $r$을 훈련 스텝 $t$에 따라 $r_t = r \min(1, t/T)$로 점진적으로 증가시켜 부드러운 전환을 유도합니다.
    * **희소성 정규화**: 정보 손실을 보완하고 희소성을 장려하기 위해 $G$에 대한 $\ell_1$ norm 페널티 $L_m = \lambda_m \sum \|G\|_1$를 적용합니다.
    * **적용 위치**: MiT 인코더 내의 $Q, K, V$ 계산 및 Mix-FFN의 중간 레이어에 DGL을 적용합니다. MLP 디코더에서는 스테이지별 특징 맵 $\{X_i\}_{i=1}^4$의 연결(concatenation)을 덧셈(addition)으로 대체하여 연산을 추가로 줄입니다.

2. **2단계 지식 증류 (Two-stage Knowledge Distillation)**:
    * **1단계 (피처 맵 증류)**: 학생 모델의 중간 특징 맵 $\tilde{X}_i$와 교사 모델의 특징 맵 $X_i$ 간의 평균 제곱 오차(MSE)를 최소화합니다: $\mathcal{L}_1 = \sum_{i=1}^{4} \text{MSE}(\tilde{X}_i, X_i)$.
    * **2단계 (최종 로짓 증류)**: 학생 모델의 최종 로짓 $\tilde{Y}$와 정답 $Y^*$ 간의 교차 엔트로피 손실(CE) 및 교사 모델의 로짓 $Y$와의 소프트 교차 엔트로피(SCE)를 최소화합니다: $\mathcal{L}_2 = \text{CE}(\tilde{Y}, Y^*) + \lambda_s \text{SCE}(\tilde{Y}, Y)$.
    * **$L_1$ 정규화 통합**: 두 단계 모두에서 식 (4)의 희소성 정규화를 통합하여 훈련합니다.

## 📊 Results

* **ADE20K (실시간 설정, MiT-B0)**:
  * 원본 SegFormer: 37.4% mIoU, 8.4G FLOPs.
  * DynaSegFormer (30% 가지치기): 36.9% mIoU, **3.3G FLOPs**. (FLOPs 60% 이상 감소, mIoU 0.5% 하락)
  * DynaSegFormer (50% 가지치기): 35.0% mIoU, **2.7G FLOPs**.
* **Cityscapes (실시간 설정, MiT-B0)**:
  * 원본 SegFormer: 76.2% mIoU, 125.5G FLOPs.
  * DynaSegFormer (30% 가지치기): 75.1% mIoU, **62.7G FLOPs**.
* **다른 모델과의 비교**: DynaSegFormer는 DeepLabV3+ (MobileNetV2 백본)보다 2.9% 높은 mIoU를 달성하면서도 FLOPs는 5%에 불과합니다.
* DGL 레이어는 SegFormer의 파라미터 수를 약 12% 증가시키지만, 입력에 따라 인스턴스별로 중복 뉴런을 식별하고 가지치기할 수 있게 합니다.

## 🧠 Insights & Discussion

* **지식 증류의 중요성**: 2단계 지식 증류는 가지치기로 인한 성능 저하를 크게 완화하며 (증류 없는 경우 대비 mIoU 1.2% 향상), 단일 단계 증류보다 더 효과적임이 입증되었습니다. 이는 압축된 모델로 지식을 전이하는 데 효과적인 전략임을 시사합니다.
* **희소성 점진적 증가(Annealing Sparsity)의 효과**: 훈련 중 희소성을 점진적으로 증가시키는 전략은 파라미터 보정에 도움이 되어 mIoU를 0.5~0.7% 향상시킵니다.
* **동적 가지치기의 우수성**: 입력에 고정된 마스크를 사용하는 정적 가지치기(magnitude-based pruning)에 비해 동적 가지치기가 mIoU에서 2.9~3.2% 더 우수한 성능을 보여, 입력에 따른 뉴런의 중요도가 크게 달라진다는 본 논문의 관찰을 뒷받침합니다.
* **뉴런 유형별 동작 분석**: DGL 레이어는 상시 활성화되어 중요한 Type-I 뉴런을 유지하고, 입력에 따라 선택적으로 활성화되는 Type-II 뉴런을 식별하며, 대부분 비활성화되는 Type-III 뉴런을 효과적으로 가지치기 함을 확인했습니다. 이는 동적 가지치기 방식이 제한된 연산 제약 하에서 데이터를 더 잘 설명하기 위해 뉴런을 선택적으로 활용함을 의미합니다.
* **제한 사항**: 모든 모델 파라미터와 게이트 예측기 파라미터를 저장해야 하므로, 모델 크기 자체는 동적 가지치기를 적용하지 않은 모델보다 다소 증가할 수 있습니다 (MiT-B0의 경우 0.44M 파라미터 추가). 하지만 게이트 예측기의 연산 복잡도는 선형 레이어에 비해 무시할 수 있는 수준입니다.

## 📌 TL;DR

본 논문은 SegFormer의 높은 연산 비용 문제를 해결하기 위해 **동적 게이팅 선형 레이어(DGL)**와 **2단계 지식 증류**를 제안합니다. DGL은 입력 인스턴스에 따라 비정보성 뉴런을 동적으로 가지치기하며, 2단계 지식 증류는 가지치기로 인한 성능 손실을 최소화합니다. 결과적으로, ADE20K 데이터셋에서 0.5%의 mIoU 하락으로 SegFormer의 FLOPs를 60% 이상 절감하는 동시에 기존 CNN 기반 모델을 능가하는 효율적인 의미론적 분할 성능을 달성했습니다.
