# Fully Convolutional Adaptation Networks for Semantic Segmentation

Yiheng Zhang, Zhaofan Qiu, Ting Yao, Dong Liu, and Tao Mei

## 🧩 Problem to Solve

최근 딥 뉴럴 네트워크는 대규모 데이터셋에서 시각 모델 학습에 뛰어난 능력을 보여주었습니다. 그러나 특히 픽셀 수준 주석을 포함하는 데이터셋을 수집하는 것은 매우 비용이 많이 드는 과정입니다. 이에 대한 매력적인 대안은 합성 데이터(예: 컴퓨터 게임)를 렌더링하고 지상 진실(ground truth)을 자동으로 생성하는 것입니다. 하지만 합성 이미지로 학습된 모델을 실제 이미지에 단순히 적용하면 '도메인 시프트(domain shift)'로 인해 높은 일반화 오류가 발생할 수 있습니다. 이 논문은 이러한 문제를 시각적 외형(visual appearance-level) 및 표현(representation-level) 도메인 적응이라는 두 가지 관점에서 해결하고자 합니다.

## ✨ Key Contributions

- **FCAN (Fully Convolutional Adaptation Networks) 제안**: 시맨틱 분할을 위한 도메인 적응 문제를 해결하기 위해 Appearance Adaptation Networks (AAN)와 Representation Adaptation Networks (RAN)를 결합한 새로운 딥 아키텍처를 제시합니다.
- **시각적 외형 수준 적응 (AAN)**: 소스 도메인 이미지를 타겟 도메인의 "스타일"처럼 보이도록 시각적 외형 수준에서 변환하여 도메인 불변성을 구축합니다.
- **표현 수준 적응 (RAN)**: 적대적 학습(adversarial learning) 방식으로 도메인 불변 표현(domain-invariant representations)을 학습하여 도메인 간의 차이를 최소화합니다.
- **ASPP (Atrous Spatial Pyramid Pooling) 확장**: RAN의 도메인 판별기(domain discriminator)에 ASPP 전략을 도입하여 필터의 수용 영역(field of view)을 확대하고 다중 스케일 표현을 활용해 판별 능력을 강화합니다.
- **최첨단 성능 달성**: GTA5에서 Cityscapes로의 전이 학습에서 기존 비지도 도메인 적응 기법보다 우수한 성능을 달성했으며, BDDS 데이터셋에서는 47.53%의 mIoU를 기록하며 새로운 최첨단 성능을 수립했습니다.

## 📎 Related Works

- **시맨틱 분할 (Semantic Segmentation)**: FCN (Fully Convolutional Networks) [16]의 발전 이후 Dilated Convolution [36], RefineNet [13], DeepLab [1], PSPNet [37] 등 다양한 멀티스케일 특징 앙상블 및 컨텍스트 정보 보존 기법들이 제안되었습니다. 픽셀 수준 주석의 높은 비용 문제로 인해 약한 지도 학습(weak supervision) 방법(예: 인스턴스 수준 바운딩 박스 [3], 이미지 수준 태그 [22])도 연구되었습니다.
- **딥 도메인 적응 (Deep Domain Adaptation)**: 이 분야는 크게 비지도, 지도, 준지도 적응으로 나뉩니다.
  - **비지도 적응**: 레이블이 없는 타겟 데이터를 활용하며, CORAL [28] (MMD 활용) 및 ADDA [31] (적대적 학습) 등이 있습니다.
  - **본 논문과의 차별점**: 본 연구는 주로 시맨틱 분할 작업에 대한 비지도 적응에 중점을 둡니다. 가장 유사한 선행 연구는 FCNWild [9]로, 완전 컨볼루션 적대적 학습을 사용하지만, FCAN은 시각적 외형 수준과 표현 수준 적응을 모두 활용하여 도메인 간 격차를 더욱 체계적으로 해소합니다.

## 🛠️ Methodology

FCAN은 시각적 외형 적응을 위한 AAN (Appearance Adaptation Networks)과 표현 적응을 위한 RAN (Representation Adaptation Networks)의 두 가지 핵심 구성 요소로 이루어져 있습니다.

- **1. Appearance Adaptation Networks (AAN)**:
  - **목표**: 소스 도메인 이미지를 타겟 도메인의 "스타일"처럼 보이도록 시각적으로 유사하게 만듭니다.
  - **과정**: 사전 학습된 CNN (예: ResNet-50)을 사용하여 특징 맵을 추출합니다. 소스 이미지 $x_s$의 고수준 콘텐츠를 보존하고 타겟 도메인 $X_t$의 "스타일"을 $x_o$에 부여하기 위해 백색 노이즈 이미지에서 시작하여 반복적으로 이미지를 조정합니다.
  - **손실 함수**: 콘텐츠 보존을 위한 $x_s$와 $x_o$의 특징 맵 거리 최소화 항과, 타겟 도메인의 평균 스타일($\bar{G}_l^t$)과 $x_o$의 스타일($G_l^o$) 거리 최소화 항으로 구성된 $L_{\text{AAN}}(x_o)$를 최소화합니다.
    $$L_{\text{AAN}}(x_o) = \sum_{l \in L} w_l^s \text{Dist}(M_l^o, M_l^s) + \alpha \sum_{l \in L} w_l^t \text{Dist}(G_l^o, \bar{G}_l^t)$$
    여기서 $M_l$은 특징 맵, $G_l$은 특징 맵 간의 상관관계(스타일)를 나타냅니다.
- **2. Representation Adaptation Networks (RAN)**:
  - **목표**: 적대적 학습을 통해 도메인 불변 표현을 학습하여 도메인 시프트의 영향을 줄입니다.
  - **구성**:
    1. **공유 FCN**: ResNet-101 기반의 FCN을 사용하여 두 도메인의 이미지 표현을 추출합니다. 이 FCN $F$는 두 도메인 간에 구별 불가능한 표현을 학습합니다.
    2. **도메인 판별기 $D$**: FCN의 출력으로부터 각 이미지 영역이 어느 도메인에 속하는지 판별합니다.
    3. **ASPP (Atrous Spatial Pyramid Pooling)**: 판별기의 다중 스케일 표현 학습 능력을 강화하기 위해, 다양한 샘플링 레이트를 갖는 dilated convolution 레이어를 병렬로 사용하여 FCN 출력에 적용합니다.
  - **최적화**:
    1. **적대적 손실 ($L_{adv}$)**: FCN은 판별기를 속이려 하고, 판별기는 두 도메인의 표현을 구별하려 하는 minimax 게임을 수행합니다.
    2. **시맨틱 분할 손실 ($L_{seg}$)**: 레이블이 있는 소스 도메인 이미지에 대한 픽셀 수준 분류 손실을 동시에 최적화합니다.
    3. **총 목적 함수**: 다음의 minimax 함수를 최적화합니다.
       $$\max_{F} \min_{D} \{L_{adv}(X_s, X_t) - \lambda L_{seg}(X_s)\}$$
- **훈련 전략**: FCN을 소스 도메인에서 분할 손실만으로 사전 학습한 후, RAN을 분할 손실과 적대적 손실을 사용하여 공동으로 미세 조정합니다.

## 📊 Results

- **GTA5 $\to$ Cityscapes 전이 학습**:
  - AAN을 통한 외형 적응은 시맨틱 분할 성능을 일관되게 향상시켰으며, AAN과 RAN을 결합했을 때 46.21%의 mIoU를 달성했습니다.
  - **구성 요소별 기여 (Ablation Study)**: FCN(29.15%)을 기준으로 Adaptive Batch Normalization (ABN) (+6.36%), Adversarial Domain Adaptation (ADA) (+5.78%), Convolutional Region Discriminator (Conv) (+1.88%), ASPP (+1.64%), AAN (+1.79%)이 순차적으로 성능을 향상시켜 최종 FCAN은 46.60%의 mIoU를 기록했습니다. Multi-scale (MS) 기법을 적용한 FCAN(MS)은 47.75%까지 성능을 높였습니다.
  - **최첨단 방법과의 비교**: Domain Confusion (37.64%), ADDA (38.30%), FCNWild (42.04%) 대비 FCAN은 46.60% (MS 포함 시 47.75%)로 가장 우수한 성능을 보여주었으며, 19개 카테고리 중 17개에서 최고 성능을 달성했습니다.
- **준지도 적응 (Semi-Supervised Adaptation)**: Cityscapes에서 소수의 레이블된 타겟 데이터를 활용했을 때, FCAN은 지도 학습 FCN보다 뛰어난 성능을 보였습니다. 1,000개의 레이블된 이미지를 사용했을 때, FCAN은 69.17%의 mIoU를 달성했습니다.
- **BDDS 데이터셋 결과**: GTA5에서 BDDS로 전이 학습 시, FCAN은 43.35%의 mIoU를 달성하여 FCNWild(39.37%)보다 3.98% 향상된 성능을 보였습니다. 앙상블 기법을 적용한 FCAN(MS+EN)은 47.53%의 mIoU를 기록하며 새로운 최첨단 성능을 수립했습니다.

## 🧠 Insights & Discussion

- **외형 및 표현 수준 적응의 시너지**: FCAN은 시각적 외형 수준과 표현 수준 도메인 적응을 결합함으로써 도메인 시프트 문제를 더욱 효과적으로 해결할 수 있음을 입증했습니다. 두 가지 관점의 적응은 상호 보완적이며 시맨틱 분할 성능을 크게 향상시킵니다.
- **RAN의 효과**: 적대적 학습을 기반으로 하는 RAN은 도메인 불변 표현을 학습하는 데 매우 효과적이며, 특히 ASPP 전략을 통해 다중 스케일 객체에 대한 적응 성능을 개선했습니다.
- **AAN의 잠재적 한계**: AAN이 적응된 타겟 이미지에 노이즈를 합성할 수 있어, 특히 객체 경계에서 분할 안정성에 영향을 미칠 수 있다는 한계가 관찰되었습니다. 이는 외형 적응 과정에서 세부적인 픽셀 정보의 정확도를 유지하는 것의 중요성을 강조합니다.
- **미래 연구 방향**: AAN에서 이미지의 시맨틱 콘텐츠를 다른 통계적 패턴으로 렌더링하는 보다 진보된 기술을 탐구하고, FCAN을 실내 장면 분할 또는 인물 분할과 같은 다른 특정 분할 시나리오로 확장할 계획입니다.

## 📌 TL;DR

이 논문은 합성 데이터에서 실제 데이터로의 시맨틱 분할 시 발생하는 '도메인 시프트' 문제를 해결하기 위해 **FCAN (Fully Convolutional Adaptation Networks)**을 제안합니다. FCAN은 **AAN (Appearance Adaptation Networks)**을 통해 시각적 외형 수준에서 소스 이미지를 타겟 도메인의 스타일로 변환하고, **RAN (Representation Adaptation Networks)**을 통해 적대적 학습 방식으로 도메인 불변 표현을 학습합니다. RAN은 ASPP (Atrous Spatial Pyramid Pooling)를 활용하여 판별기의 다중 스케일 능력을 강화합니다. 실험 결과, FCAN은 GTA5에서 Cityscapes 및 BDDS로의 전이 학습에서 기존 최첨단 비지도 적응 기법들을 능가하는 우수한 시맨틱 분할 성능을 달성했으며, 특히 BDDS에서 47.53%의 mIoU라는 새로운 기록을 세웠습니다. 이는 외형 및 표현 수준 적응의 결합이 도메인 시프트 문제 해결에 매우 효과적임을 입증합니다.
