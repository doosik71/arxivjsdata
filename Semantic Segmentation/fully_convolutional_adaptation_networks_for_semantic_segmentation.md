# Fully Convolutional Adaptation Networks for Semantic Segmentation

Yiheng Zhang, Zhaofan Qiu, Ting Yao, Dong Liu, and Tao Mei

## 🧩 Problem to Solve

* **데이터 라벨링 비용**: 픽셀 단위 주석(pixel-level annotations)이 필요한 시맨틱 분할(semantic segmentation)과 같은 비전 태스크를 위한 대규모 데이터셋 구축은 매우 비싸고 시간이 많이 소요됩니다.
* **합성 데이터의 한계**: 컴퓨터 게임 등에서 합성 데이터를 자동으로 생성하여 사용하면 라벨링 비용을 절감할 수 있지만, 합성 데이터로 학습된 모델을 실제 이미지에 적용할 경우 "도메인 시프트(domain shift)" 현상으로 인해 성능 저하가 발생합니다.
* **도메인 불일치 해결**: 이 논문은 합성 소스 도메인(예: GTA5)에서 학습한 시맨틱 분할 모델을 레이블 없는 실제 타겟 도메인(예: Cityscapes)에 효과적으로 적용하기 위한 비지도 도메인 적응(unsupervised domain adaptation) 문제를 다룹니다.

## ✨ Key Contributions

* **FCAN (Fully Convolutional Adaptation Networks) 제안**: 시맨틱 분할을 위한 새로운 딥 아키텍처인 FCAN을 제안합니다. 이는 **시각적 외형 수준(appearance-level)** 및 **표현 수준(representation-level)** 도메인 적응을 결합하여 도메인 불일치 문제를 해결합니다.
* **AAN (Appearance Adaptation Networks) 개발**: 이미지의 시각적 외형을 한 도메인에서 다른 도메인의 "스타일"로 변환하여 외형 수준의 도메인 불변성을 구축합니다. 이는 소스 이미지의 의미론적 내용을 보존하면서 타겟 도메인의 시각적 스타일을 적용합니다.
* **RAN (Representation Adaptation Networks) 개발**: 적대적 학습(adversarial learning) 방식을 통해 도메인 불변 표현(domain-invariant representations)을 학습합니다. 도메인 판별자(domain discriminator)를 속여 소스 및 타겟 표현이 구별 불가능하도록 만듭니다.
* **ASPP (Atrous Spatial Pyramid Pooling) 확장**: RAN에서 적대적 학습 능력을 향상시키기 위해 ASPP 전략을 확장하여 다양한 스케일의 특징을 활용합니다.
* **최고 성능 달성**: GTA5에서 Cityscapes로의 전이 학습에서 기존 비지도 적응 기법들을 능가하는 우수한 성능을 달성했으며, BDDS(dashcam video) 데이터셋에서는 비지도 설정에서 47.5% mIoU의 새로운 기록을 세웠습니다.

## 📎 Related Works

* **시맨틱 분할 (Semantic Segmentation)**:
  * FCN(Fully Convolutional Networks) [16] 기반의 다양한 기법들: Dilated Convolution [36], RefineNet [13], DeepLab [1] 등 다중 스케일 특징 앙상블, PSPNet [37] 등 컨텍스트 정보 보존.
  * 픽셀 수준 주석의 높은 비용으로 인해 바운딩 박스 [3]나 이미지 수준 태그 [22]와 같은 약한 감독(weak supervision) 활용 연구도 진행되었습니다.
* **딥 도메인 적응 (Deep Domain Adaptation)**:
  * **비지도 적응 (Unsupervised Adaptation)**: 레이블된 타겟 데이터가 없는 경우를 다룹니다.
    * CORAL [28]: Maximum Mean Discrepancy (MMD)를 사용하여 분포의 통계적 속성을 맞춥니다.
    * ADDA [31]: 적대적 훈련을 통해 적응 모델을 최적화합니다.
    * FCNWild [9]: 시맨틱 분할에 특화된 완전 컨볼루션 적대적 훈련을 사용합니다.
  * **FCAN의 차별점**: FCNWild가 오직 완전 컨볼루션 적대적 훈련만을 활용하는 것과 달리, FCAN은 시각적 외형 수준과 표현 수준의 도메인 적응을 모두 고려하여 도메인 간의 간극을 더욱 원칙적인 방식으로 해소합니다.

## 🛠️ Methodology

FCAN은 **AAN (Appearance Adaptation Networks)**과 **RAN (Representation Adaptation Networks)**의 두 가지 주요 구성 요소로 이루어져 있습니다.

### 1. Appearance Adaptation Networks (AAN)

* **목표**: 소스 이미지의 의미론적 내용(semantic content)을 보존하면서 타겟 도메인의 시각적 "스타일"을 적용하여 이미지의 시각적 외형을 도메인 불변으로 만듭니다.
* **과정**:
    1. **초기화**: 백색 잡음(white noise) 이미지 $x_o$에서 시작합니다.
    2. **CNN 활용**: 사전 훈련된 CNN(ResNet-50)을 사용하여 특징 맵을 추출합니다.
    3. **내용 보존**: $x_o$의 특징 맵 $M_l^o$과 소스 이미지 $x_s$의 특징 맵 $M_l^s$ 간의 유클리드 거리(Euclidean distance)를 최소화하여 의미론적 내용을 보존합니다.
    4. **스타일 적응**: $x_o$의 스타일 특징 $G_l^o$(Gram matrix)과 타겟 도메인의 평균 스타일 $\bar{G}_l^t$ 간의 거리를 최소화하여 타겟 도메인의 스타일을 합성합니다.
    5. **전체 손실 함수**: 내용 손실과 스타일 손실을 결합하여 $L_{AAN}(x_o)$를 정의하고, 경사하강법을 통해 $x_o$를 업데이트합니다.
        $$
        L_{AAN}(x_o) = \sum_{l \in L} w_l^s \text{Dist}(M_l^o, M_l^s) + \alpha \sum_{l \in L} w_l^t \text{Dist}(G_l^o, \bar{G}_l^t)
        $$
        여기서 $\alpha$는 내용과 스타일의 균형을 맞추는 가중치($10^{-14}$)입니다.

### 2. Representation Adaptation Networks (RAN)

* **목표**: 도메인 판별자(domain discriminator)를 속여 소스 및 타겟 도메인의 표현이 서로 구별 불가능하도록 만들면서 도메인 불변 표현을 학습합니다.
* **구성**:
    1. **공유 FCN (Feature Extractor)**: Dilated FCN (ResNet-101 기반)을 사용하여 이미지에서 표현 $F(x)$를 추출합니다.
    2. **도메인 판별자 (Domain Discriminator)**: FCN에서 추출된 특징을 입력받아 해당 이미지 영역이 소스 또는 타겟 도메인에서 왔는지 예측합니다.
    3. **ASPP (Atrous Spatial Pyramid Pooling) 확장**: 도메인 판별자의 판별 능력을 강화하기 위해, 다양한 샘플링 레이트(sampling rate)를 가진 여러 팽창 컨볼루션(dilated convolutional) 레이어를 병렬로 사용하여 다중 스케일 컨텍스트를 통합합니다.
* **손실 함수**:
    1. **적대적 손실 ($L_{adv}$)**: GAN 원리를 따르며, FCN은 판별자를 속이려 하고 판별자는 두 도메인을 구별하려 합니다.
    2. **분할 손실 ($L_{seg}$)**: 소스 도메인 이미지에 대해 픽셀 수준 분류를 위한 표준 크로스 엔트로피 손실을 계산합니다.
    3. **전체 목표 함수**: $L_{adv}$와 $L_{seg}$를 결합하여 RAN을 최적화합니다.
        $$
        \max_F \min_D \{ L_{adv}(X_s, X_t) - \lambda L_{seg}(X_s) \}
        $$
        여기서 $\lambda$는 트레이드오프 파라미터(5)입니다.
* **훈련 전략**: RAN을 소스 도메인에서 분할 손실만으로 사전 훈련한 후, 분할 손실과 적대적 손실을 함께 사용하여 미세 조정합니다.

## 📊 Results

* **AAN의 효과**: AAN을 통해 이미지를 적응시키는 것이 시맨틱 분할 성능을 일관되게 향상시켰으며, 특히 AAN과 RAN을 결합했을 때 GTA5에서 Cityscapes로의 전이 학습에서 가장 높은 46.21% mIoU를 달성하여 두 방식의 상호 보완성을 입증했습니다.
* **FCAN 구성 요소별 기여 (Ablation Study)**: FCN (29.15% mIoU)을 기준으로 ABN (35.51%), ADA (41.29%), Conv (43.17%), ASPP (44.81%), AAN (46.60%)이 순차적으로 mIoU 성능을 향상시켜 각 구성 요소의 유효성을 확인했습니다.
* **최신 기술과의 비교 (Cityscapes)**: FCAN (46.60% mIoU, MS 적용 시 47.75%)은 DC [30] (37.64%), ADDA [31] (38.30%), FCNWild [9] (42.04%) 등 기존 비지도 도메인 적응 방법들을 크게 능가했습니다. 특히 19개 카테고리 중 17개에서 최고 성능을 보였습니다.
* **반지도 적응 (Semi-Supervised Adaptation)**: 소량의 레이블된 타겟 데이터(예: Cityscapes 이미지 50개)를 활용하는 반지도 설정에서 FCAN이 지도 학습된 FCN보다 훨씬 우수한 성능을 보여주었습니다.
* **BDDS 데이터셋 결과**: BDDS를 타겟 도메인으로 사용할 때 FCAN(MS+EN)은 47.53% mIoU를 달성하여 FCNWild [9] (39.37%)를 크게 능가하는 새로운 최고 성능 기록을 세웠습니다.

## 🧠 Insights & Discussion

* **복합적 적응 방식의 효용성**: 시각적 외형 수준(AAN)과 표현 수준(RAN)이라는 두 가지 관점에서 도메인 적응을 수행하는 것이 시맨틱 분할의 도메인 시프트 문제를 해결하는 데 매우 효과적임을 입증했습니다. 이 두 가지 접근 방식은 상호 보완적으로 작동하여 도메인 불일치를 더 근본적으로 해소합니다.
* **AAN의 역할**: AAN은 이미지의 "스타일"을 변환하여 시각적 외형을 도메인 불변으로 만듭니다. 이는 저수준 픽셀 정보를 적응시키는 데 중요하며, 소스 이미지의 의미론적 내용을 효과적으로 보존하면서 타겟 도메인의 시각적 특징을 받아들입니다.
* **RAN의 역할**: RAN은 적대적 학습을 통해 도메인 불변 표현을 학습합니다. 특히, 도메인 판별자를 영역(region) 수준으로 확장하고 ASPP를 활용하여 다중 스케일 컨텍스트를 통합함으로써 판별 능력을 강화하고 더 강력한 도메인 불변 특징을 추출합니다.
* **반지도 학습의 잠재력**: 소량의 레이블된 타겟 데이터를 활용하는 반지도 설정에서 FCAN이 지도 학습된 FCN보다 지속적으로 우수한 성능을 보여, 실제 시나리오에서의 활용 가능성을 높였습니다.
* **향후 연구**: AAN에서 이미지 렌더링 기술을 더욱 발전시키고, FCAN을 실내 장면 분할과 같이 합성 데이터를 쉽게 생성할 수 있는 다른 특정 분할 시나리오로 확장하는 것이 고려됩니다.

## 📌 TL;DR

본 논문은 시맨틱 분할에서 합성 데이터와 실제 데이터 간의 도메인 시프트 문제를 해결하기 위해 **FCAN(Fully Convolutional Adaptation Networks)**을 제안합니다. FCAN은 **AAN(Appearance Adaptation Networks)**을 통해 소스 이미지를 타겟 도메인 스타일로 변환하여 시각적 외형 수준의 불변성을 확보하고, **RAN(Representation Adaptation Networks)**을 통해 적대적 학습 방식으로 도메인 불변 표현을 학습합니다. RAN은 FCN과 영역 수준 도메인 판별자, 그리고 ASPP를 결합하여 다중 스케일 특징을 활용합니다. 실험 결과, FCAN은 GTA5에서 Cityscapes로의 전이 학습에서 기존 비지도 도메인 적응 기법들을 크게 능가하며, BDDS 데이터셋에서는 47.53% mIoU의 새로운 최고 성능을 달성하여 외형 및 표현 수준의 복합적 적응이 도메인 시프트를 효과적으로 해소함을 입증했습니다.
