# 2차원 이미지 의미론적 분할을 위한 딥러닝 기반 아키텍처에 대한 서베이
Irem Ulku, Erdem Akagündüz

## 🧩 Problem to Solve
의미론적 분할(Semantic segmentation)은 이미지 내 모든 픽셀에 클래스 레이블을 할당하는 컴퓨터 비전의 핵심 문제입니다. 자율주행 차량, 로봇 보조 수술, 지능형 군사 시스템과 같은 복잡한 로봇 시스템 구축에 필수적입니다. 이 문제의 본질은 단순히 이미지에 포함된 객체의 클래스를 식별하는 것을 넘어, 원본 이미지 픽셀 해상도에서 레이블을 정확하게 지역화하는 것입니다. 이 서베이 논문은 특히 2D 이미지를 사용하는 딥러닝 기반 의미론적 분할 방법론의 최근 발전 동향을 체계적으로 분석하고, 이러한 기술이 직면한 근본적인 문제점(예: 정교한 지역화 및 스케일 불변성)을 해결하기 위한 노력을 탐구합니다.

## ✨ Key Contributions
*   **종합적인 서베이**: 2D 이미지의 딥러닝 기반 의미론적 분할 분야의 최신 과학적 발전을 포괄적으로 다룹니다.
*   **데이터셋 및 평가 분석**: 공개 이미지 데이터셋과 성능 평가(정확도 및 계산 복잡성) 기술에 대한 심층적인 분석을 제공합니다.
*   **시대별 분류**: 이 분야의 진화를 '심층 학습 이전 및 초기 심층 학습 시대', '완전 합성곱(FCN) 시대', 'FCN 이후 시대'의 세 가지 주요 기간으로 구분하여 연대기적으로 접근 방식들을 분석합니다.
*   **기술적 분석**: 정교한 지역화(fine-grained localisation) 및 스케일 불변성(scale invariance)과 같은 분야의 근본적인 문제들을 해결하기 위한 다양한 방법론들을 기술적으로 분석합니다.
*   **방법론 비교표**: 언급된 모든 시대의 방법론들을 표로 정리하여 각각의 기여를 설명하고, 연대기적 발전과 계산 효율성에 대한 통찰력을 제공합니다.
*   **미래 방향 제시**: 현재 분야의 도전 과제와 해결 정도를 논의하고, 향후 연구 방향(약지도 학습, 제로/퓨샷 학습, 도메인 적응, 실시간 처리, 문맥 정보 활용 등)을 제시합니다.

## 📎 Related Works
이 서베이 논문은 기존의 의미론적 분할 관련 서베이 연구들을 분석하고 본 연구의 차별점을 강조합니다.
*   **특정 문제 중심의 서베이**: 일부 서베이는 자율주행, 의료 시스템 등 특정 응용 분야에 초점을 맞춰 해당 분야 내 접근 방식들을 비교하지만, 분야 전반에 걸친 기술적 비전과 미래 방향 제시에 한계가 있습니다.
*   **일반적 개요 중심의 서베이**: 다른 서베이들은 일반적인 개요를 제공하지만, 딥러닝 기반 방법론, 특히 FCN 이후의 혁신적인 발전에 대한 깊이 있는 분석이 부족합니다. 심층 학습 이전에도 의미론적 분할이 연구되었으나, FCN 이후에야 실제적인 기여가 이루어졌습니다.
*   **포괄적이나 일반적인 분류**: (Garcia-Garcia et al. 2017) 및 (Minaee et al. 2020)과 같은 최근 서베이들은 다양한 모달리티(2D, RGB-D, 3D)를 포괄하지만, 방법론 분류가 너무 광범위하여 기술 진화에 대한 상세한 논의가 부족합니다.
*   **상세 분류와 연대기적 상관관계 부족**: (Lateef and Ruichek 2019)는 상세한 하위 범주를 제공하지만, 제안된 기술들이 시간적으로 어떻게 상호 연관되어 발전했는지에 대한 논의가 빠져 있습니다.

본 서베이 논문은 이러한 한계점을 극복하고, 새로운 도전 과제들을 정의하며, 제안된 맥락 내에서 모든 연구들의 연대기적 진화를 제시하는 데 중점을 둡니다.

## 🛠️ Methodology
본 서베이는 딥러닝 기반 2D 의미론적 분할 아키텍처의 발전을 크게 세 가지 시대로 분류하여 분석합니다.

1.  **데이터셋 및 성능 평가**:
    *   **주요 2D 이미지 데이터셋**: PASCAL VOC, COCO, ADE20K(일반 목적) 및 Cityscapes(도심 거리)와 같은 대규모 공개 데이터셋의 특징과 도전 과제를 설명합니다.
    *   **성능 평가 지표**: 정확도 측정(픽셀 정확도(PA), 평균 픽셀 정확도(mPA), IoU(Intersection over Union), 평균 IoU(mIoU), 빈도 가중 IoU(FwIoU), 정밀도-재현율 곡선(PRC) 기반 지표(F-score, PRC-AuC, 평균 정밀도(AP), 평균 AP(mAP)), 하우스도르프 거리(Hausdorff Distance, HD))과 계산 복잡성 측정(실행 시간, 메모리 사용량)을 상세히 설명합니다.

2.  **시대별 의미론적 분할 방법론 분석**:
    *   **FCN 이전 시대 (Pre- and Early Deep Learning Approaches)**:
        *   **심층 학습 이전**: 수공예 특징과 그래프 모델(MRF, CRF)을 사용한 방법론. 일부 CRF는 현재 딥러닝 모델의 **정제(refinement) 계층**으로 활용됩니다.
        *   **초기 심층 학습**: 이미지 분류를 위한 CNN(AlexNet, VGG)을 분할 네트워크로 변환하려는 초기 시도들. 완전 연결 계층의 한계와 추상 특징 추출의 부족으로 만족스럽지 못한 성능을 보였습니다.
    *   **완전 합성곱 네트워크 (FCN) 시대**:
        *   **혁신**: (Shelhamer et al. 2017)에 의해 제안된 FCN은 완전 연결 계층을 제거하고 합성곱 계층만으로 구성되어, 임의 해상도 이미지에 대해 픽셀 단위 예측을 가능하게 했습니다.
        *   **주요 특징**: 추론 속도 향상, 임의 이미지 해상도 처리, 그리고 정보 손실을 줄이는 **스킵 연결(skip connections)** 도입이 핵심 기여입니다.
    *   **FCN 이후 시대 (Post-FCN Approaches)**:
        *   FCN의 한계점(특징 계층 내 레이블 지역화 손실, 전역 문맥 정보 처리 능력 부족, 다중 스케일 처리 메커니즘 부재)을 극복하는 데 초점을 맞춥니다.
        *   **정교한 지역화를 위한 기술**:
            *   **인코더-디코더 아키텍처 (Encoder-Decoder Architecture)**: U-Net, SegNet과 같이 인코더에서 공간 차원을 줄이고 디코더에서 세부 정보를 복구하며 스킵 연결로 정보 손실을 최소화합니다.
            *   **공간 피라미드 풀링 (Spatial Pyramid Pooling, SPP)**: 다양한 입력 크기에 대해 고정된 크기의 특징 맵을 생성하여 다중 스케일 문맥 정보를 통합합니다.
            *   **특징 연결 (Feature Concatenation)**: 다른 소스나 계층에서 추출된 특징을 융합하여 문맥 지식을 강화합니다.
            *   **확장 합성곱 (Dilated Convolution)**: 풀링 없이 유효 수용 필드(effective receptive field)를 확장하여 특징 맵 해상도를 유지하면서 더 넓은 문맥을 포착합니다.
            *   **조건부 무작위 필드 (Conditional Random Fields, CRFs)**: CNN의 픽셀 분류 결과의 정밀도를 높이기 위해 후처리 정제 계층으로 사용됩니다. (최근에는 사용이 감소하는 추세)
            *   **순환 접근 방식 (Recurrent Approaches)**: RNN, LSTM, 어텐션 메커니즘을 활용하여 순차적 정보나 장거리 의존성을 모델링하여 분할 정확도를 향상시킵니다.
        *   **스케일 불변성 (Scale-Invariance)**: 다중 스케일 학습을 통해 스케일 불변성을 달성하는 방법을 논의합니다.
        *   **객체 탐지 기반 방법 (Object Detection-based Methods)**: RCNN, YOLO, SSD 등 객체 탐지 개념을 의미론적 분할(특히 인스턴스 분할)에 통합하여 객체 지역화와 분할을 동시에 수행하는 접근 방식들을 소개합니다.

3.  **방법론의 진화 및 미래 방향**:
    *   주요 방법론의 성능, 계산 효율성, 핵심 아이디어를 담은 표($\text{Table } 1$)를 제공하여 전체 분야의 진화를 요약합니다.
    *   약지도 학습, 제로/퓨샷 학습, 도메인 적응, 실시간 처리, 문맥 정보 활용 등 향후 연구의 유망한 방향들을 제시합니다.

## 📊 Results
이 서베이는 딥러닝 기반 2D 의미론적 분할 분야의 주요 결과와 동향을 다음과 같이 요약합니다.

*   **FCN의 혁명**: FCN(Fully Convolutional Networks)은 완전 연결 계층을 제거하고 임의 크기 입력에 대한 픽셀 단위 예측을 가능하게 함으로써 의미론적 분할 분야의 중요한 전환점이 되었습니다. 이는 추론 속도를 크게 향상시키고 스킵 연결을 통해 지역화 정보를 보존하는 새로운 아키텍처 가능성을 열었습니다.
*   **FCN 이후의 발전**: FCN 이후의 연구들은 FCN의 한계, 즉 특징 계층에서의 지역화 손실, 전역적 문맥 정보 처리의 어려움, 다중 스케일 처리 부족 문제를 해결하는 데 집중했습니다. 인코더-디코더 구조(U-Net, SegNet), 공간 피라미드 풀링(PSPNet, DeepLab 시리즈), 확장 합성곱(Dilated Convolution), 특징 연결(Feature Concatenation) 등이 이러한 문제를 해결하기 위해 제안되었습니다.
*   **CRF의 점진적 포기**: 초기 DeepLab 버전에서 유용했던 조건부 무작위 필드(CRF)와 같은 그래프 모델 기반의 정제(refinement) 모듈은 느린 계산 속도로 인해 최신 아키텍처에서는 점차 사용되지 않는 경향을 보입니다.
*   **최근 성능 향상 둔화와 새로운 접근법**: 2019년과 2020년에 발표된 연구들에서는 이전과 같은 획기적인 성능 향상을 보이지 않았습니다. 이로 인해 연구자들은 객체 탐지 기반 방법(Mask R-CNN, YOLACT, SOLO)이나 신경망 아키텍처 탐색(Neural Architecture Search, NAS) 기반 접근 방식(EfficientNet-NAS, DCNAS)과 같은 실험적인 해결책에 집중하고 있습니다.
*   **실시간 처리의 중요성**: YOLACT, SOLO v2, Deep Snake, SwiftNetRN18-Pyr와 같은 방법론들은 실시간 인스턴스 분할을 달성하며 산업 응용 분야에서 중요한 돌파구를 마련했습니다.
*   **지속적인 핵심 과제**: 전역 문맥 정보와 정교한 지역화 정보의 효율적인 통합은 여전히 해결되지 않은 핵심 과제로 남아 있으며, 이 문제를 해결하기 위한 연구가 계속될 것입니다.

## 🧠 Insights & Discussion
*   **핵심 도전 과제**: 2D 의미론적 분할 문제의 궁극적인 과제는 픽셀 레이블의 `정교한 지역화($\text{fine-grained localisation}$)`입니다. 이는 단순히 지역적인 관심사를 넘어, `전역 문맥($\text{global context}$)` 정보가 방법론의 실제 성능을 결정하는 데 매우 중요함을 보여줍니다.
*   **지역-전역 의미론적 간극 해소 노력**: 이러한 `지역($\text{local}$)` 정보와 `전역($\text{global}$)` 문맥 간의 간극을 해소하기 위한 다양한 접근 방식들이 연구되었습니다. 여기에는 그래프 모델, 문맥 통합 네트워크(context aggregating networks), 순환적 접근 방식(recurrent approaches), 그리고 어텐션 기반 모듈(attention-based modules) 등이 포함됩니다. 이러한 연구 노력은 앞으로도 지속될 것으로 예상됩니다.
*   **공개 챌린지의 영향**: PASCAL VOC, Cityscapes 등과 같은 `공개 챌린지($\text{public challenges}$)`는 학계 및 산업계 연구 그룹 간의 끊임없는 경쟁을 촉진하여 이 분야의 발전을 가속화하는 데 지대한 영향을 미쳤습니다. 이는 2D 의료 영상과 같이 `더 특정한 주제($\text{more specific subjects}$)`에 대한 유사한 공개 이미지 세트와 챌린지를 만드는 것을 장려해야 함을 시사합니다.
*   **한계 및 미래 방향**:
    *   **어려운 픽셀 수준 주석**: 현재 방법들은 시간과 비용이 많이 드는 `픽셀 수준 주석($\text{pixel-level annotations}$)`에 크게 의존합니다. 이를 해결하기 위해 `약지도 학습($\text{Weakly-Supervised Semantic Segmentation, WSSS}$)` (이미지 수준 레이블, 스크리블, 바운딩 박스 사용) 연구가 활발합니다.
    *   **일반화 능력 부족**: 새로운 도메인이나 클래스에 대한 `일반화($\text{generalization}$)` 능력이 부족합니다. 이를 위해 `제로/퓨샷 학습($\text{Zero-/Few-Shot Learning}$)`과 `도메인 적응($\text{Domain Adaptation}$)` (합성 데이터와 실제 데이터 간의 도메인 시프트 해결) 연구가 중요합니다.
    *   **실시간 성능 요구**: `실시간 처리($\text{Real-Time Processing}$)`를 위한 컴팩트하고 얕은 모델 아키텍처, 효율적인 합성곱 유형(깊이별 분리 합성곱, 그룹 합성곱), 신경망 아키텍처 탐색(NAS) 등이 제안되고 있지만, 이는 종종 정확도와의 `상충 관계($\text{trade-off}$)`를 가집니다.
    *   **문맥 정보의 중요성**: `문맥 정보($\text{Contextual Information}$)`를 효과적으로 집약하고 활용하여 픽셀 표현을 강화하는 것이 또 다른 유망한 연구 방향입니다. `픽셀 단위 대비 손실($\text{pixel-wise contrastive loss}$)`과 같은 새로운 손실 함수도 탐구되고 있습니다.

## 📌 TL;DR
*   **문제**: 2D 이미지의 픽셀 단위 의미론적 분할(semantic segmentation)은 자율주행 등 복잡 시스템에 필수적이지만, 픽셀 수준의 `정교한 지역화($\text{fine-grained localisation}$)`와 `전역 문맥($\text{global context}$)` 통합이 주요 과제입니다.
*   **방법**: 이 서베이는 딥러닝 기반 2D 의미론적 분할 아키텍처의 발전을 '심층 학습 이전/초기', 'FCN 시대', 'FCN 이후' 세 시대로 나누어 연대기적/기술적으로 분석합니다. 주요 데이터셋, 평가 지표, 그리고 지역화, 스케일 불변성, 객체 탐지 기반 방법 등의 기술적 해결책을 상세히 다룹니다.
*   **주요 결과**: `FCN($\text{Fully Convolutional Network}$)`의 등장은 완전 연결 계층 제거와 스킵 연결 도입으로 혁명적이었으며, 이후 연구는 FCN의 `지역화 손실($\text{loss of localisation}$)` 및 `전역 문맥($\text{global context}$)` 처리 한계 극복에 집중했습니다. 속도 문제로 `CRF($\text{Conditional Random Field}$)`와 같은 후처리 정제 모듈은 점차 사용되지 않는 추세이며, 최근에는 `NAS($\text{Neural Architecture Search}$)` 기반 모델과 `객체 탐지 기반($\text{object detection-based}$)` 실시간 방법론이 주목받고 있습니다. `전역 문맥($\text{global context}$)`과 `지역 정보($\text{localisation information}$)`의 효율적 통합이 여전히 가장 큰 도전 과제이며, `약지도 학습($\text{Weakly-Supervised}$)`, `제로/퓨샷 학습($\text{Zero-/Few-Shot Learning}$)` 및 `도메인 적응($\text{Domain Adaptation}$)`이 향후 연구의 핵심 방향입니다.