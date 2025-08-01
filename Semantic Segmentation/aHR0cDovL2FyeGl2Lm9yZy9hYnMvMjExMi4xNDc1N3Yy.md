# A Simple Baseline for Open-Vocabulary Semantic Segmentation with Pre-trained Vision-language Model

Mengde Xu, Zheng Zhang, Fangyun Wei, Yutong Lin, Yue Cao, Han Hu, and Xiang Bai

## 🧩 Problem to Solve

시맨틱 분할(Semantic Segmentation)은 이미지의 모든 픽셀에 카테고리 레이블을 할당하는 근본적인 컴퓨터 비전 작업입니다. 그러나 기존 시맨틱 분할 데이터셋은 높은 주석 비용 때문에 제한된 수의 카테고리만을 포함하고 있으며, 이로 인해 모델이 풍부한 의미를 처리하는 능력이 제한됩니다.

최근 사전 훈련된 비전-언어 모델(예: CLIP)은 이미지 수준의 개방형 어휘(open-vocabulary) 분류에서 뛰어난 성능을 보였지만, 이러한 모델의 이미지 수준 인식 능력을 픽셀 수준 시맨틱 분할과 같은 광범위한 비전 문제에 효과적으로 전이하는 방법은 명확하지 않습니다. 특히, 시맨틱 분할은 픽셀 단위로 작동하는 반면 CLIP은 이미지 단위로 작동하여 `그레인 불일치(granularity mismatch)`가 발생합니다. 기존의 FCN(Fully Convolutional Networks) 기반 프레임워크에 CLIP을 직접 통합하는 방식은 이러한 불일치로 인해 만족스럽지 못한 성능을 보였습니다.

## ✨ Key Contributions

* 사전 훈련된 CLIP 모델을 활용한 개방형 어휘 시맨틱 분할을 위한 간단하면서도 효과적인 `2단계 프레임워크`를 제안합니다.
* 시맨틱 분할 작업을 `클래스 불특정 마스크 제안 생성`과 `마스크 제안에 대한 개방형 어휘 분류`의 두 하위 작업으로 분리합니다.
* 교차-데이터셋 설정(cross-dataset setting)에서 FCN 기반 접근 방식보다 우수한 성능을 입증하여 제안된 프레임워크의 뛰어난 일반화 능력을 보여줍니다.
* 제안된 간단한 프레임워크가 기존 최신 제로-샷(zero-shot) 시맨틱 분할 방법을 크게 능가함을 입증합니다: Pascal VOC 2012 데이터셋에서 +29.5 hIoU, COCO Stuff 데이터셋에서 +8.9 hIoU 개선 (자체 학습 적용 시).
* 이러한 단순성과 강력한 성능을 바탕으로, 이 프레임워크가 향후 연구를 촉진하는 `강력한 기준선(baseline)` 역할을 할 수 있기를 기대합니다.

## 📎 Related Works

* **비전-언어 사전 훈련 (Vision-Language Pre-training)**: CLIP과 같이 대규모의 노이즈가 있는 웹 데이터로 이미지와 텍스트 개념을 연결하여 제로-샷/개방형 어휘 이미지 분류에서 강력한 성능을 보여줍니다. 본 연구는 CLIP을 강력한 비전-카테고리 대응 모델로 활용합니다.
* **시맨틱 분할 (Semantic Segmentation)**: 픽셀 단위 분류 문제로 모델링하는 FCN 및 그 변형들이 지배적이었으나, 최근 MaskFormer는 세그먼트 생성과 분류로 작업을 분리하여 경쟁력 있는 성능을 보였습니다.
* **제로-샷 학습 및 개방형 어휘 학습 (Zero-Shot Learning & Open-vocabulary Learning)**: 보이지 않는 클래스(unseen classes)를 위해 학습 가능한 표현을 배우는 데 중점을 둡니다. 개방형 어휘 학습은 임의의 클래스 인식을 위한 실현 가능한 방법을 구축하는 데 더 집중하며 추가 정보 활용을 허용합니다.
* **제로-샷 시맨틱 분할 (Zero-shot Semantic Segmentation)**: ZS3Net, CSRL, CaGNet, SPNet 등과 같은 초기 연구들은 단어 임베딩을 통해 픽셀 수준 특징을 합성하거나 비전 특징을 시맨틱 공간에 매핑하는 방식을 탐구했습니다. 최근에는 자체 학습(self-training) 기법도 연구되었습니다. 본 논문은 기존 연구들이 비전-언어 사전 훈련 모델을 활용하지 않았다는 점에서 차별화되며, FCN 기반인 LSeg이나 외부 접지 데이터셋을 사용하는 Openseg과 같은 동시 연구들과도 다릅니다.

## 🛠️ Methodology

본 논문에서 제안하는 개방형 어휘 시맨틱 분할을 위한 2단계 프레임워크는 다음과 같습니다.

1. **마스크 제안 생성 (Mask Proposal Generation)**:
    * **목표**: 클래스에 구애받지 않는(class-agnostic) 이진 마스크 제안($M_p$) 집합을 생성합니다.
    * **방법**:
        * GPB-UCM [2], Selective Search [46], MaskFormer [9] 세 가지 방법을 평가했습니다.
        * `MaskFormer`를 기본 마스크 제안 생성기로 채택했습니다. MaskFormer는 학습된 클래스(`seen classes`)로 훈련되었음에도 불구하고, 보이지 않는 클래스(`unseen classes`)에 대해서도 고품질의 마스크 제안을 생성하는 뛰어난 일반화 능력을 보여주었습니다.

2. **CLIP을 통한 영역 분류 (Region Classification via CLIP)**:
    * **목표**: 생성된 각 마스크 제안을 사전 훈련된 CLIP 모델을 사용하여 해당 카테고리로 분류합니다.
    * **두 가지 전략**:
        * **CLIP 이미지 인코더 직접 적용**: 각 마스크 제안에 대해 이미지에서 마스킹된 영역을 잘라내어 (예: 배경을 0으로 채우기) $224 \times 224$로 크기를 조정한 후 CLIP의 이미지 인코더에 입력합니다. CLIP 텍스트 인코더는 "A photo of [CLASS]"와 같은 프롬프트 템플릿을 사용하여 클래스 임베딩을 생성하고, 이미지 임베딩과의 코사인 유사도를 계산하여 분류합니다. 이 방식은 `unseen classes`에 강하지만, `seen classes`의 훈련 데이터를 활용하지 못합니다.
        * **재훈련된 비전 인코더 (Retrained Vision Encoder)**: CLIP의 텍스트 인코더에서 생성된 텍스트 특징을 고정된 분류기 가중치로 사용하여 `seen classes` 데이터로 이미지 인코더를 재훈련합니다. 이는 이미지 인코더가 텍스트 인코더와 동일한 임베딩 공간에 시각적 특징을 임베딩하도록 장려하여 `unseen classes`에 대한 일반화 능력을 얻게 합니다. 이 방법은 MaskFormer의 훈련 과정에 쉽게 통합될 수 있습니다.
        * 두 전략은 상호 보완적이므로, 기본적으로 두 전략의 결과를 `앙상블(ensemble)`하여 최종 분류 확률을 얻습니다.
    * **프롬프트 설계 (Prompt Design)**:
        * **수제 프롬프트 (Hand-Crafted Prompt)**: CLIP에 제공된 기존 수제 프롬프트(예: ImageNet-1K용)를 재사용하며, 훈련 데이터에서 가장 효과적인 프롬프트를 선택합니다.
        * **학습 기반 프롬프트 (Learning-Based Prompt)**: 프롬프트 토큰을 학습 가능한 파라미터로 설정하여 `seen classes`에서 훈련하고 `unseen classes`로 일반화하는 프롬프트 학습 기법을 탐구합니다.

3. **마스크 예측 통합 (Mask Prediction Assembly)**:
    * 여러 마스크 제안이 겹칠 수 있으므로, 픽셀 단위 시맨틱 분할 결과를 생성하기 위한 간단한 통합 메커니즘을 사용합니다.
    * 특정 픽셀 $q$에 대해 $i$번째 카테고리에 속할 예측 확률 $C_i(q)$는 다음과 같이 정의됩니다:
        $$ C_i(q) = \frac{\sum_k M_{p}{_k}(q)C_{p}{_k}(i)}{\sum_k M_{p}{_k}(q)} $$
        여기서 $M_{p}{_k}(q)$는 $k$번째 마스크 제안 $M_{p}{_k}$에서 픽셀 $q$의 예측 확률을 나타내며, $C_{p}{_k}(i)$는 마스크 제안 $M_{p}{_k}$가 $i$번째 카테고리에 속할 예측 확률입니다. 최종적으로 픽셀 $q$는 가장 높은 예측값을 가진 카테고리로 분류됩니다.

## 📊 Results

* **교차-데이터셋 설정**: COCO Stuff 데이터셋에서 훈련하고 Cityscapes, Pascal Context, ADE20K 등 다른 데이터셋에서 미세 조정 없이 평가했을 때, 제안된 2단계 접근 방식은 FCN 접근 방식보다 현저히 우수한 성능(예: Cityscapes에서 +13.1 mIoU, Pascal Context에서 +19.6 mIoU)을 보이며 뛰어난 일반화 능력을 입증했습니다.
* **제로-샷 설정**:
  * **COCO Stuff**: 자체 학습(self-training) 없이 37.8 hIoU (unseen mIoU 36.3)를 달성하여 기존 최고 CaGNet [20]을 +19.5 hIoU 능가했습니다. 자체 학습을 추가하면 41.5 hIoU (unseen mIoU 43.6)를 달성하여 기존 최고 STRICT [40]를 +8.9 hIoU 능가했습니다.
  * **Pascal VOC 2012**: 자체 학습 없이 77.5 hIoU (unseen mIoU 72.5)를 달성하여 CaGNet [20]을 +37.7 hIoU 크게 능가했습니다. 자체 학습을 추가하면 79.3 hIoU (unseen mIoU 78.1)를 달성하여 STRICT [40]를 +29.5 hIoU 능가했습니다.
* **어블레이션 연구 (Ablation Studies)**:
  * 마스크 제안 생성 방식 중 MaskFormer가 가장 좋은 성능을 보였으며, 데이터셋 간에도 제안 생성의 일반화 능력이 우수함을 입증했습니다.
  * 재훈련된 비전 인코더와 CLIP 비전 인코더 직접 적용 전략의 앙상블이 `seen` 및 `unseen classes` 모두에서 성능을 크게 향상시켰습니다.
  * CLIP ViT-B/16 백본이 가장 좋은 성능을 보였으며, 학습 가능한 프롬프트가 수제 프롬프트보다 훨씬 우수했습니다.
  * `thing` 클래스와 `stuff` 클래스 간의 성능 격차가 존재했으나, 자체 학습을 통해 이 격차를 줄일 수 있었습니다. 이는 CLIP 사전 훈련 데이터의 편향 가능성을 시사합니다.

## 🧠 Insights & Discussion

본 연구의 2단계 프레임워크는 픽셀 수준 시맨틱 분할과 이미지 수준 비전-언어 모델(CLIP) 간의 그레인 불일치를 효과적으로 해결합니다. 마스크 제안 생성과 마스크 분류를 분리함으로써, MaskFormer(학습된 클래스에서 훈련됨)가 보이지 않는 클래스에 대해서도 일반화 가능한 고품질 마스크를 생성할 수 있게 하고, CLIP은 이 마스크들에 대한 개방형 어휘 분류를 담당합니다.

CLIP의 직접 분류와 재훈련된 비전 인코더를 앙상블하는 전략은 `seen` 클래스와 `unseen` 클래스 모두에서 균형 잡힌 높은 성능을 달성하는 데 필수적입니다. 이 결과는 성능 향상이 단순히 대규모 사전 훈련 데이터 때문이 아니라, 제안된 2단계 아키텍처의 이점 덕분임을 보여줍니다. 또한, 학습 기반 프롬프트가 CLIP 모델을 시맨틱 분할 작업에 적용하는 데 매우 효과적이라는 것을 확인했습니다.

한 가지 한계는 `unseen` 카테고리의 `thing` 클래스와 `stuff` 클래스 간에 성능 차이가 있다는 점입니다. 이는 CLIP의 사전 훈련 데이터에 잠재적인 편향이 있음을 시사하며, 자체 학습(self-training)을 통해 이러한 격차를 완화할 수 있습니다. 이러한 단순하지만 효과적인 접근 방식은 개방형 어휘 시맨틱 분할 분야의 미래 연구를 위한 강력한 기준선이 될 것으로 기대됩니다.

## 📌 TL;DR

이 논문은 사전 훈련된 CLIP 모델을 활용하여 개방형 어휘 시맨틱 분할을 위한 간단한 2단계 프레임워크를 제안합니다. 이 프레임워크는 먼저 클래스 불특정 마스크 제안을 생성한 다음 (MaskFormer를 통해) CLIP을 사용하여 이 제안들을 분류함으로써 이미지 수준의 CLIP과 픽셀 수준의 분할 간의 그레인 불일치를 효과적으로 해소합니다. 제안된 방법은 기존 제로-샷 및 교차-데이터셋 시맨틱 분할 방법보다 훨씬 우수한 성능을 달성하며, 이 분야의 새로운 강력한 기준선을 제시합니다.
