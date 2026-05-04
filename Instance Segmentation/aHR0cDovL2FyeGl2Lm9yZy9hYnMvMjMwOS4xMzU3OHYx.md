# A SAM-based Solution for Hierarchical Panoptic Segmentation of Crops and Weeds Competition

Khoa Dang Nguyen, Thanh-Hai Phung, Hoang-Giang Cao (2023)

## 🧩 Problem to Solve

본 논문은 농업 환경에서 작물(Crop)과 잡초(Weed)를 구분하고, 이를 계층적으로 분할하는 Hierarchical Panoptic Segmentation 문제를 해결하고자 한다. 농업 분야의 Panoptic Segmentation은 단순한 클래스 분류를 넘어, 필드 구성의 종합적인 이해를 가능하게 하며, 구체적으로는 작물과 잡초의 분할, 식물 개체 분할, 그리고 잎(Leaf) 단위의 인스턴스 분할을 목표로 한다.

특히 본 연구는 8th Workshop on Computer Vision in Plant Phenotyping and Agriculture (CVPPA)에서 주최한 PhenoBench 데이터셋 기반의 챌린지에 대응한다. 이 문제의 핵심은 설탕무(Sugar beet)와 잡초를 정확하게 식별하고 차별화하는 것이며, 객체의 부분(Part)을 구분하는 계층적 분할을 통해 각 식물 인스턴스에 고유한 라벨을 부여해야 한다는 점에서 난이도가 높다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 인스턴스 분할 성능이 뛰어난 Segment Anything Model (SAM) 계열의 모델을 활용하되, SAM에 필요한 프롬프트(Prompt) 입력을 객체 탐지(Object Detection) 모델을 통해 자동으로 생성하는 파이프라인을 구축하는 것이다.

중심적인 설계 전략은 다음과 같다.

1. **HQ-SAM의 채택**: 복잡한 구조나 얇은 선을 가진 농작물 이미지의 특성을 고려하여, 일반 SAM보다 정밀한 마스크 생성이 가능한 HQ-SAM(Segment Anything in High-Quality)을 사용하였다.
2. **하이브리드 객체 탐지 프롬프트**: 단일 모델의 한계를 극복하기 위해 DINO와 YOLO-v8 두 가지 모델을 결합하였다. 특히 잡초 탐지에는 YOLO-v8이, 잎과 작물 탐지에는 DINO가 효과적이라는 점을 이용하여 상호 보완적인 탐지 전략을 세웠다.
3. **효율적인 미세 조정(Fine-tuning)**: 거대 모델인 ViT-Huge HQ-SAM을 효율적으로 학습시키기 위해 LoRA(Low-Rank Adaptation) 기법을 도입하여 메모리 효율성을 높였다.

## 📎 Related Works

논문은 Panoptic Segmentation이 시맨틱 분할(Semantic Segmentation)과 인스턴스 분할(Instance Segmentation)을 결합하여 장면의 전체적인 이해를 돕는 기술임을 명시한다.

기존 연구 및 접근 방식과의 차별점은 다음과 같다.

- **SAM의 한계 극복**: 일반적인 SAM은 11억 개의 마스크로 학습되어 범용성이 높지만, PhenoBench 데이터셋과 같이 얇은 선이 많거나 배경이 복잡한 이미지에서는 정교한 분할이 어렵다. 이를 해결하기 위해 본 연구는 learnable HQ-Output Token을 사용하는 HQ-SAM을 도입하였다.
- **프롬프트 생성 방식**: SAM은 프롬프트(점, 박스 등)가 필수적인 모델이다. 본 연구는 이를 수동으로 입력하는 대신, 미세 조정된 DINO와 YOLO-v8을 통해 정밀한 Bounding Box를 생성하여 SAM의 입력으로 사용하는 자동화된 파이프라인을 제안하였다.

## 🛠️ Methodology

### 전체 시스템 구조

전체 파이프라인은 **객체 탐지(Object Detection) 모듈**과 **SAM 기반 분할(Segmentation) 모듈**의 두 단계로 구성된다. 먼저 객체 탐지 모델이 이미지에서 작물, 잡초, 잎의 Bounding Box를 추출하면, 이 박스들이 HQ-SAM의 프롬프트로 입력되어 최종적인 인스턴스 마스크를 생성한다.

### 상세 구성 요소 및 학습 절차

#### 1. SAM 기반 분할 (HQ-SAM)

- **모델 선택**: 복잡한 구조의 객체 분할 능력을 높이기 위해 HQ-SAM의 ViT-Base 및 ViT-Huge 모델을 사용하였다.
- **LoRA (Low-Rank Adaptation)**: ViT-Huge와 같은 거대 모델의 전체 파라미터를 미세 조정하는 것은 비효율적이므로, LoRA를 통해 이미지 인코더와 마스크 디코더에 저차원 행렬을 추가하여 학습 가능한 파라미터 수를 줄였다. 이때 Rank 값은 $4$로 설정했을 때 최적의 성능을 보였다.
- **Box Augmentation**: 객체 탐지 모델이 생성한 박스가 완벽하지 않을 수 있음을 고려하여, 학습 시 Ground-truth 박스 좌표에 박스 길이의 $10\%$에서 최대 $20$ 픽셀 범위의 랜덤 노이즈를 추가하는 증강 기법을 적용하였다.

#### 2. 객체 탐지 (DINO & YOLO-v8)

- **DINO**: 잎(1 클래스)과 식물(작물 및 잡초, 2 클래스)을 탐지하는 두 개의 디텍터를 별도로 학습시켰다. 특히 `DINO-Focal-Large-4scale` 구조가 가장 좋은 성능을 보였다.
- **YOLO-v8**: 실시간 성능이 뛰어난 YOLO-v8 nano 모델을 사용하였다. 실험 결과 YOLO-v8은 특히 잡초 탐지에서 강점을 보였다.
- **하이브리드 전략**: DINO와 YOLO-v8의 결과를 결합하여, $\text{IoU} > 0.5$인 중복 박스를 제거하고 YOLO-v8이 추가로 찾아낸 잡초 박스(너비와 높이가 $50$ 픽셀 이상인 경우)를 최종 리스트에 포함시키는 방식을 사용하였다.

#### 3. 학습 설정

- **최적화**: AdamW 옵티마이저를 사용하였으며, 모멘텀 $0.9$, Weight decay $1\text{e-}4$, 학습률 $1\text{e-}5$를 적용하였다.
- **스케줄링**: Cosine annealing 스케줄을 사용하였으며, $\text{mean IoU}$가 $20$ 에포크 동안 개선되지 않을 경우 Early stopping을 수행하였다.
- **AMP (Automatic Mixed Precision)**: 메모리 제약을 완화하기 위해 사용하였으나, 잎 탐지기 학습에서는 AMP 적용 시 Average Precision(AP)이 하락하는 현상이 발견되어 선택적으로 적용하였다.

## 📊 Results

### 실험 환경 및 지표

- **데이터셋**: PhenoBench 데이터셋
- **평가 지표**: $\text{IoU}$ (Soil, Weed), $\text{PQ}$ (Leaf, Crop), $\text{PQ+}$, $\text{PQ}$
- **비교 대상**: Panoptic DeepLab, Mask R-CNN, Mask2Former, HAPT

### 주요 결과

본 연구의 최종 모델은 **$\text{PQ+}$ 점수 $81.33$**을 기록하며 매우 우수한 성능을 보였다.

- **정량적 결과 (Table 2 기준)**:
  - $\text{IoU (soil)}$: $99.18$
  - $\text{IoU (weed)}$: $70.66$
  - $\text{PQ (leaf)}$: $73.81$
  - $\text{PQ (crop)}$: $81.66$
  - $\text{PQ+}$: $81.33$
- **분석**: 기존의 Mask2Former($\text{PQ+} \approx 69.99$)나 HAPT($\text{PQ+} \approx 65.27$)와 비교했을 때, 모든 지표에서 압도적인 성능 향상을 보였다. 특히 잡초(Weed)의 $\text{IoU}$와 작물(Crop)의 $\text{PQ}$에서 큰 폭의 상승이 있었다.

## 🧠 Insights & Discussion

### 강점 및 유효성

본 논문은 SAM의 강력한 제로샷(Zero-shot) 분할 능력을 유지하면서, 도메인 특화 데이터(PhenoBench)로 미세 조정을 수행함으로써 농업 환경의 특수한 형태(얇은 잎, 복잡한 겹침)를 효과적으로 처리할 수 있음을 입증하였다. 특히 LoRA를 이용한 효율적인 학습 방식과 서로 다른 특성을 가진 두 탐지 모델(DINO, YOLO-v8)을 앙상블한 전략이 성능 향상의 핵심 요인이었다.

### 한계 및 논의사항

- **모델 복잡도**: HQ-SAM과 DINO, YOLO-v8을 모두 사용하는 파이프라인은 추론 속도 면에서 단일 모델보다 느릴 수밖에 없다. 실제 농기계에 탑재하여 실시간으로 작동시키기 위해서는 모델 경량화 연구가 추가로 필요할 것으로 보인다.
- **AMP의 영향**: 잎 탐지 작업에서 AMP가 성능을 저하시킨다는 발견은 흥미롭다. 이는 데이터의 특성이나 클래스의 정밀도 요구 수준에 따라 혼합 정밀도 학습이 부정적인 영향을 줄 수 있음을 시사하며, 작업별 최적화 설정의 중요성을 강조한다.

## 📌 TL;DR

본 연구는 농업용 계층적 Panoptic Segmentation을 위해 **DINO/YOLO-v8 기반의 프롬프트 생성기**와 **LoRA로 미세 조정된 HQ-SAM**을 결합한 파이프라인을 제안하였다. 이 접근 방식은 특히 얇은 구조의 식물 분할에 강점을 보이며, $\text{PQ+} 81.33$이라는 높은 성적을 거두었다. 이 연구는 거대 분할 모델(SAM)을 특정 도메인에 효율적으로 적응시키는 방법론을 제시함으로써, 향후 정밀 농업을 위한 자동화된 식물 표현형 분석(Plant Phenotyping) 연구에 중요한 기여를 할 것으로 기대된다.
