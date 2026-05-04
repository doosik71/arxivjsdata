# TernausNetV2: Fully Convolutional Network for Instance Segmentation

Vladimir I. Iglovikov, Selim S. Seferbekov, Alexander V. Buslaev, Alexey A. Shvets (2018)

## 🧩 Problem to Solve

본 논문은 고해상도 위성 이미지에서 개별 건축물(building)을 인스턴스 수준에서 추출하는 Instance Segmentation 문제를 해결하고자 한다. 위성 이미지를 통한 자동 건축물 추출은 도시 계획 및 세계 인구 모니터링에 있어 매우 중요하다.

기존의 Semantic Segmentation 방식은 픽셀 단위의 분류만을 수행하므로, 건축물들이 서로 밀집해 있거나 맞닿아 있는 경우 이를 하나의 거대한 덩어리(connected blob)로 인식하는 한계가 있다. 즉, 서로 다른 개별 인스턴스를 구분하지 못하는 문제가 발생한다. 또한, 위성 데이터는 RGB 외에도 다양한 다중 분광(multispectral) 채널을 포함하고 있으나, 이를 효율적으로 활용하여 인스턴스 분할 성능을 높이는 방법론이 필요하다. 따라서 본 논문의 목표는 다중 분광 정보를 활용하면서도, 복잡한 2단계 네트워크 없이 Fully Convolutional Network(FCN) 구조만으로 효율적인 인스턴스 분할을 수행하는 TernausNetV2를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Semantic Segmentation을 위한 네트워크 구조를 유지하면서, 출력 채널을 확장하여 인스턴스 분리를 위한 추가 정보를 예측하게 함으로써 Instance Segmentation으로 확장하는 것이다.

주요 기여 사항은 다음과 같다.

1. **인스턴스 분리를 위한 다중 출력 설계**: 단순히 건축물 영역(binary mask)만 예측하는 것이 아니라, 객체들이 서로 맞닿아 있거나 매우 근접한 영역(touching borders)을 예측하는 추가 채널을 도입하였다. 이를 통해 후처리 단계에서 Watershed Transform을 적용하여 개별 인스턴스를 분리할 수 있게 하였다.
2. **강력하고 효율적인 Encoder 채택**: 기존 TernausNet의 VGG11 인코더를 In-place Activated Batch Normalization(ABN)이 적용된 WideResNet-38로 교체하여 메모리 효율성과 성능을 동시에 확보하였다.
3. **다중 분광 채널로의 Transfer Learning 기법**: RGB 이미지로 사전 학습된 가중치를 유지하면서 11개의 다중 분광 채널을 입력으로 받을 수 있도록 입력 레이어를 확장하고, 특정한 학습 스케줄을 통해 RGB에서 다중 분광 데이터로의 부드러운 전이를 가능케 하였다.

## 📎 Related Works

기존의 Instance Segmentation 접근 방식은 Object Proposal을 사용하는 2단계 네트워크, Conditional Random Fields(CRF), Template Matching 또는 Recurrent Neural Networks(RNN) 등을 사용하는 복잡한 구조가 주를 이루었다.

또한, U-Net과 같은 Encoder-Decoder 구조의 FCN은 Semantic Segmentation에서 뛰어난 성능을 보였으며, 특히 Skip Connection을 통해 저수준 특징 맵과 고수준 특징 맵을 결합함으로써 정밀한 픽셀 수준의 localization이 가능함을 입증하였다. TernausNet은 이러한 U-Net의 인코더를 VGG11로 대체하여 성능을 높인 모델이다.

본 논문에서 언급된 다른 접근 방식 중에는 semantic segmentation, distance transform의 gradient, 그리고 energy level을 각각 예측하는 3개의 네트워크를 쌓아 Watershed Transform에 활용하는 방법이 있으나, 본 논문의 TernausNetV2는 이보다 훨씬 직관적이고 단순한 구조를 지향하며 동일한 목적을 달성한다.

## 🛠️ Methodology

### 전체 아키텍처 및 구성 요소

TernausNetV2는 기본적으로 U-Net의 Encoder-Decoder 구조를 따른다.

- **Encoder**: ImageNet으로 사전 학습된 ABN WideResNet-38의 처음 5개 합성곱 블록을 사용한다. ABN(In-place Activated Batch Normalization)은 배치 정규화 층과 활성화 층을 병합하여 메모리 사용량을 최대 50%까지 줄여주며, 이는 더 큰 배치 사이즈나 고해상도 이미지 처리를 가능하게 한다.
- **Decoder**: 5개의 디코더 블록으로 구성되며, 각 블록은 인코더의 대응하는 블록과 Skip Connection으로 연결되어 특징 맵을 Concatenation한다. 각 블록은 두 세트의 $3 \times 3$ Convolution과 ReLU 활성화 함수, 그리고 Nearest Neighbor Upsampling 층으로 이루어져 있다.
- **Output Layer**: 최종적으로 $1 \times 1$ Convolution을 통해 출력 채널을 2개로 줄인다.
  - **Channel 1**: 건축물 전체의 이진 마스크(Binary Mask)
  - **Channel 2**: 건축물 간의 맞닿은 경계(Touching Borders)

### 다중 분광 입력 및 전이 학습

입력 데이터는 RGB와 8개의 추가 채널을 포함한 총 11채널 이미지이다. RGB로 사전 학습된 인코더를 활용하기 위해 다음과 같은 전략을 사용한다.

1. 첫 번째 Convolution 레이어를 11채널 입력이 가능하도록 교체한다.
2. 기존 RGB 가중치를 처음 3개 채널에 복사하고, 나머지 8개 채널의 가중치는 0으로 초기화한다.
3. **학습 스케줄**: 첫 번째 에포크에서는 인코더를 동결(freeze)하고 디코더만 학습시킨다. 이후 두 번째 에포크부터 모든 레이어를 해제하여 end-to-end로 학습함으로써, 네트워크가 RGB 정보에서 다중 분광 정보로 서서히 적응하도록 유도한다.

### 손실 함수 (Loss Function)

네트워크는 두 개의 독립적인 이진 마스크를 예측해야 하므로, Binary Cross Entropy($H$)와 Soft Jaccard Loss($J$)를 결합하여 사용한다.

Soft Jaccard Index는 다음과 같이 정의된다:
$$J = \frac{1}{n} \sum_{c=1}^{2} w_c \frac{n \sum_{i=1}^{n} (y_i^c \hat{y}_i^c)}{n \sum_{i=1}^{n} (y_i^c + \hat{y}_i^c - y_i^c \hat{y}_i^c)}$$
여기서 $y_i^c$는 정답 레이블, $\hat{y}_i^c$는 예측 확률이다. 최종 손실 함수 $L$은 다음과 같다:
$$L = \alpha H + (1 - \alpha)(1 - J)$$
본 연구에서는 $\alpha = 0.7$을 사용하여 픽셀 단위 분류 정확도와 IoU(Intersection over Union)를 동시에 최적화하였다.

### 추론 및 후처리 (Inference & Post-processing)

인공지능 모델이 출력한 두 채널의 마스크를 이용하여 인스턴스를 분리한다.

1. 예측된 이진 마스크에서 맞닿은 경계(touching borders) 영역을 뺀다.
2. 이를 통해 생성된 영역을 Seed로 사용하고, 원래의 이진 마스크를 영역으로 하여 **Watershed Transform**을 수행한다.
3. 이 과정을 통해 서로 붙어 있던 건축물들이 개별 인스턴스로 분리된다.

## 📊 Results

### 실험 설정

- **데이터셋**: SpaceNet 데이터셋 (WorldView-3 위성 이미지, 30cm 해상도).
- **도시 범위**: 라스베이거스, 파리, 상하이, 하르툼 4개 도시.
- **이미지 크기**: 입력 $650 \times 650$ 픽셀 (추론 시 $672 \times 672$로 패딩 후 크롭).
- **학습 환경**: 4x GTX 1080 Ti, Adam optimizer (lr=1e-4), 800 에포크 학습.
- **데이터 증강**: Random resize (0.5~1.5), Random rotation (0~360도), Contrast/Brightness/Gamma correction 적용.

### 결과

DeepGlobe-CVPR 2018 건축물 탐지 챌린지의 공개 리더보드 점수를 기준으로, TernausNetV2는 **0.74**의 점수를 기록하며 다른 방법론 대비 우수한 성능을 보였다. 별도의 앙상블(bagging, TTA 등)이나 도시별 미세 조정(fine-tuning) 없이도 state-of-the-art 결과에 도달하였다.

## 🧠 Insights & Discussion

### 강점 및 해석

TernausNetV2는 매우 복잡한 Instance Segmentation 파이프라인(예: Mask R-CNN의 Proposal 단계)을 거치지 않고도, 단순한 FCN 구조와 후처리(Watershed)의 조합만으로 높은 성능을 낼 수 있음을 보여주었다. 특히, "맞닿은 경계"를 예측한다는 직관적인 아이디어가 인스턴스 분리 문제를 효과적으로 해결하였다. 또한, ABN WideResNet-38의 채택은 메모리 효율성을 극대화하여 고해상도 위성 이미지 처리에 실질적인 이점을 제공하였다.

### 한계 및 논의사항

본 논문은 후처리 단계에서 Watershed Transform에 크게 의존하고 있다. 만약 네트워크가 'touching borders'를 정확하게 예측하지 못할 경우, 인스턴스 분리가 과하게 일어나거나(over-segmentation) 여전히 분리되지 않는(under-segmentation) 문제가 발생할 수 있다. 또한, 위성 이미지의 특성상 도시마다 건축 양식이 매우 다르므로, 본 논문에서 수행하지 않은 도시별 미세 조정(city-specific fine-tuning)이 추가된다면 더 높은 성능 향상이 가능할 것으로 판단된다.

## 📌 TL;DR

TernausNetV2는 ABN WideResNet-38 인코더를 탑재한 U-Net 기반의 FCN으로, 건축물 마스크와 함께 '맞닿은 경계'를 동시에 예측하여 Watershed Transform으로 개별 건축물을 분리하는 Instance Segmentation 모델이다. RGB 사전 학습 가중치를 다중 분광 채널(11채널)로 확장하는 효율적인 전이 학습 전략을 제시하였으며, DeepGlobe-CVPR 2018 챌린지에서 0.74의 높은 점수를 기록하며 위성 이미지 분석에서의 실용성과 성능을 입증하였다.
