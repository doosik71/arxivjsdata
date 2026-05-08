# EfficientPS: Efficient Panoptic Segmentation

Rohit Mohan, Abhinav Valada (2021)

## 🧩 Problem to Solve

본 논문은 자율 주행 로봇의 핵심 기능인 주변 환경 이해를 위해 'stuff'(배경, 도로, 하늘 등)와 'thing'(차량, 보행자 등)을 동시에 인식하는 Panoptic Segmentation 작업을 효율적으로 수행하는 것을 목표로 한다.

기존의 Panoptic Segmentation 접근 방식은 다음과 같은 문제점을 가지고 있다. 첫째, Semantic Segmentation과 Instance Segmentation을 위해 별도의 네트워크를 사용하거나 단순하게 결합하는 방식은 계산 오버헤드가 크고 학습의 중복성이 발생하며, 두 네트워크 간의 예측 결과가 서로 충돌하는 현상이 나타난다. 둘째, 현재 SOTA 성능을 보이는 모델들은 주로 ResNet-101이나 ResNeXt-101과 같은 무거운 Backbone을 사용하여 파라미터 수가 매우 많고 추론 속도가 느리다는 단점이 있다.

따라서 본 연구의 목표는 연산 효율성을 극대화하면서도 기존의 무거운 모델들보다 우수한 성능을 내는 Efficient Panoptic Segmentation(EfficientPS) 아키텍처를 설계하는 것이다.

## ✨ Key Contributions

EfficientPS의 핵심 설계 아이디어는 전체 파이프라인의 모든 단계에서 효율성과 성능의 최적의 트레이드-오프(Trade-off)를 찾는 것이다.

1. **효율적인 공유 Backbone**: EfficientNet-B5를 기반으로 하며, 정보의 단방향 흐름이라는 기존 FPN의 한계를 극복하기 위해 양방향 정보 흐름을 가능하게 하는 2-way Feature Pyramid Network (FPN)를 제안한다.
2. **스케일별 최적화된 Semantic Head**: 특징의 크기에 따라 Fine-grained한 특징을 추출하는 LSFE(Large Scale Feature Extractor)와 Long-range context를 캡처하는 DPC(Dense Prediction Cells)를 분리하여 적용하고, 이를 정렬하는 MC(Mismatch Correction) 모듈을 통해 경계선 정밀도를 높인다.
3. **경량화된 Instance Head**: Mask R-CNN을 기반으로 하되, 모든 표준 컨볼루션을 Depthwise Separable Convolution으로 교체하여 파라미터 수를 획기적으로 줄였다.
4. **적응형 Panoptic Fusion Module**: Semantic Head와 Instance Head의 출력 Logit 사이의 충돌을 해결하기 위해, 각 예측의 확신도(Confidence)에 따라 가중치를 조절하여 융합하는 파라미터 프리(Parameter-free) 융합 모듈을 제안한다.
5. **KITTI Panoptic 데이터셋 공개**: 벤치마크 데이터셋인 KITTI에 Panoptic Annotation을 직접 추가하여 새로운 데이터셋을 구축하고 공개하였다.

## 📎 Related Works

기존의 Panoptic Segmentation 연구는 크게 Top-down(Proposal-based) 방식과 Bottom-up(Proposal-free) 방식으로 나뉜다.

- **Top-down 방식**: Mask R-CNN과 같은 Instance Segmentation 모델을 기반으로 하며, 객체의 스케일 변화에 강인하지만 'stuff' 클래스와 'thing' 클래스 간의 겹침(Overlap) 문제가 발생한다.
- **Bottom-up 방식**: 픽셀 단위의 예측 후 그룹화하는 방식으로, 주로 Semantic Segmentation 기반의 모델들이 사용된다. 연산 속도는 빠를 수 있으나 객체의 크기 변화가 심한 경우 성능이 떨어진다.

본 논문은 객체의 큰 스케일 변화를 효과적으로 처리하기 위해 Top-down 방식을 채택하되, 기존 모델들이 사용하는 무거운 Backbone과 단순한 융합 방식의 한계를 극복하여 효율성을 높인 점이 차별점이다.

## 🛠️ Methodology

### 전체 파이프라인

EfficientPS는 **[Shared Backbone $\rightarrow$ Parallel Heads (Semantic & Instance) $\rightarrow$ Panoptic Fusion Module]**의 구조를 가진다.

### 1. Network Backbone

- **Encoder**: EfficientNet-B5를 사용한다. 이때, Localization 성능을 저하시킬 수 있는 Squeeze-and-Excitation(SE) 연결을 제거하고, 다중 GPU 학습 시 그라디언트 추정치를 개선하기 위해 모든 Batch Normalization을 Synchronized Inplace Activated Batch Normalization(iABN sync)으로 교체하였다.
- **2-way FPN**: 기존 FPN의 Top-down 경로 외에 Bottom-up 경로를 추가하여 양방향으로 정보를 융합한다. 각 스케일에서 $1 \times 1$ 컨볼루션으로 채널을 256으로 맞춘 뒤, 양방향 경로의 합을 구하고 $3 \times 3$ Depthwise Separable Convolution을 통해 최종 $P_4, P_8, P_{16}, P_{32}$ 특징 맵을 생성한다.

### 2. Semantic Segmentation Head

입력 특징 맵의 스케일에 따라 서로 다른 모듈을 적용한다.

- **LSFE (Large Scale Feature Extractor)**: $P_4, P_8$과 같은 대형 스케일 특징에서 세밀한 특징을 추출하기 위해 두 개의 $3 \times 3$ Depthwise Separable Convolution을 사용한다.
- **DPC (Dense Prediction Cells)**: $P_{16}, P_{32}$와 같은 소형 스케일 특징에서 광범위한 문맥(Context)을 캡처하기 위해 서로 다른 Dilation rate를 가진 병렬 Dilated Convolution 브랜치를 사용한다.
- **MC (Mismatch Correction)**: 소형 스케일의 문맥 특징과 대형 스케일의 세밀한 특징 사이의 불일치를 해결하고 정렬하여 객체의 경계선을 정교하게 다듬는다.

학습 시에는 가중치 픽셀 로그 손실(Weighted per-pixel log-loss)을 사용하며, 수식은 다음과 같다.
$$L_{pp}(\Theta) = -\sum_{i,j} w_{i,j}(p^*_{i,j})\log p_{i,j}$$
여기서 $w_{i,j}$는 예측 성능이 낮은 하위 25% 픽셀에 가중치를 부여하여 어려운 영역의 학습을 유도한다.

### 3. Instance Segmentation Head

Mask R-CNN의 구조를 유지하면서 효율성을 위해 다음을 수정하였다.

- 모든 표준 Convolution $\rightarrow$ Depthwise Separable Convolution
- Batch Normalization $\rightarrow$ iABN sync
- ReLU $\rightarrow$ Leaky ReLU

손실 함수는 RPN의 Objectness score loss($L_{os}$), Object proposal loss($L_{op}$), 그리고 두 번째 단계의 Classification loss($L_{cls}$), Bounding box loss($L_{bbx}$), Mask segmentation loss($L_{mask}$) 다섯 가지를 동일한 가중치로 합산하여 최적화한다.
$$L_{instance} = L_{os} + L_{op} + L_{cls} + L_{bbx} + L_{mask}$$

### 4. Panoptic Fusion Module

두 헤드의 출력을 적응적으로 융합하는 과정이다.

1. **필터링**: Confidence 임계값과 Overlap 임계값을 기준으로 불필요한 인스턴스를 제거하고 정렬한다.
2. **Logit 추출**: 인스턴스 헤드에서 나온 Mask Logit을 $ML_A$, 세만틱 헤드의 출력 중 해당 클래스 채널의 Logit을 $ML_B$라고 한다.
3. **적응형 융합**: 다음 수식을 통해 두 Logit을 융합하여 $FL$을 구한다.
$$FL = (\sigma(ML_A) + \sigma(ML_B)) \odot (ML_A + ML_B)$$
여기서 $\sigma$는 Sigmoid 함수이며 $\odot$은 Hadamard product(원소별 곱)이다. 이는 두 예측이 일치할 때 점수를 증폭시키고, 불일치할 때 감쇠시키는 효과를 준다.
4. **최종 출력**: 융합된 인스턴스 결과를 먼저 캔버스에 배치하고, 남은 빈 공간을 세만틱 헤드의 'stuff' 예측값으로 채워 최종 Panoptic Segmentation 결과를 생성한다.

## 📊 Results

### 실험 환경 및 데이터셋

- **데이터셋**: Cityscapes, KITTI (신규), Mapillary Vistas, Indian Driving Dataset (IDD)
- **지표**: Panoptic Quality (PQ), Segmentation Quality (SQ), Recognition Quality (RQ), AP, mIoU
- **하드웨어**: NVIDIA Titan X GPU

### 주요 결과

1. **성능**: Cityscapes 벤치마크 리더보드에서 PQ 66.4%로 **1위**를 기록하였다. 또한 Mapillary Vistas, KITTI, IDD 모든 데이터셋에서 기존 SOTA 모델들을 상회하는 성능을 보였다.
2. **효율성**:
    - **파라미터 수**: 40.89M으로 UPSNet(45.0M), Seamless(51.4M), Panoptic-DeepLab(46.7M)보다 적다.
    - **추론 속도**: $1024 \times 2048$ 해상도 기준 166ms로 가장 빠른 속도를 기록하였다.
3. **데이터셋 기여**: KITTI 데이터셋에 대한 Panoptic Annotation을 공개하여 향후 연구의 기반을 마련하였다.

### 정성적 분석

시각화 결과, EfficientPS는 심하게 가려진(Occluded) 객체를 탐지하는 능력이 뛰어나며, 특히 IDD와 같은 비정형 환경에서 도로와 인도(Sidewalk)의 경계선을 더 정확하게 구분하는 모습을 보였다. 이는 2-way FPN의 양방향 정보 흐름과 Semantic Head의 경계선 정밀화 능력이 기여한 결과로 분석된다.

## 🧠 Insights & Discussion

### 강점

- **최적의 효율성**: 모델의 경량화(Depthwise Separable Conv)와 강력한 Backbone(EfficientNet)의 조합으로 성능 하락 없이 연산 비용을 크게 낮추었다.
- **적응형 융합**: 단순한 합산이나 곱셈이 아닌, 확신도 기반의 융합 식을 통해 'thing'과 'stuff' 간의 충돌을 효과적으로 해결하였다.
- **양방향 특징 추출**: 2-way FPN을 통해 저수준의 텍스처 정보와 고수준의 시맨틱 정보를 동시에 활용함으로써 객체 인식 능력을 높였다.

### 한계 및 논의

- **출력 스트라이드(Output Stride)**: EfficientPS는 연산 효율을 위해 출력 스트라이드를 32로 설정하였다. 이로 인해 스트라이드를 16으로 사용하는 일부 Bottom-up 방식(예: Panoptic-DeepLab)보다 'stuff' 클래스에 대한 PQ 점수가 약간 낮게 나타나는 경향이 있다. 이는 효율성과 정밀도 사이의 트레이드-오프 결과이다.
- **가정**: 본 모델은 Top-down 방식의 우수성을 가정하고 설계되었으며, 향후 Bottom-up 방식의 강점(세밀한 세만틱 분할)을 결합한 하이브리드 구조에 대한 연구가 필요하다.

## 📌 TL;DR

본 논문은 효율적인 Panoptic Segmentation을 위한 **EfficientPS** 아키텍처를 제안한다. Modified EfficientNet-B5와 **2-way FPN**을 통해 가벼우면서도 강력한 Backbone을 구축하였고, **LSFE/DPC/MC**가 포함된 새로운 Semantic Head와 경량화된 Instance Head를 설계하였다. 특히, 두 헤드의 결과를 적응적으로 융합하는 **Panoptic Fusion Module**을 통해 예측 충돌을 해결하였다. 결과적으로 Cityscapes를 포함한 4개의 주요 벤치마크에서 SOTA 성능을 달성함과 동시에 가장 적은 파라미터와 가장 빠른 추론 속도를 기록하여, 실제 자율 주행 시스템에 적용 가능한 매우 효율적인 모델임을 입증하였다.
