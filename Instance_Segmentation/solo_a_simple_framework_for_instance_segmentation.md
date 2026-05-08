# SOLO: A Simple Framework for Instance Segmentation

Xinlong Wang, Rufeng Zhang, Chunhua Shen, Tao Kong, Lei Li

## 🧩 Problem to Solve

기존 인스턴스 분할(Instance Segmentation) 방법론들은 객체의 수가 가변적이고 픽셀 단위 분할이 필요하다는 점에서 복잡하며 다음과 같은 한계를 가졌습니다.

* **간접적인 접근 방식**: 대부분 "감지 후 분할(detect-then-segment)" 전략(예: Mask R-CNN)을 따르거나, 픽셀 임베딩을 예측한 후 픽셀을 클러스터링하여 개별 인스턴스를 분리합니다.
* **의존성**: 이러한 방법들은 정확한 바운딩 박스 감지나 후처리(예: 픽셀 그룹화)에 크게 의존합니다.
이 논문은 바운딩 박스 제안이나 픽셀 그룹화 과정 없이, 입력 이미지로부터 원하는 객체 카테고리와 인스턴스 마스크를 직접 예측하는 간단하고 직접적인 프레임워크를 제안하여 이러한 문제들을 해결하고자 합니다.

## ✨ Key Contributions

* **"인스턴스 카테고리" 개념 도입**: 인스턴스의 위치에 따라 각 픽셀에 카테고리를 할당하는 새로운 관점을 제시하여 인스턴스 분할을 재구성합니다.
* **SOLO(Segmenting Objects by LOcations) 프레임워크**: 바운딩 박스 감지나 픽셀 클러스터링 없이 이미지를 객체 카테고리 및 인스턴스 마스크로 직접 매핑하는 간단하고 직접적이며 빠른 인스턴스 분할 프레임워크를 제안합니다.
* **SOLO 변형 제안**: 기본 원리에 따라 Vanilla SOLO, Decoupled SOLO, Dynamic SOLO(SOLOv2) 등 몇 가지 효율적인 변형 모델을 개발했습니다.
* **Matrix NMS 알고리즘**: 중복 예측을 병렬 행렬 연산으로 효율적으로 제거하는 Matrix NMS를 제안했습니다. 이는 기존 NMS보다 속도와 정확도 모두에서 우수한 성능을 보입니다.
* **최고 수준의 성능 달성**: MS COCO 데이터셋에서 인스턴스 분할, 객체 감지(마스크 부산물), 파놉틱 분할 및 인스턴스 레벨 이미지 매팅 등 다양한 작업에서 속도와 정확도 면에서 SOTA(State-of-the-Art) 결과를 달성했습니다.
* **Box-Free 패러다임**: 바운딩 박스나 앵커 박스에 대한 의존성을 완전히 제거하여 FCN(Fully Convolutional Network)의 고유한 장점을 활용합니다.

## 📎 Related Works

* **하향식(Top-down) 인스턴스 분할**: FCIS [15], Mask R-CNN [16], PANet [26], Mask Scoring R-CNN [27], HTC [17], TensorMask [18], YOLACT [25]. (SOLO는 이들과 달리 박스에 제약받지 않고 직접 마스크를 예측합니다.)
* **상향식(Bottom-up) 인스턴스 분할**: Associative Embedding [19], Discriminative Loss [20], SGN [21], SSAP [22]. (SOLO는 픽셀 간 관계나 픽셀 그룹화 없이 인스턴스 마스크 주석만으로 직접 학습합니다.)
* **직접 인스턴스 분할**: AdaptIS [28], PolarMask [29]. (SOLO는 이들보다 더 직접적으로 마스크와 semantic category를 한 번에 예측합니다.)
* **동적 컨볼루션(Dynamic convolutions)**: Dynamic filter [31], Deformable Convolutional Networks [32], Pixel-adaptive convolution [33], CondInst [34]. (SOLOv2는 이 개념을 인스턴스 분할에 적용하여 위치에 따라 세그멘터를 동적으로 학습합니다.)
* **NMS(Non-maximum suppression)**: Soft-NMS [9], NMS refinement [10], MaxPoolNMS [11], Fast NMS [25], Adaptive NMS [36]. (Matrix NMS는 기존 NMS의 순차적 처리 및 하드 제거 문제를 개선합니다.)

## 🛠️ Methodology

1. **문제 정의 (Problem Formulation)**
    * 인스턴스 분할을 두 가지 동시 예측 문제로 재구성합니다.
    * **Semantic Category 예측**: 입력 이미지를 $S \times S$ 그리드로 개념적으로 나눕니다. 객체의 중심이 속한 그리드 셀이 해당 객체의 semantic category를 예측합니다. 출력은 $S \times S \times C$ 차원입니다.
    * **인스턴스 마스크 생성**: 각 양성 그리드 셀은 해당 객체 인스턴스의 마스크를 생성합니다. 출력은 $H_I \times W_I \times S^2$ 차원이며, $k$-th 채널은 그리드 $(i,j)$에 해당하는 인스턴스 분할을 담당합니다 ($k = i \cdot S + j$).
    * **공간 변이 (Spatial Variance)**: 컨볼루션이 위치에 민감한 예측을 수행하도록 `CoordConv` [38]를 사용하여 정규화된 픽셀 좌표를 입력 피처에 추가합니다.

2. **네트워크 아키텍처 (Network Architecture)**
    * ResNet [40] 백본과 FPN [6]을 활용하여 다양한 스케일의 피처 맵을 생성하고, 이 피처 맵을 입력으로 두 개의 병렬 브랜치(카테고리 예측 및 인스턴스 마스크 생성)를 사용합니다.
    * **Vanilla SOLO**: 가장 간단한 아키텍처로, 입력 이미지를 $S^2$ 채널의 출력 텐서 $M$으로 직접 매핑합니다.
    * **Decoupled SOLO**: 출력 텐서 $M \in \mathbb{R}^{H \times W \times S^2}$를 두 개의 출력 텐서 $X \in \mathbb{R}^{H \times W \times S}$ 및 $Y \in \mathbb{R}^{H \times W \times S}$로 분리합니다. 그리드 위치 $(i,j)$의 마스크는 $m_k = x_j \otimes y_i$ (요소별 곱셈)로 정의되어 출력 공간을 $H \times W \times 2S$로 줄입니다.
    * **Dynamic SOLO (SOLOv2)**: 마스크 예측을 마스크 커널 학습($G \in \mathbb{R}^{S \times S \times D}$)과 마스크 피처 학습($F \in \mathbb{R}^{H \times W \times E}$)으로 분리합니다. 최종 마스크는 동적으로 생성된 커널을 사용하여 $M_{i,j} = G_{i,j} \sim F$ (동적 컨볼루션)로 생성됩니다.
    * **Decoupled Dynamic SOLO**: Dynamic SOLO에서 마스크 커널 $G$를 $A$개 그룹으로 나누고 마스크 피처 $F$의 채널을 줄여 더욱 통합된 알고리즘을 제안합니다. 마스크는 $M = \text{sigmoid}(G_1 \sim F) \otimes \dots \otimes \text{sigmoid}(G_A \sim F)$로 예측됩니다.

3. **학습 및 추론 (Learning and Inference)**
    * **레이블 할당**: 객체 마스크의 중심 영역에 해당하는 그리드 셀을 양성 샘플로 간주하고, 각 양성 샘플에 이진 마스크를 할당합니다.
    * **손실 함수**: $L = L_{\text{cate}} + \lambda L_{\text{mask}}$로 정의됩니다.
        * $L_{\text{cate}}$: semantic category 분류를 위한 Focal Loss [2].
        * $L_{\text{mask}}$: 마스크 예측을 위한 Dice Loss [46] (효과성 및 학습 안정성 때문에 선택).
    * **추론 파이프라인**:
        * 백본 네트워크와 FPN을 통해 카테고리 점수 $p_{i,j}$를 얻습니다.
        * 낮은 신뢰도 예측을 임계값(0.1)으로 필터링합니다.
        * Vanilla/Decoupled SOLO는 상위 500개 마스크를 NMS에 전달합니다.
        * Dynamic SOLO는 예측된 마스크 커널로 마스크 피처에 컨볼루션을 수행하여 소프트 마스크를 생성합니다.
        * 소프트 마스크를 임계값(0.5)으로 이진 마스크로 변환합니다.
        * **Maskness**: 예측된 소프트 마스크의 전경 픽셀에 대한 평균값을 계산하여 마스크 품질을 나타내는 Maskness를 분류 점수에 곱하여 최종 신뢰도를 결정합니다.
        * **Matrix NMS**: 병렬로 동작하는 Matrix NMS를 사용하여 중복 예측을 효율적으로 제거합니다.

## 📊 Results

* **MS COCO 인스턴스 분할**:
  * SOLOv2(ResNet-101 백본)는 39.7%의 마스크 $AP$를 달성하여 기존 SOTA 방법들을 능가하며, 특히 큰 객체($AP_L$)에서 +5.0% $AP$ 향상을 보였습니다.
  * 속도-정확도 트레이드오프에서 Mask R-CNN, YOLACT 등 경쟁 모델보다 뛰어난 성능을 보였습니다. (예: ResNet-50 백본으로 38.8% mask $AP$ @ 18 FPS, 경량 버전은 37.1% mask $AP$ @ 31.3 FPS).
  * Mask R-CNN보다 훨씬 세밀한 마스크, 특히 객체 경계 부분에서 고품질 예측을 제공합니다.
* **LVIS 인스턴스 분할**: SOLOv2는 Mask R-CNN baseline 대비 약 1% $AP$ 향상(Res-50-FPN: 25.5% vs 24.6% $AP$)을 달성했으며, 큰 객체($AP_L$)에서 6.7% $AP$ 향상을 보였습니다.
* **Cityscapes 인스턴스 분할**: SOLOv2는 Mask R-CNN보다 0.9% $AP$ 향상(Res-50-FPN: 37.4% vs 36.5% $AP$)을 기록했습니다.
* **SOLO 및 SOLOv2의 ABLATION 연구**:
  * **그리드 수 및 FPN**: FPN을 사용한 다중 스케일 예측은 단일 스케일 예측 대비 35.8% $AP$로 성능을 크게 향상시켰습니다.
  * **CoordConv**: `CoordConv` 사용 시 3.6% $AP$ 향상을 가져왔으며, 위치 민감성 부여에 중요합니다.
  * **손실 함수**: Dice Loss가 BCE 및 Focal Loss보다 뛰어난 성능과 학습 안정성을 보였습니다 (35.8% $AP$).
  * **Matrix NMS**: 기존 NMS 대비 Matrix NMS는 속도(1ms 미만 vs 22ms)와 정확도(0.4% $AP$) 모두에서 우수했습니다.
* **확장성**: 객체 감지, 파놉틱 분할, 인스턴스 레벨 이미지 매팅에서도 SOTA 또는 경쟁력 있는 성능을 달성했습니다. 특히, 바운딩 박스 기반 훈련 없이 마스크에서 직접 생성된 바운딩 박스로 객체 감지에서 44.9% $AP$를 달성했습니다.

## 🧠 Insights & Discussion

* **인스턴스 분할에 대한 새로운 패러다임**: SOLO는 기존의 복잡하고 간접적인 "감지 후 분할" 또는 "클러스터링" 방식에서 벗어나, "인스턴스 카테고리"라는 직관적인 개념을 도입하여 인스턴스 분할 문제를 단순한 픽셀 단위 분류 문제로 재정의했습니다. 이는 바운딩 박스 제안에 의존하지 않는 엔드투엔드(end-to-end) 솔루션을 가능하게 합니다.
* **간결함과 효율성**: SOLO 프레임워크는 기존의 복잡한 방법론들보다 훨씬 간단한 구조에도 불구하고 강력한 성능을 보여주며, 실시간 인스턴스 분할에 적합함을 입증했습니다. 이는 바운딩 박스에 대한 의존성을 제거하고 FCN의 본질적인 장점을 활용한 결과입니다.
* **Matrix NMS의 혁신**: NMS는 기존 객체 감지 및 인스턴스 분할 시스템의 주요 병목 중 하나였습니다. Matrix NMS는 이 과정을 병렬 행렬 연산으로 재해석하여 속도와 정확도 모두에서 상당한 개선을 이루었으며, 이는 시스템 전반의 효율성을 크게 향상시킵니다.
* **뛰어난 유연성과 일반화 능력**: SOLO는 인스턴스 분할뿐만 아니라 객체 감지, 파놉틱 분할, 그리고 새로운 태스크인 인스턴스 레벨 이미지 매팅에 이르기까지 다양한 하위 비전 작업에 최소한의 수정으로 쉽게 확장될 수 있음을 보여주었습니다. 이는 SOLO 프레임워크의 광범위한 적용 가능성과 견고함을 증명합니다.
* **고품질 마스크 생성**: 특히 객체 경계 부분에서 Mask R-CNN과 같은 기존 방법보다 훨씬 높은 품질의 마스크를 생성하는 능력은 이미지 편집, 증강 현실과 같이 정밀한 픽셀 단위 조작이 필요한 애플리케이션에 대한 중요한 가능성을 열어줍니다.

## 📌 TL;DR

본 논문은 인스턴스 분할 문제를 "인스턴스 카테고리" 개념을 도입하여 직접적인 픽셀 분류 문제로 재구성하는 **SOLO(Segmenting Objects by LOcations)** 프레임워크를 제안합니다. SOLO는 바운딩 박스 감지나 픽셀 그룹화 없이 입력 이미지로부터 객체 카테고리 및 고품질 인스턴스 마스크를 직접 예측합니다. Vanilla, Decoupled, Dynamic (SOLOv2) 변형들을 통해 효율성을 높였으며, 특히 병렬 처리가 가능한 **Matrix NMS**를 개발하여 NMS의 속도와 정확도를 혁신적으로 개선했습니다. SOLO는 MS COCO 등 주요 벤치마크에서 **최고 수준의 성능**을 달성함과 동시에 객체 감지, 파놉틱 분할, 인스턴스 레벨 이미지 매팅 등 다양한 비전 작업에 **높은 유연성으로 확장**될 수 있음을 입증하며, 인스턴스 레벨 인식 태스크를 위한 강력하고 효율적인 기반을 제공합니다.
