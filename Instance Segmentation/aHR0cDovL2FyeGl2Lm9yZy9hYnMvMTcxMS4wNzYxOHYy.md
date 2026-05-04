# S4Net: Single Stage Salient-Instance Segmentation

Ruochen Fan, Ming-Ming Cheng, Qibin Hou, Tai-Jiang Mu, Jingdong Wang, Shi-Min Hu (2019)

## 🧩 Problem to Solve

본 논문은 **Salient Instance Segmentation(돌출 인스턴스 분할)** 문제를 해결하고자 한다. 일반적인 객체 검출(Object Detection)이 단순히 바운딩 박스를 생성하는 것에 그치고, 일반적인 인스턴스 분할(Instance Segmentation)이 미리 정의된 특정 카테고리에 의존하는 것과 달리, 본 연구의 목표는 장면 내에서 시각적으로 가장 눈에 띄는(salient) 객체들을 카테고리에 구애받지 않고(category-agnostic) 개별적으로 분리하여 고품질의 마스크를 생성하는 것이다.

이 문제의 중요성은 이미지 편집, 로봇 인지 등 다양한 컴퓨터 비전 응용 분야에서 사용자가 관심을 가질만한 영역을 자동으로 추출하는 기초 단계로 활용될 수 있다는 점에 있다. 특히, 기존의 CNN 기반 방법론들이 객체 내부의 특징(appearance variations)에만 집중하여 객체와 주변 배경 간의 특징 분리(feature separation) 능력을 간과하고 있다는 점을 지적하며, 이를 해결하여 실시간으로 동작하는 정교한 분할 프레임워크를 구축하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 전통적인 Figure-ground segmentation 방식(예: GrabCut)에서 영감을 얻어, **객체 영역과 주변 배경 영역 간의 특징 대비를 명시적으로 모델링**하는 것이다. 이를 위해 다음과 같은 기여를 제시한다.

1.  **S4Net 프레임워크**: 실시간 처리가 가능하면서도 높은 성능을 내는 end-to-end 단일 단계(single-stage) 돌출 인스턴스 분할 프레임워크를 제안한다.
2.  **RoIMasking Layer**: 기존의 RoIPooling이나 RoIAlign이 관심 영역(RoI) 내부의 특징만을 추출하는 것과 달리, RoI 주변의 배경 정보를 함께 활용하여 전경과 배경의 분리를 명시적으로 학습시키는 새로운 레이어를 설계하였다.
3.  **Salient Instance Discriminator (SID)**: 수용 영역(receptive field)을 확장하여 돌출 인스턴스를 더 정교하게 구분할 수 있는 분할 브랜치를 설계하였다.

## 📎 Related Works

본 논문은 돌출 인스턴스 분할과 밀접한 세 가지 분야를 언급한다.

1.  **Salient Object Detection**: 가장 눈에 띄는 객체를 검출하고 분할하지만, 이진 분류(binary problem) 형태이므로 개별 인스턴스를 구분하는 능력이 부족하며 객체의 무결성(integrity)을 유지하기 어렵다는 한계가 있다.
2.  **Object Detection**: 바운딩 박스를 생성하는 데 집중하며, 픽셀 수준의 정교한 분할 마스크를 생성하지 않는다.
3.  **Semantic Instance Segmentation (예: Mask R-CNN)**: 정교한 분할이 가능하지만, 모든 객체가 돌출된(salient) 것은 아니며, 미리 정의된 카테고리 집합에 의존하므로 미지의 카테고리를 처리할 수 없는 class-agnostic 특성이 결여되어 있다.

특히, 기존의 Mask R-CNN과 같은 방식은 RoI 내부 특징에만 집중하여 인스턴스 간의 구분을 명확히 하지 못한다는 점이 본 연구의 차별점이다.

## 🛠️ Methodology

### 전체 시스템 구조
S4Net은 크게 **Bounding Box Detector**와 **Segmentation Branch**라는 두 개의 형제 브랜치로 구성되며, ResNet-50과 FPN(Feature Pyramid Network)을 백본으로 공유한다. 

1.  **Bounding Box Detector**: FPN의 다중 레벨 특징을 활용하여 객체의 위치와 클래스를 예측하는 단일 단계 검출기이다.
2.  **Segmentation Branch**: 검출기에서 예측된 바운딩 박스와 백본의 특징 맵(stride 8)을 입력으로 받아, RoIMasking 레이어를 거쳐 최종적으로 인스턴스 마스크를 생성한다.

### RoIMasking Layer
본 논문의 핵심으로, RoI 내부뿐만 아니라 주변 맥락을 활용하기 위해 세 가지 단계를 거쳐 발전시켰다.

*   **Binary RoIMasking**: 제안된 바운딩 박스 내부 영역은 $1$, 외부는 $0$으로 설정하여 특징 맵에 곱한다. 이는 기존 RoIAlign과 달리 원래의 해상도와 종횡비를 유지한다.
*   **Expanded Binary RoIMasking**: 마스크의 영역을 확장하여 더 많은 배경/맥락 정보를 포함시킨다.
*   **Ternary RoIMasking**: 가장 발전된 형태로, 바운딩 박스 내부는 $1$, 주변의 일정 영역은 $-1$, 그 외 지역은 $0$으로 설정한다. 특징 맵의 값에 $-1$을 곱함으로써 전경과 배경의 특징 부호를 반전시켜, 네트워크가 전경-배경 간의 대비를 명시적으로 학습하게 한다.

### Segmentation Branch (SID)
RoIMasking 이후의 단계인 Salient Instance Discriminator는 수용 영역을 넓히기 위해 다음과 같은 구조를 갖는다.
*   $1 \times 1$ Convolution을 통한 채널 압축 (256 채널)
*   Skip Connection 및 Dilated Convolution (dilation rate 2) 적용
*   $3 \times 3$ Max Pooling (stride 1) 적용
이 구조는 Mask R-CNN의 분할 브랜치와 파라미터 수는 비슷하지만 훨씬 넓은 수용 영역을 가진다.

### 손실 함수 (Loss Function)
모델은 다음의 다중 작업 손실 함수를 통해 end-to-end로 학습된다.

$$L = L_{obj} + L_{coord} + L_{seg}$$

여기서 $L_{obj}$는 객체 존재 여부에 대한 분류 손실, $L_{coord}$는 좌표 회귀 손실(Smooth $L_1$ loss), $L_{seg}$는 분할 손실(Cross-entropy loss)이다. 특히 $L_{obj}$의 경우, 양성 샘플($P$)과 음성 샘플($N$)의 불균형을 해결하기 위해 다음과 같이 각각 계산하여 합산한다.

$$L_{obj} = -\left( \frac{1}{N_P} \sum_{i \in P} \log p_i + \frac{1}{N_N} \sum_{j \in N} \log(1-p_j) \right)$$

## 📊 Results

### 실험 설정
*   **데이터셋**: 1,000장의 이미지로 구성된 돌출 인스턴스 분할 데이터셋을 사용 (학습 500, 검증 200, 테스트 300).
*   **지표**: mAP 0.5, mAP 0.7 및 폐색(occlusion)이 있는 샘플에 대한 $mAP^O$를 측정하였다.
*   **비교 대상**: RoIPool, RoIAlign 및 기존의 MSRNet.

### 주요 결과
1.  **RoIMasking의 효과**: Ternary RoIMasking이 RoIAlign 대비 mAP 0.7에서 약 2.1%p 향상된 성능을 보였다. 특히 폐색이 있는 이미지($mAP^O$)에서 더 강건한 성능을 나타냈다.
2.  **맥락 영역의 크기 ($\alpha$)**: 바운딩 박스 크기 대비 확장 계수 $\alpha = 1/3$일 때 최적의 성능을 보였으며, 너무 크거나 작으면 성능이 하락하였다.
3.  **속도 및 효율성**:
    *   ResNet-50 기반: $320 \times 320$ 이미지 기준 40 FPS (GTX 1080 Ti).
    *   MobileNet 기반: 90.9 FPS까지 가속 가능.
4.  **SOTA 비교**: MSRNet과 비교했을 때 mAP 0.5 기준 약 21%p라는 압도적인 성능 향상을 달성하였다.

### 응용 실험
S4Net을 약지도 학습 기반 시맨틱 분할(Weakly-supervised Semantic Segmentation)의 힌트(heuristic cues)로 사용한 결과, PASCAL VOC 벤치마크에서 기존의 돌출 객체 검출(SOD) 힌트를 사용했을 때보다 높은 성능(ResNet-101 기준 61.8%)을 기록하였다.

## 🧠 Insights & Discussion

본 논문은 딥러닝 모델이 단순히 내부 특징만을 학습하는 것이 아니라, 전경-배경의 '대비'라는 고전적인 시각 원리를 네트워크 구조(RoIMasking)에 직접 통합했을 때 성능이 크게 향상됨을 입증하였다. 특히 Ternary Masking을 통해 배경 영역에 음수 값을 부여함으로써 특징 공간에서의 분리도를 높인 점이 매우 영리한 설계이다.

**강점**:
*   단일 단계 구조로 실시간성을 확보하면서도 고정밀 분할이 가능하다.
*   카테고리 무관하게 돌출된 인스턴스를 찾을 수 있어 범용성이 높다.

**한계 및 논의**:
*   본 모델은 여전히 Bounding Box Detector의 성능에 의존한다. 만약 초기 검출 단계에서 박스가 잘못 생성된다면 RoIMasking이 이를 보완하기 어려울 수 있다.
*   $\alpha$ 값에 대한 최적화가 실험적으로 이루어졌는데, 이미지의 스케일이나 객체 밀도에 따라 이 최적값이 달라질 가능성이 있다.

## 📌 TL;DR

S4Net은 실시간으로 작동하는 단일 단계 돌출 인스턴스 분할 프레임워크이다. 기존의 RoI 추출 방식이 내부 특징에만 집중하는 한계를 극복하기 위해, 주변 배경의 부호를 반전시켜 특징 대비를 극대화하는 **Ternary RoIMasking** 레이어를 제안하였다. 이를 통해 MSRNet 등 기존 모델보다 월등한 정확도를 달성했으며, 특히 폐색 상황에서도 강건한 분할 성능을 보여준다. 이 연구는 향후 실시간 이미지 편집이나 약지도 학습 기반의 분할 연구에 중요한 기초 기술로 활용될 가능성이 크다.