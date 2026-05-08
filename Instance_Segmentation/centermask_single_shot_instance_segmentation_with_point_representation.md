# CenterMask: single shot instance segmentation with point representation

Yuqing Wang, Zhaoliang Xu, Hao Shen, Baoshan Cheng, Lirong Yang (2020)

## 🧩 Problem to Solve

본 논문은 One-stage instance segmentation에서 발생하는 두 가지 핵심적인 난제를 해결하고자 한다. 첫째는 동일한 카테고리에 속한 여러 객체 인스턴스들을 어떻게 효과적으로 구분할 것인가 하는 인스턴스 차별화(object instances differentiation) 문제이다. 특히 객체들이 서로 겹쳐 있는(overlapping) 상황에서 기존의 Global-area 기반 방식들은 인스턴스를 분리하는 데 어려움을 겪는다. 둘째는 픽셀 단위의 정밀한 위치 정보를 보존하는 픽셀-와이즈 특징 정렬(pixel-wise feature alignment) 문제이다. 기존의 One-stage 방식들은 마스크 표현 방식의 한계로 인해 경계선 부분이 거칠게 표현되는 픽셀 정렬 문제가 발생하며, 이를 해결하려는 시도(예: TensorMask)는 연산 복잡도를 높여 속도를 크게 저하시키는 결과를 초래한다. 따라서 본 연구의 목표는 단순하고 빠르면서도 정확한, Anchor-box free 기반의 One-stage instance segmentation 모델을 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 인스턴스 세그멘테이션 작업을 두 개의 병렬적인 서브태스크, 즉 'Local Shape 예측'과 'Global Saliency 생성'으로 분해하여 처리하는 것이다.

1. **Local Shape representation**: 객체의 중심점(center point) 표현을 이용하여 각 인스턴스에 대한 거친(coarse) 마스크를 예측함으로써, 객체가 겹쳐 있는 상황에서도 인스턴스들을 효과적으로 분리한다.
2. **Global Saliency Map**: 이미지 전체에 대해 픽셀 단위의 세그멘테이션을 수행하여 정밀한 디테일을 제공하고 픽셀 정렬 문제를 자연스럽게 해결한다.
3. **Assembly**: 인스턴스 인식 능력이 뛰어나지만 정밀도가 낮은 Local Shape와, 정밀도는 높지만 인스턴스 구분 능력이 없는 Global Saliency Map을 결합하여 최종 마스크를 생성한다.

## 📎 Related Works

인스턴스 세그멘테이션 연구는 크게 Two-stage와 One-stage 방식으로 나뉜다.

- **Two-stage 방식**: Mask R-CNN, PANet, Mask Scoring R-CNN 등이 대표적이며, '탐지 후 세그멘테이션(detect-then-segment)' 패러다임을 따른다. RoIAlign과 같은 모듈을 통해 픽셀 정렬 문제를 해결하여 높은 정확도를 보이지만, 구조가 복잡하고 처리 속도가 느리다는 한계가 있다.
- **One-stage 방식**:
  - **Global-area 기반**: InstanceFCN, YOLACT 등이 있으며, 전체 이미지의 특징 맵을 생성한 뒤 이를 조합해 마스크를 만든다. 픽셀 정렬은 우수하지만 객체 중첩 시 성능이 저하된다.
  - **Local-area 기반**: PolarMask(폴라 좌표계 표현), TensorMask(4D 텐서 표현) 등이 있다. 인스턴스 구분은 상대적으로 용이하나, 폴리곤 표현의 한계로 마스크가 부정확하거나(PolarMask), 복잡한 정렬 연산으로 인해 속도가 매우 느린(TensorMask) 문제가 있다.

CenterMask는 위 두 방식의 장점을 모두 취하기 위해 Local Shape와 Global Saliency 브랜치를 모두 포함하는 구조를 제안하여 차별점을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조

CenterMask의 전체 파이프라인은 공유 백본 네트워크(Backbone) 뒤에 5개의 헤드(Head)가 병렬로 연결된 구조이다. 각 헤드는 Heatmap, Offset, Shape, Size, Saliency 맵을 예측하며, 최종적으로 Local Shape와 Global Saliency Map을 곱하여 인스턴스 마스크를 생성한다.

### 주요 구성 요소 및 역할

1. **Local Shape Prediction**:
    - **Shape Head**: 중심점 위치에서의 특징을 추출하여 $S \times S$ 크기의 고정된 2D 이진 배열(shape vector)을 예측한다.
    - **Size Head**: 해당 객체의 예측 높이($h$)와 너비($w$)를 예측한다.
    - 이후, 고정 크기의 Shape vector를 예측된 $h \times w$ 크기로 리사이징(resize)하여 각 인스턴스의 거친 마스크를 형성한다.

2. **Global Saliency Generation**:
    - Fully Convolutional Backbone을 통해 이미지 전체의 Saliency Map을 생성한다.
    - 이는 시맨틱 세그멘테이션과 유사하게 픽셀이 전경(foreground)에 속하는지를 예측하며, Sigmoid 함수를 통해 이진 분류를 수행한다.
    - 설정에 따라 클래스 구분 없는(Class-agnostic) 방식이나 클래스별(Class-specific) 방식으로 동작할 수 있다.

3. **Mask Assembly**:
    - 예측된 중심점의 위치와 크기를 기반으로 Global Saliency Map에서 해당 영역을 크롭(crop)하여 $G^k$를 얻고, 대응하는 Local Shape $L^k$를 준비한다.
    - 두 행렬에 Sigmoid 함수를 적용한 후, Hadamard product(원소별 곱셈)를 통해 최종 마스크 $M^k$를 생성한다.
    $$M^k = \sigma(L^k) \odot \sigma(G^k)$$

### 학습 목표 및 손실 함수

모델은 다음 네 가지 손실 함수의 합으로 학습된다.
$$\mathcal{L}_{seg} = \lambda_p \mathcal{L}_p + \lambda_{off} \mathcal{L}_{off} + \lambda_{size} \mathcal{L}_{size} + \lambda_{mask} \mathcal{L}_{mask}$$

- **Center Point Loss ($\mathcal{L}_p$)**: 중심점 위치 및 카테고리를 예측하기 위한 손실 함수로, Focal Loss가 수정된 픽셀 단위 로지스틱 회귀를 사용한다.
- **Offset Loss ($\mathcal{L}_{off}$)**: 출력 스트라이드로 인한 이산화 오차를 보정하기 위해 $L1$ loss를 사용한다.
- **Size Loss ($\mathcal{L}_{size}$)**: 예측된 객체 크기와 실제 크기 간의 차이를 $L1$ loss로 계산한다.
- **Mask Loss ($\mathcal{L}_{mask}$)**: 최종 조립된 마스크 $M^k$와 Ground Truth 마스크 $T^k$ 사이의 픽셀 단위 Binary Cross Entropy(BCE)를 계산한다.
$$\mathcal{L}_{mask} = \frac{1}{N} \sum_{k=1}^{N} Bce(M^k, T^k)$$

## 📊 Results

### 실험 설정

- **데이터셋**: MS COCO instance segmentation benchmark.
- **백본**: Hourglass-104, DLA-34.
- **비교 지표**: mask AP, FPS.
- **구현 세부사항**: 입력 해상도 $512 \times 512$, Adam 옵티마이저 사용, NMS 없이 상위 100개 점수 포인트만 반환.

### 주요 결과

- **정량적 성능**: Hourglass-104 백본 사용 시 **34.5 mask AP**와 **12.3 FPS**를 달성하였다. DLA-34 백본 사용 시에는 **32.5 mask AP**와 **25.2 FPS**를 기록하여 속도와 정확도 사이의 우수한 트레이드-오프를 보여주었다.
- **비교 분석**:
  - Two-stage 모델인 Mask R-CNN과 One-stage인 TensorMask보다 AP는 약간 낮으나, 속도는 각각 4배, 5배 더 빠르다.
  - YOLACT 및 PolarMask와 비교했을 때, 특히 DLA-34 모델은 YOLACT보다 더 빠르면서 높은 AP를 기록하였으며, Hourglass-104 모델은 PolarMask보다 1.6 포인트 더 높은 AP를 기록하며 속도 또한 더 빨랐다.
- **Ablation Study**:
  - Local Shape 브랜치만 있을 때보다 Global Saliency를 추가했을 때 약 5포인트의 성능 향상이 있었다.
  - Class-specific 설정이 Class-agnostic 설정보다 2.4 포인트 더 높게 나타나, 클래스 정보가 인스턴스 구분에 도움을 줌을 확인하였다.
- **일반화 성능**: CenterMask의 모듈을 FCOS 검출기에 이식했을 때, ResNeXt-101-FPN 기반으로 **38.5 mAP**를 달성하여 타 모델(PolarMask 등)보다 우수한 성능과 범용성을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 인스턴스 세그멘테이션의 난제인 '인스턴스 구분'과 '픽셀 정렬'을 서로 다른 두 가지 경로(Local Shape, Global Saliency)로 분리하여 해결함으로써 효율성을 극대화하였다.

시각화 결과 분석에 따르면, Local Shape 브랜치는 객체가 심하게 겹쳐 있는 상황에서도 각 인스턴스를 분리하는 능력이 탁월하지만 마스크의 경계가 거칠다는 특징이 있다. 반면, Global Saliency 브랜치는 픽셀 단위의 정밀한 세그멘테이션이 가능하지만, 객체가 겹치면 이를 구분하지 못하고 하나로 뭉뚱그려 예측하는 경향이 있다. 이 두 브랜치를 결합함으로써 각각의 약점을 상쇄하고 강점을 극대화하는 시너지 효과를 거두었음을 알 수 있다.

다만, 하이퍼파라미터 설정이 기존 CenterNet의 설정을 그대로 따랐기 때문에, CenterMask에 최적화된 파라미터를 재탐색한다면 성능을 더욱 향상시킬 수 있을 것이라는 점이 한계이자 향후 개선 가능성으로 제시되었다.

## 📌 TL;DR

CenterMask는 One-stage instance segmentation을 위해 **Local Shape(인스턴스 구분)**와 **Global Saliency(픽셀 정밀도)**라는 두 가지 병렬 경로를 제안하고, 이를 결합하여 마스크를 생성하는 단순하고 효율적인 프레임워크이다. COCO 데이터셋에서 높은 속도와 정확도의 균형을 보여주었으며, FCOS와 같은 다른 검출기에도 쉽게 적용 가능한 범용성을 갖추고 있어, 향후 실시간 인스턴스 세그멘테이션 및 파놉틱 세그멘테이션 연구에 중요한 기초가 될 가능성이 높다.
