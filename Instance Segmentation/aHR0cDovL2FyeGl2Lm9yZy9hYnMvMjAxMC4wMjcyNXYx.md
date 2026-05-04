# Vec2Instance: Parameterization for Deep Instance Segmentation

N. Lakmal Deshapriya, Matthew N. Dailey, Manzul Kumar Hazarika, Hiroyuki Miyazaki (2020)

## 🧩 Problem to Solve

본 논문은 컴퓨터 비전의 주요 난제 중 하나인 인스턴스 분할(Instance Segmentation) 문제를 해결하고자 한다. 인스턴스 분할은 이미지 내의 각 픽셀에 레이블을 지정하는 동시에, 동일한 클래스에 속하더라도 서로 다른 개체(instance)를 구분하여 인식해야 하는 작업이다.

기존의 인스턴스 분할 방식, 특히 Mask R-CNN과 같은 모델들은 높은 정확도를 보이지만, 구조가 매우 복잡하여 구현 및 운용에 어려움이 있다. 또한, 많은 CNN 기반 모델들이 풀링(Pooling) 연산을 통해 이동 불변성(Shift Invariance)을 확보하는 과정에서 위치 정보(Positional Information)를 손실하며, 이는 정밀한 인스턴스 마스크 생성에 방해가 된다. 따라서 본 연구의 목표는 복잡한 파이프라인을 단순화하면서도, 인스턴스의 복잡한 형상을 효율적으로 추정할 수 있는 새로운 파라미터화(Parameterization) 기반의 딥러닝 아키텍처인 `Vec2Instance`를 제안하는 것이다.

## ✨ Key Contributions

`Vec2Instance`의 핵심 아이디어는 인스턴스 마스크를 하나의 다변수 함수(Multivariate Function)로 정의하고, 이 함수의 파라미터를 CNN이 직접 예측하게 만드는 것이다.

기존 방식이 픽셀 단위의 분류나 바운딩 박스 회귀에 집중했다면, 본 연구는 인스턴스의 형상을 결정짓는 가중치와 편향(Weights and Biases) 자체를 파라미터 벡터로 간주한다. 즉, 하나의 CNN이 다른 작은 신경망(MLP 디코더)의 가중치를 예측하는 '신경망을 예측하는 신경망' 구조를 설계하였다. 이를 통해 복잡한 마스크 생성 과정을 단순한 파라미터 회귀 문제로 변환하였으며, 보편적 근사 정리(Universal Approximation Theorem)에 근거하여 충분한 은닉층이 있다면 어떤 복잡한 인스턴스 형상도 근사할 수 있다는 직관을 활용하였다.

## 📎 Related Works

논문에서는 다음과 같은 기존 연구들을 언급하며 차별점을 제시한다.

- **Mask R-CNN**: Faster R-CNN의 구조에 FCN(Fully Convolutional Networks) 브랜치를 추가하여 각 ROI(Region of Interest)에 대해 마스크를 생성한다. 매우 성공적인 모델이지만 구조가 지나치게 복잡하다는 단점이 있다.
- **DCAN (Deep Contour Awareness Networks)**: 의미론적 분할(Semantic Segmentation)과 객체의 경계(Edge) 분할을 동시에 수행하여 인스턴스 인식 능력을 부여한다.
- **Deep Watershed Transform**: 픽셀에서 경계까지의 거리와 방향 벡터를 학습하여 위치 인식 능력을 직접적으로 제공한다.
- **YOLO (You Only Look Once)**: 객체 탐지(Object Detection)를 위해 중심점과 바운딩 박스 파라미터를 단일 네트워크에서 직접 회귀하는 end-to-end 방식을 사용한다. `Vec2Instance`는 YOLO의 이러한 파라미터 회귀 개념을 단순한 사각형이 아닌 복잡한 인스턴스 형상으로 확장한 것이다.

## 🛠️ Methodology

### 전체 시스템 구조
`Vec2Instance`는 두 개의 독립적인 CNN과 하나의 고정된 디코더(Decoder)로 구성된 파이프라인을 가진다.

1. **Centroid Estimation CNN**: 이미지 전체에서 각 인스턴스의 중심점(Centroid) 위치를 예측한다.
2. **Instance Segmentation CNN**: 예측된 중심점 주변의 패치(Patch)를 입력받아, 해당 인스턴스의 형상을 결정하는 파라미터 벡터를 예측한다.
3. **Mask Generation Decoder**: 예측된 파라미터 벡터를 가중치로 사용하여, 픽셀 좌표를 입력받아 마스크 값(0 또는 1)을 출력하는 작은 MLP이다.

### 상세 구성 요소 및 절차

#### 1. Centroid Estimation CNN
입력 RGB 위성 이미지 타일에서 건물 중심점의 존재 여부를 이진 이미지 형태로 출력한다. 수용 영역(Receptive Field)을 넓히기 위해 Dilated Convolution을 사용하며, 최종 출력은 $1 \times 1$ 컨볼루션 층을 통해 단일 채널의 확률 맵으로 생성된다.

#### 2. Instance Segmentation CNN 및 Decoder
이 네트워크는 특정 인스턴스가 중심에 위치한 이미지 패치를 입력으로 받는다.
- **CNN 부분**: 인스턴스의 특징을 추출하여 총 257개의 요소를 가진 파라미터 벡터를 출력한다.
- **Decoder 부분**: 학습 가능한 파라미터가 없는 고정된 구조의 MLP이다.
    - **구조**: 64개의 유닛을 가진 단일 은닉층으로 구성된다.
    - **동작**: 입력으로 픽셀 좌표 $(x, y)$를 받으며, CNN이 예측한 257개의 파라미터를 가중치와 편향으로 사용하여 출력값 $\hat{y} \in \{0, 1\}$을 계산한다.
    - **파라미터 구성**: $(2 \times 64 \text{ weights} + 64 \text{ biases}) + (64 \times 1 \text{ weights} + 1 \text{ bias}) = 257$개.

#### 3. 학습 및 추론 절차
- **학습**: 두 CNN은 분리되어 학습된다. 손실 함수로는 예측된 중심점/마스크와 Ground Truth 사이의 Root Mean Squared Error (RMSE)를 사용하며, Adam 옵티마이저를 통해 최적화한다.
- **추론 (Prediction)**:
    1. Centroid Estimation CNN을 통해 후보 중심점들을 찾는다.
    2. 각 중심점 주변 패치를 Instance Segmentation CNN에 넣어 파라미터 벡터를 얻는다.
    3. 디코더를 통해 마스크를 재구성한다.
    4. 중첩된 마스크들은 Non-Maximum Suppression (NMS)을 통해 가장 확률이 높은 것만 남기고 제거한다.

## 📊 Results

### 실험 설정
- **데이터셋**: SpaceNet Challenge AOI 2 (Vegas) 건물 풋프린트 데이터셋.
- **전처리**: $650 \times 650$ 타일을 $256 \times 256$으로 리샘플링하고, 인스턴스 분할 네트워크 학습을 위해 개별 건물 패치($64 \times 64$)를 추출하였다.
- **비교 대상**: Mask R-CNN (SOTA 인스턴스 분할), U-Net (SOTA 의미론적 분할).

### 주요 결과
- **정량적 성능**:
    - **Vec2Instance**: 전체 픽셀 정확도(Overall Pixel-wise Accuracy) $89\%$, IoU $61\%$.
    - **Mask R-CNN**: 전체 픽셀 정확도 $91\%$, IoU $65\%$.
    - **U-Net**: 전체 픽셀 정확도 $96\%$, IoU $84\%$.
- **효율성**: `Vec2Instance`의 총 파라미터 수는 약 $0.35\text{M}$ (Centroid CNN $166\text{K}$ + Instance CNN $183\text{K}$)로, Mask R-CNN($44\text{M}$)에 비해 압도적으로 적다. 학습 시간 또한 4.3시간으로 Mask R-CNN(8.6시간)보다 짧다.

### 분석
U-Net이 가장 높은 정확도를 보였으나, U-Net은 인스턴스 인식 능력이 내장되어 있지 않아 별도의 후처리가 필요하다는 한계가 있다. 반면 `Vec2Instance`는 Mask R-CNN과 유사한 성능을 내면서도 구조가 훨씬 단순하고 파라미터 수가 매우 적다는 강점을 가진다.

## 🧠 Insights & Discussion

### 강점 및 한계
본 연구의 가장 큰 강점은 인스턴스 분할 문제를 파라미터 회귀 문제로 단순화하여 모델의 복잡도를 획기적으로 낮춘 점이다. 특히 Transpose Convolution과 같은 무거운 연산을 제거하고 작은 MLP 디코더를 통해 마스크를 생성함으로써 효율성을 높였다.

그러나 한계점도 명확하다. 실험 결과, 정형화되지 않은 특이한 모양의 건물, 나무에 가려진 건물, 혹은 배경과 경계가 모호한 건물들에 대해 성능이 저하되는 경향(False Negative 증가)이 나타났다. 이는 파라미터화된 함수가 특정 복잡도를 넘어서는 형상을 근사하는 데 한계가 있을 수 있음을 시사한다.

### 확장 가능성 및 비판적 해석
저자들은 이 방식이 얼굴 재구성(Face Reconstruction)이나 3D 형상 복원과 같이 고차원 공간에서의 파라미터화가 필요한 작업으로 확장될 수 있음을 보여주었다. 실제로 얼굴 이미지의 RGB 강도를 파라미터화하여 T-SNE로 시각화한 결과, 의미 있는 잠재 공간(Latent Space)이 형성됨을 확인하였다.

비판적으로 보았을 때, 본 모델은 인스턴스 분할의 '정확도' 자체보다는 '효율적인 파라미터 표현 방식'의 가능성을 제시하는 데 집중하고 있다. U-Net과의 성능 차이는 인스턴스 분할을 위해 중심점 예측이라는 전단계(Centroid Estimation)를 거치면서 발생하는 오차가 전체 성능 하락의 주원인이 됨을 보여준다.

## 📌 TL;DR

본 논문은 인스턴스 마스크를 가변적인 함수 파라미터의 집합으로 정의하고, 이를 CNN으로 예측하여 복원하는 `Vec2Instance` 아키텍처를 제안한다. Mask R-CNN과 같은 복잡한 구조 없이도 유사한 성능을 달성하였으며, 특히 파라미터 수를 획기적으로 줄여(약 $0.35\text{M}$) 모델의 효율성을 극대화하였다. 이 연구는 인스턴스 분할뿐만 아니라 얼굴 생성, 3D 복원 등 다양한 형상 파라미터화 작업에 응용될 수 있는 단순하고 직관적인 프레임워크를 제공한다.