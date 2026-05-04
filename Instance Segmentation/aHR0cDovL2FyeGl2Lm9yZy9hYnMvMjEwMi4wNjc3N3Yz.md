# INSTA-YOLO: Real-Time Instance Segmentation

Eslam Mohamed, Abdelrahman Shaker, Ahmad El-Sallab, Mayada Hadhoud (2020)

## 🧩 Problem to Solve

본 논문은 실시간 Instance Segmentation의 효율성과 속도를 개선하는 것을 목표로 한다. 일반적인 Instance Segmentation은 객체 검출(Object Detection) 후 검출된 영역 내에서 세그멘테이션을 수행하는 2단계(Two-stage) 파이프라인을 따른다. 이러한 방식은 다음과 같은 문제점을 가진다:

1. **계산 복잡도**: 픽셀 단위의 예측을 위해 고비용의 업샘플링(Up-sampling) 과정이 필수적이다.
2. **처리 속도**: 두 단계가 순차적으로 실행되어야 하므로 실시간 애플리케이션에 적용하기에는 속도가 느리다.
3. **방향성 객체 처리의 한계**: 원격 탐사(Remote Sensing)나 LiDAR 데이터와 같이 방향성이 있는 바운딩 박스(Oriented Bounding Box)가 필요한 작업에 적합하지 않다.

따라서 본 연구의 목표는 업샘플링 과정 없이 객체의 마스크를 효율적으로 예측할 수 있는 1단계(One-stage) 엔드-투-엔드 딥러닝 모델인 **Insta-YOLO**를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Instance Segmentation을 픽셀 단위의 분류 문제로 보지 않고, **객체의 윤곽선(Contour)을 2D 카테시안 좌표계(Cartesian space) 상의 점들로 예측하는 회귀 문제로 정의**한 것이다.

주요 기여 사항은 다음과 같다:
- **새로운 CNN 아키텍처**: YOLOv3를 기반으로 하여 픽셀 단위 예측 및 업샘플링 없이 실시간으로 동작하는 Instance Segmentation 모델을 구축하였다.
- **적응형 폴리곤 표현(Adaptive Polygon Representation)**: 객체의 곡률에 따라 점의 밀도를 조절하여 효율적으로 형태를 표현하는 데이터 생성 방식을 도입하였다.
- **새로운 손실 함수**: 객체의 위치 추정 정밀도를 높이기 위해 Log Cosh loss와 새로운 Cartesian IoU loss를 제안하였다.
- **방향성 객체로의 확장성**: 별도의 각도 예측 파라미터 없이도 자연스럽게 방향성이 있는 객체를 검출할 수 있음을 입증하였다.

## 📎 Related Works

논문에서는 기존의 접근 방식을 세 가지 범주로 나누어 설명한다.

1. **Object Detection**: YOLO, SSD와 같은 1단계 검출기와 Faster R-CNN과 같은 2단계 검출기가 있다. 1단계 방식은 빠르지만 정밀도가 낮고, 2단계 방식은 정밀하지만 실시간 처리가 어렵다.
2. **Instance Segmentation**:
   - **Two-stage**: Mask R-CNN, PANet 등이 있으며, 검출 후 마스크를 생성한다. 매우 느리다는 단점이 있다.
   - **One-stage**: YOLACT, CenterMask 등이 있다. YOLACT는 프로토타입 마스크를 생성하여 결합하는 방식을 사용하지만, 여전히 업샘플링 과정이 병목 현상을 일으킨다.
3. **Bounding Polygon 기반 방식**: ExtremeNet, PolarMask, FourierNet 등이 객체의 중심점을 기준으로 폴리곤을 예측한다. 하지만 Hourglass-104와 같은 무거운 백본을 사용하거나, 특정 기하학적 가정(구형 근사 등)으로 인해 정밀도나 속도 면에서 한계가 있다.

Insta-YOLO는 이러한 기존 방식들과 달리 업샘플링을 완전히 제거하고, 효율적인 카테시안 좌표 기반의 폴리곤 회귀를 통해 속도와 정확도의 균형을 맞추었다.

## 🛠️ Methodology

### 1. 데이터 생성 (Data Generation)
픽셀 단위의 세그멘테이션 마스크를 폴리곤 형태로 변환해야 한다. 본 논문은 고정된 간격의 샘플링 대신 **적응형 단계(Adaptive Step)** 방식을 제안한다.
- **Douglas-Peucker 알고리즘**을 사용하여 곡선에서는 더 많은 점을 배치하고, 직선 구간에서는 최소한의 점만 사용하여 객체의 형태를 가장 잘 나타내는 최적의 점 집합을 찾는다.
- 이를 통해 네트워크가 곡선 부분에 더 집중하게 하여 효율성과 정확도를 동시에 높인다.

### 2. 네트워크 아키텍처 (Network Architecture)
백본 네트워크는 **YOLOv3**를 기반으로 하며, 출력 레이어를 다음과 같이 수정하였다.
- 기존 YOLOv3가 바운딩 박스의 좌표 $(x, y, w, h)$를 예측했다면, Insta-YOLO는 **객체 마스크를 구성하는 $N$개의 정점(Vertices) 좌표와 마스크 신뢰도(Confidence)**를 예측한다.
- 출력 레이어의 구조는 다음과 같다:
$$\text{output layer} = n_{\text{anchors}} [n_{\text{vertices}} \times 2 + 1 + n_{\text{classes}}]$$
(여기서 $n_{\text{vertices}} \times 2$는 $x, y$ 좌표 쌍을 의미한다.)

### 3. 손실 함수 (Loss Functions)
전체 손실 함수는 다음과 같이 정의된다:
$$\text{Insta-YOLO loss} = \text{Classification loss} + \text{Confidence loss} + \text{Localization loss}$$

#### 3.1 Classification 및 Confidence Loss
이 부분은 YOLOv3의 설계를 그대로 계승하며, 각각 클래스 확률의 제곱 오차와 객체 존재 여부에 대한 신뢰도를 측정한다.

#### 3.2 Localization Loss
본 논문의 핵심 기여 부분으로, 단순 MSE 대신 다음과 같은 조합을 사용한다:
$$\text{Localization loss} = \lambda \cdot \text{Regression Loss} + (1 - \lambda) \cdot \text{IoU Loss}$$

- **Regression Loss**: 이상치(Outlier)에 덜 민감하도록 **Log Cosh loss**를 사용한다. 이는 작은 오차에 대해서는 MSE처럼 동작하고, 큰 오차에 대해서는 L1 loss처럼 동작하여 미분 가능하면서도 강건한 학습을 가능하게 한다.
- **IoU Loss**: 점들의 좌표가 약간만 달라져도 큰 페널티를 주는 Regression loss의 단점을 보완하기 위해 IoU loss를 추가하였다. 
  - **Cartesian IoU Loss**: 예측된 폴리곤과 정답(GT) 폴리곤의 교집합과 합집합 면적을 직접 계산하여 손실을 구한다. 폴리곤의 면적은 다음과 같은 신발 끈 공식(Shoelace formula)을 사용하여 계산한다:
    $$\text{Area} = \frac{1}{2} |(x_1y_2 - y_1x_2) + (x_2y_3 - y_2x_3) + \cdots + (x_ny_1 - y_nx_1)|$$
- **학습 전략 (Warm-up)**: 학습 초기에는 $\lambda$ 값을 높게 설정하여 모델이 먼저 대략적인 정점 위치를 잡게 하고(Regression 중심), 점차 $\lambda$를 낮추어 IoU loss가 곡률 정보를 세밀하게 학습하도록 유도한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: Carvana, Cityscapes, Airbus Ship 데이터셋 사용.
- **구현**: ResNet-50 인코더 기반의 YOLOv3 구조, Adam 옵티마이저 ($\text{lr}=1\text{e}^{-4}$), 80 에포크 학습.
- **평가 지표**: $\text{AP}_{50}$, $\text{AP}_{75}$ 및 FPS (GTX-1080 GPU 기준).

### 2. 주요 결과
- **정확도 및 속도**: Cityscapes 데이터셋에서 Cartesian IoU loss를 적용했을 때 $\text{AP}_{50}$ 89%를 달성하였으며, 속도는 **56 FPS**로 YOLACT(32 FPS)나 PolarMask(28 FPS)보다 훨씬 빠르다.
- **Carvana 데이터셋**: $\text{AP}_{50}$ 99%를 기록하며 YOLACT 및 Mask R-CNN과 대등한 성능을 보이면서도 속도는 압도적으로 빠르다.
- **방향성 바운딩 박스 (Oriented Bounding Boxes)**: Airbus Ship 데이터셋 실험 결과, 기존의 Oriented YOLO (YOLO3D)보다 정확도가 5% 높고 속도는 2.7배 빠르다. 이는 폴리곤 기반 예측이 각도 인코딩 문제(Angle encoding problem)를 자연스럽게 해결하기 때문이다.

## 🧠 Insights & Discussion

### 강점
- **극단적인 속도 향상**: 업샘플링과 2단계 구조를 완전히 제거함으로써 실시간 성능을 확보하였다.
- **기하학적 유연성**: 픽셀 기반이 아닌 정점 기반 예측을 통해 방향성 객체 검출과 같은 작업에 추가적인 수정 없이 바로 적용 가능하다는 점이 매우 강력하다.
- **손실 함수의 최적화**: Log Cosh와 Cartesian IoU loss의 조합, 그리고 $\lambda$ 스케줄링을 통한 Warm-up 전략이 모델의 수렴과 정밀도를 효과적으로 높였다.

### 한계 및 논의사항
- **정점 수의 고정**: 하이퍼파라미터인 $N$(정점의 수)이 고정되어 있어, 매우 복잡한 형태의 객체는 표현력이 떨어질 수 있다.
- **자기 교차 문제**: IoU loss는 폴리곤 선이 꼬이는 'Self-intersecting' 문제에 취약하다. 저자들은 이를 위해 초기 학습 단계에서 Regression loss를 우선시하는 전략을 사용했지만, 완전히 해결되었는지에 대한 정량적 분석은 부족하다.
- **클래스 제한**: Cityscapes 결과가 "vehicles" 클래스에 대해서만 보고되었으므로, 다양한 클래스가 섞인 복잡한 씬에서의 일반화 성능에 대한 추가 검증이 필요하다.

## 📌 TL;DR

본 논문은 Instance Segmentation을 위해 픽셀 단위 예측 대신 **객체의 윤곽선 정점을 직접 예측하는 1단계 모델 Insta-YOLO**를 제안한다. 업샘플링 과정을 제거하고 적응형 폴리곤 표현과 Cartesian IoU loss를 도입함으로써, 기존 SOTA 모델들보다 **약 2배 이상의 속도(56 FPS)**를 내면서도 경쟁력 있는 정확도를 달성하였다. 특히 각도 예측 없이도 방향성 객체를 효율적으로 처리할 수 있어, 자율주행이나 원격 탐사 분야의 실시간 객체 분할에 매우 유용할 것으로 기대된다.