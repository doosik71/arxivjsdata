# 3D-SIS: 3D Semantic Instance Segmentation of RGB-D Scans

Ji Hou, Angela Dai, Matthias Nießner (2019)

## 🧩 Problem to Solve

본 논문은 일반적인 RGB-D 스캔 데이터에서 **3D semantic instance segmentation(3D 시맨틱 인스턴스 분할)**을 수행하는 문제를 해결하고자 한다. 3D 공간에서의 시맨틱 이해는 자율주행 자동차, 드론, 보조 로봇, 그리고 AR/VR 기기와 같은 현대 컴퓨터 비전 응용 분야에서 상호작용을 가능하게 하는 핵심적인 요소이다.

기존의 많은 연구는 단일 2D 이미지 분석에 집중해 왔으나, 실제 환경에서는 비디오 스트림이나 RGB-D 센서와 같은 다중 뷰(multi-view) 데이터를 사용하는 경우가 많다. 특히 인스턴스 분할의 경우, 여러 프레임에 걸쳐 동일한 객체를 식별하고 연결하는 인스턴스 연관(instance association) 문제가 발생하며, 이를 단일 이미지 단위로 처리할 경우 공간적 일관성을 유지하기 어렵다는 한계가 있다. 따라서 본 논문의 목표는 다중 뷰 RGB 입력과 3D 기하학적 구조(geometry)를 효과적으로 융합하여, 공간적으로 일관된 시맨틱 라벨과 3D 레이아웃을 동시에 예측하는 end-to-end 네트워크를 구축하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 **기하학적 신호(geometric signal)와 색상 신호(color signal)를 공동으로 학습(joint learning)**하여 인스턴스 예측의 정확도를 높이는 것이다. 주요 기여 사항은 다음과 같다.

1. **Joint 2D-3D Feature Learning**: 3D 스캔의 기하학적 정보와 RGB 입력의 색상 정보를 end-to-end 방식으로 결합하여 3D 객체 바운딩 박스 검출 및 시맨틱 인스턴스 분할을 수행하는 최초의 접근 방식을 제안한다.
2. **Fully-Convolutional 3D Architecture**: 훈련은 장면의 일부인 chunk 단위로 수행하지만, 추론 시에는 전체 3D 환경에 대해 단 한 번의 forward pass로 예측이 가능한 fully-convolutional 구조를 채택하여 효율성과 일관성을 확보하였다.
3. **성능 향상**: 실세계 데이터셋인 ScanNetV2에서 기존 state-of-the-art 방법들보다 mAP(mean Average Precision)를 13.5 이상 향상시키는 성과를 거두었다.

## 📎 Related Works

### 관련 연구 및 한계
- **2D Object Detection 및 Instance Segmentation**: Mask R-CNN과 같은 모델들은 단일 이미지에서 뛰어난 성능을 보이지만, 3D 공간에서의 객체 간 관계나 공간적 배치를 이해하는 데 한계가 있다.
- **3D Object Detection**: Sliding Shapes나 Frustum PointNet과 같은 연구들이 3D 검출을 시도하였다. Frustum PointNet은 2D에서 검출 후 3D로 투영하는 방식을 취하지만, 2D와 3D 특징을 공동으로 학습하여 최적화하는 구조는 아니다.
- **3D Instance Segmentation**: SGPN은 포인트 클라우드(point cloud) 기반의 클러스터링 접근 방식을 사용한다. 하지만 이는 다중 뷰 RGB 데이터를 기하학적 구조와 명시적으로 매핑하여 융합하는 방식과는 차이가 있다.

### 차별점
3D-SIS는 단순히 2D 결과를 3D로 투영하는 것이 아니라, **명시적인 공간 매핑(explicit spatial mapping)**을 통해 2D CNN 특징과 3D 볼륨 그리드 특징을 결합한다. 이를 통해 다중 뷰의 정보를 통합적으로 활용하며, 전체 씬에 대한 일관된 예측이 가능하다.

## 🛠️ Methodology

### 전체 파이프라인
3D-SIS는 크게 **3D Detection(검출)** 파이프라인과 **3D Mask Prediction(마스크 예측)** 파이프라인으로 구성된다. 전체 프로세스는 $\text{RGB-D 입력} \rightarrow \text{특징 추출 및 융합} \rightarrow \text{바운딩 박스 제안} \rightarrow \text{클래스 분류} \rightarrow \text{복셀 단위 인스턴스 마스크 생성}$ 순으로 진행된다.

### 주요 구성 요소 및 역할

#### 1. Back-projection Layer for RGB Features
2D 이미지의 해상도는 3D 복셀 그리드보다 훨씬 높기 때문에, 메모리 효율성을 위해 2D CNN을 통해 먼저 특징을 요약한다.
- **특징 추출**: 사전 학습된 ENet 기반의 2D 세그멘테이션 네트워크를 사용하여 각 픽셀의 특징 벡터를 추출한다.
- **투영(Back-projection)**: 카메라의 내적/외적 파라미터(pose alignment)를 이용하여 2D 픽셀 특징을 3D 볼륨 그리드의 복셀로 투영한다.
- **뷰 풀링(View Pooling)**: 여러 각도에서 찍힌 이미지들이 동일한 복셀에 투영될 때, element-wise max pooling을 통해 다중 뷰 특징을 통합한다.

#### 2. 3D Feature Backbones
기하학적 정보(TSDF)와 투영된 RGB 특징을 처리하기 위해 두 개의 별도 백본을 사용한다.
- **Detection Backbone**: TSDF와 RGB 특징을 각각 3D ResNet 블록으로 처리한 후 결합(concatenation)한다. 이후 3D 합성곱을 통해 공간 해상도를 4배 감소시켜 서로 다른 수용 영역(receptive field)을 가진 특징 맵을 생성한다. 이는 '작은 객체'와 '큰 객체'를 모두 잡기 위한 전략이다.
- **Mask Backbone**: 인스턴스 마스크의 정밀도를 높이기 위해 공간 해상도를 유지하는 3D 합성곱 층을 사용한다.

#### 3. 3D-RPN 및 3D-RoI Pooling
- **3D-RPN**: Detection 백본의 특징 맵 위에 정의된 앵커(anchor)를 기반으로 객체의 존재 여부(objectness)와 바운딩 박스 위치를 예측한다.
- **3D-RoI Pooling**: 예측된 바운딩 박스 영역의 특징을 $4 \times 4 \times 4$ 크기의 고정된 블록으로 풀링하여 MLP(Multi-Layer Perceptron)를 통해 최종 클래스를 분류한다.

#### 4. Per-Voxel Instance Segmentation
검출된 바운딩 박스와 마스크 백본의 특징을 사용하여 복셀 단위의 이진 마스크를 생성한다. 최종적으로 예측된 클래스 라벨에 해당하는 채널의 마스크를 선택한다.

### 주요 방정식 및 손실 함수

**바운딩 박스 회귀(Bounding Box Regression)**:
앵커의 중심 $\mu_{anchor}$와 크기 $\phi_{anchor}$를 기준점으로 하여 실제 박스의 중심 $\mu$와 크기 $\phi$ 사이의 로그 비율을 예측한다.
$$\Delta x = \frac{\mu - \mu_{anchor}}{\phi_{anchor}}, \quad \Delta w = \ln\left(\frac{\phi}{\phi_{anchor}}\right)$$
(다른 축 $y, z, h, l$에 대해서도 동일하게 적용된다.)

**손실 함수(Loss Functions)**:
- **Objectness 및 Classification**: 이진 및 다중 클래스 교차 엔트로피 손실(Cross Entropy Loss)을 사용한다.
- **Bounding Box Regression**: 예측값과 실제값 사이의 차이를 줄이기 위해 Huber loss를 사용한다.
- **Instance Mask**: 복셀 단위의 마스크 예측을 위해 이진 교차 엔트로피 손실(Binary Cross Entropy Loss)을 사용한다.

## 📊 Results

### 실험 설정
- **데이터셋**: 합성 데이터셋인 SUNCG와 실세계 데이터셋인 ScanNetV2를 사용하였다.
- **평가 지표**: $\text{mAP@0.25}$ 및 $\text{mAP@0.5}$ (IoU 임계값 기준 평균 정밀도)를 사용하여 성능을 측정하였다.
- **비교 대상**: Mask R-CNN(2D-3D 투영), SGPN, Frustum PointNet 등과 비교하였다.

### 정량적 결과
- **실세계 데이터(ScanNetV2)**: 3D-SIS는 기존 SOTA 방법들보다 $\text{mAP@0.5}$ 기준 상당한 성능 향상을 보였으며, 특히 real-world 데이터에서 mAP가 13.5 이상 증가하였다.
- **합성 데이터(SUNCG)**: $\text{mAP@0.25}$ 기준 32.2를 기록하며 기존 방법들을 상회하였다.

### 분석 결과
- **색상-기하학 융합의 효과**: Geometry만 사용했을 때보다 RGB 특징을 함께 사용했을 때 성능이 크게 향상되었다.
- **다중 뷰의 영향**: 훈련 시 사용되는 RGB 이미지의 수(1뷰 $\rightarrow$ 3뷰 $\rightarrow$ 5뷰)가 증가할수록 검출 및 분할 성능이 일관되게 향상되는 것을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 의의
- **공간적 일관성**: 개별 프레임 단위로 처리하고 나중에 병합하는 방식이 아니라, 전체 3D 씬을 하나의 볼륨으로 처리함으로써 인스턴스 경계의 일관성을 획기적으로 높였다.
- **유연한 추론**: Fully-convolutional 구조 덕분에 훈련 시에는 작은 chunk 단위로 학습했지만, 테스트 시에는 매우 큰 규모의 전체 씬(예: $45\text{m} \times 45\text{m}$ 공간)에 대해서도 단일 pass로 효율적인 추론이 가능하다.

### 한계 및 논의사항
- **메모리 제약**: 3D 볼륨 데이터의 특성상 메모리 사용량이 매우 크다. 이를 해결하기 위해 훈련 시 chunk 단위로 나누어 학습하는 방식을 취했으나, 이는 최적의 하이퍼파라미터 설정이 필요함을 시사한다.
- **가정**: 본 모델은 RGB-D 스캔 데이터의 6DoF pose가 정확하게 정렬되어 있다는 가정하에 동작한다. 만약 SLAM/Odometry 단계에서 오차가 크다면 back-projection 단계에서 심각한 성능 저하가 발생할 가능성이 있다.

## 📌 TL;DR

3D-SIS는 RGB-D 스캔 데이터에서 **다중 뷰 RGB 특징과 3D 기하학적 TSDF 특징을 end-to-end로 융합 학습**하는 3D 시맨틱 인스턴스 분할 네트워크이다. 3D-RPN과 3D-RoI 풀링을 통해 객체를 검출하고 복셀 단위의 정밀한 마스크를 생성하며, fully-convolutional 설계를 통해 전체 씬에 대한 일관된 추론을 가능하게 했다. 실세계 데이터셋에서 mAP를 13.5 이상 끌어올리며 탁월한 성능을 입증하였으며, 이는 향후 자율주행 및 AR/VR의 공간 이해 연구에 중요한 기준점이 될 것으로 보인다.