# OG: Equip vision occupancy with instance segmentation and visual grounding

Zichao Dong, Hang Ji, Weikun Zhang, Xufeng Huang, Junbo Chen (2023)

## 🧩 Problem to Solve

본 논문은 3D 공간의 기하학적 구조와 시맨틱 레이블을 추론하는 Occupancy Prediction 작업에서 두 가지 핵심적인 한계를 해결하고자 한다.

첫째, 기존의 Occupancy Prediction은 기본적으로 Semantic Segmentation 수준에 머물러 있어, 동일한 클래스에 속하는 서로 다른 객체들을 개별적으로 구분하는 Instance Segmentation 능력이 결여되어 있다. 로봇 공학 및 자율 주행에서 주변 환경의 개별 객체를 식별하는 것은 하위 인지 작업 및 안전한 경로 계획을 위해 매우 중요하다.

둘째, 최근 2D 영역에서는 텍스트나 포인트 등의 프롬프트를 통해 특정 객체를 찾아내는 Visual Grounding 기술이 발전하였으나, 이를 3D Occupancy 영역으로 확장하여 복셀(Voxel) 단위의 Grounding을 수행하는 연구는 아직 미비한 상태이다.

따라서 본 논문의 목표는 vanilla occupancy 파이프라인에 Instance Segmentation 능력을 부여하고, 이를 통해 텍스트 기반의 Visual Grounding이 가능한 **Occupancy Grounding (OG)** 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 복잡한 2D-3D 변환 과정과 인스턴스 구분 문제를 해결하기 위해 다음의 두 가지 설계를 도입한 것이다.

1. **Affinity Field Prediction**: 복셀들을 인스턴스 단위로 클러스터링하기 위해, 각 복셀에서 해당 인스턴스의 중심점까지의 상대적 거리를 예측하는 Affinity Field를 도입하였다. 이는 기존 Occupancy 모델에 쉽게 추가할 수 있는 plug-and-play 모듈 형태로 설계되었다.
2. **2D-3D Association Strategy**: Grounded-SAM의 2D 인스턴스 마스크와 3D Occupancy 인스턴스를 정렬하기 위한 전략을 제안하였다. 2D 픽셀을 3D 스캔라인(Scan-line) 복셀로 투영하고, 앞서 예측한 Affinity Field와 클러스터링 알고리즘(DBSCAN)을 결합하여 2D 프롬프트에 대응하는 3D 복셀 그룹을 정확하게 식별한다.

## 📎 Related Works

### 1. Segmentation Task

기존의 Segmentation은 단순히 픽셀/복셀의 클래스를 분류하는 Semantic Segmentation, 개별 객체를 구분하는 Instance Segmentation, 그리고 이 둘을 통합한 Panoptic Segmentation으로 발전해 왔다. 하지만 이러한 2D 기반 방식들은 자율 주행과 같은 환경에서 객체의 3D 기하학적 상태를 포착하지 못한다는 한계가 있다.

### 2. Occupancy Perception

MonoScene, OccDepth, VoxFormer, SurroundOcc 등 다양한 모델들이 단일 또는 다중 카메라 입력을 통해 3D 공간의 점유 상태와 시맨틱을 예측하는 연구를 수행해 왔다. 그러나 대부분의 연구가 시맨틱 레이블링에 집중되어 있으며, 인스턴스 구분(Instance-level)에 대한 접근은 거의 이루어지지 않았다.

### 3. Visual Grounding

GLIP, Grounding DINO, SAM(Segment Anything Model) 등은 텍스트나 박스 프롬프트를 통해 이미지 내 특정 영역을 정밀하게 분할하는 능력을 보여주었다. 특히 Grounded-SAM은 텍스트 기반 탐지와 정밀 분할을 통합하여 강력한 2D Visual Grounding 성능을 제공한다. 본 논문은 이러한 2D 능력을 3D Occupancy 공간으로 전이시키는 것을 목표로 한다.

## 🛠️ Methodology

### 1. Affinity Field를 통한 Instance Segmentation

본 연구는 인스턴스 구분을 위해 Bounding Box를 먼저 예측하는 2-stage 방식 대신, 복셀 간의 관계를 직접 예측하는 Affinity Field Regression 방식을 사용한다.

**중심점 계산 및 Ground Truth 정의**:
특정 인스턴스에 속한 $n$개의 복셀이 있을 때, 해당 인스턴스의 기하학적 중심점 $(\bar{x}, \bar{y}, \bar{z})$는 다음과 같이 계산된다.
$$\left( \bar{x}, \bar{y}, \bar{z} \right) = \left( \frac{1}{n} \sum_{i=1}^{n} x_i, \frac{1}{n} \sum_{n=1}^{n} y_i, \frac{1}{n} \sum_{i=1}^{n} z_i \right)$$
각 복셀 $A$의 Ground Truth affinity 값은 해당 복셀의 원래 위치에서 중심점을 뺀 벡터값으로 정의된다. 빈 공간(empty voxels)의 값은 0으로 설정한다.

**학습 과정 및 손실 함수**:
Affinity 예측 헤드 $\Psi$는 MonoScene의 Segmentation 헤드와 동일한 구조를 가지며, 출력 채널만 $\Delta(x, y, z)$에 대응하는 3개로 변경된다. 손실 함수로는 Mean Square Error (MSE)를 사용하며, 빈 복셀을 제외하기 위한 마스크 $M$을 적용한다.
$$\mathcal{L}_{aff} = \text{MSE}(\Psi(F^{3D}) \cdot M, A)$$
전체 손실 함수는 기존의 Occupancy 손실($\mathcal{L}_{ori}$)에 Affinity 손실을 추가한 형태이다.
$$\mathcal{L}_{total} = \mathcal{L}_{ori} + \lambda \mathcal{L}_{aff}$$

### 2. 2D-3D Adapter 및 Visual Grounding

텍스트 프롬프트를 통해 3D 공간의 특정 인스턴스를 추출하기 위해 핀홀 카메라 모델(Pinhole Camera Model)을 기반으로 한 **Pixel to Voxel Transformation and Clustering** 알고리즘을 제안한다.

**추론 절차**:

1. **2D segmentation**: 사용자의 텍스트 프롬프트를 Grounded-SAM에 입력하여 2D 인스턴스 마스크를 생성한다.
2. **Pixel $\rightarrow$ Voxel 투영**: 마스크에 해당하는 모든 2D 픽셀을 카메라의 내/외부 파라미터를 이용하여 3D 공간의 스캔라인(Scan-line) 형태의 복셀 리스트(Candidate voxels)로 변환한다.
3. **Filtering**: 변환된 복셀들 중 MonoScene이 예측한 시맨틱 레이블이 배경(background)이 아닌 것들만 추출하여 foreground voxel 리스트를 구성한다.
4. **Clustering**: 추출된 foreground 복셀들을 대상으로, 예측된 Affinity Field 정보를 활용하여 DBSCAN과 같은 클러스터링 알고리즘을 적용한다.
5. **Occlusion Handling**: 만약 투영된 라인 상에 여러 개의 클러스터가 발견될 경우, 가려짐(occlusion) 현상을 고려하여 카메라에서 가장 가까운 클러스터를 최종 결과로 반환한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: 실내 장면 1,449개로 구성된 NYUv2 데이터셋을 사용하였다.
- **분할**: 학습 데이터 795개, 추론 데이터 654개를 사용하였으며, 스케일은 1:4로 설정하였다.
- **구현 상세**: MonoScene의 백본을 유지한 채 Affinity 예측 헤드만 추가하였으며, 8대의 V100 GPU를 사용하여 50 epoch 동안 학습하였다.

### 2. 결과 분석

본 논문은 정량적인 수치보다는 정성적인 시각화 결과(Fig. 2)를 통해 성능을 입증한다.

- **Instance Segmentation**: 서로 다른 객체들이 동일한 시맨틱 클래스일지라도 개별적인 색상으로 구분되어 출력되는 것을 확인하였다.
- **Visual Grounding**: "sofa"와 같은 텍스트 프롬프트를 입력했을 때, Grounded-SAM의 2D 마스크가 3D 공간의 특정 소파 인스턴스 복셀들과 정확하게 매칭되어 추출되는 결과를 보여주었다.

## 🧠 Insights & Discussion

### 강점

- **확장성**: Affinity Field 기반의 모듈 설계 덕분에 기존의 어떤 Occupancy 프레임워크에도 최소한의 수정만으로 적용 가능한 plug-and-play 특성을 가진다.
- **최초의 시도**: Occupancy Prediction 분야에서 Instance Segmentation과 Visual Grounding을 동시에 구현하려 한 첫 번째 시도라는 점에서 학술적 가치가 크다.

### 한계 및 비판적 해석

- **계산 복잡도**: Grounded-SAM이라는 무거운 외부 모델을 파이프라인에 통합함으로써 전체 시스템의 추론 속도가 저하되는 문제가 있다. 2D 모델과 3D 모델이 분리되어 있어 실시간성 확보가 어려울 것으로 보인다.
- **단순한 정렬 전략**: 2D-3D 정렬 시 단순히 가장 가까운 클러스터를 선택하는 방식은 복잡한 가려짐 상황에서 오작동할 가능성이 있다.
- **평가 지표의 부재**: 정량적인 벤치마크 지표(예: mIoU, instance-level accuracy 등)가 제시되지 않고 시각적 결과에 의존하고 있어, 객관적인 성능 측정에 한계가 있다.

## 📌 TL;DR

본 논문은 3D Occupancy Prediction에 **Affinity Field**를 도입하여 인스턴스 구분 능력을 부여하고, 이를 **Grounded-SAM**과 결합하여 텍스트 기반의 3D 객체 추출(Visual Grounding)을 가능하게 한 연구이다. 2D 마스크를 3D 스캔라인으로 투영하고 클러스터링하는 전략을 통해 2D-3D 간의 정렬 문제를 해결하였다. 향후 Grounding 모듈을 Occupancy 모델 내부에 직접 통합하여 효율성을 높이는 연구가 필요할 것으로 보인다.
