# TrackNet: Simultaneous Object Detection and Tracking and Its Application in Traffic Video Analysis

Chenge Li, Gregory Dobler, Xin Feng, Yao Wang (2019)

## 🧩 Problem to Solve

본 논문은 객체 검출(Object Detection)과 객체 추적(Object Tracking)이 일반적으로 서로 독립적인 두 가지 프로세스로 처리되는 기존의 방식에 의문을 제기한다. 일반적인 '검출 기반 추적(tracking-by-detection)' 파이프라인은 모든 프레임에서 객체를 성공적으로 검출한 뒤, 이 검출 결과들을 연관(association)시키는 방식을 취한다. 이러한 방식은 매 프레임마다 검출기를 실행해야 하므로 계산 비용이 높으며, 개별 프레임의 검출 성능에 전체 추적 성능이 종속된다는 한계가 있다.

따라서 본 연구의 목표는 단일 네트워크를 통해 객체 검출과 추적을 동시에 수행하는 통합 프레임워크를 구축하는 것이다. 연구진은 비디오 세그먼트 내에서 움직이는 객체를 둘러싼 3D 튜브(3D Tube)를 직접 검출함으로써, 공간적 외형(Spatial Appearance)과 시간적 움직임(Temporal Motion) 특징을 동시에 활용하여 이 문제를 해결하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 비디오의 연속된 프레임 집합인 Group of Pictures (GoP)를 하나의 3D 볼륨으로 간주하고, 그 안에서 객체의 궤적을 나타내는 **Bounding Tube**를 검출하는 것이다. 구체적인 주요 기여 사항은 다음과 같다.

1.  **Bounding Tube 개념의 도입**: 매 프레임 Bounding Box를 검출하고 이를 연결하는 대신, 시간 축을 포함한 3D 튜브를 직접 예측함으로써 단 한 번의 Forward Pass로 객체의 시공간적 위치를 획득한다.
2.  **Tilted Anchor Tubes**: 기존의 튜브 제안 방식이 정지된 상태의 튜브(Stationary Tubes)만을 사용했던 것과 달리, Optical Flow에서 유도된 Motion Vector(MV)를 활용하여 기울어진 형태의 앵커 튜브를 생성함으로써 움직이는 객체에 대한 초기 제안 성능을 높였다.
3.  **Two-Stream Backbone 및 Spatial Transformer**: 외형 정보를 위한 VGG16과 움직임 정보를 위한 C3D-like 네트워크를 결합하고, Spatial Transformer를 통해 다양한 시점(Viewing Angle)에서 오는 특징들을 통일된 매니폴드로 매핑하였다.

## 📎 Related Works

기존의 객체 추적 시스템은 크게 검출 기반 추적(DBT)과 검출 없는 추적(DFT)으로 나뉜다. DBT는 매 프레임 검출이 선행되어야 하며, DFT는 초기 프레임에서 수동 초기화가 필요하다는 단점이 있다.

최근의 튜브 제안(Tube Proposal) 기반 연구들은 앵커 튜브를 생성할 때 동일한 Bounding Box를 여러 프레임에 복제하는 'Stationary Tubes' 방식을 사용하였다. 하지만 이는 움직이는 객체와 앵커 튜브 간의 Overlap(IoU)을 낮추어 학습 효율을 떨어뜨리며, 단순히 공간적 특징만을 풀링(Pooling)할 경우 명시적인 움직임 정보를 활용하지 못한다는 한계가 있다. TrackNet은 앵커 단계부터 움직임 벡터를 반영하고 3D Convolution을 통해 시공간 특징을 동시에 추출함으로써 기존 방식과 차별화된다.

## 🛠️ Methodology

### 1. 전체 파이프라인 구조
TrackNet은 크게 세 단계의 프로세스로 구성된다: **특징 추출 및 공간 변환 $\rightarrow$ Tube Proposal Network (TPN) $\rightarrow$ Post-TPN 분류 및 정밀화**.

### 2. Two-Stream Feature Extraction & Spatial Transformer
- **Two-Stream Backbone**: VGG16(2D CNN)은 객체의 외형에, C3D(3D CNN)는 객체의 움직임에 집중하여 특징을 추출한다. 추출된 특징 맵들은 'Squashing' Convolutional Layer($1 \times 1$ 커널)를 통해 채널 수를 128로 줄인 후 결합된다.
- **Spatial Transformer**: 다양한 각도에서 촬영된 객체의 외형을 통일하기 위해 Affine Transformation을 수행한다. 학습 가능한 모듈이 변환 파라미터 $\theta$를 출력하며, 이를 통해 특징 맵 $U$를 변형된 특징 맵 $V$로 매핑한다.

### 3. Tube Proposal Network (TPN)
TPN은 각 픽셀 위치에서 여러 개의 앵커 튜브를 생성하고, 해당 튜브가 객체를 포함하고 있는지(Objectness score)와 실제 위치와의 오차(Offset)를 예측한다.

- **Tilted Anchor Tubes**: 정지 튜브 $T^s$에 평균 움직임 벡터 $mv$를 적용하여 기울어진 튜브 $T^t$를 생성한다. 프레임 인덱스 $t$에 따른 박스 위치는 다음과 같다.
  $$T^t[t,:,:,k] = T^s[t,:,:,k] + mv \times (t-1)$$
- **Tube Offset Regression**: 앵커 튜브를 실제 Ground Truth 튜브에 맞게 '구부리는' 작업을 수행한다. 두 가지 옵션이 제시되었으며, TPN 단계에서는 파라미터 수를 줄이고 부드러운 움직임을 강제하기 위해 **선형 보간(Linear Interpolation, LP)** 방식을 사용한다. 이는 시작 프레임과 종료 프레임의 오프셋만 예측하고 중간 프레임은 선형적으로 보간하는 방식이다.

### 4. Post-TPN: Classification and Refinement
TPN에서 제안된 튜브 중 Objectness score가 높은 것들을 대상으로 더 정밀한 분류(예: 자동차, 버스, 밴)와 위치 정밀화를 수행한다. 이때 **Tube Pooling**을 사용하는데, 튜브 내 모든 Bounding Box의 합집합(Union) 영역에서 특징을 추출하여 FC Layer로 전달한다.

### 5. 학습 목표 및 손실 함수
모델은 분류 손실($L_{cls}$)과 회귀 손실($L_{reg}$)을 동시에 최적화하는 Multi-task Loss를 사용한다.
$$L(s_i, p_i) = \lambda_1 L_{cls}(l_i, s_i) + \lambda_2 \sum_{t=1}^T L_{reg}(tar_{i,t}, p_{i,t}) + \lambda_3 L_{cls}^{TPN}(l_i, s_i) + \lambda_4 \sum_{t=1}^T L_{reg}^{TPN}(tar_{i,t}, p_{i,t}) + \lambda_5 L_{smooth}$$
여기서 $L_{smooth}$는 튜브의 궤적이 갑자기 변하지 않도록 강제하는 Smoothness loss이며, 회귀 손실에는 Smooth $L_1$ loss가 사용된다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: UA-DETRAC (실제 도로 교통 영상 데이터셋).
- **평가 지표**: COCO API를 사용하여 Average Precision (AP) 및 Average Recall (AR)을 측정하였다.
- **학습 트릭 (Skip Frames)**: 차량의 속도가 제각각인 점을 해결하기 위해, 연속된 8프레임 대신 랜덤한 간격(skip factor $s \in [0, 5]$)으로 프레임을 샘플링하여 학습시켜 속도 변화에 대한 강건성을 확보하였다.

### 2. 주요 결과
- **구성 요소 분석 (Ablation Study)**: VGG 브랜치와 Spatial Transformer를 추가했을 때 False Positive(오검출)가 눈에 띄게 감소하였으며, 정밀도가 향상되었다.
- **보간법의 효과**: Linear Interpolation(LP) 방식이 파라미터 수를 줄이면서도 암시적인 Smoothness Regularization 역할을 하여 성능을 높였다.
- **특징 차원(Squash Dimension)의 영향**: 특징 맵의 차원을 128에서 512로 늘렸을 때, mAP가 $37.47\%$에서 $40.45\%$로 향상되었다. 이는 차원 축소가 계산 효율성은 높이지만 일부 세부 정보를 손실시켜 정밀한 위치 추정에 영향을 줌을 시사한다.
- **시각적 결과**: 다양한 조명 조건과 날씨에서도 강건하게 작동하며, 빠른 차량은 '성긴(sparse)' 튜브로, 느린 차량은 '밀집된(dense)' 튜브로 생성하는 특성을 보였다.

## 🧠 Insights & Discussion

### 강점
본 논문은 객체 검출과 추적을 하나의 3D 튜브 검출 문제로 통합함으로써, 기존의 '검출 후 연관' 방식이 가진 계산 효율성 및 종속성 문제를 효과적으로 완화하였다. 특히 움직임 벡터를 이용한 Tilted Anchor와 시공간 통합 특징 추출 구조는 동적인 환경인 교통 영상 분석에서 높은 정밀도를 보여준다.

### 한계 및 비판적 해석
논문에서도 언급되었듯이, **정밀한 위치 추정(Precise Localization)** 능력이 다소 부족하다. 이는 특징 추출이 프레임 단위가 아닌 GoP(8프레임) 단위로 이루어지기 때문에 시간적 해상도가 낮아지는 결과가 초래되었기 때문이다. 또한, 튜브의 궤적을 선형 보간으로 근사하는 방식은 대부분의 교통 상황에서는 유효하나, 급격한 방향 전환이나 비선형적 움직임을 보이는 객체에 대해서는 한계가 있을 것으로 판단된다.

## 📌 TL;DR

TrackNet은 비디오를 3D 볼륨으로 처리하여 객체의 궤적인 **Bounding Tube**를 직접 검출하는 통합 네트워크이다. VGG(외형)와 C3D(움직임)의 Two-stream 구조와 Motion Vector 기반의 Tilted Anchor를 통해 검출과 추적을 동시에 수행한다. UA-DETRAC 데이터셋에서 유효성을 입증하였으며, 특히 교통 영상과 같이 움직임이 비교적 일정한 환경에서 효율적인 다중 객체 추적 솔루션을 제공한다. 향후 멀티 스케일 풀링과 복잡한 움직임 패턴 학습을 통해 정밀도를 높일 가능성이 크다.