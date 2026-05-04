# YOLACT++: Better Real-time Instance Segmentation

Daniel Bolya, Chong Zhou, Fanyi Xiao, and Yong Jae Lee (2019/2020)

## 🧩 Problem to Solve

본 논문은 실시간(30fps 이상)으로 동작하면서도 경쟁력 있는 정확도를 가진 Instance Segmentation 모델을 개발하는 것을 목표로 한다. 

기존의 State-of-the-art(SOTA) 모델인 Mask R-CNN이나 FCIS와 같은 2-stage 방식은 높은 정확도를 보이지만, Bounding Box 영역 내에서 특징을 다시 추출하는 're-pooling'(예: RoI-pool/align) 과정이 필수적이다. 이러한 과정은 본질적으로 순차적(sequential)인 특성을 가지므로 연산 속도를 높이는 데 한계가 있으며, 결과적으로 실시간 성능을 달성하기 어렵다. 반면, 기존의 1-stage 방식들은 속도는 빠르지만 정확도가 현저히 낮거나 과도한 후처리 과정이 필요하다는 문제점이 있다. 따라서 본 연구는 re-pooling 단계 없이 병렬 처리가 가능한 구조를 통해 속도와 정확도라는 두 마리 토끼를 잡고자 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Instance Segmentation이라는 복잡한 작업을 두 개의 병렬적인 하위 작업으로 분리하여 처리하는 것이다.

1.  **Prototype Masks 생성**: 이미지 전체에 대해 인스턴스에 독립적인 일련의 프로토타입 마스크 세트를 생성한다.
2.  **Mask Coefficients 예측**: 각 인스턴스별로 프로토타입 마스크들을 어떻게 선형 결합할지를 결정하는 계수를 예측한다.

최종 마스크는 이 두 결과물을 선형 결합한 뒤, 예측된 Bounding Box로 크롭(crop)하여 생성한다. 이러한 설계는 무거운 re-pooling 과정을 제거하여 연산 오버헤드를 극도로 낮추면서도, 전체 이미지 해상도를 활용하므로 고품질의 마스크를 생성할 수 있게 한다. 또한, YOLACT++에서는 Deformable Convolution, 최적화된 Anchor 설정, Fast Mask Re-scoring 브랜치를 추가하여 성능을 더욱 고도화하였다.

## 📎 Related Works

기존의 Instance Segmentation 연구는 크게 다음과 같이 분류된다.

- **2-stage 방법론 (예: Mask R-CNN)**: ROI(Region of Interest)를 먼저 제안하고 이후 각 ROI에 대해 마스크를 생성한다. 정확도는 매우 높지만, ROI별로 특징을 다시 풀링해야 하므로 실시간 구현이 불가능하다.
- **1-stage 방법론 (예: FCIS, PolarMask)**: 개념적으로는 더 빠르지만, 여전히 re-pooling과 유사한 연산이나 복잡한 후처리가 필요하여 진정한 의미의 실시간(30fps) 성능에는 미치지 못한다.
- **기타 방법론**: Semantic Segmentation 후 픽셀 클러스터링이나 경계 검출을 수행하는 방식들이 있으나, 다단계 과정과 비용이 많이 드는 클러스터링 절차로 인해 실시간 적용이 어렵다.

YOLACT는 이러한 기존 방식들과 달리 마스크 조립(assembly) 단계를 단순한 행렬 곱셈으로 구현하여 연산 효율성을 극대화했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 시스템 구조
YOLACT는 기존의 1-stage Object Detector(예: RetinaNet)에 마스크 브랜치를 추가한 구조이다. 전체 과정은 **Prototype Generation $\rightarrow$ Mask Coefficient Prediction $\rightarrow$ Mask Assembly** 순으로 진행된다.

### 2. 주요 구성 요소
- **Protonet (Prototype Generation Branch)**: FCN(Fully Convolutional Network)을 사용하여 이미지 크기의 프로토타입 마스크 세트를 생성한다. FPN(Feature Pyramid Network)의 $P_3$ 레이어에서 특징을 추출하고 이를 입력 이미지 크기의 1/4로 업샘플링하여 $k$개의 채널(각 채널이 하나의 프로토타입)을 출력한다. 마지막 레이어에 $\text{ReLU}$를 적용하여 해석 가능성을 높였다.
- **Mask Coefficient Branch**: Object Detection 헤드에 병렬로 추가된 브랜치이다. 각 앵커(anchor)에 대해 $k$개의 마스크 계수를 예측하며, 출력값에 $\tanh$를 적용하여 프로토타입을 더하거나 뺄 수 있도록 설계하였다.
- **Mask Assembly**: 예측된 계수와 프로토타입 마스크를 선형 결합하고 $\text{sigmoid}$ 함수를 통과시켜 최종 마스크를 생성한다.

### 3. 주요 방정식 및 손실 함수
마스크 조립 과정은 다음과 같은 행렬 연산으로 정의된다.
$$M = \sigma(PC^T)$$
여기서 $P$는 $h \times w \times k$ 크기의 프로토타입 마스크 행렬이고, $C$는 NMS를 통과한 $n$개 인스턴스에 대한 $n \times k$ 크기의 계수 행렬이다.

학습을 위한 전체 손실 함수는 분류 손실($L_{cls}$), 박스 회귀 손실($L_{box}$), 마스크 손실($L_{mask}$)의 가중 합으로 구성된다.
$$L_{total} = L_{cls} + 1.5 L_{box} + 6.125 L_{mask}$$
이때 $L_{mask}$는 조립된 마스크 $M$과 Ground Truth 마스크 $M_{gt}$ 사이의 픽셀 단위 Binary Cross Entropy(BCE)로 계산된다.

### 4. YOLACT++의 개선 사항
- **Fast Mask Re-scoring**: 분류 신뢰도와 실제 마스크 품질 사이의 괴리를 해결하기 위해, 예측된 마스크의 IoU를 직접 회귀하는 가벼운 FCN 브랜치를 추가하여 마스크를 재정렬한다.
- **Deformable Convolutions (DCN)**: Backbone의 $C_3$부터 $C_5$ 레이어의 $3 \times 3$ 컨볼루션을 DCN으로 대체하여 다양한 스케일과 회전의 인스턴스를 더 잘 샘플링하도록 한다. 속도 저하를 막기 위해 모든 레이어가 아닌 특정 간격(interval=3)으로 DCN을 배치하였다.
- **Optimized Anchors**: FPN 레벨당 앵커의 스케일을 다양화하여 객체 검출의 재현율(Recall)을 높였다.

### 5. Fast NMS
전통적인 NMS의 순차적 연산을 병렬화하기 위해, GPU 가속 행렬 연산을 이용한 Fast NMS를 제안하였다. 모든 인스턴스의 IoU 행렬 $X$를 계산한 후, 상삼각 행렬(upper triangle)을 이용하여 자신보다 점수가 높은 박스에 의해 억제되는지를 한 번에 판단한다.
$$K_{kj} = \max_{i} (X_{kij}) \quad \forall k, j$$
이후 $K < t$ (threshold) 조건을 통해 남길 박스를 결정한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: MS COCO test-dev, Pascal 2012 SBD.
- **하드웨어**: 단일 NVIDIA Titan Xp.
- **지표**: Mask mAP, FPS (Frames Per Second).

### 2. 주요 결과
- **속도 및 정확도**: YOLACT++ (ResNet-50) 모델은 **33.5 fps**의 속도로 **34.1 mAP**를 달성하였다. 이는 Mask R-CNN보다 약 3.9배 빠르면서도 정확도 차이는 단 1.6 mAP에 불과한 수준이다.
- **마스크 품질**: Re-pooling 과정이 없기 때문에 대형 객체에 대해 Mask R-CNN이나 FCIS보다 훨씬 정교한 경계선을 생성한다. 실제로 $AP_{95}$ 지표에서 Mask R-CNN(1.3)보다 높은 1.6을 기록하며 고정밀 마스크 생성 능력을 입증하였다.
- **박스 성능**: YOLOv3와 비교했을 때 유사한 속도에서 경쟁력 있는 검출 성능을 보였으며, 마스크 브랜치의 연산 오버헤드는 단 6ms에 불과함을 확인하였다.

## 🧠 Insights & Discussion

### 1. 강점 및 통찰
- **Translation Variance의 자발적 학습**: FCN은 본래 Translation Equivariant하지만, ResNet과 같은 현대적 네트워크의 패딩(padding) 효과로 인해 네트워크가 픽셀의 상대적 위치를 인식할 수 있게 된다. YOLACT의 프로토타입들은 이를 이용하여 이미지의 특정 구역을 분할하는 'Partition map' 역할을 자발적으로 학습함을 발견하였다.
- **Temporal Stability**: 2-stage 방식은 ROI 제안 단계의 변동성 때문에 영상에서 마스크가 떨리는 현상이 발생하지만, YOLACT는 프로토타입이 안정적으로 유지되므로 훨씬 안정적인 비디오 마스크를 생성한다.

### 2. 한계 및 분석
- **Localization Failure**: 좁은 영역에 너무 많은 객체가 밀집해 있을 경우, 각 객체를 개별 프로토타입으로 분리하지 못하고 foreground 마스크처럼 뭉쳐서 출력하는 경향이 있다.
- **Leakage**: 마스크를 생성한 후 Bounding Box로 크롭하는 방식에 의존하므로, 예측된 박스가 너무 클 경우 주변의 다른 객체 노이즈가 마스크에 포함되는 'Leakage' 현상이 발생한다.

### 3. 비판적 해석
논문은 YOLACT와 Mask R-CNN의 성능 차이가 마스크 생성 알고리즘 자체가 아닌, Backbone Detector의 박스 예측 성능 차이에서 기인한다고 분석한다. 실제로 예측된 마스크를 GT로 대체했을 때의 성능 향상 폭보다 박스 성능의 차이가 더 크다는 점이 이를 뒷받침한다. 즉, 더 강력한 Object Detector를 결합한다면 실시간성을 유지하면서도 SOTA급 정확도를 달성할 가능성이 매우 높다.

## 📌 TL;DR

본 논문은 프로토타입 마스크 생성과 인스턴스별 계수 예측을 병렬로 처리하는 새로운 구조를 통해, **최초로 경쟁력 있는 정확도를 가진 실시간(>30fps) Instance Segmentation 모델인 YOLACT++를 제안**하였다. 특히 re-pooling 과정을 제거하여 연산 속도를 극대화하고 대형 객체의 마스크 품질을 높였으며, Fast NMS와 Deformable Convolution 등의 최적화를 통해 효율성을 극대화하였다. 이 연구는 고속 마스크 생성이 필수적인 자율주행, 로보틱스 및 임베디드 시스템 분야의 실시간 비전 애플리케이션에 중요한 기반이 될 것으로 기대된다.