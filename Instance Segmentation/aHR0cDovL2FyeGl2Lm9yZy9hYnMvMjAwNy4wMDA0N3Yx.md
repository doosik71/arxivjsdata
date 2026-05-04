# A Survey on Instance Segmentation: State of the art

Abdul Mueed Hafiz, Ghulam Mohiuddin Bhat (2020)

## 🧩 Problem to Solve

본 논문은 컴퓨터 비전의 핵심 과제 중 하나인 Instance Segmentation의 발전 과정과 최신 기술 동향을 분석하는 것을 목표로 한다.

Instance Segmentation은 이미지 내의 객체 클래스를 분류하는 Image Classification, 객체의 위치를 Bounding Box로 찾는 Object Detection, 그리고 픽셀 단위로 클래스를 예측하는 Semantic Segmentation의 개념을 모두 통합한 과제이다. 구체적으로는 동일한 클래스에 속하더라도 서로 다른 개별 인스턴스(Instance)를 구분하여 각각에 대해 픽셀 단위의 마스크(Mask)를 생성해야 한다.

이 문제는 자율 주행, 로보틱스, 보안 감시 시스템 등 정밀한 객체 인식과 분리가 필요한 실제 응용 분야에서 매우 중요하다. 논문은 특히 segmentation의 정확도(Accuracy)와 효율성(Efficiency) 사이의 트레이드-오프, 다양한 스케일의 객체 탐지, 가려짐(Occlusion) 및 이미지 품질 저하 문제 등을 해결하는 기술적 흐름을 다룬다.

## ✨ Key Contributions

본 논문의 주요 기여는 Instance Segmentation 분야의 방대한 연구 결과들을 체계적으로 분류하고, 기술적 진화 과정을 분석하여 연구자들에게 종합적인 가이드를 제공하는 것이다.

1. **기술적 분류 체계(Taxonomy) 제시**: Instance Segmentation 기법을 마스크 제안 기반 분류, 탐지 후 세그멘테이션, 픽셀 라벨링 후 클러스터링, 밀집 슬라이딩 윈도우 방식의 네 가지 범주로 체계화하였다.
2. **기술 진화 타임라인 분석**: R-CNN부터 시작하여 Mask R-CNN, PANet, HTC, GCNet, YOLACT에 이르기까지 주요 알고리즘의 발전 흐름을 분석하고 각 모델의 핵심 아이디어를 설명한다.
3. **주요 데이터셋 및 벤치마크 분석**: MS COCO, Cityscapes, MVD 등 표준 데이터셋의 특성과 각 모델의 성능 지표(AP)를 비교 분석하였다.
4. **미해결 과제 제시**: 연산 자원 제약, 실시간 처리 속도, 작은 객체 탐지 등 향후 연구가 필요한 방향성을 명시하였다.

## 📎 Related Works

논문은 이미지 추론의 발전 과정을 '거친 추론(Coarse inference)'에서 '세밀한 추론(Fine inference)'으로의 진화로 설명한다.

1. **전통적 접근 방식**: 초기에는 SIFT, HOG와 같은 수작업 기반의 Local Descriptor와 Bag of Words, Fisher Vector 등을 통해 특징을 추출하였다. 그러나 이러한 방식은 도메인 전문가의 세밀한 엔지니어링이 필요하며 일반화 성능이 떨어진다는 한계가 있었다.
2. **Deep Learning의 도입**: CNN(Convolutional Neural Networks)의 등장으로 특징 추출 과정이 자동화되었으며, AlexNet, VGGNet을 거쳐 ResNet, DenseNet과 같이 네트워크의 깊이가 깊어지면서 표현력이 비약적으로 향상되었다.
3. **기존 프레임워크의 한계**:
    - **R-CNN 계열**: 다단계 파이프라인으로 인해 학습 및 테스트 속도가 매우 느리고 최적화가 어렵다.
    - **Multi-scale 문제**: CNN의 하위 레이어는 세부 정보는 많으나 시맨틱 정보가 부족하고, 상위 레이어는 시맨틱 정보는 강하지만 해상도가 낮아 작은 객체를 탐지하는 데 어려움이 있다.
    - **기하학적 변환 및 가려짐**: DCNN은 본질적으로 공간적 불변성(Spatial Invariance)이 부족하며, 객체가 겹쳐 있는 경우 정보 손실이 발생한다.

## 🛠️ Methodology

본 논문은 개별 방법론을 제안하는 것이 아니라 기존 방법론들을 분석하는 서베이 논문이므로, 분석 대상이 된 핵심 기술들의 작동 원리를 중심으로 설명한다.

### 1. Instance Segmentation 기법의 분류

- **Classification of Mask Proposals**: 마스크 후보군을 먼저 생성하고 이를 분류하는 방식이다. (예: R-CNN)
- **Detection followed by Segmentation**: Bounding Box를 먼저 예측하고 그 내부에서 세그멘테이션을 수행하는 방식이다. (예: Mask R-CNN)
- **Labelling Pixels followed by Clustering**: 모든 픽셀에 라벨을 부여한 후 클러스터링 알고리즘으로 인스턴스를 분리한다. (예: Deep Watershed Transform)
- **Dense Sliding Window Methods**: 밀집된 슬라이딩 윈도우를 통해 마스크를 예측하는 방식이다. (예: TensorMask)

### 2. 핵심 알고리즘 상세 분석

#### Mask R-CNN

Faster R-CNN에 병렬적으로 마스크 예측 브랜치(Mask Prediction Branch)를 추가한 구조이다. FPN(Feature Pyramid Network)을 백본으로 사용하여 다양한 스케일의 특징을 추출하며, ROI Align을 통해 픽셀 정렬 문제를 해결하여 정밀한 마스크를 생성한다.

#### Non-local Neural Networks

지역적인 윈도우 내에서만 연산하는 기존 CNN의 한계를 극복하기 위해, 이미지 전체의 픽셀 간 관계를 계산하는 Non-local operation을 도입하였다. 수식은 다음과 같다.
$$y_i = \frac{1}{C(x)} \sum_j f(x_i, x_j) g(x_j)$$
여기서 $i$는 출력 위치, $j$는 모든 가능한 위치를 의미하며, $f$는 두 위치 간의 유사도를 계산하는 함수, $g$는 입력 신호의 표현을 계산하는 함수이다.

#### Path Aggregation Network (PANet)

특징 피라미드(FPN)의 상향식 경로를 보강하여 하위 레이어의 정밀한 위치 정보를 상위 레이어로 효율적으로 전달하는 Bottom-up path augmentation을 제안하였다. 또한 Adaptive Feature Pooling을 통해 모든 레벨의 특징을 활용한다.

#### Hybrid Task Cascade (HTC)

Bounding Box 예측과 Mask 예측을 상호 보완적으로 엮어 다단계(Cascade)로 처리한다. 특히, 박스 예측 결과가 업데이트되면 이를 마스크 예측에 반영하는 인터리빙(Interleaving) 구조를 가지며, 전체 이미지에 대한 시맨틱 세그멘테이션 브랜치를 추가하여 공간적 문맥(Spatial Context)을 강화하였다.

#### GCNet (Global Context Network)

Non-local Network의 연산 비용을 줄이기 위해, 쿼리 위치에 상관없이 글로벌 문맥이 유사하다는 점에 착안하여 Query-independent formulation을 제안하였다. Global Context(GC) Block은 다음과 같이 정의된다.
$$t_i = x_i + W_{v2} \text{ReLU}(\text{LN}(W_{v1} \sum_j \frac{e^{W_k x_j}}{\sum_m e^{W_k x_m}} x_j))$$
이 구조는 SENet과 유사하면서도 글로벌 문맥을 효율적으로 모델링하여 성능을 높였다.

#### YOLACT

실시간 성능을 위해 프로토타입 마스크(Prototype Masks) 생성과 각 인스턴스별 마스크 계수(Mask Coefficients) 예측을 병렬로 수행한 뒤, 이들의 선형 조합으로 최종 마스크를 생성하는 Fully Convolutional 구조를 가진다.

## 📊 Results

논문은 MS COCO 데이터셋을 중심으로 주요 모델들의 성능을 비교하였다.

### 정량적 결과 (MS COCO Average Precision, AP)

- **Hybrid Task Cascade (HTC)**: 추가 학습 데이터를 사용했을 때 $43.9\%$로 가장 높은 성능을 기록하였다.
- **PANet**: $42.0\%$의 AP를 달성하며 2018년 챌린지에서 1위를 차지하였다.
- **GCNet**: $41.5\%$의 성능을 보였다.
- **Mask R-CNN**: $37.1\%$로 기준점이 되는 성능을 보여준다.
- **YOLACT**: $29.8\%$의 AP를 기록하였으나, 33 FPS라는 매우 빠른 속도로 실시간 처리가 가능함을 입증하였다.

### 사용된 데이터셋 특성

- **MS COCO**: 20만 장의 이미지, 80개 이상의 클래스를 포함하며 복잡한 공간적 배치를 가지고 있어 표준 벤치마크로 사용된다.
- **Cityscapes**: 도시 거리 장면 이미지 5,000장(정밀 주석)을 포함하며, 자율 주행 연구에 특화되어 있다.
- **Mapillary Vistas Dataset (MVD)**: 25,000장의 이미지와 66개 클래스를 포함하며, 다양한 날씨와 조명 조건의 거리 장면을 제공한다.

## 🧠 Insights & Discussion

### 1. 2-Stage vs Single-Stage

- **2-Stage (Region-based)**: Mask R-CNN과 같이 제안 영역을 먼저 뽑고 세그멘테이션을 수행하는 방식은 연산량은 많지만 정확도가 매우 높으며, 특히 작은 객체 탐지에 유리하다.
- **Single-Stage (Unified)**: YOLACT와 같이 한 번에 예측하는 방식은 전처리가 없고 백본이 가벼워 속도가 매우 빠르지만, 작은 객체 탐지 성능이 상대적으로 떨어진다.

### 2. 강점 및 약점 분석

- **Detection $\rightarrow$ Segmentation**: 학습이 비교적 쉽고 일반화 성능이 좋지만, 복잡한 학습 파이프라인에 의존한다.
- **Pixel Labelling $\rightarrow$ Clustering**: 최신 시맨틱 세그멘테이션 기법을 활용할 수 있으나, 픽셀 단위 연산으로 인해 계산 비용이 매우 높고 정확도가 상대적으로 낮다.
- **Dense Sliding Window**: 탐색되지 않은 영역이 많아 잠재력이 크지만, 알고리즘이 복잡하고 최적화가 어렵다.

### 3. 한계점 및 비판적 해석

논문은 현재의 기술들이 고품질의 데이터셋(COCO 등)에서 벤치마킹되고 있음을 지적한다. 실제 환경에서 발생하는 이미지 노이즈, 조명 변화, 심한 가려짐(Occlusion) 상황에 대한 강건성(Robustness) 연구는 여전히 부족한 실정이다. 또한, 하드웨어 자원(GPU 수 등)에 따른 성능 차이가 크기 때문에, 알고리즘의 단순함과 하드웨어 요구량 사이의 균형을 맞추는 것이 실용화의 핵심이 될 것이다.

## 📌 TL;DR

본 논문은 Instance Segmentation 기술의 진화 과정을 Image Classification $\rightarrow$ Object Detection $\rightarrow$ Semantic Segmentation $\rightarrow$ Instance Segmentation의 흐름으로 정의하고, 이를 해결하기 위한 다양한 딥러닝 아키텍처를 체계적으로 분석한 서베이 보고서이다.

**핵심 요약:**

- **기술적 흐름**: Mask R-CNN의 등장 이후, 글로벌 문맥 반영(Non-local, GCNet), 경로 최적화(PANet), 다단계 정제(HTC), 실시간성 확보(YOLACT) 방향으로 발전하였다.
- **Trade-off**: 정확도를 중시하는 2-Stage 방식과 속도를 중시하는 Single-Stage 방식의 뚜렷한 차이가 존재한다.
- **미래 방향**: 하드웨어 제약 극복, 실시간 최적화, 작은 객체 탐지 성능 향상, 그리고 신체 부위 검출(Human Parsing)과 같은 세부 영역으로의 확장이 기대된다.
