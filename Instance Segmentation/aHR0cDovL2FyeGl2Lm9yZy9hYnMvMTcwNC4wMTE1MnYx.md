# Pose2Instance: Harnessing Keypoints for Person Instance Segmentation

Subarna Tripathi, Maxwell D. Collins, Matthew Brown, Serge Belongie (2017)

## 🧩 Problem to Solve

본 논문은 다수의 인물이 등장하는 이미지에서 각 개인을 픽셀 단위로 분리해내는 **Person Instance Segmentation** 문제를 해결하고자 한다. Instance Segmentation은 단순히 객체의 범주를 분류하는 Semantic Segmentation보다 더 어려운 과제로, 특히 서로 비슷한 외관을 가진 인스턴스들을 개별적으로 분리하는 것이 매우 까다롭다.

기존의 일반적인 Instance Segmentation 방법론들은 객체의 범주와 상관없이 범용적인 딥러닝 모델을 사용한다. 그러나 저자들은 인체라는 특수한 대상에 대해서는 **인체 구조(Human-specific domain knowledge)**라는 강력한 사전 지식을 활용할 수 있을 것이라는 점에 주목하였다. 따라서 본 연구의 목표는 인체 핵심 지점인 **Keypoints** 정보를 인스턴스 분할의 Prior(사전 정보)로 활용하여 segmentation 성능을 향상시키는 프레임워크를 제안하고, 그 효과를 정량적으로 검증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 인체 Keypoints로부터 유도된 **Distance Transform(거리 변환)** 또는 **Keypoint Heatmaps**를 딥러닝 네트워크의 입력으로 사용하여, 인체의 대략적인 형태(Gross shape)를 가이드로 제공하는 것이다.

이를 위해 두 가지 접근 방식을 제안한다. 첫째, Oracle(정답) Keypoints가 주어졌을 때 기존 Semantic Segmentation 모델의 추론 단계에서 이를 결합하여 성능을 높이는 방법이다. 둘째, Keypoint 추정과 Instance Segmentation을 동시에 학습하며, 추정된 Pose 정보를 Segmentation의 조건(Condition)으로 사용하는 **Pose2Instance** 딥러닝 모델을 제안한다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 언급한다.

- **Semantic & Instance Segmentation**: DeepLab과 FCN이 Semantic Segmentation의 돌파구를 마련하였으며, 이후 다양한 Instance Segmentation 방법들이 제안되었다. 그러나 대부분의 방법은 객체 범주별 명시적인 형태(Shape) 학습을 고려하지 않는다.
- **Human Pose Estimation**: 정적 이미지나 비디오에서 인체의 관절 위치를 찾는 연구들이 활발히 진행되었으며, 최근에는 CNN 기반의 모델들이 높은 성능을 보이고 있다.
- **Joint Pose Estimation and Segmentation**: PoseCut과 같이 Pose와 Segmentation을 동시에 수행하려는 시도가 있었다. PoseCut은 CRF(Conditional Random Field) 기반으로 인체 스켈레톤의 Distance Transform을 Prior로 사용하지만, MAP(Maximum A Posteriori) 솔루션을 찾기 위한 최적화 과정이 필요하여 추론 시간이 매우 오래 걸린다는 한계가 있다.

**차별점**: 제안된 Pose2Instance는 최적화 과정 없이 단 한 번의 Forward pass만으로 예측을 수행하며, 딥러닝 네트워크 내에서 Shape-to-Segmentation 매핑을 학습 가능한 파라미터로 구현하여 효율성과 성능을 동시에 잡고자 한다.

## 🛠️ Methodology

본 논문은 Pose prior를 통합하는 방법을 두 단계로 나누어 설명한다.

### 1. Pose2Instance Inference Only (Constrained Setup)
Oracle Keypoints가 제공된다는 가정하에, 추가 학습 없이 추론 단계에서만 Pose 정보를 활용하는 방법이다.

- **RAG (Region Adjacency Graph) 생성**: SLIC superpixels와 Sobel 연산자를 사용하여 이미지의 엣지 강도에 기반한 RAG $G=(V, E)$를 구축한다.
- **Pose Prior 생성**: 
    - Oracle Keypoints로 연결된 스켈레톤 라인이 포함된 superpixel 노드들에 가장 높은 확률을 부여한다.
    - Floyd-Warshall 최단 경로 알고리즘을 사용하여 RAG 상에서 **Distance Transform**을 수행한다.
    - 결과물인 Point-wise softmax 값은 각 인스턴스의 대략적인 형태를 나타내는 **Pose-instance map**이 된다.
- **최종 추론**: $\text{DeepLab-people}$ 모델의 스코어와 위에서 구한 $\text{Pose-instance map}$을 원소별 곱셈(Element-wise multiplication) 한 뒤, $\text{argmax}$를 취하여 최종 인스턴스를 분할한다.

### 2. Learning Pose2Instance (Realistic Setup)
Oracle Keypoints 없이 모델이 스스로 Pose와 Segmentation을 함께 학습하는 방식이다. 기본 아키텍처는 Atrous convolution이 적용된 modified VGG (DeepLab 스타일)를 사용한다.

두 가지 네트워크 구조를 탐색하였다.
- **Multitask Model (Pose and Seg)**: 
    - 공유된 CNN 백본 뒤에 두 개의 병렬 출력 레이어를 둔다.
    - 하나는 2-class segmentation 레이어, 다른 하나는 17-channel pose estimation 레이어이다.
    - 각각 Softmax와 Sigmoid 활성화 함수 뒤에 Cross-entropy loss를 사용하여 학습한다.
- **Cascaded Model (Pose2Seg)**: 
    - Pose estimation 레이어의 출력인 17-channel heatmap을 $1 \times 1$ Convolution 레이어에 통과시켜 **Shape likelihood**를 생성한다.
    - 이 Shape likelihood를 Segmentation feature map과 결합하여 최종 Segmentation 결과물을 낸다.
    - 이 구조는 포즈 정보를 직접적으로 segmentation의 조건으로 사용하며, $1 \times 1$ conv 커널을 통해 포즈 맵을 세그멘테이션 마스크로 매핑하는 법을 학습한다.

## 📊 Results

### 실험 설정
- **데이터셋**: COCO 데이터셋 중 Instance segmentation과 Human keypoints 어노테이션이 모두 존재하는 이미지들(Train: 45,174장, Val: 21,634장)을 사용하였다.
- **지표**: $AP_r$ (Average Precision)을 $IoU=0.5$ 및 $IoU=[0.5, 0.9]$ 구간에서 측정하였다.

### 주요 결과
1. **Constrained Setup (Oracle Keypoints)**:
    - Oracle Bounding Box를 사용한 베이스라인 대비, Oracle Keypoints를 활용한 Pose2Instance 추론 방식이 $IoU$ 임계값에 따라 **10% ~ 12%의 상대적 성능 향상**을 보였다. 이는 단순한 박스 정보보다 인체 스켈레톤 구조 정보가 훨씬 유용함을 시사한다.

2. **Realistic Environment (No Oracle Keypoints)**:
    - **Segmentation only** 모델 대비 **Cascaded (Pose2Seg)** 모델이 $IoU=0.5$에서 $AP_r$이 $0.79 \rightarrow 0.82$로 상승하였으며, 상대적 개선도는 **3.8% ~ 10.5%**에 달했다.
    - Multitask 모델 또한 성능 향상이 있었으나, Cascaded 모델이 더 우수한 결과를 보였다.

| Methods | $AP_r$ ($IoU=0.5$) | $AP_r$ ($IoU=[0.5, 0.95]$) |
| :--- | :---: | :---: |
| DeepLab Seg only | 0.79 | 0.38 |
| Multitask: Pose and Seg | 0.80 | 0.40 |
| Cascaded: Pose2Seg | $\mathbf{0.82}$ | $\mathbf{0.42}$ |

## 🧠 Insights & Discussion

**강점 및 의의**
- 최신 딥러닝 모델에서도 인체 포즈라는 도메인 지식을 추가하는 것이 유의미한 성능 향상을 가져온다는 것을 입증하였다.
- 특히 인물들이 서로 겹쳐 있거나 밀집해 있는 상황에서, 포즈 정보를 통한 가이드가 단순한 Bounding Box보다 훨씬 정교하게 인스턴스를 분리해낼 수 있음을 확인하였다.
- 제안된 방법론은 DeepLab과 같은 구조를 공유하는 다른 최신 세그멘테이션 모델에도 쉽게 적용 가능하다.

**한계 및 비판적 해석**
- 본 연구에서 사용한 VGG 기반의 Pose estimator 성능 한계로 인해, 사람이 매우 밀집된 복잡한 장면에서는 여전히 예측 결과가 불완전한 경우가 존재한다. 즉, Segmentation 성능의 상한선이 Pose estimation의 정확도에 종속되어 있다.
- 실험에서 $1 \times 1$ Convolution을 통한 Shape likelihood 생성이 유효함을 보였으나, 더 복잡한 형태의 Pose-to-Shape 매핑 네트워크를 사용했을 때의 잠재적 이득에 대해서는 다루지 않았다.

## 📌 TL;DR

본 논문은 인체 Keypoints 정보를 Prior로 활용하여 Person Instance Segmentation 성능을 높이는 **Pose2Instance** 프레임워크를 제안한다. Oracle Keypoints를 이용한 추론 실험과, Pose-Segmentation을 결합하여 학습하는 Cascaded 모델을 통해 기존 Segmentation 전용 모델 대비 최대 **10.5%의 성능 향상**을 거두었다. 이 연구는 복잡한 자연 장면에서 인체 구조 정보를 딥러닝 모델에 통합하는 효과적인 방법을 제시하였으며, 향후 비디오 세그멘테이션 등으로 확장될 가능성이 크다.