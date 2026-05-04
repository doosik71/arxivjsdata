# INSTANCE SEGMENTATION OF BUILDINGS USING KEYPOINTS

Qingyu Li, Lichao Mou, Yuansheng Hua, Yao Sun, Pu Jin, Yilei Shi, Xiao Xiang Zhu (2020)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 고해상도 원격 탐사 이미지(Remote Sensing Imagery)에서 건물 인스턴스 분할(Instance Segmentation) 시 발생하는 경계 모호성 문제이다.

기존의 Semantic Segmentation 및 Instance Segmentation 방법들은 네트워크 내부의 Pooling layer로 인해 공간 정보가 손실되는 경향이 있으며, 이로 인해 생성된 세그멘테이션 마스크의 경계선이 뭉개지거나 흐릿하게(blurred boundaries) 표현되는 한계가 있다. 건물은 도시 계획, 토지 이용 관리 및 모니터링 등 다양한 응용 분야에서 매우 중요한 요소이며, 특히 건물은 인공 구조물로서 뚜렷한 모서리(corner points)를 가진다는 기하학적 특성이 있다. 따라서 본 연구의 목표는 이러한 건물의 기하학적 특성을 활용하여 경계가 날카롭고 정교하게 보존된 건물 분할 모델을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 건물을 직접적인 마스크 형태로 학습하는 대신, 건물을 구성하는 여러 개의 **Keypoints(핵심점)**를 검출하고 이를 연결하여 닫힌 다각형(Closed Polygon)으로 재구성하는 Bottom-up 방식의 인스턴스 분할 구조를 제안한 것이다.

이 접근 방식의 중심적인 직관은 건물의 형태와 구조를 결정짓는 결정적인 요소가 모서리 점들이라는 점에 있다. 직접적인 마스크 예측 대신 Keypoints를 검출하고 이를 기하학적으로 연결함으로써, 기존 딥러닝 모델들이 겪었던 경계 흐림 현상을 극복하고 건물의 날카로운 경계선과 세부 기하학적 디테일을 보존할 수 있다.

## 📎 Related Works

논문에서는 Keypoints를 사용하여 건물 분할을 수행하는 기존 연구로 **PolyMapper**를 언급한다. PolyMapper는 CNN-RNN 구조를 통해 Keypoints를 예측하고 그룹화하는 방식을 취한다. 하지만 본 논문은 다음과 같은 두 가지 측면에서 PolyMapper와 차별점을 가진다.

1. **Keypoint Detection 방식**: PolyMapper는 먼저 건물 경계의 Heatmap mask를 생성한 뒤 추가적인 Convolutional layer를 통해 후보 Keypoints를 얻는 중간 단계가 존재한다. 반면, 제안 방법은 이러한 중간 학습 과정 없이 입력 이미지에서 직접적으로 Keypoints를 검출한다.
2. **Grouping 방식**: PolyMapper가 딥러닝 기반의 특징 학습을 통해 점들을 그룹화하는 것과 달리, 본 논문의 방법은 딥러닝을 배제하고 순수하게 기하학적 거리 기반의 Grouping 방식을 채택하여 효율성을 높였다.

## 🛠️ Methodology

### 전체 파이프라인

제안된 시스템은 **CNN $\rightarrow$ Region Proposal Network (RPN) $\rightarrow$ Fully Convolutional Network (FCN)** 순으로 구성된 2단계(Two-stage) 구조를 가진다.

1. **특징 추출 및 제안**: 먼저 CNN을 통해 특징 맵(Feature maps)을 추출하고, RPN이 특징 맵 위를 슬라이딩하며 건물이 존재할 가능성이 높은 후보 영역(Candidate bounding boxes)을 생성한다.
2. **RoI 정렬 및 Keypoint 예측**: 생성된 각 Proposal에 대해 RoIAlign을 통해 지역 특징을 획득하고, FCN을 적용하여 해당 영역 내 Keypoints의 Heatmap을 예측한다.
3. **다각형 재구성**: 예측된 Heatmap에서 Keypoints를 추출하고, 이를 기하학적으로 연결하여 건물의 최종 세그멘테이션 마스크(Polygon map)를 생성한다.

### Keypoints Detection 및 학습 목표

네트워크는 입력 패치에 대해 Keypoints의 위치를 나타내는 Heatmap $Y \in (0,1)^{H \times W}$를 예측한다. 학습을 위한 Ground Truth는 각 Keypoint를 가우시안 커널의 중심으로 하는 Gaussian heatmap $P \in (0,1)^{H \times W}$를 사용하여 생성한다.

Keypoint 추정을 위해 긍정(positive) 위치와 부정(negative) 위치 사이의 불균형을 해소할 수 있는 **Modified Focal Loss**를 사용하며, 그 식은 다음과 같다.

$$L_{polygon} = -\frac{1}{N} \sum_{i=1}^{H} \sum_{j=1}^{W} \begin{cases} (1-P_{ij})^\alpha \log(P_{ij}) & \text{if } Y_{ij}=1 \\ (1-Y_{ij})^\beta (P_{ij})^\alpha \log(1-P_{ij}) & \text{otherwise} \end{cases}$$

여기서 $N$은 패치 내 객체의 수이며, 하이퍼파라미터 $\alpha=2, \beta=4$로 설정되었다. 전체 네트워크의 손실 함수는 클래스 분류 손실($L_{cls}$), 바운딩 박스 회귀 손실($L_{box}$), 그리고 위에서 정의한 다각형 손실($L_{polygon}$)의 합으로 정의된다.

$$L = L_{cls} + L_{box} + L_{polygon}$$

### Keypoint 추출 및 Grouping 알고리즘

1. **ExtractPeak**: 예측된 Heatmap에서 임계값 $\tau=0.1$보다 큰 픽셀들을 선택하고, $3 \times 3$ 윈도우 내에서 지역 최댓값(locally maximum)인 지점을 최종 Keypoint로 추출한다.
2. **Geometric Grouping**: 추출된 점들을 다음과 같은 순서로 연결하여 다각형을 형성한다.
    - 가장 왼쪽, 오른쪽, 위, 아래 중 하나인 극단적 Keypoint(extreme keypoint)를 시작점으로 선택한다.
    - 현재 점과 가장 가까운 이웃 점(nearest neighbour)을 찾아 엣지를 생성하고, 해당 점을 다시 시작점으로 설정한다.
    - 이 과정을 반복하여 마지막 점이 처음 시작점과 만날 때까지 연결함으로써 폐곡선 다각형을 완성한다.

## 📊 Results

### 실험 설정

- **데이터셋**: AIRS (Aerial Imagery for Roof Segmentation) 데이터셋에서 개별 건물이 중앙에 위치한 $512 \times 512$ 크기의 패치 1,680개를 추출하여 사용하였다. (Train: 1400, Val: 140, Test: 140)
- **구현 환경**: Keras 프레임워크, NVIDIA Tesla P100 GPU 사용. SGD 옵티마이저(Momentum 0.9)로 40 에포크(epochs) 동안 학습하였다.
- **평가 지표**:
  - 마스크 정확도: F1-Score, Intersection Over Union (IoU)
  - 경계선 정확도: Structural Similarity Index (SSIM), F-Measure

### 정량적 결과

제안 방법은 기존의 Semantic Segmentation 모델인 FCN-8s와 Instance Segmentation 모델인 Mask R-CNN보다 우수한 성능을 보였다.

| Method | F1-Score | IoU | SSIM | F-Measure |
| :--- | :---: | :---: | :---: | :---: |
| FCN-8s | 89.01% | 83.15% | 92.58% | 8.29% |
| Mask R-CNN | 94.73% | 90.22% | 96.82% | 9.63% |
| **Proposed method** | **95.08%** | **90.81%** | **96.93%** | **11.29%** |

특히 경계선 정확도를 측정하는 **F-Measure**에서 Mask R-CNN 대비 유의미한 향상을 보였으며, 이는 제안 방법이 건물의 기하학적 구조와 날카로운 경계를 훨씬 더 잘 보존함을 입증한다.

## 🧠 Insights & Discussion

본 논문의 가장 큰 강점은 마스크를 직접 예측하는 방식에서 벗어나, 건물의 기하학적 특성인 '모서리(Keypoints)'에 집중함으로써 딥러닝 모델의 고질적인 문제인 경계 흐림 현상을 효과적으로 해결했다는 점이다. 특히 정량적 지표 중 F-Measure의 상승은 단순한 면적 일치도(IoU)를 넘어, 실제 건물의 외곽선 형태가 지상 실측 데이터(Ground Reference)와 매우 유사하게 생성되었음을 의미한다.

다만, 본 논문에서 제시한 Grouping 방식은 매우 단순한 '최근접 이웃 연결' 방식을 사용하고 있다. 이는 건물이 단순한 볼록 다각형(convex polygon) 형태일 때는 효과적이지만, 매우 복잡한 형태의 건물이나 Keypoint 검출에 노이즈가 섞였을 경우 다각형의 위상(topology)이 꼬일 가능성이 있다. 또한, Keypoint 추출을 위한 임계값 $\tau$ 설정이 결과에 영향을 미칠 수 있으나 이에 대한 민감도 분석은 명시되지 않았다.

그럼에도 불구하고, 이 연구는 픽셀 단위의 마스크를 벡터 형태의 다각형으로 변환하는 과정이 필수적인 벡터화(Vectorization) 작업에 매우 유리한 기반을 제공한다는 점에서 실용적 가치가 높다.

## 📌 TL;DR

이 논문은 고해상도 항공 이미지에서 건물을 분할할 때 발생하는 경계 흐림 문제를 해결하기 위해, 건물을 **Keypoints의 집합**으로 정의하고 이를 연결하여 다각형 마스크를 생성하는 Bottom-up 인스턴스 분할 방법을 제안한다. Mask R-CNN과 같은 기존 방식보다 경계선 보존 능력이 탁월하며, 특히 기하학적 디테일이 중요한 건물 벡터화 작업에 매우 유용한 접근 방식을 제시하였다.
