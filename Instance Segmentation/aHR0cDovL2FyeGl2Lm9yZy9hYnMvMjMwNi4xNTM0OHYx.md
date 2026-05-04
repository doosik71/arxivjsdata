# PANet: LiDAR Panoptic Segmentation with Sparse Instance Proposal and Aggregation

Jianbiao Mei, Yu Yang, Mengmeng Wang, Xiaojun Hou, Laijian Li, Yong Liu (2023)

## 🧩 Problem to Solve

본 논문은 LiDAR Point Cloud를 이용한 Panoptic Segmentation(LPS)에서 발생하는 두 가지 주요 문제를 해결하고자 한다.

첫째, 기존의 Clustering 기반 방식들이 인스턴스의 중심점을 회귀(regression)하기 위해 학습 가능한 Offset branch에 과도하게 의존한다는 점이다. LiDAR 데이터의 희소성(sparsity), 불균일한 밀도, 그리고 다양한 객체 크기로 인해 이상적인 기하학적 shift를 예측하는 것은 매우 어렵다.

둘째, 기존의 Clustering 알고리즘들은 트럭이나 버스와 같은 대형 객체를 여러 개의 작은 인스턴스로 과분할(over-segmentation)하는 경향이 있다. 이는 결과적으로 인스턴스 분할의 완결성을 떨어뜨리는 요인이 된다.

따라서 본 연구의 목표는 학습 가능한 Offset branch에 대한 의존성을 제거하면서도, 대형 객체에 대한 분할 성능을 향상시킨 새로운 LPS 프레임워크인 PANet을 제안하는 것이다.

## ✨ Key Contributions

PANet의 핵심 아이디어는 학습이 필요 없는 **Sparse Instance Proposal (SIP)** 모듈과 분절된 인스턴스를 통합하는 **Instance Aggregation (IA)** 모듈을 결합하는 것이다.

1.  **SIP 모듈**: "Sampling-Shifting-Grouping"이라는 단계적 구조를 통해 원본 포인트 클라우드에서 효율적으로 인스턴스 제안(proposal)을 생성한다. 특히 Balanced Point Sampling(BPS)을 통해 균일한 시드 포인트를 추출하고, Bubble Shrinking(BS)을 통해 이를 중심점으로 이동시킨 후, Connected Component Labeling(CCL)으로 그룹화한다.
2.  **IA 모듈**: SIP 과정에서 발생할 수 있는 대형 객체의 과분할 문제를 해결하기 위해 KNN-Transformer를 도입한다. 인스턴스 간의 유사도(affinity)를 계산하여 동일 객체로 판단되는 분절된 인스턴스들을 병합함으로써 완결성을 높인다.
3.  **Plug-and-Play 특성**: SIP 모듈은 비학습(non-learning) 방식으로 설계되어, 특정 백본 네트워크나 데이터셋에 종속되지 않고 쉽게 확장하여 적용할 수 있다.

## 📎 Related Works

LPS 방법론은 크게 Detection-based와 Clustering-based 접근법으로 나뉜다.

-   **Detection-based**: 3D Object Detection 네트워크를 사용하여 인스턴스를 먼저 찾고 세그멘테이션을 수행한다. 하지만 이 방식은 검출기(detector)의 정확도에 전체 성능이 종속되는 한계가 있다.
-   **Clustering-based**: 중심점 회귀 및 클러스터링 알고리즘을 통해 포인트를 그룹화한다. 최근 DSNet과 같은 연구들이 학습 가능한 Dynamic Shifting 모듈을 제안했으나, 예측된 중심점이 실제 정답(Ground-truth) 중심점과 일치하지 않아 학습이 저해되는 문제가 존재한다.

PANet은 이러한 기존 방식과 달리 Offset branch를 완전히 제거한 SIP 모듈을 제안하며, 단순히 GNN을 이용해 많은 수의 제안을 병합하는 기존 연구와 달리, 대형 객체의 분절 문제에 집중하여 효율적인 KNN-Transformer 기반의 IA 모듈을 사용한다는 점에서 차별성을 갖는다.

## 🛠️ Methodology

### 1. Backbone Design
PANet은 multi-scale sparse 3D CNN과 소형 2D U-Net을 결합하여 3D 및 2D 특징을 동시에 추출한다. 
-   입력 포인트 클라우드를 Voxelization 하여 $\text{L} \times \text{H} \times \text{W}$ 해상도의 voxel-wise feature $F_v^0$를 생성한다.
-   이를 z-축으로 투영하여 BEV(Bird's Eye View) feature $F_b$를 얻는다.
-   Sparse 3D CNN을 통해 multi-scale 3D 특징을 추출하고, 2D U-Net을 통해 BEV 특징을 인코딩한다.
-   최종적으로 이들을 다시 포인트 단위로 역투영(back-project)하여 결합한 뒤, MLP를 통해 융합된 특징 $f_p \in \mathbb{R}^{N \times 64}$를 생성한다. 이 특징은 Semantic head와 Instance branch의 입력으로 사용된다.

### 2. Sparse Instance Proposal (SIP)
SIP 모듈은 학습 없이 다음 세 단계를 거쳐 인스턴스 제안을 생성한다.

**A. Balanced Point-Sampling (BPS)**
FPS(Farthest Point Sampling)의 높은 계산 비용과 Random sampling의 불균일한 분포(센서와 가까울수록 밀도가 높은 문제)를 해결하기 위해 BPS를 제안한다. 
-   입력 포인트 클라우드를 voxel block으로 나누고, 각 voxel 내 포인트들의 평균값을 시드 포인트(seed point)로 설정한다.
-   이 방식은 포인트 밀도가 아닌 voxel 해상도와 객체 점유 상태에 영향을 받으므로 거리 범위에 따라 균일한 시드 분포를 생성하며, 샘플링과 할당을 동시에 수행하여 효율적이다.

**B. Bubble Shrinking (BS)**
추출된 시드 포인트 $X \in \mathbb{R}^{M \times 3}$를 인스턴스 중심점으로 이동시키는 과정이다.
-   카테고리별 최소 반지름 $r_c$를 기준으로, 같은 카테고리에 속하며 거리가 $r_c$보다 작은 시드 포인트들을 연결하여 그래프 $G=(V, E)$를 구성한다.
-   각 시드 포인트를 중심(bubble)으로 하는 이웃 포인트들의 평균으로 시드 위치를 반복적으로(L=4회) 갱신한다.
-   이 과정에서 인접 행렬 $K$는 처음에 한 번만 계산되어 계산 오버헤드를 줄인다.

**C. Point Grouping**
이동된 시드 포인트 $X'$를 대상으로 Connected Component Labeling(CCL) 알고리즘을 적용한다. 시드 포인트 간의 거리 임계값(BS에서 사용한 $r_c$의 절반)과 시맨틱 카테고리를 기준으로 연결 성분을 찾으며, 같은 연결 성분에 속한 시드 포인트들이 지배하는 모든 포인트에 동일한 인스턴스 ID를 부여한다.

### 3. Instance Aggregation (IA)
SIP에서 발생한 대형 객체의 과분할 문제를 해결하기 위해 제안된 모듈이다.

**A. Global Feature Extraction**
각 인스턴스 제안 $i$에 대해 포인트 특징 $F_i$와 위치 $P_i$를 결합하고 Max Pooling을 적용하여 글로벌 인스턴스 특징 $g_i$를 생성한다.
$$g_i = \text{MaxPool}(\text{MLP}([F_i, P_i]))$$

**B. KNN-Transformer**
인스턴스 간의 상호작용을 모델링하기 위해 KNN-Transformer를 사용한다. 각 인스턴스 중심 $p_i$를 기준으로 $K$개의 최근접 이웃 $\{g_j\}_{j \in \mathcal{N}(i)}$를 찾고, Attention 메커니즘을 통해 강화된 특징 $\hat{g}_i$를 계산한다.
$$\hat{g}_i = \text{softmax}\left(\frac{q_i \circ k_i}{\sqrt{C}}\right) \circ v_i$$
여기서 $q_i, k_i, v_i$는 $g_i$와 $\{g_j\}$로부터 선형 변환을 통해 생성된 쿼리, 키, 밸류 벡터이다.

**C. Instance Affinity 및 병합**
두 인스턴스 제안 $i, j$ 사이의 유사도 $s_{i,j}$를 다음과 같이 계산한다.
$$s_{i,j} = \text{sigmoid}[\text{MLP}([\hat{g}_i, \hat{g}_j, |p_i - p_j|])]$$
이 값은 Binary Cross-Entropy loss $\mathcal{L}_{aff}$를 통해 학습되며, 추론 단계에서는 $s_{i,j}$가 특정 임계값을 넘으면 두 인스턴스를 하나로 병합(merge)한다. 병합 과정 또한 CCL 알고리즘을 통해 효율적으로 수행된다.

## 📊 Results

### 1. 실험 설정
-   **데이터셋**: SemanticKITTI 및 nuScenes 검증/테스트 셋.
-   **지표**: Panoptic Quality (PQ), Segmentation Quality (SQ), Recognition Quality (RQ) 및 mIoU.

### 2. 주요 결과
-   **SemanticKITTI**: PANet은 PQ 기준 61.7%를 기록하며 기존의 Clustering-based 방법들(DSNet, Panoptic-PolarNet, EfficientLPS)보다 크게 앞섰다. 특히 mIoU는 SCAN과 유사한 수준을 유지하면서 PQ는 4.5% 더 높게 나타났다.
-   **nuScenes**: 모든 지표에서 State-of-the-art(SOTA) 성능을 달성하였다 (PQ 69.2%).
-   **정성적 결과**: 시각화 결과, PANet은 밀집된 장면(crowded scenes)과 대형 객체(트럭, 버스) 분할에서 타 모델 대비 우수한 성능을 보였다.

### 3. Ablation Study
-   **구성 요소 영향**: BPS와 CCL 기반 그룹화만으로도 기본 성능이 확보되었으며, BS(Bubble Shrinking)를 추가했을 때 $\text{PQ}_{\text{Th}}$가 1.5% 상승했다. IA 모듈은 특히 트럭(Truck) 클래스에서 PQ를 3.1% 향상시켜 대형 객체 병합에 효과적임을 입증했다.
-   **Clustering 알고리즘 비교**: MeanShift, DBScan, HDBScan 등 전통적인 방식보다 PANet의 SIP 방식이 더 높은 PQ를 보였으며, 학습 기반의 DS 모듈보다 정확도와 속도(약 13배 빠름) 면에서 모두 우수했다.
-   **Sampling 알고리즘 비교**: BPS는 FPS와 대등한 정확도를 보이면서도 실행 시간은 FPS보다 3배, Random sampling보다 빠르게 나타났다.

## 🧠 Insights & Discussion

PANet의 가장 큰 강점은 인스턴스 생성 과정(SIP)에서 학습 가능한 파라미터를 완전히 제거했다는 점이다. 이는 모델의 복잡도를 낮출 뿐만 아니라, Offset branch 학습의 어려움을 원천적으로 해결하고 다양한 백본에 즉시 적용 가능한 범용성을 제공한다.

특히 BPS $\rightarrow$ BS $\rightarrow$ CCL로 이어지는 파이프라인은 계산 효율성이 극대화되어 실시간 응용 가능성을 높였다. 또한, 많은 LPS 모델들이 대형 객체의 과분할 문제를 간과하는 반면, PANet은 KNN-Transformer 기반의 IA 모듈을 통해 이를 명시적으로 해결하려 노력했다는 점에서 학술적 가치가 크다.

다만, 본 논문에서 제시된 $r_c$(최소 반지름)와 같은 하이퍼파라미터들이 경험적으로 설정되었다는 점은 한계로 볼 수 있다. 서로 다른 센서 환경이나 데이터셋에서 이러한 임계값들을 어떻게 최적으로 설정할 것인가에 대한 자동화된 방법론은 향후 연구 과제로 남아 있다.

## 📌 TL;DR

PANet은 학습 가능한 Offset branch 없이도 정밀한 LiDAR Panoptic Segmentation을 수행하는 프레임워크이다. 균일한 샘플링(BPS)과 반복적 중심 이동(BS)을 통해 효율적으로 인스턴스 제안을 생성하는 **SIP 모듈**과, 대형 객체의 과분할을 막기 위해 인스턴스 간 유사도를 학습하여 병합하는 **IA 모듈**이 핵심이다. 이 연구는 SemanticKITTI와 nuScenes에서 SOTA 성능을 달성하였으며, 특히 대형 객체 분할 성능과 추론 속도를 획기적으로 개선하여 자율주행 시스템의 실시간 3D 장면 이해에 크게 기여할 것으로 기대된다.