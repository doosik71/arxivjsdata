# PointGroup: Dual-Set Point Grouping for 3D Instance Segmentation

Li Jiang, Hengshuang Zhao, Shaoshuai Shi, Shu Liu, Chi-Wing Fu, Jiaya Jia (2020)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 3D 포인트 클라우드(point cloud) 데이터에서의 **Instance Segmentation**이다. Instance Segmentation은 단순히 각 포인트의 semantic label을 예측하는 것을 넘어, 동일한 클래스에 속하는 개별 객체들을 서로 다른 ID로 구분해내는 작업이다.

3D 포인트 클라우드는 2D 이미지와 달리 데이터가 정렬되지 않고 구조화되어 있지 않아 기존의 2D 방식들을 직접 적용하기 어렵다. 특히, 동일한 카테고리에 속하는 두 객체가 3D 공간상에서 매우 가깝게 배치되어 있을 경우, 이를 하나의 객체로 오인하여 그룹화하는 문제가 빈번하게 발생한다. 따라서 본 논문의 목표는 객체 사이의 빈 공간(void space)과 semantic 정보를 효율적으로 활용하여 포인트 그룹화(grouping) 성능을 향상시키는 엔드투엔드(end-to-end) 아키텍처를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Dual-Set Point Grouping**으로, 포인트의 원래 좌표계와 학습된 오프셋(offset)을 통해 이동시킨 좌표계라는 두 가지 보완적인 좌표 세트를 모두 사용하여 클러스터링을 수행하는 것이다.

1.  **Dual-Set Point Grouping**: 포인트들을 각 객체의 중심(centroid) 방향으로 이동시킨 shifted coordinate set을 생성하고, 이를 original coordinate set과 함께 활용하여 인접한 동일 클래스 객체들을 더 정교하게 분리한다.
2.  **ScoreNet**: 생성된 수많은 후보 클러스터들 중 어떤 것이 실제 객체일 가능성이 높은지 평가하는 별도의 네트워크를 설계하여, NMS(Non-Maximum Suppression) 단계에서 최적의 인스턴스를 선택할 수 있게 한다.
3.  **Bottom-up Framework**: semantic segmentation과 offset prediction을 동시에 수행하고, 이후 클러스터링과 score 예측으로 이어지는 일관된 bottom-up 구조를 제안하여 SOTA(State-of-the-art) 성능을 달성하였다.

## 📎 Related Works

3D Instance Segmentation 연구는 크게 두 가지 방향으로 나뉜다.

1.  **Detection-based (Top-down)**: 3D bounding box를 먼저 예측하고 그 내부에서 마스크를 학습하는 방식이다. 3D-BoNet, GSPN 등이 이에 해당하며, 박스 예측의 정확도에 의존한다는 특징이 있다.
2.  **Segmentation-based (Bottom-up)**: 포인트별 semantic label을 먼저 예측한 후, 포인트 간의 유사도나 임베딩을 이용해 그룹화하는 방식이다. SGPN, ASIS 등이 대표적이다.

기존의 bottom-up 방식들은 주로 포인트 임베딩(embedding)에 의존하여 클러스터링을 수행했으나, 이는 인접한 객체들을 분리하는 데 한계가 있었다. PointGroup은 임베딩 대신 **좌표 이동(offset shifting)**과 **Dual-set clustering**이라는 기하학적 접근 방식을 도입하여 기존 방식들과 차별화를 꾀하였다.

## 🛠️ Methodology

### 1. 전체 파이프라인
전체 시스템은 **Backbone Network $\rightarrow$ Clustering Part $\rightarrow$ ScoreNet**의 세 단계로 구성된다.

### 2. Backbone Network 및 브랜치
입력 데이터인 포인트 세트 $P$에 대해 U-Net 기반의 backbone(Submanifold Sparse Convolution 및 Sparse Convolution 사용)을 통해 특징 $F$를 추출한다. 이후 두 개의 브랜치로 나뉜다.

-   **Semantic Segmentation Branch**: MLP를 통해 각 포인트의 semantic score를 예측하며, Cross Entropy Loss($L_{sem}$)를 통해 학습한다.
-   **Offset Prediction Branch**: 각 포인트를 해당 인스턴스의 중심 $\hat{c}_i$로 이동시키기 위한 오프셋 벡터 $o_i$를 예측한다. 학습을 위해 다음 두 가지 손실 함수를 사용한다.
    -   **Regression Loss**: 예측된 오프셋과 실제 중심-포인트 간 거리의 차이를 $L_1$ loss로 계산한다.
    $$L_{oreg} = \frac{1}{\sum m_i} \sum_{i} ||o_i - (\hat{c}_i - p_i)|| \cdot m_i$$
    -   **Direction Loss**: 오프셋의 크기보다 방향성을 강조하여 포인트들이 중심 방향으로 정확히 이동하게 유도한다. 이는 마이너스 코사인 유사도로 정의된다.
    $$L_{odir} = -\frac{1}{\sum m_i} \sum_{i} \frac{o_i}{||o_i||_2} \cdot \frac{\hat{c}_i - p_i}{||\hat{c}_i - p_i||_2} \cdot m_i$$

### 3. Clustering Algorithm (Dual-Set Point Grouping)
예측된 semantic label과 오프셋을 바탕으로 포인트들을 그룹화한다.
-   **Original Set ($P$)**: 원래 좌표 $\mathbf{p}_i$를 기준으로 반경 $r$ 이내의 동일 라벨 포인트들을 BFS(Breadth-First Search) 방식으로 그룹화한다.
-   **Shifted Set ($Q$)**: 예측된 오프셋을 더한 좌표 $\mathbf{q}_i = \mathbf{p}_i + \mathbf{o}_i$를 기준으로 그룹화한다. 동일 객체의 포인트들은 $\mathbf{q}_i$ 공간에서 중심점 주변으로 밀집되므로, 인접한 서로 다른 객체들을 분리하는 데 유리하다.
-   최종 후보 클러스터 세트 $C$는 $C_p$와 $C_q$의 합집합($C = C_p \cup C_q$)으로 구성된다.

### 4. ScoreNet
후보 클러스터 $C_i$의 품질을 평가하기 위해 ScoreNet을 사용한다.
-   **구조**: 클러스터의 포인트 특징과 좌표를 voxelize하여 소형 U-Net에 입력하고, Max-pooling과 MLP를 거쳐 최종 스코어 $s_{c}$를 산출한다.
-   **학습**: 정답 인스턴스와의 IoU(Intersection over Union)를 기반으로 soft label $\hat{s}_{c}$를 생성하여 Binary Cross Entropy Loss($L_{cscore}$)로 학습한다.
    $$\hat{s}_{c}^i = \begin{cases} 0 & \text{iou}_i < \theta_l \\ 1 & \text{iou}_i > \theta_h \\ \frac{\text{iou}_i - \theta_l}{\theta_h - \theta_l} & \text{otherwise} \end{cases}$$

### 5. 학습 및 추론
-   **학습**: $L = L_{sem} + L_{odir} + L_{oreg} + L_{cscore}$를 통합하여 엔드투엔드로 학습한다.
-   **추론**: ScoreNet의 점수를 기준으로 NMS를 수행하여 최종 인스턴스를 결정한다.

## 📊 Results

### 1. 실험 설정
-   **데이터셋**: ScanNet v2, S3DIS
-   **지표**: mAP (IoU threshold 0.5 기준 $AP_{50}$), mPrec, mRec
-   **주요 파라미터**: Voxel size $0.02m$, Clustering radius $r = 0.03m$, 최소 포인트 수 $N_\theta = 50$

### 2. 정량적 결과
-   **ScanNet v2**: $AP_{50}$ 기준 **63.6%**를 달성하여, 이전 최고 성능(54.9%) 대비 절대치 기준 **8.7%p 향상**되었다.
-   **S3DIS**: $AP_{50}$ 기준 **64.0%**, $\text{mPrec}_{50}$ **69.6%**, $\text{mRec}_{50}$ **69.2%**를 기록하며 타 모델들을 큰 격차로 앞섰다.

### 3. Ablation Study
-   **Coordinate Set 영향**:
    -   $P$만 사용 시: 인접한 동일 클래스 객체들을 하나로 묶는 오류(mis-grouping)가 잦다.
    -   $Q$만 사용 시: 객체 경계 부분의 오프셋 예측 부정확성으로 인해 경계 영역 처리가 미흡하다.
    -   $P \& Q$ 모두 사용 시: 두 세트의 상호보완적 특성 덕분에 최적의 성능을 보인다.
-   **Radius $r$ 영향**: $r=0.03m$일 때 가장 높은 성능을 보였으며, 너무 작으면 포인트 밀도가 낮은 곳에서 그룹 성장이 안 되고, 너무 크면 인접 객체가 합쳐지는 문제가 발생한다.
-   **ScoreNet의 필요성**: 단순 semantic probability 평균값을 사용하는 것보다 ScoreNet을 통한 스코어링이 $AP_{50}$ 기준 약 5%p 더 높은 성능을 보였다.

## 🧠 Insights & Discussion

본 논문은 3D 공간의 기하학적 특성, 즉 객체들이 보통 '빈 공간'에 의해 분리되어 있다는 점에 착안하여 매우 실용적인 해결책을 제시하였다. 특히 포인트들을 중심점으로 모으는 **Offset Prediction**과 이를 활용한 **Dual-Set Clustering**은 복잡한 임베딩 공간을 설계하는 대신 직관적인 좌표 변환을 통해 인스턴스 분리 문제를 해결했다는 점에서 강점이 있다.

다만, 저자들도 언급했듯이 클러스터링의 전제 조건이 semantic label의 정확도에 의존한다는 한계가 있다. 만약 semantic segmentation 단계에서 오류가 발생하면, 이후의 grouping 단계에서 이를 복구하기 어렵다. 향후 연구에서 언급된 'progressive refinement module' 같은 기법이 도입된다면 이러한 semantic 부정확성 문제를 완화할 수 있을 것으로 보인다.

## 📌 TL;DR

PointGroup은 3D 인스턴스 분할을 위해 **원래 좌표($P$)와 중심점 방향으로 이동시킨 좌표($Q$)를 모두 사용하여 클러스터링**하는 bottom-up 프레임워크이다. 이를 통해 인접한 동일 클래스 객체들을 정교하게 분리하며, ScoreNet을 통해 최적의 후보를 선택함으로써 ScanNet v2와 S3DIS 데이터셋에서 SOTA 성능을 달성하였다. 이 연구는 3D 씬 이해 및 로봇 내비게이션 등 정밀한 객체 구분이 필요한 분야에 크게 기여할 가능성이 높다.