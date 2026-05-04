# ClickSeg: 3D Instance Segmentation with Click-Level Weak Annotations

Leyao Liu, Tao Kong, Minzhao Zhu, Jiashuo Fan, and Lu Fang (2023)

## 🧩 Problem to Solve

3D 인스턴스 세그멘테이션(Instance Segmentation) 모델을 학습시키기 위해서는 일반적으로 모든 포인트에 대해 정밀한 라벨이 지정된 Dense labels가 필요하다. 그러나 이러한 데이터셋을 구축하는 것은 막대한 인적 비용과 시간을 소모한다. 예를 들어, ScanNet 데이터셋의 경우 한 장면당 평균 22.3분이 소요되며, 포인트 수가 수백만 개에 달하는 데이터셋에서는 이 문제가 더욱 심각해진다.

본 논문이 해결하고자 하는 핵심 문제는 **극도로 제한된 약한 지도(Weakly Supervised) 환경, 즉 인스턴스당 단 하나의 포인트(One point per instance)만 라벨링된 상태에서 어떻게 고품질의 3D 인스턴스 세그멘테이션을 수행할 것인가**이다. 인스턴스 세그멘테이션은 단순한 시맨틱 세그멘테이션과 달리 개별 인스턴스를 분리해야 하므로, 매우 제한된 라벨만으로는 달성하기 매우 어려운 도전적인 과제이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 인스턴스당 하나의 클릭(Click) 정보만을 활용하여 온라인으로 정밀한 Pseudo labels(가상 라벨)를 생성하고, 이를 통해 모델을 점진적으로 학습시키는 프레임워크를 제안하는 것이다.

1.  **Click-level Pseudo Label Generation**: 기존의 Mean-shift clustering 방식 대신, 어노테이션된 포인트들을 초기 시드(Initial seeds)로 설정한 k-means clustering을 사용하여 Pseudo labels를 생성한다. 이는 어노테이션된 포인트가 반드시 해당 인스턴스의 중심에 있지 않더라도, k-means의 반복적인 업데이트를 통해 중심을 찾아감으로써 더 정확한 라벨을 생성할 수 있게 한다.
2.  **Enhanced Similarity Metrics**: 단순히 임베딩 거리만 사용하는 것이 아니라, 시맨틱 유사도(Semantic similarity)와 공간적 거리(Spatial distance)를 결합한 새로운 유사도 지표를 설계하여 Pseudo label의 정확도를 높였다.
3.  **Online Training Framework**: 학습 과정 중에 Pseudo labels를 실시간으로 재생성하여 모델이 점진적으로 더 정확한 라벨로 학습하도록 유도하며, 이는 학습 시간을 크게 늘리지 않으면서도 성능을 극대화한다.

## 📎 Related Works

### 관련 연구 및 한계
- **3D Segmentation**: 포인트 기반(Point-based) 방식과 복셀 기반(Voxel-based) 방식으로 나뉘며, 최근에는 계산 효율성을 위해 SparseConv 등이 널리 사용된다.
- **Instance Segmentation**: 주로 Top-down 방식의 Proposal-based와 Bottom-up 방식의 Grouping-based로 나뉜다. 본 논문은 Grouping-based 방식을 따른다.
- **Weakly Supervised Segmentation**: 이미지 분야에서는 Click-level, Scribble-level 등의 연구가 활발하며, 3D 분야에서도 시맨틱 세그멘테이션에 대한 약한 지도 학습 연구(MPRM, SQN, OTOC 등)가 진행되었다.
- **Existing Weakly Supervised Instance Segmentation**: 최근 제안된 SegGroup은 슈퍼복셀(Supervoxel) 기반의 어노테이션을 요구한다. 하지만 이는 작업자가 각 인스턴스의 가장 중심이 되는 슈퍼복셀을 찾아내야 하므로 인간이 수행하기에 어렵고, 한 번 복원된 학습셋의 오류가 고정된다는 한계가 있다.

### 차별점
ClickSeg는 작업자가 인스턴스 내의 **임의의 포인트 하나만** 선택하면 되므로 어노테이션 비용을 획기적으로 낮추었으며, 고정된 라벨이 아닌 학습 과정 중 업데이트되는 Pseudo labels를 사용하여 기존 방식보다 높은 정확도를 달성한다.

## 🛠️ Methodology

### 전체 파이프라인 및 구조
ClickSeg는 3D UNet을 백본으로 하며, Submanifold Sparse Convolution을 사용하여 효율성을 높였다. 네트워크는 각 포인트 $i$에 대해 세 가지 출력값을 생성한다:
- **Embedding vector** $e_i$: 인스턴스 구분을 위한 특징 벡터
- **Semantic probability** $s_i$: 클래스 분류를 위한 확률값
- **Offset vector** $o_i$: 해당 포인트가 속한 인스턴스의 중심으로 향하는 벡터

### 학습 목표 및 손실 함수
전체 손실 함수는 시맨틱 교차 엔트로피 손실($L_{CE}$), 오프셋 회귀 손실($L_{regress}$), 그리고 임베딩 학습을 위한 판별 손실(Discriminative loss)의 합으로 구성된다.

임베딩 학습을 위해 사용되는 판별 손실은 다음과 같다:
- **Variance loss**: 동일 인스턴스 내 포인트들의 임베딩 거리를 좁힌다.
$$L_{var} = \frac{1}{C} \sum_{c=1}^{C} \frac{1}{N_c} \sum_{i=1}^{N_c} [ ||\mu_c - e_i|| - \delta_v ]_+^2$$
- **Distance loss**: 서로 다른 인스턴스 간의 임베딩 거리를 멀게 유지한다.
$$L_{dist} = \frac{1}{C(C-1)} \sum_{c_A=1}^{C} \sum_{c_B=1, c_B \neq c_A}^{C} [ 2\delta_d - ||\mu_{c_A} - \mu_{c_B}|| ]_+^2$$
여기서 $C$는 인스턴스 수, $\mu_c$는 인스턴스 $c$의 평균 임베딩, $\delta_v$와 $\delta_d$는 각각 임계값이다.

### Pseudo Label Generation (Click Annotation Version)
본 논문의 핵심인 Pseudo label 생성 과정은 다음과 같다:
1.  **초기 확장**: 낮은 수준의 특징을 이용해 슈퍼복셀 파티션을 수행하고, 클릭된 포인트의 라벨을 해당 슈퍼복셀 전체로 확장하여 초기 학습을 진행한다.
2.  **K-means Clustering**: 어노테이션된 포인트들을 초기 시드로 설정하여 k-means clustering을 수행한다. 각 클러스터는 반드시 하나의 어노테이션된 포인트를 포함해야 하며, 이를 통해 모든 unlabeled 포인트가 가장 적절한 인스턴스에 할당된다.
3.  **유사도 지표($S_{ij}$)**: k-means 및 추론 시 사용되는 유사도는 다음과 같이 정의된다:
$$S_{ij} = Q_{ij} * \exp \left( -\left( \frac{D_{e_{ij}}}{\sigma_e} \right)^2 - \left( \frac{D_{p_{ij}}}{\sigma_p} \right)^2 \right)$$
    - $D_{e_{ij}} = ||e_i - e_j||$: 임베딩 거리
    - $D_{p_{ij}}$: 공간적 거리. 초기에는 $||p_i - p_j||$를 사용하다가, 학습 후 오프셋이 학습되면 $||(p_i + o_i) - (p_j + o_j)||$를 사용한다.
    - $Q_{ij} = \frac{s_i \cdot s_j}{||s_i|| ||s_j||}$: 시맨틱 유사도. 서로 다른 부위라도 같은 시맨틱 클래스라면 동일 인스턴스일 가능성이 높다는 직관을 반영한다.

### 추론 절차 (Inference)
- **Instance Segmentation**: 학습된 유사도 지표 $S_{ij}$를 기반으로 Mean-shift clustering을 수행하여 인스턴스를 그룹화한다.
- **Semantic Segmentation**: 유사한 포인트들은 같은 라벨을 가져야 한다는 점을 이용해, 주변 포인트들의 시맨틱 확률을 융합(Fusion)하는 단계를 거친다:
$$\hat{s}_i = \frac{\sum_{j=1}^{N} I_{ij} * S_{ij} * s_j}{\sum_{j=1}^{N} I_{ij} * S_{ij}}$$
여기서 $I_{ij}$는 유사도 $S_{ij}$가 임계값 $\gamma$보다 클 때만 1이 되는 지시함수이다.

## 📊 Results

### 실험 설정
- **데이터셋**: ScanNetV2 및 S3DIS (Area-5)
- **지표**: 인스턴스 세그멘테이션은 mAP@50, 시맨틱 세그멘테이션은 mIoU를 사용한다.
- **어노테이션 설정**: 인스턴스당 단 하나의 임의의 포인트만 라벨링한다 (ScanNetV2 기준 전체 데이터의 약 0.02% 수준).

### 주요 결과
1.  **인스턴스 세그멘테이션 (ScanNetV2)**:
    - ClickSeg는 약 0.02%의 매우 적은 라벨만으로 Fully-supervised baseline 성능의 약 90%에 해당하는 **53.9% mAP@50 (Test split)**를 달성하였다.
    - 기존의 약한 지도 학습 방법인 SegGroup(44.5% mAP)보다 훨씬 높은 성능을 보였다.
2.  **시맨틱 세그멘테이션 (ScanNetV2)**:
    - 동일한 클릭 설정 하에서 OTOC(69.1% mIoU)보다 우수한 **70.3% mIoU (Test split)**를 기록하였다.
3.  **S3DIS 결과**:
    - 0.004%의 매우 희소한 라벨을 사용했음에도 인스턴스 mAP@50 45.7%를 기록하며, Fully-supervised baseline(53.4%)에 근접하는 결과를 보였다.

### Ablation Study
- **K-means 효과**: Nearest Neighbor search보다 K-means를 사용했을 때 mAP@50가 1.4% 상승하여, 시드 포인트가 중심이 아닐 때의 문제를 완화함을 입증하였다.
- **유사도 지표 영향**: 임베딩 거리 $\rightarrow$ 공간 거리 추가 $\rightarrow$ 시맨틱 유사도 추가 순으로 성능이 단계적으로 향상되었으며, 최종적으로 모든 요소를 결합했을 때 51.9% mAP로 가장 높았다.

## 🧠 Insights & Discussion

### 강점 및 해석
본 논문은 3D 인스턴스 세그멘테이션에서 **"클릭 한 번"**이라는 극도로 적은 정보만으로도 유의미한 성능을 낼 수 있음을 보여주었다. 특히 시맨틱 세그멘테이션과 인스턴스 세그멘테이션을 멀티태스크로 학습시키면서, 시맨틱 정보가 인스턴스의 경계를 잡는 데 도움을 주고, 반대로 인스턴스 정보가 시맨틱 예측을 부드럽게(Smoothing) 만드는 상호보완적 관계를 잘 활용하였다.

### 한계 및 비판적 논의
- **미라벨링 인스턴스 문제**: 논문에서 언급했듯이, 모든 인스턴스에 대해 최소 하나 이상의 포인트가 라벨링되었다는 가정이 필요하다. 만약 일부 인스턴스가 라벨링되지 않았다면, k-means 과정에서 다른 인스턴스에 흡수되는 성능 저하가 발생할 수 있다.
- **의존성**: Pseudo labels의 품질이 초기 모델의 예측 성능에 의존하므로, 초기 학습 단계에서 잘못된 라벨이 생성될 경우 이를 완전히 극복하는 메커니즘에 대한 더 상세한 분석이 필요하다.

## 📌 TL;DR

ClickSeg는 3D 인스턴스 세그멘테이션을 위해 **인스턴스당 단 하나의 포인트만 라벨링**하는 극단적인 약한 지도 학습 프레임워크이다. 어노테이션된 포인트를 초기 시드로 하는 **k-means clustering**과 **시맨틱-공간-임베딩이 결합된 새로운 유사도 지표**를 통해 온라인으로 Pseudo labels를 생성하여 학습하며, 이를 통해 매우 적은 비용으로 Fully-supervised 모델의 약 90% 성능을 달성하였다. 이 연구는 향후 3D 데이터 어노테이션 비용을 획기적으로 줄이는 실용적인 방향성을 제시한다.