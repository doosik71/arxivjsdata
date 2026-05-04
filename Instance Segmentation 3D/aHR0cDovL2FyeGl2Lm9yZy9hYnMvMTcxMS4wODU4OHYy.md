# SGPN: Similarity Group Proposal Network for 3D Point Cloud Instance Segmentation

Weiyue Wang, Ronald Yu, Qiangui Huang, Ulrich Neumann (2019)

## 🧩 Problem to Solve

본 논문은 3D 포인트 클라우드(Point Cloud)에서의 **인스턴스 분할(Instance Segmentation)** 문제를 해결하고자 한다. 2D 이미지 분야에서는 인스턴스 분할 연구가 비약적으로 발전하였으나, 3D 분야는 이에 비해 크게 뒤처져 있는 상황이다. 기존의 3D 볼륨 데이터(Volumetric data) 기반 CNN 방식은 메모리 소모와 계산 비용이 매우 높다는 한계가 있다.

따라서 본 연구의 목표는 포인트 클라우드라는 효율적인 표현 방식을 유지하면서도, 포인트 간의 관계를 학습하여 개별 객체 인스턴스를 정확하게 분리해낼 수 있는 단순하고 직관적인 딥러닝 프레임워크인 **SGPN(Similarity Group Proposal Network)**을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 포인트 클라우드의 인스턴스 분할 결과를 **유사도 행렬(Similarity Matrix)** 형태로 표현하는 것이다.

1.  **유사도 기반의 인스턴스 표현**: 임베딩된 특징 공간(Embedded feature space)에서 두 포인트 간의 거리를 계산하여 유사도를 측정하고, 이를 통해 포인트 그룹 제안(Group Proposal)을 생성한다.
2.  **다중 출력 브랜치 설계**: 하나의 네트워크에서 유사도 행렬(Similarity Matrix), 신뢰도 맵(Confidence Map), 시맨틱 분할 맵(Semantic Segmentation Map)의 세 가지 출력을 동시에 예측하는 구조를 제안한다.
3.  **Double-Hinge Loss 도입**: 포인트 쌍의 관계를 세 가지 유사도 클래스로 정의하고, 이에 따라 특징 공간에서의 거리를 최적화하는 손실 함수를 설계하여 학습의 효율성과 정확도를 높였다.
4.  **유연한 구조**: 2D CNN 특징을 심리스하게 통합하여 성능을 향상시킬 수 있는 확장성을 보여주었다.

## 📎 Related Works

### 1. 객체 탐지 및 인스턴스 분할 (Object Detection and Instance Segmentation)
2D 분야에서는 R-CNN, Mask R-CNN 등 영역 제안(Region Proposal) 기반의 방식이 주류를 이루고 있다. 3D 분야에서도 볼륨 기반의 RPN이나 RGB-D 이미지 기반의 경계 상자(Bounding Box) 회귀 방식이 제안되었으나, 포인트 클라우드 자체에서 직접적으로 인스턴스 분할을 학습하는 연구는 부족한 상태였다.

### 2. 3D 딥러닝 (3D Deep Learning)
기존의 Voxel 기반 CNN은 메모리 효율성이 낮고, Octree 기반 CNN은 유연성이 부족하다는 단점이 있다. 본 논문은 포인트 세트를 직접 처리하는 PointNet 및 PointNet++의 아키텍처를 기반으로 하여 이러한 효율성 문제를 해결하였다.

### 3. 유사도 메트릭 학습 (Similarity Metric Learning)
Siamese CNN이나 Associative Embedding과 같이 특징 공간에서 유사한 객체들을 가깝게 배치하는 메트릭 학습 기법들이 존재한다. SGPN은 이러한 개념을 차용하되, 단순히 그룹을 묶는 것을 넘어 유사도 행렬을 통해 가변적인 수의 인스턴스 제안을 생성하도록 설계하여 차별점을 두었다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
SGPN은 입력 포인트 클라우드 $P$를 PointNet/PointNet++ 기반의 특징 추출 네트워크에 통과시켜 전역 및 지역 특징 행렬 $F$를 생성한다. 이후 네트워크는 세 개의 독립적인 브랜치로 나뉜다.

-   **Similarity Matrix Branch**: 포인트 쌍 간의 유사도를 계산하여 그룹 제안을 생성한다.
-   **Confidence Map Branch**: 각 그룹 제안이 실제 객체 인스턴스일 확률(신뢰도)을 예측한다.
-   **Semantic Segmentation Branch**: 각 포인트의 시맨틱 클래스(범주)를 예측한다.

### 2. 주요 구성 요소 및 수식 설명

#### 유사도 행렬 (Similarity Matrix)
포인트 $P_i$와 $P_j$가 동일한 인스턴스에 속한다면 특징 공간에서 매우 가깝게 위치해야 한다는 직관에 기반한다. 유사도 $S_{ij}$는 두 포인트의 특징 벡터 $F^{SIM}_i$와 $F^{SIM}_j$ 사이의 $L_2$ Norm으로 정의된다.
$$S_{ij} = \|F^{SIM}_i - F^{SIM}_j\|_2$$
이 행렬의 각 행은 하나의 인스턴스 후보(Group Proposal)로 간주된다.

#### Double-Hinge Loss ($\mathcal{L}^{SIM}$)
포인트 쌍 $\{P_i, P_j\}$의 관계를 세 가지 클래스로 정의하고, 클래스 번호가 커질수록 특징 공간에서의 거리가 멀어지도록 강제한다.
-   클래스 1: 동일 인스턴스 소속
-   클래스 2: 시맨틱 클래스는 같으나 서로 다른 인스턴스 소속
-   클래스 3: 시맨틱 클래스 자체가 다름

손실 함수 $\ell(i, j)$는 다음과 같다.
$$\ell(i,j) = \begin{cases} \|F^{SIM}_i - F^{SIM}_j\|_2 & C_{ij} = 1 \\ \alpha \max(0, K_1 - \|F^{SIM}_i - F^{SIM}_j\|_2) & C_{ij} = 2 \\ \max(0, K_2 - \|F^{SIM}_i - F^{SIM}_j\|_2) & C_{ij} = 3 \end{cases}$$
여기서 $\alpha > 1$, $K_2 > K_1$이며, $\alpha$는 시맨틱 분할 브랜치와의 간섭을 줄이고 서로 다른 인스턴스를 더 강력하게 밀어내기 위한 가중치이다.

#### 신뢰도 맵 (Confidence Map) 및 시맨틱 분할 (Semantic Segmentation)
-   **Confidence Map**: 예측된 그룹 $S_i$와 실제 정답 그룹 $G_i$ 사이의 $\text{IoU}$ 값을 정답으로 하여 $L_2$ Loss를 통해 학습한다.
-   **Semantic Segmentation**: 각 포인트별로 클래스 확률을 예측하며, 클래스 불균형을 해소하기 위해 Median Frequency Balancing이 적용된 가중 교차 엔트로피(Weighted Cross Entropy) 손실 함수를 사용한다.

### 3. 추론 및 그룹 병합 절차 (Group Proposal Merging)
테스트 단계에서는 다음과 같은 절차를 통해 최종 인스턴스를 결정한다.
1.  **Pruning**: 예측된 신뢰도가 $Th_C$보다 낮거나, 그룹 내 포인트 수가 $Th_{M2}$보다 적은 제안을 제거한다.
2.  **Merging**: $\text{IoU}$가 $Th_{M1}$보다 큰 그룹들 중 포인트 수가 가장 많은 그룹을 선택하는 Non-Maximum Suppression(NMS) 방식을 적용하여 중복 제안을 제거한다.
3.  **Assignment**: 각 포인트를 자신이 속한 최종 그룹에 할당한다.

## 📊 Results

### 1. 실험 설정
-   **데이터셋**: S3DIS (실내 장면), NYUV2 (부분 스캔), ShapeNet (객체 파트 분할).
-   **비교 대상**: Seg-Cluster (시맨틱 분할 후 BFS 기반 클러스터링), PointNet 기반 3D 탐지 모델.
-   **평가 지표**: Average Precision (AP), Mean IoU.

### 2. 주요 결과
-   **S3DIS**: SGPN은 Seg-Cluster 대비 월등한 AP 성능을 보였다. 특히 동일한 시맨틱 레이블을 가졌으나 서로 인접해 있는 작은 객체들을 분리하는 능력이 탁월했다. 3D 객체 탐지 작업에서도 기존 PointNet 방식보다 높은 AP를 기록하였다.
-   **NYUV2**: partial scan 데이터에서도 효과적임을 입증했다. 특히 2D CNN 특징을 결합한 **SGPN-CNN** 모델은 단독 모델보다 높은 성능을 보였으며, 이는 기하학적 관계와 외형 특징을 동시에 활용했기 때문이다.
-   **ShapeNet**: 파트 분할 실험에서 PointNet++보다 높은 mIoU를 기록하였으며, 특히 의자 다리와 같이 구분하기 어려운 영역에서도 정교한 인스턴스 분리가 가능함을 확인하였다.

### 3. 효율성 분석
SGPN은 매우 빠른 추론 속도를 보인다. NYUV2 데이터셋 기준으로 SGPN은 약 170ms, SGPN-CNN은 300ms가 소요되며, 이는 기존의 RPN 기반 3D 탐지 모델들이 수 초(seconds) 단위의 시간을 소모하는 것에 비해 매우 효율적이다.

## 🧠 Insights & Discussion

### 강점
본 연구는 포인트 클라우드를 '그리드' 중심이 아닌 '형태(Shape)' 중심으로 해석하여, 포인트 간의 관계(유사도)를 통해 인스턴스를 정의했다는 점에서 매우 자연스러운 접근 방식을 취했다. 또한, 유사도, 신뢰도, 시맨틱 정보를 동시에 학습함으로써 각 브랜치가 서로를 보완하는 시너지 효과를 거두었다.

### 한계 및 비판적 해석
가장 큰 한계는 유사도 행렬의 크기가 포인트 수 $N_p$에 대해 제곱($O(N_p^2)$)으로 증가한다는 점이다. 이로 인해 수십만 개 이상의 포인트를 가진 매우 큰 장면을 한 번에 처리하는 것이 불가능하다. 논문에서는 이를 해결하기 위해 시드(Seed) 포인트를 활용하는 방안을 제시하였으나, 구체적인 구현 결과는 포함되지 않았다.

또한, NYUV2 실험에서 Tight Bounding Box를 사용함으로써 Amodal detection(보이지 않는 부분까지 포함하는 탐지)을 수행하는 기존 연구보다 IoU가 낮게 측정되는 경향이 있으나, 이는 실제 관측된 포인트 기반의 분할이라는 점에서 타당한 결과로 해석된다.

## 📌 TL;DR

SGPN은 3D 포인트 클라우드에서 포인트 간의 유사도를 학습하는 **유사도 행렬(Similarity Matrix)** 표현 방식을 도입하여 인스턴스 분할을 수행하는 프레임워크이다. 유사도, 신뢰도, 시맨틱 예측을 동시에 수행하는 구조와 Double-Hinge Loss를 통해 인접한 동일 클래스 객체들을 효과적으로 분리해냈으며, 기존 방식 대비 매우 빠른 추론 속도와 높은 정확도를 달성했다. 이 연구는 3D 장면 이해에서 포인트 간의 관계적 특징 학습이 중요함을 시사하며, 향후 대규모 포인트 클라우드 처리 최적화 연구의 기반이 될 가능성이 높다.