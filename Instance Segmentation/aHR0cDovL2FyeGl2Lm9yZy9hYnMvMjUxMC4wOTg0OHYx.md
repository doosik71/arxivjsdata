# Cell Instance Segmentation: The Devil Is in the Boundaries

Peixian Liang, Yifan Ding, Yizhe Zhang, Jianxu Chen, Hao Zheng, Hongxiao Wang, Yejia Zhang, Guangyu Meng, Tim Weninger, Michael Niemier, X. Sharon Hu, Danny Z Chen (2020)

## 🧩 Problem to Solve

본 논문은 세포 인스턴스 분할(Cell Instance Segmentation)에서 개별 세포를 정확하게 구분해내는 문제를 다룬다. 현재의 최신(SOTA) 방법론들은 주로 딥러닝 기반의 시맨틱 분할(Semantic Segmentation)을 통해 전경(foreground) 픽셀과 배경(background) 픽셀을 먼저 구분한 뒤, 전경 픽셀들을 클러스터링하여 개별 인스턴스를 식별하는 방식을 취한다.

이 과정에서 대부분의 방법론은 거리 맵(distance maps), 열 확산 맵(heat diffusion maps), 또는 고정된 각도의 별 모양 다각형(star-shaped polygons)과 같은 픽셀 단위의 목적 함수(pixel-wise objectives)를 사용한다. 그러나 이러한 픽셀 단위 접근 방식은 세포 인스턴스가 가진 형태(shape), 곡률(curvature), 볼록성(convexity)과 같은 중요한 기하학적 특성을 충분히 반영하지 못한다는 한계가 있다. 따라서 본 논문의 목표는 픽셀 단위가 아닌 경계 단위의 특성을 활용하여 이러한 기하학적 정보를 보존하고, 더 정확하게 세포 인스턴스를 분할하는 새로운 클러스터링 방법인 Ceb(Cell boundaries)를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 전경 픽셀을 인스턴스로 나누는 문제를 '경계 분류 문제'로 전환하는 것이다. 주요 기여 사항은 다음과 같다.

1.  **Ceb 프레임워크 제안**: 픽셀 단위의 목적 함수 대신, 경계 수준의 특징(boundary-level features)과 라벨을 활용하는 딥러닝 기반 경계 분류기를 통해 전경 픽셀을 클러스터링한다.
2.  **Boundary Signature 개발**: 잠재적 전경-전경 경계와 인접한 배경-전경 경계에서 픽셀을 샘플링하여 경계의 기하학적 특성을 반영하는 새로운 특징 표현 방식인 'Boundary Signature'를 제안하였다.
3.  **알고리즘 개선**: 모든 잠재적 전경-전경 경계를 생성하기 위해 Watershed 알고리즘을 수정하였으며, 생성된 경계에 라벨을 부여하기 위한 최적 인스턴스 매칭(optimal instance matching) 방법을 제시하였다.
4.  **Temporal Instance Consistency 도입**: 2D 비디오 데이터셋의 경우, 프레임 간의 일관성을 활용하여 분할 성능을 더욱 향상시키는 매칭 및 선택 방법을 제안하였다.

## 📎 Related Works

### 1. 세포 인스턴스 분할 (Cell Instance Segmentation)
기존 연구는 크게 두 가지로 나뉜다.
- **시맨틱 분할 기반 접근법**: 전경-배경을 먼저 구분하고 이후 클러스터링하는 방식으로, Hover-Net(거리 맵), CellPose(열 확산 맵), StarDist(방사형 맵) 등이 대표적이다. 하지만 이들은 픽셀 단위 최적화를 수행하므로 구조적 기하학적 특성을 잃기 쉽다.
- **영역 기반 접근법 (Region-based)**: Mask R-CNN과 같이 그리드나 앵커를 사용하여 영역별 분류를 수행한다. 그러나 세포가 밀집된 환경에서는 바운딩 박스가 겹치거나 불균형 문제가 발생하여 부적합할 수 있다.

### 2. 경계 생성 방법 (Boundary Generation Methods)
Watershed나 Active Contours와 같은 전통적 방법이 있으며, 최근에는 이를 딥러닝과 결합하는 시도가 있었다. Ceb는 단순한 픽셀 수준 특징을 넘어, 딥러닝 분류기를 통해 경계 라벨을 결정함으로써 시맨틱 분할의 장점과 경계 기반 방법의 구조 보존 능력을 동시에 확보하고자 한다.

### 3. 분할 트리 (Segmentation Trees)
슈퍼픽셀을 생성하고 이를 계층적으로 병합하는 방식이다. 그러나 이러한 방법은 주로 노드 특징에 의존하므로 인스턴스 영역의 기하학적 특징을 모델링하기 어렵다는 한계가 있다.

## 🛠️ Methodology

Ceb 프레임워크는 크게 다섯 가지 단계로 구성된다.

### 1. Seed Generation
시맨틱 분할 모델(예: U-Net)에서 생성된 확률 맵(probability map)을 입력으로 하여 인스턴스 시드(seed)를 생성한다. 구체적으로 Instance Candidate Forest (ICF)를 구축하여 확률 값의 임계값 변화에 따른 연결 성분(connected components)의 계층 구조를 분석하고, 리프 노드의 지역 최댓값을 시드로 선택한다.

### 2. Boundary Generation
생성된 시드와 확률 맵을 사용하여 수정된 Watershed 알고리즘을 적용한다. 이를 통해 가능한 모든 전경-전경 경계(region-region boundaries)와 그 경계들로 나누어진 영역(regions)을 생성한다. 결과적으로 세포 분할 문제는 어떤 경계가 '진짜'이고 어떤 경계가 '가짜'인지를 선택하는 문제로 변환된다.

### 3. Boundary Label Assignment (학습 단계)
학습을 위해 생성된 경계에 True/False 라벨을 부여해야 한다. 이를 위해 Ground Truth(GT) 인스턴스 마스크와 예측된 후보 인스턴스들 사이의 최적 매칭 문제를 정수 선형 계획법(Integer Linear Programming, ILP)으로 해결한다.
목적 함수는 다음과 같다.
$$ \text{GI-matching}(G, I) = \max_{f} \sum_{i \in G} \sum_{j \in I} M_{i,j} f_{i,j} $$
여기서 $M_{i,j}$는 IoU(Intersection-over-Union) 기반의 매칭 점수이며, $f_{i,j} \in \{0, 1\}$은 매칭 여부를 나타낸다. 매칭된 인스턴스 내부의 경계는 False로, 외부 경계는 True로 할당된다.

### 4. Boundary Signature Extraction
각 경계의 기하학적 특성을 포착하기 위해 'Boundary Signature'를 추출한다.
- 경계의 두 끝점(endpoints)을 찾고, 각 끝점 주변의 '포크 도로(fork road)' 형태의 픽셀들을 샘플링한다.
- 샘플링된 픽셀들을 이진 마스크(binary mask) 형태로 변환하여 이미지 형태의 특징 표현을 생성한다. 이는 MNIST 이미지와 유사한 형식이 된다.

### 5. Boundary Classification
추출된 Boundary Signature를 입력으로 하고, 앞서 할당된 True/False 라벨을 정답으로 하여 가벼운 이진 분류기(ResNet-18)를 학습시킨다. 손실 함수로는 Focal Loss를 사용한다.
추론 단계에서는 분류기가 예측한 확률 값이 0.5 이상인 경계만 유지하고, False로 판정된 경계로 나누어진 영역들은 서로 병합하여 최종 인스턴스를 생성한다.

### 6. Temporal Instance Consistency (비디오 데이터셋)
비디오 데이터의 경우 프레임 간 일관성을 활용하는 $\text{Ceb+temporal}$ 방식을 사용한다.
- **Initial State**: 높은 신뢰도의 경계를 통해 명확한 인스턴스를 먼저 선택한다.
- **Iterative Matching and Selection**: 이전 프레임($w-1$)과 다음 프레임($w+1$)의 선택된 인스턴스를 현재 프레임($w$)의 미선택 후보들과 매칭하는 $\text{SSM}(\text{Selected-Selected Matching})$과 $\text{SUM}(\text{Selected-Unselected Matching})$ 과정을 반복 수행하여 상태를 업데이트한다.

## 📊 Results

### 실험 설정
- **데이터셋**: 4개의 비디오 데이터셋(CTC)과 2개의 2D 데이터셋(BBBC039, TissueNet)을 사용하였다.
- **비교 대상**:
    - 시맨틱 분할 백본: U-Net, DCAN, Res2Net.
    - 클러스터링 방법: 0.5-Th, Otsu's, DenseCRF, MaxValue, H-EMD, GAC, ACWE, Watershed.
    - SOTA 방법론: CellPose, StarDist, Mask R-CNN, InstanSeg, CellViT 등.
- **지표**: F1-score, AJI (Average Jaccard Index), AP (Average Precision).

### 주요 결과
- **클러스터링 성능**: Ceb는 모든 백본 모델에서 기존의 픽셀 클러스터링 방법(Watershed, H-EMD 등)보다 일관되게 높은 성능을 보였다. 특히 DCAN 백본 사용 시 DIC-HeLa 데이터셋에서 F1 점수를 3.4%, AJI를 1.0% 향상시켰다.
- **SOTA 대비 성능**: Ceb는 최신 인스턴스 분할 방법론들과 비교했을 때 매우 경쟁력 있는 성능을 보였으며, 일부 지표에서는 이를 능가하였다.
- **Temporal 효과**: $\text{Ceb+temporal}$ 방식은 일반 Ceb보다 성능이 더 향상되었으며, 이는 비디오 데이터셋에서 세포 추적 정확도(TRA)의 상승으로 이어졌다.
- **2D 데이터셋**: BBBC039와 TissueNet에서도 Res2Net 백본과 결합했을 때 기존 클러스터링 방법 대비 F1, AJI, AP 모든 지표에서 유의미한 상승을 기록하였다.

## 🧠 Insights & Discussion

### 강점 및 분석
- **기하학적 특성 포착**: Boundary Signature 시각화 결과, True 경계는 'X'자 형태를 띠는 경향이 있고 False 경계는 'T'자나 'H'자 형태의 직각 구조를 가지는 특성이 발견되었다. 이를 통해 딥러닝 분류기가 경계의 기하학적 구조를 효과적으로 학습했음을 알 수 있다.
- **과분할/과소분할 해결**: 시각적 비교 결과, 기존 Watershed나 ACWE 방식에서 빈번하게 발생하는 과분할(over-segmentation) 및 과소분할(under-segmentation) 문제를 Ceb가 효과적으로 억제함을 확인하였다.

### 한계 및 비판적 해석
- **계산 복잡도**: 시맨틱 분할 자체보다 시드 생성(Seed Generation)과 경계 시그니처 추출(Boundary Signature Extraction) 단계에서 훨씬 많은 시간이 소요된다(단일 이미지 기준 각각 4.5s, 4.8s). 저자는 이를 Python 루프의 문제로 분석하며 C++/CUDA 기반 병렬화가 필요함을 명시하였다.
- **의존성**: 본 방법론은 시맨틱 분할 모델의 확률 맵에 전적으로 의존한다. 따라서 기반이 되는 시맨틱 분할 모델의 성능이 낮을 경우 최종 결과에 직접적인 영향을 미칠 수밖에 없는 구조적 한계가 있다.
- **데이터셋 특성**: PhC-C2DH-U373 데이터셋에서는 AUC-ROC가 낮게 나타났는데, 이는 해당 데이터셋의 세포 모양이 매우 불규칙하여 경계 시그니처의 특징 품질이 저하되었기 때문으로 분석된다.

## 📌 TL;DR

본 논문은 세포 인스턴스 분할 시 픽셀 단위의 목적 함수가 세포의 기하학적 특성을 놓친다는 점에 착안하여, **경계 수준의 특징(Boundary Signature)을 학습하고 분류하는 Ceb 프레임워크**를 제안하였다. 이 방법은 시맨틱 분할 결과물 위에서 작동하며, 수정된 Watershed 알고리즘과 ResNet-18 기반의 경계 분류기를 통해 전경 픽셀을 정확하게 클러스터링한다. 실험 결과, 기존의 픽셀 클러스터링 방식들을 압도하며 SOTA 방법론에 근접하거나 능가하는 성능을 보였다. 특히 비디오 데이터에서 시간적 일관성을 통합함으로써 분할 및 추적 성능을 극대화하였다. 이 연구는 단순 픽셀 예측을 넘어 구조적 특징을 활용한 인스턴스 분할의 중요성을 제시하며, 향후 정밀한 의료 영상 분석 및 세포 생물학 연구에 기여할 가능성이 높다.