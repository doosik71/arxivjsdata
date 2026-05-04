# Deep Learning Based 3D Segmentation: A Survey

Yong He, Hongshan Yu, Xiaoyan Liu, Zhengeng Yang, Wei Sun, Saeed Anwar and Ajmal Mian (2024)

## 🧩 Problem to Solve

본 논문은 컴퓨터 비전 및 그래픽스 분야에서 매우 중요하지만 도전적인 과제인 3D 세그멘테이션(3D Segmentation) 기술의 전반적인 동향을 분석하는 것을 목표로 한다. 3D 세그멘테이션은 자율 주행, 모바일 로봇, 산업 제어, 그리고 증강 및 가상 현실(AR/VR)과 같은 광범위한 응용 분야에서 3D 장면 내 객체의 세밀한 레이블을 예측하는 핵심 기술이다.

기존의 3D 세그멘테이션 방법론은 수작업으로 설계된 특징(hand-crafted features)과 머신러닝 분류기에 의존하여 일반화 능력이 부족하다는 한계가 있었다. 최근 2D 컴퓨터 비전의 성공에 힘입어 딥러닝 기술이 도입되었으나, 3D 데이터는 포인트 클라우드(Point Cloud)의 불규칙성이나 고해상도 복셀(Voxel) 변환 시 발생하는 막대한 계산 비용과 같은 고유한 문제점을 가지고 있다. 또한, 기존의 서베이 논문들은 RGB-D나 포인트 클라우드 등 특정 데이터 모달리티에만 집중되어 있어, 모든 3D 데이터 표현 방식과 응용 도메인을 포괄적으로 다룬 최신 분석 보고서가 부재한 상황이다. 따라서 본 논문은 지난 6년간의 220개 이상의 연구를 분석하여 3D 세그멘테이션의 체계적인 프레임워크를 제공하고자 한다.

## ✨ Key Contributions

본 논문의 중심적인 기여는 3D 세그멘테이션을 세 가지 수준(Semantic, Instance, Part)으로 정의하고, 이를 처리하는 다양한 딥러닝 방법론을 데이터 표현 방식에 따라 체계적으로 분류 및 분석한 점에 있다.

핵심적인 설계 아이디어는 3D 데이터를 RGB-D, 투영 이미지(Projected Images), 복셀(Voxel), 포인트 클라우드(Point Cloud), 메쉬(Mesh), 3D 비디오로 구분하고 각 표현 방식별 네트워크 아키텍처의 장단점을 분석하는 것이다. 특히, 단순한 나열을 넘어 세그멘테이션 파이프라인의 구성 요소(이웃 탐색, 특징 추출, 다운샘플링 등)를 상세히 분석하여 연구자들이 자신의 목적에 맞는 방법론을 선택할 수 있는 가이드를 제공한다.

## 📎 Related Works

논문은 3D 세그멘테이션을 세 가지 세부 작업으로 구분하여 관련 연구를 설명한다.
1. **Semantic Segmentation**: 객체의 클래스 레이블(예: 의자, 탁자)을 예측하는 작업이다.
2. **Instance Segmentation**: 동일 클래스 내에서도 서로 다른 개체(예: 의자 1, 의자 2)를 구분하는 작업이다.
3. **Part Segmentation**: 개체를 더 세부적인 구성 요소(예: 의자의 등받이, 다리)로 분해하는 작업이다.

기존의 서베이 연구들은 RGB-D 세그멘테이션이나 원격 탐사 이미지, 혹은 포인트 클라우드 일반론에 치우쳐 있었다. 본 논문은 이러한 한계를 극복하고, 3D 세그멘테이션이라는 구체적인 작업에 집중하여 데이터 표현 방식과 네트워크 구조 간의 상관관계를 심층적으로 분석함으로써 기존 연구들과 차별점을 둔다.

## 🛠️ Methodology

본 논문은 3D 세그멘테이션의 방법론을 데이터 표현 방식과 작업의 목적에 따라 다음과 같이 체계화하여 설명한다.

### 1. 3D Semantic Segmentation
데이터 표현 방식에 따라 다섯 가지 카테고리로 분류한다.
- **RGB-D 기반**: RGB 이미지와 깊이(Depth) 정보를 함께 사용한다. 주로 2채널 네트워크를 사용하며, 성능 향상을 위해 Multi-task learning, Depth encoding, Multi-scale network, Transformer 기반 퓨전 모듈 등을 결합한다.
- **투영 이미지 기반**: 3D 장면을 2D 이미지로 투영하여 2D CNN을 활용한다. Multi-view projection과 Spherical projection(구형 투영) 방식이 대표적이다.
- **복셀 기반**: 3D 공간을 정규 격자인 복셀로 나누어 3D CNN을 적용한다. 계산 효율을 위해 Octree 구조나 Sparse Convolution(희소 합성곱)을 사용한다.
- **포인트 기반**: 포인트 클라우드에 직접 적용하며, 다음과 같이 세분화된다.
    - **MLP 기반**: PointNet, PointNet++와 같이 공유 MLP와 대칭 함수(Max-pooling)를 사용한다.
    - **Point Convolution 기반**: 포인트의 기하학적 특성을 반영한 합성곱 커널을 학습한다 (예: KPConv, PointConv).
    - **Graph Convolution 기반**: 포인트 간의 관계를 그래프 구조로 모델링하여 특징을 추출한다 (예: DGCNN).
    - **Transformer 기반**: Self-attention 메커니즘을 통해 전역적인 문맥 정보를 학습한다 (예: Point Transformer).
- **3D 비디오 기반**: 연속된 프레임 간의 시공간적(Spatio-temporal) 정보를 활용하며, RNN이나 4D Convolution을 적용한다.

### 2. 3D Instance Segmentation
객체 제안(Proposal) 생성 여부에 따라 두 가지 방향으로 나뉜다.
- **Proposal-Based**: 먼저 3D Bounding Box 등의 제안 영역을 예측한 후, 해당 영역 내에서 마스크를 생성한다. Detection 기반과 Detection-free 기반으로 나뉜다.
- **Proposal-Free**: 각 포인트의 특징 임베딩(Embedding)을 학습한 뒤, 클러스터링(Clustering)을 통해 인스턴스를 구분한다. 최근에는 Dynamic Convolution이나 Transformer Decoder를 사용하여 제안 단계 없이 직접 인스턴스를 예측하는 방식이 주목받고 있다.

### 3. 3D Part Segmentation
데이터의 정규성 여부에 따라 구분한다.
- **Regular Data**: 투영 이미지나 복셀을 사용하며, 고해상도 복셀의 계산 비용 문제를 해결하기 위한 전략이 핵심이다.
- **Irregular Data**: 메쉬(Mesh)나 포인트 클라우드를 사용하며, 표면의 기하학적 세밀함을 포착하는 것이 중요하다.

### 4. 주요 평가 지표 (Equations)
논문에서는 세그멘테이션 성능을 측정하기 위해 다음과 같은 수식을 정의한다.

**Semantic Segmentation 지표:**
- Overall Accuracy (OA): $$\text{OA} = \frac{\sum_{i=0}^{K} p_{ii}}{\sum_{i=0}^{K} \sum_{j=0}^{K} p_{ij}}$$
- Mean Accuracy (mA): $$\text{mA} = \frac{1}{K+1} \sum_{i=0}^{K} \frac{p_{ii}}{\sum_{j=0}^{K} p_{ij}}$$
- Mean Intersection over Union (mIoU): $$\text{mIoU} = \frac{1}{K+1} \sum_{i=0}^{K} \frac{p_{ii}}{\sum_{j=0}^{K} p_{ij} + \sum_{j=0}^{K} p_{ji} - p_{ii}}$$

**Instance Segmentation 지표:**
- Average Precision (AP): $$\text{AP} = \frac{\sum_{c=0}^{K} \sum_{i=0}^{N_c} c_{ii}}{\sum_{c=0}^{K} (c_{ii} + \sum_{j=0}^{N_c} c_{ij})}$$
- Mean Average Precision (mAP): $$\text{mAP} = \frac{1}{K+1} \sum_{c=0}^{K} \text{AP}_c$$

## 📊 Results

### 1. 실험 설정
- **데이터셋**: S3DIS, ScanNet, Semantic3D, SemanticKITTI, ShapeNet, NYUDv2, SUN-RGBD 등이 사용되었다.
- **지표**: mIoU, mAP, OA, mA 등을 통해 정량적 평가를 수행하였다.

### 2. 주요 결과 분석
- **Semantic Segmentation**: 포인트 기반 방법론 중 Point Transformer v3와 같은 최신 Transformer 모델들이 기존의 MLP나 CNN 기반 모델보다 높은 정확도를 보였다. 특히 대규모 장면 처리에서는 RandLA-Net과 같은 효율적인 아키텍처가 강점을 가진다.
- **Instance Segmentation**: Proposal-free 방식이 Proposal-based 방식보다 전반적으로 우수한 성능을 보였다. 이는 제안 영역 생성 단계에서 발생하는 오차를 피하고, 포인트 클라우드의 전역적 특징을 직접 활용할 수 있기 때문이다. 특히 Spherical Mask 모델이 ScanNet 데이터셋에서 81.2%의 mAP를 기록하며 SOTA 성능을 보였다.
- **Part Segmentation**: ShapeNet 데이터셋에서 대부분의 모델이 유사한 성능을 보였는데, 이는 데이터셋의 객체들이 합성(synthetic) 데이터로서 정규화되어 있고 배경 소음이 적어 기하학적 특징 추출이 상대적으로 쉽기 때문으로 분석된다.

## 🧠 Insights & Discussion

### 1. 강점 및 기여도
본 논문은 파편화되어 있던 3D 세그멘테이션 연구들을 데이터 표현 방식과 작업 수준이라는 두 가지 축으로 완벽하게 정리하였다. 특히, 단순한 모델 비교를 넘어 '이웃 탐색 $\rightarrow$ 특징 추상화 $\rightarrow$ 다운샘플링'으로 이어지는 포인트 클라우드 처리 파이프라인의 구성 요소를 분석하여, 향후 새로운 모델을 설계할 때 고려해야 할 핵심 모듈을 명확히 제시하였다.

### 2. 한계 및 향후 과제
논문은 3D 세그멘테이션의 네 가지 주요 도전 과제를 제시한다.
- **데이터 부족 및 품질 문제**: 2D에 비해 어노테이션 비용이 매우 높다. 이를 해결하기 위해 합성 데이터 생성, 전이 학습(Transfer Learning), 준지도 학습(Semi-supervised Learning)의 필요성을 강조한다.
- **일반화 및 강건성**: 센서 종류나 환경 변화에 따라 성능이 급격히 저하되는 문제가 있다. Domain Adaptation과 멀티모달(Multi-modal) 접근법이 해결책으로 제시된다.
- **계산 복잡도**: 3D 데이터의 고차원성으로 인해 메모리와 연산량이 막대하다. 모델 압축(Model Compression)과 효율적인 희소 표현(Sparse Representation) 학습이 필수적이다.
- **해석 가능성**: 딥러닝 모델의 결정 과정이 불투명하다. XAI(Explainable AI) 기법을 통해 3D 세그멘테이션 결과의 근거를 시각화하고 분석하는 연구가 필요하다.

## 📌 TL;DR

본 논문은 딥러닝 기반 3D 세그멘테이션(Semantic, Instance, Part)의 최신 동향을 모든 데이터 표현 방식(RGB-D, Point, Voxel, Mesh 등)을 망라하여 분석한 종합 서베이 보고서이다. 특히 포인트 기반 방법론의 진화 과정(MLP $\rightarrow$ Conv $\rightarrow$ Graph $\rightarrow$ Transformer)을 체계적으로 정리하였으며, Proposal-free 방식의 우수성과 데이터 부족 및 계산 복잡도라는 현실적인 한계를 지적하였다. 이 연구는 3D 비전 분야의 연구자들이 데이터 특성에 맞는 최적의 아키텍처를 선택하고, 향후 대형 모델(Large Models) 및 멀티모달 학습 방향으로 나아갈 수 있는 학술적 이정표를 제공한다.