# SegGroup: Seg-Level Supervision for 3D Instance and Semantic Segmentation

An Tao, Yueqi Duan, Yi Wei, Jiwen Lu, and Jie Zhou (2022)

## 🧩 Problem to Solve

본 논문은 3D 포인트 클라우드(Point Cloud)의 인스턴스 분할(Instance Segmentation) 및 시맨틱 분할(Semantic Segmentation)에서 발생하는 막대한 주석 비용(Annotation Cost) 문제를 해결하고자 한다. 기존의 강력한 지도 학습(Strong Supervision) 방식은 장면 내의 모든 포인트에 대해 레이블을 지정하는 Point-level labels를 요구하며, 이는 매우 많은 시간과 노동력을 필요로 한다. 예를 들어 ScanNet 데이터셋의 경우, 오버세그멘테이션(Over-segmentation)을 적용하더라도 한 장면당 평균 22.3분이 소요된다.

최근 일부 약지도 학습(Weakly-supervised learning) 연구들이 등장하였으나, 장면 수준(Scene-level)이나 서브클라우드 수준(Subcloud-level)의 레이블은 인스턴스별 특정 정보(Instance-specific information)를 제공하지 못해 인스턴스 분할 작업에 적용하기 어렵다는 한계가 있다. 따라서 본 논문의 목표는 최소한의 주석 비용으로도 강력한 지도 학습에 근접하는 성능을 낼 수 있는 효율적인 약지도 학습 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 3D 장면 분할에서 **인스턴스의 위치(Location) 정보가 매우 중요하다**는 직관에서 출발한다. 모든 포인트에 레이블을 붙이는 대신, 각 인스턴스당 단 하나의 포인트(가장 대표적인 세그먼트 내의 포인트)만 클릭하여 위치를 지정하는 **Seg-level supervision** 방식을 제안한다.

이를 위해 저자들은 다음과 같은 기여를 하였다:

1. 인스턴스당 하나의 포인트만 클릭하여 위치를 표시하는 저비용 주석 방식을 설계하였다.
2. **SegGroup**이라는 세그먼트 그룹화 네트워크를 제안하여, 희소한 Seg-level 레이블로부터 포인트 수준의 의사 레이블(Pseudo labels)을 생성하는 계층적 그룹화 메커니즘을 구현하였다.
3. 생성된 의사 레이블을 기존의 강력한 지도 학습 모델(예: PointGroup, MinkowskiNet)의 학습에 직접 활용함으로써, 매우 적은 주석 비용으로도 높은 성능을 달성할 수 있음을 입증하였다.

## 📎 Related Works

### 1. Point Cloud Segmentation

기존의 3D 분할 방식은 크게 Voxel-based와 Point-based로 나뉜다. 인스턴스 분할의 경우 Detection-based(경계 상자 추출 후 마스크 생성)와 Segmentation-based(시맨틱 레이블 예측 후 그룹화) 전략이 주로 사용된다. 대부분의 방법론은 모든 포인트에 레이블이 있는 강력한 지도 학습에 의존한다.

### 2. Weakly-supervised Segmentation

약지도 학습 분야에서는 Scene-level labels, Subcloud-level labels, 또는 무작위로 샘플링된 일부 포인트 레이블 등을 사용하는 연구들이 진행되었다. 하지만 이러한 방식들은 인스턴스별 위치 정보를 제공하지 못해 인스턴스 분할에는 부적합하다. OTOC [28]와 같은 최신 연구가 인스턴스당 한 점을 클릭하는 방식을 제안했으나, 이는 시맨틱 분할에 치중되어 있어 인스턴스 분할 작업으로 직접 확장하기 어렵다.

본 논문은 Seg-level supervision을 통해 시맨틱과 인스턴스 분할 모두에 적용 가능한 일반적인 프레임워크를 제공함으로써 기존 연구와 차별점을 갖는다.

## 🛠️ Methodology

본 논문은 두 단계(Two-stage) 접근 방식을 취한다. 첫 단계에서는 SegGroup 네트워크를 통해 의사 레이블을 생성하고, 두 번째 단계에서는 이 레이블을 사용하여 표준 지도 학습 모델을 훈련시킨다.

### 1. Seg-level Annotation

먼저 Normal-based graph cut 방법을 통해 장면을 여러 개의 작은 세그먼트로 나누는 오버세그멘테이션을 수행한다. 주석자는 각 인스턴스에서 가장 크고 대표적인 세그먼트 내의 포인트 하나만 클릭하여 시맨틱 클래스와 인스턴스 ID를 부여한다. 이 클릭 한 번으로 해당 포인트가 속한 전체 세그먼트에 레이블이 확장되며, 이를 **Seg-level labels**라고 한다. 이 방식은 장면당 약 1.93분으로 주석 시간을 획기적으로 단축한다.

### 2. SegGroup Network

SegGroup은 세그먼트 그래프(Segment Graph)를 입력으로 받아, 레이블이 없는 세그먼트들을 인접한 레이블 있는 세그먼트들로 계층적으로 그룹화하여 의사 레이블을 생성한다.

#### 전체 구조

- **Input**: 각 노드가 세그먼트를 나타내고 엣지가 인접성을 나타내는 세그먼트 그래프와 포인트 클라우드 데이터.
- **Pipeline**: Structural Grouping Layer $\rightarrow$ Semantic Grouping Layer 1 $\rightarrow$ Semantic Grouping Layer 2 $\rightarrow$ Final Clustering.

#### 주요 구성 요소

- **Feature Extractor**: 공유된 EdgeConv 네트워크를 사용하여 각 세그먼트의 로컬 특징을 추출한다. 층이 깊어질수록 세그먼트가 병합되므로 더 거시적인(Macroscopic) 정보를 추출하게 된다.
- **Graph Convolution (GCN)**: 동일 인스턴스 내 노드 간의 차이는 줄이고, 다른 인스턴스 간의 차이는 넓히는 역할을 한다.
  - 유사도 계수 $e_{ij}^l$ 계산:
    $$e_{ij}^l = \exp(-\lambda \| \mathbf{h}_i^l - \mathbf{h}_j^l \|^2)$$
  - 정규화된 계수 $a_{ij}^l$:
    $$a_{ij}^l = \frac{e_{ij}^l}{e_{ii}^l + \sum_{k \in N_i^l} e_{ik}^l}$$
  - 노드 특징 업데이트:
    $$\mathbf{h}_i^{l'} = \sigma \left( a_{ii}^l \mathbf{W}^l \mathbf{h}_i^l + \sum_{k \in N_i^l} a_{ik}^l \mathbf{W}^l \mathbf{h}_k^l \right)$$
- **Clustering**: 두 인접 노드가 서로 다른 인스턴스에 속하지 않고, 특징 간의 거리 $\text{dist}(\mathbf{h}_a, \mathbf{h}_b) < \tau$인 경우 두 노드를 병합한다. 병합된 노드의 특징은 Max-pooling으로 결정된다.
- **Final Clustering**: 모든 레이어를 거친 후에도 남아있는 레이블 없는 노드들을 가장 유사한 인접 노드에 그리디(Greedy)하게 병합하여 최종적으로 모든 포인트에 레이블을 부여한다.

### 3. Network Training

SegGroup은 EM-like 알고리즘을 통해 최적화된다.

1. **Forward-propagation**: 고정된 파라미터로 의사 레이블을 생성하고, 분류기(Classifier)를 통해 시맨틱 점수를 얻는다.
2. **Backward-propagation**: 생성된 의사 레이블을 고정한 상태에서 Cross-entropy loss를 사용하여 SegGroup과 분류기의 파라미터를 최적화한다.

## 📊 Results

### 실험 설정

- **데이터셋**: ScanNet (1,201개 훈련 장면, 312개 검증 장면, 100개 테스트 장면).
- **평가 지표**: 인스턴스 분할은 $\text{AP}, \text{AP}_{50}, \text{AP}_{25}$를, 시맨틱 분할은 $\text{mIoU}$를 사용한다.
- **비교 대상**: PointGroup (Strongly supervised), CSC [25], OTOC [28] (Weakly supervised).

### 주요 결과

- **의사 레이블 품질**: SegGroup이 생성한 의사 레이블은 Ground-truth와 매우 유사하며, 특히 벽(Wall), 바닥(Floor)과 같이 구조가 단순한 클래스에서 높은 $\text{mIoU}$를 보였다.
- **인스턴스 분할 성능**: $\text{AP}$ 기준 $\text{SegGroup(PointGroup)}$은 $24.6\%$를 기록하여, 강력한 지도 학습 모델들의 성능에 근접했으며, 특히 동일한 포인트 주석 비율을 가진 CSC-50($22.9\%$)보다 우수한 성능을 보였다.
- **시맨틱 분할 성능**: $\text{mIoU}$ 기준 MinkowskiNet 사용 시 $62.7\%$를 달성하였다. 이는 OTOC(69.1%)보다는 낮지만, 다른 약지도 학습 방법들보다 경쟁력 있는 수치이다.
- **주석 효율성**: 고정된 시간 예산(Annotation Budget) 하에서 비교했을 때, SegGroup은 1,201개 전 scenes를 Seg-level로 학습시킨 결과가 104개 scenes만 Point-level로 학습시킨 결과보다 월등히 뛰어났다(인스턴스 분할 $\text{AP}_{50}$ 기준 $43.4\%$ vs $19.9\%$).

## 🧠 Insights & Discussion

본 연구는 3D 장면 이해에서 **정확한 인스턴스 위치 정보가 매우 효율적인 감독 신호(High-yield supervision signal)**가 될 수 있음을 입증하였다.

**강점**:

- 주석 시간을 장면당 22.3분에서 1.93분으로 약 11배 이상 단축하면서도 성능 하락을 최소화하였다.
- 계층적 그룹화 구조를 통해 희소한 위치 정보로부터 밀도 높은 포인트 수준 레이블을 성공적으로 복원하였다.

**한계 및 논의**:

- 시맨틱 분할 성능이 OTOC보다 낮은 이유는 OTOC의 에너지 함수 기반 최적화가 시맨틱 정보 추출에 더 특화되어 있기 때문으로 분석된다. 그러나 OTOC는 인스턴스 분할에 직접 적용할 수 없다는 치명적인 한계가 있으며, SegGroup은 두 작업 모두를 수행할 수 있다는 범용성을 가진다.
- 또한, SegGroup은 OTOC에 비해 훨씬 적은 파라미터(0.15M vs 30.11M)를 사용하여 계산 효율성이 높다.

## 📌 TL;DR

이 논문은 3D 포인트 클라우드 분할에서 모든 포인트에 레이블을 붙이는 대신, 인스턴스당 단 하나의 포인트만 클릭하는 **Seg-level supervision** 방식을 제안한다. 이를 위해 세그먼트 그래프 상에서 계층적으로 노드를 병합하는 **SegGroup** 네트워크를 통해 고품질의 의사 레이블을 생성하며, 이를 통해 주석 비용을 획기적으로 낮추면서도 강력한 지도 학습 모델에 근접한 성능을 달성하였다. 이 연구는 향후 대규모 3D 데이터셋 구축 시 주석 비용 문제를 해결하는 핵심적인 방법론이 될 가능성이 높다.
