# Radar Instance Transformer: Reliable Moving Instance Segmentation in Sparse Radar Point Clouds

Matthias Zeller, Vardeep S. Sandhu, Benedikt Mersch, Jens Behley, Michael Heidingsfeld, Cyrill Stachniss (2023)

## 🧩 Problem to Solve

본 논문은 희소하고(sparse) 노이즈가 많은 레이더 포인트 클라우드(radar point clouds)에서 **이동 인스턴스 세그멘테이션(Moving Instance Segmentation)** 문제를 해결하고자 한다.

자율 주행 로봇과 차량의 안전한 주행을 위해서는 주변 환경에서 어떤 객체가 움직이고 있으며, 각각의 에이전트가 얼마나 존재하는지를 정확히 식별하는 것이 필수적이다. 레이더 센서는 카메라나 LiDAR와 달리 기상 악화 상황에서도 강인하며, Doppler velocity를 통해 동적 객체에 대한 직접적인 정보를 제공한다는 강력한 장점이 있다.

그러나 레이더 데이터는 다중 경로 전파(multi-path propagation)와 센서 자체의 특성으로 인해 노이즈가 심하고 데이터가 매우 희소하다는 한계가 있다. 기존의 접근 방식들은 이동 객체 세그멘테이션(Moving Object Segmentation, MOS)과 인스턴스 세그멘테이션(Instance Segmentation)을 별개로 처리하거나, 데이터 집계(aggregation) 과정에서 높은 지연 시간(latency)을 발생시키고, 복셀화(voxelization) 과정에서 정보 손실이 발생하는 문제가 있었다. 따라서 본 연구의 목표는 단일 네트워크 내에서 MOS와 인스턴스 세그멘테이션을 동시에 수행하는 파놉틱(panoptic) 구조를 통해, 희소한 레이더 데이터에서도 신뢰할 수 있는 이동 인스턴스 식별 능력을 확보하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **전체 해상도를 유지하는 백본(Full-resolution Backbone)**과 **시간적 정보를 효율적으로 통합하는 어텐션 메커니즘**을 결합하여 정보 손실을 최소화하고 객체 간의 구분 능력을 극대화하는 것이다.

주요 기여 사항은 다음과 같다:
1. **Sequential Attentive Feature Encoding (SAFE)**: 여러 프레임의 데이터를 전체 네트워크에 통과시키는 대신, 현재 스캔의 특징(feature)을 이전 스캔들의 정보로 보강하는 효율적인 시간적 인코딩 모듈을 제안한다.
2. **Full-resolution Backbone**: 희소한 레이더 데이터의 특성을 고려하여, 네트워크 전체에서 원래의 포인트 수를 유지하는 병렬 경로를 구축함으로써 정보 손실을 방지한다.
3. **Attentive Instance Transformer Head**: 로컬(local) 및 글로벌(global) 어텐션을 통해 인스턴스 간의 유사도를 계산하여, 클래스에 의존하지 않는 신뢰할 수 있는 인스턴스 할당을 가능하게 한다.
4. **Graph-based Instance Assignment**: 어텐션 기반의 인접 행렬을 이용해 그래프의 모듈성(Modularity)을 최대화하는 방식으로 인스턴스를 분할한다.
5. **신규 벤치마크 구축**: RadarScenes 데이터셋을 확장하여 포인트 클라우드 기반의 이동 인스턴스 세그멘테이션 벤치마크를 최초로 제시한다.

## 📎 Related Works

기존의 포인트 클라우드 처리 방식은 크게 네 가지로 구분된다:
- **Projection-based**: 2D 이미지나 범위 이미지(range image)로 변환하여 CNN을 적용한다. 계산 효율은 좋으나 이산화 아티팩트(discretization artifacts)와 역투영 오류(back projection errors)로 인해 희소한 레이더 데이터에서는 정보 손실이 심각하다.
- **Voxel-based**: 3D 공간을 복셀로 나누어 처리함으로써 역투영 오류를 줄이지만, 여전히 이산화로 인한 정보 손실이 발생하며 레이더 데이터의 희소성을 처리하기에 부적합한 면이 있다.
- **Point-based**: 포인트 클라우드를 직접 처리하여 공간 정보를 보존한다. PointNet++ 등이 대표적이나, 많은 경우 계산 부담을 줄이기 위해 super-voxels나 오버 세그멘테이션 클러스터를 사용하는데, 이는 단일 포인트가 하나의 인스턴스를 대표할 수 있는 레이더 데이터에서는 부적절하다.
- **Transformer-based**: Self-attention 메커니즘을 통해 전역적인 문맥을 파악한다. 최근 연구들이 우수한 성과를 보이고 있으나, 레이더 데이터의 특수한 노이즈와 희소성을 해결하기 위한 최적화된 구조는 부족한 실정이다.

본 논문은 이러한 기존 방식들의 한계를 극복하기 위해, 포인트 기반의 Transformer 구조를 채택하되 시간적 정보의 효율적 통합과 해상도 유지에 집중하여 차별점을 둔다.

## 🛠️ Methodology

### 1. Sequential Attentive Feature Encoding (SAFE)
SAFE 모듈은 현재 스캔 $P_t$의 특징을 이전 $T$개의 스캔들($P_{t-T}, \dots, P_{t-1}$)의 정보로 보강한다. 모든 스캔을 네트워크에 통과시키는 대신, 현재 스캔만 전체 네트워크를 통과시키고 이전 스캔들은 SAFE 모듈 내에서만 처리하여 계산 효율성을 높인다.

- **정렬 및 인코딩**: 이전 스캔들을 현재 스캔의 좌표계로 정렬(pose alignment)한 후, KPConv 레이어를 통해 특징을 추출한다.
- **Intra-attention**: 현재 스캔의 특징을 쿼리($Q$), 이전 스캔들의 특징을 키($K$)와 밸류($V$)로 설정한다.
  $$Q=X_t W_Q, \quad K=X_p W_K, \quad V=X_p W_V$$
- **로컬 어텐션**: $k$-최근접 이웃($kNN$)을 통해 국소 영역으로 제한하여 어텐션을 수행하며, 상대적 위치 인코딩 $R$을 추가하여 정밀한 위치 정보를 반영한다.
- **최종 특징**: 어텐션 가중치 $A$를 통해 가중합을 구하고, 이를 현재 스캔의 특징과 결합하여 시간적으로 보강된 특징 $X_{SAFE}$를 생성한다.

### 2. Backbone
백본은 정보 손실을 막기 위해 **전체 해상도(Full-resolution)**를 유지하는 구조를 가진다. 
- **구조**: 4개의 스테이지($S_1, \dots, S_4$)로 구성되며, 각 스테이지는 Point Transformer 블록으로 이루어져 있다.
- **병렬 처리**: U-Net과 달리 전체 해상도 경로를 병렬로 유지하며, 다운샘플링된 고차원 특징을 다시 업샘플링하여 원래 해상도의 포인트들에 병합한다.
- **샘플링**: FPS(Farthest Point Sampling)와 max pooling을 사용하여 해상도를 단계적으로 $N \to N/2 \to N/4 \to N/8$로 줄이며, 다시 trilinear interpolation으로 복원한다.

### 3. Moving Instance Transformer Head
이 모듈은 MOS(Moving Object Segmentation)와 인스턴스 할당을 동시에 수행한다.
- **MOS 예측**: MLP를 통해 각 포인트가 $\text{static}$인지 $\text{moving}$인지 예측한다.
- **로컬 유사도 ($S_{loc}$)**: $kNN$ 영역 내에서 포인트 간의 유사도를 dot-product 어텐션으로 계산하고 sigmoid 함수를 적용한다.
  $$S_{loc_{i,j}} = \text{sigmoid}(Q_{b^*i,j} K_{b^*i,j}^\top + R_{b_{i,j}})$$
- **글로벌 유사도 ($S_{glob}$)**: 계산 비용을 줄이기 위해 $\text{moving}$으로 예측된 포인트들($N_{mov}$)에 대해서만 전역 유사도를 계산한다.
  $$S_{glob_{i,j}} = \text{sigmoid}(Q_{b^{**}i,j} K_{b^{**}i,j}^\top)$$

### 4. Graph-based Instance Assignment
최종 인스턴스 ID를 부여하기 위해 반경 그래프(radius graph) $G=(V, E)$를 생성하고 그래프 파티셔닝을 수행한다.
- **어텐션 기반 인접 행렬**: 단순 거리 기반의 인접 행렬 $A_{adj}$에 글로벌 유사도 $S_{glob}$를 원소별 곱(element-wise product)하여, 단순 거리상으로는 가깝지만 실제로는 다른 인스턴스인 경우를 걸러낸다.
  $$A_{adj}^{attn} = S_{glob} \odot A_{adj}$$
- **모듈성(Modularity) 최대화**: 다음의 모듈성 $Q$를 최대화하는 방향으로 그래프를 분할한다.
  $$Q = \frac{1}{4m} \sum_{i,j} \left( A_{adj_{i,j}} - \frac{k_i k_j}{2m} \right) s_i s_j$$
  여기서 $k_i, k_j$는 노드의 차수, $m$은 엣지의 총 수, $s$는 클러스터 할당 벡터이다. 이 최적화 과정은 Spectral approach와 Vertex moving 방법을 통해 수행된다.

## 📊 Results

### 실험 설정
- **데이터셋**: RadarScenes(대규모, 고해상도) 및 View-of-Delft(중규모)를 사용하였다.
- **지표**: Panoptic Quality (PQ), Intersection over Union (IoU), Segmentation Quality (SQ), Recognition Quality (RQ)를 측정하였다.
- **비교 대상**: Mask3D, Stratified Transformer, Gaussian Radar Transformer(GRT), 4DMOS 등 최신 모델 및 HDBSCAN/Mean-shift 기반 클러스터링 조합을 비교군으로 설정하였다.

### 주요 결과
- **정량적 성과**: RadarScenes 테스트 셋에서 **PQ 86.5, IoU 92.6**을 기록하며 모든 베이스라인을 압도하였다. 특히 Mask3D보다 $\text{PQ}_{mov}$ 측면에서 10%p 이상의 향상을 보였으며, 레이더 특화 모델인 GRT보다도 $\text{IoU}_{mov}$에서 5%p 이상 높게 나타났다.
- **일반화 능력**: View-of-Delft 데이터셋에서도 $\text{IoU}_{mov}$ 기준 63.3%를 달성하며 GRT(62.3%)와 Stratified Transformer(55.6%)보다 우수한 성능을 입증하였다.
- **효율성**: 평균 추론 시간은 **31.7ms (약 31Hz)**로, 센서의 프레임 레이트(17Hz)보다 빨라 실시간 적용 가능성을 보여주었다. 파라미터 수는 3.8M으로 다른 Transformer 기반 모델들보다 훨씬 적다.

## 🧠 Insights & Discussion

### 강점 및 분석
- **시간적 정보의 중요성**: Ablation study를 통해 이전 스캔 $T=2$개만 추가해도 성능이 유의미하게 향상됨을 확인하였다. $T$를 더 늘리면 성능이 소폭 상승하나 계산 비용이 급증하므로 $T=2$가 최적의 트레이드오프 지점임을 밝혔다.
- **해상도 유지의 효과**: 백본의 top-level Transformer 블록을 제거했을 때 성능이 하락하는 것을 통해, 희소한 레이더 데이터에서 전체 해상도를 유지하며 고차원 특징을 추출하는 것이 중요함을 입증하였다.
- **그래프 파티셔닝의 유연성**: 어텐션 기반 인접 행렬을 사용함으로써, 오프셋 예측(offset prediction) 없이도 큰 객체와 작은 객체(보행자 등)를 동시에 신뢰성 있게 구분할 수 있었다.

### 한계 및 비판적 해석
- **Doppler Velocity의 한계**: Doppler 속도는 Radial velocity(방사 속도)만을 측정하므로, 센서와 수직으로 움직이는 Tangential movement는 감지하기 어렵다. 이로 인해 교차로 장면 등에서 일부 원거리 객체 식별에 어려움이 있을 수 있다.
- **데이터 의존성**: RadarScenes 데이터셋의 정밀한 포즈 정보에 의존하고 있으며, 실제 환경에서 포즈 추정(Odometry)의 오차가 클 경우 SAFE 모듈의 정렬 성능이 떨어질 가능성이 있다.

## 📌 TL;DR

본 논문은 희소하고 노이즈가 많은 레이더 포인트 클라우드에서 **이동 인스턴스 세그멘테이션**을 수행하기 위한 **Radar Instance Transformer**를 제안한다. 효율적인 시간적 특징 인코딩(SAFE), 정보 손실을 최소화하는 전체 해상도 백본, 그리고 로컬/글로벌 어텐션과 그래프 파티셔닝을 결합한 헤드를 통해 SOTA 성능을 달성하였다. 특히, 클래스 정보 없이도 인스턴스를 구분할 수 있는 구조를 통해 '롱테일' 클래스 문제에 유연하게 대처할 수 있으며, 실시간 추론 속도를 확보하여 자율 주행 시스템의 센서 리던던시(redundancy)를 강화하는 데 기여할 수 있는 연구이다.