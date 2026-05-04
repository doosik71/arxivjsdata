# Geodesic-Former: a Geodesic-Guided Few-shot 3D Point Cloud Instance Segmenter

Tuan Ngo and Khoi Nguyen (2022)

## 🧩 Problem to Solve

본 논문은 3D 포인트 클라우드 분야에서 새로운 과제인 **Few-shot 3D Point Cloud Instance Segmentation (3DFSIS)** 문제를 정의하고 이를 해결하는 것을 목표로 한다. 3DFSIS는 특정 타겟 클래스를 대표하는 소수의 어노테이션된 포인트 클라우드(support scenes)가 주어졌을 때, 쿼리 장면(query scene) 내에서 해당 타겟 클래스의 모든 인스턴스를 분할하는 작업이다.

이 문제는 3D 포인트 단위의 인스턴스 분할 어노테이션 비용이 매우 높다는 점에서 실용적인 중요성을 가진다. 특히 LiDAR 센서로 수집된 3D 포인트 클라우드는 객체의 표면 근처에서는 조밀하지만 그 외의 영역에서는 희소하거나 비어 있는 **밀도 불균형(density imbalance)** 특성을 가진다. 이러한 특성 때문에 단순한 Euclidean distance는 서로 다른 객체를 구분하는 데 한계가 있으며, 이는 Few-shot 설정에서 모델의 일반화 성능을 더욱 떨어뜨리는 요인이 된다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Geodesic distance(측지선 거리)**를 활용하여 LiDAR 포인트 클라우드의 밀도 불균형 문제를 해결하고, 이를 Transformer Decoder의 가이드 신호로 사용하는 것이다.

주요 기여 사항은 다음과 같다.

1. **새로운 태스크 정의**: 3D 포인트 클라우드 기반의 Few-shot 인스턴스 분할이라는 새로운 문제를 제안하였다.
2. **데이터셋 벤치마크 구축**: ScannetV2와 S3DIS 데이터셋을 활용하여 3DFSIS를 평가하기 위한 새로운 데이터 분할(splits) 방식을 제안하였다.
3. **Geodesic-Former 제안**: Geodesic distance embedding을 Positional Encoding으로 사용하는 Transformer Decoder와 Dynamic Convolution을 결합한 새로운 아키텍처를 제안하여 기존 SOTA 3DIS 방법론들을 뛰어넘는 성능을 달성하였다.

## 📎 Related Works

본 논문은 기존의 인스턴스 분할 연구를 세 가지 관점에서 분석한다.

1. **3D Instance Segmentation (3DIS)**: Proposal-based(예: 3D-SIS, 3D-BoNet)와 Proposal-free(예: PointGroup, DyCo3D) 방식으로 나뉜다. 하지만 이들은 훈련셋과 테스트셋의 클래스가 동일하다고 가정하며, 대량의 학습 데이터가 필요하므로 Few-shot 설정에는 부적합하다.
2. **Few-shot 2D Instance Segmentation**: Mask R-CNN 등을 확장하여 소량의 샘플로 학습하지만, 2D 이미지는 격자 구조의 밀집 데이터인 반면 3D 포인트 클라우드는 무질서하고 희소한 구조이므로 직접적인 적용이 불가능하다.
3. **Few-shot 3D Semantic Segmentation (3DF3S)**: 최근 제안된 태스크로, 포인트별 시맨틱 라벨을 예측하지만 동일 클래스 내의 서로 다른 인스턴스를 구분해야 하는 3DFSIS보다 상대적으로 쉬운 문제이다.

본 연구는 특히 Transformer의 Attention 메커니즘이 순서가 없는(unordered) 포인트 클라우드 데이터에 자연스럽게 적합하다는 점에 주목하여, 이를 3DIS 및 3DFSIS에 최초로 적용하였다.

## 🛠️ Methodology

### 전체 파이프라인

Geodesic-Former의 전체 구조는 공유된 Backbone을 통해 특징을 추출하고, 지원 장면(support scene)에서 타겟 클래스의 특징 벡터를 추출한 뒤, 이를 쿼리 장면의 특징과 결합하여 인스턴스 마스크를 생성하는 구조이다.

### 주요 구성 요소 및 절차

**1. Context 및 Anchor Points 준비**
먼저 쿼리 특징 $F_q \in \mathbb{R}^{N_q \times d}$와 지원 특징 벡터 $f_s \in \mathbb{R}^{1 \times d}$를 결합하여 Context points 특징 $F_c \in \mathbb{R}^{N_q \times d}$를 생성한다. 이때 다음과 같은 수식을 사용한다.
$$F^c = W_{proj} * [F^q \odot f^s; F^q - f^s; F^q]$$
여기서 $\odot$은 채널별 곱셈, $[\cdot;\cdot]$은 결합(concatenation)을 의미하며, 이를 통해 지원 클래스의 특성이 반영된 정제된 특징을 얻는다. 이후 Farthest Point Sampling(FPS)과 유사도 네트워크(MLP)를 통해 타겟 클래스와 외형적 유사성이 높은 후보점인 **Anchor points** $F_a \in \mathbb{R}^{N_a \times d}$를 추출한다.

**2. Geodesic Distance Embedding 계산**
Euclidean distance의 한계를 극복하기 위해 장면의 기하학적 구조를 인코딩하는 Geodesic distance를 계산한다.

- Ball query 알고리즘을 통해 각 포인트가 최대 $\kappa$개의 인접 포인트와 연결된 유향 희소 그래프(directed sparse graph)를 생성한다.
- Dijkstra 알고리즘을 사용하여 각 Anchor point에서 모든 Context point까지의 최단 경로 길이를 계산하여 Geodesic distance를 구한다.
- 계산된 거리는 $\sin/\cos$ 함수를 이용해 고차원 임베딩 $G_i \in \mathbb{R}^{N_q \times d}$로 변환된다.

**3. Transformer Decoder**
DETR의 구조를 차용한 Decoder는 Anchor points($F_a$)와 Context points($F_c$)를 입력으로 받는다.

- **Self-attention**: Anchor points 간의 관계를 파악하여 전체적인 객체 구조를 캡처한다.
- **Cross-attention**: Anchor와 Context points 간의 관계를 통해 객체 정보를 정밀하게 표현한다.
- 특히, 일반적인 좌표 임베딩 대신 위에서 계산한 **Geodesic distance embedding $G$를 Positional Encoding으로 사용**하여 기하학적 구조를 가이드한다. 최종적으로 각 Anchor point에 대응하는 Dynamic Convolution 커널 $W_i$를 생성한다.

**4. Dynamic Convolution**
Mask head를 통해 생성된 마스크 특징 $F_{mask}$와 Geodesic distance embedding $G_i$를 결합하여 최종 인스턴스 마스크 $bm_i$를 생성한다.
$$bm_i = \text{Conv}([F_{mask}; G_i], W_i)$$
여기서 $W_i$는 Transformer Decoder가 생성한 커널이며, 이를 통해 각 객체에 특화된 동적 필터링이 수행된다.

### 학습 전략

학습은 두 단계로 진행된다.

1. **Pretraining**: 기본 클래스(base classes)를 사용하여 표준 3DIS 태스크로 Backbone, Mask head, Decoder를 학습시킨다. 이때 Hungarian algorithm을 이용해 예측값과 Ground Truth 간의 최적 매칭을 수행하며 Dice loss와 Focal loss를 사용한다.
2. **Episodic Training**: 테스트 환경과 유사하게 지원 장면과 쿼리 장면을 무작위로 샘플링하여 학습함으로써 Few-shot 시나리오에 적응시킨다. 이 단계에서는 Backbone과 Mask head를 동결하고 Transformer Decoder와 유사도 네트워크를 학습시킨다.

## 📊 Results

### 실험 설정

- **데이터셋**: ScannetV2(18개 인스턴스 클래스)와 S3DIS(12개 메인 클래스)를 사용하였으며, 알파벳 순서 등으로 훈련/테스트 셋을 엄격히 분리하였다.
- **비교 대상**: SOTA 3DIS 방법론인 DyCo3D, PointGroup, HAIS를 Few-shot 설정으로 수정하여 비교하였다.
- **지표**: ScannetV2는 mAP, $AP_{50}$를 사용하였고, S3DIS는 mCov, mPrec, mRec를 사용하였다.

### 주요 결과

- **정량적 성과**: Geodesic-Former는 모든 지표에서 기존 방법론들을 큰 차이로 압도하였다. ScannetV2 기준, 1-shot에서는 mAP +4.4, 5-shot에서는 mAP +6.8의 성능 향상을 보였다.
- **정성적 분석**: 특히 얇은 객체(shower curtain), 거대한 객체(table), 불완전한 형태의 객체(window) 분할에서 강점을 보였다. 이는 Transformer Decoder와 Geodesic embedding이 객체의 다양한 규모와 형태를 잘 처리함을 입증한다.
- **Geodesic 거리의 효용성**: 시각화 결과, Euclidean distance로는 구분되지 않는 인접 객체들이 Geodesic distance에서는 매우 멀게 측정되어, 객체 간 분리 능력이 탁월함을 확인하였다.

## 🧠 Insights & Discussion

**강점 및 분석**

- **기하학적 가이드의 중요성**: 단순한 좌표 정보보다 Geodesic distance가 포인트 클라우드의 실제 표면 구조를 더 잘 반영하며, 이것이 인스턴스 구분 성능 향상의 핵심 동력임을 확인하였다.
- **분류 헤드의 제거**: Few-shot 설정에서는 훈련 단계의 배경(BG)이 테스트 단계에서는 전경(FG)이 되는 'FG/BG 혼동' 문제가 발생한다. 본 논문은 명시적인 분류 헤드 대신 유사도 네트워크를 통해 Anchor point를 필터링함으로써 이 문제를 효과적으로 회피하였다.

**한계 및 논의사항**

- **Shot 수 증가에 따른 한계**: 5-shot 이상의 데이터가 주어졌을 때 성능 향상 폭이 미미했다. 이는 현재 지원 특징들을 단순히 평균(averaging) 내어 사용하는 방식의 한계이며, 향후 여러 지원 장면의 기하학적 구조를 더 정교하게 통합하는 연구가 필요함을 시사한다.
- **오분류 사례**: 외형이 매우 유사한 객체(예: 소파와 스툴)의 경우 여전히 동일 인스턴스로 오분류하는 경향이 있어, 세밀한 외형 차이를 구분하는 능력이 추가로 요구된다.

## 📌 TL;DR

본 논문은 **Few-shot 3D Point Cloud Instance Segmentation**이라는 새로운 문제를 정의하고, 이를 해결하기 위해 **Geodesic distance**를 활용한 **Geodesic-Former**를 제안하였다. 핵심은 LiDAR 데이터의 밀도 불균형을 극복하기 위해 측지선 거리를 Transformer Decoder의 Positional Encoding으로 사용하여 객체 간 구분을 명확히 하고, Dynamic Convolution을 통해 정밀한 마스크를 생성하는 것이다. 이 연구는 3D 포인트 클라우드 분석에서 기하학적 구조 기반의 가이드가 Few-shot 학습의 일반화 성능을 비약적으로 높일 수 있음을 보여주었으며, 향후 3D 비전의 효율적인 어노테이션 전략 연구에 중요한 기초가 될 것으로 보인다.
