# CompetitorFormer: Competitor Transformer for 3D Instance Segmentation

Duanchu Wang, Jing Liu, Haoran Gong, Yinghui Quan, Di Wang (2025)

## 🧩 Problem to Solve

본 논문은 Transformer 기반의 3D 인스턴스 분할(Instance Segmentation) 모델에서 발생하는 **inter-query competition(쿼리 간 경쟁)** 문제를 해결하고자 한다. 현재의 Transformer 기반 방법론들은 고정된 수의 인스턴스 쿼리(Instance Query)를 사용하며, 일반적으로 이 쿼리의 수는 실제 씬(Scene)에 존재하는 인스턴스의 수보다 훨씬 많게 설정된다.

이로 인해 여러 개의 쿼리가 동일한 인스턴스를 예측하는 현상이 발생하며, 최종적으로는 이분 매칭(Bipartite Matching)을 통해 단 하나의 쿼리만이 최적화된다. 특히 디코더의 초기 계층에서 쿼리들 간의 예측 점수 차이가 크지 않아, 지배적인(dominant) 쿼리가 빠르게 자신을 구분해내지 못하는 문제가 발생한다. 이러한 경쟁 상태는 모델의 학습 수렴 속도를 늦추고 최종적인 분할 정확도를 저하시키는 주요 원인이 된다. 따라서 본 연구의 목표는 쿼리 간의 공간적, 경쟁적 관계 및 시맨틱 정보를 활용하여 경쟁을 완화하고 지배적인 쿼리가 빠르게 등장하도록 돕는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 쿼리들 사이의 경쟁 상태를 명시적으로 모델링하여, 정답에 가까운 쿼리는 강화하고 중복된 쿼리는 억제하는 **competition-oriented designs**를 설계하는 것이다. 이를 위해 다음과 같은 세 가지 핵심 모듈을 제안하며, 이를 통합하여 **CompetitorFormer**라 명명하였다.

1. **Query Competition Layer (QCL)**: 쿼리 간의 리더(leader)와 래가드(laggard) 관계를 정의하고, 정적 임베딩을 통해 매칭된 쿼리의 자신감은 높이고 매칭되지 않은 쿼리는 억제한다.
2. **Relative Relationship Encoding (RRE)**: 쿼리 간의 상대적 경쟁 상태를 양자화하여 관계 인코딩 테이블을 구축하고, 이를 self-attention의 bias로 추가하여 쿼리 간의 관계를 정교화한다.
3. **Rank Cross Attention (RCA)**: Cross-attention 과정에서 쿼리 간의 유사도 차이를 정규화하여, 가장 유사도가 높은 지배적 쿼리가 특징을 더 많이 흡수하도록 유도하고 나머지 쿼리와의 격차를 벌린다.

## 📎 Related Works

### 3D 인스턴스 분할 연구

기존 연구는 크게 세 가지 방향으로 분류된다.

- **Proposal-based**: 3D 바운딩 박스를 먼저 예측한 후 세부 분할을 수행하는 하향식(top-down) 방식이다.
- **Grouping-based**: 포인트별 임베딩을 학습하여 클러스터링하는 상향식(bottom-up) 방식이다.
- **Transformer-based**: 쿼리를 통해 인스턴스 마스크를 직접 예측하는 엔드투엔드 파이프라인으로, 현재 SOTA 성능을 보이고 있다. 하지만 느린 수렴 속도와 쿼리 간 경쟁 문제는 여전히 과제로 남아있다.

### 경쟁 메커니즘 및 Attention 클러스터링

최근 2D 검출 모델인 EASE-DETR에서 쿼리 간의 경쟁이 성능에 영향을 미친다는 점이 밝혀졌다. 또한, Cross-attention을 클러스터링 알고리즘으로 해석하여 최적화하려는 시도(CMT-deeplab, KMax-deeplab 등)가 있었으나, 본 논문은 3D 공간의 복잡성을 고려하여 공간 정보뿐만 아니라 시맨틱 및 상대적 랭킹 관계를 통합하여 경쟁을 완화한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 파이프라인

시스템은 크게 **Sparse 3D U-Net** $\rightarrow$ **Flexible Pooling** $\rightarrow$ **Query Decoder** 순으로 구성된다.

- **Sparse 3D U-Net**: 입력 포인트 클라우드를 복셀화하여 포인트별 특징 $P' \in \mathbb{R}^{N \times C}$를 추출한다.
- **Flexible Pooling**: 슈퍼포인트(superpoint) 또는 복셀(voxel) 기반의 평균 풀링을 통해 특징 $F \in \mathbb{R}^{M \times C}$를 생성한다.
- **Query Decoder**: 초기 쿼리 $Q^0$와 특징 $F$를 입력받아 분류 점수 $P_{cls}^l$, IoU 점수 $S_{IoU}^l$, 인스턴스 마스크 $M_{Mask}^l$을 예측한다.

### 쿼리 경쟁 상태 준비 (Preparation)

각 디코더 계층에서 쿼리 $i$의 경쟁 점수 $k_i^l$를 다음과 같이 정의한다.
$$k_i^l = \max(p_i^l) \cdot s_i^l$$
두 쿼리 $i, j$ 사이의 상대적 차이 $C_{Score}^l(i, j) = k_i^l - k_j^l$를 통해 리더/래가드 관계 $C_{Rank}^l(i, j)$를 결정한다.
$$C_{Rank}^l(i, j) = \begin{cases} +1 & \text{if } C_{Score}^l(i, j) \ge 0 \\ -1 & \text{if } C_{Score}^l(i, j) < 0 \end{cases}$$
또한, 예측된 마스크 간의 IoU를 계산하여 $C_{IoU}^l(i, j)$를 구함으로써 경쟁의 강도를 측정한다.

### Query Competition Layer (QCL)

QCL은 각 디코더 계층 이전에 위치하며, 쿼리의 분류 자신감을 조정한다.

1. 각 쿼리에 대해 가장 강력한 경쟁자 집합 $B^l$을 $C_{IoU}$ 기반으로 선정한다.
2. $C_{Rank}$를 이용해 리더 쿼리 목록 $I_{leader}^l$과 래가드 쿼리 목록 $I_{laggard}^l$을 구성한다.
3. 두 종류의 정적 임베딩 $E_{Le}^l$ (리더용)과 $E_{La}^l$ (래가드용)을 사용하여 경쟁 인식 임베딩 $E_{fuse}^l$를 생성하고, 이를 기존 쿼리 $Q^{l-1}$와 결합하여 업데이트한다.
$$E_{fuse}^l = \text{MLP}(\hat{E}_{La}^l \| \hat{E}_{Le}^l)$$
$$\hat{Q}^l = \text{MLP}(Q^{l-1} \| E_{fuse}^l)$$

### Relative Relationship Encoding (RRE)

Self-attention 내에서 쿼리 간의 상대적 관계를 인코딩한다.

1. 상대적 경쟁 상태 $R_{state}^l = C_{Rank}^l \cdot C_{IoU}^l$를 정의하고, 이를 양자화하여 이산적인 정수 $\hat{R}_{state}^l$로 변환한다.
2. 이 값을 인덱스로 사용하여 관계 인코딩 테이블 $T^l$에서 임베딩 $w_{rel}$을 추출한다.
3. 추출된 $w_{rel}$을 쿼리 벡터 $v_q$ 및 키 벡터 $v_k$와 내적하여 관계 바이어스(relationship bias)를 계산하고, 이를 self-attention 가중치에 더한다.
$$\text{relbias}_{i,j} = w_{rel}^{i,j} \cdot v_q^i + w_{rel}^{i,j} \cdot v_k^j$$

### Rank Cross Attention (RCA)

기존의 Cross-attention은 $\text{softmax}(\text{Q}\text{F}^T)$를 사용하지만, RCA는 지배적 쿼리가 특징을 독점하도록 설계되었다.

1. 쿼리와 특징 간의 내적 유사도 $X = \text{Q}\text{F}^T$를 계산한다.
2. $X$를 다음과 같이 정규화하여 최댓값은 유지하되 나머지 값들은 상대적으로 감소시킨 $X_{norm}$을 생성한다.
$$X_{norm} = \frac{X - \min_N X}{\max_N X - \min_N X}$$
3. 최종 유사도 $\hat{Z}$를 계산할 때 $X$와 $X_{norm}$의 곱을 사용하여 softmax를 적용한다.
$$\hat{Z} = \text{softmax}_M(X \cdot X_{norm})$$

## 📊 Results

### 실험 설정

- **데이터셋**: ScanNetv2, ScanNet200, S3DIS, STPLS3D.
- **지표**: mAP, $\text{mAP}_{50}$, $\text{mAP}_{25}$.
- **베이스라인**: SPFormer, Mask3D, MAFT, OneFormer3D.

### 주요 결과

- **ScanNetv2**: SPFormer에 CompetitorFormer를 적용한 C-SPFormer는 hidden test set에서 mAP가 +3.1, $\text{mAP}_{50}$이 +3.0 상승하였으며, 리더보드 Top-2를 기록하였다.
- **S3DIS**: OneFormer3D와 결합하여 mAP +0.8, $\text{mAP}_{50}$ +0.9의 향상을 보이며 SOTA를 달성하였다.
- **ScanNet200**: SOTA 대비 mAP 4.9%, $\text{mAP}_{50}$ 2.2% 향상이라는 매우 유의미한 결과를 얻었다.
- **STPLS3D**: 모든 기존 방법론을 앞서며 mAP +0.2, $\text{mAP}_{50}$ +0.1 향상을 보였다.

### 소거 연구 (Ablation Study)

SPFormer를 기준으로 각 모듈의 기여도를 분석한 결과:

- **QCL**: 매칭된 쿼리의 분류 점수는 높이고, 매칭되지 않은 쿼리의 점수는 낮추는 효과가 확인되었다 (mAP +1.3).
- **RRE**: $\text{mAP}_{50}$ 향상에 특히 기여하며, QCL과 파라미터를 공유하더라도 추가적인 성능 이득을 제공한다 (mAP +1.1).
- **RCA**: 매칭된 쿼리와 정답 마스크 간의 IoU를 높이고, 매칭되지 않은 쿼리를 정답에서 멀어지게 하여 경쟁을 완화한다 (mAP +1.0).

## 🧠 Insights & Discussion

### 강점 및 해석

본 연구는 Transformer 기반 3D 인스턴스 분할에서 단순히 쿼리 수를 늘리는 것이 아니라, **쿼리 간의 상호작용(경쟁)**을 명시적으로 제어함으로써 성능을 높였다는 점에 큰 의의가 있다. 특히, 제안된 모듈들이 'plug-and-play' 방식으로 설계되어 다양한 기존 SOTA 프레임워크(SPFormer, Mask3D 등)에 쉽게 통합될 수 있으며, 일관된 성능 향상을 보였다는 점이 인상적이다.

### 한계 및 향후 과제

- **쿼리 수의 고정**: QCL 모듈은 쿼리의 수가 일정하게 유지되는 구조에서만 작동한다. 따라서 쿼리 선택 메커니즘이 동적인 OneFormer3D와 같은 모델에서는 QCL의 통합이 어려워 성능 향상 폭이 제한적이다.
- **적용 범위**: 현재는 인스턴스 분할에 집중되어 있으나, 3D 객체 검출(Object Detection)이나 파놉틱 분할(Panoptic Segmentation)로의 확장 가능성이 열려 있다.

## 📌 TL;DR

본 논문은 3D 인스턴스 분할 Transformer 모델에서 발생하는 **inter-query competition** 문제를 해결하기 위해 **CompetitorFormer**를 제안한다. QCL(분류 점수 조정), RRE(관계 바이어스 추가), RCA(지배적 쿼리 강화)의 세 가지 모듈을 통해 중복 쿼리를 억제하고 최적의 쿼리가 빠르게 수렴하도록 유도한다. 실험 결과, ScanNetv2, ScanNet200 등 주요 데이터셋에서 기존 SOTA 모델들의 성능을 상회하는 결과를 얻었으며, 이는 쿼리 간의 경쟁 상태 모델링이 3D 장면 이해에 매우 중요하다는 것을 시사한다.
