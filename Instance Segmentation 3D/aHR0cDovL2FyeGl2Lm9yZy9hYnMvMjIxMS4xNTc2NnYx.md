# Superpoint Transformer for 3D Scene Instance Segmentation

Jiahao Sun, Chunmei Qing, Junpeng Tan, Xiangmin Xu (2023)

## 🧩 Problem to Solve

본 논문은 3D 장면의 인스턴스 분할(Instance Segmentation) 문제를 해결하고자 한다. 3D 인스턴스 분할은 희소한 포인트 클라우드(point clouds)에서 개별 객체를 탐지하고 각 객체에 대한 정밀한 마스크(mask)를 생성해야 하는 도전적인 과제이다.

기존의 접근 방식은 크게 두 가지로 나뉘며, 각각 다음과 같은 한계를 가진다:
1. **Proposal-based 방법 (Top-down):** 3D 바운딩 박스(bounding box)를 먼저 생성한 후 내부의 마스크를 예측한다. 그러나 3D 공간에서는 바운딩 박스의 자유도(DoF)가 높아 정밀한 예측이 어렵고, 낮은 품질의 제안(proposal)이 전체 성능을 저하시키는 문제가 있다.
2. **Grouping-based 방법 (Bottom-up):** 포인트별 시맨틱 라벨과 중심점 오프셋(offset)을 학습하여 포인트들을 그룹화한다. 하지만 시맨틱 분할 결과에 의존하므로 초기 예측 오류가 전파될 가능성이 크고, 네트워크 학습과 독립적인 중간 집계(aggregation) 단계가 필요하여 연산 시간이 오래 걸린다는 단점이 있다.

따라서 본 논문의 목표는 이러한 Top-down과 Bottom-up 방식의 단점을 극복하고, 중간 집계 단계 없이 end-to-end로 학습 가능한 효율적인 3D 인스턴스 분할 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문은 **SPFormer**라는 새로운 프레임워크를 제안하며, 핵심 아이디어는 다음과 같다:

- **하이브리드 파이프라인 설계:** Bottom-up 방식의 슈퍼포인트(superpoint) 생성과 Top-down 방식의 쿼리 기반 인스턴스 제안을 결합하였다.
- **Superpoint Transformer 도입:** 포인트 클라우드를 슈퍼포인트로 그룹화하여 연산량을 줄이고, 학습 가능한 쿼리 벡터(query vectors)가 Superpoint Cross-Attention 메커니즘을 통해 인스턴스 정보를 캡처하도록 설계하였다.
- **End-to-End 학습 구조:** 슈퍼포인트 마스크 기반의 이분 매칭(bipartite matching)을 통해 복잡한 중간 집계 단계와 NMS(Non-Maximum Suppression) 같은 후처리 과정 없이 학습 및 추론이 가능하게 하였다.

## 📎 Related Works

논문에서는 기존 연구를 다음과 같이 분류하고 차별점을 제시한다:
- **Proposal-based Methods:** 3D-BoNet, GICN 등이 있으며, 주로 3D 바운딩 박스 생성에 의존한다. 저자들은 포인트 클라우드의 특성상 기하학적 중심을 찾기 어렵고 바운딩 박스 예측의 불확실성이 크다는 점을 지적한다.
- **Grouping-based Methods:** PointGroup, HAIS, SoftGroup 등이 있으며, 포인트 간의 임베딩이나 오프셋을 이용해 그룹화한다. 이러한 방식은 학습되지 않은 중간 집계 단계가 필수적이며, 시맨틱 예측의 정확도에 너무 의존한다는 한계가 있다.
- **Transformer in 2D Instance Segmentation:** Mask2Former와 같은 2D 분야의 성공적인 트랜스포머 적용 사례에서 영감을 얻었다. 다만, 3D 포인트 클라우드에 트랜스포머를 직접 적용하면 계산 복잡도가 너무 높으므로, '슈퍼포인트'라는 중간 표현체를 통해 이를 해결하고자 하였다.

## 🛠️ Methodology

### 전체 시스템 구조
SPFormer는 크게 두 단계로 구성된다: **Bottom-up Grouping Stage**와 **Top-down Proposal Stage**이다.

### 1. Bottom-up Grouping Stage
- **Sparse 3D U-net:** 입력 포인트 클라우드 $P \in \mathbb{R}^{N \times 6}$ (색상 RGB, 좌표 XYZ)로부터 포인트별 특징 $P' \in \mathbb{R}^{N \times C}$를 추출한다.
- **Superpoint Pooling Layer:** 사전에 계산된 슈퍼포인트(geometric regularity를 기반으로 인접한 유사 포인트들의 집합)를 이용하여 포인트별 특징을 평균 풀링(average pooling)한다. 결과적으로 $M$개의 슈퍼포인트 특징 $S \in \mathbb{R}^{M \times C}$가 생성되며, 이는 연산량을 획기적으로 줄이는 가교 역할을 한다.

### 2. Top-down Proposal Stage (Query Decoder)
쿼리 디코더는 **Instance Branch**와 **Mask Branch**로 나뉜다.

- **Mask Branch:** 단순 MLP를 통해 마스크 인식 특징 $S_{mask} \in \mathbb{R}^{M \times D}$를 추출한다.
- **Instance Branch:** 일련의 트랜스포머 디코더 층으로 구성되며, $K$개의 학습 가능한 쿼리 벡터 $Z$를 사용한다.
- **Superpoint Cross-Attention:** 쿼리 벡터가 슈퍼포인트 특징으로부터 인스턴스 정보를 캡처하는 핵심 과정이다. 수식은 다음과 같다:
  $$\hat{Z}^{\ell} = \text{softmax}\left(\frac{QK^T}{\sqrt{D}} + A^{\ell-1}\right)V$$
  여기서 $Q$는 쿼리 벡터의 선형 투영이며, $K, V$는 슈퍼포인트 특징의 투영값이다. $A^{\ell-1}$은 이전 층의 예측 결과에서 임계값 $\tau$를 기준으로 생성된 **Superpoint Attention Mask**이며, 이는 쿼리가 배경이 아닌 전경(foreground) 인스턴스에만 집중하도록 강제한다.

### 3. Prediction Head 및 Iterative Prediction
- **예측 항목:** 쿼리 벡터 $Z^{\ell}$를 통해 각 쿼리의 클래스 확률 $p_i$, IoU 인식 점수 $s_i$, 그리고 슈퍼포인트 마스크 $M^{\ell}$를 예측한다.
- **마스크 생성:** 쿼리 벡터 $Z^{\ell}$와 Mask Branch의 $S_{mask}$를 곱한 후 시그모이드 함수를 통과시켜 생성한다.
- **Iterative Prediction:** 트랜스포머의 느린 수렴 속도를 해결하기 위해, 모든 디코더 층의 출력물을 예측 헤드에 전달하여 학습시킨다. (추론 시에는 마지막 층의 결과만 사용)

### 4. Bipartite Matching 및 Loss Function
- **Bipartite Matching:** 헝가리안 알고리즘(Hungarian algorithm)을 사용하여 예측된 제안(proposal)과 실제 정답(GT) 사이의 최적 매칭을 찾는다. 매칭 비용 $C_{ik}$는 다음과 같이 정의된다:
  $$C_{ik} = -\lambda_{cls} \cdot p_{i, c_k} + \lambda_{mask} \cdot C_{mask_{ik}}$$
  여기서 $C_{mask_{ik}}$는 BCE(Binary Cross-Entropy)와 Dice Loss의 합으로 구성된다.
- **최종 손실 함수:**
  $$L = \beta_{cls} \cdot L_{cls} + \beta_s \cdot L_s + \beta_{mask} \cdot (L_{bce} + L_{dice})$$
  클래스 분류 손실($L_{cls}$), IoU 점수 손실($L_s$), 마스크 손실($L_{bce}, L_{dice}$)의 가중치 합으로 구성된다.

## 📊 Results

### 실험 설정
- **데이터셋:** ScanNetv2 (실내 장면 1613개), S3DIS (6개 영역, 272개 장면).
- **지표:** mAP (mean Average Precision), $AP_{50}$, $AP_{25}$, mPrec, mRec.

### 주요 결과
- **ScanNetv2 성능:** Hidden test set에서 **mAP 54.9%**를 달성하여 기존 SOTA 방법보다 **4.3% 향상**되었다. 특히 기존 방법들이 어려워하던 'counter' 카테고리에서 10% 이상의 큰 성능 향상을 보였다.
- **S3DIS 성능:** $AP_{50}$ 기준 SOTA 결과를 달성하여 일반화 능력을 입증하였다.
- **추론 속도:** ScanNetv2 기준 프레임당 **247ms**의 빠른 속도를 기록하였다. 이는 중간 집계 단계와 NMS 후처리가 없기 때문이다. (Table 4 참조)

### Ablation Study (주요 발견)
- **Superpoint의 중요성:** 슈퍼포인트 풀링 없이 포인트 특징을 직접 디코더에 넣을 경우 성능이 급격히 하락한다. (계산 복잡도 및 softmax 문제)
- **매칭 방식:** 바운딩 박스 기반 매칭보다 마스크 기반 매칭이 mAP 기준 6.4% 더 높은 성능을 보였다.
- **트랜스포머 구조:** Iterative prediction과 Attention mask를 적용했을 때 성능이 유의미하게 상승하였다.

## 🧠 Insights & Discussion

### 강점
SPFormer는 기존의 Bottom-up 방식이 가진 '느린 집계 속도'와 Top-down 방식이 가진 '부정확한 박스 제안'이라는 두 가지 난제를 동시에 해결하였다. 특히 슈퍼포인트를 트랜스포머의 입력값으로 사용함으로써, 3D 데이터의 방대한 양을 효율적으로 압축하면서도 기하학적 특성을 유지한 점이 탁월하다.

### 한계 및 논의사항
- **슈퍼포인트 의존성:** 본 모델은 사전에 계산된 슈퍼포인트에 의존한다. 만약 슈퍼포인트 생성 알고리즘 자체가 잘못된 그룹화를 수행한다면, 이후의 트랜스포머 디코더가 이를 복구하기 어려울 수 있다.
- **쿼리 개수 설정:** 실험 결과 쿼리 벡터의 개수가 너무 적거나 많으면 성능이 떨어진다. 3D 장면의 인스턴스 개수가 2D보다 일반적으로 많기 때문에 $K=800$ 정도에서 성능이 포화되는 경향을 보이는데, 이는 장면의 복잡도에 따라 동적으로 쿼리 수를 조절할 필요가 있음을 시사한다.

## 📌 TL;DR

SPFormer는 3D 포인트 클라우드를 효율적으로 처리하기 위해 **슈퍼포인트(Superpoint)**와 **트랜스포머 쿼리(Query)** 메커니즘을 결합한 end-to-end 인스턴스 분할 모델이다. 중간 집계 단계와 NMS 후처리를 제거하여 추론 속도를 높였으며, ScanNetv2와 S3DIS 벤치마크에서 SOTA 성능을 달성하였다. 이 연구는 3D 장면 이해를 위한 효율적인 포인트 압축 및 쿼리 기반 탐지 프레임워크의 가능성을 제시하였다.