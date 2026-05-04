# Mask3D: Mask Transformer for 3D Semantic Instance Segmentation

Jonas Schult, Francis Engelmann, Alexander Hermans, Or Litany, Siyu Tang, Bastian Leibe (2023)

## 🧩 Problem to Solve

본 논문은 3D 장면의 semantic instance segmentation 문제를 해결하고자 한다. 3D instance segmentation은 입력된 3D 포인트 클라우드에서 각 포인트가 어떤 클래스에 속하는지(semantic segmentation)와 동시에, 동일한 클래스 내에서 서로 다른 개별 객체들을 구분하여 이진 마스크(binary foreground mask) 형태로 추출하는(instance segmentation) 작업이다.

기존의 SOTA(State-of-the-Art) 모델들은 주로 객체의 중심점을 예측하는 voting mechanism과 이를 기반으로 포인트들을 묶어주는 geometric clustering 기법에 의존해 왔다. 하지만 이러한 방식은 중심점(center)이나 바운딩 박스(bounding box)와 같이 수동으로 선택된 기하학적 특성에 의존하며, 클러스터링을 위한 하이퍼파라미터(예: 반지름 $\text{radii}$)를 세밀하게 튜닝해야 하는 번거로움이 있다. 또한, 모델이 인스턴스 마스크를 직접 예측하는 것이 아니라 voting 결과라는 프록시(proxy)를 통해 학습되므로, 마스크 자체를 직접적으로 최적화할 수 없다는 한계가 존재한다.

따라서 본 논문의 목표는 수동으로 설계된 voting 및 grouping 메커니즘 없이, Transformer 아키텍처를 활용하여 3D 포인트 클라우드로부터 인스턴스 마스크를 직접적으로 예측하는 end-to-end 모델을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 2D 이미지 세그멘테이션에서 성공을 거둔 Mask2Former와 DETR의 개념을 3D 영역으로 확장하는 것이다. 중심적인 설계 아이디어는 다음과 같다.

1.  **Instance Query 도입**: 각 객체 인스턴스를 하나의 'instance query'로 표현한다. 이 쿼리는 Transformer decoder를 통해 포인트 클라우드의 특징(feature)들을 반복적으로 참조하며, 해당 인스턴스의 기하학적 및 세만틱 정보를 학습한다.
2.  **Direct Mask Prediction**: 별도의 클러스터링 단계 없이, 학습된 instance query와 포인트 특징 간의 유사도(similarity score)를 계산하여 직접적으로 인스턴스 마스크를 생성한다.
3.  **Hand-crafted Components 제거**: 기존 방식의 핵심이었던 center voting, Non-Maximum Suppression(NMS), 그리고 복잡한 grouping 휴리스틱을 완전히 제거하여 모델의 일반성을 높이고 튜닝 비용을 줄였다.

## 📎 Related Works

### 기존 연구 및 한계
- **Bottom-up approaches**: 포인트들을 고차원 특징 공간으로 매핑한 후, 가까운 특징끼리 묶는 방식(contrastive learning 기반)을 사용한다. 하지만 오프라인 클러스터링 단계가 필요하며 end-to-end 학습이 어렵다.
- **Top-down approaches**: Mask R-CNN과 유사하게 먼저 바운딩 박스로 객체를 검출한 뒤 내부 마스크를 생성한다. 앵커 박스(anchor box) 설정에 의존하거나 단일 스케일의 scene descriptor를 사용하여 다양한 크기의 객체를 처리하는 데 한계가 있다.
- **Voting-based approaches**: 최근 SOTA 모델들이 채택한 방식으로, 포인트들이 객체의 중심을 투표(voting)하게 하고 이를 그룹화한다. 성능은 뛰어나나 수동 튜닝이 많이 필요하고 마스크를 직접 최적화하지 못한다.

### Mask3D의 차별점
Mask3D는 위 방식들과 달리 Transformer decoder를 사용하여 쿼리 기반으로 인스턴스를 직접 예측한다. 이는 2D의 Mask2Former와 유사한 접근법으로, 3D instance segmentation 분야에서 처음으로 경쟁력 있는 성능을 보여준 Transformer 기반 모델이라는 점에서 차별성을 갖는다.

## 🛠️ Methodology

### 전체 파이프라인
Mask3D는 **Sparse Feature Backbone $\rightarrow$ Transformer Decoder $\rightarrow$ Mask Module**의 구조로 이루어져 있다.

### 1. Sparse Feature Backbone
MinkowskiEngine 기반의 sparse convolutional U-net을 사용하여 포인트 클라우드 $P$를 처리한다.
- 입력 데이터를 복셀(voxel)화하고, 인코더-디코더 구조를 통해 다중 해상도(multi-scale) 특징 맵 $F_r$을 추출한다.
- $r=0$인 전체 해상도 특징 맵 $F_0$는 최종 마스크 생성에 사용되며, $r \ge 1$인 저해상도 특징 맵들은 Transformer decoder의 cross-attention 입력으로 사용된다.

### 2. Transformer Decoder & Query Refinement
$K$개의 instance query $X \in \mathbb{R}^{K \times D}$를 사용하여 각 인스턴스를 표현한다.
- **Cross-Attention**: 각 쿼리는 backbone에서 추출된 voxel features $F_r$을 참조하여 정보를 업데이트한다.
- **Self-Attention**: 쿼리들끼리 서로 통신하여 중복된 인스턴스 예측을 방지한다.
- **Masked Cross-Attention**: 이전 레이어에서 예측된 중간 마스크 $B$를 활용하여, 쿼리가 자신의 마스크 영역 내부의 복셀에만 집중하도록 강제함으로써 효율성을 높인다. 수식은 다음과 같다.
$$X = \text{softmax}\left(\frac{QK^T}{\sqrt{D}} + B'\right)V, \quad B'_{ij} = -\infty \cdot [B_{ij} = 0]$$

### 3. Mask Module
최종적으로 정제된 쿼리 $X$와 포인트 특징 $F_0$를 이용하여 마스크와 클래스를 예측한다.
- **Binary Mask 생성**: 쿼리를 MLP $f_{\text{mask}}(\cdot)$를 통해 포인트 특징 공간으로 투영한 후, 내적(dot product)을 통해 유사도를 계산한다.
$$B = \{b_{i,j} = [\sigma(F_0 f_{\text{mask}}(X)^T)_{i,j} > 0.5]\}$$
- **Semantic Class 예측**: 쿼리를 선형 투영 후 softmax를 통해 $C+1$개의 클래스(배경 포함) 중 하나로 분류한다.

### 4. 학습 절차 및 손실 함수
예측된 마스크와 정답(GT) 마스크 간의 일대일 대응을 찾기 위해 **Bipartite Graph Matching (Hungarian Method)**을 사용한다. 매칭 비용 $C(k, \hat{k})$는 Dice loss, BCE loss, CE loss의 가중치 합으로 계산된다.
최종 마스크 손실 함수는 다음과 같다.
$$\mathcal{L}_{\text{mask}} = \lambda_{\text{BCE}} \mathcal{L}_{\text{BCE}} + \lambda_{\text{dice}} \mathcal{L}_{\text{dice}}$$
여기에 클래스 분류를 위한 $\mathcal{L}_{\text{CE}}$가 추가되어 전체 네트워크를 최적화한다.

## 📊 Results

### 실험 설정
- **데이터셋**: ScanNet v2, ScanNet200, S3DIS, STPLS3D의 4가지 벤치마크 사용.
- **지표**: mean Average Precision (mAP)을 주요 지표로 사용하며, S3DIS에서는 mPrec/mRec (IoU 50%)도 측정한다.
- **기준선(Baseline)**: SoftGroup, HAIS, PointGroup 등 최신 voting 및 grouping 기반 모델들과 비교한다.

### 정량적 결과
Mask3D는 모든 데이터셋에서 기존 SOTA를 크게 상회하는 성능을 기록하였다.
- **ScanNet test**: mAP 기준 기존 대비 $+6.2$ 상승.
- **S3DIS 6-fold**: mAP 기준 $+10.1$ 상승.
- **STPLS3D**: mAP 기준 $+11.2$ 상승.
- **ScanNet200**: mAP 기준 $+12.4$ 상승.

특히, 클래스 불균형이 심한 ScanNet200에서도 꼬리 부분(tail) 클래스들의 성능이 크게 향상된 것을 확인할 수 있다.

### 정성적 결과 및 분석
- **비정형 객체 처리**: center-voting 기반의 SoftGroup은 U자형 테이블과 같이 비볼록(non-convex)한 형태의 객체 중심점을 예측하는 데 어려움을 겪지만, Mask3D는 기하학적 제약 없이 마스크를 직접 예측하므로 이를 정확하게 분할한다.
- **밀집 객체 분리**: 여러 개의 의자가 밀집해 있는 경우, 기존 방식은 중심점이 겹쳐 하나의 큰 인스턴스로 오인하는 경향이 있으나 Mask3D는 이를 개별적으로 잘 분리한다.

## 🧠 Insights & Discussion

### 강점
Mask3D는 domain-agnostic한 Transformer 구성 요소를 사용하여 3D instance segmentation을 성공적으로 수행하였다. 특히 수동 튜닝이 필요한 voting/grouping 과정을 없앴음에도 불구하고 성능이 비약적으로 향상되었다는 점은, 3D 영역에서도 쿼리 기반의 직접적인 마스크 예측 방식이 매우 유효함을 시사한다.

### 한계 및 해결책
- **Merged Instances**: Transformer의 attention 메커니즘이 전체 포인트 클라우드를 참조할 수 있기 때문에, 세만틱과 기하학적 특성이 매우 유사한 두 객체가 멀리 떨어져 있음에도 불구하고 하나의 인스턴스로 묶이는 경우가 발생한다.
- **DBSCAN 적용**: 이를 해결하기 위해 본 논문은 후처리 단계에서 DBSCAN 클러스터링을 적용하여 공간적으로 분리된 마스크들을 개별 인스턴스로 쪼개는 방법을 제안하였으며, 이를 통해 성능을 추가로 개선하였다.

### 비판적 해석
본 모델은 MinkowskiEngine 기반의 강력한 sparse backbone에 크게 의존하고 있다. 하지만 보조 실험(Tab. V)을 통해 다른 backbone(StratifiedFormer 등)에서도 동작함을 보였으며, 모델 크기를 줄인 버전에서도 성능 하락이 적음을 보여 아키텍처 자체의 견고함을 입증하였다.

## 📌 TL;DR

Mask3D는 3D semantic instance segmentation 분야에 Transformer 기반의 쿼리-마스크 예측 구조를 처음으로 도입하여 SOTA를 달성한 연구이다. 기존의 복잡한 center-voting 및 geometric grouping 과정을 완전히 제거하고, instance query를 통해 마스크를 직접 예측함으로써 모델의 단순성과 성능을 동시에 잡았다. 이 연구는 향후 3D scene understanding 연구가 수동 설계된 휴리스틱에서 벗어나 end-to-end 학습 가능한 Transformer 구조로 전환되는 중요한 전환점이 될 가능성이 높다.