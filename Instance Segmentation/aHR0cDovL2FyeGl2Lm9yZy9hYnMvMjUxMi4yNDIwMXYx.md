# BATISNet: Instance Segmentation of Tooth Point Clouds with Boundary Awareness

Yating Cai, Yanghui Xu, Zehua Hu, Jiazhou Chen, Jing Huang (2025)

## 🧩 Problem to Solve

치아 포인트 클라우드의 정확한 분할(Segmentation)은 임상 진단 보조 및 치료 계획 수립에 있어 매우 중요하다. 기존의 많은 연구들은 치아 분할을 **Semantic Segmentation** 문제로 접근하여, 서로 다른 유형의 치아 간의 시맨틱 특징을 추출하는 데 집중하였다.

그러나 실제 임상 환경에서는 다음과 같은 문제들로 인해 기존 방식이 한계를 보인다.
- **치아의 밀집 구조**: 치아들이 매우 조밀하게 배치되어 있어 인접한 치아 간의 경계가 불분명하다.
- **복잡한 임상 사례**: 치아 결손(Missing teeth), 부정교합 및 치아 위치 이상(Malposed teeth)과 같은 다양한 변수가 존재한다.
- **경계 모호성**: 스캔 해상도의 한계와 치아 간의 작은 틈으로 인해 정확한 경계 획정(Delineation)이 어렵다.

따라서 본 논문의 목표는 이러한 복잡한 임상 시나리오에서도 강건하고 정확하게 개별 치아를 분리할 수 있는 **Instance Segmentation** 네트워크인 BATISNet을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 치아 분할을 단순한 클래스 분류(Semantic)가 아닌, 개별 객체를 식별하는 **Instance Segmentation** 작업으로 재정의하는 것이다.

중심적인 설계 포인트는 다음과 같다.
1. **Boundary-Aware Instance Network**: PointMLP 기반의 효율적인 특징 추출기와 인스턴스 마스크 임베딩 메커니즘을 결합하여, 치아의 개수나 배열이 불규칙한 상황에서도 개별 치아를 정확히 분리한다.
2. **Boundary-Aware Loss**: 인스턴스 간의 경계 영역에 집중하여 감독하는 손실 함수를 설계함으로써 치아 간의 유착(Adhesion) 문제와 경계 모호성을 효과적으로 해결한다.
3. **Proposal-free Approach**: 기존의 인스턴스 분할 방식이 의존하던 Bounding Box나 Centroid 같은 중간 제안(Proposal) 과정 없이, 쿼리 기반의 임베딩을 통해 직접 마스크를 예측한다.

## 📎 Related Works

### 3D Tooth Semantic Segmentation
기존 연구들은 곡률(Curvature)이나 윤곽선(Contour lines) 같은 수작업 특징 설계에 의존하거나, 최근에는 GNN(TeethGNN) 및 Transformer(TSegFormer) 기반의 딥러닝 방식을 사용한다. 하지만 이들은 미리 정의된 16종의 치아 카테고리 레이블에 의존하므로, 치아가 결손되었거나 위치가 비정상적인 경우 성능이 급격히 저하되는 한계가 있다.

### 3D Tooth Instance Segmentation
인스턴스 분할 방식은 가변적인 치아 개수를 처리할 수 있어 더 유연하다. 하지만 TSegNet과 같은 기존 방식들은 먼저 치아의 중심점(Centroid)이나 Bounding Box를 예측한 뒤 그 내부를 분할하는 2단계 방식을 사용한다. 이는 전역적인 문맥 정보(Global Context)가 부족하여 강건성이 떨어진다는 단점이 있다.

### 3D Tooth Boundary Detection
경계 검출은 인접 객체를 구분하는 데 필수적이다. 기존의 곡률 기반 방식은 곡률이 높은 영역에 과도하게 집중하여, 곡률이 낮은 영역의 경계를 놓치는 경향이 있다. 본 논문은 이를 해결하기 위해 end-to-end 최적화가 가능한 경량화된 경계 손실 함수를 제안한다.

## 🛠️ Methodology

### 1. Feature Extraction Backbone
입력으로 포인트 좌표와 법선 벡터(Normal vectors)를 사용하며, PointMLP를 기반으로 한 **U-Net 구조의 Encoder-Decoder**를 채택하였다.

- **Encoder**: FPS(Farthest Point Sampling)로 다운샘플링을 수행하고, K-NN을 통해 이웃 특징을 추출한다. 특히 기하학적 어댑티브 표현 능력을 높이기 위해 다음과 같은 **Geometric Affine Transformation** 모듈을 적용한다.
$$\hat{F} = \alpha \cdot \frac{F - \mu}{\sigma + \epsilon} + \beta$$
여기서 $\mu$와 $\sigma$는 K-최근접 이웃 포인트들의 평균과 표준편차이며, $\alpha, \beta$는 학습 가능한 파라미터이다.
- **Decoder**: 업샘플링과 보간법을 통해 해상도를 회복하고, Encoder의 특징을 Skip Connection으로 통합한다.
- **Global-Local Feature Aggregation**: 전역 정보와 지역 정보를 동시에 활용하기 위해, 모든 Encoder 레이어의 특징을 결합한 뒤 Max Pooling을 통해 전역 특징 $F_{global}$을 추출하고, 이를 최종 Decoder 출력 $F_{dec}$와 결합한다.
$$F_{bb} = \text{Cat}\left(F_{dec}, \text{Repeat}\left(\text{MaxPooling}(\text{Cat}(F_{enc}^l))\right)\right)$$

### 2. Instance Segmentation Module
Mask2Former와 OneFormer에서 영감을 받아, 치아 분할을 이진 마스크 예측과 마스크 분류 문제로 정의하였다.

- **Query-based Embedding**: 학습 가능한 치아 인스턴스 쿼리 $Q^{ins} \in \mathbb{R}^{M \times C_{ins}}$와 백본에서 추출된 특징 $F_{bb}$를 사용하여 인스턴스 마스크 임베딩 $E^m \in \mathbb{R}^{M \times N}$을 생성한다.
- **Three-branch Architecture**:
    1. **Mask Prediction**: 포인트 임베딩 $E^{pp}$와 마스크 임베딩 $E^m$을 곱하여 이진 마스크 맵을 생성한다.
    2. **Category Prediction**: 각 마스크의 치아 유형(클래스)을 분류한다.
    3. **Confidence Prediction**: 해당 마스크의 신뢰도 점수를 예측한다.
- 최종적으로 Non-Maximum Suppression(NMS)을 통해 최적의 마스크를 선택한다.

### 3. Boundary-Aware Loss Function
인스턴스 간의 유착을 방지하기 위해 경계 포인트 집합 $B$에 대해서만 적용되는 **Instance Boundary Loss ($L_{ibl}$)**를 도입하였다. 이는 Focal Loss의 형태를 띠며, 분류하기 어려운 경계 지점에 높은 가중치를 부여한다.
$$L_{ibl} = -\frac{1}{|B|} \sum_{i \in B} \sum_{c=1}^{C} (1 - p_{i,c})^\gamma \cdot y_{i,c} \cdot \log(p_{i,c})$$
전체 손실 함수는 다음과 같이 구성된다.
$$L = \lambda_{cls} L_{cls} + \lambda_{mask} L_{mask} + \lambda_{obj} L_{obj} + \lambda_{ibl} L_{ibl}$$

### 4. Post-processing (Graph Cut)
예측 결과의 일관성을 높이고 고립된 오류 예측을 제거하기 위해 Graph Cut 최적화를 수행한다. 전체 에너지 함수 $E(L)$을 최소화하는 방식으로 레이블을 정교화한다.
$$E(L) = \sum_{p \in P} U_p(L_p) + \lambda \cdot \sum_{(p,q) \in E} w_{pq} \cdot I[L_p \neq L_q]$$
여기서 $U_p$는 데이터 항(Data term)으로 네트워크의 예측 확률을 사용하며, $w_{pq}$는 평활화 항(Smoothness term)으로 포인트 간의 거리 $d_{pq}$와 법선 벡터의 코사인 유사도 $nsim_{pq}$를 기반으로 계산된다.

## 📊 Results

### 실험 설정
- **데이터셋**: MICCAI Challenge 데이터셋 (학습 254개, 테스트 125개 모델). 모든 모델은 16,000개 포인트로 다운샘플링되었다.
- **평가 지표**: Accuracy (ACC), Intersection over Union (IoU), Dice coefficient, 그리고 인스턴스 수준의 성능을 측정하는 Average Precision (AP) 및 mAP를 사용하였다.

### 주요 결과
- **SOTA 비교**: BATISNet은 OA, mACC, mIoU 모든 지표에서 기존 SOTA 모델들을 압도하였다. 특히 Baseline인 PointMLP 대비 mIoU가 4.13% 향상되었다.
- **인스턴스 정밀도**: mAP 지표에서 THISNet과 같은 다른 인스턴스 분할 방식보다 최소 3% 이상의 성능 향상을 보였으며, 특히 IoU 임계값이 높은(엄격한) 조건에서 그 격차가 더 커졌다.
- **복잡한 사례 처리**: 치아 결손, 매복치(Partially erupted teeth), 사랑니(Wisdom teeth)가 포함된 데이터셋에서도 매우 강건한 성능을 보였다. 이는 Semantic 방식이 개별 치아 내부에서 일관되지 않은 레이블을 생성하는 것과 대조적이다.

### Ablation Study
- **인스턴스 모듈의 중요성**: 표준 MLP 기반의 Semantic head로 교체했을 때 모든 지표가 크게 하락하여, 인스턴스 임베딩의 필요성이 입증되었다.
- **Boundary Loss & Graph Cut**: Boundary Loss는 mAcc 향상에 기여하며, Graph Cut은 OA, mIoU, mAP 전반의 수치를 끌어올리는 효과가 확인되었다.

## 🧠 Insights & Discussion

### 강점
본 논문은 치아 분할을 단순 분류가 아닌 인스턴스 분리 문제로 접근함으로써, 정형화되지 않은 치아 구조(결손, 부정교합)에 대해 매우 높은 유연성을 확보하였다. 특히 경계 전용 손실 함수($L_{ibl}$)와 Graph Cut 후처리를 통해 의료 데이터에서 가장 치명적인 문제인 '인접 치아 간 유착' 문제를 효과적으로 해결한 점이 돋보인다.

### 한계 및 향후 과제
현재 모델은 입력 데이터를 16,000개 포인트로 단순화하여 처리한다. 하지만 이러한 단순화 과정에서 미세한 기하학적 디테일이 손실될 수 있다. 저자들은 향후 연구에서 백본의 연산 복잡도를 낮추어, 고해상도 포인트 클라우드를 직접 처리할 수 있는 방향으로 개선해야 한다고 언급하였다.

### 비판적 해석
본 연구는 Proposal-free 방식의 인스턴스 분할을 치아 도메인에 성공적으로 적용하였다. 하지만 사용된 데이터셋의 사랑니 비율이 약 5%로 매우 낮아, 희귀 케이스에 대한 일반화 성능이 충분히 검증되었는지에 대해서는 추가적인 대규모 데이터 검증이 필요할 것으로 보인다.

## 📌 TL;DR

**BATISNet**은 치아 포인트 클라우드 분할을 **Semantic $\rightarrow$ Instance Segmentation**으로 전환하여, 치아 결손이나 부정교합 같은 복잡한 임상 사례에서도 개별 치아를 정확히 분리해내는 네트워크이다. **PointMLP 기반 U-Net 백본**, **쿼리 기반 마스크 임베딩**, 그리고 **경계 특화 손실 함수($L_{ibl}$)**를 통해 인접 치아 간의 유착 문제를 해결하였으며, SOTA 수준의 mAP 성능을 달성하였다. 이 연구는 향후 정밀한 디지털 치과 진단 및 치료 계획 자동화 시스템의 핵심 모듈로 활용될 가능성이 매우 높다.