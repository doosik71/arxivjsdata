# Hierarchical Aggregation for 3D Instance Segmentation

Shaoyu Chen, Jiemin Fang, Qian Zhang, Wenyu Liu, Xinggang Wang (2021)

## 🧩 Problem to Solve

3D 포인트 클라우드(Point Cloud)에서의 인스턴스 분할(Instance Segmentation)은 3D 장면 이해의 핵심적인 과제이다. 기존의 bottom-up 방식(클러스터링 기반 방법)은 다음과 같은 이유로 인스턴스를 정확하게 분리하는 데 어려움을 겪는다. 첫째, 포인트 클라우드는 데이터의 양이 매우 방대하며, 장면마다 포함된 인스턴스의 개수와 크기가 매우 다양하다. 둘째, 개별 포인트가 가진 특징(좌표 및 색상)이 매우 약하기 때문에, 포인트 수준의 특징과 인스턴스 정체성 사이에는 큰 의미적 격차(Semantic Gap)가 존재한다.

이러한 문제로 인해 기존 방법들은 하나의 객체가 여러 개로 쪼개지는 과분할(Over-segmentation)이나, 서로 다른 객체가 하나로 묶이는 과소분할(Under-segmentation) 문제에 취약하다. 본 논문의 목표는 포인트와 포인트 세트 간의 공간적 관계를 최대한 활용하여, 이러한 분할 오류를 해결하고 연산 효율성을 높인 새로운 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **계층적 집계(Hierarchical Aggregation)** 전략이다. 이는 한 번에 인스턴스를 생성하는 대신, 단계적인 프로세스를 통해 점진적으로 인스턴스 제안(Proposal)을 생성하는 방식이다.

1.  **Point Aggregation**: 낮은 대역폭(Bandwidth)을 사용하여 포인트를 예비 세트로 묶어 과분할 가능성을 낮춘다.
2.  **Set Aggregation**: 동적 대역폭을 적용하여 작은 조각(Fragment)들을 주요 인스턴스(Primary Instance)에 병합함으로써 완전한 인스턴스를 형성한다.
3.  **Intra-instance Prediction**: 생성된 인스턴스 내부에서 노이즈 포인트를 필터링하고 마스크 품질 점수를 예측하는 서브 네트워크를 통해 정밀도를 높인다.

결과적으로 본 연구는 NMS(Non-Maximum Suppression) 과정이 필요 없는 간결한 단일 전방향 추론(Single-forward inference) 파이프라인을 구축하여 높은 정확도와 매우 빠른 추론 속도를 동시에 달성하였다.

## 📎 Related Works

3D 인스턴스 분할 연구는 크게 두 가지 접근 방식으로 나뉜다. 

- **Proposal-based methods**: 객체 제안(Proposal)을 먼저 생성하고 그 내부에서 마스크를 예측한다. 2D의 Mask R-CNN과 유사한 구조를 가지며, 3D-BoNet이나 GICN 등이 이에 해당한다. 이러한 방식은 대개 밀집된 제안을 생성해야 하므로 연산량이 많고 NMS와 같은 후처리 과정이 필수적이다.
- **Clustering-based methods**: 포인트별 레이블을 예측한 후 클러스터링을 통해 인스턴스를 생성한다. PointGroup, MTML 등이 대표적이며, 본 논문의 HAIS 역시 이 패러다임을 따른다. 그러나 기존 클러스터링 방식은 복잡하고 시간이 많이 걸리는 절차를 요구하며, 포인트 수준의 임베딩에만 의존하여 인스턴스 수준의 보정이 어렵다는 한계가 있다.

HAIS는 클러스터링 기반 방식을 채택하되, **Set Aggregation**과 **Intra-instance Prediction**을 도입하여 객체 수준에서의 보정을 수행함으로써 기존 방법론과의 차별성을 갖는다.

## 🛠️ Methodology

### 1. 전체 파이프라인 구조
HAIS는 크게 네 가지 단계로 구성된다: **Point-wise Prediction $\rightarrow$ Point Aggregation $\rightarrow$ Set Aggregation $\rightarrow$ Intra-instance Prediction**.

### 2. Point-wise Prediction Network
입력 포인트 클라우드를 정규 볼륨 그리드로 변환한 후, Submanifold Sparse Convolution 기반의 3D UNet 구조를 사용하여 특징을 추출한다. 추출된 특징($F_{point}$)은 두 개의 브랜치로 나뉜다.
- **Semantic Label Prediction**: MLP와 Softmax를 통해 각 포인트의 세만틱 클래스를 예측한다.
- **Center Shift Vector Prediction**: 포인트에서 해당 인스턴스의 중심까지의 오프셋 벡터 $\Delta x_i \in \mathbb{R}^3$를 예측한다. 학습 시 사용되는 손실 함수 $L_{shift}$는 다음과 같다.
$$L_{shift} = \frac{1}{\sum_{p_i \in P} 1(p_i \in P_{fg})} \cdot \sum_{p_i \in P} L(p_i)$$
$$L(p_i) = w(p_i) \cdot \|\Delta x^{gt}_i - \Delta x^{pred}_i\|_1 \cdot 1(p_i \in P_{fg})$$
여기서 $w(p_i) = \min(\|\Delta x^{gt}_i\|_2, 1)$는 중심에 가까운 포인트의 가중치를 낮추어 학습의 안정성을 높이는 가중치 항이다.

### 3. Point Aggregation
예측된 중심 이동 벡터를 사용하여 포인트의 좌표를 이동시킨다.
$$x^{shift}_i = x^{origin}_i + \Delta x_i$$
이후 동일한 세만틱 레이블을 가지고, 이동된 좌표 간의 거리가 고정된 대역폭 $r_{point}$보다 작은 포인트들을 하나의 독립된 세트로 묶는다. 이 단계에서 생성된 결과물은 여전히 불완전한 상태일 수 있다.

### 4. Set Aggregation
Point Aggregation의 결과는 크게 **주요 인스턴스(Primary Instances)**와 **조각(Fragments)**으로 나뉜다. 조각들을 주요 인스턴스에 병합하기 위해 다음과 같은 조건을 검사한다.
1. 조각 $m$과 동일한 클래스를 가진 주요 인스턴스 $n$ 중 기하학적 중심이 가장 가까워야 한다.
2. 두 중심 간의 거리가 동적 대역폭 $r^{set}$보다 작아야 한다.
$$r^{set} = \max(r_{size}, r_{cls}), \quad r_{size} = \alpha \sqrt{S^{prim}_n}$$
여기서 $S^{prim}_n$은 주요 인스턴스의 크기(포인트 수)이며, 인스턴스가 클수록 더 넓은 범위의 조각을 흡수하도록 설계되었다.

### 5. Intra-instance Prediction Network
병합 과정에서 잘못 포함된 노이즈를 제거하기 위해 인스턴스 내부 특징을 다시 학습한다.
- **Mask Branch**: 이진 마스크를 예측하여 인스턴스의 전경과 배경을 구분한다. $IoU > 0.5$인 샘플만 학습에 사용하여 모호성을 제거하며, 손실 함수 $L_{mask}$를 통해 최적화한다.
- **Certainty Score Branch**: 마스크를 통해 필터링된 전경 특징만을 사용하여 인스턴스의 신뢰도 점수를 예측한다. 이는 GT 마스크와의 $IoU$를 정답으로 하여 $L_{score}$로 학습된다.

### 6. 전체 학습 및 추론
전체 네트워크는 다음과 같은 통합 손실 함수를 통해 End-to-End로 학습된다.
$$L = L_{seg} + L_{shift} + L_{mask} + L_{score}$$
추론 시에는 포인트가 단 하나의 인스턴스에만 할당되므로 중복 예측이 발생하지 않는다. 따라서 NMS 없이 신뢰도 점수만으로 최종 결과를 랭킹하여 출력하는 단일 전방향 추론이 가능하다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: ScanNet v2 (18개 클래스), S3DIS (13개 클래스).
- **지표**: ScanNet v2에서는 $AP_{50}$을, S3DIS에서는 $mCov, mWCov, mPrec, mRec$를 사용하였다.

### 2. 정량적 결과
- **ScanNet v2**: $AP_{50}$ 기준 **69.9%**를 달성하여 벤치마크 1위를 기록하였다. 이는 이전 SOTA 방법(GICN)보다 6.1% 향상된 수치이며, 18개 클래스 중 12개에서 최고 성능을 보였다.
- **S3DIS**: 모든 주요 지표($mCov, mWCov, mPrec, mRec$)에서 기존 방법들을 유의미하게 상회하며 높은 일반화 성능을 입증하였다.

### 3. 추론 속도 분석
HAIS는 후처리가 없는 간결한 구조 덕분에 압도적인 속도를 보여준다. ScanNet v2 검증 세트 전체 추론에 단 128초가 소요되었으며, **프레임당 평균 추론 시간은 410ms**이다. 이는 PointGroup이나 GICN 등 기존 SOTA 방법들보다 훨씬 빠른 속도이다.

## 🧠 Insights & Discussion

본 논문은 Bottom-up 클러스터링 방식의 고질적인 문제인 과분할 문제를 **계층적 집계(Point $\rightarrow$ Set)**라는 단순하지만 강력한 구조로 해결하였다. 특히 포인트 수준의 예측에서 발생하는 오차를 세트 수준의 동적 대역폭 병합으로 보정하고, 최종적으로 인스턴스 내부 네트워크를 통해 정밀도를 다듬는 단계적 접근 방식이 주효했다.

또한, 대다수의 3D 인스턴스 분할 모델이 복잡한 후처리(NMS, Iterative Clustering)에 의존하여 실시간 적용이 어려웠던 반면, HAIS는 구조적 설계를 통해 NMS-free를 구현함으로써 실용성을 극대화하였다. 

다만, 본 논문에서 제시한 $r_{point}$나 $\alpha$와 같은 하이퍼파라미터들이 데이터셋의 특성에 따라 민감하게 작용할 가능성이 있으며, 이에 대한 자동 최적화 방안은 명시되지 않았다. 또한, 학습 단계에서 Set Aggregation을 사용하지 않고 추론 시에만 적용한다는 점이 성능에 구체적으로 어떤 영향을 미치는지에 대한 심층 분석이 더 필요할 것으로 보인다.

## 📌 TL;DR

HAIS는 3D 포인트 클라우드 인스턴스 분할을 위해 **포인트 집계 $\rightarrow$ 세트 집계 $\rightarrow$ 내부 정밀화**로 이어지는 계층적 구조를 제안한다. 이 방법은 과분할 문제를 해결하여 ScanNet v2 벤치마크 1위($AP_{50}: 69.9\%$)를 달성했으며, NMS가 필요 없는 단일 전방향 추론을 통해 프레임당 410ms라는 매우 빠른 속도를 구현하였다. 이 연구는 실시간성이 중요한 자율주행이나 로보틱스 분야의 3D 장면 이해 시스템에 즉시 적용될 가능성이 매우 높다.