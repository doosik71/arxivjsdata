# PointInst3D: Segmenting 3D Instances by Points

Tong He, Wei Yin, Chunhua Shen, Anton van den Hengel (2022)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 3D 인스턴스 분할(3D Instance Segmentation)에서 기존의 SOTA(State-of-the-art) 방법론들이 의존하고 있는 클러스터링(Clustering) 단계의 한계를 극복하는 것이다.

기존의 클러스터링 기반 방식들은 다음과 같은 몇 가지 치명적인 문제점을 가지고 있다. 첫째, 휴리스틱(Heuristics)이나 그리디 알고리즘(Greedy algorithms)에 의존하는 경향이 있어 데이터 통계 변화에 취약하며 로버스트함이 떨어진다. 둘째, 시맨틱 분할(Semantic segmentation)이나 중심점 오프셋 예측(Centroid offset prediction)과 같은 중간 단계의 결과에 의존하는 '태스크 간 의존성(Inter-task dependencies)'이 존재한다. 이로 인해 중간 단계에서 발생한 오류가 최종 결과까지 누적되는 오류 전파(Error accumulation) 문제가 발생하며, 이는 결과적으로 인스턴스의 단편화(Fragmentation)나 과도한 병합(Merging)으로 이어진다.

따라서 본 논문의 목표는 클러스터링 단계와 태스크 간 의존성을 완전히 제거하고, 포인트별 예측(Per-point prediction) 방식의 단순하고 유연한 Fully-convolutional 3D 인스턴스 분할 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 가장 중심적인 아이디어는 **Optimal Transport(OT)**를 이용한 **동적 타겟 할당(Dynamic Target Assignment)**이다.

단순히 포인트별 예측 방식을 도입하면, 어떤 샘플링 포인트가 어떤 인스턴스 마스크를 예측해야 하는지에 대한 타겟 설정의 모호성(Ambiguity) 문제가 발생한다. 2D 이미지에서는 객체의 중심 영역이 정확한 예측을 제공한다는 'Center Prior'를 사용할 수 있지만, 3D 포인트 클라우드는 데이터가 표면에 분포하고 밀도가 불규칙하여 이러한 정적인 거리 기반 전략을 적용하기 어렵다.

이를 해결하기 위해 저자들은 예측값과 실제 정답(Ground Truth) 사이의 유사도를 기반으로 타겟을 최적으로 할당하는 Optimal Transport 접근법을 제안하였다. 이를 통해 휴리스틱한 튜닝 없이도 각 샘플링 포인트에 가장 적합한 타겟 마스크를 동적으로 할당함으로써 학습의 안정성과 정확도를 높였다.

## 📎 Related Works

### 2D 이미지의 타겟 할당 (Target Assignment in 2D Images)

2D 객체 탐지 분야에서는 앵커 기반(Anchor-based) 방식의 IoU 임계값 설정이나, 앵커 프리(Anchor-free) 방식의 Center Prior를 활용한 샘플 선택 전략이 주로 사용되었다. 최근에는 ATSS와 같이 동적 임계값을 사용하거나, OTA(Optimal Transport Assignment)처럼 타겟 할당 문제를 최적 운송 문제로 정식화하여 해결하려는 시도가 있었다.

### 3D 포인트 클라우드 인스턴스 분할 (Instance Segmentation on 3D Point Cloud)

3D 분야는 데이터의 불규칙성과 희소성으로 인해 주로 Bottom-up 방식의 클러스터링 방법론이 주류를 이루었다.

- **Metric-based pipelines:** 포인트 간의 임베딩 공간에서의 거리를 학습하고 Mean-shift 등으로 클러스터링하는 방식(ASIS 등)이 있으나, 하이퍼파라미터 의존도가 높고 일반화 능력이 떨어진다.
- **Grouping-based pipelines:** PointGroup은 시맨틱 라벨과 중심점 예측을 결합해 클러스터를 생성하며, DyCo3D는 동적 컨볼루션(Dynamic Convolution)을 도입해 마스크를 디코딩한다.
- **한계:** 이러한 방식들은 여전히 클러스터링 단계의 성능에 종속적이며, 시맨틱 분할 결과가 틀릴 경우 이를 복구할 수 없는 구조적 한계가 있다.

## 🛠️ Methodology

### 전체 시스템 구조

PointInst3D는 Sparse Convolution 기반의 UNet 구조를 백본(Backbone)으로 사용하며, 입력으로 좌표(Coordinates)와 특징(Features)을 받는다. 이 네트워크는 인스턴스 마스크를 디코딩하기 위한 마스크 특징 $F_m \in \mathbb{R}^{N \times d'}$를 출력한다. 전체 파이프라인은 클러스터링 과정 없이, 샘플링된 포인트들이 직접 인스턴스 마스크를 예측하는 포인트별 예측 방식으로 동작한다.

### 인스턴스 헤드 및 마스크 예측

Farthest Point Sampling(FPS) 전략을 통해 $K$개의 포인트를 샘플링한다. 각 샘플링 포인트 $k$는 하나의 특정 인스턴스 마스크 또는 배경을 예측하는 책임을 진다. $k$번째 포인트에 의한 예측 마스크 $\hat{M}_k$는 다음과 같이 정의된다.

$$\hat{M}_k = \text{Conv}_{1 \times 1}(F_m \oplus C^k_{rel}, \text{mlp}(f^b_k))$$

여기서 $f^b_k$는 백본에서 추출된 $k$번째 샘플링 포인트의 특징이며, $\text{mlp}(\cdot)$는 이 특징을 통해 동적 컨볼루션의 가중치(Weights)를 생성한다. $F_m \oplus C^k_{rel}$은 공유된 마스크 특징과 $k$번째 포인트 기준의 상대 좌표 임베딩을 결합한 입력값이다.

### Optimal Transport 기반 동적 타겟 할당

훈련 과정에서 각 예측 $\hat{M}_k$에 어떤 정답 마스크 $M_t$를 할당할지 결정하기 위해 최적 운송(Optimal Transport) 문제를 정의한다.

1. **정의:** 공급자(Suppliers)는 실제 인스턴스 마스크 $\{M_t\}_{t=1}^T$와 배경 마스크 $M_{T+1}$이며, 수요자(Demanders)는 샘플링된 포인트들의 예측값 $\{\hat{M}_k\}_{k=1}^K$이다.
2. **비용 함수:** 운송 비용 $C_{tk}$는 예측값과 정답 사이의 Dice Loss로 정의된다.
   $$C_{tk} = \begin{cases} L_{\text{dice}}(M_t, \hat{M}_k) & t \le T \\ L_{\text{dice}}(1-M_t, 1-\hat{M}_k) & t = T+1 \end{cases}$$
3. **최적화:** Sinkhorn-Knopp 알고리즘을 사용하여 다음과 같은 최적 운송 계획 $U^*$를 찾는다.
   $$U^* = \arg \min_{U \in \mathbb{R}^{(T+1) \times K}_+} \sum_{t,k} C_{tk} U_{tk}$$
   (제약 조건: 모든 공급량과 수요량의 합이 일치해야 하며, 각 예측 포인트는 정확히 하나의 타겟을 가져야 함)

### 보조 헤드(Auxiliary Head) 및 학습 절차

OT 기반 할당 시, 학습 초기에는 예측값이 무작위적이므로 모든 예측이 배경 타겟으로 할당되는 Trivial Solution에 빠질 위험이 있다. 이를 방지하기 위해 **보조 인스턴스 헤드(Auxiliary Instance Head)**를 도입한다.

- 보조 헤드는 샘플링 포인트의 원래 인스턴스 라벨에 기반한 정적 타겟(Static Target)으로 학습된다.
- OT 비용 행렬 $C_{tk}$를 계산할 때 보조 헤드의 예측값을 사용함으로써 초기 학습의 안정성을 확보한다.
- 최종 손실 함수는 다음과 같이 보조 손실 $L_a$와 메인 태스크 손실 $L_m$의 합으로 구성된다.
  $$L = w_a \sum_{k=1}^K L_a(M^a_k, \hat{M}^a_k) + \sum_{k=1}^K L_m(M^m_k, \hat{M}^m_k)$$
  이때 $w_a$는 학습이 진행됨에 따라 $0.99$의 비율로 감쇠되어 메인 태스크에 더 집중하게 한다.

## 📊 Results

### 실험 설정

- **데이터셋:** ScanNetV2, S3DIS
- **평가 지표:** ScanNet에서는 mAP, AP@50을 사용하였고, S3DIS에서는 mCov, mWCov, mPrec, mRec를 사용하였다.
- **비교 대상:** PointGroup, DyCo3D, HAIS, SSTNet 등

### 주요 결과

1. **인스턴스 분할 성능:**
   - **S3DIS:** 6-fold 교차 검증 결과, mCov 71.5%, mRec 74.0% 등을 기록하며 HAIS 등의 기존 모델보다 우수한 성능을 보였다.
   - **ScanNet:** 작은 백본(Ours-S)만으로도 mAP 39.6%를 달성하였으며, 큰 백본(Ours-L)에서는 더 높은 성능을 보였다. 특히 DyCo3D 대비 mAP가 약 4.2% 향상되었다.
2. **3D 객체 탐지(Detection):** 예측된 마스크에 Axis-aligned bounding box를 피팅하여 평가한 결과, ScanNet AP@50에서 51.0%를 기록하며 DyCo3D(45.3%)와 3D-MPA(49.2%)를 앞질렀다.
3. **효율성:** DyCo3D와 동일한 GPU 환경에서 추론 속도가 약 26% 더 빨랐으며, mAP는 1.8% 더 높았다.

### 어블레이션 연구 (Ablation Study)

- **Center Prior의 한계:** 3D 포인트 클라우드에 2D의 Center Prior를 적용해 본 결과, mAP 상승폭이 매우 적어(0.4%) 3D에서의 정적 샘플 선택의 어려움을 확인하였다.
- **Dynamic Targets의 효과:** OT 기반의 동적 할당을 도입했을 때 baseline 대비 mAP가 3.1% 향상되었다.
- **Auxiliary Supervision의 효과:** 보조 헤드를 통해 정규화를 수행했을 때 mAP가 추가적으로 2.8% 향상되어, 최종적으로-OT와 보조 헤드를 모두 사용했을 때 가장 높은 성능을 보였다.

## 🧠 Insights & Discussion

### 강점 및 통찰

본 논문은 3D 인스턴스 분할에서 당연하게 여겨졌던 '클러스터링' 단계를 과감히 제거하고, 이를 포인트별 예측 문제로 치환하였다. 특히 3D 데이터의 기하학적 특성상 정적인 샘플링 전략이 작동하지 않는다는 점을 발견하고, 이를 예측값 기반의 **Optimal Transport**라는 수학적 최적화 문제로 해결한 점이 매우 인상적이다. 이는 모델 구조를 단순화하면서도 성능을 높이는 'Simpler but Stronger'의 방향성을 잘 보여준다.

### 한계 및 비판적 해석

- **샘플링 전략:** 현재 FPS(Farthest Point Sampling)를 사용하고 있으나, 저자 스스로 언급했듯이 더 정보량이 많은(Informative) 샘플링 전략이 있다면 성능을 더 끌어올릴 수 있을 것이다.
- **카테고리 결정 방식:** 인스턴스 마스크를 생성한 후, 해당 마스크 내 포인트들의 시맨틱 예측값 중 다수결(Majority voting)로 카테고리를 결정하는 방식은 단순하지만, 시맨틱 분할의 오류가 그대로 전이될 가능성이 있다.
- **계산 복잡도:** OT 할당은 훈련 단계에서만 수행되므로 추론 속도에는 영향이 없으나, $K$값이 매우 커질 경우 훈련 시의 메모리와 계산 비용이 증가할 수 있다.

## 📌 TL;DR

**요약:**
PointInst3D는 기존 3D 인스턴스 분할의 고질적인 문제인 클러스터링 의존성과 그로 인한 오류 전파 문제를 해결하기 위해, **클러스터링-프리(Clustering-free)** 및 **의존성-프리(Dependency-free)** 프레임워크를 제안한다. 핵심은 **Optimal Transport**를 이용해 샘플링 포인트와 인스턴스 타겟을 동적으로 최적 할당하는 것이며, 이를 통해 더 단순한 구조로도 ScanNet과 S3DIS 데이터셋에서 SOTA 수준의 성능과 더 빠른 추론 속도를 달성하였다.

**의의:**
본 연구는 3D 인스턴스 분할에서도 2D의 포인트 기반 예측 방식이 가능함을 증명하였으며, 특히 3D 특화 타겟 할당 전략(OT)을 제시함으로써 향후 3D 씬 이해(Scene Understanding) 연구에 있어 더 유연하고 효율적인 아키텍처 설계의 방향성을 제시하였다.
