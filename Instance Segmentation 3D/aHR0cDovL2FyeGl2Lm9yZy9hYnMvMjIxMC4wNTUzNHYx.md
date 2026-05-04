# Learning Inter-Superpoint Affinity for Weakly Supervised 3D Instance Segmentation

Linghua Tang, Le Hui, and Jin Xie (2022)

## 🧩 Problem to Solve

3D 포인트 클라우드(point cloud)의 인스턴스 분할(instance segmentation)은 실내 내비게이션, 증강 현실, 로보틱스 등 다양한 분야에서 필수적인 기술이다. 그러나 완전 지도 학습(fully supervised learning) 방식은 방대한 양의 정밀한 주석(annotation) 데이터가 필요하며, 3D 데이터의 특성상 이러한 라벨링 작업은 매우 많은 시간과 비용이 소요된다.

본 논문이 해결하고자 하는 핵심 문제는 **매우 제한적인 주석(인스턴스당 단 하나의 포인트 라벨)만으로도 어떻게 고성능의 3D 인스턴스 분할을 달성할 것인가**이다. 즉, 라벨의 희소성(sparsity) 문제를 극복하고 효과적으로 인스턴스 경계를 구분할 수 있는 약지도 학습(weakly supervised learning) 프레임워크를 구축하는 것이 목표이다.

## ✨ Key Contributions

본 연구의 중심 아이디어는 두 가지 단계의 전략으로 요약된다.

1.  **Inter-Superpoint Affinity Mining**: 포인트 수준의 희소한 라벨을 슈퍼포인트(superpoint) 수준으로 확장하고, 슈퍼포인트 간의 시맨틱(semantic) 및 공간적(spatial) 관계를 고려한 어피니티(affinity, 유사도)를 학습하여 Random Walk 기반의 라벨 전파를 통해 고품질의 의사 라벨(pseudo label)을 생성한다.
2.  **Volume-Aware Instance Refinement**: 약지도 학습에서는 인스턴스의 정확한 경계를 찾기 어렵다는 점에 착안하여, 객체의 부피 정보(복셀 수 및 반지름)라는 물리적 제약 조건을 도입하여 인스턴스 클러스터링 결과를 정교화한다.

## 📎 Related Works

### 기존 연구 및 한계
- **완전 지도 학습 기반 방식**: PointGroup, SoftGroup, GraphCut 등이 높은 성능을 보이지만, 막대한 양의 라벨링 데이터에 의존한다.
- **약지도/반지도 학습 기반 방식**: 
    - SPIB [19]는 바운딩 박스(bounding box)를 감독 신호로 사용하지만, 박스 라벨링 역시 여전히 비용이 많이 든다.
    - SegGroup [25]는 인스턴스당 한 점을 클릭하는 방식을 사용하며 의사 라벨을 생성하지만, 판별적인 인스턴스 특징을 학습하는 능력이 부족하여 의사 라벨의 품질이 낮다는 한계가 있다.

### 차별점
본 논문은 단순한 의사 라벨 생성을 넘어, **슈퍼포인트 그래프 상에서 어피니티를 적응적으로 학습**하고 **객체의 부피 제약 조건(volume constraint)**을 직접적으로 학습 과정에 통합함으로써, 더 적은 라벨로도 완전 지도 학습 방식에 근접하는 성능을 목표로 한다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 구조
본 프레임워크는 크게 두 단계(Stage)로 구성된다.
- **Stage 1**: 슈퍼포인트 그래프를 구축하고, 어피니티 마이닝을 통해 라벨을 전파하여 초기 의사 라벨을 생성하고 네트워크를 학습시킨다.
- **Stage 2**: Stage 1에서 얻은 결과를 바탕으로 객체의 부피 정보를 학습하고, 이를 이용해 인스턴스 분할 결과를 정밀하게 수정(refinement)한다.

### 2. 주요 구성 요소 및 상세 설명

#### A. Backbone Network
먼저, 비지도 방식의 오버세그멘테이션(oversegmentation)을 통해 포인트 클라우드를 **슈퍼포인트(superpoint)**들로 나누고, 이들을 노드로 하는 **슈퍼포인트 그래프 $G=(V, E)$**를 구축한다.
- **특징 추출**: 3D U-Net을 통해 포인트 특징을 추출한 후 평균 풀링(average pooling)으로 슈퍼포인트 특징을 만든다. 이후 ECC(Edge-Conditioned Convolutions)를 사용하여 그래프 구조의 특징을 반영한 최종 슈퍼포인트 임베딩 $X$를 추출한다.

#### B. Inter-Superpoint Affinity Mining
두 인접 슈퍼포인트 $i$와 $j$ 사이의 어피니티 $A_{ij}$를 다음과 같이 학습한다.

$$A_{ij} = \frac{\exp(\sigma(\phi(X_i), \psi(X_j)) * \gamma(p_i - p_j))}{\sum_{k \in N_i} \exp(\sigma(\phi(X_i), \psi(X_k)) * \gamma(p_i - p_k))}$$

여기서 $\sigma(\cdot, \cdot)$는 시맨틱 유사도를 측정하는 내적(dot product)이며, $\gamma(p_i - p_j)$는 중심 좌표의 차이를 이용해 공간적 유사도를 측정하는 MLP이다. 학습된 어피니티를 통해 업데이트된 임베딩 $\tilde{X}_i$는 다음과 같다.

$$\tilde{X}_i = A_{ij} \cdot \rho(X_j) + X_i$$

#### C. Semantic-Aware Random Walk (라벨 전파)
학습된 어피니티 $A$와 예측된 시맨틱 라벨 $S^c$(클래스 $c$에 대해 동일 클래스면 1, 아니면 0)를 결합하여 가중치 행렬 $P^c$를 계산한다.

$$P^c = M \odot S^c \odot A$$

이후 전이 확률 행렬 $T^c = D^{-1}P^c$를 유도하고, $t$번의 반복 전파(iteration)를 통해 의사 라벨 $I_j$를 할당한다.

$$I_j = I_k, \quad \text{where } k = \arg\max_{i} (\hat{T}^c_{ij}), \quad \hat{T}^c = (T^c)^t$$

#### D. Volume-Aware Instance Refinement
객체의 부피 정보를 활용해 인스턴스를 정교화한다.
- **부피 예측**: Stage 1의 모델로 생성된 의사 인스턴스를 통해 각 객체의 복셀 수($u$)와 반지름($r$)을 계산하고 이를 ground truth로 사용하여 네트워크를 재학습시킨다.
- **부피 인식 클러스터링 (Algorithm 1)**: 예측된 반지름 $r_i$를 이용해 인접 슈퍼포인트를 그룹화하고, 그룹의 평균 복셀 수 $\bar{w}$가 예측된 부피 $w$의 일정 임계값($\beta \bar{w}$)보다 클 경우에만 유효한 인스턴스로 인정한다.

### 3. 학습 절차 및 손실 함수

**Stage 1**: 시맨틱 손실($L_{sem}$), 오프셋 손실($L_{offset}$), 그리고 어피니티 손실($L_{aff}$)을 함께 최적화한다.
$$L_{stage1} = L_{sem} + L_{offset} + L_{aff}$$
- $L_{aff}$는 Discriminative Loss를 사용하여 동일 객체 내 슈퍼포인트는 가깝게, 서로 다른 객체는 멀게 배치한다.

**Stage 2**: 어피니티 손실을 제외하고 부피 손실($L_{volume}$)을 추가한다.
$$L_{stage2} = L_{sem} + L_{offset} + L_{volume}$$
- $L_{volume}$은 예측된 복셀 수와 반지름의 $L_1$ 거리를 최소화하는 방식으로 정의된다.

## 📊 Results

### 실험 설정
- **데이터셋**: ScanNet-v2, S3DIS.
- **평가 지표**: $\text{AP}$, $\text{AP}_{50}$, $\text{AP}_{25}$ (ScanNet-v2), $\text{mCov}$, $\text{mWCov}$, $\text{mPrec}$, $\text{mRec}$ (S3DIS).
- **약지도 설정**: 각 인스턴스당 단 하나의 포인트만 라벨링하여 사용.

### 주요 결과
- **ScanNet-v2**: 본 방법(3D-WSIS)은 $\text{AP}_{25}$ 기준 67.5%를 달성하여, 기존 약지도 학습 방법인 SegGroup(62.9%)보다 약 5% 성능 향상을 보였다. 특히 바운딩 박스를 사용한 SPIB보다 더 적은 정보(점 하나)만으로도 더 높은 성능을 기록했다.
- **S3DIS**: 6-fold cross validation 및 Area 5 실험 모두에서 기존 약지도 방식들을 상회하며, 일부 완전 지도 학습 방식(SGPN 등)보다 우수한 결과를 보였다.

### Ablation Study
- **의사 라벨의 효과**: Random Walk 반복 횟수가 증가함에 따라 성능이 향상되며, Stage 2의 부피 정제 과정을 거쳤을 때 성능이 크게 도약함을 확인했다.
- **의사 라벨의 품질**: 단순 Random Walk보다 어피니티 제약과 시맨틱 제약을 추가했을 때, 생성되는 의사 라벨의 양은 줄어들지만 정확도(Accuracy)는 비약적으로 상승(39.9% $\rightarrow$ 81.9%)함을 입증했다.

## 🧠 Insights & Discussion

본 논문은 3D 인스턴스 분할에서 가장 비용이 많이 드는 '라벨링' 문제를 효과적으로 해결했다. 특히 단순히 데이터 증강이나 단순한 의사 라벨링에 의존하지 않고, **슈퍼포인트 그래프**라는 구조적 특징과 **물리적 부피 제약**이라는 도메인 지식을 딥러닝 파이프라인에 잘 녹여낸 점이 강점이다.

**비판적 해석 및 한계**:
- **슈퍼포인트 의존성**: 제안 방법은 비지도 오버세그멘테이션 결과에 크게 의존한다. 만약 초기 슈퍼포인트 분할이 잘못되어 하나의 슈퍼포인트에 서로 다른 두 인스턴스가 섞여 들어간다면, 이후의 라벨 전파 과정에서 오류가 누적될 가능성이 있다.
- **하이퍼파라미터**: $\lambda$ (반지름 필터링) 및 $\beta$ (복셀 수 필터링)와 같은 하이퍼파라미터가 경험적으로 설정되었는데, 객체의 크기가 매우 다양할 경우 최적의 공통 값을 찾기 어려울 수 있다.

## 📌 TL;DR

본 논문은 **인스턴스당 단 하나의 포인트 라벨**만 사용하는 매우 가벼운 약지도 학습 기반의 3D 인스턴스 분할 프레임워크를 제안한다. 핵심은 **시맨틱-공간 어피니티를 학습한 Random Walk 기반 라벨 전파**와 **객체 부피 정보를 이용한 인스턴스 정제**이다. 이를 통해 ScanNet-v2와 S3DIS 데이터셋에서 기존 약지도 학습 SOTA 성능을 달성했으며, 데이터 라벨링 비용을 획기적으로 줄이면서도 높은 정밀도를 확보할 수 있음을 보여주었다.