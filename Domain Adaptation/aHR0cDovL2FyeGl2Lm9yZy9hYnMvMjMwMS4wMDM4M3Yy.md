# Discriminative Radial Domain Adaptation

Zenan Huang, Jun Wen, Siheng Chen, Linchao Zhu, and Nenggan Zheng (2023)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 도메인 적응(Domain Adaptation, DA) 과정에서 발생하는 **Domain Shift** 현상이다. 일반적으로 머신러닝 모델은 학습 데이터(Source)와 테스트 데이터(Target)가 동일한 분포를 가진다고 가정하지만, 실제 환경에서는 두 도메인 간의 분포 차이로 인해 모델의 성능이 크게 저하된다.

기존의 도메인 적응 방법론, 특히 적대적 도메인 적응(Adversarial Domain Adaptation) 방식은 도메인 불변 특징(Domain-invariant features)을 학습하여 분포를 일치시키는 데 집중한다. 그러나 이러한 방식은 다음과 같은 두 가지 주요 한계를 가진다:
1. **최적화의 어려움**: 적대적 학습의 min-max 게임은 최적화가 까다로우며, 도메인 간 격차가 크거나 데이터 분포가 복잡할 경우 잘못된 특징 정렬(False feature alignment)로 인해 모델이 붕괴될 위험이 있다.
2. **특징 판별력 저하**: 특징의 전이 가능성(Transferability)을 높이려는 시도가 오히려 특징의 판별력(Discriminability)을 훼손하여 분류 성능을 떨어뜨리는 상충 관계가 존재한다.

따라서 본 논문의 목표는 특징의 전이 가능성과 판별력을 동시에 확보할 수 있는 새로운 도메인 적응 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 특징 공간 내에서 각 카테고리가 중심에서 바깥쪽으로 뻗어 나가는 **Radial Structure(방사형 구조)**를 형성한다는 관찰에서 시작된다. 모델이 판별적으로 학습될수록 서로 다른 클래스의 특징들은 서로 다른 방향으로 확장되며 고유한 방사형 구조를 갖게 된다.

주요 기여 사항은 다음과 같다:
- **Discriminative Radial Domain Adaptation (DRDA)** 제안: 적대적 학습 대신, 소스와 타겟 도메인의 방사형 구조를 일치시켜 도메인 간 격차를 줄인다.
- **정렬 프로세스의 분리 (Decoupled Alignment)**: 정렬 과정을 '전역 등거리 변환(Global Isometric Transformation)'과 '지역 앵커 정제(Local Anchor Refinement)'의 두 단계로 나누어, 구조 정렬 과정에서 특징의 판별력이 손상되는 것을 방지한다.
- **구조 강화 기법 도입**: Optimal Transport(OT)를 이용하여 샘플들을 지역 앵커(Local Anchor) 주변으로 밀집시키고, 분류기의 예측과 기하학적 할당 간의 일관성(Consensus Regularization)을 강제하여 구조의 판별력을 높인다.

## 📎 Related Works

### 1. Domain Adaptation
기존의 DA 연구는 크게 두 가지 방향으로 진행되었다. 하나는 MMD(Maximum Mean Discrepancy)나 OT(Optimal Transport)와 같은 지표를 사용하여 도메인 간의 분포 차이를 직접 최소화하는 방식이고, 다른 하나는 DANN이나 ADDA와 같이 적대적 학습을 통해 도메인 판별자를 속이는 방식으로 도메인 불변 특징을 학습하는 방식이다. 하지만 전자는 조건부 분포의 차이를 완전히 해결하기 어렵고, 후자는 앞서 언급한 판별력 훼손 문제가 발생한다.

### 2. Discriminative Structure Learning
Center Loss나 Contrastive Loss와 같이 클래스 내 거리는 좁히고 클래스 간 거리는 넓히는 판별적 학습 기법들이 연구되어 왔다. 최근에는 각 카테고리의 중심(Prototype)을 정렬하는 방식(예: MSTN, Prototypical Networks)이 제안되었으나, 본 논문의 DRDA는 단순한 중심점 정렬을 넘어 전체적인 방사형 구조(전역 앵커와 지역 앵커의 관계)를 모델링한다는 점에서 차별화된다.

## 🛠️ Methodology

### 1. Radial Structure Construction
각 도메인은 하나의 전역 앵커(Global Anchor)와 $k$개의 지역 앵커(Local Anchor)로 구성된 방사형 구조 $G = \{a, N\}$로 표현된다.

- **Global Anchor ($a$)**: 도메인 내 모든 특징의 중심점(Centroid)으로 정의된다.
$$a^s = \frac{1}{n_s} \sum_{i=1}^{n_s} G_\theta(x^s_i), \quad a^t = \frac{1}{n_t} \sum_{j=1}^{n_t} G_\theta(x^t_j)$$
- **Local Anchors ($N = \{a_i\}$)**: 각 카테고리의 중심점이다. 소스 도메인은 레이블을 통해 직접 계산하며, 타겟 도메인은 의사 레이블(Pseudo-labels)을 통해 계산한다.
- **Egocentric Representation ($V$)**: 구조의 형태적 차이를 비교하기 위해 전역 앵커를 원점으로 하는 상대 좌표 $v_i = a_i - a$를 사용한다.

### 2. Radial Structure Alignment
구조 정렬은 전역 변환과 지역 정제로 분리되어 수행된다.

#### (1) Global Isometric Transformation
- **Translation Reduction**: 소스와 타겟의 전역 앵커 간 거리를 최소화하여 두 분포의 중심을 일치시킨다.
$$L_{global} = \|a^s - a^t\|_F$$
- **Rotation Reduction**: **Stiefel Layer** $S(\cdot)$를 도입하여 타겟 특징을 회전시킨다. Stiefel Manifold $\mathcal{V}_k(\mathbb{R}^d)$ 상에서 정의된 이 레이어는 특징의 노름(norm)을 보존하면서 회전 변환만을 수행하여 타겟 구조를 소스 구조에 맞게 회전시킨다.

#### (2) Local Refinement
전역 정렬 이후, 두 구조의 세부 형태를 일치시키기 위해 **Gromov-Wasserstein (GW) distance**를 사용한다. GW 거리는 좌표계의 이동, 회전, 순열에 불변하며 오직 내부 구조의 형태만을 비교한다. 본 논문에서는 계산 효율을 위해 전송 계획(Transport plan) $\pi$를 일대일 대응으로 고정하고, 다음과 같은 내부 거리 함수 $c(v_i, v_j)$를 정의하여 형태 차이를 최소화한다.
$$c(v_i, v_j) = \left[1 - \frac{\langle v_i, v_j \rangle}{\|v_i\| \|v_j\|}\right] + \lambda_{dist} \frac{1}{2} \|v_i - v_j\|_2^2$$
이 함수는 각 벡터 간의 **각도 차이(Cosine distance)**와 **길이 차이(Euclidean distance)**를 동시에 고려한다.

### 3. Radial Structure Enhancement
학습된 방사형 구조가 실제 데이터 분포를 잘 대표하도록 두 가지 제약을 추가한다.

- **Structure Faithfulness (OT Distance)**: 샘플들이 해당 지역 앵커에 가깝게 뭉치도록 엔트로피 정규화된 Optimal Transport 거리를 최소화한다.
$$L_{ot} = OT^\epsilon_\theta(X^s, N^s) + OT^\epsilon_\theta(X^t, N^t)$$
- **Semantic Meaningfulness (Consensus Regularization)**: OT를 통해 할당된 기하학적 레이블 $Q$와 분류기가 예측한 소프트 레이블 $P$ 사이의 KL-Divergence를 최소화하여, 기하학적 구조와 시맨틱 정보의 일치성을 확보한다.
$$L_R = KL(Q^s \| P^s) + H(P^s) + KL(Q^t \| P^t) + H(P^t)$$

### 4. Overall Optimization
최종 목적 함수는 다음과 같이 정의되며, 각 항은 가중치 $\lambda$에 의해 조절된다.
$$\min_{\theta, \phi, \Delta} L_{ce} + \lambda_T L_{global} + \lambda_\phi \phi(V^s, V^t) + \lambda_{ot} L_{ot} + \lambda_R L_R$$
여기서 $L_{ce}$는 소스 도메인의 분류 손실이며, $SG[\cdot]$ (Stop-gradient) 연산을 통해 앵커 업데이트와 네트워크 업데이트를 분리하여 학습의 안정성을 높인다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: Office-31, Office-Home, Office-Caltech10.
- **백본**: ResNet-50.
- **비교 대상**: DANN, JAN, CDAN+E, ALDA, MDD 등 최신 DA 방법론.
- **평가 지표**: 평균 분류 정확도(Average Classification Accuracy).

### 2. 주요 결과
- **Single-source UDA**: Office-31과 Office-Home에서 대부분의 전이 작업에 대해 SOTA 성능을 달성하였다. 특히 Office-Home과 같이 카테고리가 많고 도메인 간 격차가 큰 어려운 작업일수록 DRDA의 성능 향상 폭이 더 컸다.
- **Multi-source UDA**: 여러 소스 도메인을 하나의 구조로 통합하여 학습함으로써 일반화 성능이 향상되었으며, Office-Caltech10에서 타 방법론들을 압도하는 성능을 보였다.
- **Domain-Agnostic & Generalization**: 타겟 도메인이 특정되지 않은 상황이나, 학습 시 보지 못한 새로운 도메인에 대한 테스트에서도 높은 강건성을 보였다. 이는 방사형 구조를 통한 정렬이 단순한 분포 일치를 넘어 의미 있는 특징 추출을 유도했음을 시사한다.

## 🧠 Insights & Discussion

### 1. 방사형 구조의 유효성
본 논문의 실험(Fig. 3)을 통해 학습이 진행됨에 따라 특징들이 실제로 전역 앵커를 중심으로 방사형으로 확장되는 것을 확인하였다. 이는 선형 분류기의 작동 방식(각도와 노름의 중요성)과 일치하며, 이러한 구조를 정렬하는 것이 판별력을 유지하는 효율적인 방법임을 입증한다.

### 2. Stiefel Layer와 Decoupling의 중요성
Stiefel Layer를 제거했을 때 성능이 하락하는 것을 통해, 전역 회전 성분을 먼저 제거하는 것이 매우 중요함을 알 수 있다. 초기 단계에서 발생하는 무작위 회전 오차는 지역 정렬 과정에서 잘못된 정렬(Negative alignment)을 유발하며, 이는 복구가 불가능한 성능 저하로 이어진다. 따라서 전역 변환과 지역 정제를 분리하여 처리하는 전략이 필수적이다.

### 3. 각도 성분(Angular term)의 영향
내부 거리 함수에서 각도 차이를 제거했을 때 성능이 베이스라인 수준으로 떨어진다. 이는 고차원 공간에서 유클리드 거리보다 각도 기반의 거리가 분류 성능과 더 밀접한 관련이 있으며, 방사형 구조 모델링의 핵심이 각도 정보에 있음을 보여준다.

### 4. 한계 및 논의
본 방법론은 의사 레이블(Pseudo-labels)을 사용하여 타겟 지역 앵커를 계산한다. 따라서 초기 단계에서 의사 레이블의 정확도가 매우 낮을 경우, 잘못된 구조가 형성될 위험이 있다. 이를 방지하기 위해 본 논문에서는 전역 정렬을 통해 타겟 레이블의 신뢰도를 먼저 높이는 전략을 사용하고 있다.

## 📌 TL;DR

본 논문은 도메인 적응 시 발생하는 **'전이 가능성 vs 판별력'의 상충 관계**를 해결하기 위해, 특징 공간의 **방사형 구조(Radial Structure)**를 정렬하는 **DRDA**를 제안한다. 전역 등거리 변환(Stiefel Layer 이용)과 지역 형태 정제(GW distance 이용)를 분리하여 수행함으로써 판별력을 보존하며 도메인 간 격차를 줄인다. 실험적으로는 단일/다중 소스 적응 및 도메인 일반화 작업에서 SOTA 성능을 달성하였으며, 이는 방사형 구조 기반의 정렬이 복잡한 도메인 환경에서도 매우 강건하게 작동함을 보여준다.