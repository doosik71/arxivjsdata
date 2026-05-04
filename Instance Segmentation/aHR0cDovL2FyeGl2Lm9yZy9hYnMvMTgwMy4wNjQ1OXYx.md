# Learning to Cluster for Proposal-Free Instance Segmentation

Yen-Chang Hsu, Zheng Xu, Zsolt Kira, Jiawei Huang (2018)

## 🧩 Problem to Solve

본 논문은 이미지 내의 각 픽셀에 대해 인스턴스 라벨을 부여하는 **Instance Segmentation** 문제를 해결하고자 한다. Instance Segmentation은 단순한 의미론적 분류(Semantic Segmentation)를 넘어, 동일한 클래스에 속하는 서로 다른 개별 객체들을 구분해내야 하는 과제이다.

특히 기존의 많은 방식들이 객체의 제안 영역을 먼저 생성하는 Proposal-based 방식에 의존하고 있는데, 이는 경계 상자(Bounding Box)의 품질에 성능이 크게 좌우되며 도로의 차선과 같이 얇고 긴 형태의 객체를 탐지하는 데 한계가 있다. 또한, Proposal-free 방식이라 하더라도 대개 '표현 학습'과 '클러스터링'이라는 두 단계의 파이프라인을 거치게 되어 연산 효율성과 최적화 측면에서 개선의 여지가 있다.

따라서 본 연구의 목표는 Proposal 생성 단계 없이, Fully Convolutional Network(FCN)의 단일 forward pass만으로 픽셀 단위의 클러스터링을 수행하여 인스턴스를 분리해내는 end-to-end 학습 목적 함수를 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 픽셀 간의 **쌍별 관계(Pairwise Relationship)**를 직접적인 학습 감독 신호로 사용하는 새로운 손실 함수를 설계한 것이다. 구체적인 기여 사항은 다음과 같다.

1. **End-to-End Pixel Clustering**: 픽셀들이 동일한 인스턴스에 속하는지 여부를 정의하는 관계 $R$을 이용하여 FCN이 직접 픽셀 클러스터링을 수행하도록 학습시키는 새로운 목적 함수를 제안하였다.
2. **Graph Coloring 이론의 접목**: 네트워크가 출력할 수 있는 인덱스(ID)의 수는 제한적이지만, 이미지 내 인스턴스의 수는 가변적이고 무제한적일 수 있다. 이를 해결하기 위해 그래프 채색(Graph Coloring) 이론을 도입하여, 인접한 인스턴스는 서로 다른 ID를 갖게 하고 멀리 떨어진 인스턴스는 ID를 재사용하게 함으로써 제한된 ID로 무제한의 인스턴스를 구분할 수 있게 하였다.
3. **실용적 검증**: 제안한 방법론이 차선 검출(Lane Detection) 및 Cityscapes 데이터셋에서 강력한 성능을 보임을 입증하였으며, 특히 외부 데이터 없이도 높은 효율성과 실시간성(~55 FPS)을 달성하였다.

## 📎 Related Works

기존의 Instance Segmentation 접근 방식은 크게 두 가지로 나뉜다.

**Proposal-based methods**는 Mask R-CNN과 같이 객체 탐지기를 통해 Bounding Box를 먼저 예측한 후, 해당 영역 내에서 foreground-background segmentation을 수행하는 'detect-then-segment' 패러다임을 따른다. 하지만 이러한 방식은 Bounding Box의 정확도에 의존하며, 형태가 정형화되지 않은 객체에 취약하다는 한계가 있다.

**Proposal-free methods**는 픽셀 레벨의 특징 표현(feature vector, energy level 등)을 학습한 뒤, 후처리 단계에서 클러스터링 알고리즘을 적용하여 픽셀들을 그룹화한다. 하지만 이러한 방식은 대개 두 단계로 분리되어 있어 end-to-end 학습이 어렵고, 클러스터링 알고리즘의 하이퍼파라미터 설정에 민감하다는 단점이 있다.

본 논문은 이러한 기존의 Proposal-free 방식과 달리, 별도의 중간 표현 학습 단계를 두지 않고 FCN이 직접 픽셀에 클러스터 인덱스를 할당하도록 학습시킨다는 점에서 차별점을 갖는다.

## 🛠️ Methodology

### 1. Instance Labeling Objective

본 연구는 인스턴스 라벨링을 픽셀 간의 관계 $R(p_i, p_j)$를 학습하는 문제로 정의한다. 두 픽셀 $p_i, p_j$가 동일한 인스턴스에 속하면 $R=1$, 아니면 $R=0$이다. FCN의 출력은 각 픽셀이 특정 인스턴스 인덱스에 할당될 확률 분포 $P = [t_1, \dots, t_n]$ (Multinomial distribution)으로 정의된다.

**동일 인스턴스 손실 ($L^+$):** 두 픽셀이 같은 인스턴스에 속한다면, 이들의 출력 확률 분포는 서로 유사해야 한다. 이를 위해 대칭적인 KL-Divergence를 사용한다.
$$L(p_i, p_j)^+ = D_{KL}(P_i || P_j) + D_{KL}(P_j || P_i)$$
여기서 $D_{KL}(P_i || P_j) = \sum_{k=1}^{n} t_{i,k} \log \frac{t_{i,k}}{t_{j,k}}$이다.

**상이 인스턴스 손실 ($L^-$):** 두 픽셀이 서로 다른 인스턴스에 속한다면, 확률 분포는 일정 거리 $\sigma$ 이상으로 서로 달라야 한다. 이를 위해 Hinge-loss를 적용한다.
$$L(p_i, p_j)^- = L_h(D_{KL}(P_i || P_j), \sigma) + L_h(D_{KL}(P_j || P_i), \sigma)$$
여기서 $L_h(e, \sigma) = \max(0, \sigma - e)$이며, 마진 $\sigma$는 2로 설정되었다.

**최종 Contrastive Loss:** 위 두 식을 관계 $R$에 따라 결합하여 픽셀 쌍에 대한 손실 함수를 구성한다.
$$L(p_i, p_j) = R(p_i, p_j)L(p_i, p_j)^+ + (1 - R(p_i, p_j))L(p_i, p_j)^-$$

### 2. Background Handling 및 전체 손실 함수

배경(Background)은 이미지에서 대부분의 면적을 차지하므로, 배경 픽셀들에 대해서는 쌍별 관계가 아닌 단일 픽셀 분류(Unary prediction) 방식으로 처리한다. 인덱스 0을 배경으로 예약하고, Binary Cross Entropy와 유사한 형태의 손실 함수 $L_{bg}$를 사용한다.
$$L_{bg} = -\frac{1}{N} \sum_{i} (I_{bg,i} \log t_{i,0} + (1 - I_{bg,i}) \log(\sum_{k=1}^{n} t_{i,k}))$$

전체 인스턴스 세그멘테이션 손실 함수는 다음과 같다.
$$L_{ins} = L_{pair} + L_{bg}$$
여기서 $L_{pair}$는 샘플링된 픽셀 쌍들에 대한 평균 손실 값이다.

### 3. Unlimited Instances via Graph Coloring

제한된 수의 ID $n$으로 무제한의 인스턴스를 구분하기 위해, 본 논문은 그래프 채색 이론을 도입한다. 인접한 인스턴스는 서로 다른 색(ID)을 가져야 하지만, 멀리 떨어진 인스턴스는 동일한 색을 재사용할 수 있다는 원리를 이용한다.

이를 위해 학습 시 샘플링 전략을 수정하여, 공간적 거리 $|p_i p_j|$가 임계값 $\epsilon$ (실험적으로 256 픽셀) 이하인 픽셀 쌍들에 대해서만 $L_{pair}$를 적용한다. 거리가 $\epsilon$보다 먼 픽셀들은 서로 어떤 ID를 갖든 손실 함수에 영향을 주지 않으므로, 네트워크는 자연스럽게 인접한 객체들끼리만 ID가 겹치지 않도록 학습하게 된다.

최종 추론 시에는 동일한 ID를 가진 픽셀들 중 **Connected Components(연결 성분)**를 추출함으로써 각각의 개별 인스턴스를 복원한다.

### 4. Network Architecture

네트워크는 **Feature Pyramid Network(FPN)** 구조를 기반으로 한다.

- **Backbone**: Pre-trained ResNet을 사용하여 특징 맵을 추출한다.
- **Neck**: 각 단계의 특징 맵을 Up-sampling하고 element-wise summation을 통해 결합하여 최종 특징 맵 $M$을 생성한다.
- **Heads**: 특징 맵 $M$ 위에 Task-specific 레이어를 추가한다. 인스턴스 ID 할당을 위해 $3\times3$ Conv 레이어와 $1\times1$ Conv 레이어(출력 차원 $n+1$)를 배치한다. 또한 다중 클래스 대응을 위해 Semantic Segmentation 헤드와 객체 중심(Object Center) 예측 헤드를 추가하여 Multi-task learning을 수행한다.

## 📊 Results

### 1. Lane Detection (TuSimple Dataset)

- **설정**: 차선을 10픽셀 너비의 마스크로 변환하여 인스턴스 세그멘테이션 문제로 정의하였다.
- **결과**: 2017 CVPR Autonomous Driving Challenge에서 **2위**를 차지하였다.
- **분석**: 외부 데이터를 사용한 1위 모델과 성능 차이가 미미하며, 특히 외부 데이터 없이 달성한 최고 성적이라는 점에서 의의가 있다. 또한 약 55 FPS의 실시간 추론 속도를 보였다.

### 2. Cityscapes Instance Segmentation

- **지표**: Average Precision (AP)을 측정하였다.
- **결과**: Proposal-free 방식 중 상위 4위 안에 들었으며, **15.1% AP**를 기록하였다. 특히 동일하게 그래프 라벨링 개념을 사용한 JGD(9.8% AP)보다 훨씬 우수한 성능을 보였다.
- **분석**: Semantic Segmentation의 품질이 인스턴스 AP에 큰 영향을 미침을 확인하였다. GT-Seg(정답 세그멘테이션)를 사용했을 때 AP가 38.4%까지 상승하는 것을 통해, 제안한 인스턴스 마스크 생성 능력 자체는 매우 강력함을 입증하였다.

## 🧠 Insights & Discussion

**강점:**

- Bounding Box 제안 단계 없이 FCN의 단일 통과만으로 인스턴스를 분리하는 효율적인 end-to-end 프레임워크를 제시하였다.
- 그래프 채색 이론을 딥러닝의 목적 함수에 녹여내어, 고정된 출력 차원으로 가변적인 수의 객체를 처리하는 창의적인 해결책을 제시하였다.
- 차선 검출과 같은 특수 목적의 작업뿐만 아니라 Cityscapes와 같은 일반적인 도시 장면 데이터셋에서도 범용성을 입증하였다.

**한계 및 비판적 해석:**

- **ID 충돌 문제**: 인접한 두 객체가 우연히 동일한 ID를 할당받을 경우, Connected Component 추출 단계에서 두 객체가 하나로 병합되는 현상이 발생한다. 이는 정성적 결과에서도 확인되는 주요 실패 사례이다.
- **과분할(Over-segmentation)**: 하나의 객체가 여러 개의 세그먼트로 쪼개지는 현상이 발생하며, 이를 해결하기 위해 Object Center 예측을 통한 병합 후처리를 도입하였으나 완벽하지는 않다.
- **AP 지표의 한계**: 시각적으로는 매우 훌륭한 결과를 내놓음에도 불구하고, AP 수치가 낮게 나오는 경향이 있다. 이는 신뢰도 점수(Confidence score) 할당 방식의 한계일 수 있으며, 저자 또한 이 점을 지적하였다.

## 📌 TL;DR

본 논문은 FCN이 직접 픽셀 단위의 클러스터링을 수행하도록 하는 **Pairwise Relationship 기반의 새로운 학습 목적 함수**를 제안한다. 특히 **그래프 채색(Graph Coloring)** 개념을 도입하여 제한된 수의 ID로 무제한의 인스턴스를 구분할 수 있게 하였으며, 이를 통해 Proposal-free 방식의 Instance Segmentation을 end-to-end로 구현하였다. 이 연구는 특히 실시간성이 중요하거나 Bounding Box 기반의 접근이 어려운 특수 형태의 객체 탐지 분야에 중요한 기여를 할 가능성이 높다.
