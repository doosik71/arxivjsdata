# Occlusion-Ordered Semantic Instance Segmentation

Soroosh Baselizadeh, Cheuk-To Yu, Olga Veksler and Yuri Boykov (2025)

## 🧩 Problem to Solve

본 논문은 단일 이미지에서 객체의 인스턴스 분할(Instance Segmentation)과 더불어, 객체 간의 상대적인 깊이 순서(Relative Depth Order)를 동시에 추론하는 **Occlusion-Ordered Semantic Instance Segmentation (OOSIS)** 문제를 해결하고자 한다.

일반적인 시맨틱 인스턴스 분할은 2D 정보만을 제공하므로 3D 장면 이해에 한계가 있다. 이를 해결하기 위해 단안 깊이 추정(Monocular Depth Estimation)을 결합하는 방식이 주로 사용되지만, 절대적인 깊이(Absolute Depth)를 픽셀 수준에서 정밀하게 예측하는 것은 매우 어려운 작업이며, 특히 멀리 있는 객체나 얇은 객체의 경우 신뢰도가 급격히 떨어진다. 반면, 가려짐(Occlusion) 현상을 이용한 상대적 깊이 순서는 절대적 깊이보다 추정하기가 더 쉬우며, 거리와 관계없이 더 안정적인 3D 정보를 제공한다. 따라서 본 논문의 목표는 가려짐 경계(Occlusion Boundary) 정보를 활용하여 인스턴스 분할과 상대적 깊이 순서를 동시에 예측하는 통합 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **OOSIS의 통합 정식화**: 인스턴스 분할과 가려짐 순서 추정을 별개의 단계로 처리하지 않고, 이를 하나의 **레이블링 문제(Labeling Problem)**로 정식화하여 Conditional Random Field (CRF)를 통해 동시에 해결하는 딥러닝 기반 접근 방식을 최초로 제안하였다.
2. **지향성 가려짐 경계(Oriented Occlusion Boundaries) 예측**: 가려짐의 방향(occluder $\rightarrow$ occludee)을 포함하는 경계 예측 모델을 개발하였으며, 이는 기존의 회귀(Regression) 기반 방식보다 우수한 성능을 보이는 분류(Classification) 기반 방식이다.
3. **새로운 평가 지표 OAIR Curve**: 인스턴스 마스크의 정확도(Recall)와 가려짐 순서의 정확도(Accuracy)를 동시에 평가할 수 있는 **OAIR (Occlusion-ordered Accuracy vs. Instance Recall) curve** 지표를 제안하였다.
4. **모호하지 않은 인스턴스 분할(Unambiguous Instance Segmentation)**: 각 픽셀이 최대 하나의 인스턴스에만 할당되는 구조를 통해, 기존의 검출 기반(detect-and-segment) 방식보다 깔끔한 분할 결과를 얻었으며, 특정 조건에서 최신 트랜스포머 기반 모델보다 더 정확한 마스크를 생성함을 보였다.

## 📎 Related Works

- **Semantic Instance Segmentation**: Mask R-CNN과 같은 검출 기반 방식이나 Panoptic Segmentation 등이 존재하지만, 이들은 2D 분할에만 집중하며 깊이 정보를 제공하지 않는다.
- **Relative Depth Ordering**: 딥러닝 이전의 연구들은 크기나 $y$ 좌표 같은 단순한 휴리스틱에 의존하였다. 최근의 딥러닝 기반 연구 중에서는 깊이 정보를 직접 예측하여 순서를 매기는 방식이 있었으나, 본 논문은 가려짐 경계라는 국소적(local)이고 더 쉬운 단서를 사용하여 더 높은 신뢰도를 확보하였다.
- **Occlusion Order Prediction**: 기존 연구들은 이미 분할된 인스턴스 마스크 쌍이 주어졌을 때 그 순서만을 예측하는 방식이었다. 본 논문은 분할과 순서 예측을 동시에 수행한다는 점에서 차별화된다.
- **Oriented Occlusion Boundary Prediction**: 기존의 경계 예측 모델들은 경계 존재 여부와 방향 예측을 분리하여 처리하거나 회귀 방식으로 접근했으나, 본 논문은 이를 분류 문제로 통합하여 성능을 높였다.

## 🛠️ Methodology

본 연구의 파이프라인은 크게 두 단계로 구성된다.

### 1. Joint Semantic Segmentation and Oriented Occlusion Boundary Estimation

첫 번째 단계에서는 CNN(수정된 PSPNet)을 사용하여 시맨틱 세그멘테이션과 지향성 가려짐 경계를 동시에 예측한다.

- **지향성 경계 모델링**: 픽셀 $p$가 가려짐 경계인지 나타내는 이진 변수 $B_p$와, 경계일 경우 그 법선 벡터의 방향을 나타내는 변수 $O_p \in D$를 정의한다.
- **네트워크 구조**:
  - **Boundary Head ($b_p$)**: $\Pr(B_p = 1)$을 예측한다.
  - **Orientation Head ($e_p$)**: 경계라는 가정하에 방향의 조건부 확률 $\Pr(O_p = d | B_p = 1)$을 예측한다.
  - 최종 예측값 $o_p$는 다음과 같이 결합된다:
        $$o_p := [b_p \odot e_p, 1 - b_p]$$
- **손실 함수**:
    $$L = \sum_p \text{CE}(S_p | s_p) + \text{CE}_w(B_p | b_p) + B_p \cdot \text{CE}(E_p | e_p)$$
    여기서 $\text{CE}_w$는 경계 픽셀의 희소성을 해결하기 위한 가중치 적용 교차 엔트로피이다.

### 2. CRF Occlusion-Order Labeling

두 번째 단계에서는 예측된 시맨틱 맵과 가려짐 경계를 입력으로 하여, 픽셀 $p$에 가려짐 순서 레이블 $x_p$를 할당하는 CRF 최적화를 수행한다.

- **레이블 정의**: $x_p = 0$은 배경이며, $x_p > 0$인 정수는 가려짐 순서를 의미한다. 값이 클수록 앞에 있는 객체(occluder)이며, 같은 값을 가진 연결 성분이 하나의 인스턴스를 형성한다.
- **에너지 함수**:
    $$E(x) = \sum_{p \in P} u_p(x_p) + \lambda \sum_{(p,q) \in N} v(x_p, x_q) + \mu \sum_{(p,q) \in O} o(x_p, x_q)$$
  - **Unary Term ($u_p$)**: 시맨틱 세그멘테이션의 배경 확률 $\sigma_p$를 이용하여, $\sigma_p$가 높으면 $x_p=0$을 유도한다.
  - **Smoothness Term ($v$)**: 인접 픽셀이 서로 다른 레이블을 가질 때 페널티를 주어 공간적 일관성을 유지한다.
  - **Occlusion Term ($o$)**: 가려짐 관계 $(p, q) \in O$ (즉, $p$가 $q$를 가림)에 대해:
        $$o(x_p, x_q) = c_\infty \cdot [x_p < x_q] - [x_p > x_q]$$
        이는 $x_p > x_q$일 때 에너지를 낮추어(보상), 가려지는 객체보다 가리는 객체가 더 큰 레이블을 갖도록 강제한다.
- **최적화 (Jump Move Algorithm)**:
    레이블을 0부터 시작하여 1씩 증가시키는 **Jump Move** 방식을 사용한다. Smoothness 항이 non-submodular하여 직접적인 Graph-cut 최적화가 어렵기 때문에, 이를 **Submodular Upper Bound**로 대체하여 최적화하며, 에너지가 더 이상 감소하지 않을 때까지 반복한다.

## 📊 Results

### 실험 설정

- **데이터셋**: KINS (자율주행 장면, 14,991장), COCOA (자연 장면, 5,073장).
- **비교 대상 (Baselines)**:
  - 분할 모델(PanopticMask2Former, E2EC, Mask-RCNN)과 순서 예측 모델(OrderNet, InstaOrderNet, Monocular Depth)을 각각 조합한 파이프라인.
- **지표**: OAIR Curve (Recall vs Accuracy), mAP, WCS (Weighted Coverage Score).

### 주요 결과

- **OOSIS 성능**: 제안 방법(Joint-Labeling)은 모든 데이터셋에서 베이스라인보다 높은 OAIR 커브를 보였다. 특히 KINS 데이터셋에서 단안 깊이 기반 순서 예측보다 가려짐 기반 예측이 훨씬 정확함을 입증하였다.
- **전역 일관성**: 페어별 분류기(OrderNet 등)를 사용한 베이스라인은 가려짐 그래프에 사이클(Cycle)이 발생하는 오류가 잦았으나, 본 방법은 설계상 **Cycle-free**한 전역 일관적 순서를 보장한다.
- **마스크 품질**: 모호하지 않은(unambiguous) 분할 성능을 측정하는 WCS 지표에서, 더 강력한 아키텍처를 가진 PanopticMask2Former보다 더 높은 정확도를 기록하였다.
- **경계 예측 성능**: 지향성 가려짐 경계 예측 모델은 기존 SOTA 모델인 P2ORM보다 모든 지표(ODS, OIS, AP)에서 우수하였다.

## 🧠 Insights & Discussion

### 강점 및 통찰

- **통합 모델의 효율성**: 분할과 순서 예측을 별개로 수행하는 대신, 가려짐 경계라는 기하학적 단서를 레이블링 과정에 직접 통합함으로써 두 작업이 서로를 보완하게 하였다.
- **상대적 깊이의 신뢰성**: 단안 깊이 추정은 거리에 따라 오차가 커지지만, 가려짐 기반의 상대적 순서는 거리와 무관하게 일정 수준의 신뢰도를 유지한다는 점을 실험적으로 보여주었다.
- **Unambiguous Segmentation**: 픽셀당 하나의 레이블만 할당하는 제약 조건이 오히려 더 정교한 인스턴스 경계를 찾는 데 도움이 되었으며, 이는 CRF의 에너지 최적화 과정에서 가려짐 경계가 가이드 역할을 했기 때문으로 분석된다.

### 한계 및 향후 과제

- **부분 순서(Partial Order)의 한계**: 가려짐 체인(monotonic chain)으로 연결되지 않은 두 객체 간의 상대적 깊이는 알 수 없다.
- **상호 가려짐(Mutual Occlusion)**: 두 객체가 서로를 가리는 복잡한 상황은 처리할 수 없다.
- **추론 속도**: CPU 기반의 CRF 최적화로 인해 이미지당 약 12~13초가 소요되어 실시간 적용에는 한계가 있다.

## 📌 TL;DR

본 논문은 인스턴스 분할과 상대적 깊이 순서 추정을 동시에 해결하는 **OOSIS**라는 새로운 과제를 정의하고, 이를 위해 **CNN 기반의 지향성 가려짐 경계 예측**과 **CRF 기반의 통합 레이블링 최적화**를 제안하였다. 제안 방법은 기존의 '분할 후 순서 예측' 방식보다 정확도가 높고 전역적으로 일관된 순서를 보장하며, 특히 가려짐 정보가 단안 깊이 추정보다 상대적 깊이를 판단하는 데 더 신뢰할 수 있는 단서임을 입증하였다. 이는 향후 3D 장면 이해, 이미지 캡셔닝, 시각적 질의응답(VQA) 등의 분야에서 중요한 기초 기술로 활용될 가능성이 크다.
