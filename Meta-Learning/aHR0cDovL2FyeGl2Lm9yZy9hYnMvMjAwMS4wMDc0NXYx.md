# Automated Relational Meta-Learning

Huaxiu Yao, Xian Wu, Zhiqiang Tao, Yaliang Li, Bolin Ding, Ruirui Li, Zhenhui Li (2020)

## 🧩 Problem to Solve

본 논문은 메타 러닝(Meta-learning)에서 발생하는 **Task Heterogeneity**(태스크 이질성) 문제를 해결하고자 한다. 기존의 많은 메타 러닝 알고리즘들은 모든 태스크에 대해 전역적으로 공유되는 meta-learner(예: 매개변수 초기화, meta-optimizer 등)를 학습한다. 그러나 학습해야 할 태스크들이 서로 다른 분포에서 추출된 경우, 이러한 전역 공유 방식은 각 태스크의 특성을 충분히 반영하지 못해 성능이 저하되는 문제가 발생한다.

태스크 이질성을 해결하기 위해 일부 연구에서는 태스크별 표현(task-specific representation)을 학습하여 전역 지식을 맞춤형으로 조정하려 했으나, 관련성이 높은 태스크 간의 지식 일반화 능력이 떨어지는 한계가 있었다. 또한, 계층적 구조를 도입한 방법(예: HSML)은 사람이 직접 설계한(hand-crafted) 구조에 의존하므로, 복잡한 태스크 간 관계를 포착하기 어렵고 튜닝에 많은 비용이 소요된다는 단점이 있다. 따라서 본 논문의 목표는 태스크 간의 관계를 자동으로 추출하고 이를 구조화하여, 새로운 태스크가 주어졌을 때 가장 관련 있는 지식을 빠르게 찾아 적용할 수 있는 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 지식 베이스(Knowledge Base)의 지식 그래프(Knowledge Graph) 구조에서 영감을 얻어, **Meta-knowledge Graph**(메타 지식 그래프)를 자동으로 구축하고 이를 통해 태스크 맞춤형 지식을 제공하는 것이다.

주요 기여 사항은 다음과 같다:
1. **자동화된 메타 지식 그래프 구축**: 과거 태스크들로부터 교차 태스크 관계를 자동으로 추출하여 메타 지식 그래프를 생성함으로써, 새로운 태스크 학습 시 최적의 구조적 지식을 제공한다.
2. **성능 향상**: 2D Regression 및 Few-shot Image Classification 실험을 통해 기존의 state-of-the-art 메타 러닝 알고리즘보다 우수한 성능을 입증하였다.
3. **해석 가능성(Interpretability) 제공**: 구축된 메타 지식 그래프의 정점(vertex)과 간선(edge)을 분석함으로써, 모델이 태스크 간의 어떤 관계(예: 형태적 유사성, 텍스처 유사성 등)를 학습했는지 시각적으로 확인하고 해석할 수 있게 한다.

## 📎 Related Works

본 연구는 Gradient-based meta-learning 연구 흐름에 기반한다. 

- **Globally shared methods**: MAML, Meta-SGD와 같이 전역적으로 공유되는 초기화 값이나 최적화 경로를 학습하는 방식이다. 이는 구현이 간단하지만, 태스크 분포가 다양할 때 발생하는 Task Heterogeneity 문제에 취약하다.
- **Task-specific methods**: MT-Net, MUMOMAML, HSML 등은 태스크별 특성을 반영하여 전역 지식을 조정한다. 특히 HSML은 계층적 클러스터링 구조를 사용하여 일반화와 맞춤화의 균형을 맞추려 했으나, 구조 설계가 수동적이라는 한계가 있다.
- **Non-parametric methods**: ProtoNet, TADAM 등은 거리 메트릭을 학습하여 퓨샷 분류를 수행한다.

ARML은 기존의 수동적 구조 설계에서 벗어나, 데이터로부터 직접 **Relational Structure**를 자동 추출하여 그래프 형태로 관리한다는 점에서 기존 연구들과 차별화된다.

## 🛠️ Methodology

ARML의 전체 파이프라인은 (1) Prototype-based sample structuring, (2) Automated meta-knowledge graph construction and utilization, (3) Task-specific knowledge fusion and adaptation의 세 단계로 구성된다.

### 1. Prototype-based Sample Structuring
태스크 내 샘플들 간의 관계를 정의하기 위해 **Prototype-based Relational Graph** $\mathcal{R}_i$를 구축한다. 이는 이상치(abnormal samples)의 영향을 줄이기 위해 raw sample 대신 프로토타입(prototype)을 정점으로 사용한다.

- **Classification**: 각 클래스의 샘플들을 임베딩 함수 $E$를 통해 투영한 후 평균을 내어 프로토타입 $c^k_i$를 생성한다.
  $$c^k_i = \frac{1}{N^{tr}_k} \sum_{j=1}^{N^{tr}_k} E(x_j)$$
- **Regression**: 클래스 정보가 없으므로, 할당 행렬 $P_i$를 학습하여 샘플들을 $K$개의 클러스터로 묶어 프로토타입을 생성한다.
- **Edge Weight**: 두 프로토타입 $c^j_i$와 $c^m_i$ 사이의 간선 가중치는 다음과 같이 유사도를 기반으로 계산된다.
  $$A^R_i(c^j_i, c^m_i) = \sigma(W^r(|c^j_i - c^m_i|/\gamma_r) + b_r)$$

### 2. Automated Meta-knowledge Graph Construction and Utilization
과거 태스크들의 지식을 저장하는 **Meta-knowledge Graph** $G = (H^G, A^G)$를 구축한다. 여기서 $H^G$는 학습 가능한 정점 특징 행렬이며, $A^G$는 정점 간의 관계를 나타내는 인접 행렬이다.

- **Super-graph $\mathcal{S}_i$ 생성**: 새로운 태스크 $\mathcal{T}_i$가 들어오면, 해당 태스크의 릴레이셔널 그래프 $\mathcal{R}_i$와 메타 지식 그래프 $G$를 결합하여 슈퍼 그래프 $\mathcal{S}_i$를 만든다. 이때 프로토타입 $c^j_i$와 메타 정점 $h_k$ 사이의 연결 가중치는 Euclidean distance에 Softmax를 적용하여 결정된다.
  $$A^S(c^j_i, h_k) = \frac{\exp(-\| (c^j_i - h_k)/\gamma_s \|^2_2/2)}{\sum_{k'=1}^G \exp(-\| (c^j_i - h_{k'})/\gamma_s \|^2_2/2)}$$
- **Information Propagation**: GNN(Graph Neural Network)의 메시지 패싱 메커니즘을 통해 메타 지식 그래프 $G$로부터 $\mathcal{R}_i$로 관련 지식을 전파한다. 이를 통해 풍부해진 프로토타입 표현 $\hat{C}^R_i$를 얻는다.

### 3. Task-specific Knowledge Fusion and Adaptation
추출된 지식을 바탕으로 전역 공유 초기값 $\theta^0$를 태스크에 맞게 조정(Modulation)한다.

- **Task Representation**: Auto-encoder 구조를 사용하여 raw 프로토타입 $C^R_i$로부터 태스크 표현 $q_i$를, GNN을 거친 프로토타입 $\hat{C}^R_i$로부터 $t_i$를 각각 추출한다. 이때 학습 안정성을 위해 재구성 손실(Reconstruction Loss) $L_q$와 $L_t$를 도입한다.
- **Modulating Function**: 다음과 같은 게이팅 메커니즘을 통해 태스크 맞춤형 초기값 $\theta^0_i$를 생성한다.
  $$\theta^0_i = \sigma(W^g(t_i \oplus q_i) + b_g) \circ \theta^0$$
- **최종 목적 함수**:
  $$\min_{\Phi} L_{all} = \sum_{i=1}^I L(f_{\theta^0_i - \alpha \nabla_\theta L(f_\theta, D^{tr}_i)}, D^{ts}_i) + \mu_1 L_t + \mu_2 L_q$$

## 📊 Results

### 실험 설정
- **데이터셋**: 
  - **2D Regression**: Sinusoids, Line, Quadratic 등 6가지 함수군을 사용한 2D 회귀 문제.
  - **Few-shot Classification**: Plain-Multi(CUB, Texture, Aircraft, Fungi) 및 Art-Multi(이미지에 Blur, Pencil 필터 적용하여 이질성 증폭) 데이터셋.
- **비교 대상**: MAML, Meta-SGD (Global), MT-Net, MUMOMAML, HSML (Task-specific), ProtoNet, TADAM (Non-parametric).

### 주요 결과
- **2D Regression**: ARML이 MSE 기준 가장 낮은 오차를 기록하며, 전역 공유 모델 및 기존 태스크 맞춤형 모델보다 우수한 성능을 보였다.
- **Few-shot Classification**: 특히 태스크 이질성이 강한 **Art-Multi** 데이터셋에서 ARML의 성능 향상이 두드러졌다. 5-way 1-shot 기준, ARML은 다른 baseline들을 상회하는 정확도를 보였으며, 특히 수동 구조 기반의 HSML보다 높은 성능을 기록하여 릴레이셔널 구조의 이점을 입증하였다.
- **분석**: 메타 지식 그래프 분석 결과, 특정 정점들이 '곡선(curve)', '선(line)', '텍스처(texture)', '블러(blur)'와 같은 구체적인 의미적 특징을 캡처하고 있음을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 메타 러닝에서 태스크 이질성을 해결하기 위해 '자동화된 그래프 구조'라는 접근 방식을 제안하여 성공적인 결과를 얻었다.

**강점 및 해석**:
- **유연한 지식 활용**: 고정된 계층 구조(HSML)와 달리, ARML은 프로토타입-프로토타입, 프로토타입-메타지식, 메타지식-메타지식 간의 관계를 동시에 탐색하므로 훨씬 유연하게 지식을 전이할 수 있다.
- **해석 가능성**: 학습된 메타 지식 그래프를 통해 모델이 어떤 공통 특징을 기반으로 태스크를 구분하고 지식을 공유하는지 시각적으로 분석할 수 있다는 점이 매우 큰 강점이다.

**한계 및 논의사항**:
- **계산 복잡도**: 매 태스크마다 슈퍼 그래프를 구축하고 GNN 전파 과정을 거쳐야 하므로, 계산 비용이 증가할 가능성이 있다.
- **정점 수의 영향**: 실험 결과 정점 수($G$)가 약 8개일 때 성능이 포화되는 경향을 보였으나, 이는 실험에 사용된 데이터셋의 복잡도에 따른 결과일 수 있으며, 더 방대한 데이터셋에서는 더 많은 정점이 필요할 것이다.
- **범용성**: 현재는 피처 공간과 라벨 공간이 공유되는 태스크들에 집중하고 있으며, 서로 다른 피처/라벨 공간을 가진 태스크 간의 관계 확장 문제는 향후 과제로 남아 있다.

## 📌 TL;DR

본 논문은 태스크 이질성 문제를 해결하기 위해 **자동으로 구축되는 메타 지식 그래프(Meta-knowledge Graph)**를 도입한 ARML 프레임워크를 제안한다. 이 모델은 태스크의 프로토타입 간 관계를 그래프로 구조화하고 GNN을 통해 과거의 경험적 지식을 전이함으로써, 새로운 태스크에 최적화된 모델 초기값을 생성한다. 결과적으로 퓨샷 학습과 회귀 문제에서 기존 모델보다 높은 성능을 보였으며, 모델의 의사결정 과정을 그래프 구조로 해석할 수 있는 가능성을 제시하였다.