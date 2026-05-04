# Model-Agnostic Graph Regularization for Few-Shot Learning

Ethan Shen, Maria Brbić, Nicholas Monath, Jiaqi Zhai, Manzil Zaheer, Jure Leskovec (2021)

## 🧩 Problem to Solve

본 논문은 데이터가 매우 제한적인 상황에서 새로운 클래스를 분류해야 하는 Few-Shot Learning (FSL) 문제에서 클래스 간의 관계 정보(Knowledge Graph)를 어떻게 효과적으로 활용할 것인가를 다룬다.

기존의 그래프 기반 FSL 방법론들은 지식 그래프(Knowledge Graph)를 보조 정보로 활용하여 유의미한 성과를 거두었으나, 대개 매우 복잡한 아키텍처를 가지고 있으며 수많은 하위 컴포넌트로 구성되어 있다. 이러한 복잡성은 그래프 정보가 실제로 성능 향상에 어떤 영향을 미치는지에 대한 심층적인 이해를 방해하며, 빠르게 발전하는 다양한 meta-learning 모델들에 유연하게 적용하기 어렵게 만든다. 따라서 본 연구의 목표는 특정 모델에 종속되지 않고(model-agnostic), 단순하면서도 강력하게 그래프 정보를 통합할 수 있는 정규화(regularization) 방법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 복잡한 그래프 신경망(GNN) 구조를 설계하는 대신, 클래스 레이블 간의 그래프 구조가 모델 파라미터의 학습을 가이드하도록 하는 **Graph Regularization** 목적 함수를 도입하는 것이다.

중심적인 직관은 그래프 상에서 서로 가까운 이웃 관계에 있는 클래스들이 모델 내에서도 유사한 표현(representation) 또는 파라미터를 갖도록 강제하는 것이다. 이를 위해 `node2vec`의 무작위 보행(random walk) 기반 임베딩 개념을 차용하여, 그래프의 지역적 구조와 전역적 커뮤니티 특성을 보존하는 정규화 항을 기존의 분류 손실 함수에 추가하였다. 이 방식은 모델의 내부 구조를 변경하지 않고 손실 함수만 수정하면 되므로, 어떤 FSL 모델에도 적용 가능한 model-agnostic한 특성을 가진다.

## 📎 Related Works

Few-Shot Learning의 기존 접근 방식은 크게 세 가지로 나뉜다. 첫째는 task 간 전이 가능한 메트릭을 학습하는 Metric-based approach (예: Prototypical Networks), 둘째는 빠른 적응을 위한 초기화를 학습하는 Optimization-based approach (예: MAML, LEO), 셋째는 사전 학습 후 새로운 task에 맞게 미세 조정하는 Transfer learning 기반의 Fine-tuning 방식이다.

최근에는 WordNet과 같은 지식 그래프를 활용해 base 클래스에서 novel 클래스로 지식을 전이하는 연구들이 진행되었다. 대표적으로 KGTN과 같은 모델은 Gated Graph Neural Network (GGNN)를 사용하여 정보를 전파한다. 그러나 이러한 방법들은 파라미터 수가 많고 구조가 복잡하여, 본 논문에서 제안하는 단순한 정규화 방식과 대비된다. 본 연구는 복잡한 전파 메커니즘 없이도 단순한 정규화만으로 기존의 복잡한 그래프 임베딩 모델보다 더 나은 성능을 낼 수 있음을 보여준다.

## 🛠️ Methodology

### 1. 전체 파이프라인

본 제안 방법은 기존의 FSL 모델(Base Learner)에 그래프 정규화 항을 추가하여 함께 최적화하는 단순한 구조이다. 전체 목적 함수는 다음과 같이 정의된다:
$$\text{Total Loss} = \text{Classification Loss} + \lambda L_{graph}(G, \theta)$$
여기서 $\text{Classification Loss}$는 사용되는 Base Learner에 따라 다르며(예: Cross-Entropy 또는 거리 기반 손실), $L_{graph}$는 그래프 구조를 보존하기 위한 정규화 항이다.

### 2. Graph Regularization ($\text{L}_{graph}$)

그래프 정규화는 `node2vec`의 목적 함수를 기반으로 하며, 그래프 내의 노드 표현 $\theta$가 이웃 노드들의 유사도를 보존하도록 설계되었다. 수식은 다음과 같다:

$$L_{graph}(G, \theta) = -\sum_{y \in Y} \left[ -\log Z_y + \sum_{n \in N(y)} \frac{1}{T} \text{sim}(\theta_n, \theta_y) \right]$$

- $\theta$: 노드 표현(모델의 클래스 파라미터 또는 프로토타입).
- $\text{sim}(\cdot)$: 두 노드 간의 유사도 함수.
- $N(y)$: 소스 노드 $y$에서 random walk를 통해 얻은 이웃 노드 집합.
- $T$: 온도(temperature) 하이퍼파라미터.
- $Z_y$: 분배 함수(partition function)이며, 계산 효율성을 위해 negative sampling으로 근사한다.
  $$Z_y = \sum_{v \in Y} \exp\left(\frac{1}{T} \text{sim}(\theta_y, \theta_v)\right)$$

### 3. 모델별 적용 전략 (Augmentation Strategies)

Base Learner의 특성에 따라 유사도 함수 $\text{sim}(\cdot)$과 정규화 대상 $\theta$를 다르게 설정한다.

- **Metric-Based Models (예: ProtoNet):**
  클래스별 프로토타입 $p$를 $\theta$로 설정한다. 유사도 함수로는 음의 유클리드 거리($-\|p_i - p_j\|^2_2$)를 사용한다.
- **Optimization-Based Models (예: LEO):**
  잠재 클래스 인코딩 $z$를 $\theta$로 설정하여 inner-loop 적응 과정에서 정규화를 수행한다. 유사도 함수로는 내적(inner product) 또는 코사인 유사도를 사용한다.
- **Fine-tuning Models (예: SGM, $\text{S2M2}^R$):**
  마지막 분류 층(classifier)의 파라미터 $\theta$를 정규화 대상으로 하며, 코사인 유사도를 $\text{sim}(\cdot)$으로 사용한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋:** Mini-ImageNet, ImageNet-FS.
- **그래프 정보:** WordNet의 계층 구조(IS-A 관계)를 사용하여 DAG(Directed Acyclic Graph) 형태의 그래프를 구축하였다.
- **지표:** 평균 정확도(Average Accuracy) 및 Top-5 정확도를 측정하였다.

### 2. 정량적 결과

- **Mini-ImageNet:** $\text{S2M2}^R$ 모델에 그래프 정규화를 적용했을 때 1-shot에서 $66.93\%$, 5-shot에서 $83.35\%$의 정확도를 기록하며, 복잡한 그래프 모델인 KGTN보다 높은 성능을 보였다.
- **ImageNet-FS:** SGM 모델에 적용 시, 기본 모델 대비 최대 $6.7\%$의 성능 향상을 보였으며, 기존의 그래프 기반 모델인 KTCH나 KGTN을 모두 능가하는 결과를 얻었다.
- **Model-Agnostic 검증:** ProtoNet, LEO, $\text{S2M2}^R$라는 서로 다른 세 가지 계열의 모델 모두에서 성능이 일관되게 향상됨을 확인하였다. 특히 데이터가 매우 적은 1-shot 설정에서 이득이 더 컸다.

### 3. 합성 데이터 실험 (Synthetic Dataset)

이진 트리 구조의 합성 데이터를 통해 정규화의 효과를 분석하였다.

- **결정 경계(Decision Boundary) 분석:** Support set의 샘플이 Query set과 멀리 떨어져 있어 일반화가 어려운 상황에서, 그래프 정규화를 적용한 모델이 훨씬 더 정확한 결정 경계를 생성함을 시각적으로 확인하였다.
- **Task 난이도와의 관계:** 태스크의 난이도 $\Omega_\phi$ (쿼리 샘플이 잘못 분류될 확률의 log-odds 평균)를 정의하여 분석한 결과, 태스크가 어려울수록(Hardness가 높을수록) 그래프 정규화로 인한 성능 이득이 더 크게 나타났다.

## 🧠 Insights & Discussion

본 논문은 복잡한 GNN 아키텍처 없이도 단순한 정규화 항만으로 클래스 간의 관계 정보를 충분히 활용할 수 있음을 입증하였다. 특히 다음과 같은 통찰을 제공한다.

첫째, 그래프 정규화는 support set이 불충분하거나(low-shot), 제공된 샘플이 클래스의 대표성을 띠지 못해(uninformative) 학습이 어려운 상황에서 결정 경계를 가이드하는 강력한 제약 조건으로 작용한다.

둘째, 기존의 그래프 기반 FSL 모델들이 거둔 성과가 반드시 복잡한 아키텍처 덕분이 아니라, 단순히 클래스 간의 유사도를 보존하려는 목적 함수 덕분이었을 가능성을 시사한다.

한계점으로는 WordNet과 같은 외부 지식 그래프가 반드시 존재해야 한다는 점이 있으며, 그래프의 품질(계층의 깊이 등)이 성능에 직접적인 영향을 미친다는 점이 ablation study를 통해 확인되었다.

## 📌 TL;DR

이 논문은 Few-Shot Learning에서 클래스 간 관계 정보를 활용하기 위해, 특정 모델에 종속되지 않는 **Graph Regularization** 방법을 제안한다. `node2vec` 기반의 유사도 보존 손실 함수를 기존 모델의 목적 함수에 추가함으로써, Metric-based, Optimization-based, Fine-tuning 모델 모두에서 성능 향상을 이끌어냈다. 특히 데이터가 극히 적거나 난이도가 높은 태스크에서 매우 효과적이며, 복잡한 그래프 신경망 모델보다 더 단순하면서도 우수한 성능을 보였다. 이 연구는 향후 다양한 FSL 모델에 지식 그래프를 쉽게 통합할 수 있는 표준적인 방법론을 제시했다는 점에서 의의가 있다.
