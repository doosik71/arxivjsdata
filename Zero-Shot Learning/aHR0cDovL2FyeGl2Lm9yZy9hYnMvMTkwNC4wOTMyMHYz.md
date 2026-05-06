# Context-Aware Zero-Shot Recognition

Ruotian Luo, Ning Zhang, Bohyung Han, Linjie Yang (2019)

## 🧩 Problem to Solve

본 논문은 Zero-Shot Learning (ZSL) 분야에서 기존 방법론들이 가진 한계점을 해결하고자 한다. 전통적인 ZSL 방식은 주로 학습 데이터에 포함되지 않은 Unseen category의 객체를 인식하기 위해, Semantic similarity가 높은 Seen category로부터 지식을 전이(Knowledge transfer)하는 방식에 의존한다. 즉, 단일 객체의 시각적 특성과 시맨틱 임베딩 간의 관계만을 고려하여 독립적으로 클래스를 추론한다.

하지만 인간은 새로운 객체를 마주했을 때, 단순히 객체 자체의 외형뿐만 아니라 주변 객체들과의 관계 및 장면의 맥락(Context)을 활용하여 정체를 추론하는 능력을 가지고 있다. 예를 들어, 처음 보는 붉은 원반 형태의 객체가 있더라도, 주변에 사람과 개가 있고 이들이 함께 놀고 있다는 맥락이 있다면 이를 '프리스비(frisbee)'라고 추론할 수 있다.

따라서 본 논문의 목표는 객체 간의 관계 prior(inter-object relation prior)와 시각적 맥락을 활용하여 Unseen category의 객체를 더 정확하게 인식하는 **Context-Aware Zero-Shot Recognition** 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 객체 개별의 특성(Instance-level)뿐만 아니라, 이미지 내 객체 쌍(Pairwise) 간의 기하학적 관계와 지식 그래프(Knowledge Graph)에 정의된 관계 정보를 결합하여 추론하는 것이다.

이를 위해 저자들은 다음과 같은 설계를 제안한다:

1. **CRF(Conditional Random Field) 기반의 통합 추론**: 개별 객체의 클래스 확률을 나타내는 Unary potential과 객체 간의 관계 가능성을 나타내는 Pairwise potential을 CRF 구조로 통합하여, 이미지 내 모든 객체의 레이블을 공동으로 추론(Joint reasoning)한다.
2. **관계 지식 그래프의 활용**: $\langle \text{subject, predicate, object} \rangle$ 형태의 튜플로 구성된 관계 지식 그래프를 구축하여, 특정 클래스 쌍 사이에 존재 가능한 관계(Relation)를 정의하고 이를 추론 과정에 반영한다.
3. **기하학적 특징 기반의 관계 추론**: 객체 간의 상대적 위치와 크기 정보를 이용한 기하학적 구성 특징(Geometric configuration feature)을 통해 관계 가능성을 예측하는 MLP 모델을 설계하였다.

## 📎 Related Works

### 1. Zero-Shot Learning (ZSL)

기존 ZSL 연구들은 주로 Attribute, Semantic embedding, Knowledge graph(예: WordNet) 등을 사용하여 Seen $\to$ Unseen category로의 지식 전이를 수행하였다. 최근에는 GCN(Graph Convolutional Network)을 이용해 분류기 가중치를 전파하는 방식이 제안되었다. 그러나 이러한 방법들은 객체를 독립적으로 처리하며, 장면 내의 시각적 맥락을 고려하지 않는다.

### 2. Context-Aware Detection

객체 검출(Object Detection) 분야에서는 이미 오래전부터 주변 맥락 정보를 활용해 왔다. 최근의 딥러닝 기반 접근법(예: Scene Graph Generation) 또한 객체 간 관계를 학습하여 검출 성능을 높인다. 하지만 이러한 방법들은 대부분 Fully-supervised 설정에서 설계되었으며, 학습 데이터가 없는 Unseen category에 대해서는 적용하기 어렵다는 한계가 있다.

### 3. Knowledge Graphs

지식 그래프는 이미지 분류, 시각적 추론 등 다양한 비전 작업에 활용되어 왔다. 본 논문은 이러한 지식 그래프의 개념을 ZSL의 관계 추론 단계에 도입하여, 정답 레이블이 없는 상황에서도 객체 간의 상식적인 관계를 통해 클래스를 보정할 수 있도록 하였다.

## 🛠️ Methodology

### 전체 파이프라인

본 프레임워크는 이미지와 객체 바운딩 박스 $\{B_i\}$를 입력으로 받아 각 지역의 클래스 $c_i$를 예측한다. 전체 과정은 **Instance-level ZSL 추론 $\to$ Relationship 추론 $\to$ CRF 통합 추론** 순으로 진행된다.

### CRF 모델 정의

이미지 내 $N$개 객체의 클래스 할당 확률은 다음과 같은 CRF 에너지 함수로 정의된다:
$$P(c_1 \dots c_N | B_1 \dots B_N) \propto \exp \left( \sum_{i} \theta(c_i | B_i) + \gamma \sum_{i \neq j} \phi(c_i, c_j | B_i, B_j) \right)$$
여기서 $\theta$는 Unary potential, $\phi$는 Pairwise potential이며, $\gamma$는 두 잠재 함수의 균형을 조절하는 가중치이다.

### 주요 구성 요소

**1. Instance-level Zero-Shot Inference (Unary Potential)**

- Fast R-CNN 기반의 네트워크를 통해 각 지역의 특징 $f_i \in \mathbb{R}^{d_f}$를 추출한다.
- 클래스 확률 $P_c(c_i) = \text{softmax}(W f_i)$를 계산하며, 여기서 $W$는 Seen category를 위해 학습된 $W_S$와 외부 지식을 통해 추정된 Unseen category를 위한 $W_U$의 결합으로 구성된다.
- Unary potential은 $\theta_i(c_i) = \log P_c(c_i | B_i)$로 정의된다.

**2. Relationship Inference (Pairwise Potential)**

- 두 객체 $B_i, B_j$ 사이의 상대적 기하학적 특징 $g_{ij}$를 다음과 같이 정의하여 평행 이동 및 스케일 불변성을 확보한다:
$$g_{ij} = \left[ \log \frac{|x_i - x_j|}{w_i}, \log \frac{|y_i - y_j|}{h_i}, \log \frac{w_j}{w_i}, \log \frac{h_j}{h_i} \right]^\top$$
- 이 특징 $g_{ij}$를 MLP에 통과시켜 특정 관계 $\hat{r}_k$가 존재할 확률인 관계 잠재 함수 $\ell(\hat{r}_k | B_i, B_j)$를 얻는다.
- 최종 Pairwise potential $\phi$는 지식 그래프 내에 $\langle c_i, \hat{r}_k, c_j \rangle$ 튜플이 존재하는지 확인하는 지시 함수 $\delta$를 사용하여 계산한다:
$$\phi(c_i, c_j | B_i, B_j) = \sum_{k} \delta(\hat{r}_k; c_i, c_j) \ell(\hat{r}_k; B_i, B_j)$$

**3. 학습 절차 및 손실 함수**

- **1단계**: Seen category에 대해 Instance-level ZSL 모듈을 학습한다.
- **2단계**: Relationship 추론 모듈을 학습한다. 이때 관계에 대한 정답 레이블이 없으므로, 다른 객체들의 정답 레이블이 주어졌을 때 해당 객체의 레이블을 맞추는 Pseudo-likelihood 기반의 손실 함수를 사용한다:
$$L = -\sum_{i} \log P(c^*_i | c^*_{\setminus i})$$

### 추론 (Inference)

최종 예측은 MAP(Maximum A Posteriori) 추론을 통해 수행하며, 효율적인 계산을 위해 Mean Field Inference 알고리즘을 사용한다. 또한 계산 복잡도를 줄이기 위해 Instance-level 추론에서 상위 $K$개의 클래스만 선택하여 재정렬하는 Pruning 기법을 적용한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Visual Genome (VG) 데이터셋을 사용하였다 (Seen 478개, Unseen 130개 클래스).
- **평가 지표**: Per-class 및 Per-instance Accuracy를 측정하였으며, Generalized ZSL 설정에서는 Seen/Unseen 성능의 조화 평균(Harmonic Mean, HM)을 측정하였다.
- **비교 대상**: Word Embedding (WE), CONSE, GCN, SYNC 등 기존 ZSL 베이스라인 모델들을 사용하고, 여기에 제안하는 Context-aware inference를 추가하여 성능 변화를 관찰하였다.

### 정량적 결과

- **전반적 성능 향상**: 모든 베이스라인 모델에서 Context-aware 모듈을 추가했을 때 Unseen category에 대한 인식 성능이 유의미하게 향상되었다.
- **Generalized ZSL에서의 효과**: 특히 GCN과 SYNC 모델과 결합했을 때, Seen/Unseen 모두에서 성능이 향상되었으며 HM 수치가 크게 증가하였다.
- **Top-K Refinement**: Instance-level 추론의 Top-5 결과 중 실제 정답이 포함되어 있을 때, 관계 정보를 통해 Top-1 정답을 정확하게 찾아내는 재정렬(Reranking) 능력이 입증되었다.
- **Zero-Shot Detection**: EdgeBoxes 제안 영역을 입력으로 사용한 검출 작업에서도 Recall@100 지표가 향상됨을 확인하였다.

### 정성적 결과

- 단순한 시각적 특징만으로는 구분하기 어려운 객체(예: Pie $\to$ Pizza, Furniture $\to$ Chair)를 주변 객체(예: 소시지, 타포린 등)와의 관계를 통해 정확한 클래스로 보정하는 사례가 확인되었다.

## 🧠 Insights & Discussion

### 강점 및 분석

- **기하학적 정보의 중요성**: 관계 추론 시 외형 특징(Appearance feature)을 추가하는 것보다 기하학적 정보($+G$)만을 사용하는 것이 Unseen category 성능에 더 유리했다. 이는 외형 특징을 사용할 경우 Seen category에 과적합(Overfitting)되어 Unseen category로의 일반화 성능이 떨어지기 때문으로 분석된다.
- **그래프 연결도의 영향**: 분석 결과, 관계 지식 그래프에서 연결도(Degree)가 높은 클래스일수록, 그리고 테스트 셋에서 등장 빈도가 높은 클래스일수록 Context-aware 추론을 통한 성능 향상 폭이 컸다. 이는 활용 가능한 관계 정보가 많을수록 추론의 근거가 명확해짐을 시사한다.

### 한계 및 논의

- **데이터 의존성**: 본 모델은 관계 지식 그래프에 의존한다. 만약 그래프에 정의되지 않은 새로운 관계가 나타나거나 그래프 자체가 불완전할 경우 추론 성능이 제한될 수 있다.
- **계산 복잡도**: 객체 수가 많은 이미지의 경우 CRF 추론 비용이 증가하며, 이를 위해 Top-K pruning을 사용하고 있으나 이는 근본적인 계산량 감소보다는 근사치에 가깝다.

## 📌 TL;DR

본 논문은 객체를 독립적으로 인식하던 기존 ZSL의 한계를 넘어, **객체 간의 기하학적 관계와 관계 지식 그래프를 CRF 프레임워크로 통합하여 Unseen category를 인식하는 Context-Aware ZSL**을 제안하였다. 실험을 통해 시각적 맥락 정보가 ZSL 성능, 특히 Generalized ZSL 설정에서 정답을 재정렬하는 데 매우 효과적임을 입증하였다. 이 연구는 향후 Few-shot learning이나 Scene Graph 기반의 고차원 시각 인식 연구에 중요한 기초를 제공할 것으로 기대된다.
