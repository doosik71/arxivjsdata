# A SHORT NOTE ON THE DECISION TREE BASED NEURAL TURING MACHINE

Yingshi Chen (2020)

## 🧩 Problem to Solve

본 논문은 오랫동안 서로 독립적으로 발전해 온 두 가지 모델인 튜링 머신(Turing Machine)과 결정 트리(Decision Tree) 사이의 이론적 접점을 찾는 것을 목표로 한다. 구체적으로는 신경 튜링 머신(Neural Turing Machine, NTM)의 외부 메모리 구조와 미분 가능한 포레스트(Differentiable Forest)의 구조적 유사성에 주목한다.

기존의 NTM은 외부 메모리 뱅크를 읽고 쓰기 위해 미분 가능한 어텐션(Attention) 메커니즘을 사용하며, 대개 신경망을 컨트롤러(Controller)로 활용한다. 반면, 미분 가능한 결정 트리는 전통적인 결정 트리의 단순성과 해석 가능성을 유지하면서 역전파(Backpropagation)를 통한 학습이 가능하게 하여 특히 정형 데이터(Tabular data)에서 강력한 성능을 보인다. 저자는 미분 가능한 포레스트가 사실상 NTM의 특수한 사례라는 점을 증명하고, 이를 바탕으로 결정 트리 기반의 NTM 구조인 RaDF(Response augmented differential forest)를 제안함으로써 두 모델의 결합 가능성을 제시하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 미분 가능한 포레스트(Differentiable Forest)가 신경 튜링 머신(NTM)의 특수한 형태라는 이론적 연결 고리를 발견한 것이다.

중심적인 직관은 미분 가능한 결정 트리의 리프 노드(Leaf node)들이 NTM의 외부 메모리에 저장된 응답 벡터(Response vector)를 읽고 쓰는 헤드(Head) 역할을 수행한다는 점이다. 이러한 통찰을 통해 컨트롤러로는 미분 가능한 포레스트를 사용하고, 외부 메모리로는 리프 노드들에 대응하는 응답 벡터들을 저장하는 **Response augmented differential forest (RaDF)** 구조를 설계하였다.

## 📎 Related Works

논문에서는 NTM과 미분 가능한 결정 트리라는 두 가지 핵심 관련 연구를 소개한다.

첫째, NTM은 외부 메모리 뱅크를 통해 정보를 저장하고 검색하는 능력을 갖춘 재귀적 머신으로, Memory Augmented Neural Networks (MANN)나 Differentiable Neural Computer 등으로도 불린다. 이는 기존 LSTM 등의 모델보다 정보 저장 및 검색 능력이 뛰어나 기계 번역, 추천 시스템, SLAM 등 다양한 분야에 적용되었으나, 컨트롤러가 모두 일반적인 신경망으로 구성되었다는 특징이 있다.

둘째, 미분 가능한 결정 트리는 기존 GBDT(Gradient Boosted Decision Trees)와 같은 트리 모델이 가진 비미분성(Non-differentiability) 문제를 해결하여 SGD나 Adam과 같은 경사 하강법 최적화 알고리즘을 사용할 수 있게 한 연구들이다. Stochastic routing을 이용한 모델, NODE(Neural Oblivious Decision Ensembles), ANT(Adaptive Neural Trees), DNDT(Deep Neural Decision Trees) 등이 제안되었으며, 일부 연구에서는 이러한 미분 가능한 포레스트가 LightGBM이나 XGBoost 같은 최신 GBDT 라이브러리보다 높은 정확도를 보였다고 보고되었다.

저자는 기존의 미분 가능한 트리 연구들이 NTM과의 구조적 연관성을 인지하지 못했음을 지적하며, 본 연구가 이를 명시적으로 연결함으로써 모델 개선의 방향성을 제시한다고 주장한다.

## 🛠️ Methodology

### 전체 구조 및 파이프라인

RaDF는 크게 두 가지 모듈로 구성된다. 하나는 $K$개의 미분 가능한 결정 트리로 이루어진 **컨트롤러 $T$**이며, 다른 하나는 외부 메모리 뱅크인 **응답 뱅크 $Q$**이다. $Q$의 각 셀은 트리의 리프 노드들에 대응하는 응답 벡터를 저장한다. 입력 데이터 $x$가 들어오면 컨트롤러(트리들)가 경로 확률을 계산하고, 이를 통해 메모리 $Q$에서 응답을 읽어 최종 예측값을 도출한다.

### 주요 구성 요소 및 동작 원리

**1. 게이팅 함수 (Gating Function)**
각 내부 노드에서 샘플 $x$가 왼쪽 또는 오른쪽 자식 노드로 갈 확률을 결정하기 위해 미분 가능한 게이팅 함수 $g$를 사용한다. 학습 가능한 파라미터 $A$와 임계값 $b$에 대해 다음과 같이 정의된다.
$$g(A, x, b) = \sigma(Ax - b)$$
여기서 $\sigma$는 시그모이드(Sigmoid) 함수와 같은 활성화 함수이다.

**2. 리프 노드 도달 확률 (Leaf Node Probability)**
루트 노드부터 리프 노드 $j$까지의 경로에 있는 모든 노드의 게이팅 값들의 곱으로 리프 노드 $j$에 도달할 확률 $p_j$를 계산한다.
$$p_j = \prod_{n \in \{n_1, \dots, n_d\}} g_n$$

**3. 메모리 읽기 및 예측 (Read Operation & Prediction)**
각 리프 노드 $j$는 외부 메모리 $Q$에서 응답 벡터 $q_j$를 읽어온다. 단일 트리 $h$의 출력 $Q_h(x)$는 각 리프 노드 응답의 가중 평균으로 계산된다.
$$Q_h(x) = \sum_{j \in \text{leaf of } h} p_j q_j$$
최종 예측값 $\hat{y}$는 $K$개 트리의 평균 결과로 산출된다.
$$\hat{y}(x) = \frac{1}{K} \sum_{h=1}^{K} Q_h(x)$$

### 학습 절차 및 손실 함수

전체 파라미터 $\Theta = (A, b, Q)$에 대해 다음과 같은 일반적인 손실 함수 $L$을 최소화하는 방향으로 학습한다.
$$L(\Theta : x, y) = \frac{1}{K} \sum_{h=1}^{K} L_h(A, b, Q : x, y)$$
분류 문제의 경우 교차 엔트로피(Cross-entropy)를, 회귀 문제의 경우 MSE나 MAE 등을 사용한다. 최적화는 SGD(Stochastic Gradient Descent)를 통해 수행하며, NTM의 업데이트 방식에 따라 응답 뱅크 $Q$를 다음과 같이 업데이트한다.
$$Q^i_t = Q^i_{t-1} [1 - w^i_t e_t] + w^i_t a_t$$
여기서 $w^i_t$는 가중치, $e_t$는 erase 벡터, $a_t$는 add 벡터이다.

## 📊 Results

본 논문은 이론적인 연결 고리를 제시하는 'Short Note' 형식의 논문으로, 새로운 벤치마크 실험 결과나 정량적인 성능 지표를 구체적으로 제시하고 있지는 않는다.

다만, 서론과 관련 연구 섹션에서 미분 가능한 포레스트 모델이 기존의 최적화된 GBDT 라이브러리(LightGBM, Catboost, XGBoost)보다 높은 정확도를 보였다는 기존 연구[19]를 인용하며, RaDF의 기반이 되는 구조적 우수성을 간접적으로 언급한다. 저자는 상세한 실험 결과는 향후 후속 논문에서 다룰 것임을 명시하였다.

## 🧠 Insights & Discussion

본 논문은 서로 다른 영역에서 발전한 NTM과 미분 가능한 결정 트리를 하나의 프레임워크로 통합했다는 점에서 이론적 가치가 있다. 특히 NTM의 컨트롤러를 일반적인 신경망이 아닌 결정 트리로 대체함으로써, NTM이 가질 수 있는 블랙박스적 특성을 완화하고 정형 데이터에 최적화된 메모리 네트워크 구조를 설계할 수 있는 가능성을 열었다.

하지만 논문에서 제시된 RaDF의 구체적인 성능 향상 폭이 실험적으로 검증되지 않았다는 점은 한계로 남는다. 또한, NTM의 핵심인 '동적 메모리 업데이트(erase/add 벡터를 이용한 쓰기)'가 결정 트리의 정적인 응답 뱅크 구조에서 구체적으로 어떤 이점을 주는지에 대한 심층적인 분석이 부족하다. 단순히 리프 노드가 메모리 셀에 대응된다는 정의를 넘어, 트리 기반 컨트롤러가 신경망 컨트롤러보다 메모리 주소 지정(Addressing) 측면에서 어떤 효율성을 가지는지에 대한 논의가 추가될 필요가 있다.

## 📌 TL;DR

본 논문은 미분 가능한 포레스트가 신경 튜링 머신(NTM)의 특수한 사례임을 밝히고, 이를 기반으로 결정 트리를 컨트롤러로, 리프 노드 응답을 외부 메모리로 사용하는 **Response augmented differential forest (RaDF)**를 제안한다. 이는 메모리 증강 신경망의 컨트롤러를 트리 구조로 확장하여, 특히 정형 데이터 처리에서 해석 가능성과 성능을 동시에 잡을 수 있는 새로운 연구 방향을 제시한다.
