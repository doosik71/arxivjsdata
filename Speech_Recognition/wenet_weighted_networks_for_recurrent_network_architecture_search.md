# WeNet: Weighted Networks for Recurrent Network Architecture Search

Zhiheng Huang, Bing Xiang (2019)

## 🧩 Problem to Solve

본 논문은 딥러닝 모델의 성능을 결정짓는 핵심 요소인 네트워크 구조(Architecture)를 수동으로 설계하는 대신, 자동으로 탐색하는 Neural Architecture Search(NAS)를 recurrent network(순환 신경망) 분야에 효율적으로 적용하는 문제를 해결하고자 한다.

기존의 NAS 방식들은 이미지 분류나 언어 모델링에서 우수한 성과를 거두었으나, 탐색 공간이 너무 넓어 막대한 연산 자원이 소모되거나(RL, EA 기반), 연속적인 완화(continuous relaxation)를 사용하는 과정에서 복잡한 파라미터 공유 및 교차 업데이트 과정이 필요하다는 한계가 있었다. 특히 recurrent network의 경우, 긴 의존성(long-distance dependency)을 모델링할 수 있는 최적의 셀(cell) 구조를 찾는 것이 매우 중요하며, 이를 효율적이고 정확하게 수행하는 것이 본 연구의 목표이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Weighted Networks (WeNets)**라는 개념을 도입하여, 여러 개의 후보 네트워크에 가중치를 부여하고 이를 경사 하강법(Gradient Descent)으로 최적화하는 것이다.

중심적인 직관은 Mixture of Experts(MoE)와 유사하게, 다양한 구조의 네트워크들을 병렬로 배치하고 각 네트워크의 중요도를 나타내는 가중치 $w$를 학습시키는 것이다. 이때 모델의 파라미터와 네트워크의 가중치를 동시에 학습함으로써, 어떤 구조가 해당 작업에 가장 적합한지를 데이터 기반으로 직접적으로 찾아낼 수 있다. 또한, 'network batch size'라는 개념을 도입하여 대규모의 후보군을 효율적으로 탐색할 수 있는 알고리즘을 제안하였다.

## 📎 Related Works

본 논문은 특히 **DARTS (Differentiable Architecture Search)**와 비교하여 차별점을 제시한다.

1. **탐색 공간의 정의**: DARTS는 가능한 모든 연산의 연속적인 완화(continuous relaxation)를 통해 전체 탐색 공간을 고려하지만, WeNet은 미리 정의된 특정 네트워크 집합 내에서 가중치를 학습함으로써 탐색 공간을 제한하여 효율성을 높인다.
2. **파라미터 공유(Parameter Sharing)**: DARTS는 메모리 한계를 극복하기 위해 파라미터를 공유하지만, WeNet은 파라미터를 공유하지 않는다. 저자들은 파라미터 공유 시 서로 다른 노드와 연산자가 동일한 파라미터를 업데이트하면서 발생하는 '업데이트 충돌(updating collision)'이 성능 저하를 유발한다고 주장하며, 비공유 방식이 더 정확한 중요도 평가를 가능하게 한다고 설명한다.
3. **데이터셋 활용**: DARTS는 구조 파라미터를 업데이트하기 위해 검증 데이터셋(validation set)이 별도로 필요하지만, WeNet은 훈련 데이터셋만으로도 가중치 업데이트가 가능하다.
4. **업데이트 방식**: DARTS는 모델 파라미터와 구조 파라미터를 교대로 업데이트해야 하지만, WeNet은 이를 동시에 업데이트하여 절차를 단순화한다.

또한, **Mixture of Experts (MoE)** 연구와도 관련이 있으나, MoE가 추론 시점에 gating network를 통해 전문가를 선택하는 것과 달리, WeNet은 탐색 과정에서 최적의 단일 구조를 찾기 위한 가이드로 가중치를 사용하며, 최종 추론 시에는 가중치를 제거하고 선택된 단일 네트워크만 사용한다.

## 🛠️ Methodology

### 1. Network Search Space

본 논문에서 정의하는 recurrent cell은 $L$개의 노드로 구성된 방향성 비순환 그래프(Directed Acyclic Graph, DAG) 형태이다.

- **입력 및 초기 노드**: 현재 시점의 입력 $x_t$와 이전 은닉 상태 $h_{t-1}$을 입력으로 받는다. 초기 노드 $s_0$는 다음과 같이 계산된다.
  $$c = \sigma(W_x x_t), \quad h = \tanh(W_h h_{t-1}), \quad s_0 = h_{t-1} + c(h - h_{t-1})$$
- **중간 노드 계산**: 각 노드 $i$ ($i=1, \dots, L-1$)는 이전 노드 $j \in \{0, \dots, i-1\}$ 중 하나를 조상 노드로 선택하고, 활성화 함수 $o_i$를 적용하여 상태를 계산한다.
  $$s_i = o_i(W_{ji} s_j)$$
  여기서 $o_i$는 $\tanh, \text{relu}, \text{sigmoid}, \text{identity}$ 중 하나이다.
- **최종 출력**: 셀의 최종 출력 $h_t$는 모든 중간 노드의 평균으로 구한다.
  $$h_t = \frac{1}{L-1} \sum_{i=1}^{L-1} s_i$$

### 2. Weighted Networks (WeNets)

WeNet은 $n$개의 후보 네트워크 $\{N_0, \dots, N_{n-1}\}$와 각각의 가중치 $\{w_0, \dots, w_{n-1}\}$로 구성된다. 모든 후보 네트워크는 동일한 입력을 받고 동일한 크기의 출력을 내놓는다. 최종 출력 $y$는 다음과 같이 각 네트워크 출력의 가중 합으로 계산된다.
$$y = \sum_{i=1}^{n} w_i N_i(x)$$
여기서 가중치 $w_i$는 softmax 함수를 통해 정규화되며, 이는 각 네트워크의 중요도를 의미한다.

### 3. Architecture Search Algorithm

탐색 과정은 크게 두 단계로 나뉜다.

**Step 1: Random Network Generation**

- 총 $T$개의 네트워크를 무작위로 생성한다. 각 네트워크의 각 레벨 $l$에 대해 조상 노드 $node \in \{0, \dots, l\}$와 활성화 함수 $op \in \{\tanh, \text{relu}, \text{sigmoid}, \text{identity}\}$를 무작위로 샘플링하여 구조를 확정한다.

**Step 2: Search Algorithm**

- **입력**: 전체 탐색 대상 수 $T$, 네트워크 배치 크기 $B$, 시드 네트워크 크기 $K$.
- **절차**:
  1. 전체 후보군(pool)에서 $B$개의 네트워크를 추출하여 `candidates`를 구성한다.
  2. 이전 단계에서 살아남은 상위 $K$개의 시드 네트워크(`seed`)를 `candidates`에 추가한다.
  3. 구성된 WeNet 구조를 사용하여 훈련 데이터로 가중치 $w$와 모델 파라미터를 동시에 학습시킨다.
  4. 학습 후 가중치 $w$가 가장 높은 상위 $K$개의 네트워크를 새로운 `seed`로 선정한다.
  5. 모든 $T$개의 네트워크를 처리할 때까지 반복하며, 최종적으로 가장 가중치가 높은 네트워크를 반환한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: Penn Treebank (PTB), WikiText-2 (WT2)
- **지표**: Perplexity (pplx) $\rightarrow$ 값이 낮을수록 성능이 좋음.
- **설정**: $T=10\text{k}, B=100, K=20$, 셀 노드 수 $L=8$. 탐색 시에는 각 노드에 Batch Normalization을 적용하여 그래디언트 폭주를 방지하였다.

### 2. 정량적 결과

- **Penn Treebank (PTB)**:
  - 일반적인 Random 구조는 61.5 pplx를 기록하였다.
  - DARTS(second order)는 56.1 pplx를 기록하여 매우 강력한 성능을 보였다.
  - WeNet은 1,500 epoch 학습 시 57.9 pplx로 ENAS와 유사한 수준이었으나, **6,000 epoch까지 학습했을 때 54.87 pplx라는 새로운 SOTA(State-of-the-art) 성과를 달성**하였다.
- **WikiText-2 (WT2) 전이 성능**:
  - PTB에서 발견한 구조를 WT2에 적용했을 때, WeNet은 66.6 pplx를 기록하여 ENAS(70.4), DARTS(66.9), NAONet(67.0)보다 우수한 전이 능력을 보여주었다.

### 3. 효율성

- 전체 탐색 비용(4회 반복 실행 기준)은 1 GPU day 이내로, ENAS나 DARTS와 비슷하며 기존 RL 기반 NAS보다는 훨씬 빠르다.

## 🧠 Insights & Discussion

### 1. 강점 및 분석

- **장기 학습 안정성**: WeNet 구조는 DARTS 구조와 초기 2,000 epoch까지는 유사하거나 DARTS가 더 우세한 경향을 보였으나, 그 이후부터는 WeNet이 지속적으로 성능이 향상되는 양상을 보였다. 이는 WeNet이 찾은 구조가 더 깊은 최적화 가능성을 가지고 있음을 시사한다.
- **구조적 공통점**: WeNet과 DARTS가 발견한 구조를 비교했을 때, 레벨 0부터 4까지의 노드 및 연산자 쌍이 완전히 일치하는 부분 구조(substructure)가 발견되었다. 이는 해당 구조가 시퀀스 데이터의 장거리 모델링에 매우 효과적임을 의미한다.
- **파라미터 비공유의 이점**: 파라미터 공유를 하지 않음으로써 각 후보 네트워크의 순수한 성능을 평가할 수 있었고, 이것이 결과적으로 더 나은 구조 발견으로 이어졌다고 분석한다.

### 2. 한계 및 비판적 해석

- **학습 시간의 의존성**: WeNet이 SOTA 성능을 내기 위해서는 6,000 epoch라는 매우 긴 학습 시간이 필요했다. 이는 구조 탐색 자체는 효율적일지 몰라도, 최종 모델의 성능을 끌어올리기 위한 훈련 비용이 크다는 점을 보여준다.
- **탐색 공간의 제한**: 본 연구는 무작위로 생성된 $T$개의 네트워크 집합 내에서 최적을 찾는 방식이므로, 이론적으로 가능한 모든 조합을 탐색하는 DARTS 방식보다 최적해를 놓칠 가능성이 존재한다. 다만, 본 논문에서는 이 제한된 탐색 공간 내에서도 충분히 강력한 구조를 찾을 수 있음을 입증하였다.

## 📌 TL;DR

본 논문은 Recurrent Network의 구조 탐색을 위해 **Weighted Networks (WeNet)**라는 새로운 프레임워크를 제안한다. 여러 후보 구조에 가중치를 부여하고 이를 모델 파라미터와 함께 SGD로 학습시켜 최적의 구조를 찾는 방식이다. 이 방법은 파라미터 공유 없이 효율적인 '네트워크 배치' 탐색을 수행하며, 결과적으로 Penn Treebank 데이터셋에서 **54.87 pplx라는 SOTA 성능**을 달성하고 타 데이터셋으로의 **우수한 전이 능력**을 보였다. 이 연구는 복잡한 연속적 완화 없이도 가중치 기반의 단순한 접근법으로 고성능의 순환 신경망 구조를 찾을 수 있음을 증명하였다.
