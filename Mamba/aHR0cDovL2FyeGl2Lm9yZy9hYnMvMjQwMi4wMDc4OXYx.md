# Graph-Mamba: Towards Long-Range Graph Sequence Modeling with Selective State Spaces

Chloe Wang, Oleksii Tsepa, Jun Ma, Bo Wang (2024)

## 🧩 Problem to Solve

본 논문은 그래프 데이터에서 노드 간의 장거리 의존성(long-range dependencies)을 효율적으로 모델링하는 문제를 해결하고자 한다. 기존의 Graph Transformer는 모든 노드 간의 상호작용을 계산하는 Attention 메커니즘을 사용하여 강력한 성능을 보이지만, 연산 복잡도가 노드 수의 제곱인 $O(N^2)$에 비례하여 대규모 그래프에 적용하기에는 확장성(scalability) 문제가 심각하다.

이를 해결하기 위해 기존 연구들은 무작위 샘플링이나 휴리스틱 기반의 Sparse Attention을 도입하여 연산량을 줄이려 했다. 그러나 이러한 방식은 데이터의 특성에 따라 동적으로 문맥을 추론하는 데이터 의존적 문맥 추론(data-dependent context reasoning) 능력이 부족하다는 한계가 있다. 따라서 본 연구의 목표는 Mamba와 같은 Selective State Space Models(SSMs)를 그래프 네트워크에 통합하여, 연산 효율성을 유지하면서도 입력 데이터에 따라 유연하게 문맥을 선택하여 처리하는 Graph-Mamba를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 순차 데이터 모델링에 최적화된 Mamba의 선택적 상태 공간 모델(Selective SSM)을 비순차적인 그래프 구조에 맞게 변형하여 적용하는 것이다.

1. **입력 의존적 노드 필터링:** Mamba의 선택 메커니즘을 통해 장거리 문맥 중에서 중요한 정보만을 선택적으로 필터링하여 반영함으로써, 기존의 무작위 샘플링 기반 Sparse Attention보다 정교한 문맥 추론을 가능하게 한다.
2. **그래프 특화 적응 전략:** SSM의 단방향성(unidirectionality) 문제를 해결하기 위해 노드의 중요도(예: degree)에 따라 순서를 정하는 노드 우선순위 지정(Node Prioritization) 기법과, 순서 편향을 줄이기 위한 순열 기반 학습(Permutation-based training) 레시피를 제안하였다.
3. **효율성 및 성능의 동시 달성:** 선형 시간 복잡도 $O(L)$를 달성하여 대규모 그래프에서도 메모리 사용량과 연산량(FLOPs)을 획기적으로 줄이면서도, 최신 SOTA 모델들과 대등하거나 더 우수한 예측 성능을 보였다.

## 📎 Related Works

### Graph Neural Networks (GNNs)

GCN, GIN, GAT와 같은 메시지 패싱 신경망(MPNN)은 인접 노드 간의 정보를 집계하여 지역적 구조를 잘 포착한다. 하지만 이들은 1-WL 동형 테스트라는 표현력의 한계가 있으며, 층이 깊어질수록 노드 특징이 유사해지는 Over-smoothing 문제로 인해 장거리 의존성을 포착하는 데 어려움이 있다.

### Graph Transformers

Transformer의 Attention 메커니즘은 모든 노드가 서로 상호작용할 수 있게 하여 장거리 연결성을 효과적으로 모델링한다. 그러나 $O(N^2)$의 복잡도로 인해 대형 그래프에서는 메모리 부족(OOM) 문제가 발생한다. 이를 해결하기 위해 BigBird나 Performer 같은 Sparse Attention 기법이 제안되었으나, 이들은 주로 순차 데이터용으로 설계되어 그래프 구조에 직접 적용 시 성능 저하가 발생하거나, 단순한 무작위 샘플링에 의존하는 경향이 있다.

### State Space Models (SSMs)

S4와 같은 구조적 상태 공간 모델은 선형 시간 복잡도로 장거리 시퀀스를 모델링할 수 있다. 최신 모델인 Mamba는 여기에 선택 메커니즘(Selection Mechanism)을 추가하여 입력 값에 따라 상태 업데이트를 동적으로 제어할 수 있게 하였다. 본 논문은 이러한 Mamba의 특성을 그래프의 노드 선택 과정으로 해석하여 적용하였다.

## 🛠️ Methodology

### 전체 시스템 구조

Graph-Mamba는 기존의 모듈형 프레임워크인 GraphGPS를 기반으로 하며, 기존의 Attention 모듈을 **Graph-Mamba Block (GMB)**으로 대체한다. 전체 파이프라인은 다음과 같이 구성된다.

1. **입력 단계:** Structural Encoding(SE)과 Positional Encoding(PE)을 노드 및 엣지 임베딩에 결합한다.
2. **GMB 레이어 (K개 적층):**
    * **MPNN 모듈:** GatedGCN 등을 사용하여 지역적 문맥을 집계하고 노드/엣지 임베딩을 업데이트한다.
    * **GMB 모듈:** Mamba 블록을 사용하여 글로벌 문맥을 모델링하고 노드 임베딩을 업데이트한다.
    * **결합:** MPNN과 GMB의 출력을 MLP를 통해 통합하여 최종 노드 임베딩을 생성한다.

### Mamba의 선택적 SSM 메커니즘

기본적인 SSM은 다음과 같은 선형 상미분 방정식(ODE)으로 정의된다.
$$h'(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t)$$

이를 이산화(discretization)하면 다음과 같은 재귀식으로 표현된다.
$$h_t = \bar{A}h_{t-1} + \bar{B}x_t$$
$$y_t = Ch_t$$

여기서 Mamba의 핵심인 **선택 메커니즘**은 매개변수 $\bar{A}, \bar{B}, C$를 입력 $x$의 함수로 만들어 데이터에 따라 문맥을 선택하게 하는 것이다. 구체적으로 이산화 단계 크기 $\Delta$를 입력의 함수로 정의함으로써, RNN의 게이팅 메커니즘과 유사하게 현재 입력 $x_t$가 이전 문맥 $h_{t-1}$을 얼마나 유지하고 새로운 정보를 얼마나 받아들일지를 결정한다.

### 그래프 적응 전략 (Graph Adaptation)

Mamba는 순차적으로 데이터를 스캔하므로, 시퀀스의 뒤쪽에 위치한 노드일수록 더 많은 이전 노드의 문맥을 참조할 수 있다. 이를 그래프에 적용하기 위해 두 가지 전략을 사용한다.

1. **노드 우선순위 지정 (Node Prioritization):**
    그래프를 시퀀스로 변환할 때, 노드 차수(degree)와 같은 휴리스틱을 사용하여 중요한 노드를 시퀀스의 뒤쪽에 배치한다. 이를 통해 중요한 노드가 더 많은 문맥 정보에 접근할 수 있도록 하여 정보 손실을 최소화한다.
2. **순열 기반 학습 및 추론 (Permutation Recipe):**
    * **학습 시:** 동일한 차수를 가진 노드들 사이에는 무작위 노이즈를 추가하여 순서를 섞음으로써, 특정 순서에 과적합되는 것을 방지하고 순열 불변성(permutation invariance)을 높인다.
    * **추론 시:** 여러 번의 무작위 순열을 적용해 GMB를 실행하고 그 결과값들을 평균 내어 최종 출력을 얻는다.

## 📊 Results

### 실험 설정

* **데이터셋:** Long-Range Graph Benchmark (LRGB) 및 GNN Benchmark의 10개 데이터셋을 사용하였다. 특히 노드 수가 150~1,400개에 달하는 대규모 그래프 데이터셋(Peptides-Func, MalNet-Tiny 등)에 집중하였다.
* **비교 대상:** GCN, GIN, GatedGCN, GPS+Transformer, GPS+Performer, GPS+BigBird, Exphormer.
* **지표:** Accuracy, F1 score, MAE, AP 등 작업별 최적 지표를 사용하였다.

### 주요 결과

1. **예측 성능:** 대규모 그래프 데이터셋 5종 중 4종에서 Graph-Mamba가 SOTA 성능을 기록하였다. 특히 Sparse Attention 방식인 Exphormer보다 높은 성능을 보였으며, 이는 데이터 의존적인 선택 메커니즘의 우수성을 입증한다.
2. **연산 효율성:**
    * **FLOPs 및 메모리:** MalNet-Tiny 데이터셋 실험 결과, Transformer의 비용이 노드 수에 따라 이차 함수적으로 증가하는 반면, Graph-Mamba는 선형적으로 증가하였다.
    * **메모리 절감:** Transformer 대비 GPU 메모리 사용량을 최대 74%까지 줄였으며, Exphormer와 비교해도 약 40%의 메모리 및 연산량 감소를 달성하였다.
3. **절제 연구 (Ablation Study):**
    * 단순한 무작위 순열(Node Level Permutation)만 적용해도 기본 모델보다 성능이 크게 향상되었다.
    * 여기에 노드 차수 기반의 우선순위 지정(Degree Prioritization)을 결합했을 때 가장 높은 성능을 보여, 그래프 특화 설계의 중요성이 확인되었다.

## 🧠 Insights & Discussion

본 논문은 순차 데이터 모델링의 정점인 Mamba를 그래프라는 비순차 구조에 성공적으로 이식하였다. 가장 주목할 점은 SSM의 **단방향 스캔 특성**을 오히려 활용하여, 중요 노드에게 더 많은 문맥을 부여하는 '우선순위 지정' 전략으로 승화시킨 점이다.

**강점:**

* Transformer의 표현력과 SSM의 효율성을 동시에 확보하였다.
* 하드웨어 가속(SRAM 활용)이 적용된 Mamba 구현을 통해 실제 추론 및 학습 속도를 획기적으로 개선하였다.

**한계 및 논의사항:**

* 그래프를 시퀀스로 변환(Flattening)하는 과정에서 정보 손실이 발생할 수 있으며, 본 논문에서는 차수(degree)라는 단순 휴리스틱에 의존하였다.
* 향후 연구에서는 데이터를 통해 최적의 시퀀스 변환 전략을 학습하거나, 그래프 토폴로지를 더 정교하게 주입하는 방법이 필요할 것으로 보인다.

## 📌 TL;DR

Graph-Mamba는 **Selective SSM(Mamba)**을 그래프 네트워크에 도입하여, 기존 Graph Transformer의 $O(N^2)$ 복잡도 문제를 해결하고 **선형 시간 복잡도 $O(L)$**를 달성한 모델이다. 특히 **노드 우선순위 지정**과 **순열 기반 학습**이라는 그래프 특화 전략을 통해, 연산 비용을 획기적으로 줄이면서도(메모리 최대 74% 절감) 장거리 의존성 포착 성능을 SOTA 수준으로 끌어올렸다. 이 연구는 대규모 그래프 데이터셋의 효율적인 학습을 위한 새로운 방향성을 제시한다.
