# Graph Mamba: Towards Learning on Graphs with State Space Models

Ali Behrouz, Farnoosh Hashemi (2024)

## 🧩 Problem to Solve

본 논문은 그래프 표현 학습(Graph Representation Learning)에서 기존의 주요 방법론들이 가진 한계를 극복하고자 한다.

첫째, Message-Passing Neural Networks (MPNNs)는 국소적 메시지 전달 메커니즘에 의존하므로, 그래프의 깊이가 깊어질 때 정보가 소실되는 over-squashing 문제와 노드 간 특징이 지나치게 유사해지는 over-smoothing 문제가 발생하며, 결과적으로 long-range dependencies를 캡처하는 능력이 부족하다.

둘째, Graph Transformers (GTs)는 전역 attention 메커니즘을 통해 long-range interaction을 직접 모델링함으로써 위 문제들을 해결하려 하지만, 입력 크기 $n$에 대해 $O(n^2)$의 시간 및 메모리 복잡도를 가지므로 대규모 그래프에 적용하기 어렵다. 또한, GTs는 그래프 구조에 대한 inductive bias가 부족하여 복잡하고 계산 비용이 높은 Positional Encoding (PE) 및 Structural Encoding (SE)에 크게 의존하는 경향이 있다.

따라서 본 논문의 목표는 Mamba와 같은 Selective State Space Models (SSMs)를 그래프 데이터에 적응시켜, 계산 복잡도를 선형 수준으로 유지하면서도 MPNN의 국소적 한계와 GT의 확장성 문제를 동시에 해결하는 Graph Mamba Networks (GMNs) 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 순차적 데이터 처리에 최적화된 Selective SSM을 그래프 구조에 맞게 변형하여, 선형 복잡도로 전역적 문맥을 파악하는 것이다. 주요 기여 사항은 다음과 같다.

1. **GMNs 설계 레시피 제안**: SSM을 그래프에 적용할 때 발생하는 도전 과제들을 분석하고, 이를 해결하기 위한 5단계 과정(Tokenization, PE/SE, Local Encoding, Token Ordering, Bidirectional Selective SSM Encoder)을 정의하였다.
2. **유연한 Tokenization 메커니즘**: Random Walk를 이용해 노드 수준과 서브그래프 수준의 토큰화를 단일 파라미터 $m$으로 조절할 수 있는 방식을 제안하여, 데이터 특성에 따라 inductive bias와 long-range dependency 캡처 능력 사이의 균형을 맞출 수 있게 하였다.
3. **Bidirectional SSM Encoder 설계**: SSM의 순차적/인과적(causal) 특성으로 인해 발생하는 정보 손실을 막기 위해, 입력을 양방향으로 스캔하는 구조를 도입하여 순열(permutation)에 강건한 특성을 부여하였다.
4. **이론적 정당성 확보**: GMNs가 모든 그래프 함수에 대한 universal approximator임을 증명하였으며, 적절한 PE를 사용할 경우 WL-test보다 강력한 표현력을 가짐을 이론적으로 보였다.

## 📎 Related Works

### Message-Passing Neural Networks (MPNNs)

GCN, GIN, GAT 등이 대표적이며 국소 이웃 정보를 반복적으로 집계한다. 하지만 표현력이 1-WL isomorphism test로 제한되며, over-smoothing 및 over-squashing 문제로 인해 원거리 노드 간의 관계를 학습하는 데 한계가 있다.

### Graph Transformers (GTs)

전역 attention을 통해 모든 노드 쌍의 상호작용을 모델링하여 MPNN의 한계를 극복하려 한다. 그러나 $O(n^2)$의 복잡도로 인해 대규모 그래프 적용이 불가능하며, 이를 해결하기 위해 Sparse Attention이나 Subgraph Tokenization 기법이 제안되었으나, 여전히 높은 계산 비용이나 MPNN 기반 인코딩의 한계(over-smoothing 등)를 그대로 안고 있다.

### State Space Models (SSMs)

S4, Mamba 등이 대표적이며, 선형 시간 복잡도로 긴 시퀀스를 처리할 수 있다. 특히 Mamba는 입력 데이터에 따라 상태 전이를 조절하는 selection mechanism을 도입하여 Transformer 수준의 성능을 달성하였다. 다만, SSM은 기본적으로 시퀀스 데이터를 위해 설계되었으므로, 비인과적(non-causal)이고 순서가 없는 그래프 구조에 직접 적용하는 데에는 어려움이 있다.

## 🛠️ Methodology

### 전체 파이프라인

GMNs는 그래프 $G=(V, E)$를 토큰 시퀀스로 변환한 뒤, 이를 양방향 SSM 인코더에 통과시켜 최종 노드 표현을 학습하는 구조이다. 전체 과정은 다음과 같은 단계로 구성된다.

### 1. Tokenization 및 Encoding

노드 $v$에 대해, 길이 $\hat{m} \in \{0, \dots, m\}$인 random walk를 $M$번 샘플링하고, 이들이 유도하는 서브그래프를 토큰으로 정의한다.

- **Neighborhood Sampling**: 각 노드 $v$에 대해 $\hat{m}$ 단계의 이웃 정보를 포함하는 서브그래프 $G[T_{\hat{m}}(v)]$를 생성한다. 이때 파라미터 $s$를 통해 샘플링 횟수를 조절하여 시퀀스 길이를 늘릴 수 있으며, 이는 SSM의 성능을 높이는 효과를 준다.
- **Local Encoding**: 생성된 서브그래프 토큰을 벡터화하기 위해 MPNN(예: Gated-GCN)이나 RWF(Random Walk Features) 인코더 $\phi(\cdot)$를 사용한다.
- **Positional/Structural Encoding (Optional)**: 필요 시 Laplacian eigenvectors나 Random-walk structural encodings를 초기 노드 특징에 결합한다.

### 2. Token Ordering

SSM은 순차적 인코더이므로 토큰의 순서가 중요하다.

- **서브그래프 토큰 ($m \ge 1$)**: 계층적 구조를 반영하기 위해 역순(Reverse order)으로 배치한다. 즉, $\hat{m}=m$ (가장 넓은 범위)부터 $\hat{m}=0$ (노드 자신) 순으로 배치하여, 국소적 정보가 전역적 문맥을 반영할 수 있도록 한다.
- **노드 토큰 ($m=0$)**: 노드 자체를 토큰으로 사용할 경우, 중요도에 따라 정렬한다. 본 논문에서는 계산 효율성을 위해 노드의 degree를 기준으로 정렬하였다.

### 3. Bidirectional Selective SSM Encoder

그래프의 비순차적 특성을 해결하기 위해, 데이터를 정방향과 역방향으로 두 번 스캔한다.

**정방향 스캔 (Forward Pass):**
입력 시퀀스 $\Phi$에 대해 다음과 같은 연산을 수행한다.
$$\Phi_{input} = \sigma(\text{Conv}(W_{input} \text{LayerNorm}(\Phi)))$$
$$B = W_B \Phi_{input}, \quad C = W_C \Phi_{input}, \quad \Delta = \text{Softplus}(W_\Delta \Phi_{input})$$
이후 이산화된 SSM을 통해 $y_{forward}$를 산출한다.

**역방향 스캔 (Backward Pass):**
시퀀스의 순서를 뒤집은 $\Phi_{inverse}$를 입력으로 하여 동일한 구조의 SSM을 통해 $y_{backward}$를 산출한다.

**최종 출력:**
$$y_{output} = W_{out}(y_{forward} + y_{backward})$$
이렇게 얻어진 노드 표현들은 다시 한번 Bidirectional Mamba 레이어에 입력되어 그래프 전체의 long-range dependency를 학습하게 된다.

### 4. 복잡도 분석

GMNs의 시간 복잡도는 $O(M \cdot s \cdot (m+1) \cdot (|V|+|E|))$이며, 이는 그래프 크기에 대해 선형적이다. 이는 $O(n^2)$의 복잡도를 가지는 GTs에 비해 비약적인 효율성 향상을 의미한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Long-Range Graph Benchmark (LRGB), GNN Benchmark (MNIST, CIFAR10 등), Heterophilic Benchmark, OGBN-Arxiv 등 다양한 특성의 데이터셋을 사용하였다.
- **비교 대상**: MPNNs (GCN, GIN), GTs (GPS, Exphormer, NAGphormer), SSM 기반 모델 (S4G, GRED) 및 Baseline (GPS + Mamba)과 비교하였다.

### 주요 결과

1. **Long-Range 성능**: LRGB 데이터셋(COCO-SP, PascalVOC-SP 등)에서 GMNs는 모든 baseline을 압도하는 성능을 보였다. 이는 bidirectional scan과 selection mechanism이 불필요한 정보를 필터링하고 원거리 의존성을 효과적으로 캡처했기 때문이다.
2. **Heterophilic 데이터**: 서로 다른 특성을 가진 노드들이 연결된 heterophilic 그래프에서도 우수한 성능을 기록하며, over-smoothing 문제에 강건함을 입증하였다.
3. **효율성 (Scalability)**: OGBN-Arxiv 및 MalNet-Tiny 실험에서 GPS 대비 메모리 사용량이 획기적으로 적으며, GPU 메모리 사용량이 노드 수에 따라 선형적으로 증가함을 확인하였다 (Figure 3).
4. **Ablation Study**: Bidirectional Mamba 구조가 성능 향상에 가장 크게 기여했으며, PE/SE나 MPNN 없이도 경쟁력 있는 성능을 낼 수 있음을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 통찰

본 연구는 Transformer의 전역 Attention이 주는 성능 이점을 유지하면서도, SSM의 선형 복잡도를 그래프 도메인에 성공적으로 이식하였다. 특히, 단순히 Mamba 블록을 교체하는 것이 아니라 **'토큰화 $\rightarrow$ 역순 정렬 $\rightarrow$ 양방향 스캔'**으로 이어지는 체계적인 파이프라인을 구축함으로써, SSM의 인과적 제약을 극복하고 그래프의 구조적 정보를 보존하였다.

### 비판적 해석 및 한계

- **하이퍼파라미터 민감도**: $M$(walk 수), $m$(walk 길이), $s$(샘플링 횟수) 등의 파라미터가 성능에 영향을 미치며, 데이터셋마다 최적값이 다르다. 이는 사용자가 최적의 설정을 찾기 위해 상당한 탐색 비용을 들여야 함을 의미한다.
- **토큰화의 오버헤드**: 이론적 복잡도는 선형이지만, 전처리 단계에서 수행되는 random walk 샘플링 및 서브그래프 인코딩 과정이 실제 실행 시간에서 어느 정도의 비중을 차지하는지에 대한 더 상세한 분석이 필요하다.

### 결론적 논의

논문은 "Transformer나 복잡한 PE/SE가 좋은 성능을 내기 위해 충분(sufficient)할 수는 있지만, 필수(necessary)적인 것은 아니다"라는 점을 시사한다. 이는 적절한 토큰화 전략과 효율적인 SSM 구조만으로도 고성능 그래프 학습이 가능함을 보여준 중요한 결과이다.

## 📌 TL;DR

본 논문은 Selective SSM(Mamba)을 그래프 학습에 적용한 **Graph Mamba Networks (GMNs)**를 제안한다. 서브그래프 기반의 유연한 토큰화와 양방향 스캔 메커니즘을 통해, **$O(n^2)$의 복잡도를 가진 Graph Transformer의 성능을 유지하면서도 선형 복잡도 $O(n)$으로 확장성 문제를 해결**하였다. 특히 long-range dependency 캡처와 대규모 그래프 처리에 탁월한 효율성을 보이며, 향후 초거대 그래프 데이터 분석을 위한 새로운 아키텍처적 방향성을 제시한다.
