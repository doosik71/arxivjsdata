# Kronecker Decomposition for GPT Compression

Ali Edalati, Marzieh Tahaei, Ahmad Rashid, Vahid Partovi Nia, James J. Clark, Mehdi Rezagholizadeh (2021)

## 🧩 Problem to Solve

본 논문은 GPT-2와 같은 Auto-regressive Transformer 기반의 사전 학습 언어 모델(Pre-trained Language Models, PLMs)이 가진 과도한 파라미터 수로 인해 발생하는 배포의 어려움을 해결하고자 한다. GPT 모델은 방대한 데이터와 수억 개에서 수십억 개의 파라미터를 통해 뛰어난 성능을 보이지만, 이러한 특성은 메모리와 계산 능력이 제한된 디바이스에서의 추론 및 학습을 매우 어렵게 만든다.

기존의 모델 압축 연구는 주로 BERT와 같은 Encoder-based 모델에 집중되어 왔으며, GPT 계열의 압축 연구는 상대적으로 부족한 실정이다. 특히 기존의 대표적인 압축 모델인 DistilGPT2는 방대한 데이터셋(OpenWebText)으로 장시간 사전 학습을 수행해야 한다는 효율성 문제가 있으며, 자연어 이해(NLU) 작업에서 BERT 계열에 비해 성능이 낮다는 한계가 있다. 따라서 본 연구의 목표는 Kronecker Decomposition을 활용하여 파라미터 수를 줄이면서도, 효율적인 학습 과정과 높은 NLU 성능을 갖춘 압축된 GPT 모델인 **KnGPT2**를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 거대한 가중치 행렬을 두 개의 훨씬 작은 행렬의 Kronecker Product로 분해하여 표현함으로써 파라미터 수와 계산 복잡도를 획기적으로 줄이는 것이다.

주요 기여 사항은 다음과 같다:

1. **GPT 모델 압축에 Kronecker Decomposition 최초 적용**: 기존에 BERT 압축 등에 사용되었던 기법을 GPT-2 모델의 Embedding 레이어와 Transformer 레이어(MHA, FFN)에 적용하여 모델 크기를 줄였다.
2. **학습 효율성 증대**: DistilGPT2와 같은 기존 모델이 방대한 데이터로 여러 에포크를 학습해야 했던 것과 달리, 본 연구에서는 분해 후 손실된 정보를 복구하기 위해 전체 데이터의 10%만을 사용하여 단 1 에포크의 가벼운 사전 학습(Light Pre-training)을 수행한다.
3. **Intermediate Layer Knowledge Distillation (ILKD) 도입**: 단순한 출력값의 모방이 아니라 Embedding, Attention distribution, Hidden state 등 중간 레이어의 지식을 전수함으로써 압축으로 인한 성능 저하를 최소화하였다.

## 📎 Related Works

논문에서는 모델 압축을 위한 기존의 네 가지 주요 방향성인 저비트 양자화(Low-bit Quantization), 가지치기(Pruning), 지식 증류(Knowledge Distillation), 행렬 분해(Matrix Decomposition)를 언급한다.

Kronecker Decomposition과 관련하여, 과거에는 주로 Fully-connected network나 소규모 CNN, IoT용 언어 모델 등에 적용되었으며, 최근에는 BERT 모델을 압축한 KroneckerBERT 연구가 수행되었다. 하지만 GPT-2와 같은 대규모 Auto-regressive 모델에 이를 적용한 사례는 본 연구가 처음임을 명시하고 있다.

비교 대상으로 설정된 DistilGPT2는 Knowledge Distillation을 통해 GPT-2 Small(124M)을 82M 파라미터 규모로 줄인 모델이다. 그러나 DistilGPT2는 학습에 소요되는 시간과 데이터 양이 매우 많다는 한계가 있다.

## 🛠️ Methodology

### 1. Kronecker Product 및 분해 원리

Kronecker Product($\otimes$)는 두 행렬 $A \in \mathbb{R}^{m_1 \times n_1}$와 $B \in \mathbb{R}^{m_2 \times n_2}$를 입력받아 $m=m_1m_2, n=n_1n_2$ 크기의 블록 행렬을 생성하는 연산이다.
$$A \otimes B = \begin{bmatrix} a_{11}B & \cdots & a_{1n_1}B \\ \vdots & \ddots & \vdots \\ a_{m_11}B & \cdots & a_{m_1n_1}B \end{bmatrix}$$

이를 통해 가중치 행렬 $W \in \mathbb{R}^{m \times n}$를 $W \approx A \otimes B$로 근사하면, 저장해야 할 파라미터 수가 $mn$개에서 $m_1n_1 + m_2n_2$개로 크게 감소한다.

### 2. GPT-2 모델의 압축 적용

본 논문은 GPT-2의 주요 선형 매핑 레이어들에 Kronecker 분해를 적용한다.

- **Embedding Layer**: $W^E \in \mathbb{R}^{v \times d}$ (여기서 $v$는 vocabulary size, $d$는 embedding dimension)를 $A^E \in \mathbb{R}^{v \times d/f}$와 $B^E \in \mathbb{R}^{1 \times f}$로 분해한다. 이를 통해 각 단어 임베딩 $E_i$는 $A^E_i \otimes B$로 계산되며 계산 복잡도는 $O(d)$로 유지된다.
- **Transformer Layers**: Multi-Head Attention(MHA)의 Query, Key, Value 행렬($W^Q, W^K, W^V$)과 Feed-Forward Network(FFN)의 두 가중치 행렬($W^c_{fc}, W^c_{proj}$)을 Kronecker factor로 분해한다.

**초기화 방법**: 원본 모델 $W$에 가장 가까운 Kronecker factor $\hat{A}, \hat{B}$를 찾기 위해 다음의 최적화 문제를 푼다.
$$(\hat{A}, \hat{B}) = \text{argmin}_{(A,B)} \|W - A \otimes B\|_F^2$$
이 문제는 $W$를 적절히 reshape한 후 Rank-1 SVD(Singular Value Decomposition) 근사를 통해 해결한다.

### 3. Intermediate Layer Knowledge Distillation (ILKD)

분해 직후의 모델은 성능이 크게 저하되므로, Teacher 모델(GPT-2)의 지식을 Student 모델(KnGPT2)에 전달하는 ILKD를 수행한다.

- **Embedding Loss**: Teacher와 Student 임베딩 간의 MSE를 사용한다.
  $$L_{\text{Embedding}}(x) = \text{MSE}\{E^S(x), E^T(x)\}$$
- **Attention Loss**: Attention 분포 간의 KL-Divergence를 사용한다.
  $$L_{\text{Attention}}(x) = \sum_{l} \text{KL}\{\text{Att}^S_l(x), \text{Att}^T_l(x)\}$$
- **Hidden States Loss**: FFN의 최종 출력값 간의 MSE를 사용한다.
  $$L_{\text{Hidden States}}(x) = \sum_{l} \text{MSE}\{H^S_l(x), H^T_l(x)\}$$

**최종 손실 함수**:
$$\text{Loss}(x, y) = \alpha_1 L_{\text{Embedding}}(x) + \alpha_2 L_{\text{Attention}}(x) + \alpha_3 L_{\text{Hidden States}}(x) + \alpha_4 L_{\text{Cross Entropy}}(x, y)$$

## 📊 Results

### 1. 실험 설정

- **대상 모델**: GPT-2 Small (124M parameters)
- **비교 모델**: DistilGPT2 (82M parameters)
- **압축 설정**: KnGPT2의 파라미터 수를 약 83M으로 맞추기 위해 Embedding 레이어와 Transformer 레이어 중 홀수 번째 레이어들을 factor 2로 압축하였다.
- **평가 지표**: WikiText-103 (Perplexity), GLUE benchmark (Average Score)

### 2. 주요 결과

- **Language Modeling (LM)**: WikiText-103 데이터셋에서 KnGPT2는 Perplexity $20.5$를 기록하여, 더 많은 데이터로 더 오래 학습된 DistilGPT2($23.7$)보다 우수한 성능을 보였다.
- **GLUE Benchmark**:
  - **Dev Set**: KnGPT2 + ILKD는 평균 $79.25$를 기록하여 DistilGPT2($76.8$) 및 DistilGPT2 + KD($76.73$)보다 월등히 높았으며, 원본 GPT-2 Small($79.81$)에 근접하였다.
  - **Test Set**: KnGPT2 + ILKD는 평균 $77.42$를 기록하여 DistilGPT2($74.52$) 대비 성능 향상이 뚜렷했으며, 원본 GPT-2 Small($77.56$)과 거의 동일한 수준의 성능을 달성하였다.
- **학습 효율성**: KnGPT2는 DistilGPT2가 사용한 데이터의 $1/10$만 사용하여 단 1 에포크만 학습하였으며, 학습 시간 또한 $6.5$시간으로 매우 짧았다 (DistilGPT2는 $\sim 90$시간 이상 소요).

### 3. Ablation Study

사전 학습의 효과를 분석한 결과, 단순 LM 사전 학습보다 KD를 결합한 사전 학습이 downstream task(MNLI 등)에서 더 좋은 성능을 보였다. 다만, LM에서의 Perplexity 감소가 항상 downstream task의 성능 향상으로 직접 연결되지는 않는다는 점이 관찰되었다.

## 🧠 Insights & Discussion

본 논문은 Kronecker Decomposition이 대규모 언어 모델의 파라미터를 효과적으로 줄이는 동시에, 적절한 초기화와 가벼운 지식 증류(KD) 과정을 거치면 원본 모델에 근접하는 성능을 유지할 수 있음을 입증하였다.

특히 주목할 점은 **학습 효율성**이다. 기존의 DistilGPT2가 모델 구조 자체를 작게 만든 후 방대한 데이터로 다시 학습시키는 방식을 취했다면, KnGPT2는 기존 학습된 가중치를 수학적으로 분해하여 초기값을 잡고, 부족한 부분만 살짝 보완하는 방식을 취함으로써 학습 비용을 획기적으로 낮추었다.

다만, 본 연구는 GPT-2 Small 모델에 한정하여 실험이 진행되었다는 한계가 있다. 더 큰 규모의 GPT-3나 다른 Transformer 변형 모델에서도 동일한 압축률과 성능 복구 효율이 나타날지는 추가적인 연구가 필요하다. 또한, 압축 계수(Compression Factor)를 더 높였을 때 성능 하락폭이 어느 정도일지에 대한 상세한 분석은 부족한 상태이다.

## 📌 TL;DR

본 연구는 **Kronecker Decomposition**을 GPT-2 모델에 적용하여 파라미터 수를 획기적으로 줄인 **KnGPT2**를 제안한다. 가중치 행렬을 작은 행렬들의 곱으로 분해하고, **Intermediate Layer Knowledge Distillation (ILKD)**와 매우 가벼운 사전 학습을 통해 성능을 복구하였다. 실험 결과, KnGPT2는 DistilGPT2보다 훨씬 적은 데이터와 학습 시간만으로도 더 뛰어난 언어 모델링 성능과 GLUE 벤치마크 성능(원본 모델 수준)을 달성하였다. 이는 향후 거대 언어 모델의 경량화 및 온디바이스 배포에 있어 매우 효율적인 방법론이 될 가능성이 높다.
