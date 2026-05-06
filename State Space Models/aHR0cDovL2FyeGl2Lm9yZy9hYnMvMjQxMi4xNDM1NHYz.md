# State Space Models are Strong Text Rerankers

Jinghua Yan, Zhichao Xu, Ashim Gupta, Vivek Srikumar (2024/2025)

## 🧩 Problem to Solve

본 연구는 자연어 처리(NLP)와 정보 검색(IR) 분야에서 표준으로 자리 잡은 Transformer 아키텍처의 효율성 문제를 해결하고자 한다. Transformer는 긴 시퀀스에 대해 추론 시 시간 복잡도가 $O(L)$, 공간 복잡도가 $O(LD)$(여기서 $L$은 시퀀스 길이, $D$는 hidden state의 차원)로 증가하며, 특히 매우 긴 문맥(long context)을 처리할 때 추론 효율성이 급격히 떨어진다는 한계가 있다.

이러한 문제를 해결하기 위해 최근 State Space Models(SSMs), 특히 Mamba와 같은 모델들이 제안되었다. SSM은 추론 시 시간 복잡도를 $O(1)$로 줄일 수 있는 잠재력을 가지고 있다. 그러나 텍스트 재순위화(Text Reranking) 작업은 쿼리와 문서 간의 세밀한 상호작용(fine-grained interaction)과 긴 문맥에 대한 깊은 이해가 필수적이다. 본 논문은 과연 SSM 기반 아키텍처가 이러한 까다로운 재순위화 작업에서 Transformer 수준의 성능을 낼 수 있는지, 그리고 실제 효율성 측면에서 이점이 있는지 분석하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 다음과 같다.

1. **SSM 기반 재순위화 모델의 성능 검증**: Mamba-1과 Mamba-2 아키텍처를 다양한 규모와 사전 학습 목표에 따라 벤치마킹하여, SSM이 유사한 파라미터 규모의 Transformer 모델과 대등한 텍스트 재순위화 성능을 낼 수 있음을 입증하였다.
2. **Mamba-1 vs Mamba-2 비교 분석**: Mamba-2가 Mamba-1보다 성능과 효율성 모든 면에서 우수함을 확인하였다.
3. **실제 효율성 분석**: 이론적인 복잡도와 달리, 실제 구현 단계에서는 Flash Attention이 적용된 Transformer가 Mamba보다 학습 및 추론 속도 면에서 더 효율적임을 밝혀냈으며, 그 이유를 연산자 레벨의 프로파일링을 통해 분석하였다.

## 📎 Related Works

**1. Transformer 기반 텍스트 랭킹**
기존의 재순위화 연구는 BERT, RoBERTa, T5와 같은 사전 학습된 Transformer 모델을 미세 조정(Fine-tuning)하는 방식이 주류였다. 특히 쿼리와 문서를 결합하여 입력하고 단일 스칼라 점수를 예측하는 Cross-Encoder 구조가 표준으로 사용되어 왔다. 하지만 이 방식은 시퀀스 길이에 따라 계산 비용이 이차적으로 증가하는 문제가 있다.

**2. Transformer 대안 아키텍처**
S4(Structured State Space Sequence Models)와 RWKV 같은 모델들이 Transformer의 효율성 문제를 해결하기 위해 제안되었다. 특히 Mamba는 '선택적 상태 공간 모델(Selective SSM)'을 통해 입력에 따라 파라미터가 변하도록 설계하여, 기존 SSM의 정보 압축 한계를 극복하고 Transformer에 근접한 표현력을 확보하였다.

**3. 차별점**
기존 연구들이 주로 언어 모델링(Language Modeling)이나 일반적인 시퀀스 작업에 집중한 반면, 본 연구는 정보 검색의 핵심 단계인 '재순위화(Reranking)'라는 특수한 작업에 집중하여 SSM의 효용성을 심층 분석하였다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. State Space Model (SSM) 기본 구조

SSM은 1차원 입력 시퀀스 $x(t)$를 잠재 상태(latent state) $h(t)$를 통해 출력 $y(t)$로 매핑한다. 연속 시간 시스템에서의 정의는 다음과 같다.
$$h'(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t)$$
이를 이산화(discretization)하면 다음과 같은 재귀 형태로 표현된다.
$$h_t = \bar{A}h_{t-1} + \bar{B}x_t$$
$$y_t = \bar{C}h_t$$
여기서 $\bar{A}$와 $\bar{B}$는 discretization rule에 의해 결정된다. 이 구조는 RNN처럼 재귀적으로 계산할 수 있어 추론 효율성이 높으며, 동시에 CNN처럼 합성곱(convolution) 형태로 변환하여 병렬 학습이 가능하다.

### 2. Mamba-1 및 Mamba-2 아키텍처

- **Mamba-1**: 기존 SSM의 고정된 파라미터를 입력값에 따라 변하는 '선택적(Selective)' 파라미터 $(\Delta, B, C)$로 변경하였다. 이를 효율적으로 계산하기 위해 Hardware-aware한 Selective Scan 알고리즘을 사용한다.
- **Mamba-2**: 행렬 $A$를 스칼라 값과 단위 행렬의 곱으로 제한하여 단순화하였다. 또한 Transformer의 head 구조와 유사하게 SSM head dimension $P$를 도입하여 $D = P \times \#heads$ 형태로 구성함으로써 행렬 곱셈(Matrix Multiplication)을 더 효과적으로 활용하도록 개선하였다.

### 3. 재순위화 학습 절차

- **입력 구성**: 쿼리 $q$와 문서 $d$를 연결하여 입력으로 사용한다.
  - Autoregressive 모델(Mamba, OPT): `document:{d}; query:{q}; [EOS]` 형태의 템플릿을 사용하며, $[EOS]$ 토큰의 마지막 레이어 표현을 사용한다.
  - Encoder 모델(BERT 등): `[CLS]; query:{q}; document:{d}` 형태를 사용하며, $[CLS]$ 토큰의 표현을 사용한다.
- **손실 함수**: 하드 네거티브(hard negatives)를 포함한 Softmax Loss를 사용하여 학습한다.
$$\text{Loss} = -\frac{1}{|S|} \sum_{(q_i, d^+_i) \in S} \log \frac{\exp f_\theta(q_i, d^+_i)}{\exp f_\theta(q_i, d^+_i) + \sum_{j \in D^-_i} \exp f_\theta(q_i, d^-_i)}$$
- **출력**: 언어 모델 상단에 선형 레이어(Linear layer)를 추가하여 최종 관련성 점수(scalar score) $f_\theta(q, d)$를 산출한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: MS MARCO (Passage/Document), TREC DL19/DL20, BEIR (13개 데이터셋)
- **평가 지표**: MRR@10, NDCG@10
- **비교 모델**: BERT, RoBERTa, ELECTRA (Encoder), BART (Enc-Dec), OPT, Llama-3.2-1B (Decoder), Mamba-1, Mamba-2

### 2. 성능 결과

- **Passage Reranking**: Mamba 모델들은 유사한 크기의 Transformer 모델들과 대등한 성능을 보였다. 특히 Mamba-2-370M은 BERT-large와 유사한 수준의 성능을 기록하였다. 다만, 15T 토큰으로 사전 학습된 Llama-3.2-1B가 가장 높은 성능을 보였다.
- **Document Reranking**: 긴 문맥 처리 능력을 테스트한 결과, Mamba-2-780M 모델이 1,536 토큰 길이(LongP) 설정에서 sub-1B 파라미터 모델 중 가장 우수한 성능을 보였다.
- **일반화 성능(Out-of-domain)**: BEIR 벤치마크 결과, Mamba-2-1.3B가 OPT-1.3B보다 더 높은 평균 NDCG@10을 기록하며 강건함을 보였다.

### 3. 효율성 결과

- **학습 처리량(Training Throughput)**: Mamba-2가 Mamba-1보다 훨씬 빠르지만, Flash Attention이 적용된 Transformer(OPT-FlashAttn)보다는 현저히 느렸다.
- **추론 속도(Inference Speed)**: 이론적인 $O(1)$ 복잡도에도 불구하고, 실제 재순위화 추론 속도는 Transformer와 비슷하거나 오히려 약간 느린 결과를 보였다. (표 5 참고)

## 🧠 Insights & Discussion

**1. 왜 Mamba의 추론 속도가 기대보다 느린가?**
일반적으로 Mamba의 효율성은 '생성(Generation)' 작업처럼 여러 번의 forward pass가 필요한 경우에 극대화된다. 하지만 재순위화 작업은 단 한 번의 forward pass로 점수를 계산하는 구조이므로, SSM의 재귀적 이점이 발휘될 기회가 없다.

**2. 연산자 레벨의 병목 현상 (Profiling Analysis)**

- **Mamba-1**: `aten::is_nonzero`, `aten::item`과 같은 스칼라 값 추출 연산이 실행 시간의 상당 부분을 차지한다. 이는 텐서 연산의 효율성을 저해하는 병목 지점이 된다.
- **Mamba-2**: 구조적 개선을 통해 스칼라 추출 연산을 제거하고 행렬 곱셈(`MambaSplitConv1D`) 비중을 높였다. 이로 인해 Mamba-1보다 효율성이 크게 향상되었으나, 여전히 I/O 최적화가 극대화된 Flash Attention 기반 Transformer에는 미치지 못한다.

**3. 결론적 해석**
SSM은 텍스트 재순위화 작업에서 Transformer를 대체할 수 있을 만큼 강력한 성능을 가졌음이 입증되었다. 특히 Mamba-2는 성능과 메모리 효율성 면에서 큰 진전을 이루었다. 하지만 실제 하드웨어 가속 및 I/O 최적화 측면에서는 여전히 Transformer가 우위에 있으며, 이를 극복하기 위한 추가적인 아키텍처 최적화가 필요하다.

## 📌 TL;DR

본 논문은 **Mamba-1 및 Mamba-2 모델이 텍스트 재순위화 작업에서 Transformer와 대등한 성능을 낼 수 있음**을 실험적으로 증명하였다. 특히 **Mamba-2는 Mamba-1보다 성능과 메모리 효율이 더 뛰어나다**. 하지만 **실제 학습 및 추론 속도는 Flash Attention을 사용하는 Transformer가 더 빠르며**, 이는 재순위화 작업이 단일 forward pass로 이루어지기 때문에 SSM의 이론적 이점이 사라지고 구현상의 오버헤드가 부각되기 때문이다. 이 연구는 SSM이 IR 작업의 유망한 대안이 될 수 있음을 시사하며, 향후 하드웨어 최적화 및 하이브리드 모델 연구의 필요성을 제시한다.
