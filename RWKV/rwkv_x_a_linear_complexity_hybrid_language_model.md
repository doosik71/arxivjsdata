# RWKV-X: A Linear Complexity Hybrid Language Model

Haowen Hou, Zhiyi Huang, Kaifeng Tan, Rongchang Lu, Fei Richard Yu (2025)

## 🧩 Problem to Solve

현대 거대 언어 모델(LLM)의 근간이 되는 Transformer 아키텍처는 입력 시퀀스 길이에 대해 제곱 시간 복잡도($O(N^2)$)를 가지는 Self-Attention 메커니즘으로 인해, 매우 긴 컨텍스트를 처리할 때 막대한 계산 비용과 메모리 병목 현상이 발생한다. 이를 해결하기 위해 Mamba와 같은 State Space Models(SSMs)나 RWKV와 같은 Linear RNN 계열의 모델들이 제안되었으며, 이들은 선형 시간 복잡도($O(N)$)를 달성하여 효율성을 크게 높였다.

그러나 이러한 선형 복잡도 모델들은 효율성에도 불구하고 긴 컨텍스트에 대한 이해력(Long-context understanding)이 떨어진다는 치명적인 한계가 있다. 예를 들어, RWKV-7 모델의 경우 일정 길이(약 28K 토큰)까지는 높은 정확도를 보이지만, 그 이상의 길이에서는 Passkey retrieval(특정 키에 대응하는 값을 찾는 작업) 성능이 급격히 저하되는 현상이 관찰된다. 기존의 하이브리드 모델(예: Jamba, Zamba)들은 Full Attention 레이어를 일부 섞어 이 문제를 완화하려 했으나, 결과적으로 여전히 제곱 복잡도를 유지하게 되어 초장거리 시퀀스 처리에서의 확장성 문제가 여전히 남아 있다. 따라서 본 논문의 목표는 훈련 시 선형 복잡도를 유지하면서도, 추론 시 일정한(constant) 메모리와 속도를 보장하며, 동시에 매우 긴 컨텍스트를 효과적으로 처리할 수 있는 새로운 하이브리드 아키텍처를 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 RWKV-7의 효율적인 단거리 모델링 능력과 새롭게 제안된 Sparse Attention의 장거리 컨텍스트 포착 능력을 결합한 **RWKV-X** 아키텍처를 제안한 것이다.

가장 중심적인 아이디어는 Full Attention의 $O(N^2)$ 복잡도를 피하기 위해, 모든 토큰이 아닌 가장 관련성이 높은 일부 데이터 블록만을 선택적으로 참조하는 **Top-kChunk Sparse Attention**을 도입한 것이다. 이를 통해 훈련 단계에서는 $O(N)$의 선형 복잡도를 달성하고, 추론 단계에서는 KV 캐시 관리 메커니즘을 통해 $O(1)$의 상수 시간 복잡도를 구현하였다. 결과적으로 RWKV-X는 짧은 컨텍스트에서의 성능을 유지하면서도, 최대 100만 토큰에 이르는 초장거리 시퀀스를 안정적인 속도와 메모리 사용량으로 디코딩할 수 있는 확장 가능한 백본을 제공한다.

## 📎 Related Works

### Linear Complexity Language Models

Transformer의 연산 효율성 문제를 해결하기 위해 Linear Attention, State Space Models(SSMs), Linear RNNs 등이 연구되었다. 특히 RWKV 시리즈는 RNN의 재귀적 특성과 Transformer의 병렬 학습 능력을 결합하여 선형 복잡도를 달성했다. RWKV-7은 동적 상태 진화(Dynamic state evolution)를 통해 인컨텍스트 학습 능력을 향상시켰으나, 여전히 매우 긴 시퀀스에서의 정보 회수 능력에는 한계가 있었다.

### Hybrid Language Models

최근 Jamba나 Zamba와 같은 하이브리드 모델들이 SSM과 Transformer를 결합하여 성능과 효율성의 균형을 맞추려 시도했다. 하지만 이러한 모델들은 여전히 Full Attention 레이어에 의존하고 있어, 시퀀스 길이가 증가함에 따라 메모리 사용량이 제곱으로 증가하는 근본적인 한계를 극복하지 못했다.

### Sparse Attention

Native Sparse Attention이나 MoBA(Mixture of Block Attention)와 같은 연구들은 입력 컨텍스트를 블록 단위로 나누어 중요한 부분만 참조함으로써 연산량을 줄이려 했다. 하지만 MoBA의 경우, 자동 회귀 디코딩(Autoregressive decoding) 과정에서 KV 캐시가 시퀀스 길이에 따라 선형적으로 증가하여, 추론 시 메모리 사용량을 일정하게 유지할 수 없다는 단점이 있었다.

## 🛠️ Methodology

### 전체 아키텍처

RWKV-X는 RWKV-7 블록과 Sparse Attention 블록이 주기적으로 교차 배치된 하이브리드 구조이다. RWKV-7 블록이 짧은 범위의 의존성을 효율적으로 처리한다면, Sparse Attention 블록은 긴 범위의 문맥 정보를 캡처하는 역할을 수행한다.

### Top-kChunk Sparse Attention

이 메커니즘은 전체 시퀀스를 크기 $B$의 동일한 청크(Chunk)들로 나눈 뒤, 쿼리 토큰 $q$와 가장 관련이 깊은 $k$개의 청크만을 선택하여 어텐션을 수행한다.

1. **청크 관련성 계산**: 각 청크 $i$에 대해, 해당 청크 내 모든 키 벡터들의 평균값과 쿼리 $q$의 내적을 통해 관련성 점수 $s_i$를 계산한다.
   $$s_i = q \cdot \left( \frac{1}{B} \sum_{j=1}^{B} k_j^{(i)} \right), \quad i=1, \dots, n$$
2. **Top-k 선택**: 계산된 점수 중 상위 $k$개의 청크 인덱스 집합 $I$를 선택한다.
   $$I = \text{TopK}(\{s_i\}_{i=1}^n, k)$$
3. **제한적 어텐션 수행**: 선택된 청크 $I$에 속하는 키-값 쌍에 대해서만 표준 어텐션을 계산한다.
   $$\text{Attn}(q, K_I, V_I) = \text{softmax} \left( \frac{q K_I^\top}{\sqrt{d_k}} \right) V_I$$

### KV 캐시 관리 (KV Cache Management)

추론 시 메모리 사용량을 상수로 유지하기 위해, SnapKV에서 영감을 얻은 압축 전략을 사용한다. 과거의 캐시를 '이전 캐시 상태'($K_{past}, V_{past}$)와 '최근 관찰 윈도우'($K_{obs}, V_{obs}$)로 분리한다. 관찰 윈도우의 쿼리와 이전 캐시 간의 어텐션 점수를 합산하여 중요도 벡터 $C$를 구하고, 상위 $m$개의 중요한 항목만을 유지하고 나머지는 제거한다. 이를 통해 메모리 예산 $m$ 내에서 일정한 공간 복잡도를 유지한다.

### 학습 절차 및 손실 함수

RWKV-X는 처음부터 학습시키지 않고 LLaMA Pro의 블록 확장 방식을 채택하여 두 단계로 학습한다.

1. **정렬 사전학습(Alignment Pretraining)**: MiniPile 데이터셋을 사용하여 짧은 텍스트(1024 토큰)로 학습한다. 이때 기존 RWKV-7 블록은 동결하고 새로 추가된 Sparse Attention 블록만 업데이트하여 파라미터를 정렬시킨다.
2. **장거리 컨텍스트 지속 사전학습(Long-context Continual Pretraining)**: ProLong-64K 데이터셋을 사용하여 64K 토큰 길이로 학습하며, 모든 파라미터를 해제하여 공동 최적화한다.

특히, 장거리 의존성을 가진 중요한 토큰에 더 많은 가중치를 부여하는 **Long-context Cross-Entropy (LongCE)** 손실 함수를 사용하여 모델이 긴 문맥 속의 핵심 정보를 더 잘 포착하도록 유도한다.

## 📊 Results

### 장거리 컨텍스트 평가 (S-NIAH)

RULER 벤치마크의 Single Needle-In-A-Haystack(S-NIAH) 테스트 결과, RWKV-X-3.6B 모델은 64K 토큰 길이의 Passkey retrieval 작업에서 거의 완벽한 정확도를 달성하였다. 이는 동일 규모의 RWKV-7이나 Mamba2 등의 모델들이 특정 길이 이후 성능이 급격히 하락하는 것과 대조적이며, 장거리 정보 회수 능력이 획기적으로 개선되었음을 보여준다.

### 단거리 컨텍스트 평가

단거리 언어 이해 벤치마크(MMLU, ARC 등)에서 RWKV-X(3.6B)는 평균 점수 71.9를 기록하여, RWKV-7(2.9B, 72.8) 및 Qwen2.5-3B(71.4)와 대등한 수준의 성능을 보였다. 이는 장거리 처리 능력을 추가했음에도 불구하고 기존의 일반적인 언어 모델링 성능을 희생하지 않았음을 의미한다.

### 효율성 분석

- **Prefill 지연 시간**: 128K 토큰 길이에서 RWKV-X는 Flash-Attention v3를 사용하는 Transformer보다 약 1.37배 빠른 속도를 보였으며, 시퀀스 길이가 길어질수록 이 격차는 더 벌어지는 선형 스케일링 특성을 보였다.
- **디코딩 지연 시간**: RWKV-X-3.6B는 컨텍스트 길이가 100만 토큰까지 증가하더라도 디코딩 시간이 거의 일정하게 유지되는 놀라운 안정성을 보여주었다.

## 🧠 Insights & Discussion

### 주요 분석 및 통찰

1. **LongCE 손실 함수의 중요성**: Ablation study 결과, LongCE를 사용하지 않았을 때 8K 이상의 길이에서 성능이 급격히 떨어지는 현상이 확인되었다. 이는 단순한 데이터 학습보다 중요한 토큰에 가중치를 두는 학습 전략이 장거리 일반화 능력에 핵심적임을 시사한다.
2. **최적의 하이브리드 비율**: 전체 레이어 중 약 25%를 Sparse Attention 레이어로 구성했을 때 검증 손실(Validation loss)이 가장 낮게 나타났다. 이는 순수 RWKV-7나 순수 Transformer보다 적절한 혼합 구조가 더 효율적임을 입증한다.
3. **위치 인코딩의 불필요성**: ROPE나 절대적 위치 임베딩을 적용한 것보다 위치 인코딩을 전혀 사용하지 않았을 때 성능이 미세하게 더 좋거나 비슷했다. 이는 RWKV의 재귀적 구조 자체가 이미 충분한 암시적 위치 정보를 제공하고 있음을 의미한다.

### 한계점

Top-kChunk 선택 방식은 휴리스틱한 접근이므로 일부 의미적으로 중요한 의존성을 놓칠 가능성이 있다. 또한, 현재 구현상으로는 Sparse Attention의 디코딩 속도가 순수 RWKV보다 느린 부분이 있어, 향후 커널 최적화 등의 엔지니어링 작업이 필요하다.

## 📌 TL;DR

RWKV-X는 RWKV-7의 효율성과 Top-kChunk Sparse Attention의 장거리 문맥 포착 능력을 결합한 하이브리드 언어 모델이다. 훈련 시 선형 복잡도 $O(N)$를, 추론 시 상수 시간 복잡도 $O(1)$를 달성하여 100만 토큰 이상의 초장거리 시퀀스도 안정적으로 처리할 수 있다. 특히 64K Passkey retrieval에서 거의 완벽한 성능을 보였으며, 이는 효율적인 선형 모델이 가진 고질적인 장거리 기억 상실 문제를 해결할 수 있는 유망한 방향성을 제시한다. 향후 초장거리 문맥을 다루는 일반 목적의 LLM 백본으로 활용될 가능성이 매우 높다.
