# Linearizing Large Language Models

Jean Mercat, Igor Vasiljevic, Sedrick Keh, Kushal Arora, Achal Dave, Adrien Gaidon, Thomas Kollar (2024)

## 🧩 Problem to Solve

본 논문은 Transformer 아키텍처의 핵심인 Softmax Attention이 가진 추론 비용 문제를 해결하고자 한다. Transformer는 학습 효율성과 스케일링 성능이 매우 뛰어나지만, 추론 시 메모리와 연산 비용이 시퀀스 길이에 따라 선형적으로 증가한다는 치명적인 단점이 있다.

반면, Recurrent Neural Networks (RNNs)나 Linear Transformers는 고정된 크기의 hidden state를 유지하므로 추론 비용이 일정하며 효율적이다. 하지만 기존의 Linear Attention 모델들은 Softmax Attention에 비해 성능 및 스케일링 능력이 떨어지며, 이를 극복하기 위해 제안된 최신 모델들(RWKV, Mamba 등)은 막대한 양의 데이터와 계산 자원을 투입해 처음부터 다시 사전 학습(Pre-training)해야 한다는 부담이 있다.

따라서 본 연구의 목표는 이미 성능이 검증된 거대 언어 모델(LLM)을 적은 비용으로 RNN으로 변환하는 효율적인 방법론을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **SUPRA (Scalable UPtraining for Recurrent Attention)**라는 기법을 제안한 것이다. 이는 처음부터 Linear 모델을 학습시키는 대신, 이미 사전 학습된 Transformer 모델을 RNN으로 변환하는 'Uptraining' 방식을 취한다.

중심 아이디어는 기존의 Softmax Attention을 Linear Kernel로 교체하고, 학습 안정성을 위해 새로운 정규화 전략(GroupNorm)과 상대적 위치 인코딩(RoPE)을 결합하는 것이다. 이를 통해 전체 사전 학습 비용의 약 5%만 사용하여도 최신 Linear 모델들과 경쟁 가능한 성능의 RNN을 구축할 수 있음을 보였다.

## 📎 Related Works

본 논문에서는 다음과 같은 관련 연구들을 다룬다.

- **Linear Transformers**: Softmax를 선형 커널로 대체하여 RNN처럼 동작하게 함으로써 추론 효율성을 높였으나, 일반적인 Transformer보다 성능이 낮다는 한계가 있다.
- **Modern Recurrent Models (RWKV, Mamba, Griffin)**: 새로운 게이팅 구조나 상태 공간 모델(State-Space Models, SSMs)을 도입하여 성능 격차를 줄였으나, 대규모 사전 학습 비용이 매우 높다.
- **T2R (Kasai et al., 2021)**: Transformer를 RNN으로 변환하려는 초기 시도로, MLP를 이용해 Softmax Attention을 근사하는 방식을 제안했다. 그러나 대규모 모델로 확장 시 불안정성이 높고 성능 하락이 심하다는 문제가 있었다.

SUPRA는 T2R과 달리 Softmax를 단순히 근사하는 것이 아니라, Linear Kernel로 완전히 교체하고 GroupNorm을 도입함으로써 대규모 모델에서도 안정적인 변환이 가능하도록 차별화하였다.

## 🛠️ Methodology

### 전체 구조 및 변환 절차

SUPRA는 사전 학습된 Transformer의 Attention 블록을 Linear Attention 구조로 교체한 후, 소량의 데이터로 추가 학습(Uptraining)을 진행한다. 이 모델은 학습 시에는 Transformer처럼 병렬 처리가 가능하며, 추론 시에는 RNN처럼 재귀적으로 동작한다.

### 주요 구성 요소 및 동작 원리

**1. Linear Kernel 및 유사도 함수**
기존의 $\text{sim}(q, k) = \exp(q^T k / \sqrt{d})$ 형태의 Softmax 유사도를 다음과 같은 선형 커널 형태로 변경한다.
먼저, 쿼리($q$)와 키($k$)를 투영하기 위해 가중치를 공유하는 작은 MLP $\phi(x) = \text{relu}(Wx + b)$를 도입한다. 여기에 Rotary Positional Embedding (RoPE)를 적용하여 최종 유사도 함수를 다음과 같이 정의한다.
$$\text{sim}(q_i, k_j) = \text{RoPE}(\phi(q_i)) \cdot \text{RoPE}(\phi(k_j))$$

**2. 정규화 전략: GroupNorm**
기존 Linear Attention의 합산 기반 정규화($\sum \text{sim}$)는 수치적으로 불안정하여 대규모 모델 학습 시 성능 저하를 유발한다. 이를 해결하기 위해 본 논문은 GroupNorm을 도입하여 출력값을 정규화한다.

**3. 감쇠 인자 (Decay Factor)**
컨텍스트의 영향을 조절하기 위해 고정된 감쇠 벡터 $\gamma \in (0, 1)^h$ (여기서 $h$는 head의 수)를 도입한다. 이를 통해 과거 정보의 영향력을 부드럽게 조절한다.

**4. 최종 Attention 수식**
위의 요소들을 종합한 SUPRA의 Attention 연산은 다음과 같다.
$$v'_i = \text{GroupNorm} \left( \sum_{j=1}^i \gamma^{i-j} \text{sim}(q_i, k_j) v_j \right)$$

### 추론 절차 (Recurrent Form)

추론 시에는 다음과 같이 상태 $s_i$를 업데이트하는 RNN 형태로 동작하여 메모리 비용을 상수로 유지한다.

- 상태 업데이트: $s_i = s_{i-1} + \text{RoPE}(\phi(k_i)) v_i^T$
- 결과 산출: $v'_i = \text{GroupNorm}(\text{RoPE}(\phi(q_i))^T s_i)$

## 📊 Results

### 실험 설정

- **대상 모델**: Llama2-7B, Mistral-7B (사전 학습된 Transformer)
- **비교 대상**: RWKV-5, Mamba-7B (처음부터 학습된 Linear 모델)
- **평가 지표**:
  - 일반 NLU 벤치마크: HellaSwag, PIQA, ARC-E, ARC-C, MMLU 등
  - 장문 컨텍스트(Long-context) 평가: SCROLLS 벤치마크의 Qasper, NarrativeQA

### 주요 결과

1. **일반 언어 이해 성능**:
   - Mistral-SUPRA 모델은 매우 적은 Uptraining만으로도 RWKV-5를 능가하며, 1.2T 토큰으로 처음부터 학습시킨 Mamba-7B와 경쟁 가능한 성능을 보였다.
   - 이는 기존 Transformer의 강력한 사전 학습 지식을 유지하면서 RNN의 효율성을 얻었음을 의미한다.

2. **장문 컨텍스트 처리**:
   - Linear 모델(SUPRA 포함)은 학습된 시퀀스 길이보다 긴 입력에 대해 어느 정도 성능을 유지하는 경향을 보였다.
   - 하지만 절대적인 성능 면에서는 Vanilla Transformer(특히 YaRN 기법을 적용한 모델)가 훨씬 강력하며, Linear 모델과 Transformer 사이에 여전히 상당한 성능 격차가 존재함이 확인되었다.

3. **Ablation Study**:
   - 정규화 방식의 중요성을 확인했다. T2R 방식의 근사 기법은 대규모 모델에서 불안정했으나, GroupNorm을 사용한 SUPRA는 매우 안정적으로 학습되었다.
   - 또한, Softmax Attention을 명시적으로 근사하려는 2단계 학습법은 성능 향상에 도움이 되지 않았으며, 단순히 구조를 교체하고 학습시키는 것이 더 효과적임을 발견했다.

## 🧠 Insights & Discussion

### 강점 및 의의

SUPRA는 막대한 계산 자원 없이도 고성능의 RNN LLM을 구축할 수 있는 실용적인 레시피를 제공한다. 특히, proprietary 데이터로 학습된 강력한 Transformer 모델이 있다면, 이를 Linear 모델로 빠르게 전환하여 RNN의 장점을 실험할 수 있다는 점에서 가치가 크다.

### 한계 및 비판적 해석

- **In-Context Learning (ICL) 능력 결여**: MMLU와 같은 5-shot 평가에서 성능이 크게 하락했다. 이는 Linear Attention 구조가 Transformer의 핵심 능력인 ICL 능력을 완전히 계승하지 못함을 시사한다.
- **장문 처리의 실질적 한계**: 이론적으로는 무한한 길이를 처리할 수 있으나, 실제 NLU 작업에서는 Transformer보다 성능이 낮다. 저자들은 이를 해결하기 위해 더 정교한 게이팅 메커니즘이나 고차 Linear Attention이 필요할 것이라고 분석한다.
- **근사의 부재**: 부록(Appendix A)의 분석 결과, SUPRA의 Linear Attention 행렬은 원래 Transformer의 Softmax Attention 행렬을 근사하는 것이 아니라, 완전히 새로운 매핑을 학습하는 것임이 밝혀졌다.

## 📌 TL;DR

본 논문은 사전 학습된 Transformer LLM을 적은 비용(약 5%의 학습 비용)으로 RNN으로 변환하는 **SUPRA** 기법을 제안한다. MLP 커널, RoPE, 그리고 GroupNorm을 결합하여 학습 안정성을 확보하였으며, 이를 통해 RWKV나 Mamba와 같은 최신 Linear 모델에 필적하는 성능을 달성했다. 다만, RNN으로의 변환 후에도 In-Context Learning 및 복잡한 장문 이해 능력에서는 여전히 Softmax Transformer와의 성능 격차가 존재한다는 점을 확인하였다. 이 연구는 향후 효율적인 재귀적 LLM 연구를 가속화하는 중요한 도구가 될 것으로 보인다.
