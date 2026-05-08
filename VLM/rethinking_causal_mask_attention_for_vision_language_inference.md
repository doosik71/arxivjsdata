# Rethinking Causal Mask Attention for Vision-Language Inference

Xiaohuan Pei, Tao Huang, Yanxiang Ma, Chang Xu (2025)

## 🧩 Problem to Solve

본 논문은 autoregressive Vision-Language Models (VLMs)에서 기본적으로 사용되는 Causal Mask Attention 메커니즘의 한계점을 지적한다. 현재 대부분의 VLM은 Large Language Models (LLMs)의 설계를 그대로 계승하여, 모든 토큰이 자신의 이전 토큰만을 참조하도록 하는 엄격한 left-to-right 인과적 마스킹(causal masking)을 적용한다.

그러나 텍스트와 달리 시각적 정보는 본질적으로 비순차적(non-sequential)이며, 이미지 내의 특정 영역은 순서와 상관없이 전체적으로 처리되어야 한다. 따라서 시각적 토큰(vision tokens)에 대해 엄격한 causal mask를 적용하는 것은 모델이 추론에 필수적인 미래 컨텍스트(future context)의 세만틱 큐를 활용하는 능력을 저해하며, 이는 결과적으로 VLM의 추론 성능을 제한하는 요소가 된다. 본 논문의 목표는 시각적 토큰에 최적화된 새로운 마스킹 전략을 제안하여, autoregressive 구조를 유지하면서도 시각적 추론 능력을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 시각적 토큰에 한해서만 미래 토큰을 미리 볼 수 있도록 하는 **Future-Aware Attention**을 도입하는 것이다.

1. **Future-Aware Causal Masks 제안**: 시각적 쿼리가 참조할 수 있는 미래 토큰의 범위를 세 가지 수준($M_f, M_{v2v}, M_{v2t}$)으로 정의하여, 작업의 성격에 따라 선택적으로 미래 정보를 활용할 수 있게 한다.
2. **Light Future Aware Attention (Merge) 메커니즘**: 미래 정보를 직접 참조할 때 발생하는 추론 지연(latency) 문제를 해결하기 위해, Prefill 단계에서 미래의 시각적 정보를 커널 풀링(kernel pooling)을 통해 압축하고 이를 과거의 특정 위치(Attention Sink 영역)에 병합하는 경량화된 기법을 제안한다.
3. **모달리티별 마스킹 영향 분석**: 시각적 토큰과 텍스트 토큰에 대해 causal mask를 완화했을 때의 영향을 실험적으로 분석하여, 시각적 토큰에는 완화된 마스크가 유리하지만 텍스트 토큰에는 엄격한 마스크가 필수적임을 입증하였다.

## 📎 Related Works

기존의 LLaVA, InternVL, Qwen-VL과 같은 VLM들은 LLM의 decoder-only 아키텍처를 그대로 사용하여 시각적 토큰과 텍스트 토큰을 하나의 시퀀스로 연결하고 동일한 causal mask를 적용한다. 일부 연구에서 해상도 인식 인코더나 정교한 토큰 상호작용 전략을 탐구했으나, LLM에서 계승된 causal masking이 시각적 토큰 처리에 미치는 영향에 대해서는 충분히 연구되지 않았다.

본 논문은 기존의 일괄적인 causal masking 방식과 달리, 시각적 정보의 비순차적 특성을 고려하여 **모달리티별로 차별화된 마스킹 전략**을 적용한다는 점에서 기존 접근 방식과 차별화된다.

## 🛠️ Methodology

### 1. Future-Aware Causal Masks

본 논문은 시각적 토큰 $V$와 텍스트 토큰 $T$를 구분하여, 시각적 쿼리($i \in V$)에 대해 다음과 같은 세 가지 마스킹 전략을 정의한다.

* **Future-Aware Full Mask ($M_f$)**: 시각적 쿼리가 미래의 모든 토큰(시각적 및 텍스트 토큰 모두)을 참조할 수 있게 한다.
    $$M^f_{i,j} = \begin{cases} 0, & \text{if } j \le i \lor (j > i \land i \in V) \\ -\infty, & \text{otherwise} \end{cases}$$
* **Future-Aware Visual-to-Visual Mask ($M_{v2v}$)**: 시각적 쿼리가 미래의 시각적 토큰만 참조하고, 미래의 텍스트 토큰은 마스킹한다.
    $$M^{v2v}_{i,j} = \begin{cases} 0, & \text{if } j \le i \lor (j > i \land i,j \in V) \\ -\infty, & \text{otherwise} \end{cases}$$
* **Future-Aware Visual-to-Textual Mask ($M_{v2t}$)**: 시각적 쿼리가 미래의 텍스트 토큰만 참조하고, 미래의 시각적 토큰은 마스킹한다.
    $$M^{v2t}_{i,j} = \begin{cases} 0, & \text{if } j \le i \lor (j > i \land i \in V, j \in T) \\ -\infty, & \text{otherwise} \end{cases}$$

### 2. Light Future Aware Attention (Merge)

미래 토큰을 직접 참조하는 방식은 디코딩 단계에서 연산 비용을 증가시킨다. 이를 해결하기 위해 본 논문은 Prefill 단계에서 미래 정보를 압축하여 과거 영역에 병합하는 방식을 제안한다.

* **커널 풀링(Kernel Pooling)**: 1D 커널 풀링을 사용하여 미래 영역의 어텐션 가중치를 집계하여 요약 점수를 생성한다.
* **병합(Merging)**: 집계된 요약 점수를 과거의 초기 토큰 위치(Attention Sink)에 병합한다. 이를 통해 디코딩 단계에서는 표준적인 lower-triangular causal mask를 그대로 사용하면서도, Prefill 단계에서 집계된 미래 정보를 활용할 수 있다.
* **최종 연산**: 수정된 어텐션 분포 $C(B, \mu)$를 기존 어텐션 계산에 추가하여 예측을 수행한다.
    $$h'_\theta(x^v, x^t; \mu) = \text{Softmax}(B(x^v, x^t) + C(B, \mu) + M^c)$$

### 3. 구현 상세

효율적인 계산을 위해 FlashAttention 프레임워크를 통합하여, 블록 단위로 마스킹 로직을 적용하고 fused softmax 연산을 통해 수치적 안정성과 메모리 효율성을 확보하였다.

## 📊 Results

### 실험 설정

* **모델**: LLaVA-7b, LLaVA-13b
* **데이터셋**: MILEBench의 29개 벤치마크 (Temporal Multi-image, Semantic Multi-image, Needle In a Haystack 등)
* **지표**: Accuracy, ROUGE-L

### 주요 결과

1. **작업별 최적 마스크**:
    * **Temporal Multi-image (시간적 다중 이미지)**: $M_f$와 $M_{v2v}$가 성능을 크게 향상시켰다. 이는 이벤트 시퀀스나 객체의 움직임 궤적을 파악하기 위해 미래의 시각적 큐가 필수적이기 때문이다.
    * **Visual Relation (시각적 관계)**: $M_{v2v}$가 가장 효과적이었다. 이미지 간의 미세한 차이를 포착하는 작업에서는 시각적 토큰 간의 상호작용이 중요하기 때문이다.
    * **Text-Rich Image QA (텍스트 풍부 이미지 QA)**: $M_{v2t}$가 높은 성능 향상을 보였다. 이미지 내의 텍스트 정보를 해석하기 위해 미래의 텍스트 토큰을 미리 참조하는 것이 유리했다.

2. **경량화 병합(Merge)의 효율성**:
    * **성능**: `Merge` 기법을 적용했을 때, 미래 정보를 직접 참조하는 방식과 대등하거나 오히려 더 나은 성능을 보였다. 특히 단 하나의 토큰(prefix size = 1)에 정보를 병합하는 것만으로도 충분한 효과를 거두었다.
    * **지연 시간(Latency)**: 디코딩 속도가 획기적으로 개선되었다. 예를 들어 $M_f$의 경우 토큰당 83.18ms에서 $M_f + \text{merge}$ 적용 시 26.53ms로 약 3배 이상의 속도 향상을 보였다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 연구는 VLM에서 무비판적으로 수용되었던 LLM의 causal masking 구조를 재검토하여, 모달리티의 특성에 맞는 마스킹의 필요성을 입증하였다. 특히, 미래 정보를 단순히 개방하는 것이 아니라 '압축 후 병합'하는 방식을 통해 성능과 효율성 사이의 Trade-off를 성공적으로 해결하였다.

### 한계 및 논의사항

* **작업 의존성**: 모든 작업에 동일한 마스크가 적용되는 것이 아니라, 작업의 성격(시각 중심 vs 텍스트 중심)에 따라 최적의 마스크가 다르다. 이는 실제 적용 시 작업 유형을 미리 파악하거나 동적으로 마스크를 선택하는 메커니즘이 필요함을 시사한다.
* **텍스트 토큰의 제약**: 실험 결과, 텍스트 토큰의 causal mask를 완화하면 오히려 성능이 저하되었다. 이는 언어의 순차적 특성이 시각 정보보다 훨씬 강하며, autoregressive 생성의 핵심인 인과성이 텍스트 영역에서는 절대적으로 중요함을 의미한다.

## 📌 TL;DR

본 논문은 VLM의 시각적 토큰에 적용되는 엄격한 Causal Mask가 추론 능력을 저하시킨다는 점을 발견하고, 이를 해결하기 위한 **Future-Aware Attention**과 이를 효율적으로 구현한 **Merge 메커니즘**을 제안한다. 제안된 방법은 특히 시간적 추론, 시각적 관계 분석, 텍스트 중심 QA 작업에서 성능을 향상시키며, 병합 기법을 통해 표준 causal decoding 수준의 빠른 추론 속도를 유지한다. 이 연구는 향후 VLM 설계 시 모달리티별 특성을 고려한 차별적 어텐션 구조 설계의 중요성을 강조한다.
