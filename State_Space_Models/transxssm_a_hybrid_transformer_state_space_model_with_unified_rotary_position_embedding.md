# TransXSSM: A Hybrid Transformer–State Space Model with Unified Rotary Position Embedding

Bingheng Wu, Jingze Shi, Yifan Wu, Nan Tang, Yuyu Luo (2025)

## 🧩 Problem to Solve

본 연구는 Transformer와 State Space Model(SSM)의 각기 다른 장점을 결합하려는 하이브리드 아키텍처에서 발생하는 **위치 인코딩의 불일치(Positional Encoding Incompatibility)** 문제를 해결하고자 한다.

Transformer는 장거리 의존성 캡처 능력이 뛰어나지만, Self-Attention의 이차 시간 복잡도($O(n^2)$)로 인해 긴 시퀀스 처리 시 계산 효율성이 급격히 떨어진다. 반면, SSM은 선형 시간 복잡도($O(n)$)를 통해 매우 긴 시퀀스에서도 높은 처리량(throughput)을 제공하지만, 위치 정보를 암시적으로만 인코딩하므로 추론 능력과 few-shot 학습 능력이 상대적으로 부족하다.

기존의 하이브리드 모델들은 두 구조를 단순히 교차 배치하는 방식을 취했다. 그러나 Transformer는 Rotary Position Embedding(RoPE)과 같은 명시적인 위치 인코딩을 사용하는 반면, SSM은 컨볼루션과 재귀(recursion)를 통해 위치를 암시적으로 표현한다. 이러한 근본적인 메커니즘의 차이는 모듈 간 인터페이스에서 정보의 단절(information gap)과 불연속성을 초래하며, 이는 결과적으로 하이브리드 모델의 성능을 저하시키는 주요 원인이 된다. 따라서 본 논문의 목표는 두 아키텍처가 동일한 위치 체계를 공유하도록 하여, 효율성과 성능을 동시에 잡은 하이브리드 모델인 TransXSSM을 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Transformer의 Self-Attention과 SSM의 상태 업데이트 메커니즘 모두에 동일하게 적용될 수 있는 **Unified Rotary Position Embedding (Unified RoPE)**를 제안한 것이다.

핵심 직관은 SSM의 상태 업데이트 신호를 Transformer의 Query($Q$) 및 Key($K$) 벡터와 유사하게 취급하는 것이다. SSM의 내부 업데이트 파라미터에 RoPE의 회전 변환을 적용함으로써, 모델 전체에 걸쳐 일관된 위치 위상(positional phase)을 유지하게 한다. 이를 통해 모듈 간의 전환 시 위치 정보가 리셋되거나 정렬이 어긋나지 않는 '스펙트럼 연속성(spectral continuity)'을 확보하였으며, 결과적으로 Transformer의 정밀한 위치 추론 능력과 SSM의 선형 효율성을 통합한 TransXSSM 아키텍처를 구현하였다.

## 📎 Related Works

기존의 하이브리드 모델링 연구들은 S4 레이어와 로컬 어텐션을 결합하거나, SSM 레이어를 Transformer 블록 앞에 배치하는 등의 시도를 해왔다. 최근에는 Mamba와 어텐션 레이어를 교차 배치하거나, Transformer의 MLP 레이어를 Mamba 레이어로 대체하는 방식이 제안되었다. 특히 Jamba는 Transformer와 Mamba 블록의 적절한 배치 비율을 통해 성능과 효율성의 균형을 맞춘 선구적인 대규모 하이브리드 모델로 평가받는다.

그러나 기존의 접근 방식들은 대부분 두 모듈을 단순히 층으로 쌓는 것에 집중했을 뿐, 위치 인코딩 방식의 차이에서 오는 불일치 문제를 깊게 다루지 않았다. 본 연구는 이러한 '위치 스펙트럼 불연속성'이 하이브리드 모델의 성능 확장을 방해하는 핵심 요소라고 지적하며, Unified RoPE를 통해 이를 해결함으로써 기존 하이브리드 모델들과 차별점을 둔다.

## 🛠️ Methodology

### 1. Unified Rotary Position Embedding (Unified RoPE)

Unified RoPE는 Self-Attention의 $Q, K$ 벡터뿐만 아니라 SSM의 상태 업데이트 벡터인 $C$와 $B$에도 동일한 회전 변환을 적용한다.

**수학적 정의:**
절대 위치 인덱스를 $m$ (query-like: $Q, C$)과 $n$ (key-like: $K, B$)이라고 할 때, Unified RoPE는 다음과 같이 정의된다.
$$f_Q(q, m) = q e^{i, m, \theta}, \quad f_K(k, n) = k e^{i, n, \theta}, \quad f_C(c, m) = c e^{i, m, \theta}, \quad f_B(b, n) = b e^{i, n, \theta}$$
여기서 $\theta$는 기본 각주파수이다. 이 복소수 표현을 실제 구현에서는 2D 회전 행렬을 이용한 실수 행렬 형태로 변환하여 적용한다.

**상대적 위치 캡처:**
이렇게 변환된 벡터들 간의 내적을 계산하면, 결과값은 두 토큰의 상대적 거리인 $m-n$에 의존하게 된다.
$$\text{attn score} = \langle f_Q(q_m, m), f_K(k_n, n) \rangle = q_m R_{\Theta, m-n}^d k_n^\top$$
$$\text{ssd score} = \langle f_C(c_m, m), f_B(b_n, n) \rangle = c_m R_{\Theta, m-n}^d b_n^\top$$
이로 인해 Transformer 레이어와 SSM 레이어는 동일한 상대적 위치 정보를 공유하며, 데이터가 어떤 모듈을 통과하더라도 일관된 위치 정보를 유지할 수 있다.

### 2. TransXSSM 아키텍처 설계

TransXSSM은 Unified RoPE를 기반으로 하며, 다음과 같은 설계 원칙을 따른다.

* **하이브리드 모듈 적층 (Hybrid Module Stacking):** SSM 레이어와 Self-Attention(SA) 레이어를 **7:1 비율**로 배치한다. 즉, 7개의 SSM 서브 레이어 이후에 1개의 SA 서브 레이어가 위치한다. 이는 SSM이 대부분의 시퀀스 처리를 효율적으로 담당하게 하고, 주기적인 SA 레이어가 글로벌 컨텍스트 믹싱과 강력한 관계 추론 능력을 주입하도록 하기 위함이다.
* **피처 정제 (Feature Refinement):** 각 SA 또는 SSM 서브 레이어 이후에는 표준 Transformer 스타일의 Feed-Forward Network(FFN)를 배치하여 특징을 정제하고 모델의 표현력을 유지한다.
* **안정성 및 정규화:** 긴 시퀀스 학습 시 발생할 수 있는 상태 벡터 드리프트나 상태 붕괴(state collapse)를 방지하기 위해, 모든 서브 레이어와 FFN 주변에 **RMSNorm**과 **잔차 연결(Residual Connection)**을 적용한다.

**전체 파이프라인:**
$\text{Input Tokens} \rightarrow \text{Embedding} \rightarrow \text{N stacked modules (7 SSMs + 1 SA, each with FFN, RMSNorm, and Unified RoPE)} \rightarrow \text{Final Norm} \rightarrow \text{LM Head} \rightarrow \text{Output}$

## 📊 Results

### 1. 실험 설정 및 지표

* **데이터셋 및 토크나이저:** Smollm-Corpus 데이터셋과 NeoX 토크나이저를 사용하였다.
* **비교 대상:** Llama3 (Pure Transformer), Mamba2 (Pure SSM), Jamba (Hybrid).
* **모델 규모:** 320M 및 1.3B 파라미터 규모에서 실험을 진행하였다.
* **평가 지표:** Perplexity(PPL), 다양한 다운스트림 벤치마크(MMLU, TriviaQA, ARC, PIQA, HellaSwag, OBQA, Winogrande)의 정확도, Throughput(it/s).

### 2. 주요 결과

* **계산 효율성:** 4K 시퀀스 길이에서 TransXSSM은 표준 Transformer 대비 **학습 속도는 42.3%, 추론 속도는 29.5% 향상**되었다. 처리량(throughput) 면에서는 순수 SSM인 Mamba2보다는 약간 낮지만, Llama3나 Jamba보다는 훨씬 높은 효율성을 보였다.
* **정량적 성능:** 1.3B 모델 기준, TransXSSM은 7개의 다양한 벤치마크 작업에서 베이스라인 모델들을 2점 이상 앞섰으며, 특히 Winogrande에서는 Llama3(55.40) 대비 62.09라는 압도적인 성능 향상을 보였다.
* **장거리 컨텍스트 처리:** "Needle-in-a-Haystack" 테스트에서 TransXSSM은 매우 높은 정확도를 기록하며, Unified RoPE가 다단계 추론(multi-hop reasoning)과 깊은 컨텍스트 모델링에 효과적임을 입증하였다.
* **확장성 (Scalability):** 320M에서 1.3B로 모델 규모를 확장했을 때, TransXSSM의 평균 성능 향상 폭(+7.22)이 Llama3(+6.37)나 Mamba2(+6.0)보다 컸다. 이는 모델 규모가 커질수록 하이브리드 구조의 이점이 더 명확해짐을 의미한다.

## 🧠 Insights & Discussion

본 논문은 Transformer와 SSM의 결합에서 가장 간과되었던 **위치 인코딩의 일관성** 문제를 정확히 짚어내어 해결하였다. Unified RoPE를 통해 두 아키텍처 간의 '언어(위치 표현)'를 통일함으로써, 정보 손실 없이 효율적인 선형 시간 복잡도와 강력한 추론 능력을 동시에 확보할 수 있었다.

**강점:**

* 단순한 레이어 교차 배치를 넘어, 수학적 근거(SSD duality)를 바탕으로 위치 인코딩을 통합하였다.
* SSM의 효율성을 유지하면서도 Transformer의 강점인 정밀한 위치 추론 능력을 보존하였다.
* 모델 규모 확장 시 성능 이득이 더 커지는 우수한 확장성을 보여주었다.

**한계 및 논의사항:**

* RoPE 자체의 한계로 인해 고차 비트(high bit) 부분이 충분히 학습되지 않아 일반화 능력이 떨어질 수 있다는 점이 언급되었다. 논문에서는 이를 해결하기 위해 학습 데이터의 유효 시퀀스 길이를 기준으로 로그 스케일 확장을 적용하였으나, 더 근본적인 해결책에 대한 논의는 부족하다.
* 7:1이라는 적층 비율이 실험적으로 도출되었으나, 작업의 성격에 따라 최적의 비율이 달라질 가능성이 있으며 이에 대한 세밀한 분석은 추가 연구가 필요해 보인다.

## 📌 TL;DR

TransXSSM은 Transformer의 RoPE를 SSM의 상태 업데이트 메커니즘으로 확장한 **Unified RoPE**를 통해, 서로 다른 위치 인코딩 방식을 사용하는 두 모델을 유기적으로 결합한 하이브리드 아키텍처이다. 이 모델은 순수 Transformer보다 훨씬 빠르고, 순수 SSM보다 추론 능력이 뛰어나며, 특히 모델 규모가 커질수록 그 효율성과 성능적 이점이 극대화된다. 이는 향후 초거대 언어 모델(LLM)의 효율적인 장거리 컨텍스트 모델링을 위한 중요한 설계 방향을 제시한다.
