# Mixtral of Experts

Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed (2024)

## 🧩 Problem to Solve

본 논문은 거대 언어 모델(LLM)의 성능을 높이기 위해 파라미터 수를 늘리면서도, 추론 시 발생하는 계산 비용과 지연 시간(latency)을 효과적으로 제어하는 문제를 해결하고자 한다. 일반적으로 모델의 파라미터가 증가하면 성능은 향상되지만, 모든 토큰에 대해 모든 파라미터를 계산해야 하므로 추론 속도가 느려지고 컴퓨팅 자원 소모가 극심해지는 트레이드-오프 관계가 존재한다.

논문의 목표는 Sparse Mixture of Experts (SMoE) 구조를 도입하여, 전체 파라미터 수는 크게 늘려 모델의 용량을 확보하되, 실제 추론 시에는 각 토큰당 극히 일부의 파라미터만 활성화함으로써 Llama 2 70B와 같은 거대 모델에 필적하거나 이를 능가하는 성능을 효율적으로 달성하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 Mistral 7B의 아키텍처를 기반으로 하되, 모든 Feed-Forward Network (FFN) 블록을 8개의 전문가(expert) 네트워크로 구성된 SMoE 레이어로 대체한 Mixtral 8x7B를 설계한 것이다.

중심적인 직관은 모든 토큰이 모든 파라미터를 사용할 필요가 없으며, Router 네트워크를 통해 각 토큰의 특성에 맞는 최적의 전문가 2개만을 선택적으로 활용하게 함으로써 계산 효율성을 극대화하는 것이다. 이를 통해 모델은 총 47B 개의 파라미터를 보유하고 있음에도 불구하고, 추론 시에는 토큰당 13B 개의 active parameters만 사용하여 빠른 추론 속도와 높은 처리량(throughput)을 동시에 달성하였다.

## 📎 Related Works

본 논문은 Transformer 아키텍처와 Mistral 7B의 설계를 계승하며, 특히 SMoE 접근 방식의 일환으로 GShard 아키텍처를 언급한다. 

기존의 GShard와 비교했을 때 Mixtral은 다음과 같은 차별점을 가진다. 첫째, GShard는 일부 블록만을 MoE로 교체하는 반면, Mixtral은 모든 FFN 서브-블록을 MoE 레이어로 대체하였다. 둘째, Mixtral은 GShard의 복잡한 게이팅 전략 대신 보다 단순하고 효율적인 Top-K 로그 기반의 Softmax 게이팅 방식을 사용한다.

## 🛠️ Methodology

### 전체 시스템 구조
Mixtral 8x7B는 Decoder-only Transformer 구조를 따른다. 가장 큰 특징은 각 레이어의 FFN이 8개의 독립적인 전문가 블록으로 구성되어 있으며, Router가 입력 토큰에 대해 가장 적합한 2개의 전문가를 선택하여 그 출력을 가중 합산하는 방식이다.

### MoE 레이어 및 라우팅 메커니즘
입력 벡터 $x$에 대해 MoE 모듈의 출력은 게이팅 네트워크(Router)의 출력값에 의해 결정되는 전문가 네트워크들의 가중 합으로 정의된다. $n$개의 전문가 네트워크 $\{E_0, E_1, \dots, E_{n-1}\}$가 있을 때, 출력은 다음과 같이 계산된다.

$$y = \sum_{i=0}^{n-1} G(x)_i \cdot E_i(x)$$

여기서 $G(x)_i$는 $i$번째 전문가에 대한 게이팅 네트워크의 출력값이며, $E_i(x)$는 해당 전문가 네트워크의 출력이다. 본 논문에서는 단순하고 성능이 뛰어난 Top-K 로그 기반의 Softmax 방식을 사용하여 $G(x)$를 구현하였다.

$$G(x) := \text{Softmax}(\text{TopK}(x \cdot W_g))$$

이때 $\text{TopK}(\ell)$ 함수는 로그 값 $\ell$ 중 상위 $K$개의 좌표에 대해서만 값을 유지하고 나머지는 $-\infty$로 처리하여, Softmax 이후 선택되지 않은 전문가의 가중치를 0으로 만든다. Mixtral에서는 $K=2$를 설정하여 토큰당 2개의 전문가만 활성화한다.

### 세부 구현 및 학습
- **전문가 함수**: 각 전문가 $E_i(x)$는 Mistral 7B와 동일한 SwiGLU 아키텍처를 사용한다.
- **최종 연산식**: 위 식들을 종합하면 입력 토큰 $x$에 대한 최종 출력 $y$는 다음과 같다.
$$y = \sum_{i=0}^{n-1} \text{Softmax}(\text{Top2}(x \cdot W_g))_i \cdot \text{SwiGLU}_i(x)$$
- **효율적 추론**: SMoE의 효율적인 실행을 위해 Megablocks CUDA 커널을 통합하여 희소 행렬 곱셈(sparse matrix multiplication)으로 FFN 연산을 처리함으로써 실행 속도를 높였다.
- **학습 설정**: 32k 토큰의 컨텍스트 사이즈로 사전 학습되었으며, 다국어 데이터를 대폭 상향 샘플링하여 학습하였다.
- **Instruction Fine-tuning**: 지도 학습 기반의 미세 조정(SFT)과 직접 선호도 최적화(Direct Preference Optimization, DPO)를 순차적으로 적용하여 Mixtral 8x7B-Instruct 모델을 생성하였다.

## 📊 Results

### 실험 설정 및 지표
Mixtral은 Llama 2 7B, 13B, 70B 및 Mistral 7B와 비교 평가되었다. 평가 항목은 상식 추론(Hellaswag, ARC-Challenge 등), 세계 지식(NaturalQuestions, TriviaQA), 독해(BoolQ, QuAC), 수학(GSM8K, MATH), 코드(HumanEval, MBPP) 및 종합 지표(MMLU, BBH)를 포함한다.

### 주요 정량적 결과
- **전반적 성능**: Mixtral 8x7B는 대부분의 벤치마크에서 Llama 2 70B 및 GPT-3.5와 비슷하거나 더 우수한 성능을 보였다.
- **특화 영역**: 특히 수학과 코드 생성 벤치마크에서 Llama 2 70B를 압도적으로 능가하였다. (예: MBPP pass@1 기준 Mixtral 60.7% vs Llama 2 70B 49.8%)
- **효율성**: 추론 시 active parameters가 13B에 불과함에도 불구하고, 70B 파라미터를 모두 사용하는 Llama 2 70B보다 우수한 성능을 내어 계산 효율성을 입증하였다.
- **다국어 성능**: 프랑스어, 독일어, 스페인어, 이탈리아어 벤치마크에서 Llama 2 70B보다 유의미하게 높은 성능을 기록하였다.
- **긴 컨텍스트 처리**: Passkey retrieval 테스트에서 컨텍스트 길이와 키 위치에 관계없이 100% 정확도를 기록하여 32k 컨텍스트 윈도우를 성공적으로 활용함을 보였다.

### Instruct 모델 평가
Mixtral 8x7B-Instruct는 MT-Bench에서 8.30점을 기록하였으며, LMSys 챗봇 아레나의 인간 평가 결과 GPT-3.5-Turbo, Claude-2.1, Gemini Pro, Llama 2 70B-chat을 모두 앞서는 성능을 보였다.

## 🧠 Insights & Discussion

### 라우팅 분석 (Routing Analysis)
본 논문에서는 전문가 선택의 패턴을 분석하여 특정 전문가가 특정 도메인(예: 수학, 생물학)에 특화되었는지 조사하였다. 분석 결과, 놀랍게도 도메인별로 뚜렷한 전문가 할당 패턴은 관찰되지 않았다.

대신, 라우터는 **구문적(syntactic) 행동**을 보이는 것으로 나타났다. 예를 들어 Python 코드의 'self'나 영어의 'Question'과 같은 특정 단어들은 서로 다른 도메인임에도 불구하고 동일한 전문가에게 할당되는 경향이 있었다. 또한, 레이어가 깊어질수록(특히 15번, 31번 레이어) 인접한 토큰들이 동일한 전문가를 선택하는 시간적 지역성(temporal locality)이 강하게 나타났다. 이는 향후 캐싱(caching) 기법을 통해 추론 속도를 더 최적화할 수 있는 가능성을 시사한다.

### 강점 및 한계
- **강점**: 모델 용량(Capacity)을 획기적으로 늘리면서도 추론 비용을 낮춘 점, 그리고 오픈 소스 라이선스(Apache 2.0)로 배포하여 접근성을 높인 점이 매우 높게 평가된다.
- **한계**: 논문에서는 추론 시의 계산 비용(Active parameters)에 집중하였으나, 전체 파라미터(47B)를 메모리에 적재해야 하므로 실제 서빙 시의 메모리 비용은 여전히 높다는 점이 언급된다. 또한, 전문가 병렬 처리(Expert Parallelism) 시 일부 전문가에게 부하가 집중되는 로드 밸런싱 문제가 발생할 수 있다.

## 📌 TL;DR

Mixtral 8x7B는 모든 FFN 레이어를 8개의 전문가로 구성하고 토큰당 2개만을 활성화하는 Sparse Mixture of Experts (SMoE) 모델이다. 이 모델은 13B 개의 active parameters만으로 Llama 2 70B 및 GPT-3.5의 성능을 능가하거나 대등한 수준으로 구현하였으며, 특히 수학, 코드, 다국어 처리에서 탁월한 능력을 보인다. 본 연구는 거대 모델의 파라미터 효율성을 극대화하는 새로운 기준을 제시하였으며, 오픈 소스 생태계에서 고성능 LLM의 보급과 최적화 연구에 중요한 기여를 할 것으로 기대된다.