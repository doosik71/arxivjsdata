# EXPLORING ADVANCED LARGE LANGUAGE MODELS WITH LLMSUITE

Giorgio Roffo(2024)

## 🧩 Problem to Solve

본 논문은 현대의 거대 언어 모델(Large Language Models, LLMs)인 ChatGPT나 Gemini 등이 가진 내재적 한계점을 해결하기 위한 다양한 기술적 접근법과 아키텍처를 분석하는 것을 목표로 한다. 구체적으로 해결하고자 하는 문제는 다음과 같다.

첫째, **지식의 시간적 단절(Temporal Knowledge Cutoffs)** 문제이다. LLM은 학습 데이터가 수집된 시점까지만의 정보만을 가지고 있으므로, 최신 정보에 대해 답변할 때 한계가 있다. 둘째, **수학적 부정확성(Mathematical Inaccuracies)**이다. LLM은 기본적으로 다음 토큰을 예측하는 확률 모델이므로, 정밀한 계산이 필요한 수학적 문제에서 오류를 범하는 경향이 있다. 셋째, **환각(Hallucinations)** 현상이다. 모델이 그럴듯해 보이지만 사실과는 다른 잘못된 정보를 생성하는 문제가 발생한다.

결과적으로 본 연구는 이러한 신뢰성과 정확성의 문제를 극복하기 위해 외부 데이터 결합, 고급 프롬프팅, 효율적인 미세 조정(Fine-tuning) 및 정렬(Alignment) 기법들을 종합적으로 정리하여 제시하고자 한다.

## ✨ Key Contributions

본 논문은 LLM의 성능과 신뢰성을 높이기 위한 핵심 아이디어를 다음과 같은 세 가지 관점에서 제시한다.

1. **외부 자원 및 도구의 통합**: 모델의 내부 파라미터에만 의존하지 않고, Retrieval Augmented Generation (RAG)을 통해 최신 데이터를 검색하고, Program-Aided Language Models (PAL)를 통해 외부 코드 인터프리터를 활용함으로써 지식의 최신성과 계산의 정확성을 확보한다.
2. **추론 및 실행 프레임워크의 고도화**: Chain-of-Thought (CoT) 프롬프팅으로 논리적 단계별 사고를 유도하고, ReAct 프레임워크를 통해 '추론(Reasoning)'과 '행동(Acting)'을 결합하여 복잡한 워크플로우를 실행하는 체계를 구축한다.
3. **효율적인 학습 및 정렬 전략**: LoRA와 같은 Parameter-Efficient Fine-Tuning (PEFT)를 통해 자원 소모를 줄이면서 특정 태스크에 최적화하며, RLHF와 ReST를 통해 모델의 출력을 인간의 선호도에 맞게 정렬하는 방법론을 제시한다.

## 📎 Related Works

논문에서는 LLM의 근간이 되는 Transformer 아키텍처와 그 변형 모델들을 소개한다.

- **Transformer**: Self-attention 메커니즘을 통해 문장 내 단어 간의 관계를 병렬적으로 처리함으로써 기존 RNN의 한계를 극복하였다.
- **모델 분류**: 입력 텍스트의 깊은 이해를 위한 Encoder-only 모델(BERT), 요약 및 번역과 같은 시퀀스-투-시퀀스 작업에 적합한 Encoder-Decoder 모델(BART, T5), 그리고 텍스트 생성에 특화된 Decoder-only 모델(GPT 시리즈, LLaMA)로 구분하여 설명한다.
- **기존 접근 방식의 한계**: 단순한 모델 크기의 확장(Scaling)만으로는 환각 문제나 최신 정보 반영 문제를 완전히 해결할 수 없으며, 모든 파라미터를 업데이트하는 Full Fine-tuning은 막대한 계산 자원이 필요하고 '치명적 망각(Catastrophic Forgetting)' 현상을 야기한다는 한계가 있다.

## 🛠️ Methodology

### 1. 성능 향상을 위한 외부 통합 프레임워크

- **Retrieval Augmented Generation (RAG)**: 외부 데이터베이스나 API에서 관련 정보를 검색(Retrieve)하여 프롬프트에 추가한 뒤 생성(Generate)하는 방식이다. 이는 모델을 재학습시키지 않고도 최신 정보를 반영하고 환각을 줄이는 효과가 있다.
- **Program-Aided Language Models (PAL)**: LLM이 직접 계산하는 대신, 계산 과정을 Python 코드로 생성하고 이를 외부 인터프리터에서 실행하여 정확한 결과값을 얻는 방식이다.
- **ReAct (Reasoning + Acting)**: '생각 $\rightarrow$ 행동 $\rightarrow$ 관찰'의 루프를 반복한다. 모델이 추론 과정을 생성(Thought)하고, 외부 도구를 사용해 행동(Action)하며, 그 결과(Observation)를 다시 추론에 반영하여 정답에 도달한다.
- **LangChain**: 위와 같은 RAG, PAL, ReAct 등을 모듈화하여 LLM 기반 애플리케이션을 쉽게 구축할 수 있게 돕는 오픈소스 프레임워크이다.

### 2. Transformer 아키텍처 및 수식

Self-attention은 다음과 같이 계산된다.
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
여기서 $Q, K, V$는 각각 Query, Key, Value 행렬이다. Multi-Head Attention은 여러 개의 attention head를 병렬로 처리하여 다양한 관점의 관계를 캡처한다.
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$

### 3. 학습 자원 최적화 및 스케일링

- **ZeRO (Zero Redundancy Optimizer)**: GPU 메모리 낭비를 줄이기 위해 Optimizer states(Stage 1), Gradients(Stage 2), Parameters(Stage 3)를 여러 GPU에 분산 배치(Sharding)한다.
- **FSDP (Fully Sharded Data Parallel)**: ZeRO의 개념을 PyTorch에 구현하여 모델 파라미터를 샤딩하고 필요할 때만 복원하여 사용함으로써 대규모 모델 학습을 가능하게 한다.
- **1-bit LLMs (BitNet b1.58)**: 가중치를 $\{-1, 0, 1\}$의 ternary 값으로 양자화하여 메모리 사용량을 획기적으로 줄이고 에너지 효율을 높인다.

### 4. 미세 조정 및 정렬 (Fine-tuning & Alignment)

- **LoRA (Low-Rank Adaptation)**: 기존 가중치 $W$를 고정한 채, 두 개의 작은 저차원 행렬 $A, B$의 곱으로 변화량을 학습한다.
    $$W \approx A \cdot B \quad (r \ll \min(d, k))$$
- **Prompt Tuning**: 입력 임베딩 앞에 학습 가능한 가상 토큰인 'Soft Prompts'를 추가하여 모델의 가중치 변경 없이 태스크에 적응시킨다.
- **RLHF vs ReST**:
  - **RLHF**: 인간의 피드백으로 보상 모델(Reward Model)을 만들고, PPO 알고리즘을 통해 정책을 업데이트한다.
  - **ReST**: 'Grow(데이터 생성)' 단계와 'Improve(필터링 및 학습)' 단계를 분리하여 오프라인으로 학습함으로써 RLHF보다 계산 효율적이고 안정적이다.
- **PPO (Proximal Policy Optimization)**: 업데이트 범위를 Trust Region 내로 제한하여 학습의 안정성을 확보한다. 전체 손실 함수는 다음과 같다.
    $$L(\theta) = \mathbb{E}_t [L_{CLIP}(\theta) - C_1 L_{VF}(\theta) + C_2 S[\pi](s_t)]$$
    여기서 $L_{CLIP}$은 정책 손실, $L_{VF}$는 가치 함수 손실, $S$는 다양성을 위한 엔트로피 항이다.

## 📊 Results

본 논문은 튜토리얼 성격의 보고서이므로 새로운 실험 결과보다는 기존 벤치마크와 기법들의 효과를 정리하여 제시한다.

- **평가 데이터셋**: GLUE, SuperGLUE, MMLU, BIG-Bench 등 자연어 이해 및 다중 작업 벤치마크를 통해 모델의 성능을 측정한다.
- **BitNet b1.58의 정량적 효과**: FP16 모델 대비 3B 파라미터 기준 GPU 메모리 사용량을 3.55배 줄였으며, 추론 속도는 최대 2.71배, 처리량(Throughput)은 8.9배 향상되었다. 특히 행렬 곱셈 연산에서 에너지 효율이 71.4배 증가했다.
- **PEFT의 효과**: LoRA를 적용한 FLAN-T5 모델이 전체 미세 조정(Full Fine-tuning)과 대등한 ROUGE 점수를 기록하면서도 계산 자원은 훨씬 적게 소모함을 확인하였다.
- **ReST의 효과**: IWSLT 2014 및 WMT 2020 기계 번역 벤치마크에서 RLHF보다 적은 계산 비용으로도 인간 평가 및 자동 지표에서 유의미한 개선을 보였다.

## 🧠 Insights & Discussion

**강점 및 분석**:
본 논문은 단일 모델의 크기를 키우는 것보다, 외부 도구(RAG, PAL)와의 결합과 효율적인 학습 전략(PEFT, ZeRO)의 조화가 실제 애플리케이션의 신뢰성을 높이는 데 훨씬 효과적임을 강조한다. 특히 ReAct와 같은 프레임워크는 LLM을 단순한 텍스트 생성기가 아닌 '에이전트'로 진화시키는 핵심 기제로 작용한다.

**한계 및 논의사항**:

- **치명적 망각(Catastrophic Forgetting)**: 단일 태스크 미세 조정 시 다른 일반 능력이 저하되는 문제가 여전히 존재하며, 이를 위해 Multitask Fine-tuning이나 PEFT의 도입이 필수적이다.
- **보상 해킹(Reward Hacking)**: RLHF 과정에서 모델이 실제 품질 개선 없이 보상 모델의 허점만을 이용해 높은 점수를 얻으려는 경향이 있으며, 이는 ReST와 같은 오프라인 학습 방식이 대안이 될 수 있다.
- **자원 제약**: 1-bit LLM과 같은 양자화 기술이 발전하고 있으나, 여전히 최첨단 모델을 학습시키기 위해서는 FSDP와 같은 고도의 분산 학습 인프라가 전제되어야 한다.

## 📌 TL;DR

본 논문은 LLM의 3대 난제(지식 단절, 계산 오류, 환각)를 해결하기 위한 최신 기술들을 집대성한 가이드라인이다. 핵심적으로는 **RAG, PAL, ReAct**를 통한 외부 도구 통합, **LoRA 및 Prompt Tuning**을 통한 효율적 적응, 그리고 **RLHF와 ReST**를 통한 인간 선호도 정렬 기법을 다룬다. 이 연구는 향후 LLM이 단순한 챗봇을 넘어, 정확한 계산과 최신 지식을 갖추고 스스로 계획을 세워 행동하는 '신뢰 가능한 AI 에이전트'로 발전하는 데 필요한 기술적 로드맵을 제공한다.
