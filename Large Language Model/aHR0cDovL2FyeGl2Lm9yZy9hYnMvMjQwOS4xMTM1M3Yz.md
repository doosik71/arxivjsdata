# THaMES: An End-to-End Tool for Hallucination Mitigation and Evaluation in Large Language Models

Mengfei Liang, Archish Arun, Zekun Wu, Cristian Munoz, Jonathan Lutch, Emre Kazim, Adriano Koshiyama, Philip Treleaven (2024)

## 🧩 Problem to Solve

본 논문은 대규모 언어 모델(Large Language Models, LLMs)에서 발생하는 핵심 문제인 Hallucination(환각 현상), 즉 사실적으로 틀렸거나 근거 없는 내용을 생성하는 현상을 해결하고자 한다. 기존의 Hallucination 탐지 및 완화 방법들은 대부분 개별적으로 존재하며, 특정 도메인의 특수한 요구사항을 충족하기에는 불충분한 경우가 많았다.

특히 기존의 벤치마크 데이터셋(예: HaluEval, DelucionQA)은 구축 과정에서 많은 인적 자원(human annotation)이 소모되며, 질문의 복잡성이나 다양성이 부족하다는 한계가 있다. 또한, 많은 프레임워크가 '탐지(Identification)' 또는 '생성(Generation)' 중 하나의 기준에만 집중하는 경향이 있어, 모델의 종합적인 강건성을 평가하기 어렵다. 따라서 본 연구의 목표는 도메인 맞춤형 데이터셋 생성, 다각적인 벤치마킹, 그리고 유연한 완화 전략을 하나의 통합된 파이프라인으로 결합한 end-to-end 도구인 THaMES를 제안하는 것이다.

## ✨ Key Contributions

THaMES의 핵심 아이디어는 데이터 생성부터 평가, 완화에 이르는 전체 과정을 자동화하여, 특정 도메인의 지식 베이스(Knowledge Base)를 기반으로 모델의 환각 능력을 정밀하게 진단하고 최적의 해결책을 찾아내는 것이다.

주요 기여 사항은 다음과 같다.

1. **자동화된 테스트셋 생성**: Weighted Sampling과 배치 처리(Batch Processing)를 통해 비용 효율적이면서도 다양성과 품질이 높은 QA 데이터셋을 자동으로 생성한다.
2. **다양한 질문 유형 도입**: 단순 질문 외에도 Reasoning, Multi-context, Situational, Distracting, Double 등 6가지 복잡한 질문 유형을 설계하여 모델의 한계를 정밀하게 테스트한다.
3. **다각적 평가 지표**: Hallucination Identification(식별)과 Generation(생성)이라는 두 가지 기준을 모두 적용하여 모델의 성능을 종합적으로 평가한다.
4. **맞춤형 완화 전략 제공**: In-Context Learning (ICL), Retrieval Augmented Generation (RAG), Parameter-Efficient Fine-tuning (PEFT) 등 다양한 전략을 제공하고, 모델과 지식 베이스에 따라 최적의 전략을 선택할 수 있게 한다.

## 📎 Related Works

논문은 Hallucination 연구를 Factuality(사실성)와 Faithfulness(충실성) 두 가지 유형으로 구분하며, 기존의 벤치마크와 완화 기법들을 언급한다.

- **기존 벤치마크의 한계**: HaluEval 및 DelucionQA와 같은 기존 데이터셋은 QA 작업을 지원하지만, 수동 주석 작업에 크게 의존하며 질문의 다양성이 떨어진다.
- **기존 완화 기법**: RAG, CoT(Chain-of-Thought), CoVe(Chain-of-Verification) 등이 제안되었으나, 이들은 특정 측면에만 집중하며 전체적인 파이프라인을 제공하지 않는다.
- **차별점**: THaMES는 단순히 평가 도구에 그치지 않고, 사용자가 제공한 임의의 말뭉치(Corpus)로부터 테스트셋을 생성하고, 이를 통해 모델을 평가한 뒤, 최적의 완화 전략을 적용하는 통합 솔루션을 제공한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

THaMES 프레임워크는 크게 세 가지 구성 요소로 이루어져 있다.

### 1. Hallucination Benchmark Testset Generation

사용자가 제공한 말뭉치로부터 synthetic QA pair(질문, 정답, 환각 답변)를 생성하는 과정이다.

- **지식 베이스 구축 및 샘플링**: `LlamaIndex`를 사용하여 텍스트 블록(nodes)을 관리하며, 데이터의 다양성을 확보하기 위해 Weighted Random Sampling을 사용한다. 각 노드 $i$의 샘플링 확률 $p_i$는 다음과 같이 계산된다.
$$p_i = \frac{w_i}{\sum_{j=1}^{n} w_j}, \quad \text{where } w_i = \frac{1}{c_i + 1}$$
여기서 $c_i$는 해당 노드의 빈도수 등을 나타내며, 이를 통해 특정 데이터에 치우치지 않고 말뭉치 전체를 고르게 커버한다.

- **질문 생성 및 진화**: GPT-4o-mini를 사용하여 6가지 유형(Simple, Reasoning, Multi-context, Situational, Distracting, Double)의 질문을 생성한다. 단순 질문을 먼저 생성한 후, 필터링 과정을 거쳐 더 복잡한 유형으로 진화(Evolution)시키는 방식을 취한다.

- **환각 답변(Hallucinated Answer) 생성**: 단순히 다른 모델이 생성한 답을 쓰는 것이 아니라, NLI 모델(deberta-v3-base-tasksource-nli)과 환각 평가 모델(HHEM-2.1-Open)을 사용하여 **Ensemble Score**를 계산한다.
$$\text{Ensemble Score} = \text{Entailment Score} + \text{Factual Consistency Score}$$
점수가 낮을수록 환각 정도가 심한 답변으로 판단하며, 이를 통해 가장 혼란을 줄 수 있는(distracting) 환각 답변을 선택한다.

### 2. Hallucination Evaluation

모델의 성능을 두 가지 관점에서 평가한다.

- **Hallucination Generation (생성 능력)**: 모델이 생성한 답변을 정답과 비교하여 평가한다. RAGAS 지표를 기반으로 Faithfulness, Relevancy, Similarity, Correctness를 측정한다. 특히 Answer Relevancy는 다음과 같이 계산된다.
$$\text{answer relevancy} = \frac{1}{N} \sum_{i=1}^{N} \frac{E_{g_i} \cdot E_o}{\|E_{g_i}\| \|E_o\|}$$
여기서 $E_{g_i}$는 생성된 질문의 임베딩, $E_o$는 원본 질문의 임베딩이다.

- **Hallucination Identification (식별 능력)**: 모델에게 정답과 환각 답변 중 하나를 무작위로 제시하고, 어떤 것이 정답인지 맞추는 능력을 Accuracy로 측정한다.
$$\text{Accuracy} = \frac{\# \text{Correct Predictions}}{\# \text{Total Predictions}}$$

### 3. Hallucination Mitigation

세 가지 주요 완화 전략을 적용한다.

- **In-Context Learning (ICL)**: 특히 **Chain-of-Verification (CoVe)** 기법을 사용한다. 모델이 답변을 생성한 후, 스스로 검증 질문을 만들고 이에 답함으로써 초기 답변의 오류를 수정하는 단계적 추론 방식이다.
- **Retrieval-Augmented Generation (RAG)**: 외부 지식을 활용하여 답변의 근거를 확보한다. THaMES는 모델이 이전 평가에서 틀렸던 사례들을 수집하여 지식 베이스로 구축하고, 유사한 질문이 들어왔을 때 이를 few-shot 컨텍스트로 제공하여 동일한 실수를 방지한다.
- **Parameter-Efficient Fine-tuning (PEFT)**: **LoRA (Low-Rank Adaptation)**를 사용하여 모델을 미세 조정한다. baseline 평가에서 실패한 QA 쌍들을 학습 데이터로 사용하여 모델이 정답을 내도록 최적화한다.

## 📊 Results

실험은 GPT-4o, GPT-4o-mini, Llama-3.1-8B-Instruct, Mistral-Nemo 등 상용 및 오픈 웨이트 모델을 대상으로 수행되었다.

- **모델별 전략 효율성**:
  - **상용 모델 (GPT-4o 계열)**: ICL(CoVe)의 효과는 제한적이었으나, **RAG 전략을 통해 환각 억제 능력이 크게 향상**되었다. 이는 이미 모델의 기본 추론 능력이 높기 때문에 추가적인 외부 지식의 주입이 더 효과적임을 시사한다.
  - **오픈 웨이트 모델 (Llama-3.1, Mistral-Nemo)**: RAG 역시 도움이 되었으나, 특히 **ICL(CoVe)이 환각 식별(Identification) 정확도를 높이는 데 매우 효과적**이었다.

- **PEFT의 효과**: Llama-3.1-8B-Instruct 모델에 LoRA 미세 조정을 적용한 결과, 텍스트 생성 성능(Answer Relevancy, Correctness, Similarity)과 식별 성능(Recall, F1-score) 모두에서 baseline 모델보다 유의미한 성능 향상을 보였다.

- **질문 유형별 분석**: Reasoning 및 Multi-context 질문에서 모델들의 성능이 전반적으로 낮았으며, 특히 오픈 웨이트 모델들이 이러한 복잡한 질문에서 더 많은 환각을 생성하는 경향을 보였다.

## 🧠 Insights & Discussion

본 논문은 모든 모델에 적용 가능한 단일 최적 전략은 없으며, **모델의 규모와 특성, 그리고 해결하려는 과제(생성 vs 식별)에 따라 최적의 완화 전략이 다르다**는 점을 실증적으로 보여주었다.

**강점**:

- 데이터 생성 $\rightarrow$ 평가 $\rightarrow$ 완화로 이어지는 end-to-end 파이프라인을 구축하여 도메인 특화 환각 문제를 체계적으로 다루었다.
- Weighted Sampling과 Ensemble Score 기반의 환각 답변 선택을 통해 테스트셋의 품질과 다양성을 확보하였다.

**한계 및 비판적 해석**:

- **리소스 제약**: 계산 자원의 한계로 인해 오픈 소스 모델의 경우 양자화(quantized) 버전을 사용하였으며, PEFT 실험을 Llama-3.1 모델 하나에만 국한하여 진행한 점이 아쉽다.
- **데이터 생성 의존성**: 테스트셋 생성 과정에서 GPT-4o-mini에 크게 의존하고 있어, 생성된 데이터의 상한선이 해당 모델의 성능에 종속될 위험이 있다.
- **인적 검증 부재**: 자동화된 파이프라인의 효율성은 높으나, 최종 데이터셋에 대한 인간 전문가의 검증 단계가 부족하여 일부 데이터에 노이즈가 포함되었을 가능성이 있다.

## 📌 TL;DR

THaMES는 LLM의 환각 현상을 체계적으로 진단하고 완화하기 위한 end-to-end 프레임워크이다. 자동화된 고품질 테스트셋 생성, 다각적 벤치마킹, 그리고 ICL/RAG/PEFT라는 세 가지 완화 경로를 제공한다. 실험 결과, GPT-4o와 같은 고성능 모델은 RAG에, Llama-3.1과 같은 모델은 ICL 및 PEFT에 더 큰 혜택을 입음을 확인하였다. 이 연구는 향후 특정 도메인(의료, 법률 등)에서 LLM의 신뢰성을 확보하고 배포하기 위한 표준화된 평가 및 최적화 파이프라인으로 활용될 가능성이 매우 높다.
