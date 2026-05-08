# Recent Trends in Unsupervised Summarization

Mohammad Khosravani, Amine Trabelsi (2024)

## 🧩 Problem to Solve

본 논문은 텍스트 요약(Text Summarization) 분야에서 발생하는 **레이블링 된 데이터셋(labeled datasets)에 대한 과도한 의존성** 문제를 해결하고자 한다. 텍스트 요약 모델을 학습시키기 위해서는 일반적으로 원문과 요약문이 쌍을 이루는 대규모 데이터가 필요하지만, 다음과 같은 이유로 이는 매우 비효율적이다.

- **데이터 구축 비용:** 뉴스 같은 특정 도메인을 제외하고, 다른 도메인이나 저자원 언어(low-resource languages)에 대한 레이블 데이터를 구축하는 것은 비용이 매우 많이 든다.
- **도메인의 가변성:** 언어는 계속 진화하며 새로운 도메인이 지속적으로 등장하므로, 모든 도메인에 대해 최신 데이터셋을 유지하는 것은 사실상 불가능하다.

따라서 본 논문의 목표는 레이블 없는 데이터로부터 학습이 가능한 **비지도 학습 기반 요약(Unsupervised Summarization)** 기술의 최신 동향을 분석하고, 이를 체계적으로 분류할 수 있는 세밀한 **분류 체계(Taxonomy)**를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 비지도 학습 기반 요약 방법론을 단순한 전략 구분이 아닌, 실제 구현 기법과 모델 구조에 따라 세분화하여 분류한 점이다.

- **세분화된 분류 체계(Fine-grained Taxonomy) 제안:** 기존의 광범위한 분류(추출적 vs 생성적)를 넘어, 학습 전략과 모델 아키텍처를 기준으로 한 계층적 구조를 제시하였다.
- **최신 연구 동향 분석:** Pretrained Language Models(PLMs)와 Large Language Models(LLMs)가 비지도 요약에 미친 영향을 상세히 분석하였다.
- **비지도 학습 외 대안 탐색:** 레이블 부족 문제를 함께 해결하는 weakly-supervised, self-supervised, zero-shot, few-shot 방법론을 함께 다루어 분석의 범위를 확장하였다.

## 📎 Related Works

기존의 요약 연구들은 주로 다음과 같은 기준으로 분류되어 왔다.

- **입력 문서의 수:** 단일 문서 요약(Single Document Summarization, SDS)과 다중 문서 요약(Multi-document Summarization, MDS).
- **요약의 초점:** 일반적 요약(Generic)과 특정 측면 중심 요약(Aspect-based).
- **요약 전략:** 추출적 요약(Extractive)과 생성적 요약(Abstractive).

그러나 저자들은 이러한 분류가 실제 사용된 아이디어와 전략의 세부 사항을 충분히 전달하지 못한다고 지적한다. 특히 비지도 학습 분야에 특화되어 방법론을 체계적으로 분류한 서베이 논문이 부족하다는 점을 들어 본 연구의 차별성을 강조한다.

## 🛠️ Methodology

본 논문은 새로운 알고리즘을 제안하는 것이 아니라, 기존 연구들을 분석하기 위한 **분류 체계(Taxonomy)**를 방법론으로 제시한다.

### 1. 생성적 방법론 (Abstractive Methods)

입력 텍스트를 이해하고 새로운 문장을 생성하는 방식으로, 크게 세 가지로 분류한다.

- **Language Model-based:** PLM(BART, T5, PEGASUS) 및 LLM(GPT-4, Llama-2)을 활용하는 방식이다. 최근에는 모델 가중치를 직접 수정하는 fine-tuning보다 **Prompt Engineering**을 통한 Zero-shot/Few-shot 학습에 집중하는 추세이다.
- **Reconstruction Networks:** 원본 문서에 노이즈를 추가한 뒤, 이를 다시 원본으로 복구하는 과정을 통해 요약 능력을 학습하는 방식이다. 주로 Auto-encoder 구조를 사용한다.
- **Other:** Non-autoregressive 구조나 Q-learning 기반의 Edit-based 접근법 등이 포함된다.

### 2. 추출적 방법론 (Extractive Methods)

원문에서 중요한 단위(단어, 문장)를 선택하여 조합하는 방식으로, 세 가지로 분류한다.

- **Selection:** 각 단위가 요약문에 포함될지 여부를 결정하는 이진 분류(Binary Classification) 문제로 접근한다.
- **Ranking:** 문장 간의 유사도나 중요도를 점수로 계산하여 상위 $k$개를 선택하는 방식이다. (예: TextRank의 발전 형태)
- **Search-based:** 목적 함수(Objective Function)를 정의하고, 단어를 추가/삭제/교체하며 요약문을 반복적으로 최적화하는 Hill-climbing search 방식이다.

### 3. 하이브리드 방법론 (Hybrid Methods)

추출적 방식과 생성적 방식의 장점을 결합한다.

- **Extract-then-Abstract:** 먼저 추출적 모델로 핵심 내용을 추린 후, 이를 생성적 모델의 입력으로 넣어 최종 요약문을 만든다. 이는 긴 문서 처리 시 효율적이다.
- **Joint Strategies:** 두 방식 모두에 적용 가능한 학습 전략을 설계하거나, 두 모델을 동시에 사용하여 상호 보완하는 방식이다.

## 📊 Results

본 논문은 서베이 논문이므로 직접적인 실험 결과보다는 기존 연구들의 결과와 지표를 정리하여 제시한다.

- **평가 지표:** 주로 **ROUGE** (R1, R2, RL) F1 스코어가 표준으로 사용되며, 최근에는 BERTScore, BLEURT, BARTScore와 같은 PLM 기반 지표가 도입되고 있다.
- **데이터셋:** 뉴스 도메인의 CNN/DailyMail, XSum, Multi-News와 리뷰 도메인의 Yelp, Amazon 등이 주로 사용된다.
- **정량적 경향:**
  - LLM(GPT-4 등)은 Zero-shot 설정에서도 PLM(BART, T5)보다 우수하거나 대등한 성능을 보이며, 특히 새로운 기준 요약문(Reference summaries)을 적용했을 때 더 높은 평가를 받는 경향이 있다.
  - 추출적 방법론은 학습 및 추론 속도가 빠르지만, 생성적 방법론에 비해 유창성(Fluency)과 응집도(Coherence)가 떨어진다.
  - 생성적 방법론은 성능은 높으나 **Hallucination(환각)** 및 **Factual Incorrectness(사실적 부정확성)** 문제가 지속적으로 발생한다.

## 🧠 Insights & Discussion

### 강점 및 기회

- **PLM/LLM의 잠재력:** 비지도 학습의 성능이 PLM의 발전과 궤를 같이한다. 특히 거대 모델의 등장으로 fine-tuning 없이도 수준 높은 요약이 가능해졌다.
- **데이터 효율성:** 레이블 없는 데이터는 구하기 쉽기 때문에, 대규모 비지도 데이터를 활용한 사전 학습은 향후 성능 향상의 핵심이 될 것이다.

### 한계 및 비판적 해석

- **평가 지표의 정체:** 모델은 급격히 발전했으나, 평가 지표는 여전히 n-gram 중첩 기반인 ROUGE에 의존하고 있다. 이는 의미적 유사성(Semantic Similarity)을 반영하지 못해 모델의 실제 성능을 왜곡할 수 있다.
- **LLM의 비용 문제:** GPT-4와 같은 모델은 뛰어난 성능을 보이지만, 추론 비용이 매우 높고 모델 가중치가 공개되지 않아 연구의 재현성과 접근성이 떨어진다.
- **사실성 문제:** 생성적 요약에서 발생하는 Hallucination은 실무 적용의 가장 큰 걸림돌이며, 이를 완전히 해결할 비지도 학습 전략은 아직 부족하다.

## 📌 TL;DR

본 논문은 레이블 데이터 없이 텍스트를 요약하는 **비지도 학습 기반 요약 기술의 최신 동향을 정리하고, 이를 [생성적 $\rightarrow$ 추출적 $\rightarrow$ 하이브리드] 및 세부 기법으로 분류한 taxonomy를 제안**한다. 최근 연구의 흐름이 추출적 방식에서 PLM/LLM 기반의 생성적 방식으로 완전히 이동했음을 보여주며, 특히 LLM의 Prompt Engineering이 핵심으로 부상했음을 분석한다. 이 연구는 향후 저자원 도메인 및 실시간 요약 시스템 구축을 위한 모델 선택의 가이드라인을 제공한다는 점에서 중요한 역할을 한다.
