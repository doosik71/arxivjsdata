# GPT-3 및 GPT-3.5 시리즈 모델의 종합적인 능력 분석

Junjie Ye, Xuanting Chen, Nuo Xu, Can Zu, Zekai Shao, Shichun Liu, Yuhan Cui, Zeyang Zhou, Chao Gong, Yang Shen, Jie Zhou, Siming Chen, Tao Gui, Qi Zhang, Xuanjing Huang

## 🧩 Problem to Solve

GPT 시리즈 모델들이 뛰어난 자연어 처리(NLP) 능력을 보여주었음에도 불구하고, 모델의 진화 과정(시간 경과)에 따른 능력 변화에 대한 종합적인 분석은 부족했습니다. 특히, GPT 시리즈 모델 훈련에 사용된 다양한 전략(예: RLHF)이 자연어 이해(NLU) 태스크 수행 능력에 어떤 영향을 미치는지에 대한 심층적인 조사가 필요합니다.

## ✨ Key Contributions

* **종합적인 능력 분석**: GPT-3 및 GPT-3.5 시리즈의 대표적인 6개 모델(davinci, text-davinci-001, code-davinci-002, text-davinci-002, text-davinci-003, gpt-3.5-turbo)의 NLU 능력을 9가지 태스크와 21개 데이터셋에 걸쳐 포괄적으로 평가했습니다.
* **훈련 전략의 영향**: 모델의 전반적인 NLU 능력이 모델 진화에 따라 점진적으로 증가하지 않으며, 특히 RLHF(인간 피드백 기반 강화 학습) 전략 도입이 인간과 유사한 응답 생성 능력을 향상시키지만 일부 NLU 태스크 해결 능력은 저해할 수 있음을 밝혔습니다.
* **모델 견고성 분석**: 모델의 성능 향상에도 불구하고, 견고성 측면에서는 큰 개선이 없었으며, 여전히 개선의 여지가 많음을 지적했습니다.
* **Davinci 모델의 명령어 이해**: davinci 모델은 zero-shot 시나리오에서 명령어 이해 능력이 부족하지만(프롬프트의 특정 키워드에 민감), few-shot 학습을 통해 프롬프트 이해 능력을 크게 향상시킬 수 있음을 보여주었습니다.
* **프롬프트 민감성**: 모든 모델이 zero-shot 및 few-shot 시나리오에서 프롬프트에 민감한 경향을 보였습니다.
* **Few-shot 학습의 한계**: few-shot 시나리오가 항상 모델 성능을 향상시키는 것은 아니며, 모델, 태스크, 프롬프트 디자인 및 예시 선택에 따라 결과가 달라짐을 확인했습니다.

## 📎 Related Works

* **대규모 언어 모델 (LLMs)**: FLAN (Wei et al., 2022), OPT (Zhang et al., 2022b), PaLM (Chowdhery et al., 2022) 등.
* **GPT 시리즈**: Generative Pre-trained Transformer (GPT) (Brown et al., 2020) 모델.
* **GPT 모델의 특정 능력 분석**:
  * GPT-3의 언어학적 지식 및 의미 정보 인식 (Zhang et al., 2022a).
  * ChatGPT (gpt-3.5-turbo)의 측면 기반 텍스트 요약 (Yang et al., 2023) 및 기계 번역 (Hendy et al., 2023) 능력.
  * ChatGPT의 zero-shot 능력 (Qin et al., 2023).
* **GPT 모델의 한계 및 견고성**:
  * ChatGPT의 성능 비교 및 단점 (Koco’n et al., 2023).
  * GPT 시리즈 모델의 NLU 태스크 견고성 테스트 (Chen et al., 2023).
* **GPT 훈련 전략**: InstructGPT (Ouyang et al., 2022), Codex (Chen et al., 2021), RLHF (Christiano et al., 2017).
* **평가 도구**: TextFlint (Gui et al., 2021) 다국어 NLP 견고성 평가 툴킷.

## 🛠️ Methodology

* **평가 대상 모델**:
  * **GPT-3 시리즈**: `davinci`, `text-davinci-001`
  * **GPT-3.5 시리즈**: `code-davinci-002`, `text-davinci-002`, `text-davinci-003`, `gpt-3.5-turbo`
* **평가 태스크 (9가지 NLU 태스크)**:
  * Aspect-based Sentiment Analysis (ABSA)
  * Machine Reading Comprehension (MRC)
  * Named Entity Recognition (NER)
  * Natural Language Inference (NLI)
  * Part-of-speech Tagging (POS)
  * Relation Extraction (RE)
  * Sentiment Classification (SC)
  * Semantic Matching (SM)
  * The Winograd Schema Challenge (WSC)
* **데이터셋**: 21개 데이터셋과 TextFlint로 생성된 변환 데이터를 사용하여 성능 및 견고성 평가.
* **평가 시나리오**:
  * **Zero-shot**: 어떠한 예시도 없이 태스크 지시만으로 평가.
  * **Few-shot (1-shot, 3-shot)**: 몇 가지 레이블링된 예시를 프롬프트에 포함하여 평가.
* **프롬프트 설계**: GitHub "promptsource"에서 태스크별 프롬프트를 수집하고 수동으로 새로운 프롬프트를 설계. 가장 성능이 좋은 3가지 프롬프트를 선정하여 사용. RE, NER, POS 태스크의 경우 원본 레이블을 특정 구문으로 매핑하여 모델의 이해를 돕도록 함.
* **API 사용**: OpenAI의 공식 API를 통해 모델 평가를 진행. 일부 모델(davinci, code-davinci-002, text-davinci-001, text-davinci-002)은 API 접근 제한으로 인해 샘플 데이터셋으로 테스트.

## 📊 Results

* **Davinci 모델의 명령어 이해**: `davinci` 모델은 zero-shot에서 "Answer"와 같은 명시적인 키워드가 없으면 답변 생성에 어려움을 겪어 명령어 이해 부족을 나타냈습니다. 하지만 few-shot 시나리오에서는 인컨텍스트 학습을 통해 NER, POS 등 복잡한 태스크의 명령어 이해도가 크게 향상되었습니다.
* **모델별 성능 차이**:
  * **Zero-shot 시나리오**:
    * ABSA, MRC, SC 태스크에서는 `code-davinci-002`가 가장 좋은 성능을 보였습니다.
    * POS, RE, SM 태스크에서는 `text-davinci-003`이 가장 우수했습니다.
    * NLI, WSC 태스크에서는 `gpt-3.5-turbo`가 뛰어났으나, POS 태스크에서는 명령어 준수에 어려움을 겪어 `text-davinci-001`과 유사한 결과를 보였습니다. 이는 `gpt-3.5-turbo`의 모델 크기와 대화 지향적인 훈련 때문일 수 있습니다.
  * **Few-shot 시나리오**:
    * 일반적으로 zero-shot보다 성능이 향상되지만, 항상 그런 것은 아니며, 모델, 태스크, 프롬프트 디자인에 따라 다르게 나타났습니다.
    * SC 태스크에서는 오히려 few-shot에서 성능이 저하되는 경우가 있었는데, 이는 긴 입력 텍스트가 모델의 문맥 판단에 영향을 미쳤을 가능성이 있습니다.
    * WSC 태스크에서는 zero-shot에서 1-shot으로 전환 시 성능이 하락하는 이례적인 결과도 관찰되었습니다.
  * **특정 모델의 경향**:
    * `text-davinci-001`은 `davinci`를 제외하고 대부분의 태스크에서 가장 약한 전반적인 능력을 보였습니다.
    * `gpt-3.5-turbo`와 `text-davinci-003`은 대부분의 태스크에서 비슷한 성능을 보였으나, MRC, POS, RE 태스크에서는 `gpt-3.5-turbo`가 약간 불리했습니다.
* **견고성**:
  * ABSA 태스크를 제외하고는 모델 간 견고성 차이가 상대적으로 미미했습니다.
  * GPT 시리즈 모델의 세대별 업데이트에도 불구하고 MRC, NER, NLI 태스크의 견고성은 크게 개선되지 않았습니다. 특히 `gpt-3.5-turbo`는 NLI 태스크의 일부 변형에서 이전 모델보다 견고성이 떨어지는 경우도 있었습니다.
  * SM 태스크에서는 few-shot 시나리오에서 성능과 견고성 모두 크게 개선되었습니다.

## 🧠 Insights & Discussion

* **사전 훈련의 중요성**: `davinci` 모델의 사례를 통해 사전 훈련이 모델에게 기본적인 이해 능력과 인컨텍스트 학습 능력을 부여함을 확인했습니다. 이는 복잡한 명령어 이해에 특히 중요합니다.
* **지도 미세 조정의 영향**: 지도 미세 조정 단계에 포함된 특정 태스크 유형이 모델의 해당 태스크 성능에 결정적인 영향을 미칠 수 있습니다. 그러나 OpenAI의 공식 문서에서 어떤 태스크가 미세 조정에 사용되었는지 명확하지 않아 추가 조사가 필요합니다.
* **인간 인지 정렬의 대가 ('Alignment Tax')**: InstructGPT 모델인 `text-davinci-002`는 `code-davinci-002`를 기반으로 하지만, 일부 태스크(SM, WSC)에서는 성능 우위를 보였으나 다른 태스크에서는 유사하거나 심지어 성능이 저하되는 현상이 나타났습니다. 이는 "alignment tax"로 알려진 현상으로, 인간 인지와의 정렬을 추구하는 과정에서 발생하는 일부 태스크 성능 저하를 의미합니다.
* **RLHF의 목적**: RLHF는 NLU 태스크의 성능을 직접적으로 개선하기보다는, 모델이 인간과 유사하고 유용한 응답을 생성하는 능력을 강화하는 데 초점을 맞춥니다. `text-davinci-003`은 RLHF를 통해 `text-davinci-002`를 개선했으나, NLU 태스크 성능은 비슷하거나 일부에서는 오히려 낮은 결과를 보여, RLHF가 태스크에 대한 깊은 이해보다는 응답의 품질에 더 기여함을 시사합니다.
* **모델 진화의 비선형성**: GPT 시리즈 모델의 진화는 모든 NLU 태스크에서 보편적인 개선으로 이어지지 않으며, 훈련 전략과 태스크 특성에 따라 상충 관계가 존재할 수 있습니다.
* **미래 연구 방향**: 모델의 태스크 해결 능력과 사용자 친화적인 응답 능력 사이의 균형을 찾는 것, 그리고 성능 향상과 동시에 모델의 견고성을 개선하는 방법에 대한 추가 연구가 필요합니다.
* **제한 사항**: OpenAI API 접근 제한으로 인해 일부 모델은 전체 데이터셋으로 테스트하지 못했으며, 연구 기간 중 출시된 GPT-4는 API 접근 불가로 평가에 포함되지 못했습니다.

## 📌 TL;DR

이 논문은 GPT-3 및 GPT-3.5 시리즈 모델(6개)의 NLU 능력을 9가지 태스크와 21개 데이터셋에 걸쳐 포괄적으로 분석했습니다. 핵심 발견은 모델의 진화가 모든 NLU 태스크에서 점진적인 성능 향상으로 이어지지 않으며, 특히 RLHF 훈련 전략이 인간 유사 응답 생성 능력은 강화하지만 일부 태스크 해결 능력은 저해할 수 있다는 것입니다. 또한, 모든 모델이 프롬프트에 민감하고, 견고성 측면에서는 여전히 개선이 필요함을 강조하며, 모델의 태스크 수행 능력과 사용자 친화적 응답 능력 간의 균형에 대한 통찰을 제공합니다.
