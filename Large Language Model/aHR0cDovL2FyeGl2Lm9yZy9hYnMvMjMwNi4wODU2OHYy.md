# WizardCoder: EMPOWERING CODE LARGE LANGUAGE MODELS WITH EVOL-INSTRUCT

Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, Daxin Jiang

## 🧩 Problem to Solve

코드 대규모 언어 모델(Code LLMs)은 다양한 코드 관련 작업에서 인상적인 성능을 보여왔지만, 일반 언어 모델 분야와 달리 명령어 파인튜닝(instruction fine-tuning) 기술은 코드 도메인에서 상대적으로 덜 연구되었습니다. 기존의 오픈소스 Code LLMs는 최첨단 비공개(closed-source) 모델에 비해 성능이 뒤처지며, Code LLMs의 내재된 코딩 능력을 최대한 활용하기 위해 코드 명령어 데이터의 복잡성을 자동으로 증가시키는 방법이 필요합니다.

## ✨ Key Contributions

* `Code Evol-Instruct`라는 새로운 코드 명령어 파인튜닝 접근 방식을 도입하여 오픈소스 Code LLMs의 성능을 크게 향상시켰습니다.
* `WizardCoder` 모델을 개발하여 기존의 모든 오픈소스 Code LLMs를 압도적인 성능으로 능가함을 입증했습니다.
  * `WizardCoder 15B`는 Anthropic의 Claude, Google의 Bard와 같은 잘 알려진 비공개 LLM보다 뛰어난 성능을 보였습니다.
  * `WizardCoder 34B`는 HumanEval 벤치마크에서 GPT3.5 (ChatGPT)와 비견할 만한 점수를 달성했으며, HumanEval+ 벤치마크에서는 이를 능가했습니다.
* 명령어 복잡성이 뛰어난 코딩 성능을 달성하는 데 핵심적인 역할을 한다는 점을 강조하는 예비 연구를 수행했습니다.

## 📎 Related Works

* **대규모 언어 모델(LLMs):** OpenAI의 GPT-3/4, Google의 PaLM/Bard, DeepMind의 Chinchilla/Gopher, Anthropic의 Claude 등 비공개 LLM들과 GPT-NeoX-20B, LLaMA1/2 등 오픈소스 LLM들을 언급합니다.
* **코드를 위한 대규모 언어 모델(Code LLMs):** OpenAI의 Codex, Google의 PaLM-Coder 등 비공개 모델과 Salesforce의 CodeGen, BigCode Project의 StarCoder, Meta의 CodeLlama 등 오픈소스 모델이 있으며, 오픈소스 모델은 비공개 모델에 비해 성능이 뒤처지는 경향이 있음을 지적합니다.
* **명령어 파인튜닝(Instruction Fine-Tuning):** 초기에는 T5, FLAN 등 다중 작업 학습을 통해 모델의 일반화 능력을 향상시키는 데 중점을 두었습니다. 이후 InstructGPT, ChatGPT와 같이 사람의 피드백이나 자기-명령(self-instruct) 방식을 통해 사용자 의도에 맞게 정렬하는 연구(Alpaca, Vicuna, WizardLM의 Evol-Instruct)가 진행되었습니다. 본 연구는 이 `Evol-Instruct` 방법을 코드 도메인에 맞게 조정합니다.

## 🛠️ Methodology

본 논문은 `Code Evol-Instruct`를 사용하여 `WizardCoder`를 훈련하는 방법론을 제시합니다.

1. **Code Evol-Instruct 개발:**
    * WizardLM의 `Evol-Instruct` 방법에서 영감을 받아 코드 도메인의 특성에 맞춰 명령어 복잡성을 자동으로 향상시킵니다.
    * 다음과 같은 새로운 기능을 포함합니다:
        * LeetCode와 같은 코딩 과제 플랫폼의 특징에 맞춰 과제 복잡성을 전략적으로 증가시키는 휴리스틱.
        * 적대적 샘플로서 오류 코드(erroneous code)를 도입하여 과제 복잡성을 높이는 방법.
        * 시간 및 공간 복잡성 요구사항을 강조하는 휴리스틱.
    * 명령어 진화(evolution)를 위한 5가지 휴리스틱 방법:
        * 기존 문제에 새로운 제약 조건 및 요구 사항 추가 (약 10단어).
        * 프로그래밍 작업의 일반적인 요구 사항을 덜 일반적이고 더 구체적인 것으로 대체.
        * 원래 문제가 몇 가지 논리적 단계로만 해결될 수 있다면 더 많은 추론 단계 추가.
        * 오류 코드 조각을 참고 자료로 제공하여 오도(misdirection) 증가.
        * 더 높은 시간 또는 공간 복잡성 요구 사항 제안 (자주 사용하지 않음).

2. **WizardCoder 훈련:**
    * StarCoder 15B 및 CodeLlama-34B-Python을 기본 모델로 활용합니다.
    * `Code Alpaca` 데이터셋(약 2만 개 샘플)에 `Code Evol-Instruct` 기술을 반복적으로 적용하여 진화된 데이터를 생성합니다.
    * 각 데이터 진화 라운드(round) 후에 진화된 데이터와 원본 데이터셋을 병합하여 Code LLMs를 파인튜닝합니다.
    * 외부 개발 세트(dev set)를 `Evol Stop` 제어 장치로 사용하여 성능이 하락하면 진화를 중단합니다.
    * 파인튜닝 설정: 배치 크기 $512$, 시퀀스 길이 $2048$, 파인튜닝 단계 $200$, 웜업 단계 $30$, 학습률 $2 \times 10^{-5}$, Cosine 학습률 스케줄러, fp16 혼합 정밀도.
    * `gpt3.5-turbo`를 사용하여 약 78k개의 진화된 샘플을 생성합니다.

## 📊 Results

* **HumanEval, HumanEval+, MBPP 벤치마크:**
  * `WizardCoder 34B`는 HumanEval+에서 GPT3.5 (ChatGPT)를 능가했습니다 (64.6% vs. 63.4%). HumanEval 및 HumanEval+ 벤치마크에서 전체 2위를 달성했습니다.
  * `WizardCoder 15B`는 Claude-Plus (59.8% vs. 53.0%) 및 Bard (59.8% vs. 44.5%)를 뛰어넘었습니다.
  * 모든 오픈소스 모델에 비해 HumanEval 및 MBPP에서 상당한 성능 우위를 보였습니다. (`WizardCoder 34B`: HumanEval $71.5\%$, MBPP $61.2\%$; `WizardCoder 15B`: HumanEval $57.3\%$, MBPP $51.8\%$).
* **MultiPL-E (8개 프로그래밍 언어) 벤치마크:**
  * Java, JavaScript, C++, PHP, R, Julia, Swift, Rust 등 8개 언어 모두에서 SOTA 오픈소스 Code LLMs보다 우수한 성능을 입증했습니다.
* **DS-1000 벤치마크:**
  * 데이터 과학 문제 해결에 있어 다른 모든 모델보다 뛰어난 성능을 보였습니다. 특히 `WizardCoder 15B`는 삽입(insertion) 모드에서 StarCoder를 크게 능가했습니다.

## 🧠 Insights & Discussion

* **진화 모델 및 라운드:** GPT-4를 진화 모델로 사용했을 때 GPT-3.5보다 성능 향상 폭이 컸지만, GPT-4의 원시 코딩 성능에 비례하지는 않았습니다. 오픈소스 CodeLlama-Instruct-34B도 효과적이었으며, 3라운드의 데이터 진화 이후 MBPP-400 개발 세트와 HumanEval에서 가장 높은 pass@1 점수를 달성했습니다.
* **복잡성 대 양(Quantity):** 성능 향상은 단순히 샘플 수나 토큰 수의 증가가 아니라 `Code Evol-Instruct` 방법이 도입하는 더 복잡한 데이터 덕분임을 보여주었습니다. 동일한 샘플/토큰 수에서도 진화된 데이터로 훈련된 모델이 초기 데이터로만 훈련된 모델보다 일관되게 우수한 성능을 보였습니다.
* **복잡성 대 유사성(Similarity):** 진화 과정이 테스트 세트와의 유사성을 높이지 않으며, 모든 라운드에서 유사성 점수가 상대적으로 낮게 유지됨을 확인했습니다. 이는 성능 향상이 데이터 유출(leakage)이나 테스트 예제와의 유사성 때문이 아니라 명령어 `복잡성` 증가에 기인한다는 점을 뒷받침합니다.
* **한계:** `WizardCoder` 모델은 여전히 GPT4에 비해 상당한 격차를 보이고 있으며, 향후 연구에서 이 격차를 줄이는 것을 목표로 합니다.

## 📌 TL;DR

* **문제:** 코드 LLM 분야에서 명령어 튜닝 연구가 부족하고, 오픈소스 모델은 비공개 모델에 비해 성능이 뒤처지는 문제를 해결하고자 함.
* **제안 방법:** `Code Evol-Instruct`는 코드 도메인에 특화된 휴리스틱(예: 오류 코드 삽입, 시간/공간 복잡성 요구사항)을 사용하여 기존 코드 명령어 데이터의 복잡성을 자동으로 증가시키는 새로운 명령어 파인튜닝 접근 방식. 이 방법을 통해 `StarCoder` 및 `CodeLlama`를 기반으로 `WizardCoder` 모델을 훈련함.
* **주요 결과:** `WizardCoder`는 HumanEval, HumanEval+, MBPP, DS-1000, MultiPL-E 등 5가지 주요 코드 생성 벤치마크에서 모든 다른 오픈소스 Code LLM을 능가하는 SOTA 성능을 달성. 특히 `WizardCoder 15B`는 Claude 및 Bard를 능가하고, `WizardCoder 34B`는 HumanEval+ 벤치마크에서 GPT3.5 (ChatGPT)를 능가함. 명령어 복잡성 증가가 뛰어난 코딩 성능에 핵심적인 역할을 함을 분석을 통해 입증.
