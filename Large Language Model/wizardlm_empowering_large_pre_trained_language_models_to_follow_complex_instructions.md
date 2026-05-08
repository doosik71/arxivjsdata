# WizardLM: EMPOWERINGLARGEPRE-TRAINEDLAN- GUAGEMODELS TOFOLLOWCOMPLEXINSTRUCTIONS

Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, Qingwei Lin, Daxin Jiang

## 🧩 Problem to Solve

거대 언어 모델(LLM)은 다양한 NLP 작업에서 큰 성공을 거두었지만, 사용자가 지정하는 복잡한 지침이나 목표를 따르는 데 어려움을 겪는 경우가 많습니다. 오픈 도메인 지침 팔로잉(instruction-following) 데이터를 통해 LLM을 훈련하는 것이 효과적임이 입증되었으나, 이러한 데이터를 수동으로 생성하는 것은 엄청난 시간과 노동력을 요구하며, 인간은 높은 복잡도의 지침을 생성하는 데 한계가 있습니다. 따라서, 다양한 난이도의 오픈 도메인 지침 데이터, 특히 더 어려운 지침들을 저비용으로 대량 생산할 수 있는 자동화된 방법론의 개발이 필요합니다.

## ✨ Key Contributions

* **Evol-Instruct 도입:** LLM 자체를 활용하여 다양한 주제와 난이도를 가진 오픈 도메인 지침을 자동으로 대량 생산함으로써 오픈 소스 LLM의 성능을 크게 향상시키는 새로운 접근 방식인 `Evol-Instruct`를 제안했습니다.
* **WizardLM 개발:** `Evol-Instruct`로 생성된 데이터를 사용하여 LLaMA를 미세 조정하여 `WizardLM` 모델을 개발했습니다. `WizardLM`은 Alpaca 및 Vicuna와 같은 대표적인 오픈 소스 LLM을 코드, 수학, GPT-4 및 인간 평가를 포함한 일련의 벤치마크에서 크게 능가합니다.
* **지침 복잡성의 중요성 강조:** 대규모 사전 학습 언어 모델의 지도 미세 조정(supervised fine-tuning)에서 지침의 복잡성이 뛰어난 성능을 달성하는 데 중요하다는 점을 심층적으로 조사하고 입증했습니다.

## 📎 Related Works

* **폐쇄형 도메인(Closed-domain) 지침 튜닝:** T5 (Raffel et al., 2020), FLAN (Wei et al., 2021), ExT5 (Aribandi et al., 2022), T0 (Sanh et al., 2022), FLAN-T5 (Chung et al., 2022) 등 초기 연구들은 다양한 NLP 작업 데이터셋에 소량의 수작업 지침을 사용하여 모델을 튜닝했습니다. 이는 특정 작업에 대한 성능을 향상시키지만, 지침의 다양성과 복잡성이 제한적이었습니다.
* **개방형 도메인(Open-domain) 지침 튜닝:** InstructGPT (Ouyang et al., 2022) 및 ChatGPT는 인간이 생성한 오픈 도메인 지침 데이터를 사용하여 큰 성공을 거두었습니다. 이후 Alpaca (Taori et al., 2023)는 Self-Instruct (Wang et al., 2022a) 방법을 통해 소수의 시드(seed) 지침으로 50k 데이터를 생성하여 LLaMA를 튜닝했습니다. Vicuna (Chiang et al., 2023)는 ShareGPT의 70k 사용자 공유 대화를 기반으로 LLaMA를 미세 조정했습니다. 본 연구는 InstructGPT 및 Vicuna와 달리 AI 생성 데이터를 사용하며, Alpaca의 Self-Instruct와 달리 생성된 지침의 난이도와 복잡성 수준을 제어할 수 있다는 점에서 차별화됩니다.

## 🛠️ Methodology

본 연구에서는 LLM을 사용하여 다양한 난이도의 지침 데이터를 자동으로 생성하는 `Evol-Instruct` 방법론을 제안합니다. 파이프라인은 크게 두 가지 구성 요소로 이루어집니다: Instruction Evolver (지침 진화기) 및 Instruction Eliminator (지침 제거기).

1. **지침 데이터 진화 정의:**
    * 초기 지침 데이터셋 $D^{(0)}$ (Alpaca 데이터셋의 52k 샘플)에서 시작합니다.
    * 각 진화 단계($t$)에서, $D^{(t)}$ 내의 지침 $I^{(t)}$을 `Evol-Instruct` 프롬프트를 사용하여 LLM을 통해 $I^{(t+1)}$으로 업그레이드하고, 이에 대한 응답 $R^{(t+1)}$을 생성하여 진화된 데이터셋 $D^{(t+1)}$을 얻습니다.
    * 이 과정을 $M$번 반복하여 $[D^{(1)} \cdots D^{(M)}]$ 진화 데이터셋을 순차적으로 얻습니다.

2. **자동 지침 데이터 진화:**
    * **Instruction Evolver:** LLM (OpenAI ChatGPT API 사용)을 활용하여 지침을 진화시킵니다.
        * **In-Depth Evolving (심층 진화):** 지침을 더 복잡하고 어렵게 만듭니다. 5가지 유형의 프롬프트가 사용됩니다.
            * `add constraints` (제약 추가): 지침에 새로운 제약/요구 사항을 추가합니다.
            * `deepening` (심화): 질문의 깊이와 폭을 증가시킵니다.
            * `concretizing` (구체화): 일반적인 개념을 더 구체적인 개념으로 대체합니다.
            * `increased reasoning steps` (추론 단계 증가): 간단한 추론으로 해결 가능한 지침을 다단계 추론을 명시적으로 요구하도록 변경합니다.
            * `complicate input` (입력 복잡화): XML, SQL, Python 코드, HTML 페이지, Shell 명령어, JSON 데이터 등 다양한 형식의 입력 데이터를 포함시켜 지침을 복잡하게 만듭니다. (In-context learning 사용)
        * **In-Breadth Evolving (폭넓은 진화):** 주제 범위와 스킬 다양성을 확장하기 위해, 주어진 지침에서 영감을 받아 동일한 도메인에 속하지만 더 희귀한(long-tailed) 새로운 지침을 생성합니다.
    * **Response Generation:** 진화된 각 지침에 대해 동일한 LLM(ChatGPT-3.5)을 사용하여 응답을 생성합니다.
    * **Elimination Evolving (제거 진화):** 다음 4가지 상황에 해당하는 진화 실패 지침을 필터링합니다.
        1. 원래 지침과 비교하여 정보 이득이 없는 경우.
        2. LLM이 응답 생성에 어려움을 겪는 경우 (예: "sorry" 포함, 80단어 미만).
        3. LLM이 생성한 응답이 구두점이나 불용어만 포함하는 경우.
        4. 진화된 지침이 프롬프트에서 단어를 명백히 복사한 경우.

3. **진화된 지침을 사용한 LLM 미세 조정:**
    * 모든 진화가 완료되면, 초기 데이터셋과 모든 epoch의 진화된 지침 데이터를 병합하고 무작위로 셔플하여 미세 조정 데이터셋을 생성합니다.
    * 공정한 비교를 위해 250k 지침 중 Vicuna와 동일한 70k 데이터를 샘플링하여 최종 훈련 데이터로 사용합니다.
    * LLaMA 13B 모델을 Vicuna의 프롬프트 형식에 따라 미세 조정하여 `WizardLM`을 훈련합니다.

## 📊 Results

* **자동 평가:** MMLU, ARC, HellaSwag, TruthfulQA, HumanEval (코드), GSM8k (수학), AlpacaEval, MT-Bench, 그리고 본 연구에서 새로 생성한 WizardEval 등 9개의 LLM 벤치마크에서 `WizardLM`의 성능을 평가했습니다.
  * `WizardLM`은 Alpaca, Vicuna, Baize, CAMEL, Tulu 등 동급 오픈 소스 모델보다 대부분의 벤치마크에서 뛰어난 성능을 보였습니다.
  * 특히 수학 (GSM8k), 코드 (HumanEval) 및 GPT-4 기반 평가 (AlpacaEval, MT-Bench, WizardEval)에서 상당한 개선을 이루었습니다. 예를 들어, GSM8k에서 Alpaca 13B는 9.2, Vicuna 13B는 12.5에 불과했지만, WizardLM 13B는 24.03을 기록했습니다.
* **인간 평가:** `WizardLM`과 Alpaca, Vicuna, ChatGPT-3.5를 대상으로 218개의 실제 인간 지침으로 구성된 `WizardEval` 데이터셋에서 블라인드 쌍대 비교(blind pairwise comparison)를 수행했습니다.
  * `WizardLM`은 Alpaca 및 Vicuna보다 현저히 우수한 결과를 달성하여 `Evol-Instruct` 방법의 효과를 입증했습니다. 평가자 간의 Kappa 점수는 0.6 이상으로 높은 일치도를 보였습니다.
* **Ablation Study:**
  * **데이터 시드, 크기, 진화 모델, 기반 모델 크기:**
    * Alpaca보다 ShareGPT를 시드 데이터로 사용했을 때 `WizardLM` 성능이 더 좋았습니다.
    * 더 큰 진화 데이터셋(250k)은 모델 역량을 향상시켰습니다.
    * `Evol-Instruct`는 ChatGPT뿐만 아니라 Llama-2-70B-Chat과 같은 다른 강력한 오픈 소스 모델로도 효과적으로 작동함을 보여, 특정 LLM에 의존적이지 않음을 입증했습니다.
    * Supernatural Instructions보다 본 연구의 진화 데이터가 더 나은 미세 조정 성능을 보였습니다.
    * 다양한 사전 학습 모델(Llama-1 65B, Llama-2 70B, Mistral-7B)에서도 `Evol-Instruct`가 적용 가능함을 확인했습니다.
  * **In-depth Evolving 분석:** 진화 라운드가 증가하고 훈련 지침의 복잡성이 점진적으로 증가함에 따라, 미세 조정된 모델의 성능 또한 비례적으로 향상됨을 확인했습니다. ChatGPT, GPT-4, 인간 평가 모두에서 난이도 점수 변화의 높은 일치도를 보였습니다.
  * **In-breadth Evolving 분석:** t-SNE와 k-means 클러스터링을 통해 `Evol-Instruct`로 생성된 지침들이 ShareGPT 및 Alpaca보다 더 넓은 주제 다양성을 가짐을 정성적으로 입증했습니다.

## 🧠 Insights & Discussion

`Evol-Instruct`는 LLM이 인간의 개입을 최소화하면서 복잡하고 다양한 지침 데이터를 자동으로 생성할 수 있는 효과적인 방법을 제시합니다. 이 연구는 지침의 난이도와 복잡성을 체계적으로 증가시키는 것이 LLM의 성능, 특히 추론, 코드 생성 및 수학적 능력 향상에 매우 중요함을 강조합니다. `WizardLM`은 이러한 지침 진화 전략의 성공적인 적용 사례를 보여주며, 기존의 인간 생성 데이터셋에 의존하는 방식의 한계를 극복하고 LLM의 잠재력을 더욱 확장할 수 있는 새로운 길을 열었습니다.

**제한 사항:** 본 연구는 자동 GPT-4 평가와 인간 평가 방법론의 확장성 및 신뢰성 측면에서의 한계를 인정합니다. 또한, 사용된 테스트 세트가 LLM이 적용될 수 있는 모든 시나리오나 도메인을 대표하지 못할 수 있습니다.

## 📌 TL;DR

**문제:** 복잡하고 다양한 오픈 도메인 지침 데이터를 수동으로 생성하는 것은 비싸고 비효율적이다.
**제안 방법:** `Evol-Instruct`는 LLM(ChatGPT)을 활용하여 기존 지침을 심층적(In-depth Evolving) 및 폭넓게(In-breadth Evolving) 진화시켜 다양한 난이도와 주제의 지침 데이터를 자동으로 생성하는 진화 알고리즘이다.
**주요 발견:** 이 방식으로 LLaMA를 미세 조정하여 `WizardLM`을 개발했으며, 이는 Alpaca, Vicuna 등 기존 모델들을 코드, 수학, GPT-4, 인간 평가 등 광범위한 벤치마크에서 크게 능가함을 입증했다. 이는 LLM의 성능 향상에 있어 지침 복잡성의 중요성을 강조한다.
