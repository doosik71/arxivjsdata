# INSTRUCTIONTUNING WITHGPT-4

Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, Jianfeng Gao

## 🧩 Problem to Solve

거대 언어 모델(LLM)은 뛰어난 일반화 능력을 가지고 있지만, 실제 세계의 작업을 수행하기 위한 자연어 지시를 따르도록 만들기 위해서는 `지시 튜닝(instruction tuning)`이 필수적입니다. 기존의 `Self-Instruct` 튜닝 방식은 주로 GPT-3.5와 같은 이전 세대 모델이나 공유된 사용자 대화 데이터를 활용했습니다. 이 연구는 최신이자 가장 강력한 독점 LLM인 GPT-4를 교사 모델로 활용하여, `LLaMA`와 같은 오픈소스 LLM의 지시 따르기 능력을 한 단계 더 발전시키고 그 효용성을 검증하는 것을 목표로 합니다.

## ✨ Key Contributions

* **GPT-4 기반 지시-응답 데이터 최초 공개**: GPT-4를 활용하여 52K개의 영어 및 중국어 지시-응답 데이터셋을 생성하고 공개했습니다.
* **GPT-4 생성 피드백 데이터 공개**: 세 가지 지시 튜닝 모델의 출력에 대한 GPT-4 기반 평가 및 비교 데이터를 공개하여 보상 모델(reward model) 학습을 지원합니다.
* **LLaMA 기반 지시 튜닝 모델 개발**: GPT-4가 생성한 데이터를 사용하여 `LLaMA-GPT4` 및 `LLaMA-GPT4-CN` 모델과 보상 모델을 개발했습니다.
* **성능 우수성 입증**: GPT-4로 생성된 데이터가 이전 SOTA 모델(예: GPT-3.5 기반 `Alpaca`)이 생성한 데이터보다 새로운 작업에서 우수한 제로샷(zero-shot) 성능을 제공함을 경험적으로 검증했습니다.
* **실용적인 통찰 제공**: LLM 기반의 범용 지시 따르기 에이전트를 구축하기 위한 실용적인 팁을 제시합니다.

## 📎 Related Works

* **지시 튜닝(Instruction Tuning)**: LLM의 능력을 향상시키기 위한 연구 방향으로, Zhong et al. (2021), Ouyang et al. (2022), Wei et al. (2021), Chung et al. (2022) 등의 연구가 있습니다. `PromptSource` (Bach et al., 2022) 및 `T0` (Sanh et al., 2021)와 같은 프롬프트 컬렉션과 멀티태스크 훈련 모델도 언급됩니다.
* **Self-Instruct 튜닝**: Wang et al. (2022a)이 제안한 방식으로, 이 연구의 기반이 됩니다.
* **LLM 정렬(Alignment)**: Askell et al. (2021), Ouyang et al. (2022) 등에서 논의된 인간 선호도에 LLM 행동을 맞추는 `RLHF` (Reinforcement Learning from Human Feedback)의 중요성을 강조합니다.
* **오픈소스 LLM 노력**: `BLOOM` (Scao et al., 2022), `GPT-J` (Wang & Komatsuzaki, 2021), `GPT-NEO` (Black et al., 2021), `OPT` (Zhang et al., 2022), `LLaMA` (Touvron et al., 2023)와 같은 기반 모델과 `Open-Assistant` (LAION-AI, 2023), `Alpaca` (Taori et al., 2023), `Vicuna` (Vicuna, 2023), `Dolly` (Databricks, 2023) 같은 챗봇 개발 사례가 있습니다.
* **GPT-4의 자체 평가 능력**: Peng et al. (2023), Bai et al. (2022), Madaan et al. (2023), Kim et al. (2023) 등의 연구에서 GPT-4가 자체 오류를 식별하고 응답 품질을 정확하게 판단할 수 있음을 보여줍니다.

## 🛠️ Methodology

1. **데이터 수집**: `Alpaca` 데이터셋(Taori et al., 2023)에서 52K개의 고유 지시를 재사용하여 GPT-4로 응답을 생성했습니다.
    * **영어 지시-응답 데이터**: 52K개 영어 지시에 대해 GPT-4가 응답을 생성했습니다. 프롬프트 엔지니어링 및 API 호출 매개변수(예: `temperature = 1.0`, `top_p = 1.0`, `max_tokens = 512`)는 Algorithm 1에 명시되어 있습니다.
    * **중국어 지시-응답 데이터**: ChatGPT를 통해 52K개 지시를 중국어로 번역한 후, GPT-4가 중국어로 응답을 생성했습니다.
    * **비교 데이터**: GPT-4에게 자체 응답을 1점에서 10점까지 평가하게 하고, GPT-4, GPT-3.5, OPT-IML 세 모델의 응답을 비교 평가하게 하여 보상 모델 학습에 사용했습니다.
    * **Unnatural Instructions 응답**: 68K개의 `Unnatural Instructions` (Honovich et al., 2022)에 대한 GPT-4 응답을 생성하여, 미세 튜닝된 모델과의 격차를 정량화하는 데 사용했습니다.
    * **데이터 통계**: GPT-4는 GPT-3.5보다 긴 응답 시퀀스를 생성하는 경향이 있으며, 동사-명사 쌍 분포에서 차이를 보였습니다.
2. **모델 훈련 (Self-Instruct Tuning)**: `LLaMA 7B` 체크포인트를 기반으로 두 모델을 지도 미세 튜닝(supervised finetuning)했습니다.
    * `LLaMA-GPT4`: 52K개 영어 GPT-4 지시-응답 데이터로 훈련.
    * `LLaMA-GPT4-CN`: 52K개 중국어 GPT-4 지시-응답 데이터로 훈련.
    * 훈련 일정은 `Alpaca` (Taori et al., 2023)와 동일하게 따랐습니다.
3. **보상 모델(Reward Models) 훈련**:
    * `OPT 1.3B` (Iyer et al., 2022)를 기반으로 보상 모델을 훈련하여 다양한 응답의 품질을 평가했습니다.
    * GPT-4가 매긴 점수 $s \in [1, 10]$를 바탕으로, 응답 쌍 $(y_{l}, y_{h})$에 대해 $s_{l} < s_{h}$인 경우 목적 함수 $\min \log(\sigma(r_{\theta}(x, y_{h}) - r_{\theta}(x, y_{l})))$를 사용하여 모델 $r_{\theta}$를 훈련했습니다.
4. **평가 벤치마크**:
    * `User-Oriented-Instructions-252` (Wang et al., 2022a): 252개의 수동 큐레이션된 사용자 중심 지시.
    * `Vicuna-Instructions-80` (Vicuna, 2023): GPT-4가 합성한 80개의 도전적인 질문.
    * `Unnatural Instructions` (Honovich et al., 2022): `text-davinci-002`가 생성한 68,478개의 샘플.
5. **평가 지표**:
    * **인간 평가 (Human Evaluation)**: `HHH` (Helpfulness, Honesty, Harmlessness) 정렬 기준에 따라 `User-Oriented-Instructions-252`에서 모델 응답을 평가했습니다.
    * **GPT-4를 이용한 자동 평가 (Automatic Evaluation with GPT-4)**: `Vicuna-Instructions-80` 질문에 대해 GPT-4가 1점부터 10점까지 모델 응답 품질을 평가했습니다.
    * **ROUGE-L**: `Unnatural Instructions` 데이터셋에 대해 모델 응답과 정답 간의 일치도를 측정했습니다.

## 📊 Results

* **인간 평가 (HHH 정렬 기준)**:
  * `LLaMA-GPT4` vs. `Alpaca` (GPT-3.5 기반):
    * **Helpfulness**: `LLaMA-GPT4`가 54.12%의 득표율로 `Alpaca` (19.74%)를 크게 앞질렀습니다.
    * **Honesty & Harmlessness**: 동점 비율이 가장 높았으나, `Alpaca`가 미세하게 우세한 경향을 보였습니다.
  * `LLaMA-GPT4` vs. `GPT-4` (교사 모델):
    * 세 가지 기준 모두에서 `LLaMA-GPT4`가 `GPT-4`와 매우 유사한 성능을 보이며, GPT-4 데이터를 통한 학습의 효과를 입증했습니다.
* **자동 평가 (GPT-4 활용, Vicuna-Instructions-80)**:
  * GPT-4 평가 결과, GPT-4 데이터로 튜닝된 LLaMA-GPT4 (7B)가 `text-davinci-003`로 튜닝된 `Alpaca` (13B)나 미튜닝 LLaMA (13B)보다 높은 성능을 보였습니다. 그러나 GPT-4 자체와는 여전히 격차가 있었습니다.
  * 보상 모델이 순위를 매긴 상위 응답 그룹이 기준선보다 더 나은 성능을 보여, 보상 모델의 유효성을 확인했습니다.
  * 중국어 평가에서도 영어와 일관된 경향을 보였습니다. GPT-4는 직접 생성한 중국어 응답보다 영어 응답을 중국어로 번역한 경우에 더 높은 성능을 보여, 영어 학습 말뭉치(corpus)의 풍부함이 더 강한 영어 지시 따르기 능력으로 이어진 것으로 분석됩니다.
* **ROUGE-L (Unnatural Instructions)**:
  * 전체 평균 ROUGE-L 점수는 `Alpaca` (0.39)가 `LLaMA-GPT4` (0.34) 및 `GPT-4` (0.37)보다 약간 높았습니다.
  * 하지만 정답 응답 길이가 4보다 길어질수록 `LLaMA-GPT4`와 `GPT-4`가 더 나은 성능을 보였는데, 이는 창의적인 시나리오에서 지시를 더 잘 따름을 시사합니다.
  * `LLaMA-GPT4`는 `GPT-4`의 동작을 더 가깝게 모방했으며, 짧은 응답의 경우 `GPT-4`와 `LLaMA-GPT4`는 "챗봇스러운" 부가적인 설명을 추가하여 ROUGE-L 점수가 낮게 나올 수 있었습니다.

## 🧠 Insights & Discussion

* **GPT-4의 교사 모델로서의 강력한 효과**: GPT-4가 생성한 지시 데이터는 오픈소스 LLM(`LLaMA-GPT4`)을 지시 튜닝하는 데 매우 효과적이며, 이를 통해 `GPT-4`와 같은 독점 SOTA 모델에 필적하는 성능을 달성할 수 있음을 입증했습니다.
* **교차 언어 일반화 가능성**: 번역된 지시와 GPT-4의 중국어 응답을 활용하여 중국어 지시 따르기 모델을 구축할 수 있음을 보여주었습니다.
* **보상 모델의 잠재적 가치**: GPT-4가 생성한 비교 데이터와 이를 통해 훈련된 보상 모델은 응답 순위화 및 향후 `RLHF`에 유용하게 활용될 수 있는 가능성을 제시합니다.
* **한계점 및 향후 연구**:
  * **데이터 및 모델 스케일**: 현재 52K개의 데이터셋과 LLaMA 7B 모델은 `Vicuna` (700K 데이터, 13B LLaMA)와 같은 다른 노력에 비해 작은 규모입니다. 향후 더 많은 GPT-4 지시-응답 데이터를 수집하고, `ShareGPT` 데이터와 결합하여 더 큰 LLaMA 모델을 훈련할 필요가 있습니다.
  * **RLHF 통합**: 현재 보상 모델은 디코딩 단계에서만 사용되었습니다. 기계 생성 피드백을 이용한 강화 학습(RLHF)을 통해 LLM을 지속적으로 훈련하는 방향으로 연구를 확장할 수 있습니다.
  * **ROUGE-L의 한계**: 짧은 창의적 응답의 경우 `GPT-4`와 `LLaMA-GPT4`가 대화형의 추가 정보를 포함하여 ROUGE-L 점수가 낮게 나올 수 있었는데, 이는 이 지표가 지시 따르기 능력을 완전히 반영하지 못할 수 있음을 시사합니다.

## 📌 TL;DR

* **문제**: 오픈소스 LLM의 지시 따르기 능력을 향상시키기 위해 최신 GPT-4를 교사 모델로 활용하는 방법을 모색.
* **방법**: GPT-4를 사용하여 52K개의 영어 및 중국어 지시-응답 데이터와 보상 모델 학습을 위한 비교 데이터를 생성. 이 데이터를 사용하여 LLaMA 7B 모델을 미세 조정(`LLaMA-GPT4`, `LLaMA-GPT4-CN`).
* **결과**: `LLaMA-GPT4`는 인간 평가(`Helpfulness`)에서 GPT-3.5 기반 `Alpaca`를 크게 능가했으며, `GPT-4` 자체와 비슷한 성능을 보였다. GPT-4 자동 평가에서도 `LLaMA-GPT4`가 `Alpaca`보다 우수함을 확인. GPT-4 데이터가 개방형 LLM의 지시 따르기 성능을 크게 향상시킬 수 있음을 입증.
