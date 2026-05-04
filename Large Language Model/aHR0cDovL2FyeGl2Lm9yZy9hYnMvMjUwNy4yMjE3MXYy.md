# Enhancing Jailbreak Attacks on LLMs via Persona Prompts

Zheng Zhang, Peilin Zhao, Deheng Ye, Hao Wang (2025)

## 🧩 Problem to Solve

본 논문은 대규모 언어 모델(LLM)의 안전 가드레일을 무력화하여 유해한 콘텐츠를 생성하도록 유도하는 **Jailbreak 공격**의 효율성을 높이는 문제를 다룬다. 기존의 Jailbreak 연구들은 주로 유해한 의도(Harmful Intent)를 직접적으로 조작하거나 표현 방식을 수정하는 데 집중해 왔다. 그러나 LLM의 상호작용 스타일이나 정체성을 결정하는 **Persona Prompt**(예: "당신은 유능한 조수입니다")가 모델의 안전 방어 체계에 어떠한 영향을 미치는지에 대한 체계적인 연구는 부족한 실정이다.

따라서 본 연구의 목표는 Persona Prompt가 LLM의 방어 기제에 미치는 영향을 분석하고, 유해한 요청에 대해 모델이 거부하지 않고 응답할 확률을 극대화하는 최적의 Persona Prompt를 자동으로 생성하는 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **유전 알고리즘(Genetic Algorithm, GA)**을 활용하여 LLM의 거부 반응을 최소화하는 Persona Prompt를 자동 진화시키는 것이다. 연구진은 특정 Persona가 LLM의 거부 메커니즘을 약화시켜, 결과적으로 다른 Jailbreak 공격 기법들이 더 쉽게 작동할 수 있는 '낮은 방어 상태'의 컨텍스트를 제공한다는 직관을 제시한다. 주요 기여 사항은 다음과 같다.

- **자동화된 Persona 진화 프레임워크**: 유전 알고리즘을 통해 LLM의 거부율(RtA)을 낮추는 최적의 Persona Prompt를 자동으로 탐색하고 생성한다.
- **방어 무력화 및 시너지 효과 입증**: 진화된 Persona Prompt가 단독으로 사용될 때보다 기존의 Jailbreak 방법론과 결합했을 때 공격 성공률(ASR)을 10~20% 가량 유의미하게 향상시킴을 보였다.
- **범용성 및 강건성 확인**: 특정 모델에서 진화시킨 Persona Prompt가 다른 모델(Cross-model)에서도 효과적으로 작동하며, 일반적인 프롬프트 수준의 방어 전략에 대해서도 강건함을 입증하였다.

## 📎 Related Works

### 기존 Jailbreak 접근 방식 및 한계
1. **최적화 기반 기법 (GCG 등)**: 그래디언트를 사용하여 적대적 접미사(Adversarial Suffix)를 생성하지만, 생성된 텍스트가 난수 형태인 경우가 많아 폐쇄형 모델로의 전이성이 낮다.
2. **프롬프트 재작성 기법 (AutoDAN, GPTFuzzer 등)**: 템플릿을 반복적으로 조합하고 강화하지만, 주로 유해한 의도 자체를 숨기는 데 집중한다.
3. **가상 시나리오 및 인코딩 기법 (PAIR, PAP, ArtPrompt 등)**: 가상 상황을 설정하거나 저자원 언어, ASCII 아트 등으로 변환하여 필터를 우회한다.

### 본 연구의 차별점
기존 연구들이 유해한 요청(Harmful Request) 자체를 수정하는 것에 집중했다면, 본 연구는 요청 앞에 붙는 **Persona Prompt**라는 별도의 구성 요소에 주목한다. Persona Prompt는 특정 유해 의도에 종속되지 않으므로, 한 번 최적화되면 다양한 종류의 공격 프롬프트에 범용적으로 결합하여 사용할 수 있다는 점에서 기존 방식과 차별화된다.

## 🛠️ Methodology

본 논문은 유전 알고리즘을 사용하여 LLM의 방어력을 낮추는 Persona Prompt를 진화시킨다. 전체 파이프라인은 **초기화 $\rightarrow$ 교차 $\rightarrow$ 변이 $\rightarrow$ 선택**의 반복적인 사이클로 구성된다.

### 1. 초기화 (Initialization)
초기 집단 $P_0$는 소설이나 영화 속 캐릭터 설명 데이터셋인 `inCharacter`에서 추출한 35개의 Persona 설명으로 구성된다. 이때, 캐릭터 이름이나 배경 지식 등 불필요한 정보를 제거하고 성격적 특성만을 남기기 위해 GPT-4o를 이용한 정제(Sanitization) 과정을 거친다.
$$P_0 = \{p_1, p_2, \dots, p_N\}, \quad N = 35$$

### 2. 교차 (Crossover)
다양한 특성의 조합 가능성을 탐색하기 위해 교차 메커니즘을 사용한다. 매 반복마다 무작위로 $M$ 쌍의 프롬프트를 선택하고, LLM을 통해 두 프롬프트의 핵심 속성을 결합한 새로운 프롬프트를 생성한다.
$$P_{cross} = \{c_k \mid c_k = \text{Crossover}(p_1^{(k)}, p_2^{(k)}), \quad k = 1, \dots, M\}$$

### 3. 변이 (Mutation)
프롬프트 공간의 무작위 탐색과 다양성 확보를 위해 변이 과정을 거친다. $M$개의 프롬프트를 무작위로 선택하여 LLM을 통해 **재작성(Rewriting), 확장(Expansion), 축소(Contraction)** 중 하나의 변환을 적용한다. 특히 프롬프트 길이를 일정하게 유지하기 위해 100단어 초과는 축소, 10단어 미만은 확장하도록 강제한다.
$$P_{mut} = \{m_k \mid m_k = \text{Mutate}(p^{(k)}), \quad k = 1, \dots, M\}$$

### 4. 선택 (Selection)
거부율(RtA, Refuse to Answer)이 가장 낮은 상위 $N$개의 프롬프트만을 남기고 나머지는 제거한다.
$$P_{t+1} = \{p \mid \text{Rank}(p) \le N, p \in P_t \cup P_{cross} \cup P_{mut}\}$$
여기서 선택 지표로 사용되는 **RtA**는 LLM이 요청을 명시적으로 거부하는 비율을 의미하며, 이 값이 낮을수록 방어 체계를 더 잘 우회한 것으로 간주한다.

## 📊 Results

### 실험 설정
- **데이터셋**: AdvBench(520개 유해 프롬프트) 및 TrustLLM 벤치마크(1,400개 유해 프롬프트)의 서브셋을 사용하였다.
- **측정 지표**:
    - **RtA (Refuse to Answer)**: 모델이 응답을 거부하는 비율 ($\downarrow$).
    - **ASR (Attack Success Rate)**: GPT-4o-mini를 평가자로 사용하여 실제 유해한 콘텐츠가 생성되었는지 판단한 성공률 ($\uparrow$).
    - **HS (Harmful Score)**: 유해성 정도를 1~5점으로 수치화한 점수 ($\uparrow$).
- **대상 모델 (Victim LLMs)**: GPT-4o-mini, GPT-4o, Qwen2.5-14B, LLaMA-3.1-8B, DeepSeek-V3.

### 주요 결과
1. **거부율의 급격한 감소**: 진화된 Persona Prompt를 사용했을 때 GPT-4o-mini, GPT-4o, DeepSeek-V3 등에서 거부율(RtA)이 50~70% 감소하였다.
2. **시너지 효과**: Persona Prompt 단독으로는 ASR이 낮았으나, 이를 GPTFuzzer, PAP, Chat-NN 등 기존 공격 기법과 결합했을 때 ASR이 10~20% 상승하였다. 이는 Persona Prompt가 1차 방어선인 '거부 메커니즘'을 약화시켜 후속 공격이 더 잘 통하게 만들기 때문이다.
3. **전이 가능성 (Transferability)**: GPT-4o-mini에서 진화시킨 Persona Prompt를 Qwen이나 LLaMA-3.1에 적용했을 때도 RtA 감소 효과가 유지되었으며, 특히 PAP와 결합 시 ASR이 10~30% 향상되었다.
4. **방어 기제에 대한 강건성**: Adaptive System Prompt, Paraphrasing, Safety-Prioritized Prompt 등의 방어 전략을 적용하더라도 여전히 유의미한 RtA 감소 효과를 보였다.

## 🧠 Insights & Discussion

### Persona Prompt의 배치 및 특성
- **배치 위치**: Persona Prompt를 **System Prompt**에 배치하는 것이 가장 효과적이었으며, User Prompt의 시작 부분, 끝 부분 순으로 효과가 낮아졌다. 이는 LLM 학습 데이터에서 Persona 지침이 주로 시스템 프롬프트나 대화 시작점에 위치하는 구조적 패턴 때문으로 분석된다.
- **진화된 프롬프트의 특징**: 성공적인 Persona Prompt들은 공통적으로 (1) 짧은 문장 사용, (2) 수사적 질문(Rhetorical questions) 활용, (3) 자기비하적 유머(Self-deprecating humor) 포함이라는 특성을 보였다.

### 알고리즘 분석
- **다양성의 중요성**: 초기 집단의 semantic diversity(의미적 다양성)가 초기 RtA 값보다 수렴 속도와 최종 성능에 더 큰 영향을 미친다는 것을 확인하였다.
- **선택 지표의 영향**: ASR을 직접 최적화하는 것보다 RtA를 낮추는 방향으로 진화시킨 프롬프트가 다른 공격 기법과 결합했을 때 훨씬 더 큰 시너지를 냈다. 이는 단순한 유해 콘텐츠 생성보다 '거부하지 않는 상태'를 만드는 것이 더 강력한 기반이 됨을 시사한다.
- **내부 메커니즘 (Attention Shift)**: Llama-3.1-8B에 대한 사례 연구 결과, Persona Prompt가 적용되면 모델의 Attention이 "fake", "reviews" 같은 유해 키워드에서 "whimsical", "humor" 같은 스타일 관련 키워드로 이동함을 확인하였다.

### 한계점
본 연구의 진화 성능은 초기 집단의 품질에 의존하며, 현재는 공개된 데이터셋의 Persona에 국한되어 있다. 향후 LLM을 이용한 자동화된 초기 집단 생성 기법이 필요하다.

## 📌 TL;DR

본 논문은 유전 알고리즘을 통해 LLM의 거부 반응을 최소화하는 **Persona Prompt를 자동으로 진화**시키는 프레임워크를 제안한다. 진화된 Persona Prompt는 모델의 1차 방어선인 거부 메커니즘을 무력화하여 **거부율(RtA)을 50~70% 감소**시키며, 기존의 다양한 Jailbreak 공격 기법과 결합했을 때 **공격 성공률(ASR)을 크게 높이는 시너지 효과**를 낸다. 이는 Persona Prompt가 모델의 Attention을 유해 키워드에서 스타일 지침으로 분산시킴으로써 방어 체계를 약화시킨다는 점을 시사하며, 향후 LLM 안전 정렬(Safety Alignment) 연구에 있어 Persona 기반 조작에 대한 대비책이 필요함을 강조한다.