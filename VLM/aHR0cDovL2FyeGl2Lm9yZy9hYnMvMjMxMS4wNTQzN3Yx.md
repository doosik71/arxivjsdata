# LLaVA-Plus: Learning to Use Tools for Creating Multimodal Agents

Shilong Liu et al. (2023)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 기존의 대규모 다중모달 모델(Large Multimodal Models, LMMs)이 가진 기능적 한계와 기존 툴 체이닝(Tool Chaining) 방식의 취약성이다.

현재 LLaVA와 같은 LMM들은 이미지 이해와 추론 능력은 뛰어나지만, 이미지 세그멘테이션(Segmentation), 정밀한 이미지 생성(Generation), 혹은 외부 지식 검색과 같은 전문적인 기술을 통합적으로 수행하는 데 어려움이 있다. 반면, LangChain과 같은 툴 체이닝 방식은 별도의 학습 없이 프롬프트 엔지니어링을 통해 외부 툴을 호출하지만, 이는 프롬프트에 지나치게 의존하므로 복잡하고 다양한 툴셋 중에서 적절한 툴을 정확하게 선택하고 그 결과를 유연하게 조합하는 능력이 부족하며 견고함(Robustness)이 떨어진다.

따라서 본 연구의 목표는 LMM이 다양한 전문 툴을 능동적으로 선택하고 사용할 수 있도록 학습시킴으로써, 현실 세계의 복잡한 다중모달 작업을 수행할 수 있는 범용 다중모달 어시스턴트를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 LMM을 단순한 답변 생성기가 아닌, 툴 사용을 계획하는 **플래너(Planner)**로 동작하게 만드는 것이다. 이를 위해 다음과 같은 기여를 수행하였다.

1. **툴 사용을 위한 다중모달 지시어 튜닝(Instruction Tuning):** LMM이 언제, 어떤 툴을 호출하고 그 결과를 어떻게 해석해야 하는지를 학습시키기 위해 새로운 다중모달 지시어 데이터셋을 구축하였다.
2. **통합 예측 포맷(Unified Prediction Format):** 모델이 추론 과정($\text{Thought}$), 실제 툴 호출($\text{Action}$), 최종 응답($\text{Value}$)을 순차적으로 생성하도록 하는 구조를 도입하여 모델의 계획 능력을 향상시켰다.
3. **확장 가능한 스킬 저장소(Skill Repository):** 객체 검출(G-DINO), 세그멘테이션(SAM), 이미지 생성(Stable Diffusion) 등 다양한 전문 모델들을 툴로 구성하여 LMM이 필요에 따라 이를 플러그인 형태로 사용할 수 있게 하였다.
4. **엔드투엔드 학습과 툴 체이닝의 결합:** 프롬프트 기반의 툴 체이닝과 달리, LMM 자체가 툴 사용법을 학습하게 함으로써 더 높은 유연성과 견고함을 확보하였다.

## 📎 Related Works

본 논문은 기존 연구를 두 가지 방향으로 분류하고 차별점을 제시한다.

- **다중모달 툴 사용 AI 에이전트:** Visual ChatGPT나 MM-ReAct와 같은 연구들은 프롬프트 엔지니어링이나 인컨텍스트 러닝(In-context learning)을 통해 툴을 호출한다. 하지만 LLaVA-Plus는 LMM을 직접 튜닝하여 플래너로 활용하며, 특히 전체 상호작용 세션 동안 원본 이미지 신호를 지속적으로 참조하여 계획 및 추론 능력을 높였다는 점에서 차별화된다.
- **범용 다중모달 모델:** Flamingo, LLaVA, GPT-4V 등은 강력한 제로샷 전이 능력을 보여주지만, 텍스트 출력에 국한되거나 특정 기능(예: 바운딩 박스 출력)만 일부 지원한다. LLaVA-Plus는 훨씬 더 넓은 범위의 전문 기술(세그멘테이션, 편집, 생성 등)과 그 조합을 가능하게 하여 기능적 범위를 획기적으로 확장하였다.

## 🛠️ Methodology

### 전체 시스템 구조 및 파이프라인

LLaVA-Plus는 LMM이 플래너 역할을 수행하고, 외부의 스킬 저장소(Skill Repository)에서 필요한 툴을 호출하는 구조이다. 전체 상호작용은 다음의 4단계로 진행된다.

1. **사용자 입력:** 사용자가 이미지 $I_q$와 함께 지시어 $X_q$를 제공한다.
2. **툴 선택 및 호출:** LMM이 $X_q$와 $I_q$를 분석하여 어떤 툴을 사용할지 결정하고, 툴에 전달할 인자(Argument)가 포함된 $X_{\text{skill use}}$를 생성한다.
3. **툴 실행:** 선택된 전문 모델이 실행되어 결과값 $X_{\text{skill result}}$를 반환한다.
4. **최종 응답 생성:** LMM이 툴 실행 결과와 이전 대화 맥락을 종합하여 최종 답변 $X_{\text{answer}}$를 생성한다.

이 과정은 다음과 같은 시퀀스로 표현된다.
$$\text{Human}: I_q \langle \backslash n \rangle X_q \langle \text{STOP} \rangle \text{Assistant}: X_{\text{skill use}} \langle \text{STOP} \rangle \text{Human}: X_{\text{skill result}} \langle \text{STOP} \rangle \text{Assistant}: X_{\text{answer}} \langle \text{STOP} \rangle$$

### 통합 예측 포맷 (Unified Prediction Format)

LMM은 모든 응답을 다음 세 가지 필드로 구성된 포맷으로 출력한다.

- $\text{Thought}$: 툴 사용 필요 여부와 어떤 툴을 사용할지에 대한 추론 과정이다.
- $\text{Action}$: JSON 형식의 함수 호출 리스트이다. $\text{APIname}$과 $\text{APIparams}$를 포함하며, 툴 호출이 필요 없는 경우 빈 리스트($[]$)가 된다.
- $\text{Value}$: 툴 실행 결과 등을 종합한 최종 자연어 응답이다.

### 학습 절차 및 손실 함수

LLaVA-Plus는 자동 회귀(Auto-regressive) 목적 함수를 사용하여 학습된다. 학습 시 사용되는 데이터는 기존 LLaVA-158K 데이터셋(통합 포맷으로 변환됨)과 새롭게 구축한 툴 사용 지시어 데이터를 결합하여 사용한다. 손실 함수는 모델이 생성하는 $\text{Assistant}$ 부분의 토큰들에 대해서만 계산되어, 모델이 적절한 툴 호출과 답변을 예측하도록 유도한다.

### 스킬 저장소 및 데이터 생성

스킬 저장소는 다음과 같은 툴들을 포함한다.

- **이해(Understanding):** G-DINO (검출), OpenSeeD (시맨틱 세그멘테이션), SAM (인스턴스 세그멘테이션), EasyOCR (OCR) 등.
- **외부 지식(Knowledge):** CLIP Retrieval을 통한 외부 정보 검색.
- **생성(Generation):** Stable Diffusion (생성), Instruct-Pix2Pix (편집).
- **시각적 프롬프트(Visual Prompts):** SAM, Semantic-SAM 등을 이용한 포인트/박스 기반 상호작용.

데이터 생성은 GPT-4를 레이블러로 사용하는 $\text{Self-instruct}$ 방식을 채택하였다. 이미지 컨텍스트를 GPT-4에 제공하고, 이에 맞는 다중모달 대화 시나리오를 생성하게 하여 학습 데이터를 구축하였다.

## 📊 Results

### 실험 설정

- **데이터셋 및 벤치마크:** LLaVA-Bench, SEED-Bench, MM-Vet, ViSiT-Bench를 사용하였다.
- **지표:** GPT-4 기반의 정성적 점수, Elo rating, CLIP score 등을 측정하였다.
- **비교 대상:** LLaVA, GPT4Tools, MM-ReAct, Bing Chat, Bard 등.

### 주요 결과

1. **기존 능력 향상:** LLaVA-Plus는 LLaVA-Bench와 SEED-Bench에서 LLaVA보다 높은 성능을 보였다. 이는 전문 툴의 인식 결과가 LMM의 컨텍스트로 제공되어 추론 능력을 보강했기 때문이다.
2. **새로운 능력 획득:** LLaVA-Bench (Tool Use) 실험에서 Grounding, Tagging, OCR 등의 작업에서 타 모델들을 압도하였다. 특히 상용 서비스인 Bing Chat이나 Bard보다 높은 성능을 보였는데, 이는 해당 시스템들이 특정 전문 툴과 정교하게 통합되지 않았기 때문으로 분석된다.
3. **실제 활용성 (ViSiT-Bench):** 실세계 지향 벤치마크인 ViSiT-Bench에서 Elo rating 기준 LLaVA를 100점 이상 앞지르며 새로운 SoTA(State-of-the-art)를 달성하였다.
4. **복합 작업 수행:** 단순히 단일 툴을 사용하는 것을 넘어, '세그멘테이션 $\rightarrow$ 마스크 기반 이미지 생성'과 같은 복합적인 워크플로우를 동적으로 생성하여 수행할 수 있음을 입증하였다.

## 🧠 Insights & Discussion

### 강점

LLaVA-Plus의 가장 큰 강점은 **확장성**이다. 새로운 전문 모델이 개발되면 이를 스킬 저장소에 추가하고 관련 지시어 데이터로 튜닝함으로써 모델 전체를 다시 학습시키지 않고도 능력을 확장할 수 있다. 또한, LMM이 단순한 인터페이스가 아니라 이미지 전체를 계속 보며 계획을 세우는 플래너로 동작하게 함으로써 툴 사용의 정확도를 높였다.

### 한계 및 비판적 해석

논문에서도 언급되었듯이, 여전히 **환각(Hallucination)** 문제가 존재하며, 때로는 부적절한 툴을 선택하는 **툴 사용 충돌(Tool use conflicts)**이 발생한다. 또한, 학습 데이터 생성에 GPT-4에 크게 의존하고 있어, 생성된 데이터의 품질이 LLaVA-Plus의 상한선을 결정짓는 구조적 한계가 있다. 툴의 출력값이 텍스트나 좌표 형태의 심볼릭 표현으로 변환되어 LMM에 전달되는데, 이 과정에서 원본 시각 정보의 손실이 발생할 가능성이 있다.

## 📌 TL;DR

LLaVA-Plus는 LMM을 전문 툴을 사용하는 플래너로 학습시킨 범용 다중모달 어시스턴트이다. $\text{Thought-Action-Value}$라는 통합 포맷과 툴 사용 전용 지시어 데이터셋을 통해 LMM이 동적으로 툴을 선택하고 조합하는 능력을 갖추게 하였다. 이를 통해 기존 LMM이 불가능했던 정밀한 객체 검출, 세그멘테이션, 이미지 편집 및 외부 지식 검색 등의 기능을 통합적으로 수행하며, 특히 ViSiT-Bench에서 SoTA를 기록하였다. 이 연구는 향후 더 많은 전문 툴을 통합한 고도화된 다중모달 AI 에이전트 개발의 중요한 기반이 될 것으로 보인다.
