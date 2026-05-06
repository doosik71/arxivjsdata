# Analyzing Modular Approaches for Visual Question Decomposition

Apoorv Khandelwal, Ellie Pavlick, and Chen Sun (2023)

## 🧩 Problem to Solve

본 논문은 Visual Question Answering (VQA) 분야에서 최근 주목받고 있는 모듈형(Modular) 또는 뉴로-심볼릭(Neuro-symbolic) 접근 방식의 실질적인 성능 향상 원인을 분석하는 것을 목표로 한다.

기존의 End-to-end 신경망 모델들은 높은 성능을 보이지만, 해석 가능성(Interpretability)과 일반화 능력(Generalization capabilities)이 부족하다는 한계가 있다. 이를 해결하기 위해 ViperGPT와 같은 모듈형 시스템은 거대 언어 모델(LLM)을 이용해 문제를 해결하기 위한 심볼릭 프로그램(Python 코드)을 생성하고, 이를 특정 작업에 특화된 모듈들을 통해 실행함으로써 논리적 추론을 수행한다.

그러나 이러한 모듈형 시스템은 내부적으로 BLIP-2와 같은 최신 End-to-end 모델을 구성 요소로 포함하고 있어, 실제 성능 향상이 모듈화된 시스템 구조(Symbolic components)에서 오는 것인지, 아니면 단순히 내장된 강력한 End-to-end 모델의 성능 덕분인지 구분하기 어렵다. 따라서 본 연구는 ViperGPT를 중심으로 모듈형 접근 방식의 기여도를 정밀하게 분석하고, 이를 대체할 수 있는 더 단순한 방식의 가능성을 탐색한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 ViperGPT의 성능 향상이 시스템의 모듈 구조 자체보다는 **작업별 특화된 모듈 선택(Task-specific module selection)과 내장된 End-to-end 모델(BLIP-2)의 성능에 크게 의존하고 있음**을 밝혀낸 것이다.

주요 직관은 다음과 같다.

1. ViperGPT의 성능 이점은 작업에 맞춰 모듈을 미리 선택하는 엔지니어링 단계에서 발생하며, 작업에 무관한(Task-agnostic) 설정에서는 BLIP-2 대비 뚜렷한 이점이 사라진다.
2. 복잡한 Python 코드 생성 방식 대신, 자연어를 이용한 단계적 질문 분해(Successive Prompting)만으로도 유사하거나 때로는 더 높은 성능을 얻을 수 있다.
3. 심볼릭 프로그램 방식은 Out-of-Distribution (OOD) 데이터셋에서 런타임 에러(Runtime error)가 증가하는 취약성을 보인다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 언급한다.

- **End-to-end 모델**: BLIP-2와 같이 이미지 인코더와 언어 모델을 결합하여 제로샷(Zero-shot) VQA를 수행하는 모델들이 주류를 이루고 있다.
- **뉴로-심볼릭/모듈형 방법론**: Neural Module Networks (NMN)와 같은 초기 연구부터, 최근 LLM을 이용해 프로그램을 생성하는 ViperGPT, Visual Programming 등이 있다. 이들은 해석 가능성을 높이고 복잡한 추론을 수행하려 한다.
- **자연어 추론 및 도구 사용**: Chain-of-Thought (CoT)와 같이 문제를 단계적으로 분해하여 푸는 LLM의 추론 방식과, 외부 도구를 호출하는 Tool-use 연구들이 진행되고 있다.

본 논문은 기존 모듈형 연구들이 주장하는 '모듈화의 이점'을 비판적으로 분석하며, 단순한 자연어 프롬프팅 기반의 분해 전략과 직접 비교함으로써 차별성을 둔다.

## 🛠️ Methodology

본 연구는 세 가지 모델 패밀리를 설계하여 비교 분석한다.

### 1. End-to-end (BLIP-2)

기준 모델로 사용되며, 이미지 인코더와 언어 모델(FlanT5-XXL)을 통해 질문에 대한 직접적인 텍스트 답변을 생성한다. 프롬프트 형식은 `"Question: {} Short answer: []"`를 사용한다.

### 2. Modular (ViperGPT)

LLM(Codex)이 주어진 질문과 API 명세서를 바탕으로 Python 프로그램을 생성하고, 이를 인터프리터가 실행하는 구조이다.

- **구성 요소**: `ImagePatch` 클래스와 `.find`, `.simple_query` 등의 함수로 구성된 API를 제공한다.
- **실행 과정**: 생성된 코드는 GLIP(객체 검출), MiDaS(깊이 추정), BLIP-2(VQA), X-VLM, InstructGPT 등의 신경망 모듈과 심볼릭 알고리즘을 호출하여 최종 답을 도출한다.
- **추론 절차**: 질문 $\rightarrow$ LLM (코드 생성) $\rightarrow$ Python 인터프리터 (모듈 호출 및 실행) $\rightarrow$ 최종 답변.

### 3. Successive Prompting

코드 생성 없이 자연어만으로 문제를 분해하는 방식이다.

- **작동 원리**: LLM(InstructGPT)과 VLM(BLIP-2)을 교대로 호출한다.
- **절차**:
    1. LLM이 원본 질문을 풀기 위한 첫 번째 후속 질문(Follow-up question)을 생성한다.
    2. VLM이 해당 후속 질문에 답한다.
    3. LLM이 이전의 모든 질문-답변 쌍을 참고하여 다음 후속 질문을 생성하거나, 최종 답변을 내놓는다.
- 이 방식은 코드의 논리적 실행 대신 LLM의 암시적 추론과 자연어 인터페이스를 활용한다.

### 평가 지표 및 수식

- **Direct Answer**: 기존 VQA 지표 외에 LLM을 이용한 `InstructGPT-eval` 지표를 사용한다.
- **Multiple Choice**: 표준 정확도(Accuracy)를 사용한다.
- **Log Likelihood**: 다지선다형 문제에서 답변 선택 시 다음 수식을 사용하여 가장 가능성이 높은 선택지를 고른다.
$$ \text{Score} = \frac{\sum_{j=m}^{m+(n-1)} \log P(x_j | x_{0:j})}{\sum_{k=m}^{m+(n-1)} L_{x_k}} $$
여기서 $L_{x_k}$는 토큰의 바이트 길이(Byte-length)를 의미한다.

## 📊 Results

### 모듈 선택의 영향 (ViperGPT 분석)

- **Task-agnostic 설정**: 각 작업에 특화된 모듈을 선택하지 않고 전체 API를 제공했을 때, ViperGPT의 BLIP-2 대비 성능 향상폭이 급격히 감소한다. (예: OK-VQA에서 $+11.1\% \rightarrow -3.6\%$)
- **Ablation Study**:
  - **Without BLIP-2**: BLIP-2 모듈을 제거해도 ViperGPT는 원래 성능의 $84\text{--}87\%$를 유지한다. 이는 다른 모듈들이 어느 정도 보완 역할을 함을 의미한다.
  - **Only BLIP-2**: 오직 BLIP-2 모듈만 사용하도록 제한했을 때, ViperGPT 성능의 약 $95\%$가 유지된다. 이는 사실상 BLIP-2가 대부분의 핵심 역할을 수행하고 있음을 시사한다.

### 프로그램 vs 프롬프팅 비교

- **Successive Prompting** 방식은 ViperGPT (Only BLIP-2) 모델과 비교했을 때, 직접 답변(Direct Answer) 설정에서 평균 $92\%$의 성능을 유지하며 유사한 성능을 보였다.
- 특히 다지선다형(Multiple Choice) 설정(A-OKVQA, ScienceQA)에서는 Successive Prompting이 ViperGPT보다 평균 $12\%$ 더 높은 성능을 기록했다.

### 일반화 능력 및 OOD 분석

- **OOD 성능 저하**: A-OKVQA 및 ScienceQA와 같은 OOD 데이터셋에서 작업 특화 예시(In-context examples)를 제공했을 때, 오히려 성능이 $2\text{--}11\%$ 하락하는 현상이 발견되었다.
- **에러 분석**: OOD 데이터셋에서 ViperGPT의 코드 실행 실패율이 급증한다. ScienceQA의 경우 런타임 에러(Runtime error)율이 $18\%$에 달하며, 구문 에러(Syntax error) 또한 $1\text{--}3\%$ 발생한다.

## 🧠 Insights & Discussion

본 논문은 모듈형 VQA 시스템이 제공하는 '논리적 프로그램 생성'의 가치에 대해 비판적인 시각을 제공한다.

**강점 및 발견**:

- ViperGPT의 성능 향상은 순수한 모듈 구조의 이점이라기보다, 내부의 강력한 VLM(BLIP-2)과 정교한 작업별 모듈 튜닝(Engineering)의 결과일 가능성이 높다.
- 자연어 기반의 단계적 분해(Successive Prompting)가 복잡한 코드 생성 없이도 동등하거나 더 우수한 성능을 낼 수 있음을 보여줌으로써, 시스템 복잡도를 낮출 수 있는 대안을 제시했다.

**한계 및 비판적 해석**:

- 심볼릭 프로그램 방식은 정해진 API 내에서는 강력하지만, 정의되지 않은 도메인(OOD)으로 확장될 때 코드 생성 오류라는 치명적인 취약점을 가진다.
- 반면, 자연어 프롬프팅 방식은 LLM의 유연한 추론 능력에 의존하므로 코드 실행 에러와 같은 하드웨어적 실패는 없으나, LLM 자체의 환각(Hallucination) 문제에 노출될 수 있다.

## 📌 TL;DR

본 연구는 ViperGPT와 같은 모듈형 VQA 시스템의 성능이 실제로는 내장된 BLIP-2 모델과 작업별 수동 튜닝에 크게 의존하고 있음을 분석하였다. 또한, 복잡한 Python 코드 생성 대신 자연어로 문제를 분해하는 **Successive Prompting** 방식이 더 단순하면서도 일부 작업(특히 다지선다형)에서는 더 뛰어난 성능을 보임을 입증하였다. 이는 향후 VQA 연구가 복잡한 심볼릭 구조 설계보다는 효율적인 자연어 추론 파이프라인 구축에 집중해야 함을 시사한다.
