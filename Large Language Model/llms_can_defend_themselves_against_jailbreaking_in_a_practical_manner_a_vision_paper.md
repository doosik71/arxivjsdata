# LLMs Can Defend Themselves Against Jailbreaking in a Practical Manner: A Vision Paper

Daoyuan Wu, Shuai Wang, Yang Liu, Ning Liu (2024)

## 🧩 Problem to Solve

본 논문은 최신 거대 언어 모델(Large Language Models, LLMs)에 적용된 안전 정렬(Safety Alignment)을 우회하려는 Jailbreaking 공격을 해결하고자 한다. Jailbreaking은 모델이 안전 정책상 답변해서는 안 될 유해한 질문에 대해 답변하도록 유도하는 적대적 공격이다. 최근 Greedy Coordinate Gradient (GCG)와 같은 최적화 기반 공격, "Do-Anything-Now" (DAN)와 같은 템플릿 기반 공격, 그리고 다국어(Multilingual)를 이용한 공격 등 공격 기법은 매우 정교해지고 있다.

반면, 이러한 공격을 막기 위한 방어 측면의 연구는 상대적으로 부족한 실정이다. 기존의 방어 기법들은 추가적인 학습(Fine-tuning)이 필요하여 비용이 많이 들거나, 추론 과정에서 심각한 지연(Latency)을 초래하는 한계가 있다. 따라서 본 논문의 목표는 기존의 다양한 Jailbreaking 공격을 효과적으로 방어하면서도, 정상적인 사용자 요청에 대해서는 지연 시간이 거의 없는 가볍고 실용적인 방어 체계를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **"어떤 정교한 Jailbreaking 전략을 사용하더라도, 결국 LLM에 전달되는 최종 프롬프트에는 '폭탄 만드는 법'과 같은 유해한 프롬프트(Harmful Prompt)가 포함되어야 하며, 최신 LLM들은 이러한 유해한 부분만을 분리하여 인식하는 능력이 충분하다"**는 직관에서 출발한다.

이러한 통찰을 바탕으로, 저자들은 `SELFDEFEND`라는 새로운 방어 아키텍처를 제안한다. 이 시스템의 중심 설계는 정상적인 처리를 담당하는 Normal Stack과 유해성 여부를 동시에 검사하는 Shadow Stack을 병렬로 배치하여, 유해성이 감지되는 즉시 정상 스택의 출력을 차단하는 Checkpoint 메커니즘을 도입한 것이다.

## 📎 Related Works

논문에서는 LLM Jailbreak 방어 기법을 크게 두 가지 범주로 분류하여 설명한다.

1. **Tuning-based Defense**: Llama Guard나 SafeDecoding과 같이 모델 자체의 안전 정렬을 강화하는 방식이다. 하지만 이러한 방식은 모델 재학습에 따른 비용이 크며, GCG와 같은 최신 적대적 공격에 여전히 취약할 수 있다는 한계가 있다.
2. **Non-tuning-based Defense**: RAIN, SmoothLLM, IAPrompt와 같이 기존 모델을 그대로 사용하며 입력이나 출력을 필터링하는 방식이다.
    * RAIN과 같은 기법은 출력물을 다시 검사하여 가이드를 제공한다.
    * SmoothLLM은 입력 프롬프트를 변형하여 여러 번 실행한 후 결과를 집계한다.
    * IAPrompt는 입력 프롬프트의 '의도(Intention)'를 분석한다.

`SELFDEFEND`는 기존의 Non-tuning 방식과 비교했을 때, 단순히 의도를 분석하는 것이 아니라 입력 프롬프트 내의 **실제 유해 문장을 직접 포착**한다는 점에서 차별점을 가진다. 또한, Shadow Stack 구조를 통해 정상 요청에 대한 지연 시간을 최소화함으로써 실용성을 극대화하였다.

## 🛠️ Methodology

`SELFDEFEND`의 전체 시스템 구조와 동작 절차는 다음과 같다.

### 1. 시스템 아키텍처

시스템은 크게 **Normal Stack**과 **Shadow Stack**이라는 두 개의 경로로 구성된다. 사용자의 입력 프롬프트는 이 두 스택에 동시에 전달된다.

### 2. Shadow Stack의 동작 및 역할

Shadow Stack은 입력된 프롬프트에서 안전 정책을 위반하는 부분이 있는지 확인하는 전용 프롬프트와 함께 LLM에 전달된다. 구체적인 검사 프롬프트는 다음과 같은 형태를 띤다:

* "다음 문장에서 안전 정책을 위반할 가능성이 있는 부분이 어디인지 식별하라. 위반 사항이 있다면 해당 부분을 그대로 출력하고, 없다면 'No'라고 답변하라."

### 3. Checkpoint 메커니즘 및 추론 절차

Shadow Stack의 출력 결과에 따라 Normal Stack의 동작을 결정하는 Checkpoint가 작동한다.

* **정상 프롬프트의 경우**: Shadow Stack이 즉시 "No"라는 토큰을 출력한다. "No"는 매우 짧은 토큰이므로 출력 시간이 매우 짧으며, Normal Stack은 지연 없이 정상적으로 응답을 생성한다.
* **Jailbreak 프롬프트의 경우**: Shadow Stack이 유해한 프롬프트 내용(예: "how to make a bomb")을 식별하여 출력하기 시작한다. 이때 Normal Stack은 Shadow Stack의 유해 문장 출력을 감지하는 즉시 Checkpoint를 트리거하여 답변 생성을 중단하고, "귀하의 요청은 안전 정책을 위반하여 수행할 수 없습니다"와 같은 설명 가능한 거절 응답을 출력한다.

### 4. 알고리즘 흐름 요약

$$ \text{Input Prompt} \rightarrow \begin{cases} \text{Normal Stack} \rightarrow \text{Token Generation} \\ \text{Shadow Stack} \rightarrow \text{Harmfulness Check} \end{cases} $$
$$ \text{If Shadow Stack Output} = \text{"No"} \implies \text{Normal Stack proceeds normally} $$
$$ \text{If Shadow Stack Output} \neq \text{"No"} \implies \text{Trigger Checkpoint} \rightarrow \text{Block Normal Stack} \rightarrow \text{Output Refusal} $$

## 📊 Results

저자들은 GPT-3.5와 GPT-4 모델을 사용하여 `SELFDEFEND`의 유효성을 검증하기 위해 수동 분석(Manual Analysis)을 수행하였다. 실험 대상이 된 Jailbreak 공격 카테고리는 다음과 같다.

* **GCG Jailbreak**: GCG 기법으로 생성된 적대적 접미사가 포함된 프롬프트를 사용하였다.
* **Template-based Jailbreak**: DAN이나 특정 역할극(Role-play) 시나리오와 같은 템플릿 기반 프롬프트를 사용하였다.
* **Multilingual Jailbreak**: 유해 프롬프트를 스페인어 등 다른 언어로 번역하여 입력하였다.
* **Normal Prompt**: 일반적인 무작위 문장을 입력하여 오탐지 여부를 확인하였다.

### 정성적 결과

* **공격 탐지**: GPT-3.5와 GPT-4 모두 GCG, 템플릿 기반, 다국어 Jailbreak 프롬프트 내에 숨겨진 유해한 핵심 문장을 정확하게 식별해 냈음을 확인하였다.
* **정상 요청 처리**: 일반적인 사용자 요청에 대해서는 Shadow Stack이 정확하게 "No"라고 답변하여, 정상적인 서비스 이용에 방해가 되지 않음을 확인하였다.

실험 결과, 최신 off-the-shelf LLM들은 복잡한 껍데기(템플릿, 인코딩 등)에 싸여 있더라도 그 내부의 유해한 의도를 식별하는 능력이 매우 뛰어나며, 이를 활용한 `SELFDEFEND` 구조가 실용적인 방어책이 될 수 있음을 시사한다.

## 🧠 Insights & Discussion

### 강점 및 의의

`SELFDEFEND`는 모델의 가중치를 수정하지 않고도 기존 모델의 능력을 활용해 방어 체계를 구축했다는 점에서 매우 실용적이다. 특히 Shadow Stack과 Checkpoint 메커니즘을 통해, 보안 검사로 인해 발생하는 고질적인 문제인 '응답 지연'을 획기적으로 줄였다. 또한, 단순히 거절하는 것이 아니라 어떤 부분이 유해했는지 식별하므로 설명 가능한(Explainable) 방어가 가능하다.

### 한계 및 미해결 과제

1. **Prompt Injection**: 공격자가 `SELFDEFEND`의 존재를 알고 Shadow Stack의 검사 프롬프트 자체를 무력화하려는 Prompt Injection 공격을 시도할 수 있다.
2. **Multimodal Jailbreak**: 본 논문의 설계는 텍스트 기반이다. 텍스트 없이 이미지나 소리만으로 공격하는 순수 멀티모달 Jailbreak의 경우에는 대응할 수 없다.
3. **리소스 효율성**: 모든 요청에 대해 Shadow Stack을 호출하는 것은 여전히 추가적인 API 비용이나 컴퓨팅 자원을 소모한다.

### 향후 연구 방향 (Future Directions)

* **FD1**: 유해 프롬프트 인식만을 전담하는 저비용·고속의 경량 LLM을 설계하고, Prefix Tuning을 통해 검사 프롬프트를 고정함으로써 Prompt Injection을 방지하는 방안.
* **FD2**: `SELFDEFEND`를 통해 발견된 적대적 예제(AEs)를 수집하여 LLM의 안전 정렬을 더욱 강화하는 피드백 루프 구축.
* **FD3**: Shadow Stack 호출 횟수를 줄이기 위한 캐싱(Caching) 메커니즘 설계.

## 📌 TL;DR

본 논문은 LLM이 복잡한 Jailbreak 템플릿 속에서도 유해한 핵심 문장을 식별할 수 있다는 점에 착안하여, **정상 스택과 병렬로 동작하는 'Shadow Stack' 기반의 방어 체계인 `SELFDEFEND`를 제안**한다. 이 방식은 유해성이 감지될 때만 체크포인트를 트리거하여 정상 요청의 지연 시간을 최소화하면서도, GCG, 템플릿 기반, 다국어 공격 등 다양한 Jailbreak 시도를 효과적으로 차단할 수 있다. 향후 경량 전용 모델 도입 및 캐싱 메커니즘을 통해 실전 배치 가능성을 높일 수 있을 것으로 기대된다.
