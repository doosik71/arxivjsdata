# Improving Arithmetic Reasoning Ability of Large Language Models through Relation Tuples, Verification and Dynamic Feedback

Zhongtao Miao, Kaiyan Zhao, Yoshimasa Tsuruoka

## 🧩 Problem to Solve

거대 언어 모델(LLM)의 추론 단계에 사용되는 현재 표현 방식에는 두 가지 주요 문제점이 있습니다. 첫째, 자연어 기반 추론(예: CoT)은 길어서 추론 비용이 높고 계산 오류나 논리적 비약이 있을 수 있으며, 자연어의 특성상 검증이 어렵습니다. 둘째, 비자연어 기반 추론(예: 프로그래밍 코드)은 자동 검증이 용이하지만, 코딩에 익숙하지 않은 사람들에게는 읽기 어렵습니다. 따라서 사람과 기계 모두에게 친숙하고, 이해하기 쉬우며, 검증이 용이한 새로운 형태의 추론 표현 방식이 필요합니다.

## ✨ Key Contributions

- **관계 튜플(Relation Tuples) 도입:** LLM 추론 단계에 세미 구조화된 표현 방식인 관계 튜플을 도입했습니다. 이는 자연어 추론 단계보다 짧고 읽기 쉬우며, 의사 코드와 유사하여 실제 프로그래밍 언어 코드로 쉽게 변환될 수 있어 기계 친화적입니다. 관계 튜플을 퓨샷(few-shot) 예시에 통합함으로써 7개 산술 데이터셋 중 4개에서 정확도 향상을 보였습니다.
- **자동 검증 프로세스 구현:** 관계 튜플 기반의 추론 단계를 검증하기 위해 로컬 코드 인터프리터를 활용한 자동 검증 프로세스를 개발했습니다. 이 로컬 코드 인터프리터는 오픈소스 여부와 관계없이 모든 LLM에 원활하게 통합될 수 있습니다.
- **동적 피드백 메커니즘 통합:** 간단하고 효과적인 동적 피드백 메커니즘을 구현했습니다. Self-Refine(Madaan et al., 2023)보다 훨씬 단순하지만, 필요한 경우에만 피드백을 제공하여 LLM의 자체 개선을 돕습니다.

## 📎 Related Works

이 연구는 LLM의 추론 능력을 향상시키기 위한 기존 연구들과 연관됩니다.

- **자연어 추론:** CoT(Wei et al., 2022), Zero-shot CoT(Kojima et al., 2022), Least-to-most prompting(Zhou et al., 2023), Self-Consistency(Wang et al., 2023), Tree-of-Thought(Yao et al., 2023), Graph-of-Thoughts(Besta et al., 2024), Buffer-of-Thoughts(Yang et al., 2024) 등 자연어 형태의 중간 추론 단계를 활용하는 방식들이 있습니다.
- **비자연어 추론 및 검증:** PAL(Gao et al., 2023)은 Python 코드를 중간 추론 단계로 사용하며, ERA-CoT(Liu et al., 2024)는 엔티티와 관계 분석을 활용합니다. GPT-4 코드 인터프리터(Zhou et al., 2024a) 및 MathCoder(Wang et al., 2024)는 코드 생성 및 실행을 통합하고, SymbolCoT(Xu et al., 2024b)는 기호 표현을, Zhou et al. (2024b)은 비공식 자연어를 형식적인 Isabelle 코드로 변환하여 추론을 검증합니다. 본 연구는 세미 구조화된 관계 튜플과 코드 생성 능력을 활용하여 추론을 검증한다는 점에서 차별화됩니다.
- **자체 개선 및 검증:** STaR(Zelikman et al., 2022)은 정확한 추론 경로를 통해 모델을 미세 조정하며, Self-Refine(Madaan et al., 2023)은 생성, 피드백 제공, 개선의 3단계로 구성됩니다. 본 연구의 동적 피드백은 필요한 경우에만 제공되는 점이 다릅니다.

## 🛠️ Methodology

제안된 ART(Arithmetic Reasoning Ability through Relation Tuples, Verification and Dynamic Feedback) 프레임워크는 다음 세 단계를 포함합니다.

- **Step 1: 관계 튜플을 사용한 추론 (Reasoning with relation tuples)**

  - 질문 $Q_{i}$가 주어지면, LLM은 자연어 문장과 그에 상응하는 관계 튜플($r_{i}, t_{i}$)을 포함하는 추론 과정 $\hat{R}_{i} = [(r_{0}, t_{0}), \dots, (r_{n-1}, t_{n-1})]$과 답변 $\hat{A}_{i}$를 생성합니다.
  - 이 단계에서는 자연어와 관계 튜플이 혼합된 8-샷(eight-shot) 인컨텍스트 학습 프롬프트(in-context learning prompt)가 사용됩니다.

- **Step 2: 관계 튜플과 로컬 코드 인터프리터를 사용한 자동 검증 (Automatic verification with relation triples and a local code interpreter)**

  - Step 1에서 생성된 추론 과정 $\hat{R}_{i}$에서 관계 튜플 목록 $T_{i} = [t_{0}, \dots, t_{n-1}]$를 추출합니다.
  - LLM은 질문 $Q_{i}$와 추출된 관계 튜플 $T_{i}$를 기반으로 Python 코드 솔루션 $C_{i} = \text{LM}(Q_{i}, T_{i})$를 생성합니다.
  - 생성된 Python 코드 $C_{i}$는 로컬 코드 인터프리터 $\text{LCI}$에 의해 실행되어 검증 답변 $\hat{A}_{v_{i}} = \text{LCI}(C_{i})$를 얻습니다.

- **Step 3: 일관성 확인 및 필요시 동적 피드백 제공 (Checking consistency and providing dynamic feedback when necessary)**
  - Step 1의 답변 $\hat{A}_{i}$와 Step 2의 검증 답변 $\hat{A}_{v_{i}}$를 비교합니다.
  - 두 답변이 일치하면 추론 과정에 계산 오류가 없다고 판단하고 최종 답변으로 확정합니다.
  - 두 답변이 불일치하는 경우, 이전 추론 단계 $\hat{R}_{i}$가 LLM에 피드백으로 다시 전송되어 새로운 추론 과정을 재생성하도록 유도합니다. 최대 3번의 재시도가 허용됩니다.
  - Self-Consistency(Wang et al., 2023) 방법과 통합하여, Step 1과 Step 2에서 얻은 모든 답변을 기록하고 가장 빈번한 답변을 최종 답변으로 선택합니다.

## 📊 Results

이 연구는 GSM8K, SVAMP, ASDIV, SingleOP, SingleEQ, AddSub, MultiArith 등 7가지 산술 데이터셋에서 ChatGPT (gpt-3.5-turbo-0301), GPT-4o, Llama3-8B-Instruct 모델을 사용하여 실험을 수행했습니다.

- **주요 결과:** ART 프레임워크는 ChatGPT (gpt-3.5-turbo-0301)에서 CoT, PAL, ModelSelection과 같은 기존 기준선보다 우수한 성능을 보였습니다. 특히 GSM8K 데이터셋에서 ModelSelection 대비 1.9%, SVAMP 데이터셋에서 2.8%의 정확도 향상을 달성했습니다.
- **관계 튜플의 역할:** 관계 튜플을 통합한 추론 과정은 7개 산술 데이터셋 중 4개에서 CoT보다 높은 정확도를 보였습니다. 이는 관계 튜플이 LLM이 다음 추론 단계를 생성하기 전에 "생각"하도록 유도하는 "일시 정지" 토큰 역할을 할 수 있음을 시사합니다.
- **프로그래밍 코드 검증의 역할:** Step 2의 검증 답변 정확도는 Step 1의 추론 답변 정확도보다 낮게 나타났으며, 특히 Llama3-8B-Instruct 모델에서 이러한 차이가 두드러졌습니다. 이는 LLM이 세미 구조화된 관계 튜플로부터 정확한 Python 코드를 생성하는 데 어려움을 겪을 수 있음을 나타냅니다. Llama3-8B-Instruct의 경우, `UnboundLocalError` 및 `SyntaxError`와 같은 실행 오류가 자주 관찰되었습니다.
- **피드백의 역할:** LLM의 코딩 능력이 향상될수록(Llama3-8B-Instruct < ChatGPT < GPT-4o), 피드백이 필요한 질문의 비율이 지속적으로 감소했습니다. 이는 동적 피드백 메커니즘이 덜 강력한 모델의 자체 개선에 특히 효과적임을 시사합니다.
- **Self-Consistency 통합:** ART 프레임워크는 Self-Consistency와 원활하게 통합될 수 있음을 보여주었으며, Self-Consistency(SC@5)를 통해 Llama3-8B-Instruct의 GSM8K 산술 추론 성능을 크게 향상시켰습니다.

## 🧠 Insights & Discussion

- **관계 튜플의 효과:** 관계 튜플은 자연어 추론과 정형 언어(프로그래밍 코드)를 연결하는 다리 역할을 하여, 추론 과정의 명확성과 검증 가능성을 높이는 데 기여합니다. 이는 LLM이 추론 과정에서 "생각"하는 중간 단계를 더 잘 구조화하도록 돕는 것으로 보입니다.
- **검증의 한계:** LLM의 프로그래밍 코드 이해 및 생성 능력에 검증 프로세스가 크게 의존한다는 점이 한계로 지적됩니다. 특히 Llama3-8B-Instruct와 같은 일부 모델은 세미 구조화된 관계 튜플에서 정확한 코드를 생성하는 데 어려움을 겪어 오류가 발생할 수 있습니다. 이는 LLM의 코드 생성 능력이 추론 품질에 중요한 영향을 미침을 보여줍니다.
- **동적 피드백의 가치:** 동적 피드백 메커니즘은 간단하지만 효과적으로 LLM의 자체 개선 능력을 이끌어냅니다. 모델의 내재적 능력에 따라 피드백의 필요성이 달라지는 흥미로운 현상도 관찰되었습니다.
- **효율성 문제:** 현재 방법은 자연어 문장과 세미 구조화된 관계 튜플의 혼합된 추론 과정을 사용하므로 추론 비용이 높습니다. 미래에는 관계 튜플만으로 추론을 수행하여 추론 비용을 줄이고 가독성을 유지하며 기계 처리를 용이하게 하는 것이 이상적일 것입니다.
- **추가 연구 가능성:** 관계 튜플 외에도 다른 검증 가능한 세미 구조화된 추론 형태가 존재할 수 있다는 점은 향후 연구의 방향을 제시합니다.

## 📌 TL;DR

- **문제:** LLM의 산술 추론은 자연어로는 검증이 어렵고, 코드는 사람이 읽기 어렵다는 한계가 있습니다.
- **방법:** 이 논문은 ART(Arithmetic Reasoning Ability through Relation Tuples, Verification and Dynamic Feedback) 프레임워크를 제안합니다. ART는 다음 세 가지 핵심 요소를 포함합니다: 1) 인간 친화적이면서 기계 친화적인 **관계 튜플**을 추론 단계에 도입하여 추론 과정을 명확히 합니다. 2) 관계 튜플을 기반으로 LLM이 생성한 코드를 **로컬 코드 인터프리터**로 자동 검증합니다. 3) LLM의 자체 개선을 유도하는 **동적 피드백 메커니즘**을 통합하여, 초기 추론과 검증 결과가 불일치할 경우 모델이 재추론하도록 합니다.
- **성과:** 이 방법은 다양한 산술 데이터셋에서 LLM의 정확도를 효과적으로 향상시켰습니다. 관계 튜플은 추론 정확도를 높이는 데 기여하며, 피드백 메커니즘은 특히 성능이 낮은 모델에서 효과적인 자체 개선을 가능하게 합니다. 또한, Self-Consistency 방법과 결합 시 성능이 더욱 향상될 수 있음을 보였습니다.
