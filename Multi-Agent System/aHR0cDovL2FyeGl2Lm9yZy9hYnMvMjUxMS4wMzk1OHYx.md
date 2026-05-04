# Multi-Agent Collaborative Framework For Math Problem Generation

Kia Karbasi, Kevin Hong, Mohammad Amin Samadi, Gregory Pottie (2025)

## 🧩 Problem to Solve

본 논문은 지능형 튜터링 시스템(Intelligent Tutoring Systems, ITS) 및 교육자를 위한 수학 문제 자동 생성(Automatic Question Generation, AQG)의 한계를 해결하고자 한다. 기존의 사전 학습된 트랜스포머 기반 언어 모델들은 자연어 생성 능력은 뛰어나지만, 교육적으로 매우 중요한 요소인 문제의 복잡도(Complexity)와 인지적 요구량(Cognitive Demands)을 정밀하게 제어하는 데 어려움을 겪는다.

수학 교육에서 학습자의 현재 지식 상태에 맞춘 적절한 난이도의 문제를 동적으로 제공하는 것은 개인화된 학습 경로를 구축하는 데 필수적이다. 따라서 본 연구의 목표는 추론 시간 연산(Inference Time Computation, ITC)을 AQG에 도입하여, 문제의 질과 난이도 제어 능력을 향상시킨 협력적 멀티 에이전트 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 단일 모델의 생성 능력에 의존하는 대신, 여러 에이전트가 상호작용하며 생성된 문제-정답 쌍을 반복적으로 정제하는 협력적 워크플로우를 구축하는 것이다. 주요 기여 사항은 다음과 같다.

1.  **AQG를 위한 두 가지 협력적 프레임워크 제안**: Teacher-Critic Cycle(TCC)과 Collective Consensus(CC)라는 서로 다른 구조의 멀티 에이전트 시스템을 설계하였다.
2.  **Bloom의 교육 목표 분류학 기반 Self-Curation 방법론 개발**: Bloom's Taxonomy의 인지적 요구량 기준을 적용하여 생성된 문제 중 교육적 가치가 높은 문제만을 선별하는 Bloom Agent를 도입하였다.
3.  **자동화된 수학 문제 평가 프레임워크 구축**: 관련성(Relevance), 중요도(Importance), 명확성(Clarity), 난이도 일치도(Difficulty Matching), 답변 가능성(Answerability)의 5가지 지표를 통해 생성된 문제의 품질을 정량적으로 평가하는 체계를 마련하였다.

## 📎 Related Works

기존의 AQG 연구들은 주로 특정 주제에 맞게 사전 학습된 모델을 미세 조정(Fine-tuning)하거나, Bloom의 분류학을 활용한 프롬프트 엔지니어링, Chain-of-Thought(CoT) 추론, 또는 방정식 제약 조건을 활용한 문장제 문제 생성 방식 등을 사용해 왔다. 그러나 이러한 방식들은 여전히 정답의 부재와 인간 평가의 주관성이라는 자연어 생성(NLG)의 근본적인 문제에 직면해 있으며, 의도한 난이도와 실제 생성된 문제 간의 불일치 문제가 지속적으로 제기되었다.

본 논문은 이러한 한계를 극복하기 위해 최근 LLM 분야에서 주목받는 추론 시간 연산(ITC) 및 Agentic Workflows를 도입한다. 특히 에이전트 간의 토론(Debate)이 사실 관계 확인 및 추론 능력을 향상시킨다는 점에 착안하여, 이를 수학 문제 생성 과정에 적용함으로써 기존의 단일 모델 생성 방식과 차별화를 꾀한다.

## 🛠️ Methodology

### 전체 시스템 구조 및 에이전트 정의
본 시스템은 주어진 지식 구성 요소(Knowledge Component, KC) 이름, 예시 문제 집합, 그리고 목표 난이도(Easy, Medium, Hard)를 입력으로 받아 문제와 정답을 출력한다. 이를 위해 다음과 같은 역할을 가진 에이전트들을 정의한다.

*   **Teacher**: 특정 KC와 난이도에 맞는 수학 문제와 정답을 생성한다.
*   **Generic Critic**: 생성된 문제의 명확성, 관련성, 난이도 일치도를 객관적으로 평가하고 피드백을 제공한다.
*   **Consensus CEO**: 대화 기록을 검토하여 최종적으로 가장 적절한 문제-정답 쌍을 선택하는 의사결정자이다.
*   **Versatile Agent**: 상황에 따라 새로운 문제를 생성하거나, 기존 문제를 수정하거나, 동료의 의견에 찬성하며 피드백을 주는 유연한 역할을 수행한다.

### 멀티 에이전트 워크플로우
논문에서는 두 가지 주요 워크플로우를 제안한다.

1.  **Teacher-Critic Cycle (TCC)**: 
    - $\text{Teacher} \rightarrow \text{Generic Critic} \rightarrow \text{Teacher}$ 순으로 이어지는 반복적 피드백 루프이다.
    - Critic이 문제의 교육적 적절성을 평가하면, Teacher가 이를 반영하여 문제를 수정한다. 이 과정은 2~5회 반복된다.
    - Auto Chain-of-Thought(AutoCoT)와 명시적 솔루션 생성(Solution Generation) 기법을 적용하여 추론 과정을 정밀화한다.

2.  **Collective Consensus (CC)**:
    - 여러 명의 Versatile Agent들이 참여하는 협력적 토론 구조이다.
    - 한 에이전트가 문제를 생성하면, 다른 에이전트들이 순차적으로 수정 제안을 하거나 동의하며 합의점을 찾아간다.
    - 최종적으로 Consensus CEO가 전체 논의 과정을 검토하여 최적의 결과물을 선정한다.

### 난이도 프롬프팅 및 Self-Curation
*   **Prompting Strategies**: 실제 학생들의 정답률 데이터를 기반으로 한 $\text{Empirical}$ (전체 난이도 예시 제공), $\text{Prompting Empirical}$ (목표 난이도 예시만 제공), $\text{Prompting Simple}$ (무작위 예시 제공)의 세 가지 전략을 비교 실험하였다.
*   **Self-Curation**: Bloom Agent가 생성된 문제에 대해 인지적 요구량 점수(1~5점)를 부여한다. 이 점수가 설정된 기준에 미달하는 문제는 과감히 폐기함으로써, 단순히 텍스트가 유사한 문제가 아닌 교육적으로 깊이 있는 문제만을 남긴다.

## 📊 Results

### 실험 설정
*   **데이터셋**: ASSISTments 데이터셋의 확장판인 Problem Bodies를 사용하였으며, 실제 학생들의 정답률(Percent Correct)을 기준으로 난이도를 구분하였다.
*   **평가 지표**: GPT-4를 평가자로 활용하여 관련성, 중요도, 명확성, 난이도 일치도, 답변 가능성을 1~5점 척도로 측정하였다.

### 주요 결과
1.  **워크플로우 성능 비교**: 
    - curated된 TCC와 CC 방식이 Baseline(Zero-shot, Few-shot) 및 non-curated 에이전트 방식보다 모든 지표에서 우수한 성능을 보였다.
    - 특히 **Difficulty Matching**과 **Relevance**에서 가장 큰 향상이 관찰되었다. 이는 반복적인 비판과 정제 과정이 의도한 인지적 난이도를 맞추는 데 효과적임을 시사한다.
2.  **난이도별 성능 추이**:
    - 문제의 목표 난이도가 높아질수록(Easy $\rightarrow$ Hard), 난이도 일치도와 평균 점수가 하락하는 경향을 보였다. 이는 LLM이 고난도 문제의 인지적 복잡성을 유지하며 생성하는 것에 여전히 어려움을 겪고 있음을 보여준다.
3.  **추론 비용과 성능**:
    - 에이전트의 수나 토론 라운드 수를 단순히 늘리는 것이 성능 향상으로 직결되지 않았으며, 특정 시점 이후로는 수익 체감(Diminishing Returns) 현상이 나타났다.
4.  **프롬프팅 전략의 영향**: 
    - 예상과 달리 Few-shot 프롬프팅 전략 간의 성능 차이는 미미하였으며, 일부 케이스에서는 Baseline Zero-Shot이 Few-Shot보다 난이도 일치도가 높게 나타나는 현상이 발견되었다.

## 🧠 Insights & Discussion

### 강점 및 성과
본 연구는 단순한 텍스트 생성을 넘어, 에이전트 간의 협력과 Bloom의 교육 분류학 기반의 필터링을 결합하여 교육적으로 가치 있는 수학 문제를 생성할 수 있음을 입증하였다. 특히 ITC(Inference Time Computation)를 통해 모델의 파라미터 업데이트 없이도 추론 단계의 구조적 설계만으로 결과물의 품질을 높였다는 점이 고무적이다.

### 한계 및 비판적 해석
1.  **평가 지표의 신뢰성**: 모든 평가가 GPT-4에 의해 자동화되어 수행되었다. 논문에서도 언급되었듯이, LLM 기반 평가의 **천장 효과(Ceiling Effect)**로 인해 시스템 간의 미세한 성능 차이가 가려졌을 가능성이 크다.
2.  **난이도 제어의 어려움**: 고난도 문제 생성 시 성능이 하락하는 점은, 현재의 멀티 에이전트 구조만으로는 복잡한 수학적 추론의 '난이도'를 정밀하게 정의하고 생성하는 데 한계가 있음을 의미한다.
3.  **Few-shot의 비효율성**: 다양한 Few-shot 전략이 큰 효과를 보이지 못한 점은 AQG 작업에서 단순 예시 제공보다 더 정교한 In-context Learning 기법이나 도메인 특화 가이드라인이 필요함을 시사한다.

## 📌 TL;DR

본 논문은 수학 문제 자동 생성(AQG)에서 난이도와 인지적 요구량을 정밀하게 제어하기 위해 **Teacher-Critic Cycle(TCC)** 및 **Collective Consensus(CC)**라는 멀티 에이전트 협력 프레임워크를 제안하였다. Bloom의 교육 목표 분류학을 이용한 Self-Curation을 통해 생성된 문제의 교육적 품질을 높였으며, 실험 결과 고난도 문제 생성의 어려움은 여전히 존재하지만, 단일 모델 기반 생성보다 난이도 일치도와 관련성 면에서 우수한 성능을 보였다. 이 연구는 향후 지능형 튜터링 시스템(ITS)에서 개인화된 교육 콘텐츠를 자동 생성하는 핵심 기술로 활용될 가능성이 높다.