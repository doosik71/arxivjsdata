# Emergent Language: A Survey and Taxonomy

Jannik Peters, Constantin Waubert de Puiseau, Hasan Tercan, Arya Gopikrishnan, Gustavo Adolpho Lucas De Carvalho, Christian Bitter, Tobias Meisen (2024/2025)

## 🧩 Problem to Solve

본 논문은 인공지능, 특히 Multi-Agent Reinforcement Learning (MARL) 환경에서 에이전트들 사이에 자발적으로 형성되는 Emergent Language (EL) 연구 분야의 파편화 문제를 해결하고자 한다. EL 연구는 에이전트들이 협력을 위해 스스로 통신 프로토콜을 설계하고 학습하는 과정을 다루지만, 현재 이 분야는 다음과 같은 심각한 문제점에 직면해 있다.

첫째, 용어의 표준화가 부족하여 연구자마다 동일한 개념을 다르게 정의하거나 사용하는 Taxonomic Inconsistency가 빈번하게 발생한다. 둘째, 측정 지표(Metric)의 선택과 사용이 일관되지 않으며, 측정하고자 하는 언어적 특성과 실제 적용된 지표 사이에 괴리가 존재한다. 셋째, 기존의 서베이 논문들은 특정 설정(Setting)이나 방법론(Method)에만 치중되어 있어, 전체적인 체계와 정량적 분석 방법을 통합적으로 제시하는 가이드라인이 부족한 실정이다.

따라서 본 논문의 목표는 181편의 학술 논문을 체계적으로 분석하여, EL의 구성 요소에 대한 포괄적인 Taxonomy를 제안하고, 이를 기반으로 일관된 표기법(Notation)과 분류된 측정 지표 리스트를 제공함으로써 향후 연구의 비교 가능성과 해석 가능성을 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 EL 연구를 위한 통합적인 분석 프레임워크를 구축한 것에 있으며, 구체적인 내용은 다음과 같다.

1. **EL의 체계적 Taxonomy 제안**: 통신 설정(Setting), 언어 게임(Game), 언어 사전 지식(Prior), 그리고 언어적 특성(Characteristics)으로 이어지는 계층적 구조를 정의하였다. 특히 언어적 특성을 음성학(Phonetics)부터 화용론(Pragmatics)까지의 6단계 언어 구조 레벨에 맞추어 세분화하였다.
2. **통합 표기법 및 측정 지표의 분류**: $\Omega$(설정), $\Phi$(의미), $L$(언어)이라는 세 가지 공간을 중심으로 한 수학적 표기법을 도입하고, 이를 바탕으로 Morphology, Syntax, Semantics, Pragmatics의 각 단계에서 사용할 수 있는 정량적 지표들을 체계적으로 정리하였다.
3. **연구 공백(Research Gap) 식별**: 대규모 문헌 분석을 통해 Semantics 분야는 활발히 연구되고 있으나, Syntax 분야는 매우 소외되어 있다는 점과, EL과 Natural Language (NL) 간의 정렬(Alignment) 문제인 'Evolution-Acquisition Dilemma'라는 핵심 과제를 도출하였다.

## 📎 Related Works

논문은 EL 관련 기존 서베이들을 세 가지 범주로 나누어 설명하며 본 연구와의 차별점을 명시한다.

- **Settings 중심 서베이**: 언어 게임 패러다임이나 학습 환경의 설계에 집중한 연구들이다. 본 논문은 특정 게임 패러다임을 넘어 더 광범위한 접근 방식을 분석한다.
- **Methods 중심 서베이**: 학습 알고리즘이나 평가 방법론의 비판적 검토에 집중한다. 본 논문은 단순한 방법론 요약을 넘어, 이를 언어학적 레벨과 연결한 Taxonomy와 통합 지표 체계를 제공한다는 점에서 차별화된다.
- **General Overview 서베이**: 분야 전반에 대한 일반적인 소개를 제공하지만, 본 논문처럼 세밀한 정량적 측정 지표의 분류와 일관된 수학적 표기법을 제시하는 경우는 드물다.

결과적으로 본 논문은 단순한 문헌 요약이 아니라, 측정 가능성(Measurability)과 정량화(Quantification)에 초점을 맞춘 '지표 중심의 프레임워크'를 제공한다는 점에서 기존 연구들과 차별된다.

## 🛠️ Methodology

본 논문은 EL 시스템을 분석하기 위해 **Setting $\rightarrow$ Meaning $\rightarrow$ Language**로 이어지는 세미오틱 사이클(Semiotic Cycle) 기반의 프레임워크를 제안한다.

### 1. 시스템 구조 및 표기법 (Notation)

- **Setting Space ($\Omega$)**: 환경 $E$, 액션 $A$, 상태 $S$ 등을 포함하며, 에이전트 $\xi_i$가 관측값 $o_\xi$를 통해 환경과 상호작용하는 단계이다.
- **Meaning Space ($\Phi$)**: 원시 데이터 $\chi$를 의미 벡터 $\phi$로 변환하는 단계이다.
    - Conceptualization: $\Psi^{con} : \chi \to \Phi$
    - Interpretation: $\Psi^{int} : \Phi \to \chi$
- **Language Space ($L$)**: 메시지 $m \in M$을 생성하고 이해하는 단계이다.
    - Production: $L^{prod} : \chi \to M$
    - Comprehension: $L^{comp} : M \to \chi$

### 2. EL Taxonomy 구성

- **Communication Setting**: 에이전트 수(Single, Dual, Population), 협력 수준(Cooperative, Semi-cooperative, Competitive), 대칭성(Symmetry), 수신자 형태(Target, Broadcast)로 구분한다.
- **Language Games**: Referential Game(참조), Reconstruction Game(재구성), Question-Answer Game(문답), Grid World Game, Continuous World Game 등으로 분류한다.
- **Language Prior**: 사전 지식 없이 자발적으로 발생하는 Evolution-based 접근법과 NL의 구조를 모방하는 Acquisition-based 접근법으로 나눈다.
- **Language Characteristics**: 언어학의 6단계 구조를 따른다.
    - **Phonetics**: 통신 채널의 특성 (Discrete vs Continuous).
    - **Phonology**: 사용되는 어휘의 종류 (Binary, Token, NL, Sound, Picture).
    - **Morphology**: 단어 및 문장 구성 규칙 (Message Length, Compression, Redundancy).
    - **Syntax**: 문법적 구조 및 규칙.
    - **Semantics**: 문자 그대로의 의미 (Grounding, Compositionality, Consistency, Generalization).
    - **Pragmatics**: 맥락에 따른 언어 사용 (Predictability, Efficiency, Positive Signaling/Listening, Symmetry).

### 3. 주요 측정 지표 및 방정식

본 논문은 각 특성별로 핵심 지표를 정의하며, 대표적인 예시는 다음과 같다.

- **Topographic Similarity (Topsim)**: 의미 공간의 거리 $\Delta \Phi$와 메시지 공간의 거리 $\Delta L$ 사이의 상관관계를 측정하여 구성성(Compositionality)을 평가한다.
  $$\rho = \text{cov}(R(\Delta L(m_i)), R(\Delta \Phi(\phi_i))) / (\sigma_{R(\Delta L)} \sigma_{R(\Delta \Phi)})$$
- **Zero-shot Evaluation**: 학습 시 보지 못한 새로운 입력(Unseen input)이나 파트너(Unseen partner)에 대해 에이전트가 얼마나 잘 수행하는지를 통해 일반화(Generalization) 능력을 측정한다.
- **Causal Influence of Communication (CIC)**: 통신 채널이 있을 때와 없을 때의 리시버 액션 엔트로피 차이를 통해 통신의 실질적 효용성을 측정한다.
  $$\text{CIC}(\tau^\xi_R) = H(a^\xi_R | \tau^\xi_R) - H(a^\xi_R | \tau^{+M}_\xi_R)$$

## 📊 Results

본 논문은 181편의 논문을 분석한 결과 다음과 같은 정량적/정성적 경향성을 발견하였다.

- **게임 유형의 분포**: Referential Game이 가장 압도적으로 많이 사용되며, 그 뒤를 Reconstruction과 Grid World Game이 잇고 있다.
- **언어 특성별 연구 집중도**: 
    - **Semantics**: 가장 활발하게 연구되는 분야이며, 특히 Compositionality와 Generalization에 대한 관심이 높다.
    - **Morphology & Pragmatics**: 중간 수준의 연구가 진행되고 있으며, 주로 메시지 길이 및 효율성 분석에 치중되어 있다.
    - **Syntax**: 가장 소외된 분야로, 단 2편의 논문만이 구체적인 구문 분석 방법을 제시하였다.
- **출판 경향**: 2020년까지는 지속적으로 증가 추세였으나, 최근에는 LLM(Large Language Models)으로의 연구 관심 이동으로 인해 EL 전용 연구의 출판 수가 다소 감소하는 경향을 보인다.

## 🧠 Insights & Discussion

### 1. Evolution-Acquisition Dilemma
연구진은 EL 연구의 근본적인 딜레마를 제시한다. 에이전트가 자신의 환경에 최적화된 언어를 자발적으로 개발하게 하는 '진화(Evolution)' 과정과, 인간과의 상호작용을 위해 NL의 구조를 강제하는 '습득(Acquisition)' 과정 사이에는 강한 트레이드-오프가 존재한다는 것이다.

### 2. EL vs LLM: Grounding의 관점
현재의 LLM은 방대한 텍스트 데이터를 통한 통계적 모방에 의존하지만, EL은 구체적인 목표(Goal)와 환경적 경험(Experience)을 통해 언어를 형성한다. 이는 LLM이 겪고 있는 '깊은 이해의 부재'와 'Grounding 문제'를 해결할 수 있는 실마리가 될 수 있으며, LLM의 표현력과 EL의 경험적 학습을 결합한 'Agentic LLM' 또는 'Cognitive Language Agents'의 필요성을 시사한다.

### 3. 비판적 해석
본 논문은 EL의 정량적 평가 체계를 잘 정리하였으나, 정작 '무엇이 좋은 EL인가'에 대한 절대적인 기준은 여전히 부재함을 인정한다. 지표의 최댓값이나 최솟값이 항상 최적의 언어를 의미하는 것은 아니며, 시스템의 목적에 따른 적절한 균형점(Balance point)을 찾는 연구가 추가적으로 필요하다.

## 📌 TL;DR

본 논문은 파편화된 Emergent Language (EL) 연구 분야를 통합하기 위해 **[통신 설정 $\rightarrow$ 언어 게임 $\rightarrow$ 언어 사전 지식 $\rightarrow$ 언어 특성]**으로 이어지는 체계적인 **Taxonomy**와 **통합 측정 지표 프레임워크**를 제안하였다. 특히 언어학적 6단계 레벨을 AI 에이전트의 통신 분석에 도입하여, 그동안 간과되었던 구문(Syntax) 및 화용(Pragmatics) 분석의 중요성을 강조하였다. 이 연구는 향후 LLM의 한계인 Grounding 문제를 해결하고, 실제 인간-에이전트 상호작용(HCI)이 가능한 수준의 인공지능 통신 시스템을 구축하는 데 중요한 이론적 기초가 될 것으로 기대된다.