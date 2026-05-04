# Multi-Turn Human–LLM Interaction Through the Lens of a Two-Way Intelligibility Protocol

Harshvardhan Mestha, Karan Bania, Shreyas Vinaya Sathyanarayana, Sidong Liu, Ashwin Srinivasan (2025)

## 🧩 Problem to Solve

본 연구는 인간 전문가와 대규모 언어 모델(LLM)이 자연어를 통해 상호작용하며 데이터 분석 작업을 수행하는 소프트웨어 시스템의 설계 문제를 다룬다. 복잡한 문제 해결 과정에서 LLM은 인간의 전문성과 창의성을 활용하여 단독으로는 찾기 어려운 해결책을 도출할 수 있는 잠재력을 가지고 있다.

그러나 기존의 설명 가능한 AI(XAI)는 주로 기계가 생성한 예측의 근거를 제공하는 일방향적 설명에 치중되어 있으며, 사용자의 이해도나 수준에 맞춘 맞춤형 설명을 제공하지 못하는 한계가 있다. 즉, 단순히 '읽을 수 있는(readable)' 설명을 제공하는 것이 곧 '이해 가능한(intelligible)' 소통을 의미하는 것은 아니다. 따라서 본 논문은 인간과 LLM 사이의 상호 이해도를 정량적으로 측정하고 관리할 수 있는 구조화된 프로토콜을 통해 상호 운용성을 높이는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 기존의 이론적 연구인 PXP(Predict and Explain) 프로토콜을 실제 구현하여 LLM과 인간 간의 상호작용에 적용하고 그 효용성을 실증적으로 분석했다는 점이다.

중심적인 아이디어는 '양방향 이해 가능성(Two-way Intelligibility)'이라는 개념을 도입하는 것이다. 이를 위해 상호작용하는 에이전트를 유한 상태 기계(Finite-State Machine)로 모델링하고, 교환되는 메시지에 네 가지 태그(Ratification, Refutation, Revision, Rejection)를 부여하여 소통의 질을 추적한다. 이를 통해 단순한 성능 향상을 넘어, 인간과 기계가 서로의 예측과 설명을 얼마나 이해하고 수용하는지를 체계적으로 분석할 수 있는 프레임워크를 제공한다.

## 📎 Related Works

논문은 먼저 Michie의 머신러닝 분류 체계를 언급한다. Michie는 ML 시스템을 성능 향상에만 집중하는 'Weak ML', 학습 내용을 인간이 이해할 수 있는 형태로 전달하는 'Strong ML', 그리고 인간의 성능까지 향상시키는 'Ultra-strong ML'로 구분하였다.

또한, Chain-of-Thought(CoT)나 Tree-of-Thought(ToT)와 같은 반복적 프롬프팅(Iterative Prompting) 연구들이 존재하지만, 본 연구는 LLM의 프롬프팅 기법을 개선하는 것이 아니라, 인간과 LLM 간의 다회차 상호작용 자체를 '이해 가능성(Intelligibility)'의 관점에서 분석하고 해석하는 방법론을 제안한다는 점에서 차별점을 갖는다. 기존 XAI가 사용자 맞춤형 설명의 부재로 인해 신뢰도를 떨어뜨리는 문제를 해결하기 위해, 본 연구는 메시지 태그 기반의 상호 확인 절차를 제안한다.

## 🛠️ Methodology

### 1. PXP 프로토콜 및 PEX 에이전트

본 연구는 예측과 설명을 수행하는 PEX(Predict and Explain) 에이전트 간의 상호작용 모델인 PXP 프로토콜을 사용한다. 에이전트 $a_n$은 상대 에이전트 $a_m$으로부터 받은 예측 $y_m$과 설명 $e_m$을 자신의 예측 $y_n$ 및 설명 $e_n$과 비교하여 메시지 태그를 결정한다.

태그 결정 로직은 다음과 같다:

- **RATIFY**: 예측과 설명 모두 일치할 때 전송한다.
- **REVISE**: 예측 또는 설명 중 하나만 일치하며, 자신의 내부 모델을 수정할 수 있을 때 전송한다.
- **REFUTE**: 예측 또는 설명 중 하나만 일치하지만, 모델 수정이 불가능할 때 전송한다.
- **REJECT**: 예측과 설명 모두 일치하지 않을 때 전송한다.

### 2. 이해 가능성의 정의

상호작용 세션 $S$에서 전송된 태그 시퀀스를 $T_{mn}$(m에서 n으로), $T_{nm}$(n에서 m으로)라고 할 때, 이해 가능성은 다음과 같이 정의된다.

- **One-Way Intelligibility**: 시퀀스에 $\text{RATIFY}$ 또는 $\text{REVISE}$가 최소 하나 포함되어 있고, $\text{REJECT}$가 전혀 없을 때 성립한다.
- **Two-Way Intelligibility**: 양측 모두에 대해 One-Way Intelligibility가 성립할 때이다.
- **Strong Intelligibility**: 모든 메시지 태그가 $\{\text{RATIFY}, \text{REVISE}\}$ 집합에 속할 때 성립한다.
- **Ultra-Strong Intelligibility**: Strong Intelligibility를 만족하면서, 최소 하나 이상의 $\text{REVISE}$ 태그가 존재할 때 성립한다.

### 3. 시스템 구현 (Blackboard System)

상호작용을 관리하기 위해 세 개의 테이블(Data, Message, Context)로 구성된 블랙보드 시스템을 구축하였다.

- **Data**: 세션 ID와 데이터 인스턴스의 쌍 $(s, x)$ 저장.
- **Message**: 송신자, 수신자, 메시지 내용(태그, 예측, 설명)을 포함하는 5-튜플 $(s, j, \alpha, \mu, \beta)$ 저장.
- **Context**: 도메인 특화 문맥 정보 $(s, j, c)$ 저장.

전체 프로세스는 $\text{INTERACT}$ 알고리즘에 따라 기계 에이전트가 먼저 예측과 설명을 제시하며 시작되며, 양측이 $\text{RATIFY}$에 도달하거나 메시지 제한 횟수 $n$에 도달할 때까지 교대로 메시지를 주고받는다. 기계 에이전트로는 Claude 3.5 Sonnet 모델을 사용하였다.

## 📊 Results

### 1. 실험 설정

- **대상 도메인**: X-ray 진단(RAD) 및 분자 합성 경로 설계(DRUG).
- **실험 종류**:
  - **Controlled**: 인간 전문가의 데이터베이스를 프록시로 사용하여 반복 실험 수행.
  - **Uncontrolled**: 실제 화학 전문가 2명(h1: 계산 화학 전문성 높음, h2: 화학 전문성 높음)이 참여.
- **측정 지표**: 이해 가능성(One-way, Two-way, Strong, Ultra-Strong)의 비율 및 기계의 성능 변화.

### 2. 정량적 결과

- **이해 가능성**: RAD와 DRUG 모두에서 One-way 및 Two-way Intelligibility의 비율이 매우 높게 나타났다. 특히 상호작용 횟수가 증가할수록 인간 에이전트 입장에서의 이해 가능성 비율이 상승하는 경향을 보였다.
- **기계 성능**: 상호작용이 진행됨에 따라 기계 에이전트의 예측 성능이 향상되었다. 이는 상호작용 과정에서 축적된 문맥(Context) 정보가 LLM의 생성 능력에 긍정적인 영향을 주었기 때문으로 분석된다.
- **특이 사항**: Strong Intelligibility의 경우, 대부분 상호작용 시작 즉시 양측이 합의하는 '퇴행적(degenerate)' 케이스($\langle \text{INIT}_m, \text{RATIFY}_h, \text{RATIFY}_m \rangle$)에서 발생했으며, 긴 토론 끝에 도달하는 진정한 의미의 Strong Intelligibility는 드물게 나타났다.

### 3. Uncontrolled 실험 결과

실제 인간 전문가와의 실험에서도 통제 실험과 유사한 경향이 관찰되었다. 흥미로운 점은 화학적 전문성이 높은 전문가보다 계산적 전문성(Computational expertise)이 높은 전문가가 더 높은 이해 가능성 비율을 보였다는 점이다. 이는 LLM의 출력 특성에 따른 우연한 결과일 수 있으나, LLM의 응답이 도메인 지식뿐만 아니라 모델의 출력 방식에 따라 이해도가 달라질 수 있음을 시사한다.

## 🧠 Insights & Discussion

본 연구는 단순한 가독성(Readability)과 진정한 의미의 이해 가능성(Intelligibility) 사이의 간극을 명확히 보여준다. LLM이 자연어로 설명을 제공할 수 있지만, 그것이 반드시 인간의 지식 구조와 일치하거나 상호 수용 가능한 형태는 아님을 시사한다.

**강점 및 시사점**:

- PXP 프로토콜을 통해 인간-AI 간의 소통 품질을 정량적으로 측정할 수 있는 방법론을 제시하였다.
- LLM이 인간의 텍스트 피드백을 통해 자신의 예측을 수정(REVISE)하고 성능을 개선할 수 있음을 실증하였다.

**한계 및 논의**:

- **Strong Intelligibility의 부재**: 대부분의 성공적인 세션이 즉각적인 합의로 끝난다는 점은, LLM이 매우 정교한 추론 과정을 거쳐 인간을 설득하거나 혹은 인간이 LLM의 초기 답변을 그대로 수용하는 경향이 있음을 의미한다. 심도 있는 상호 교정 과정이 일어나는 세션이 적다는 점은 향후 연구 과제이다.
- **학습 함수(LEARN function)의 영향**: 통제 실험에서는 검증 셋의 성능이 향상될 때만 예측을 수정하도록 제한했으나, 실제 인간과의 상호작용에서는 이러한 제약이 없다. 이로 인해 기계가 인간의 응답을 '이해'했다고 판단함에도 불구하고 실제 성능은 떨어지는 역설적인 상황이 발생할 수 있다.

## 📌 TL;DR

본 논문은 인간 전문가와 LLM 간의 다회차 상호작용에서 '양방향 이해 가능성(Two-way Intelligibility)'을 측정하기 위해 PXP 프로토콜을 구현하고 검증하였다. 의료 진단 및 분자 합성 도메인 실험 결과, 구조화된 메시지 태그($\text{RATIFY}, \text{REVISE}$ 등)를 통한 상호작용이 기계의 예측 성능을 높이고 소통의 질을 정량화할 수 있음을 확인하였다. 이 연구는 향후 인간과 AI가 협력하는 전문 시스템 설계 시, 단순한 성능 지표가 아닌 '상호 이해도'를 기반으로 한 설계 프레임워크의 중요성을 제시한다.
