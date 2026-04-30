# Agentic Reasoning for Large Language Models

Tianxin Wei et al. (2026)

## 🧩 Problem to Solve

본 논문은 기존 거대 언어 모델(Large Language Models, LLMs)이 가진 추론 능력의 근본적인 한계를 해결하고자 한다. 기존의 LLM 추론은 주로 정적인 입력값에 대해 단발성(one-shot) 또는 소수 사례(few-shot) 기반의 예측을 수행하는 패시브(passive)한 방식에 의존해 왔다. 이러한 방식은 수학적 문제 해결이나 코드 생성과 같은 폐쇄적 환경(closed-world settings)에서는 강력한 성능을 보이지만, 정보가 시간에 따라 변하고 환경과의 상호작용이 필수적인 개방형 및 동적 환경(open-ended and dynamic environments)에서는 한계를 드러낸다.

특히, 기존의 Chain-of-Thought(CoT)와 같은 기법들은 추론 과정을 명시화하여 성능을 높였으나, 여전히 정적인 컨텍스트와 짧은 호라이즌(short-horizon)의 추론을 가정한다는 문제가 있다. 따라서 본 논문의 목표는 LLM을 단순히 텍스트를 생성하는 모델이 아니라, 환경과 지속적으로 상호작용하며 계획(plan), 행동(act), 학습(learn)하는 자율적 에이전트로 재정의하는 '에이전틱 추론(Agentic Reasoning)'의 체계적인 로드맵을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 LLM의 추론을 '상호작용'의 관점에서 바라보고, 이를 세 가지 상호 보완적인 계층(Layer)으로 구조화하여 에이전틱 추론의 패러다임을 정의한 것이다.

1.  **에이전틱 추론의 3계층 구조 제안**:
    *   **기초 에이전틱 추론(Foundational Agentic Reasoning)**: 단일 에이전트의 핵심 역량인 계획, 도구 사용(Tool Use), 탐색(Search)을 다룬다.
    *   **자기 진화적 에이전틱 추론(Self-evolving Agentic Reasoning)**: 피드백, 메모리, 적응 메커니즘을 통해 에이전트가 경험으로부터 스스로 능력을 개선하는 과정을 다룬다.
    *   **집단적 다중 에이전트 추론(Collective Multi-agent Reasoning)**: 여러 에이전트가 역할을 분담하고 협력하여 개별 에이전트의 한계를 넘어서는 집단 지능을 다룬다.

2.  **최적화 설정의 구분**: 추론 능력을 향상시키는 방법을 파라미터 수정 없이 추론 시간의 연산량을 늘리는 '인컨텍스트 추론(In-context Reasoning)'과, 강화학습(RL) 및 지도 미세 조정(SFT)을 통해 모델 가중치를 직접 최적화하는 '포스트 트레이닝 추론(Post-training Reasoning)'으로 명확히 구분하여 분석하였다.

3.  **통합적인 로드맵 제공**: 이론적 정의부터 방법론, 실제 도메인 적용 사례(과학, 로봇, 의료 등), 그리고 이를 평가하기 위한 벤치마크까지를 포괄하는 통합적인 분석 보고서를 제공한다.

## 📎 Related Works

본 논문은 기존의 LLM 추론 관련 연구와 AI 에이전트 관련 연구의 접점에 위치하며, 다음과 같은 차별점을 갖는다.

*   **LLM 추론 연구와의 차이**: 기존 연구들은 주로 모델 내부의 계산 과정(예: CoT, Prompt Engineering)을 통해 어떻게 하면 더 나은 추론 결과물을 낼 것인가에 집중하였다. 반면, 본 논문은 추론이 텍스트 생성을 넘어 동적 계획, 적응형 메모리, 피드백 기반 행동으로 확장되는 '시스템 수준의 지능'에 집중한다.
*   **AI 에이전트 연구와의 차이**: 기존 에이전트 연구들은 주로 아키텍처나 시스템 설계(예: RL 기반 결정, 도구 사용 모듈)에 치중하는 경향이 있었다. 본 논문은 추론(Reasoning)을 에이전트 아키텍처의 단순한 부산물이 아니라, 단일 에이전트의 강화, 다중 에이전트의 협업, 자기 진화를 연결하는 '통합 메커니즘'으로 정의한다.

## 🛠️ Methodology

### 1. 에이전틱 추론의 정식화 (Formalization)
논문은 에이전틱 추론을 부분 관측 마르코프 결정 과정(Partially Observable Markov Decision Process, POMDP)으로 모델링한다. 에이전트의 정책 $\pi_\theta$는 내부적인 '생각(Thinking)' 과정과 외부적인 '행동(Acting)' 과정으로 분리된다.

$$ \pi_\theta(z_t, a_t | h_t) = \pi_{reason}(z_t | h_t) \cdot \pi_{exec}(a_t | h_t, z_t) $$

여기서 $z_t$는 내부 추론 트레이스(Reasoning Trace)를, $a_t$는 외부 행동(External Action)을 의미한다. 이는 에이전트가 행동을 취하기 전 $Z$ 공간에서 충분한 계산(생각)을 수행하는 'Think-then-Act' 구조를 수학적으로 명시한 것이다.

### 2. 최적화 모드 (Optimization Modes)
*   **인컨텍스트 추론 (In-context Reasoning)**: 모델 파라미터 $\theta$를 고정한 채, 추론 시간의 탐색(Search)을 통해 최적의 경로를 찾는다. Tree-of-Thoughts (ToT)나 MCTS 스타일의 접근 방식이 여기에 해당하며, 휴리스틱 가치 함수 $\hat{v}$를 최대화하는 궤적 $\tau$를 선택한다.
*   **포스트 트레이닝 추론 (Post-training Reasoning)**: RL을 통해 $\theta$를 직접 최적화한다. 특히 본 논문에서는 가치 네트워크 없이 그룹 상대적 보상을 사용하는 Group Relative Policy Optimization (GRPO)를 중요하게 다룬다. GRPO의 목적 함수는 다음과 같다.

$$ \mathcal{L}_{GRPO}(\theta) = \mathbb{E}_{q \sim P(Q)} \left[ \frac{1}{G} \sum_{i=1}^{G} \left( \min(\rho_i \hat{A}_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon)\hat{A}_i) - \beta D_{KL}(\pi_\theta \| \pi_{ref}) \right) \right] $$

여기서 $\hat{A}_i$는 그룹 내 상대적 이점(Advantage)으로, 단순한 절대 보상이 아닌 그룹 평균 대비 성능을 통해 학습 효율을 높인다.

### 3. 에이전틱 추론의 계층 구조
*   **Foundational Layer**: 계획(Planning), 도구 사용(Tool-use), 탐색(Search)을 통해 기본 능력을 구축한다.
*   **Self-evolving Layer**: 
    *   **피드백**: 자기 비판(Self-critique)과 리플렉션(Reflection)을 통해 추론 경로를 수정한다.
    *   **메모리**: 단순 텍스트 저장(Flat Memory)에서 그래프/계층 구조 메모리(Structured Memory)로 발전하며, RL을 통해 메모리 읽기/쓰기 정책을 최적화한다.
*   **Collective Layer**: 다중 에이전트 시스템(MAS)을 구축한다. 리더(Leader), 워커(Worker), 크리틱(Critic) 등의 역할 분담(Role Taxonomy)을 통해 복잡한 문제를 해결하며, 통신 프로토콜과 공유 메모리를 통해 협업한다.

## 📊 Results

논문은 제안한 프레임워크가 실제 다양한 도메인에서 어떻게 구현되고 평가되는지 상세히 분석하였다.

### 1. 도메인별 적용 사례
*   **수학 및 코드**: 단순 정답 맞추기에서 벗어나, 프로그램 탐색(Program Search)과 반복적인 디버깅 루프를 통해 새로운 수학적 정리를 발견하거나 복잡한 소프트웨어를 개발하는 에이전틱 워크플로우를 보여준다.
*   **과학적 발견**: 가설 생성 $\rightarrow$ 시뮬레이션/실험 $\rightarrow$ 결과 분석 $\rightarrow$ 가설 수정으로 이어지는 폐쇄 루프(Closed-loop) 시스템을 구축하여 신약 개발이나 재료 과학 연구를 자동화한다.
*   **로보틱스 (Embodied Agents)**: 언어적 계획을 물리적 행동(Action)으로 매핑하며, 환경 피드백을 통해 제어 정책을 실시간으로 수정하는 능력을 평가한다.
*   **의료 및 헬스케어**: 환자 기록(EHR) 분석, 가이드라인 준수, 다학제적 전문가 에이전트 간의 협의(Consensus) 과정을 통해 진단 정확도를 높인다.
*   **웹 탐색 및 연구**: 동적인 웹 환경에서 필요한 정보를 능동적으로 탐색하고, 다수의 소스에서 얻은 정보를 교차 검증하여 심층 보고서를 작성하는 능력을 보여준다.

### 2. 벤치마크 분석
논문은 평가 지표를 두 가지 차원으로 나누어 분석한다.
*   **메커니즘 중심 평가**: 도구 호출 정확도(Tool Use), 탐색 경로의 효율성(Search), 장기 기억 유지력(Memory), 협업 효율성(MAS) 등 개별 기능을 격리하여 측정한다.
*   **애플리케이션 중심 평가**: 실제 환경(예: OSWorld, WebArena, AgentClinic)에서 엔드-투-엔드(End-to-End) 성공률을 측정하여 실무 적용 가능성을 평가한다.

## 🧠 Insights & Discussion

### 강점 및 통찰
본 논문은 파편화되어 있던 'LLM 추론'과 '에이전트 설계'라는 두 분야를 '에이전틱 추론'이라는 하나의 통합된 패러다임으로 묶어냈다는 점에서 매우 높은 학술적 가치를 가진다. 특히 추론을 단순히 모델 내부의 연산이 아니라 $\text{생각} \rightarrow \text{행동} \rightarrow \text{피드백} \rightarrow \text{수정}$으로 이어지는 외적 루프로 확장한 관점이 돋보인다.

### 한계 및 미해결 과제
1.  **다중 에이전트 메모리의 최적화**: 단일 에이전트의 메모리 최적화 연구는 활발하지만, 여러 에이전트가 공유 메모리를 어떻게 효율적으로 업데이트하고 일관성을 유지할 것인지에 대한 포스트 트레이닝 연구는 여전히 부족하다.
2.  **잠재적 추론(Latent Reasoning)의 위험성**: 자연어 텍스트가 아닌 잠재 공간(Latent Space)에서 추론을 수행하려는 시도가 증가하고 있으나, 이는 추론 과정의 투명성과 해석 가능성(Interpretability)을 심각하게 훼손할 수 있다.
3.  **장기 호라이즌의 신용 할당 문제(Credit Assignment)**: 매우 긴 단계의 상호작용 후에 얻은 보상을 어떤 구체적인 추론 단계나 도구 호출 덕분인지 정확히 판별하여 학습시키는 것은 여전히 어려운 과제이다.

## 📌 TL;DR

본 논문은 LLM의 추론 패러다임을 정적인 텍스트 생성에서 동적인 상호작용 중심의 **'에이전틱 추론(Agentic Reasoning)'**으로 전환할 것을 제안한다. 이를 위해 **기초 역량 $\rightarrow$ 자기 진화 $\rightarrow$ 집단 협업**으로 이어지는 3계층 구조를 정의하고, 인컨텍스트 최적화와 포스트 트레이닝 최적화라는 두 가지 경로를 체계화하였다. 이 연구는 향후 LLM이 단순한 챗봇을 넘어, 스스로 도구를 사용하고 학습하며 협력하는 자율적 지능체(Autonomous Intelligence)로 진화하기 위한 설계 도면을 제공한다.