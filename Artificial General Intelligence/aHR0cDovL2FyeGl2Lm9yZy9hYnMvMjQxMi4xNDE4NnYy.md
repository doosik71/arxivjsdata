# Towards AI-45° Law: A Roadmap to Trustworthy AGI

Chao Yang, Chaochao Lu, Yingchun Wang, Bowen Zhou (2024)

## 🧩 Problem to Solve

본 논문은 인공 일반 지능(Artificial General Intelligence, AGI)의 발전 과정에서 AI의 능력(Capability)과 안전성(Safety) 사이의 심각한 불균형 문제를 해결하고자 한다. 현재 AI 기술은 Scaling Laws와 모델 아키텍처의 혁신으로 인해 급격하게 발전하고 있으나, 이에 대응하는 안전성 확보 조치는 매우 느리게 진행되고 있다.

기존의 AI 안전 조치들은 모델이 이미 개발된 후 취약점이 발견되었을 때 대응하는 '반응적 접근 방식(Reactive Approach)'에 의존하며, 특정 도메인에 국한된 파편화된 형태로 제공된다는 한계가 있다. 저자들은 이러한 상태를 '불구의 AI(Crippled AI)'라고 정의하며, 강력한 능력을 갖추었음에도 불구하고 안전 체계가 부족하여 잠재적으로 치명적인 위험을 초래할 수 있는 현재의 상황을 지적한다. 따라서 본 논문의 목표는 AI의 능력과 안전성이 균형 있게 발전할 수 있도록 하는 가이드라인인 $\text{AI-45}^\circ \text{ Law}$를 제안하고, 이를 실현하기 위한 구체적인 기술적 프레임워크인 '신뢰 가능한 AGI의 인과 사다리(Causal Ladder of Trustworthy AGI)'를 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 AI의 능력과 안전성의 동기화된 발전을 강조하는 이론적 토대와 이를 구현하기 위한 계층적 기술 구조를 제안한 점에 있다.

첫째, $\text{AI-45}^\circ \text{ Law}$라는 설계 원칙을 도입하였다. 이는 능력과 안전성이 좌표계 상에서 $45^\circ$ 직선을 따라 동일한 속도로 발전해야 한다는 직관을 제공한다. 이를 통해 실존적 위험을 의미하는 'Red Line'과 조기 경보 지표인 'Yellow Line'을 설정하여 위험 관리의 기준을 제시하였다.

둘째, Judea Pearl의 '인과 관계의 사다리(Ladder of Causation)'에서 영감을 얻어 $\text{Causal Ladder of Trustworthy AGI}$ 프레임워크를 제안하였다. 이는 Approximate Alignment, Intervenable, Reflectable의 세 가지 계층으로 구성되며, 단순한 데이터 상관관계 학습에서 시작해 외부 개입 가능성, 그리고 최종적으로는 자기 성찰적 추론으로 나아가는 기술적 경로를 정의한다.

셋째, AGI의 신뢰성을 다섯 가지 수준(지각, 추론, 의사결정, 자율성, 협업 신뢰성)으로 체계화한 $\text{Matrix of Trustworthy AGI}$를 구축하여, 현재의 모델들이 어느 단계에 위치하며 향후 어떤 방향으로 발전해야 하는지를 명시하였다.

## 📎 Related Works

논문은 기존의 AI 안전 조치들이 가진 한계를 지적하며 관련 연구를 검토한다. Red Teaming, Watermarking, Safety Guardrails와 같은 기법들이 현재 사용되고 있으나, 이는 특정 시나리오의 취약점만 발견하거나 쉽게 우회될 수 있으며, 사후적으로 적용되는 Guardrails의 경우 시스템의 유연성과 사용성을 저하시키는 문제가 있다.

특히 기존의 적대적 방어(Adversarial Defense) 기술들이 이미지 인식과 같은 특정 작업에 매몰되어 있어, 자연어 처리나 음성 인식과 같은 다른 도메인으로 확장하기 어렵다는 점을 언급한다. 본 논문은 이러한 파편화된 접근 방식과 달리, 인과 관계라는 수학적/논리적 기초 위에 안전성을 통합하려는 시도를 통해 기존 연구들과 차별점을 둔다.

## 🛠️ Methodology

### 1. AI-45° Law
$\text{AI-45}^\circ \text{ Law}$는 AI의 능력(Capability)과 안전성(Safety)이 평행하게 발전해야 한다는 원칙이다.
- **Red Line**: 자가 복제 및 개선, 권력 추구, 무기 개발 지원, 사이버 공격, 기만 등 인류에게 돌이킬 수 없는 재앙을 초래할 수 있는 실존적 위험 영역을 의미한다.
- **Yellow Line**: 시스템이 Red Line에 진입하기 전, 능력이 위험 수준에 도달했음을 알리는 조기 경보 임계치이다.

### 2. Causal Ladder of Trustworthy AGI
본 프레임워크는 내재적 신뢰성(Endogenous Trustworthiness)과 외재적 신뢰성(Exogenous Trustworthiness)을 모두 포함하며, 다음의 세 계층으로 구성된다.

**계층 1: Approximate Alignment Layer (근사 정렬 계층)**
인과 사다리의 '연관(Association)' 단계에 해당하며, 관찰 데이터 기반의 상관관계를 학습하여 "무엇인가(What is it)"라는 질문에 답하는 단계이다.
- **Supervised Fine-Tuning (SFT)**: 인간의 가치와 일치하는 고품질 데이터를 통해 모델을 정렬한다.
- **Machine Unlearning**: 개인정보나 오류 데이터의 영향력을 제거하여 데이터 유출 및 프라이버시 문제를 해결한다.

**계층 2: Intervenable Layer (개입 가능 계층)**
인과 사다리의 '개입(Intervention)' 단계에 해당하며, "X에 개입하면 어떤 일이 벌어지는가(What will happen if X is intervened on)"를 다룬다.
- **Learning from X Feedback**: RLHF(인간 피드백 기반 강화학습)나 RLAIF(AI 피드백 기반 강화학습)를 통해 모델의 출력을 가이드한다.
- **Mechanistic Interpretability**: 모델 내부의 뉴런 가중치나 특징에 개입하여 동작을 분석하고 안전 성능을 확인한다.
- **Controllable Generation**: 명시적/암묵적 지침을 통해 유해하지 않은 콘텐츠 생성을 제어한다.

**계층 3: Reflectable Layer (성찰 가능 계층)**
인과 사다리의 '반사실(Counterfactual)' 단계에 해당하며, "다른 선택을 했다면 어떤 결과가 나왔을까(What would have happened if a different choice had been made)"를 추론하는 단계이다.
- **Value Reflection**: 자신의 행동과 선택에 대해 깊이 숙고하여 인간의 가치에 최적화한다.
- **World Models (Mental Models)**: 외부 환경과의 상호작용을 수학적으로 표현하여 의사결정의 하류 효과를 예측하고 반사실적 추론을 수행한다.
- **Counterfactual Interpretability**: 인과적 귀속 및 매개 분석을 통해 의사결정의 근본 원인을 이해한다.

### 3. Matrix of Trustworthy AGI (신뢰성 수준)
신뢰성의 수준을 다음의 5단계로 정의하며, 단계가 올라갈수록 Reflectable Layer에 대한 의존도가 높아진다.
1. **Perception Trustworthiness (지각 신뢰성)**: 감각 데이터 수집 및 해석의 정확성과 편향 없음.
2. **Reasoning Trustworthiness (추론 신뢰성)**: 논리적, 인과적 추론 과정의 투명성과 검증 가능성.
3. **Decision-making Trustworthiness (의사결정 신뢰성)**: 물리적 세계와의 상호작용 시 가치 정렬 및 상황 인지 능력.
4. **Autonomy Trustworthiness (자율성 신뢰성)**: 자율 운용 중 윤리적 원칙 준수 및 자기 규제 능력.
5. **Collaboration Trustworthiness (협업 신뢰성)**: 다중 에이전트 및 인간-AI 협업 시의 투명성과 합의 도달 능력.

## 📊 Results

본 논문은 특정 알고리즘의 성능을 측정하는 실험 논문이 아닌 포지션 페이퍼(Position Paper)이므로, 정량적인 수치 결과보다는 기존 모델들을 제안한 매트릭스에 배치한 정성적 분석 결과를 제시한다.

- **기초 모델 (Llama 2/3, Qwen, InternLM 등)**: 주로 SFT와 RLHF를 사용하므로 $\text{Approximate Alignment Layer}$와 $\text{Intervenable Layer}$에 위치하며, 주로 $\text{Perception Trustworthiness}$ 수준에 머물러 있다고 분석한다.
- **OpenAI-o1 (preview)**: 추론 과정에서 정책 검토 및 안전성 검증을 도입했으므로, $\text{Reflectable Layer}$의 초기 형태이자 $\text{Reasoning Trustworthiness}$ 단계에 진입한 것으로 평가한다.

또한, $\text{AI-45}^\circ \text{ Law}$를 적용했을 때, 현재의 AI 발전 궤적은 능력의 증가 속도가 안전성의 증가 속도를 훨씬 앞지르고 있어 $\text{Red Line}$ 영역으로 향하고 있다는 위험성을 경고한다.

## 🧠 Insights & Discussion

본 논문은 AI 안전성을 단순한 '필터링'이나 '가드레일'의 문제가 아니라, 인과 추론이라는 인지적 능력의 확장 문제로 접근했다는 점에서 매우 강력한 통찰을 제공한다. 특히 Judea Pearl의 인과 사다리를 AGI 안전성에 접목하여, 단순 상관관계 학습 $\rightarrow$ 개입 $\rightarrow$ 반사실적 성찰로 이어지는 구체적인 기술적 로드맵을 제시한 점이 돋보인다.

다만, 몇 가지 한계와 논의 사항이 존재한다.
첫째, $\text{AI-45}^\circ \text{ Law}$에서 말하는 '능력'과 '안전성'을 정량적으로 어떻게 측정하여 $45^\circ$라는 수치를 정의할 것인지에 대한 구체적인 메트릭(Metric)이 명시되지 않았다.
둘째, $\text{Reflectable Layer}$에서 제안하는 '세계 모델(World Models)'이나 '가치 성찰'이 실제 LLM 아키텍처에서 어떻게 구현될 수 있는지에 대한 구체적인 아키텍처 설계안보다는 개념적 제안에 그치고 있다.

그럼에도 불구하고, 본 연구는 AGI로 가는 길목에서 안전성을 부수적인 요소가 아닌 핵심 능력으로 정의함으로써, 향후 연구자들이 집중해야 할 기술적 방향성을 명확히 제시하였다.

## 📌 TL;DR

본 논문은 AI의 능력과 안전성이 동일한 속도로 발전해야 한다는 $\text{AI-45}^\circ \text{ Law}$를 제안하고, 이를 달성하기 위해 '근사 정렬 $\rightarrow$ 개입 $\rightarrow$ 성찰'로 이어지는 $\text{Causal Ladder of Trustworthy AGI}$ 프레임워크를 제시한다. 또한 AGI의 신뢰성을 지각부터 협업까지 5단계로 체계화하여, 단순한 성능 향상을 넘어 인간의 가치와 일치하는 안전한 AGI를 구축하기 위한 기술적/거버넌스적 로드맵을 제공한다. 이 연구는 특히 단순한 데이터 학습을 넘어선 '성찰적 추론' 능력이 AGI 안전성의 핵심임을 시사한다.