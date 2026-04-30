# Risk assessment at AGI companies: A review of popular risk assessment techniques from other safety-critical industries

Leonie Koessler, Jonas Schuett (2023)

## 🧩 Problem to Solve

본 논문은 OpenAI, Google DeepMind, Anthropic과 같은 AGI(Artificial General Intelligence) 개발 기업들이 직면한 **치명적 위험(Catastrophic Risks)**을 관리하기 위한 체계적인 위험 평가 방법론의 부재 문제를 해결하고자 한다. AGI는 광범위한 인지 작업에서 인간과 대등하거나 그 이상의 성능을 발휘하는 시스템을 목표로 하지만, 이는 인류 멸종이나 문명 붕괴와 같은 실존적 위험(Existential Risks)을 초래할 가능성이 있다.

현재 AGI 기업들은 모델 평가(evals)나 레드팀(red teaming)과 같은 AI 특화 기술을 사용하고 있으나, 이는 전체적인 위험 관리 관점에서 불충분하다. 따라서 본 연구의 목표는 금융, 항공, 원자력, 생물학 실험실 등 이미 안전이 매우 중요한 **안전 필수 산업(Safety-critical industries)**에서 검증된 위험 평가 기법들을 검토하고, 이를 AGI 개발 환경에 어떻게 적용할 수 있을지에 대한 가이드를 제공하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 AGI의 특수한 위험 성격(낮은 확률과 극도로 높은 영향력)을 고려하여, 타 산업의 수많은 위험 평가 기법 중 AGI 기업에 실질적으로 유용한 **10가지 핵심 기법을 선별하고 체계화**했다는 점이다.

중심적인 설계 아이디어는 위험 관리의 세 단계인 **위험 식별(Risk Identification) $\rightarrow$ 위험 분석(Risk Analysis) $\rightarrow$ 위험 평가(Risk Evaluation)** 흐름에 따라 기법을 분류하고, 이를 AI 시스템 생애 주기(사전 훈련, 훈련, 배포, 모니터링)에 맞게 배치하는 프레임워크를 제시한 것이다. 단순히 기법을 나열하는 것에 그치지 않고, AGI 기업이 고려해야 할 구체적인 적용 시나리오와 한계점을 함께 논의함으로써 실행 가능한 권고안을 도출하였다.

## 📎 Related Works

논문은 기존의 AI 위험 관련 연구들을 다음과 같이 분류하여 설명한다.

1.  **기존 위험 식별 노력**: AI 발전 궤적과 위험에 대한 전문가 설문조사, 실존적 위험에 대한 위험 유형론(Risk Typologies) 및 분류 체계(Taxonomies) 연구들이 존재한다. 또한 FMEA(Failure Modes and Effects Analysis)나 HAZOP(Hazard and Operability Study) 같은 기법이 일부 AI 위험 분석에 적용된 사례가 있다.
2.  **기존 위험 분석 및 평가**: 영향 다이어그램(Influence Diagrams), 결함 나무 분석(Fault Tree Analysis, FTA), 사건 나무 분석(Event Tree Analysis, ETA) 등이 초지능(Superintelligence) 탈취 시나리오 분석에 사용되었다. 또한, 연구 프로젝트의 실존적 위험 영향을 평가하기 위한 체크리스트 등이 제안된 바 있다.

**기존 접근 방식과의 차별점**:
기존 연구들은 주로 단일 기법을 적용하거나 특정 시나리오(예: 초지능의 반란)에 집중하는 경향이 있었다. 반면, 본 논문은 ISO 31000:2018 및 IEC 31010:2019와 같은 국제 표준을 기반으로 광범위한 기법을 검토하였으며, 단순한 선형적 인과관계(FTA/ETA)를 넘어 시스템적 상호작용을 고려하는 STPA(System-Theoretic Process Analysis)와 같은 복잡한 기법의 필요성을 강조한다.

## 🛠️ Methodology

본 연구는 타 산업의 기법을 선별하기 위해 다음과 같은 제외 및 우선순위 기준을 적용하였다.

- **제외 기준**: 비즈니스 위험에만 국한된 기법, 꼬리 위험(Tail risks)을 무시하는 기법, 단순 반복 업무의 인간 신뢰성 평가 기법, 하드웨어 신뢰성 전용 기법은 제외하였다.
- **우선순위 기준**: 사건 간의 복잡한 상호작용 분석 가능 여부, 다양한 이해관계자의 관점 통합 가능 여부, 미래 전개 상황에 대한 명확성 제공 여부, 정량적 기법보다는 정성적 기법을 우선하였다.

선별된 10가지 기법의 상세 내용은 다음과 같다.

### 1. 위험 식별 (Risk Identification)
- **Scenario Analysis**: 미래에 발생 가능한 여러 시나리오를 구축하고 그로 인한 위험을 분석하는 전향적 추론(Forward reasoning) 방식이다. AGI 시장의 경쟁 구도나 지정학적 상황을 시나리오화하여 위험을 예측할 수 있다.
- **Fishbone Method (Ishikawa Analysis)**: 특정 위험(결과)으로부터 그 원인을 역추적하는 후향적 추론(Backward reasoning) 방식이다. "왜?"라는 질문을 반복하여 원인-부원인 구조를 시각화한다.
- **Risk Typologies and Taxonomies**: 위험의 전체 집합(Risk universe)을 개념적 또는 경험적으로 분류하는 체계이다. 사각지대를 없애고 조직 내 공통된 위험 이해를 돕는다.

### 2. 위험 분석 (Risk Analysis)
- **Causal Mapping**: 사건 간의 인과관계를 화살표로 연결하여 맵으로 그리는 방식이다. 피드백 루프나 복잡한 상호의존성을 파악하는 데 유리하다.
- **Delphi Technique**: 전문가 그룹에게 익명으로 설문을 반복하고 결과를 공유하며 합의를 도출하는 예측 방법이다. AGI의 위험 발생 확률을 추정할 때 유용하다.
- **Cross-impact Analysis**: 사건 간의 상관관계를 분석하는 복잡한 기법이다. 전문가의 조건부 확률 예측을 바탕으로 몬테카를로 시뮬레이션 등을 통해 일관성 있는 미래 시나리오를 생성한다.
- **Bow Tie Analysis**: 위험의 원인(왼쪽)과 결과(오른쪽)를 배치하고, 그 사이에 예방 통제(Preventive controls)와 대응 통제(Reactive controls)를 배치하여 '나비넥타이' 모양으로 시각화한다.
- **System-Theoretic Process Analysis (STPA)**: 시스템을 구성 요소의 집합이 아닌 '제어 루프'로 파악한다. 안전 제약 조건(Safety constraints)을 정의하고, 제어 조치가 부적절하게 수행되는 '안전하지 않은 제어 동작(Unsafe Control Actions, UCAs)'을 식별하여 그 원인을 분석한다.

### 3. 위험 평가 (Risk Evaluation)
- **Checklists**: 사전 정의된 상황에서 위험 여부를 판단하기 위한 질문지이다. 위험 평가를 분산화하고 조직 내 안전 문화를 확산시키는 데 효과적이다.
- **Risk Matrices**: 위험의 영향도(Consequence)와 발생 가능성(Likelihood) 또는 취약성(Vulnerability)을 축으로 하여 위험의 우선순위를 결정하는 매트릭스이다.

## 📊 Results

본 논문은 실험적 수치 결과 대신, 선정된 기법들을 AI 시스템 생애 주기에 어떻게 적용해야 하는지에 대한 **운용 프레임워크(Exemplary use case)**를 결과물로 제시한다.

- **사전 훈련 단계 (Pre-training)**: 시나리오 분석, Fishbone, STPA, Delphi 기법 등을 통해 잠재적 위험과 통제 방안을 식별한다.
- **훈련 단계 (Training)**: 지속적인 업데이트와 모니터링을 수행한다.
- **배포 전 단계 (Pre-deployment)**: 체크리스트, 위험 매트릭스, Cross-impact analysis 등을 통해 최종 배포 여부와 우선순위를 결정한다.
- **배포 및 모니터링 단계 (Deployment & Monitoring)**: 실제 발생 상황을 모니터링하고 다시 위험 식별 단계로 피드백한다.

특히 저자들은 위험 매트릭스 작성 시, AGI의 특성상 발생 가능성(Likelihood)을 예측하기 매우 어려우므로, 이를 **취약성(Vulnerability)** 지표로 대체하여 평가할 것을 권고한다. 이때 취약성은 Bow Tie 분석이나 STPA를 통해 도출된 통제 기법의 유효성을 기반으로 측정한다.

## 🧠 Insights & Discussion

**강점 및 시사점**:
본 연구는 AGI 안전이라는 모호한 영역을 ISO/IEC와 같은 산업 표준의 위험 관리 프레임워크로 끌어들였다는 점에서 큰 의미가 있다. 특히 AI 모델 내부의 정렬(Alignment) 문제뿐만 아니라, 기업의 의사결정 구조, 경쟁적 역학 관계, 인적 오류와 같은 **시스템적 관점의 위험 관리** 필요성을 강조한 점이 돋보인다.

**한계 및 비판적 해석**:
1.  **불충분성**: 저자 스스로 언급했듯이, 타 산업의 기법만으로는 AGI의 특수한 위험을 완전히 평가할 수 없다. 모델의 위험한 능력(Dangerous capabilities)을 직접 측정하는 AI 특화 평가(Evals)와 병행되어야 한다.
2.  **데이터의 부재**: 많은 기법이 과거의 사고 데이터나 경험적 근거를 바탕으로 하지만, AGI의 치명적 사고는 전례가 없는 사건(Unprecedented events)이므로 전문가의 주관적 판단에 의존할 수밖에 없는 한계가 있다.
3.  **실행의 어려움**: STPA나 Cross-impact analysis 같은 기법은 학습 곡선이 매우 가파르고 많은 시간이 소요되어, 빠른 개발 속도를 중시하는 AGI 기업 문화에서 실제로 채택될 수 있을지에 대한 의문이 남는다.

## 📌 TL;DR

본 논문은 AGI 기업들이 직면한 치명적 위험을 관리하기 위해 항공, 원자력, 금융 등 안전 필수 산업에서 사용하는 **10가지 위험 평가 기법(식별 3, 분석 5, 평가 2)**을 제안하고 이를 AI 생애 주기에 맞게 체계화하였다. 단순한 모델 평가를 넘어 시스템적 관점의 위험 관리를 도입함으로써, AGI 개발 과정에서 발생할 수 있는 실존적 위험을 보다 구조적으로 파악하고 통제할 수 있는 방법론적 토대를 제공한다. 이 연구는 향후 AGI 안전 규제 프레임워크 구축 및 기업 내 안전 문화 정착에 중요한 기초 자료로 활용될 가능성이 높다.