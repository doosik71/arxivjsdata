# A Formal Approach For Modelling And Analysing Surgical Procedures

Ioana Sandu, Rita Borgo, Prokar Dasgupta, Ramesh Thurairaja, and Luca Viganò (2024)

## 🧩 Problem to Solve

수술 절차는 일반적으로 고유하고 명확한 방식으로 정의된 '표준화'된 형태가 아니며, 많은 경우 집도의와 수술 팀원들의 머릿속에 있는 암묵적 지식(implicit knowledge)에 의존하여 수행된다. 이러한 의존성은 수술 전 계획 단계와 수술 중의 효과적인 소통 과정에서도 동일하게 나타난다. 특히 로봇 보조 수술(Robot-Assisted Surgery, RAS)의 경우, 장비로 인한 공간적 요구사항 증가와 콘솔 집도의와 환자 및 팀원 간의 물리적 분리로 인해 대인 간의 신호 전달이 방해받고 오해소통(miscommunication)이 발생할 가능성이 높다.

이러한 암묵적 지식에 대한 의존과 소통의 오류는 환자의 안전을 위협하는 심각한 위험 요소가 될 수 있다. 기존의 수술 프로세스 모델(Surgical Process Models, SPMs)이 제안되었으나, 이러한 모델들은 주로 절차의 표현에 집중했을 뿐, 실제 절차 내에서 발생할 수 있는 오류나 속성에 대해 논리적으로 추론(reasoning)하는 단계까지는 나아가지 못했다. 따라서 본 논문의 목표는 수술 절차를 정형화된 모델로 구축하고, 이를 자동화된 방식으로 분석하여 잠재적인 위험 요소와 속성 위반을 식별하는 방법론을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 수술 절차를 사이버 보안 분야의 **Security Ceremony(보안 세레모니)**로 간주하여 모델링하는 것이다. Security Ceremony는 컴퓨터 노드뿐만 아니라 인간 노드를 포함하며, 인간-컴퓨터 및 인간-인간 간의 소통, 물리적 객체의 전달 등을 포괄하는 개념이다.

수술 절차를 Security Ceremony로 모델링함으로써 얻는 주요 기여는 다음과 같다:

1. **정형 모델링**: 수술 절차를 Message Sequence Chart(MSC)로 표현하여 개념적 명확성을 제공하고 공유 가능한 형태로 만든다.
2. **오류의 정형화(Mutation)**: 인간 팀원이 저지를 수 있는 실수(예: 단계 누락, 잘못된 메시지 전달)를 **Mutation(변이)**이라는 개념으로 정의하여 정형적으로 분석할 수 있게 한다.
3. **자동화된 분석**: UPPAAL과 같은 정형 검증 도구를 사용하여, 정의된 수술 속성이 모든 가능한 실행 경로(trace)에서 유지되는지 검증하고, 위반 사례가 발생할 경우 이를 'Attack Trace' 형태로 추출하여 분석한다.

## 📎 Related Works

기존의 수술 프로세스 모델(SPMs)은 워크플로우 관리 및 컴퓨터 과학 개념을 도입하여 수술 활동의 네트워크를 단순화하거나 정형/반정형 형태로 표현해 왔다. 하지만 이러한 방식은 절차를 시각화하고 전달하는 데 치중되어 있으며, 특정 속성이 충족되는지 혹은 어떤 상황에서 오류가 발생하는지에 대한 논리적 추론 기능이 부족하다는 한계가 있다.

본 연구는 이를 해결하기 위해 사이버 보안의 Security Protocol 및 Security Ceremony 분석 기법을 도입한다. 기존의 보안 프로토콜 분석이 암호화 기술을 통한 데이터의 기밀성과 무결성에 집중했다면, 본 논문은 이를 확장하여 인간의 실수(human mistakes)가 전체 프로세스의 안전성(safety)에 미치는 영향을 분석하는 데 적용함으로써 차별점을 둔다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조

본 논문은 수술 절차를 다음과 같은 단계로 정형화한다:
`수술 절차 분석 $\rightarrow$ MSC 설계 $\rightarrow$ Role-script 정의 $\rightarrow$ State Transition Rule 구축 $\rightarrow$ Mutation 적용 $\rightarrow$ UPPAAL을 통한 속성 검증`

### 주요 구성 요소 및 역할

1. **Message Sequence Chart (MSC)**: 집도의($S$), 보조의($A$), 간호사($N$)와 같은 에이전트들이 주고받는 메시지와 수행하는 동작을 시간 순서대로 시각화한 도표이다.
2. **Algebra of Messages**: 에이전트들이 주고받는 메시지의 집합을 $T_\Sigma(V)$로 정의하며, 여기에는 상수(constants)와 변수(variables)가 포함된다.
3. **Role-script**: 각 에이전트가 수행하는 이벤트의 시퀀스이다. 이벤트 집합은 $\text{RoleActions} = \{Snd, Rcv, s\_action, Start\}$로 구성된다.
    - $Snd, Rcv$: 메시지 전송 및 수신.
    - $s\_action(Ag, a^{Ag}, O)$: 에이전트 $Ag$가 객체 $O$에 대해 동작 $a^{Ag}$를 수행함.
    - $Start$: 프로세스의 시작.

### 실행 모델 및 전이 규칙

상태(State)는 에이전트들이 가진 지식(Knowledge)의 멀티셋으로 정의되며, 전이 규칙(State Transition Rules)은 다음과 같은 형식으로 작동한다:
$$\text{premise (current state)} \xrightarrow{\text{internal checks}} \text{conclusion (new state)}$$
에이전트는 메시지를 수신하고 $\text{Check}(?X=m)$과 같은 가드(guard) 조건을 만족하면, 정해진 수술 동작을 수행하고 다음 상태로 전이한다.

### Mutation (변이) 모델링

인간의 실수를 모델링하기 위해 두 가지 Mutation을 도입한다:

- **Skip Mutation**: 에이전트가 수행해야 할 특정 동작이나 메시지 전송을 생략하는 경우이다.
- **Replace Mutation**: 동작은 수행하지만 메시지를 다른 내용으로 대체하여 전송하는 경우이다. 특히 본 논문에서는 '부정 메시지(negative message, $\text{not\_m}$)' 개념을 도입하여, 작업이 완료되지 않았음을 알리는 잘못된 신호를 모델링한다.

**Matching Mutation**: Mutation이 발생했을 때 시스템이 단순히 데드락(deadlock)에 빠지지 않고 끝까지 실행되어 속성 위반 여부를 확인할 수 있도록, 이전 단계의 Mutation에 대응하여 다음 에이전트가 동작을 조정하는 '매칭 변이'를 함께 정의한다.

### 분석 목표 및 속성 (Properties)

Linear Temporal Logic(LTL)을 사용하여 환자의 안전을 보장하는 속성을 정의한다. 예를 들어, '절단 전 클립 적용' 속성(Property 1)은 다음과 같이 정의된다:
$$\text{s\_action(S, cut)} @l \implies \text{s\_action(S, request, clips)} @i \ \& \ \text{s\_action(N, provide, clips)} @j \ \& \ \text{s\_action(A, apply, clips)} @k \ \& \ i < j < k < l$$
이는 집도의가 절단($cut$)을 수행하기 전 반드시 클립 요청 $\rightarrow$ 제공 $\rightarrow$ 적용 단계가 순차적으로 완료되어야 함을 의미한다.

## 📊 Results

### 실험 설정

- **대상 데이터셋**: 복강경 전립선 절제술(Laparoscopic Prostatectomy)의 두 단계인 **Cutting Stage**와 **Lateral Dissection Stage**를 대상으로 한다.
- **분석 도구**: UPPAAL (실시간 시스템 모델링 및 검증 도구).
- **검증 지표**: 정의된 4가지 핵심 안전 속성(Property 1~4)의 만족 여부.

### 주요 결과

1. **정상 경로 검증**: Mutation이 비활성화된 상태에서는 모든 속성이 만족되며, 기대되는 순서대로 수술이 진행됨을 확인하였다.
2. **Mutation 활성화 시 위반 사례 발견**: Mutation을 허용했을 때, UPPAAL은 속성을 위반하는 **Attack Trace(공격 경로)**를 생성하였다.
    - **Property 1 위반**: 소통 오류나 태만으로 인해 클립이 적용되지 않은 상태에서 집도의가 절단을 수행하는 경로가 발견되었다.
    - **Property 2, 3, 4 위반**: 측면 박리 단계에서 신경혈관다발(NVB) 확인이나 조직 박리 순서가 지켜지지 않은 채 다음 단계로 넘어가는 경로들이 식별되었다.

### 결과의 의미

식별된 위반 경로 중 일부는 실제 임상에서 위험한 실수일 수 있으며, 일부는 상황에 따른 '합리적인 지름길(legitimate shortcut)'일 수 있다. 이는 본 모델이 단순히 오류를 찾는 것을 넘어, 임상의가 수술 절차의 변형(variant)을 검토하고 안전성을 논의하는 도구로 활용될 수 있음을 시사한다.

## 🧠 Insights & Discussion

### 강점

본 연구는 모호한 수술 절차를 사이버 보안의 정형 방법론(Security Ceremony)과 결합하여, 인간의 실수를 수학적으로 모델링하고 자동 검증했다는 점에서 매우 혁신적이다. 특히 Mutation과 Matching Mutation의 개념을 통해 단순한 오류 발생을 넘어, 그 오류가 전체 프로세스에 어떻게 전파되어 최종적으로 환자의 안전(속성 위반)을 위협하는지를 추적할 수 있게 하였다.

### 한계 및 미해결 질문

1. **상태 공간 폭발(State Space Explosion)**: 전체 수술 과정을 한 번에 모델링할 경우 UPPAAL이 처리할 수 있는 상태 수가 너무 많아진다. 본 논문에서는 이를 피하기 위해 각 단계를 독립적으로 분석하였으나, 전체 과정을 통합 분석하기 위해서는 추상화(abstraction) 또는 최적화 기법이 필요하다.
2. **임상적 해석의 필요성**: 도구가 출력한 'Attack Trace'가 실제 의료 현장에서 항상 위험한 것은 아니므로, 최종 판단은 반드시 숙련된 전문의의 해석을 거쳐야 한다.

### 비판적 해석

수술 절차를 보안 세레모니로 치환한 접근법은 매우 효율적이지만, 현재의 모델은 메시지를 단순 상수로 처리하고 있다. 실제 수술실에서의 소통은 훨씬 더 복잡하고 비정형적인 자연어로 이루어지므로, 향후 연구에서는 이러한 비정형 소통을 어떻게 정형 모델의 메시지로 매핑할 것인가에 대한 고민이 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 수술 절차를 **Security Ceremony**로 모델링하여 인간의 실수(Mutation)가 환자 안전에 미치는 영향을 자동 분석하는 프레임워크를 제안한다. UPPAAL을 이용해 복강경 전립선 절제술의 안전 속성을 검증한 결과, 특정 실수 조합이 치명적인 안전 위반(Attack Trace)으로 이어짐을 정형적으로 입증하였다. 이 연구는 향후 로봇 보조 수술의 안전 가이드라인 제정 및 원격 수술(Telesurgery)의 보안 통신 분석에 중요한 기반이 될 것으로 기대된다.
