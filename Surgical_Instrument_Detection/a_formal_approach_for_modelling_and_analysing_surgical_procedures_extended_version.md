# A Formal Approach For Modelling And Analysing Surgical Procedures

Ioana Sandu, Rita Borgo, Prokar Dasgupta, Ramesh Thurairaja, and Luca Viganò (2024)

## 🧩 Problem to Solve

수술 절차는 대개 표준화된 문서로 정의되기보다 외과 의사와 수술팀의 머릿속에 암묵적인 지식(implicit knowledge)의 형태로 존재한다. 이러한 특성은 수술 전 계획 수립과 수술 중 팀원 간의 의사소통에 과도하게 의존하게 만들며, 이 과정에서 발생하는 의사소통 오류나 인간의 실수는 환자의 안전을 직접적으로 위협하는 심각한 결과를 초래할 수 있다.

기존에도 수술 프로세스 모델(Surgical Process Models, SPMs)이 제안되어 워크플로우 관리 관점에서 절차를 표현하려는 시도가 있었으나, 이러한 모델들은 대개 묘사적(descriptive)이거나 반형식적(semi-formal)인 수준에 그쳤다. 즉, 절차를 시각화하거나 전달하는 데는 유용하지만, 해당 절차가 안전한지, 혹은 특정 실수로 인해 어떤 위험이 발생하는지를 엄격하게 추론하고 자동화된 방식으로 분석하는 기능은 거의 제공하지 못했다는 문제가 있다. 따라서 본 논문의 목표는 수술 절차를 형식적으로 모델링하고, 인간의 실수를 체계적으로 분석하여 환자의 안전을 보장하는 자동화된 분석 방법론을 제시하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 수술 절차를 사이버 보안 분야의 **Security Ceremonies**로 개념화하여 모델링하는 것이다. Security Ceremony는 컴퓨터 노드뿐만 아니라 인간 노드, 인간-컴퓨터 및 인간-인간 간의 통신, 그리고 물리적 객체의 전달을 모두 포함하는 확장된 보안 프로토콜 개념이다.

수술 절차를 이러한 프레임워크로 정의함으로써 얻는 이점은 다음과 같다. 첫째, 절차를 **Message Sequence Chart (MSC)** 형태로 표현하여 개념적 명확성을 확보하고 공유 가능한 형태로 만들 수 있다. 둘째, 보안 프로토콜 분석에서 검증된 형식적 방법론을 수술 절차 분석에 그대로 적용할 수 있다. 특히, 인간의 실수를 **Mutation(변이)**으로 정의하여 모델링함으로써, 특정 행동을 건너뛰거나 잘못된 메시지를 전달했을 때 환자의 안전 속성(Safety Property)이 어떻게 위반되는지를 자동화된 도구를 통해 탐색할 수 있다.

## 📎 Related Works

로봇 보조 수술(Robotic-assisted surgery, RAS)의 도입으로 수술실의 물리적 구조가 변화함에 따라, 집도의가 환자와 분리되어 콘솔에서 조작하게 되면서 대인 간의 비언어적 신호가 차단되고 의사소통의 어려움이 증가했다는 연구들이 보고되었다. 이러한 환경 변화는 상황 인식(situational awareness) 저하와 팀 협업의 효율성 감소를 야기하며, 이는 결국 수술 중 중단이나 오류로 이어질 가능성을 높인다.

기존의 SPM 접근 방식은 수술 단계를 정의하고 팀원 간의 워크플로우를 최적화하는 데 집중했으나, 본 논문은 여기서 한 걸음 더 나아가 **형식적 분석(Formal Analysis)**을 도입한다. 기존 연구들이 "어떻게 수행해야 하는가"에 집중했다면, 본 연구는 "특정 실수가 발생했을 때 어떤 안전 규칙이 깨지는가"를 수학적으로 증명하고 탐색한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조 및 모델링

본 방법론은 수술 절차를 에이전트(Surgeon, Assistant, Nurse) 간의 메시지 교환과 개별 행동의 시퀀스로 모델링한다.

1. **Message Sequence Chart (MSC):** 각 에이전트를 프로세스로 정의하며, 수직 타임라인상의 행동(Surgical Actions)과 에이전트 간에 주고받는 메시지(Messages)로 구성된다.
2. **Role-Script:** 에이전트의 행동을 다음과 같은 이벤트들의 시퀀스로 정의한다.
    * $Start(Ag, K^0_{Ag})$: 에이전트의 시작 및 초기 지식 상태.
    * $Snd(Ag_s, Ag_r, m)$: 송신자 $Ag_s$가 수신자 $Ag_r$에게 메시지 $m$을 전송.
    * $Rcv(Ag_s, Ag_r, m)$: 메시지 $m$을 수신.
    * $s\_action(Ag, a_{Ag}, O)$: 에이전트 $Ag$가 객체 $O$에 대해 행동 $a_{Ag}$를 수행.

### 실행 모델 및 전이 규칙

상태 전이는 멀티셋 재작성 시스템(multi-set rewriting system)을 통해 정의된다. 상태 $S_i$에서 전제 조건(premise)이 만족되면 결론(conclusion)인 다음 상태 $S_{i+1}$로 전이된다.

$$ \frac{prem (\text{role-actions and internal checks})}{conc (\text{new state})} $$

예를 들어, 간호사(Nurse)가 집도의(Surgeon)로부터 클립 요청 메시지를 받고, 클립을 제공한 뒤 조수(Assistant)에게 이를 알리는 과정은 다음과 같은 규칙으로 표현된다.
$$\{1; K^1_S, K^1_A, K^1_N\} \xrightarrow{Rcv(S,N,?X)} \xrightarrow{Check(?X=clips\_requested)} \xrightarrow{s\_action(N,provide,clips)} \xrightarrow{Snd(N,A,clips\_provided)} \{2; K^2_S, K^2_A, K^2_N\}$$

### Mutation 및 Matching Mutation

인간의 실수를 모델링하기 위해 두 가지 Mutation을 도입한다.

* **Skip Mutation:** 특정 수술 행동이나 메시지 전송을 생략하는 경우이다.
* **Replace Mutation:** 메시지를 다른 메시지로 대체하는 경우이다. 본 논문에서는 특히 "부정 메시지(negative message, e.g., $\text{not\_m}$)"라는 개념을 도입하여, 행동은 수행했으나 잘못된 신호를 보낸 상황을 모델링한다.

Mutation이 발생하면 이후 단계의 에이전트들이 대기 상태에 빠져 교착 상태(Deadlock)가 발생할 수 있다. 이를 방지하고 분석 가능한 트레이스(Trace)를 생성하기 위해 **Matching Mutation**을 정의한다. 이는 앞선 에이전트가 특정 행동을 생략했을 때, 다음 에이전트 또한 해당 행동이 불가능함을 인지하고 그에 맞춰 행동을 생략하거나 변경하는 전이 규칙이다.

### 분석 절차 및 도구

본 연구는 **UPPAAL** 도구를 사용하여 모델을 검증한다.

1. **속성 정의:** 선형 시간 논리(Linear Temporal Logic, LTL)를 사용하여 안전 속성을 정의한다. 예를 들어 "클립-전-절개(Clip-before-cutting)" 속성은 다음과 같다.
    $$\text{s\_action}(S, \text{cut}) @l \implies \text{s\_action}(S, \text{request, clips}) @i \land \text{s\_action}(N, \text{provide, clips}) @j \land \text{s\_action}(A, \text{apply, clips}) @k \land i < j < k < l$$
2. **자동화 분석:** UPPAAL의 분석 엔진을 통해 모든 가능한 실행 경로를 탐색하며, 위 속성을 위반하는 경로가 존재하는지 확인한다. 속성이 위반될 경우, 도구는 이를 증명하는 **Attack Trace(공격 트레이스)**를 출력한다.

## 📊 Results

본 연구는 복강경 전립선 절제술(laparoscopic prostatectomy)의 두 단계인 **Cutting stage**와 **Lateral dissection stage**를 대상으로 실험을 진행하였다.

### 분석 대상 속성

* **Property 1 (Clip-before-cutting):** 절개 전 반드시 클립 요청 $\to$ 제공 $\to$ 적용 단계가 선행되어야 함.
* **Property 2 (Dissection of the pedicle):** Pedicle 절개 전 정관(VD)과 정낭(SV)의 견인, Pedicle 식별 및 소작이 선행되어야 함.
* **Property 3 (Incision of the DF):** Denonvilliers’ Fascia(DF) 절개 전 PFS 진입, visceral fascia 절개, NVB 점검이 선행되어야 함.
* **Property 4 (Check if the nerves are preserved):** NVB 점검 전 Pedicle 소작, PFS 진입, visceral fascia 절개가 선행되어야 함.

### 정량적 및 정성적 결과

1. **Mutation 비활성화 시:** 모든 속성이 만족됨을 확인하였다. 즉, 정의된 표준 절차대로 수행될 경우 안전성은 보장된다.
2. **Mutation 활성화 시:** 모든 속성에서 위반 사례가 발견되었다.
    * **Property 1 위반:** 조수가 클립을 적용하지 않았음에도 불구하고 집도의가 절개를 수행하는 트레이스가 발견되었다. 이는 의사소통 오류나 태만으로 인해 발생할 수 있는 위험 상황을 정확히 짚어낸 결과이다.
    * **Property 2, 3, 4 위반:** 각각의 단계에서 필수 전제 조건이 생략된 채 다음 단계로 넘어가는 'Attack Trace'가 생성되었다.

이 결과는 단순한 텍스트 기반의 절차서로는 발견하기 어려운 잠재적 위험 경로를 형식적 분석을 통해 자동으로 찾아낼 수 있음을 보여준다.

## 🧠 Insights & Discussion

### 강점 및 의의

본 연구는 사이버 보안의 'Security Ceremony'라는 개념을 의료 도메인에 성공적으로 이식하였다. 특히, 인간의 실수를 Mutation으로 정형화하고, 이를 통해 발생 가능한 모든 위험 시나리오를 전수 조사할 수 있는 기반을 마련했다는 점이 매우 강력하다. 이는 실제 환자에게 적용하기 전, 절차의 변형(variant)이 안전한지 혹은 위험한지를 가상 환경에서 검증할 수 있게 함으로써 의료 사고 예방에 기여할 수 있다.

### 한계 및 비판적 해석

1. **상태 공간 폭발(State Space Explosion):** UPPAAL을 사용한 분석에서 수술 전체 과정을 한꺼번에 모델링할 경우 상태 공간이 너무 커져 분석이 불가능했다. 이로 인해 각 단계를 독립적으로 분석해야 했으며, 단계 간의 상호작용을 통합적으로 분석하기 위해서는 추상화(Abstraction)나 최적화 기법이 필수적으로 요구된다.
2. **실제 적용 가능성의 해석:** UPPAAL이 출력하는 'Attack Trace'가 항상 실제 위험을 의미하는 것은 아니다. 어떤 경우에는 숙련된 의사가 상황에 따라 선택하는 효율적인 '지름길(shortcut)'일 수 있다. 따라서 도구의 결과물을 임상 전문가가 해석하는 과정이 반드시 수반되어야 한다.
3. **단순한 메시지 모델:** 현재 모델에서는 메시지가 상수로 처리되어 매우 단순하다. 실제 수술 환경에서의 복잡한 의사소통(모호한 표현, 다중 수신자 등)을 반영하기 위해서는 메시지 대수(Algebra of messages)의 확장이 필요하다.

## 📌 TL;DR

본 논문은 수술 절차를 사이버 보안의 **Security Ceremonies**로 모델링하여, 인간의 실수를 **Mutation**으로 정의하고 이를 **UPPAAL**을 통해 자동 분석하는 프레임워크를 제안하였다. 복강경 전립선 절제술 사례 연구를 통해, 표준 절차에서 벗어난 행동(Skip, Replace)이 환자의 안전 속성을 어떻게 위반하는지 'Attack Trace' 형태로 도출해 냈으며, 이는 향후 수술 절차의 안전성 검증 및 맞춤형 수술 경로 설계에 중요한 도구로 활용될 가능성이 높다.
