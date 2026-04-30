# Brain-inspired AI Agent: The Way Towards AGI

Bo Yu, Jiangning Wei, Minzhen Hu, Zejie Han, Tianjian Zou, Ye He, Jun Liu (2024)

## 🧩 Problem to Solve

본 논문은 인공 일반 지능(Artificial General Intelligence, AGI)을 달성하기 위한 경로로서의 AI 에이전트 구조를 다룬다. 현재의 AI 에이전트들은 거대 언어 모델(Large Language Models, LLMs)의 처리 능력에 크게 의존하고 있으며, 대부분 특정 작업에 특화된(task-specific) 워크플로우를 가지고 있다. 이러한 구조는 인간 뇌의 인지 처리 능력과 비교했을 때 상당한 간극이 존재하며, 범용적인 작업 수행 능력과 유연성이 부족하다는 문제점이 있다. 따라서 저자들은 인간 뇌의 기능적 메커니즘을 모방하여 더 높은 수준의 인지 지능을 갖춘 범용 AI 에이전트를 설계하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 인간 뇌의 **중규모 피질 네트워크(mesoscale cortical networks)** 구조를 AI 에이전트 아키텍처에 투영한 '뇌 영감 AI 에이전트(Brain-inspired AI Agent)' 개념을 제안한 것이다. 단순히 거시적인 인지 패턴을 흉내 내는 수준을 넘어, 뇌의 특정 피질 영역을 기능적 모듈(functional modules)로 정의하고, 이들 사이의 **기능적 연결성(functional connectivity)**을 통해 작업 흐름을 제어하는 설계를 제시한다. 즉, 뇌의 생물학적 구조와 기능적 연결망을 AI 에이전트의 모듈형 구조와 상호작용 경로로 치환함으로써 AGI로 가는 구체적인 아키텍처 방향성을 제시하였다.

## 📎 Related Works

논문은 AGI를 향한 기존의 노력들을 세 가지 관점에서 설명한다.
첫째, 딥러닝의 발전으로 CNN, RNN, 그리고 Transformer 기반의 BERT와 GPT 같은 모델들이 등장하며 AGI의 가능성을 보여주었으나, 이들은 여전히 특수 목적 AI(special purpose AI)의 범주에 머물러 있다.
둘째, AI 에이전트 연구는 자율성, 반응성, 능동성 등을 갖춘 프레임워크로 발전하여 자율 주행, 게임 AI, 로보틱스 등에 적용되었으나, 여전히 작업 중심적인 워크플로우에 의존하고 있다.
셋째, Brain-inspired AI 분야에서는 SNN(Spiking Neural Networks)이나 뇌 영역을 모델 노드로 사용하는 시도가 있었으나, 이를 종합적인 에이전트 아키텍처로 통합하여 설계하려는 연구는 부족한 실정이다.

## 🛠️ Methodology

본 논문이 제안하는 뇌 영감 AI 에이전트의 핵심 방법론은 뇌의 피질 영역을 기능적 노드로 매핑하고, 이를 연결하는 네트워크를 설계하는 것이다.

### 1. 중규모 피질 네트워크 기반 모듈화
인간 뇌의 복잡한 뉴런 회로를 직접 모델링하는 대신, 기능적으로 독립적이면서 서로 연결된 중규모(mesoscale) 피질 영역에 집중한다. 각 피질 영역은 에이전트의 **기능적 모듈(functional module)**이 되며, 모듈의 복잡도와 요구 기능에 따라 LLM이나 다른 도구들로 구현된다.
- 예: 시각적 특징 추출을 담당하는 $\text{V1}$ 영역 $\rightarrow$ CNN 또는 YOLO로 구현.
- 예: 고차원 인지 기능을 담당하는 전전두엽 피질($\text{PFC}$) $\rightarrow$ LLM으로 구현.

### 2. 기능적 연결성(Functional Connectivity) 설계
연결 구조를 설계함에 있어 정적인 해부학적 연결인 구조적 연결성(structural connectivity)보다, 작업에 따라 동적으로 활성화되는 **기능적 연결성(functional connectivity)** 모델을 따른다. 이는 병렬 작업 처리와 영역 간 정보 통합을 가능하게 한다.

### 3. 기능-피질 영역 매핑 (Brain-like Functions Mapping)
논문은 주요 뇌 기능과 그에 해당하는 피질 영역 및 네트워크를 다음과 같이 정의한다 (Table I 참조).

| 에이전트 기능 모듈 | 대응 핵심 피질 영역 | 주요 기능 | 대응 피질 네트워크 |
| :--- | :--- | :--- | :--- |
| **Perception (인지)** | Primary Visual/Auditory Cortex | 시각 및 청각 인지 | Visual/Auditory Input Network |
| **Planning (계획)** | Prefrontal Cortex (PFC) | 고수준 인지 기능, 계획 | PFC-Motor / PFC-Parietal Network |
| **Decision-making (의사결정)** | DLPFC | 논리적 추론, 결정 실행 | ACC-DLPFC-Parietal Network |
| **Action (행동)** | Primary Motor Cortex | 운동 계획 및 실행 | Motor Cortex-Motor Network |
| **Memory (기억)** | Hippocampus, PFC | 기억 검색 및 변환 | Hippocampus-Neocortex Pathway |
| **Reasoning (추론)** | DLPFC, VMPFC | 논리적 추론, 인과 분석 | PFC-Parietal / PFC-Hippocampus Pathway |
| **Reflection (성찰)** | Medial PFC | 자기 성찰, 감정 조절 | Default Mode Network (DMN) |
| **Optimization (최적화)** | DLPFC | 인지 제어, 작업 최적화 | Frontal Cortex-Lateral Prefrontal Pathway |
| **Emotion (감정)** | Amygdala, ACC | 감정 인식 및 조절 | Amygdala-VMPFC-ACC Pathway |
| **Language (언어)** | Broca's / Wernicke's Area | 언어 생성 및 이해 | Language Network (Arcuate Fasciculus) |

### 4. 작동 절차
1. **입력 및 추출**: 외부 정보가 들어오면 해당 양식(modality)에 맞는 기능적 모듈(예: 시각 피질 모듈)에서 특징을 추출한다.
2. **저장**: 추출된 정보는 관련 기억 모듈에 저장된다.
3. **활성화**: 실행 담당 영역(execution region)에서 명령을 내리면, 기능적 연결성 경로를 통해 기억 모듈의 정보가 분석 담당 영역으로 전달된다.
4. **상태 관리**: 각 기능적 노드는 현재 작업 수행 여부에 따라 활성화 상태(activation state)를 갖는다.

## 📊 Results

본 논문은 구체적인 수치적 실험 결과나 벤치마크 성능을 제시하는 실험 논문이 아니라, **개념적 아키텍처를 제안하는 제안서 성격의 논문**이다. 

대신, 저자들은 기존의 LLM 기반 단일 에이전트 아키텍처들(GEAP, WebShop, Voyager 등)과 본 제안 모델을 비교 분석한 표(Table II)를 제시한다. 분석 결과, 기존 에이전트들은 인지, 계획, 행동, 기억 중 일부 기능만을 선택적으로 구현하고 있으나, 본 논문이 제안하는 **Brain-inspired Agent**는 인지부터 언어, 감정, 성찰에 이르기까지 뇌의 모든 주요 기능을 통합적으로 포함하는 가장 포괄적인 구조임을 정성적으로 보여준다.

## 🧠 Insights & Discussion

### 강점 및 가능성
- 기존의 단순한 $\text{Perception} \rightarrow \text{Planning} \rightarrow \text{Action}$ 흐름을 넘어, 뇌과학적 근거(피질 영역 및 연결성)를 바탕으로 에이전트의 내부 구조를 체계화하였다.
- LLM을 단일 '두뇌'로 사용하는 것이 아니라, 뇌의 각 영역처럼 특화된 모듈들의 집합으로 배치함으로써 더 유연하고 범용적인 지능 구현 가능성을 제시하였다.

### 한계 및 해결 과제
저자들은 실제 구현을 위해 해결해야 할 네 가지 도전 과제를 명시하고 있다.
1. **뇌에 대한 이해 부족**: 신경과학의 발전에도 불구하고 인간 뇌의 모든 지능적 기능을 완벽히 복제하는 것은 여전히 어렵다.
2. **아키텍처 정의의 미흡**: 본 연구는 중규모 피질 영역에 집중했으나, 실제 지능에는 피질 하부 영역(subcortical areas)과 더 세밀한 신경 연결이 필수적이다.
3. **계산 자원 소모**: 수많은 기능적 노드와 연결망을 유지하고 활성화하는 데 막대한 컴퓨팅 자원이 필요하며, 이에 대한 효율적인 할당 문제가 남는다.
4. **프레임워크 통합**: 다양한 모듈과 동적 연결망을 통합적으로 관리할 수 있는 전용 프레임워크 개발이 필요하다.

## 📌 TL;DR

본 논문은 인간 뇌의 중규모 피질 영역과 그들 사이의 기능적 연결성을 모방하여, 범용 인공 지능(AGI)을 달성하기 위한 **뇌 영감 AI 에이전트 아키텍처**를 제안한다. 기존 에이전트들이 특정 작업에 특화된 단순 구조였던 것과 달리, 본 모델은 인지, 계획, 결정, 행동, 기억, 성찰, 감정, 언어 등 뇌의 주요 기능을 각각의 모듈로 매핑하고 이를 동적으로 연결하는 체계를 가진다. 이는 미래의 AI 에이전트가 단순한 도구를 넘어 인간과 유사한 인지 능력을 갖춘 범용 지능체로 진화하는 데 중요한 설계 지침을 제공할 것으로 기대된다.