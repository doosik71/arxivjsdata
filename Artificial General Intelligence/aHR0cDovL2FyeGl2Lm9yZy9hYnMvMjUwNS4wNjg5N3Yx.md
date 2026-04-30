# Embodied Intelligence: The Key to Unblocking Generalized Artificial Intelligence

Jinhao Jiang, Changlin Chen, Shile Feng, Wanru Geng, Zesheng Zhou, Ni Wang, Shuai Li, FengQi Cui, and Erbao Dong (2025)

## 🧩 Problem to Solve

본 논문은 인공지능의 최종 목표인 인공 일반 지능(Artificial General Intelligence, AGI)을 달성하기 위한 핵심 경로로서의 구체화된 지능(Embodied Intelligence, EAI)의 역할과 구조를 분석한다.

전통적인 심볼릭 AI와 현대의 좁은 AI(Narrow AI, ANI)는 논리적 추론이나 특정 데이터셋 기반의 최적화에는 능숙하지만, 복잡한 실제 환경과의 실시간 상호작용이 부족하여 범용적인 지능을 구현하는 데 한계가 있다. 특히 기존의 연구들이 특정 기술이나 개별 애플리케이션에 치중되어 있어, EAI가 어떻게 AGI의 핵심 원칙들을 충족시키고 실제로 AGI로 나아가는 가교 역할을 하는지에 대한 체계적인 분석이 부족한 상황이다. 따라서 본 논문은 EAI를 AGI 달성을 위한 기초적 접근 방식으로 정의하고, 그 기술적 구조와 AGI 원칙 간의 상관관계를 규명하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 EAI의 기술적 솔루션을 **Sub-modular Architecture(서브 모듈 구조)**와 **End-to-End Architecture(엔드-투-엔드 구조)**로 명확히 구분하고, 이를 AGI의 관점에서 체계적으로 분석했다는 점이다.

특히 DeepMind가 제시한 AGI의 6가지 핵심 원칙(모델 능력 중심, 일반화 및 성능, 인지 및 메타인지 과업, 잠재력 중심, 생태적 타당성, 발전 경로 중심)을 기준으로, EAI의 4가지 핵심 모듈(Perception, Intelligent Decision-making, Action, Feedback)이 각각 어떻게 AGI의 원칙들을 구현하는지를 이론적으로 연결하여 설명하였다. 또한, 최신 LLM(Large Language Models)의 통합이 EAI에 가져온 인식론적 변화(Symbolic Embodiment, Cognitive Scaffolding, Metacognitive Regulation)를 심도 있게 다루었다.

## 📎 Related Works

논문은 AGI와 EAI의 개념적 진화를 다음과 같은 흐름으로 설명한다.

1.  **초기 AGI 및 한계**: 1950년대부터 논의된 AGI는 자율 학습과 환경 적응력을 목표로 했으나, 초기 접근법들은 환경과의 물리적 상호작용을 간과하여 실제 세계의 복잡성을 해결하지 못했다.
2.  **Narrow AI (ANI)와의 차이**: Ray Kurzweil이 정의한 ANI는 특정 영역의 문제 해결에 집중하는 반면, AGI는 다중 도메인에서 인간 수준의 지능을 발휘하는 것을 목표로 한다.
3.  **구체화된 인지 이론**: Alan Turing의 물리적 참여 이론, Rodney Brooks의 Subsumption Architecture, Rolf Pfeifer와 Christian Scheier의 신체 구조 기반 지능 이론, 그리고 Linda Smith의 Embodiment Hypothesis(신체화 가설)를 통해 지능이 뇌의 계산만이 아니라 신체와 환경의 역동적인 상호작용에서 발생한다는 점을 강조한다.
4.  **기술적 전환점**: 2012년 이후 Deep Learning의 성숙으로 다중 모달 센서 데이터의 병렬 처리가 가능해졌으며, 최근 LLM의 도입으로 언어적 추상화와 물리적 행동 간의 매핑이 가능해지면서 EAI의 발전이 가속화되었다.

## 🛠️ Methodology

본 논문은 EAI 시스템을 구현하는 두 가지 주요 아키텍처를 제시하고, 그중 모듈형 구조의 세부 구성 요소를 상세히 분석한다.

### 1. 아키텍처 패러다임 비교
- **End-to-End Architecture**: 원시 센서 입력(Vision, LiDAR 등)을 신경망을 통해 직접 제어 신호(Actuator control)로 매핑한다. 이론적 성능 상한선이 높고 전역 최적화가 가능하지만, 데이터 요구량이 극도로 높고 내부 작동 원리를 알 수 없는 Black-box 모델이라는 단점이 있다. (예: Tesla FSD V12)
- **Modular Architecture**: 시스템을 지각, 결정, 행동, 피드백의 4개 모듈로 분리한다. 각 모듈을 독립적으로 최적화할 수 있어 개발 비용이 낮고 해석 가능성이 높으나, 모듈 간 상호의존성으로 인해 전체 성능이 가장 낮은 모듈에 제한되는 'Bucket Effect'가 발생할 수 있다.

### 2. 모듈형 아키텍처의 4대 핵심 구성 요소
시스템은 다음과 같은 폐쇄 루프(Closed-loop) 구조로 작동한다.

#### ① Perception Module (지각 모듈)
다양한 센서 데이터를 통합하여 환경 상태를 재구성한다. 프로세스는 다음과 같은 단계로 진행된다.
$$\text{Data Acquisition} \rightarrow \text{Preprocessing} \rightarrow \text{Feature Extraction} \rightarrow \text{Data Fusion} \rightarrow \text{Learning \& Inference} \rightarrow \text{Output}$$
- **Multimodal Fusion**: 시각, 청각, 촉각 데이터를 통합하여 단일 모달의 한계를 극복하며, 이를 통해 AGI의 '범용성 및 성능' 원칙을 지원한다.

#### ② Decision-Making Module (의사결정 모듈)
지각 모듈의 정보와 피드백 모듈의 데이터를 바탕으로 과업 계획을 수립한다.
- **구성**: 환경 이해 및 추론 $\rightarrow$ 과업 계획(Task Planning) $\rightarrow$ 결정 생성(Decision Generation) $\rightarrow$ 학습 및 진화 프레임워크.
- **특징**: Reinforcement Learning(RL)과 Meta-learning을 통해 경험으로부터 스스로를 최적화하며, 이는 AGI의 '잠재력 중심' 원칙과 일치한다.

#### ③ Action Module (행동 모듈)
결정 모듈의 명령을 물리적 동작으로 변환하여 실행한다.
- **핵심 기술**: 형상기억합금(SMA)이나 전기활성고분자(EAP) 같은 스마트 재료를 이용한 유연한 제어, Bionic Motion Modeling(생체 모방 동작 모델링) 등이 포함된다.
- **역할**: 물리적 제약 조건 하에서 최적의 동작 경로를 생성하여 실행 효율을 극대화한다.

#### ④ Feedback Module (피드백 모듈)
실시간으로 상호작용 상태를 모니터링하여 시스템을 조정한다.
- **Perceptual Feedback**: 환경 변화에 따라 지각 프로세스를 동적으로 조정한다.
- **Decision Feedback**: 계획된 전략의 실제 효과를 평가하여 전략을 수정한다.
- **Action Feedback**: 액추에이터의 상태를 실시간으로 감시하여 오차를 보정한다.

## 📊 Results

본 논문은 특정 실험 데이터셋을 통해 성능을 입증하는 연구가 아니라, 기존 기술과 산업 사례를 분석한 리뷰 및 프레임워크 제안 논문이다. 따라서 정량적 결과 대신 다음과 같은 산업적 구현 사례와 기술적 경향성을 결과로 제시한다.

- **End-to-End 구현 사례**: 
    - **Tesla FSD V12**: 수백만 대의 차량에서 수집한 실데이터를 통해 명시적 경로 계획 없이 직접적인 제어를 학습하는 Pure physical-world learning을 구현하였다.
    - **Huawei GOD-PDP**: 실세계 장애물 탐지(General Obstacle Detection)와 시뮬레이션 기반 결정 프로세서(Predictive Decision Processor)를 결합한 하이브리드 방식을 채택하였다.
- **데이터 확보 전략**: 미국 기업들은 NVIDIA DRIVE Sim과 같은 고도화된 시뮬레이션 기반의 합성 데이터(Synthetic Data) 생성에 집중하는 반면, 중국 기업들은 대규모 원격 조작(Teleoperation) 인프라를 통해 물리적 데이터를 직접 수집하는 전략을 취하고 있다.
- **모듈 간 상호작용 분석**: 4가지 모듈이 통합된 폐쇄 루프 시스템이 구축될 때, 시스템은 단순한 자동화를 넘어 환경에 적응하고 스스로 진화하는 지능적 특성을 보이며, 이것이 AGI로 가는 실질적인 경로임을 논증하였다.

## 🧠 Insights & Discussion

### 강점 및 통찰
- **상징적 구체화(Symbolic Embodiment)**: LLM의 도입이 단순히 텍스트 처리를 넘어, 언어적 추상 개념과 물리적 행동 패턴 사이의 양방향 매핑을 가능하게 함으로써 EAI의 차원을 높였다는 분석이 매우 날카롭다.
- **AGI 원칙과의 연결**: 모호한 AGI의 정의를 DeepMind의 6가지 원칙으로 구체화하여 EAI의 각 모듈이 어떤 기여를 하는지 논리적으로 연결한 점이 훌륭하다.

### 한계 및 미해결 과제
- **지각-행동 조정(Coordination)**: 다중 센서 데이터의 실시간 처리와 물리적 행동 사이의 피드백 지연(Latency) 및 오차를 줄이는 것이 여전히 큰 기술적 난제임을 명시하고 있다.
- **추론 능력의 부족**: 현재의 EAI는 반복적인 시행착오(Trial and Error)를 통한 학습에 의존하며, 기존 지식을 바탕으로 처음 마주하는 문제를 해결하는 '추상적 추론(Abstract Reasoning)' 능력은 여전히 부족하다.
- **장기 기억(Long-term Memory)**: 현재 시스템들은 주로 단기 기억에 의존하며, 다양한 환경으로 지식을 전이(Transfer Learning)할 수 있는 장기적 기억 구조의 구축이 필요하다.

## 📌 TL;DR

본 논문은 AGI 달성을 위한 핵심 열쇠로 **구체화된 지능(Embodied Intelligence, EAI)**을 제시한다. EAI를 **End-to-End**와 **Modular** 아키텍처로 구분하여 분석하였으며, 특히 지각-결정-행동-피드백으로 이어지는 4대 모듈의 유기적 결합이 AGI의 핵심 원칙들을 어떻게 구현하는지 체계적으로 설명하였다. 결론적으로, 지능은 단순한 계산이 아니라 물리적 신체를 통한 환경과의 끊임없는 상호작용 속에서 창발(Emergence)하며, 이러한 EAI의 발전이 좁은 AI(ANI)의 한계를 깨고 AGI로 나아가는 유일하고 실질적인 경로임을 강조하고 있다.