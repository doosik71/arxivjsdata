# On the link between conscious function and general intelligence in humans and machines

Arthur Julian, Kai Arulkumar, Shuntaro Sasai, Ryota Kanai (2022)

## 🧩 Problem to Solve

본 논문은 대중 매체에서 흔히 묘사되는 '인공 지능의 자각(awareness)과 인간 수준 혹은 그 이상의 지능(superhuman intelligence) 사이의 연결 고리'가 실제로 유효한지를 학술적으로 탐구한다. 

전통적인 AI 시스템은 특정 도메인에서는 압도적인 성능을 보이지만, 새로운 기술을 습득하는 효율성 측면에서는 인간이나 동물에 비해 매우 낮다. 이는 수백만 개의 데이터 샘플이 필요하거나, 설계 단계에서 이미 방대한 도메인 지식이 주입되어야 하기 때문이다. 즉, 현재의 AI는 '범용 지능(General Intelligence)' 관점에서 심각한 결함이 있으며, 저자들은 이 지능의 격차(intelligence gap)를 메우기 위한 방법으로 인간의 의식 기능(conscious function)에 주목한다.

논문의 핵심 목표는 의식의 기능적 이론들을 분석하여, 이것이 어떻게 범용 지능을 가능하게 하는지 밝히고, 이를 인공 지능 아키텍처에 적용할 수 있는 통합적인 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 가장 중심적인 아이디어는 **'접근 의식(Access Consciousness)'**의 기능적 메커니즘이 범용 지능의 핵심인 '빠른 일반화 능력'과 직결되어 있다는 것이다.

저자들은 세 가지 현대적 의식 이론인 **Global Workspace Theory (GWT)**, **Information Generation Theory (IGT)**, **Attention Schema Theory (AST)**가 각각 지능의 서로 다른 측면을 지원한다고 주장한다. 이 세 가지 이론을 통합하여 구현할 수 있는 구체적인 목표로 **'정신적 시간 여행(Mental Time Travel, MTT)'** 능력을 제시한다. MTT는 단순히 과거를 회상하거나 미래를 예측하는 것을 넘어, 가상의 환경과 정책 속에서 자신을 투영하여 시뮬레이션하는 능력이다. 저자들은 인공 지능이 MTT를 구현할 수 있다면, 이는 곧 인간 수준의 범용 지능에 도달했음을 의미하는 기능적 지표가 될 것이라고 제안한다.

## 📎 Related Works

### 의식 연구의 배경
논문은 의식을 두 가지 관점으로 구분하는 Ned Block의 이론을 차용한다.
- **Phenomenal Consciousness (현상적 의식):** 주관적인 느낌(qualia)에 관한 것으로, 과학적으로 측정하기 어려운 'Hard Problem'에 해당한다.
- **Access Consciousness (접근 의식):** 추론, 언어 보고, 행동 제어에 사용 가능한 정보의 상태를 의미하며, 이는 기능적으로 측정 가능하므로 본 논문의 분석 대상이 된다.

### 기존 AI 접근 방식의 한계
현재의 Deep Learning은 대규모 데이터와 모델 크기를 통해 일반화를 꾀하는 'Foundation Models' 방식과, MBRL(Model-Based RL)이나 Meta-learning과 같은 '원칙적 일반화' 방식으로 나뉜다. 그러나 대부분의 시스템은 다음과 같은 한계를 가진다.
- **데이터 효율성 부족:** 인간은 Few-shot 혹은 Zero-shot 학습이 가능하지만, AI는 수천 배 이상의 경험이 필요하다.
- **구조적 제약:** 특정 환경(예: Atari 게임)에서 학습된 모델은 유사한 다른 환경에서도 성능이 급격히 떨어진다.
- **수동적 시뮬레이션:** 현재의 World Model은 주로 관측된 데이터의 재현(Replay)이나 제한적인 미래 예측(Preplay)에 머물러 있으며, 능동적인 가상 시나리오 생성 능력이 부족하다.

## 🛠️ Methodology

### 1. 의식의 세 가지 기능적 이론
저자들은 범용 지능을 구현하기 위해 다음의 세 가지 이론적 구성 요소가 필요하다고 설명한다.

- **Global Workspace Theory (GWT):** 뇌의 다양한 모듈이 정보를 공유하는 공통의 대표 공간(representational space)이 존재한다는 이론이다. 여기서 'Attention'은 관련 정보를 워크스페이스에 진입시키는 정책이며, 진입된 정보는 시스템 전체로 방송(broadcast)되어 공유된다. 이는 AI의 Modularity 및 Transformer의 Attention 메커니즘과 연결된다.
- **Information Generation Theory (IGT):** 의식적 경험은 단순한 감각 입력이 아니라, 뇌 내부의 생성 모델(generative model)이 만들어낸 결과물이라는 이론이다. 특히 'Cognitive Map'을 통해 공간적, 시간적 일관성을 유지하며 가상 시나리오를 생성한다. 이는 AI의 World Model 및 Generative Model과 대응된다.
- **Attention Schema Theory (AST):** 의식은 주의(attention) 그 자체가 아니라, '주의 프로세스에 대한 고수준 모델(meta-representation)'을 가지는 것이라는 이론이다. 이를 통해 에이전트는 자신이 무엇에 집중하고 있는지 인식하고, 상황에 맞춰 주의 정책을 유연하게 수정할 수 있다. 이는 AI의 Meta-learning 및 Self-modeling과 연결된다.

### 2. 경험 생성의 계층 구조 (Hierarchy of Experience Generation)
저자들은 지능의 수준을 경험 생성 능력에 따라 네 단계로 정의한다.

1. **Direct Experience:** 현재의 궤적에서만 학습하는 단계.
2. **Replay:** 과거에 경험한 궤적을 재현하여 학습하는 단계 (예: Experience Replay).
3. **Preplay:** 현재 환경 내에서 가능한 미래의 궤적을 시뮬레이션하는 단계 (예: MBRL의 Planning).
4. **Mental Time Travel (MTT):** 경험하지 않은 가상의 환경과 가상의 정책을 생성하여 시뮬레이션하는 단계.

### 3. 통합 시스템 구조
저자들이 제안하는 통합 아키텍처의 흐름은 다음과 같다.
$$\text{IGT (경험 생성)} \rightarrow \text{GWT (정보 선택 및 공유)} \rightarrow \text{AST (주의 정책 제어 및 자기 인식)}$$

- **IGT 기반 생성:** Cognitive Map을 활용해 시공간적으로 일관된 가상 궤적을 생성한다.
- **GWT 기반 통합:** 생성된 정보 중 행동에 필요한 핵심 정보만을 선택하여 Global Workspace에 유지하고, 이를 다른 모듈과 공유한다.
- **AST 기반 조절:** 현재의 가상 경험이 '시뮬레이션'임을 인지하고, 목적에 따라 주의 집중 대상을 동적으로 변경하는 Meta-controller 역할을 수행한다.

## 📊 Results

본 논문은 특정 데이터셋을 이용한 실험 논문이 아니라, 이론적 분석과 기존 연구들의 매핑을 통해 결론을 도출하는 **이론적 프레임워크(Theoretical Framework)** 논문이다. 따라서 정량적인 수치 결과보다는 다음과 같은 논리적 결과(Theoretical Results)를 제시한다.

- **기능적 매핑의 성공:** GWT $\rightarrow$ Modularity/Attention, IGT $\rightarrow$ World Models, AST $\rightarrow$ Meta-learning으로 각각의 뇌 과학 이론이 현대 AI의 핵심 기술들과 일대일 대응됨을 확인하였다.
- **MTT의 필요성 증명:** 인간의 높은 지능은 단순히 데이터를 많이 본 것이 아니라, MTT를 통해 가상의 시나리오에서 '사전 학습'을 수행했기 때문임을 논리적으로 설명하였다.
- **구현 가능성 제시:** 최근의 Transformer 아키텍처, MBRL, 그리고 Meta-RL의 발전이 각각 GWT, IGT, AST의 부분적 구현체로 작동하고 있음을 분석하였다.

## 🧠 Insights & Discussion

### 강점 및 의의
본 논문은 모호한 개념인 '의식'과 '지능'의 관계를 '접근 의식'이라는 기능적 관점으로 끌어내어, AI 연구자가 추구해야 할 구체적인 공학적 목표(MTT 구현)를 제시했다는 점에서 매우 높은 가치가 있다. 특히 뇌 과학의 세 가지 주요 이론을 통합하여 하나의 시스템 아키텍처로 제안한 점이 독창적이다.

### 한계 및 비판적 해석
- **현상적 의식의 배제:** 저자들은 'Qualia'와 같은 현상적 의식을 논외로 하였으나, 실제 지능의 발현에 있어 주관적 가치 판단이나 감정적 valence가 어떤 역할을 하는지에 대한 설명이 부족하다.
- **구현의 복잡성:** 세 이론을 통합하는 것은 이론적으로는 완벽해 보이지만, 이를 실제 신경망 아키텍처로 구현했을 때 발생할 계산 복잡도와 학습 불안정성에 대한 구체적인 해결책은 제시되지 않았다.
- **가정의 의존성:** MTT가 인간에게만 고유하다는 가정과 이것이 지능의 절대적 지표라는 전제는 여전히 학계에서 논쟁 중인 사안이다.

## 📌 TL;DR

본 논문은 인간의 **접근 의식(Access Consciousness)**을 가능하게 하는 세 가지 이론(**GWT, IGT, AST**)이 범용 지능의 핵심 기제임을 주장한다. 이 세 가지 기능을 통합하여 **'정신적 시간 여행(Mental Time Travel)'** 능력을 갖춘 AI를 만드는 것이 진정한 AGI로 가는 길이 될 것이라고 제안한다. 이는 단순한 데이터 증량이 아니라, 내부 시뮬레이션-정보 공유-메타 제어로 이어지는 의식의 기능적 구조를 AI에 이식해야 함을 의미한다.