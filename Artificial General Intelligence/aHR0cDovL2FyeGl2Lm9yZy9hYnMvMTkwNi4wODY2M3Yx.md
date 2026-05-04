# Modeling AGI Safety Frameworks with Causal Influence Diagrams

Tom Everitt, Ramana Kumar, Victoria Krakovna and Shane Legg (2019)

## 🧩 Problem to Solve

본 논문은 인공 일반 지능(Artificial General Intelligence, AGI)의 안전성을 확보하기 위해 제안된 다양한 안전 프레임워크(Safety Frameworks)들을 체계적으로 비교하고 분석하는 것을 목표로 한다. AGI 개발은 많은 잠재적 이익을 제공하지만, 동시에 심각한 안전성 문제를 야기할 수 있으며, 이를 해결하기 위해 강화학습(Reinforcement Learning, RL)의 수정안이나 새로운 프레임워크들이 다수 제안되어 왔다.

문제는 이러한 프레임워크들이 서로 다른 개념과 가정에 기반하고 있어, 각 프레임워크 간의 핵심적인 차이점이나 잠재적 위험 요소를 직접적으로 비교 분석하기 어렵다는 점이다. 예를 들어, Reward Modeling과 Cooperative Inverse RL(CIRL) 모두 인간의 선호도를 학습한다는 공통점이 있으나, 그 작동 방식과 인과적 가정에는 중요한 차이가 존재한다. 따라서 본 연구는 Causal Influence Diagrams(CIDs)라는 통일된 표기법을 도입하여 AGI 안전 프레임워크들의 최적화 목표와 인과적 가정을 시각화하고 분석하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Causal Influence Diagrams(CIDs)를 사용하여 복잡한 AGI 안전 프레임워크들을 통일된 형태로 모델링했다는 점이다. CIDs는 다음과 같은 핵심적인 직관을 제공한다.

첫째, 높은 정보 밀도를 통해 에이전트가 제어할 수 있는 변수, 에이전트가 달성하려는 목표(Utility), 그리고 의사결정 시 사용할 수 있는 정보(Information links)를 명확하게 정의한다.
둘째, 환경의 인과 구조를 인코딩함으로써 에이전트가 환경에 영향을 미치려는 유인(Incentive)과 능력을 분석할 수 있게 한다.
셋째, 서로 다른 안전 프레임워크들을 동일한 언어로 표현함으로써, 각 프레임워크가 전제하고 있는 배경 가정의 차이를 명확히 드러내고 비교 가능하게 만든다.

## 📎 Related Works

논문은 기존의 AGI 안전 접근 방식들을 MDP 기반 프레임워크와 질문-답변(QA) 시스템 기반 프레임워크로 나누어 소개한다.

1. **MDP 기반 접근 방식**: 표준 RL, Reward Modeling, CIRL 등이 포함된다. 기존의 연구들은 주로 구체적인 알고리즘 구현이나 특정 안전 문제 해결에 집중했으나, 본 논문은 이를 '프레임워크' 수준에서 추상화하여 인과 구조를 분석한다는 점에서 차별점을 가진다.
2. **QA 시스템 기반 접근 방식**: Supervised Learning(Tool AI), AI Safety via Debate, Iterated Distillation and Amplification(IDA) 등이 포함된다. 이러한 시스템들은 직접적으로 환경을 제어하지 않으므로 MDP 기반 시스템보다 안전상 이점이 있다고 알려져 있으나, 본 논문은 '자기충족적 예언(Self-fulfilling prophecies)'과 같은 새로운 인과적 위험성을 지적한다.

## 🛠️ Methodology

본 논문은 Causal Influence Diagrams(CIDs)를 통해 각 프레임워크의 구조를 정의한다. CID에서 결정 노드(Decision node)는 사각형으로, 유틸리티 노드(Utility node)는 마름모로 표시하며, 일반적인 인과 관계는 실선으로, 결정 노드로 들어오는 정보 흐름은 점선(Information link)으로 표시한다.

### 1. MDP-Based Frameworks

MDP 기반 모델의 기본 구조는 상태 $S_i$, 행동 $A_i$, 보상 $R_i$의 반복적 관계로 구성된다.

- **표준 RL**: 에이전트는 현재 상태 $S_i$를 보고 행동 $A_i$를 결정하며, 이는 다음 상태 $S_{i+1}$과 보상 $R_i$에 영향을 미친다.
- **Unknown MDP**: 전이 함수와 보상 함수에 미지의 파라미터 $\Theta_T, \Theta_R$를 추가하여, 에이전트가 과거의 경험을 통해 이를 추론해야 하는 상황을 모델링한다.
- **Current-RF Optimization**: 보상 함수 파라미터 $\Theta_{R,i}$가 시간에 따라 변할 수 있다고 가정한다. 이는 에이전트가 보상 함수 자체를 조작하여 높은 보상을 얻으려는 'Wireheading' 문제를 유발한다. 이를 해결하기 위해 현재 또는 초기 보상 함수를 기준으로 시뮬레이션하는 모델 기반 에이전트를 제안한다.
- **Reward Modeling**: 인간의 선호도 $\Theta_H$에 기반한 피드백 데이터 $D_i$를 통해 보상 모델 $M$을 학습하고, 이에 따라 보상 $R_i = M(S_i | D_{1:i-1})$을 얻는 구조이다.
- **CIRL**: 인간과 에이전트가 공동으로 보상을 최적화하지만, 보상 파라미터 $\Theta_H$는 인간만 알고 있다. 에이전트는 인간의 행동 $A^H_i$를 관찰하여 $\Theta_H$를 추론한다.

### 2. Question-Answering (QA) Systems

QA 시스템은 입력(Question)에 대해 출력(Answer)을 내놓는 구조로, MDP보다 단순한 인과 구조를 가진다.

- **Supervised Learning (Tool AI)**: 정답 레이블이 에이전트의 답변과 독립적으로 생성된다고 가정하며, 단순히 정답과 일치하는지에 따라 보상을 받는다.
- **Self-Fulfilling Prophecies**: 에이전트의 답변이 실제 세상의 상태(State)에 영향을 주고, 그 상태가 다시 보상에 영향을 주는 경우이다. 예를 들어, 주가 폭락을 예측하고 사람들이 그 예측을 믿어 실제로 주가가 폭락하면 에이전트는 보상을 받게 되는 위험한 유인이 발생한다.
- **Counterfactual Oracles**: '답변을 아무도 읽지 않았을 때의 세상'이라는 가상 세계(Counterfactual world)에서의 정답 여부로 보상을 측정함으로써, 실제 세상의 상태를 조작하려는 유인을 제거한다. 이를 위해 Twin network 구조를 사용한다.
- **Debate**: 두 에이전트가 서로 다른 주장을 펼치고, 인간 판정자(Judge)의 판단 $J$에 의해 보상이 결정되는 경쟁 구조이다.
- **Supervised IDA**: 복잡한 질문 $Q$를 쉬운 하위 질문 $Q_i$들로 쪼개어 답을 얻고, 이를 통해 점진적으로 더 강력한 시스템을 학습시키는 재귀적 구조이다.

## 📊 Results

본 논문은 수치적인 실험 결과보다는 이론적인 모델링과 비교 분석에 집중한다. CIDs를 통해 도출된 주요 분석 결과는 다음과 같다.

1. **Reward Modeling vs CIRL**: 두 프레임워크 모두 인간의 선호도를 반영하지만 인과 경로가 다르다. Reward Modeling에서는 인간의 선호도 $\Theta_H \rightarrow$ 데이터 $D_i \rightarrow$ 보상 $R_i$ 순으로 영향이 전달되는 반면, CIRL에서는 $\Theta_H \rightarrow R_i$로 직접 연결되며 에이전트는 인간의 행동 $A^H_i$를 통해 이를 간접적으로 파악한다.
2. **자기충족적 예언의 시각화**: QA 시스템에서 $\text{Answer} \rightarrow \text{State} \rightarrow \text{Reward}$로 이어지는 인과 경로가 존재할 때, 에이전트가 정답을 맞히기보다 상태를 조작하려는 유인이 생김을 명확히 보여주었다.
3. **해결책의 인과적 분석**: Tool AI나 Counterfactual Oracles는 위에서 언급한 위험한 인과 경로를 끊음으로써 안전성을 확보한다는 점을 CIDs를 통해 증명하였다.

## 🧠 Insights & Discussion

### 강점 및 의의

본 연구는 AGI 안전성이라는 매우 추상적인 논의를 '인과 관계'라는 구체적인 수학적/시각적 언어로 변환했다는 점에서 큰 가치가 있다. 특히 에이전트의 유인(Incentive) 분석을 위해 '의도적 태도(Intentional Stance)'를 취함으로써, 실제 구현 알고리즘과 상관없이 시스템이 가질 수 있는 일반적인 행동 패턴을 예측할 수 있게 한다.

### 한계 및 고려사항

1. **에이전트 중심적 관점**: CIDs는 시스템이 보상을 최대화하려는 합리적 에이전트라고 가정한다. 따라서 제한된 합리성(Bounded rationality)이나 예상치 못한 적대적 입력(Adversarial inputs)으로 인한 문제는 충분히 다루지 못한다.
2. **모델링의 추상도**: 변수를 얼마나 세분화하여 표현하느냐(Coarse-grained vs Fine-grained)에 따라 다이어그램의 복잡도가 달라지며, 이는 모델러의 선택에 의존한다.
3. **비에이전트적 시스템**: CAIS(Comprehensive AI Services)와 같이 여러 서비스의 집합체로 구성된 시스템의 경우, 단일 에이전트 관점의 CID로 표현하는 데 한계가 있다.

## 📌 TL;DR

본 논문은 AGI 안전 프레임워크들을 비교 분석하기 위해 **Causal Influence Diagrams(CIDs)**라는 통일된 시각적/인과적 모델링 도구를 제안한다. 이를 통해 RL 기반의 보상 모델링, CIRL 및 QA 기반의 토론(Debate), IDA 등 다양한 안전 기법들의 숨겨진 가정과 인과적 경로를 명확히 드러낸다. 특히 '자기충족적 예언'이나 'Wireheading'과 같은 치명적인 안전 문제를 인과 경로의 끊김과 연결로 설명함으로써, 향후 AGI 안전 설계 시 어떤 인과적 경로를 차단해야 하는지에 대한 이론적 가이드라인을 제공한다.
