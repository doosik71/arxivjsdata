# Visualizations for an Explainable Planning Agent

Tathagata Chakraborti, Kshitij P. Fadnis, Kartik Talamadupula, Mishal Dholakia, Biplav Srivastava, Jeffrey O. Kephart, and Rachel K. E. Bellamy (2018)

## 🧩 Problem to Solve

본 논문은 자동 계획(Automated Planning) 시스템이 인간과 협력하는 'Human-in-the-loop' 의사 결정 과정에서 발생하는 투명성(Transparency)과 설명 가능성(Explainability)의 결여 문제를 해결하고자 한다. 

기존의 지능형 비서(Smart Assistants)들은 대부분 미리 정의된 스크립트에 따라 수동적으로 반응하는 수준에 머물러 있으며, 복잡한 환경에서 능동적으로 협력하기 위해서는 자동 계획 기술이 필요하다. 그러나 계획 시스템의 내부 의사 결정 과정이 블랙박스 형태로 존재할 경우, 인간 사용자는 시스템의 제안이나 행동을 신뢰하기 어렵다. 따라서 사용자가 시스템의 내부 상태를 인지하고 의사 결정의 근거(Rationale)를 추론할 수 있도록 돕는 시각화 체계의 구축이 필수적이다. 본 연구의 목표는 계획 에이전트의 내부 인지 과정을 외부로 표출하고, 계획의 생성 근거를 효율적으로 전달하는 시각화 역량을 갖춘 Explainable AI Planning (XAIP) 에이전트를 구현하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 계획 에이전트의 인지 과정을 단계별 추상화 수준에 따라 시각화하는 'Mr. Jones' 시스템과, 계획의 생성 근거를 모델 기반으로 설명하는 'Fresco' 시각화 도구를 제안한 것이다.

중심적인 아이디어는 시각화를 단순한 데이터 표시가 아닌 하나의 '설명 과정(Process of Explanation)'으로 정의하는 것이다. 특히, 모든 도메인 지식을 보여주는 대신, 특정 계획이 최적이 되기 위해 반드시 필요한 최소한의 조건만을 추출하여 보여줌으로써 사용자의 인지적 과부하(Cognitive Overload)를 줄이고 계획의 'Why(왜 이 계획이 선택되었는가)'에 집중하게 만드는 Model Reconciliation 기법을 시각화에 도입하였다.

## 📎 Related Works

논문은 기존의 계획 시각화 연구들이 주로 계획 생성의 '방법(How)', 즉 상태 공간 탐색 과정(Search Process)을 시각화하는 데 치중했음을 지적한다. 이러한 접근 방식은 시스템 개발자의 디버깅에는 유용할 수 있으나, 최종 사용자에게는 계획의 근거를 이해시키는 데 큰 도움이 되지 않는다.

또한, XAIP 패러다임 내에서 신뢰, 상호작용, 투명성의 중요성이 강조되고 있으며, 특히 모델 간의 차이를 해결하여 계획을 설명하는 Model Reconciliation Process (MRP) 연구들이 진행되어 왔음을 언급한다. 본 연구는 이러한 MRP 개념을 시각화 영역으로 확장하여, 사용자의 멘탈 모델을 '빈 모델(Empty Model)'로 가정하고 이를 업데이트하는 방식으로 최적의 계획에 필요한 최소 지식만을 시각화하는 차별점을 가진다.

## 🛠️ Methodology

### 1. end-to-end Planning Agent: Mr. Jones
Mr. Jones는 크게 두 가지 프로세스로 구성된다.

- **Engage**: 환경의 센서 입력(음성 전사, 이미지, 사람의 위치 등)을 모니터링하여 현재 진행 중인 태스크를 인식하는 단계이다. Probabilistic plan recognition 알고리즘을 사용하여 현재 상황에 가장 적합한 태스크의 확률 분포를 계산한다.
- **Orchestrate**: 인식된 태스크를 바탕으로 의사 결정 지원을 수행한다. 이는 (1) 실행(Execute), (2) 비판(Critique), (3) 제안(Suggest), (4) 설명(Explain)의 네 가지 액션으로 나타난다.

### 2. 'Mind of Mr. Jones' 시각화 구조
에이전트의 인지 과정을 세 가지 추상화 수준으로 나누어 시각화한다.
- **Raw Inputs**: 카메라 피드, 음성-텍스트 변환 결과 등 가공되지 않은 입력 데이터를 보여주어 시스템이 무엇을 보고 듣는지를 명시한다.
- **Lower level reasoning**: 입력 데이터에서 추출된 정보(화자 식별, 위치, 대화 주제 등)를 시각화하여 상황 인지(Situational Awareness)를 돕는다.
- **Higher level reasoning**: 계획 인식(Plan Recognition)의 결과인 태스크별 확률 분포와 그 근거(Provenance)를 시각화하여 에이전트의 현재 이해도를 보여준다.

### 3. Model-Based Plan Visualization: Fresco
Fresco는 계획의 생성 근거를 시각화하기 위해 다음과 같은 방법론을 사용한다.

#### Top-K Visualization
단일 계획이 아닌 상위 $K$개의 대안 계획을 동시에 시각화하여, 사용자가 명시적으로 모델링되지 않은 선호도를 암시적으로 드러낼 수 있도록 지원한다.

#### Visualization as Model Reconciliation
계획의 생성 근거를 설명하기 위해 '모델 조정(Model Reconciliation)' 개념을 도입한다. 
- **가정**: 사용자가 해당 도메인에 대한 지식이 전혀 없는 '빈 모델(Empty Model)'을 가지고 있다고 가정한다.
- **절차**: 계획자의 모델과 빈 모델 사이의 간극을 메우기 위해, 현재의 계획이 최적이 되도록 만드는 최소한의 조건 집합인 Minimally Complete Explanation (MCE)을 계산한다.
- **결과**: 전체 도메인 조건 중 MCE에 포함되지 않은 불필요한 조건들은 시각화에서 회색으로 처리(Gray-out)하여 제거하고, 계획에 결정적인 영향을 준 핵심 조건들만 강조하여 표시한다.

### 4. Fresco 아키텍처
Fresco의 파이프라인은 다음과 같은 흐름으로 구성된다:
$$\text{Parser} \rightarrow \text{Planner (Fast-Downward, MMP)} \rightarrow \text{Resolver} \rightarrow \text{Visualizer (D3.js)}$$
- **Parser**: 도메인 모델과 문제 인스턴스를 Python 객체로 변환하고 VAL을 통해 검증한다.
- **Planner**: Fast-Downward를 통해 계획을 생성하고, Multi-Model Planner (MMP)를 통해 설명을 생성한다.
- **Resolver**: 생성된 계획과 설명을 바탕으로 불필요한 전제 조건이나 효과를 제거하여 최적화한다.
- **Visualizer**: 최종 결과를 웹 브라우저에서 렌더링 가능한 그래픽으로 변환한다.

## 📊 Results

본 연구는 IBM의 Cognitive Environments Laboratory (CEL) 환경에서 실험을 진행하였다.

- **정성적 결과 (Demonstration 1)**: 두 명의 사용자가 상호작용할 때 Mr. Jones의 인터페이스가 실시간으로 태스크 확률 분포를 업데이트하는 모습을 보여주었다. 이를 통해 에이전트의 신념(Belief) 변화 과정을 투명하게 공개할 수 있음을 입증하였다.
- **정성적 결과 (Demonstration 2)**: 에이전트가 수집한 시각화 데이터를 샘플링하여 회의록(Minutes)을 자동으로 생성하는 기능을 시연하였다. 이는 기존의 텍스트 기반 회의록보다 더 직관적인 요약 정보를 제공할 수 있음을 보여준다.
- **Fresco의 성능**: CD(Collective Decision) 도메인 실험에서, 총 30개의 조건 중 계획의 최적성에 기여하는 핵심 조건만을 추출하여 11개만 강조하고 나머지는 제거함으로써, 시각적 복잡도를 획기적으로 줄이면서도 계획의 근거를 명확히 전달할 수 있음을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 AI 계획 시스템에서 시각화의 역할을 단순한 '결과 표시'에서 '모델 기반의 설명'으로 격상시켰다는 점에서 강점이 있다. 특히 MRP를 활용해 빈 모델로부터의 MCE를 추출하는 방식은 사용자의 인지적 부하를 최소화하면서도 논리적인 근거를 제시하는 효율적인 방법이다.

다만, 본 연구는 다음과 같은 한계와 과제를 남기고 있다.
1. **모델 획득의 병목 현상**: 여전히 XML 기반의 모델링 인터페이스를 사용하고 있어, 비전문가가 도메인 모델을 구축하기에는 어려움이 있다. 이를 해결하기 위한 그래픽 기반 모델 획득 인터페이스가 필요하다.
2. **사용자 선호도 반영**: MCE가 유일하지 않을 수 있으며, 사용자마다 중요하게 생각하는 정보가 다를 수 있다. 향후 상호작용을 통해 사용자의 선호도를 학습하고 이에 맞춤화된 시각화를 제공하는 기능이 필요하다.
3. **범용성 확장**: 현재는 특정 도메인에 적용되어 있으나, `planning.domains`와 같은 도메인 독립적인 계획 도구에 통합되어 다양한 분야에서 활용될 필요가 있다.

## 📌 TL;DR

본 논문은 투명한 의사 결정을 지원하는 XAIP 에이전트 'Mr. Jones'와 모델 기반 계획 시각화 도구 'Fresco'를 제안한다. 특히 계획의 생성 근거를 설명하기 위해 Model Reconciliation 기법을 도입, 계획의 최적성을 보장하는 최소한의 핵심 조건만을 시각화함으로써 사용자의 인지 부하를 줄이고 시스템에 대한 신뢰도를 높였다. 이 연구는 향후 AI 계획 시스템이 인간과 협력하는 실제 환경에서 '왜 이런 계획이 수립되었는가'에 대한 답을 시각적으로 제공하는 중요한 기반이 될 것으로 보인다.