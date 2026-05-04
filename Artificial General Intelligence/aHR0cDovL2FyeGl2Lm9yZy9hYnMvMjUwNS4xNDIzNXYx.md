# Toward Embodied AGI: A Review of Embodied AI and the Road Ahead

Yequan Wang, Aixin Sun (2025)

## 🧩 Problem to Solve

본 논문은 현재의 Embodied AI가 가진 한계를 분석하고, 궁극적인 목표인 Embodied AGI(Embodied Artificial General Intelligence)에 도달하기 위한 체계적인 경로를 제시하는 것을 목표로 한다.

일반적으로 AGI는 신체적 구현(Embodiment)이 필수적이라고 여겨지지만, 현재의 로봇 시스템 및 AI 모델들은 특정 작업에 국한되거나 환경 변화에 취약하며, 인간 수준의 일반화 능력을 갖추지 못하고 있다. 특히 기존의 모델들은 시각과 언어라는 제한된 모달리티에 의존하며, 실시간 응답성과 고차원적인 인간 인지 능력이 부족한 상태이다. 따라서 본 연구는 Embodied AGI를 정의하고, 현재의 기술 수준을 객관적으로 평가할 수 있는 척도를 제공하며, 이를 달성하기 위한 기술적 요구사항과 개념적 프레임워크를 제안하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Embodied AGI로 나아가기 위한 단계적 로드맵과 이를 평가하기 위한 다차원적 기준을 정립한 것이다.

첫째, 자율주행 자동차의 단계 구분에서 영감을 얻어, 단순 작업 수행부터 완전한 범용 로봇에 이르는 **5단계 taxonomy(L1~L5)**를 제안하였다. 이를 통해 현재 기술의 위치를 파악하고 미래의 목표를 명확히 정의하였다.

둘째, Embodied AGI의 능력을 측정하기 위한 **4가지 핵심 차원(Core Dimensions)**을 정의하였다. 이는 Omnimodal capabilities, Humanoid cognitive abilities, Real-time responsiveness, Generalization으로 구성된다.

셋째, L3 이상의 고수준 Embodied AI를 구현하기 위한 **개념적 프레임워크(Conceptual Framework)**를 제안하였다. 여기에는 전방향 모달리티 스트리밍을 지원하는 모델 구조와 평생 학습(Lifelong learning) 및 물리 기반 학습(Physical-oriented training)을 포함한 훈련 패러다임이 포함된다.

## 📎 Related Works

논문은 Embodied AI의 두 가지 주류 접근 방식을 설명한다.

첫째는 **End-to-End 방식**으로, Vision-Language-Action (VLA) 모델을 사용하여 시각 및 텍스트 입력을 직접 액션 토큰으로 예측하는 방식이다. 둘째는 **Plan-and-Act 방식**으로, VLM이나 LLM을 이용해 고수준 계획을 수립하고 이를 실행 가능한 코드나 명령어로 변환하여 제어하는 방식이다. 최근에는 이 두 방식을 결합한 하이브리드 방법론이나 대규모 합성 데이터를 활용한 사전 학습 전략이 연구되고 있다.

기존 연구들의 한계점은 대부분의 모델이 시각과 언어라는 bimodal 혹은 trimodal 수준에 머물러 있으며, 물리 법칙의 내재화 부족으로 인해 환경 변화에 대한 강건성(Robustness)이 떨어지고 작업 간 일반화(Inter-task generalization) 능력이 부족하다는 점이다.

## 🛠️ Methodology

### Embodied AGI의 5단계 로드맵 (L1-L5)

본 논문은 Embodied AGI의 발전 단계를 다음과 같이 정의한다.

- **L1 (Single-task completion):** 단일한 정의된 작업(예: 물체 집기)을 안정적으로 수행하는 단계이다.
- **L2 (Compositional task completion):** 고수준 명령을 하위 작업으로 분해하여 순차적으로 수행하는 단계이다.
- **L3 (Conditional general-purpose task completion):** 다양한 작업 카테고리를 다루며 조건부 일반화가 가능하고, 실시간 응답성을 갖춘 초기 단계의 범용 지능이다.
- **L4 (Highly general-purpose robots):** 물리 세계 모델을 내재화하여 미지의 작업에 대해서도 강건한 일반화 능력을 보이며, 고도의 다중 모달 추론이 가능한 단계이다.
- **L5 (All-purpose robots):** 인간과 유사한 인지 능력(자기 인식, 사회적 연결 이해 등)을 갖추고, 인간의 개입 없이 모든 일상적 요구를 충족하는 최종 단계이다.

### L3+ 로봇 뇌를 위한 개념적 모델 구조

저자들은 L3 이상의 능력을 갖추기 위해 모든 이전 시점의 정보를 바탕으로 다중 모달 출력을 생성하는 구조를 제안한다. 이를 수학적으로 다음과 같이 정의한다.

$$y_{t+1}^{a_1}, y_{t+1}^{a_2}, \dots, y_{t+1}^{a_n} = f_\theta(x_{0 \sim t}^{b_1}, x_{0 \sim t}^{b_2}, \dots, x_{0 \sim t}^{b_m})$$

여기서 $x_{0 \sim t}^{b_j}$는 텍스트, 오디오, 이미지, 비디오, 히트맵 등 입력 모달리티들의 시계열 스트림이며, $y_{t+1}^{a_i}$는 생각(thoughts), 음성(speech), 액션(action) 등 모델이 출력하는 다중 모달 응답을 의미한다. 이는 기존의 turn-based 방식이 아닌, 실시간 스트리밍 기반의 full-duplex 상호작용을 가능하게 한다.

### 학습 패러다임

L3+ 달성을 위해 다음과 같은 훈련 전략을 제시한다.

1. **Multimodal training from scratch:** 모달리티 간의 깊은 정렬을 위해 처음부터 다중 모달 모델로 학습시킨다.
2. **Lifelong Learning:** 고정된 체크포인트 방식에서 벗어나, 능동 학습(Active learning)과 지식 편집(Knowledge editing)을 통해 지속적으로 내부 상태를 업데이트한다.
3. **Physical-oriented Training:** 단순 모방 학습을 넘어 물리 법칙을 내재화하기 위해 World Models를 확장하고 물리적 상호작용의 결과를 예측하는 학습 목표를 설정한다.

## 📊 Results

본 논문은 특정 알고리즘의 성능을 측정하는 실험 논문이 아니라, 분야 전체를 조망하는 리뷰 및 제안 논문이다. 따라서 정량적인 벤치마크 결과 대신, 기존 모델들에 대한 정성적 분석 결과를 제시한다.

저자들은 GraspVLA와 같은 최신 모델들이 L1 수준의 강건성을 보이고, Helix와 같은 시스템이 L2 수준의 복합 작업 수행 능력을 보이고 있음을 분석하였다. 또한 $\pi_{0.5}$와 같은 최신 VLA 모델들이 다양한 작업 카테고리를 다루기 시작했으나, 이는 여전히 환경적 일반화에 가깝지 진정한 작업 간 일반화(L3+)에는 도달하지 못했다고 평가한다.

결론적으로, 현재의 Embodied AI 기술 수준은 **L1과 L2 사이(L1–L2)**에 위치하고 있으며, L3로 진입하기 위해서는 실시간 응답성과 전방향 모달리티 통합이라는 거대한 기술적 격차를 해소해야 함을 명시한다.

## 🧠 Insights & Discussion

본 논문은 Embodied AGI를 달성하기 위해 단순히 모델의 크기를 키우는 것이 아니라, 인간의 인지 구조를 모방한 설계가 필요함을 역설한다. 특히 **인간형 인지 능력(Humanoid Cognition)**의 네 가지 요소인 자기 인식(Self-awareness), 사회적 연결 이해(Social connection understanding), 절차적 기억(Procedural memory), 기억 재공고화(Memory reconsolidation)가 L5 수준의 로봇을 위한 필수 요건임을 강조한 점이 인상적이다.

비판적 관점에서 볼 때, 제안된 L1~L5 로드맵은 매우 직관적이지만, 각 단계 간의 경계를 결정짓는 정량적인 지표(Metric)가 구체적으로 제시되지 않았다. 또한, 제안된 모델 구조 $f_\theta$가 가져올 엄청난 계산 복잡도와 실시간 추론 시의 레이턴시 문제를 어떻게 해결할 것인지에 대한 구체적인 아키텍처 설계안보다는 개념적 방향성에 치중되어 있다. 그럼에도 불구하고, Embodied AI의 목표 지점을 명확히 하고 이를 위해 필요한 구성 요소를 체계적으로 정리했다는 점에서 학술적 가치가 높다.

## 📌 TL;DR

본 논문은 Embodied AGI로 가기 위한 5단계 로드맵(L1~L5)과 4가지 핵심 능력 차원을 제안하며, 현재의 기술 수준이 L1~L2 단계에 머물러 있음을 분석한다. 이를 극복하기 위해 실시간 다중 모달 스트리밍 구조와 평생 학습 및 물리 법칙 내재화 학습을 포함한 L3+ 로봇 뇌의 개념적 프레임워크를 제시한다. 이 연구는 향후 범용 로봇 지능을 개발하는 데 있어 중요한 기술적 이정표와 평가 기준을 제공할 것으로 기대된다.
