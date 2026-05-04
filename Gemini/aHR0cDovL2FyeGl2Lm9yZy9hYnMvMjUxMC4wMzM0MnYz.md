# Gemini Robotics 1.5: Pushing the Frontier of Generalist Robots with Advanced Embodied Reasoning, Thinking, and Motion Transfer

Gemini Robotics Team, Google DeepMind (2025)

## 🧩 Problem to Solve

범용 로봇(General-purpose robots)이 실세계에서 유용하게 작동하기 위해서는 물리적 세계에 대한 깊은 이해, 고도의 추론 능력, 그리고 일반적이면서도 정교한 제어 능력이 필수적이다. 그러나 기존의 로봇 학습 체계는 다음과 같은 주요 문제점에 직면해 있다.

첫째, 로봇 데이터의 희소성 문제이다. 특정 로봇 환경에서 수집된 데이터는 양이 제한적이며, 이는 모델의 일반화 성능을 저해한다. 둘째, 서로 다른 형태의 로봇(Embodiment) 간의 지식 전이가 어렵다는 점이다. 셋째, 복잡하고 단계가 많은 long-horizon 태스크를 수행할 때, 단순한 입출력 구조의 모델은 계획 수립 및 오류 복구 능력이 부족하여 성공률이 낮다는 문제가 있다.

본 논문의 목표는 이러한 한계를 극복하기 위해 고도의 Embodied Reasoning(ER) 능력과 Motion Transfer(MT) 메커니즘, 그리고 '행동 전 사고(Thinking before acting)' 프로세스를 결합한 새로운 로봇 파운데이션 모델 제품군인 Gemini Robotics 1.5를 제안하는 것이다.

## ✨ Key Contributions

본 연구의 핵심적인 기여는 다음 세 가지 혁신적인 설계 아이디어로 요약된다.

1. **Motion Transfer (MT) 메커니즘**: 서로 다른 하드웨어 구조를 가진 다수의 로봇(ALOHA, Bi-arm Franka, Apollo humanoid) 데이터로부터 동시에 학습하여, 특정 로봇에 국한되지 않는 통합된 물리적 운동 이해도를 구축한다. 이를 통해 한 로봇에서 학습한 기술을 다른 로봇으로 전이하는 zero-shot skill transfer를 가능하게 한다.
2. **Thinking VLA (Embodied Thinking)**: VLA(Vision-Language-Action) 모델이 행동을 출력하기 전, 자연어 형태로 내부 추론 과정을 생성하도록 설계하였다. 이는 복잡한 지시사항을 하위 단계로 분해하고, 행동의 투명성을 높이며, 상황 인식 기반의 복구 행동을 가능하게 한다.
3. **SOTA Embodied Reasoning 모델 (GR-ER 1.5)**: 시각적-공간적-시간적 이해, 태스크 계획 및 진행률 추정 등 로봇에게 필수적인 '신체화된 추론' 능력에서 최첨단 성능을 달성한 VLM을 구축하였다.

## 📎 Related Works

본 연구는 이전 세대인 Gemini Robotics 모델을 기반으로 하며, 기존의 VLA 모델들이 가졌던 단순 제어 중심의 한계를 극복하고자 한다. 기존 연구들은 대개 특정 로봇 플랫폼에 특화된 post-training을 거쳐야 했으나, 본 논문은 다중 embodiment 데이터를 통합 학습함으로써 범용성을 확보하였다.

또한, 기존의 VLM들이 일반적인 시각-언어 이해도는 높으나 실제 물리적 공간에 대한 정밀한 추론(예: 정밀한 포인팅, 물리적 제약 조건 고려)에는 약하다는 점을 지적하며, 이를 위해 최적화된 GR-ER 1.5 모델을 통해 Embodied Reasoning의 성능을 극대화하였다.

## 🛠️ Methodology

### 1. 시스템 아키텍처: Agentic Framework

본 연구는 고수준의 계획을 담당하는 **Orchestrator**와 저수준의 실행을 담당하는 **Action Model**이 협력하는 에이전트 구조를 채택한다.

- **Orchestrator (GR-ER 1.5)**: 사용자의 입력과 환경 피드백을 처리하여 전체 태스크 흐름을 제어한다. 복잡한 작업을 VLA가 실행 가능한 단순 단계로 분해하고, 성공 여부를 감지하며, 필요한 경우 웹 검색과 같은 외부 도구를 사용한다.
- **Action Model (GR 1.5)**: Orchestrator의 지시를 받아 실제 로봇의 저수준 동작으로 변환한다.

### 2. Embodied Thinking (Thinking VLA)

GR 1.5는 단순히 $\text{Observation} \rightarrow \text{Action}$으로 이어지는 것이 아니라, 중간에 자연어 기반의 사고 과정을 삽입한다.
$$\text{Observation} \rightarrow \text{Natural Language Thoughts} \rightarrow \text{Action}$$
이 프로세스는 복잡한 지시사항을 원시 스킬(primitive skills)의 시퀀스로 단순화하며, 로봇이 스스로 행동의 성공/실패를 판단하고 복구 전략을 세울 수 있게 한다.

### 3. Motion Transfer (MT)

다양한 로봇 데이터 소스로부터 통합된 물리적 상호작용 이해도를 형성하기 위한 새로운 아키텍처와 학습 레시피를 도입하였다. MT는 서로 다른 embodiment 간의 공통점을 추출하고 정렬함으로써, 데이터가 부족한 로봇 플랫폼이 다른 플랫폼의 데이터를 통해 성능을 향상시킬 수 있도록 돕는다.

### 4. 학습 데이터 및 평가 절차

- **데이터**: ALOHA, Bi-arm Franka, Apollo humanoid 로봇 데이터 및 인터넷의 대규모 텍스트, 이미지, 비디오 데이터를 사용하였다.
- **평가**: 실제 로봇을 이용한 A/B/n 테스트를 수행하여 분산을 줄였으며, 개발 속도를 높이기 위해 MuJoCo 시뮬레이터를 활용하였다. 시뮬레이션과 실제 환경 간의 순위 일관성(rank consistency)이 높음을 확인하여 개발 과정의 90% 이상을 시뮬레이션에서 진행하였다.

## 📊 Results

### 1. 일반화 성능 및 Cross-Embodiment Transfer

GR 1.5는 시각적(Visual), 지시사항(Instruction), 행동(Action), 태스크(Task)의 네 가지 축에서 기존 베이스라인 모델보다 월등한 일반화 성능을 보였다. 특히 Motion Transfer 메커니즘을 적용했을 때, 한 로봇에서만 학습된 기술을 다른 로봇이 zero-shot으로 수행하는 능력이 크게 향상되었다. (예: Bi-arm Franka에서 학습된 수직 패널 작업 기술이 ALOHA 로봇으로 전이됨)

### 2. Thinking VLA의 효과

Multi-step 벤치마크 결과, 'Thinking' 모드를 활성화했을 때 Progress Score가 유의미하게 상승하였다. 이는 복잡한 교차 모달 변환(Cross-modal translation)을 $\text{시각 정보} \rightarrow \text{언어적 사고} \rightarrow \text{행동}$의 두 단계로 분해함으로써 강건성을 높였기 때문이다.

### 3. Embodied Reasoning (GR-ER 1.5) 성능

GR-ER 1.5는 다음과 같은 지표에서 SOTA를 달성하였다.

- **Complex Pointing**: 물리적/공간적/시맨틱 제약을 고려한 정밀 포인팅 능력에서 GPT-5 및 Gemini 2.5 Pro를 압도하였다.
- **Progress Understanding**: 다양한 뷰(Multiview)에서의 성공 감지(Success Detection) 및 비디오 프레임 재배열(Unshuffling) 작업에서 높은 정확도를 보였다.
- **Inference-time Compute**: 사고 토큰(thinking tokens) 예산을 늘릴수록 Embodied Reasoning 성능이 향상되는 scaling law를 확인하였다.

### 4. Long-horizon Task 수행 능력

GR-ER 1.5(Orchestrator)와 GR 1.5(Action Model)를 결합한 에이전트 시스템은 "쓰레기 분류", "캐리어 짐 싸기"와 같은 복잡한 태스크에서 베이스라인 대비 약 2배에 가까운 Progress Score를 기록하였다. 특히 고수준 계획(Planning) 단계에서의 오류율을 크게 낮추어 전체 시스템의 신뢰도를 높였다.

## 🧠 Insights & Discussion

본 연구는 물리적 에이전트를 구축함에 있어 **'고수준의 신체화된 추론(ER)'**과 **'범용적인 저수준 제어(VLA)'**의 결합이 필수적임을 입증하였다. 단순히 거대 VLM을 오케스트레이터로 사용하는 것보다, 로봇 특화 ER 모델을 사용하는 것이 계획 수립 및 성공 감지 단계에서 훨씬 효율적이다.

또한, 'Thinking' 프로세스의 도입은 단순한 성능 향상을 넘어, 로봇의 내부 상태를 인간이 이해할 수 있는 언어로 출력함으로써 상호작용의 투명성과 안전성을 확보할 수 있다는 중요한 통찰을 제공한다.

**한계점 및 향후 연구**:
본 모델은 일반화 성능은 매우 뛰어나지만, 정밀한 조작 능력(Dexterity)은 이전 세대와 비슷한 수준에 머물러 있다. 저자들은 이를 해결하기 위해 강화 학습(RL)이나 새로운 아키텍처를 도입하여 범용성을 유지하면서도 정교함을 높이는 연구가 필요함을 언급하였다. 또한, 더 많은 양의 인간 비디오 데이터 및 합성 데이터를 활용하여 데이터 희소성 문제를 더욱 완화할 계획이다.

## 📌 TL;DR

본 논문은 범용 로봇을 위한 모델 제품군인 **Gemini Robotics 1.5**를 제안한다. 핵심 기여는 $\text{(1)}$ 서로 다른 로봇 간 기술 전이를 가능케 하는 **Motion Transfer**, $\text{(2)}$ 행동 전 사고 과정을 거치는 **Thinking VLA**, $\text{(3)}$ 공간/물리 추론에 특화된 **GR-ER 1.5** 모델의 개발이다. 이를 결합한 에이전트 시스템은 복잡한 long-horizon 작업을 성공적으로 수행하며, 이는 향후 인간 수준의 지능과 적응력을 가진 물리적 AI 에이전트 구현을 위한 핵심적인 경로를 제시한다.
