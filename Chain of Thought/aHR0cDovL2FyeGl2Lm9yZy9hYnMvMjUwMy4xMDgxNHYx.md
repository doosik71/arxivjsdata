# Thinking Machines: A Survey of LLM based Reasoning Strategies

Dibyanayan Bandyopadhyay, Soham Bhattacharjee, Asif Ekbal (2025)

## 🧩 Problem to Solve

본 논문은 대규모 언어 모델(Large Language Models, LLMs)이 보여주는 언어적 능숙함(Language Proficiency)과 실제 추론 능력(Reasoning Ability) 사이의 상당한 간극을 해결하는 것을 목표로 한다. LLM은 단순한 텍스트 생성에는 능숙하지만, 복잡한 수학적 증명이나 다단계 계획 수립과 같이 논리적 사고가 필요한 작업에서는 여전히 한계를 보인다. 

이러한 추론 능력의 결여는 특히 의료, 금융, 법률, 국방과 같이 신뢰성과 정확성이 필수적인 민감한 도메인에 AI를 배치하는 데 있어 큰 걸림돌이 된다. 따라서 본 연구의 목표는 LLM과 시각-언어 모델(Vision Language Models, VLMs)이 스스로 사고하고 자신의 응답을 재평가할 수 있도록 만드는 다양한 추론 전략들을 체계적으로 분석하고 분류하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 최신 추론 모델(OpenAI o1, DeepSeek R1 등)의 등장에 맞춰 LLM 추론 전략을 세 가지 주요 패러다임인 **강화 학습(Reinforcement Learning, RL)**, **테스트 시간 연산(Test Time Compute, TTC)**, 그리고 **자기 학습(Self-Training)**으로 구분하여 체계적인 분류 체계(Taxonomy)를 제시한 것이다. 

단순한 모델 크기 확장(Scaling)보다는 추론 시간의 최적화와 효율적인 피드백 루프를 통한 성능 향상이라는 직관에 기반하여, 초심자도 쉽게 이해할 수 있도록 탑다운(Top-down) 방식의 분석과 시각적 구조를 제공한다.

## 📎 Related Works

논문은 기존의 추론 관련 서베이 연구들(Qiao et al., 2023; Huang and Chang, 2023)을 언급하며, 본 연구가 최신 모델들의 발전 사항을 반영하여 더 최신 상태의 정보를 제공한다는 점을 강조한다. 또한, 기존의 다른 최신 서베이들(Kumar et al., 2025; Li et al., 2025b)과 비교했을 때, 추론 분야 전체를 아우르는 더 명확한 분류 체계를 제공함으로써 학술적 완성도를 높였다고 주장한다.

## 🛠️ Methodology

본 논문은 LLM의 추론 능력을 향상시키기 위한 방법론을 크게 세 가지 축으로 나누어 설명한다. 이에 앞서 기초 개념으로 언어 모델의 목적 함수와 RL 및 MCTS의 기본 원리를 정의한다.

### 1. 기초 개념 및 수식
- **언어 모델 목적 함수**: 인과적 언어 모델(Causal LM)은 이전 토큰들이 주어졌을 때 다음 토큰 $x_{n+1}$이 나타날 확률을 추정하며, 손실 함수는 다음과 같이 정의된다.
$$\mathcal{L}_{LM}(\theta) = -\sum_{t=1}^{T} \log P_\theta(x_t | x_1, \dots, x_{t-1})$$
- **강화 학습(RL)**: 에이전트(LLM)가 환경과 상호작용하며 누적 보상을 최대화하는 정책 $\pi_\theta(a|s)$를 학습한다. 기대 누적 보상 $J(\pi_\theta)$는 다음과 같다.
$$J(\pi_\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]$$
- **선호도 최적화(Preference Optimization, DPO)**: 보상 모델 없이 선호 데이터 $(x, y^+, y^-)$를 직접 사용하여 모델을 업데이트한다.
$$\mathcal{L}_{PO}(\theta) = -\sum_{(x, y^+, y^-)} \log \frac{e^{r(y^+, x)}}{e^{r(y^+, x)} + e^{r(y^-, x)}}$$
- **몬테카를로 트리 탐색(MCTS)**: 선택(Selection) $\rightarrow$ 확장(Expansion) $\rightarrow$ 시뮬레이션(Simulation) $\rightarrow$ 역전파(Backpropagation) 과정을 통해 최적의 경로를 탐색한다.

### 2. 주요 추론 전략

#### ① 강화 학습 (Reinforcement Learning)
- **언어적 강화(Verbal Reinforcement)**: 자연어 형태의 피드백을 메모리에 저장하고 이를 다음 생성에 활용하는 방식이다(예: ReAct, Reflexion).
- **보상 기반 강화(Reward-based Reinforcement)**: 
    - **과정 감독(Process Supervision)**: 최종 결과뿐 아니라 각 추론 단계마다 보상을 부여하여 더 일관된 추론 체인을 유도한다.
    - **결과 감독(Outcome Supervision)**: 최종 정답 여부에만 보상을 부여한다. 최근 DeepSeek-R1과 같은 모델들이 이 방식만으로도 높은 성능을 보였음을 언급한다.
- **탐색 및 계획(Search/Planning)**: MCTS를 결합하여 미래의 결과를 시뮬레이션하고 최적의 행동을 선택한다.

#### ② 테스트 시간 연산 (Test Time Compute, TTC)
학습 단계의 가중치 업데이트 없이 추론 시점에 더 많은 계산 자원을 투입하는 전략이다.
- **피드백 기반 개선**: 외부 도구(코드 실행기, 지식 베이스)나 다른 모델의 피드백을 받아 응답을 수정한다.
- **연산 확장(Scaling/Adaptation)**: 
    - **Token-level Scaling**: Chain-of-Thought (CoT)를 넘어 Tree-of-Thought (ToT), Graph-of-Thought (GoT)와 같이 추론 경로를 병렬적으로 확장하고 최적의 경로를 선택한다.
    - **자기 피드백(Self-Feedback)**: 모델이 스스로 생성한 답변을 스스로 검토하고 수정하는 루프를 수행한다.

#### ③ 자기 학습 (Self-Training)
모델이 스스로 생성한 추론 경로 중 정답인 것들을 선택하여 다시 학습 데이터로 사용하는 방식이다.
- **절차**: Supervised Fine-Tuning (SFT) $\rightarrow$ Rejection Fine-Tuning (정답인 경로만 선택) $\rightarrow$ Preference Tuning (DPO 등을 통해 정답 경로 선호 유도) $\rightarrow$ RL (보상 최대화) 순으로 진행된다. (예: STaR, v-STaR)

## 📊 Results

본 논문은 서베이 논문이므로 직접적인 실험 결과보다는 기존 연구들의 성과를 종합하여 제시한다.

- **과정 감독 vs 결과 감독**: 전통적으로는 과정 감독(Process Supervision)이 결과 감독보다 우수하다고 알려졌으나, 최근 DeepSeek-R1과 같은 모델들이 결과 기반 보상과 대규모 RL만으로도 강력한 추론 능력을 획득했음을 보여주었다.
- **TTC의 효율성**: 테스트 시간의 연산량을 늘리는 것이 단순히 모델 파라미터 수를 늘리는 것보다 특정 추론 작업에서 더 효과적일 수 있다는 결과들이 보고되었다.
- **모델 크기의 제약**: CoT와 같은 테스트 시간 확장 기법은 주로 100B 이상의 대형 모델에서 효과적이며, 10B 미만의 소형 모델에서는 오히려 성능이 저하되는 경우가 있다는 점이 명시되었다.

## 🧠 Insights & Discussion

### 강점 및 기회
- 본 연구는 LLM 추론의 메커니즘을 학습(Training)과 추론(Inference) 단계로 명확히 구분하여, 최신 트렌드인 '추론 시간 연산 확장'의 중요성을 잘 짚어냈다.
- 특히 RL, TTC, Self-Training이라는 세 가지 축을 통해 파편화되어 있던 추론 기법들을 하나의 프레임워크로 통합한 점이 돋보인다.

### 한계 및 도전 과제
- **과정 감독의 자동화 문제**: 단계별 보상을 위한 정밀한 레이블링은 여전히 인간의 노동 집약적인 작업에 의존하고 있어, 이를 자동화하는 것이 시급하다.
- **과잉 사고(Overthinking)**: MCTS와 같은 탐색 기반 기법은 탐색 공간이 너무 넓을 경우 불필요한 계산을 수행하여 비효율적일 수 있다.
- **사전 학습의 의존성**: TTC가 강력하더라도 기반 모델(Base Model)의 사전 학습(Pre-training) 품질이 낮으면 추론 시점의 연산 확장만으로는 한계를 극복할 수 없다.

## 📌 TL;DR

본 논문은 LLM의 언어 능력과 추론 능력 사이의 간극을 메우기 위한 전략들을 **강화 학습(RL), 테스트 시간 연산(TTC), 자기 학습(Self-Training)**의 세 가지 관점에서 종합적으로 분석한 서베이 보고서이다. 특히 최근의 o1, R1 모델들이 보여준 '추론 시간의 확장'과 '결과 기반 RL'의 중요성을 강조하며, 향후 AGI로 나아가기 위해 해결해야 할 과정 감독의 자동화 및 연산 효율성 문제를 제시한다. 이 연구는 LLM 추론 분야의 최신 지형도를 제공함으로써 향후 고도화된 추론 모델 설계의 가이드라인 역할을 할 가능성이 높다.