# Agent-R1: Training Powerful LLM Agents with End-to-End Reinforcement Learning

Mingyue Cheng, Jie Ouyang, Shuo Yu, Ruiran Yan, Yucong Luo, Zirui Liu, Daoyu Wang, Qi Liu, Enhong Chen (2025)

## 🧩 Problem to Solve

본 논문은 대규모 언어 모델(LLM)을 도구 사용과 같은 능동적인 환경 상호작용이 가능한 에이전트(Agent)로 구축하기 위한 강화학습(Reinforcement Learning, RL) 방법론과 프레임워크의 부재 문제를 해결하고자 한다.

일반적인 LLM은 정적인 텍스트 생성 작업에 최적화되어 있으나, 에이전트로 작동하기 위해서는 다회차(multi-turn)의 의사결정, 메모리 유지, 그리고 환경으로부터 오는 확률적인 피드백에 적응하는 능력이 필요하다. 하지만 기존의 RL 적용 방식은 주로 수학 문제 풀이나 코드 생성과 같은 정적인 태스크에 집중되어 있었으며, 에이전트 설정에서의 RL 적용은 학습의 불안정성, 복잡한 보상 신호 설계, 그리고 일반화 성능 부족이라는 난관에 봉착해 있다. 따라서 본 연구의 목표는 LLM 에이전트의 특성을 반영하여 Markov Decision Process (MDP) 프레임워크를 체계적으로 확장하고, 이를 실제로 구현할 수 있는 유연하고 확장 가능한 학습 프레임워크인 Agent-R1을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 LLM 에이전트의 상호작용 특성을 수학적으로 정의한 MDP 프레임워크의 확장과, 이를 기반으로 한 모듈형 학습 프레임워크 Agent-R1의 개발이다.

중심적인 직관은 LLM 에이전트의 학습이 단순한 토큰 생성의 최적화가 아니라, '에이전트의 행동'과 '환경의 피드백'을 명확히 구분하고, 최종 결과뿐만 아니라 중간 단계의 성공 여부를 반영하는 프로세스 보상(Process Reward)을 통해 정밀한 신용 할당(Credit Assignment)을 수행해야 한다는 점이다. 이를 위해 Action Mask를 도입하여 모델이 제어할 수 없는 환경 응답 부분을 제외하고 오직 모델이 생성한 액션에 대해서만 정책 업데이트를 수행하도록 설계하였다.

## 📎 Related Works

논문은 기존의 LLM 활용 방식을 세 단계로 구분하여 설명한다. 첫째는 인간이 설계한 고정된 경로를 따르는 Workflow, 둘째는 ReAct와 같이 추론과 행동의 루프를 도입한 Agentic Workflow, 마지막으로 사전 정의된 워크플로우 없이 환경과 능동적으로 상호작용하는 Autonomous Agent이다.

기존의 RL 기반 LLM 학습(예: 수학/코드 생성)은 상태 전이가 결정론적이며 보상이 매우 희소(sparse)한 특성을 가진다. 반면, LLM 에이전트 환경은 도구 호출 결과에 따라 상태가 확률적으로 변하며, 다회차 상호작용이 발생한다. 본 논문은 이러한 동적 특성을 반영하지 못하는 기존의 정적 RL 접근 방식의 한계를 지적하며, 환경 상호작용을 내재화한 MDP 모델링의 필요성을 강조한다.

## 🛠️ Methodology

### 1. LLM 에이전트를 위한 MDP 프레임워크 확장

본 연구는 Static LLM과 LLM Agent의 차이를 MDP의 네 가지 핵심 요소로 정의하여 확장한다.

**상태 공간 (State Space, $S$)**
정적 LLM의 상태가 단순한 텍스트 컨텍스트 $s_t = (w_p, w_1, \dots, w_t)$인 것과 달리, 에이전트의 상태는 다회차 상호작용 이력과 환경 피드백을 모두 포함한다.
$$s_t = (w_p, T_1, \dots, T_k, T_{partial}^{k+1})$$
여기서 $T_i$는 에이전트의 생성 토큰과 그에 따른 환경 피드백 $w_{ei}$를 포함하는 하나의 완전한 턴(turn)을 의미한다.

**액션 공간 (Action Space, $A$)**
기본적으로는 어휘 사전 $V$에서 다음 토큰을 선택하는 것이지만, 특정 토큰 시퀀스는 외부 도구를 호출하는 명령어로 해석되어 능동적인 환경 개입을 가능하게 한다.

**상태 전이 확률 (State Transition Probability, $P$)**
정적 LLM은 토큰 추가에 따른 결정론적 전이를 보이지만, 에이전트는 환경 상호작용에 따른 확률적 전이를 포함한다.
$$P(s_{t+1}|s_t, a_t) = \begin{cases} P_E(s_{t+1}|s_t, a_t), & \text{if } a_t \text{ triggers tool/env interaction} \\ P_G(s_{t+1}|s_t, a_t), & \text{otherwise (standard generation)} \end{cases}$$
여기서 $P_G$는 결정론적인 생성 전이이며, $P_E$는 외부 API 응답 등 환경의 불확실성을 반영하는 확률적 전이다.

**보상 함수 (Reward Function, $R$)**
최종 결과 보상 $r_f$ 외에 중간 단계의 성공을 평가하는 프로세스 보상 $r_p$를 도입하여 보상을 더 조밀하게(dense) 설계한다.
$$R(s_t, a_t, s_{t+1}) = \begin{cases} r_f(s_{t+1}), & \text{if } s_{t+1} \text{ is a terminal state} \\ r_p(s_t, a_t, s_{t+1}), & \text{if } a_t \text{ triggers a significant intermediate event} \\ 0, & \text{otherwise} \end{cases}$$

### 2. Agent-R1 프레임워크 구조

Agent-R1은 다회차 롤아웃(Rollout)과 학습을 효율적으로 수행하기 위해 `Tool`과 `ToolEnv`라는 두 가지 핵심 모듈을 제안한다.

- **Tool**: API 호출, 코드 실행 등 원자적 액션을 수행하는 실행기이다. `BaseTool` 클래스를 통해 도구의 이름, 설명, JSON Schema 기반의 파라미터 구조를 표준화하여 에이전트가 도구를 정확히 선택하고 사용할 수 있게 한다.
- **ToolEnv**: RL 환경의 오케스트레이터이다. 에이전트의 출력을 받아 도구 호출 여부를 판단하고, `Tool`을 실행한 뒤 그 결과를 다시 에이전트가 이해할 수 있는 상태로 변환하여 전달한다. 또한 `step()` 메서드를 통해 상태 전이와 보상 계산을 관리한다.

### 3. 다회차 궤적을 이용한 정책 최적화

롤아웃을 통해 수집된 궤적에서 에이전트가 생성한 부분과 환경이 응답한 부분을 구분하기 위해 **Action Mask**를 사용한다.

- **Advantage 계산**: 단순한 최종 보상이 아니라, `ToolEnv`에서 수집된 프로세스 보상을 GAE(Generalized Advantage Estimation) 등의 방식에 통합하여 계산한다. 이때 Action Mask를 통해 에이전트가 실제로 액션을 취한 시점에만 Advantage가 할당되도록 정렬한다.
- **Masked Policy Optimization**: Actor 모델의 손실 함수(Loss)를 계산할 때, Action Mask를 적용하여 모델이 생성한 토큰에 대해서만 그라디언트(gradient)를 업데이트한다. 이는 환경의 응답이나 프롬프트 토큰에 대해 모델이 학습하는 것을 방지한다.
- **Critic 업데이트**: Critic 모델은 프로세스 보상과 최종 보상을 모두 포함한 누적 보상을 예측하도록 MSE 손실 함수를 통해 학습된다.

## 📊 Results

### 실험 설정

- **태스크 및 데이터셋**: Multi-hop QA (HotpotQA, 2WikiMultihopQA, Musique).
- **모델 및 도구**: Qwen2.5-3B-Instruct 모델을 사용하며, KILT Wikipedia 코퍼스를 쿼리하는 `wikisearch` 도구를 활용한다.
- **비교 대상**: Naive RAG, Base Tool Call 및 다양한 RL 알고리즘(PPO, GRPO, REINFORCE++, RLOO).
- **지표**: Exact Match (EM) score.

### 주요 결과

RL을 통해 학습된 모든 에이전트가 Base Tool Call(평균 EM 0.0847)과 Naive RAG(평균 EM 0.1328)를 압도하였다. 특히 GRPO 알고리즘이 평균 EM 0.3877로 가장 높은 성능을 보였으며, PPO(0.3719)와 RLOO(0.3716)가 그 뒤를 이었다. 특히 PPO는 도메인 외 데이터셋인 Musique에서 강점을 보였다.

### ablation Study

정책 최적화 구성 요소인 Loss Mask와 Advantage Mask의 영향을 분석한 결과, 두 마스크가 모두 활성화되었을 때 최적의 성능이 나타났다. PPO의 경우 Advantage Mask를 비활성화했을 때 평균 EM이 0.3719에서 0.3136으로 크게 하락하였으며, Loss Mask를 제거했을 때도 성능 저하가 관찰되었다. 이는 정밀한 신용 할당과 타겟팅된 그라디언트 업데이트가 에이전트 학습에 필수적임을 입증한다.

## 🧠 Insights & Discussion

본 논문은 LLM 에이전트 학습에서 가장 큰 난제 중 하나인 '누구에게 공을 돌릴 것인가'의 문제, 즉 신용 할당(Credit Assignment) 문제를 MDP의 확장과 Action Masking을 통해 효과적으로 해결하였다. 특히 프로세스 보상을 도입함으로써 에이전트가 최종 정답에 도달하기 전의 중간 단계(예: 올바른 도구 호출)에서도 학습 신호를 얻을 수 있게 하여 학습 효율을 높였다.

다만, 본 연구는 Multi-hop QA라는 특정 도메인에서 검증되었으므로, 도구의 종류가 훨씬 다양하거나 환경의 피드백이 극도로 복잡한 일반적인 자율 에이전트 시나리오에서도 동일한 수준의 안정성을 보일지는 추가적인 연구가 필요하다. 또한, 보상 함수 설계 시 프로세스 보상 $r_p$를 정의하는 기준이 사람이 설계한 규칙에 의존하고 있다는 점은 향후 자동화된 보상 모델(Reward Model) 도입을 통해 해결해야 할 과제로 보인다.

## 📌 TL;DR

본 논문은 LLM 에이전트의 다회차 상호작용과 확률적 환경 전이를 반영하여 확장된 MDP 프레임워크를 제안하고, 이를 구현한 **Agent-R1** 학습 프레임워크를 소개한다. 핵심은 **프로세스 보상**의 도입과 **Action Mask**를 통한 정밀한 정책 업데이트이며, 이를 통해 Multi-hop QA 태스크에서 기존 RAG 및 단순 도구 호출 방식보다 월등히 높은 성능을 달성하였다. 이 연구는 향후 복잡한 환경에서 스스로 진화하는 자율 LLM 에이전트를 구축하기 위한 체계적인 RL 학습 기반을 제공한다.
