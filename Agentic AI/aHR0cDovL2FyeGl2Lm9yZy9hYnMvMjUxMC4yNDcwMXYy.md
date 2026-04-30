# Tongyi DeepResearch Technical Report

Tongyi DeepResearch Team (2025)

## 🧩 Problem to Solve

본 논문은 복잡하고 광범위한 정보 탐색이 필요한 **Long-horizon, Deep Information-seeking Research** 작업을 자율적으로 수행할 수 있는 에이전트 모델의 개발을 목표로 한다. Deep Research는 단순한 질의응답을 넘어, 인터넷 상에서 다단계 추론과 정보 탐색을 자율적으로 수행하여 수 시간에 걸쳐 사람이 수행해야 할 연구 과제를 수십 분 내에 완료하는 능력을 의미한다.

현재 이러한 기능을 갖춘 시스템들은 대부분 폐쇄형 소스(Closed-source)로 운영되고 있으며, 내부적인 연구 프로세스가 공개되지 않아 학계의 체계적인 방법론 정립과 공유가 부족한 상황이다. 따라서 본 연구의 목적은 자율적인 계획 수립, 검색, 추론 및 지식 합성을 가능하게 하는 **Open-source AI Researcher** 모델을 구축하고, 이를 위한 확장 가능한 훈련 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **Agentic Mid-training**과 **Agentic Post-training**을 결합한 엔드-투-엔드 훈련 패러다임을 통해, 적은 파라미터 활성화만으로도 최첨단(SOTA) 성능을 내는 연구 에이전트를 구현한 것이다.

중심적인 설계 아이디어는 다음과 같다.
1.  **에이전트 중심의 단계적 훈련**: 일반적인 LLM이 갖지 못한 에이전트적 귀납 편향(Agentic inductive bias)을 부여하기 위해 Mid-training 단계를 도입하여 Pre-training과 Post-training 사이의 간극을 메웠다.
2.  **완전 자동화된 합성 데이터 파이프라인**: 인간의 어노테이션 없이도 연구 수준의 고난도 질문과 에이전트 궤적(Trajectory)을 생성할 수 있는 확장 가능한 데이터 합성 시스템을 구축하였다.
3.  **환경 기반의 학습 전략**: Prior World, Simulated, Real-world라는 세 가지 형태의 맞춤형 환경을 설계하여, 학습의 안정성과 비용 효율성을 동시에 확보하며 현실 세계의 피드백을 반영하도록 하였다.

## 📎 Related Works

본 연구는 기본적으로 추론과 행동을 결합한 **ReAct (Reasoning and Acting)** 프레임워크를 기반으로 한다. 기존의 많은 에이전트 연구들이 복잡한 프롬프트 엔지니어링이나 정교하게 설계된 인간의 지식 구조에 의존하는 경향이 있었으나, 본 논문은 계산량의 확장이 결국 성능 향상으로 이어진다는 "The Bitter Lesson"의 철학을 따라, 단순하지만 확장 가능한 일반적 방법론을 채택하였다.

또한, 강화학습 알고리즘으로는 **GRPO (Group Relative Policy Optimization)**를 기반으로 한 최적화를 수행하며, 이는 최근 DeepSeek-R1 등에서 보여준 추론 능력 강화 방식과 궤를 같이 한다. 기존 접근 방식과의 차별점은 단순히 Post-training에 집중하는 것이 아니라, **Agentic CPT (Continual Pre-training)**라는 Mid-training 단계를 통해 모델이 에이전트로서의 기본 소양을 먼저 갖추게 한 뒤 RL을 적용했다는 점이다.

## 🛠️ Methodology

### 1. 시스템 정식화 (Formulation)
Tongyi DeepResearch의 각 타임스텝 $t$에서의 롤아웃(Rollout)은 다음 세 가지 구성 요소로 정의된다.
- **Thought ($\tau_t$):** 현재 맥락 분석, 메모리 회상, 계획 수립 및 자기 성찰을 포함하는 내부 인지 프로세스이다.
- **Action ($a_t$):** Search, Visit, Python Interpreter, Google Scholar, File Parser 등의 도구를 사용하는 외부 조작이다. 최종 단계 $a_T$는 사용자에게 제공할 최종 보고서 작성이 된다.
- **Observation ($o_t$):** 행동 후 환경으로부터 받는 피드백이며, 이는 다음 Thought의 입력이 된다.

전체 궤적 $H_T$는 다음과 같이 표현된다.
$$H_T = (\tau_0, a_0, o_0, \dots, \tau_i, a_i, o_i, \dots, \tau_T, a_T)$$

### 2. 맥락 관리 (Context Management)
장기 과제 수행 시 발생하는 컨텍스트 윈도우 초과 문제를 해결하기 위해 **Markovian state reconstruction** 기반의 맥락 관리 패러다임을 도입하였다. 모델은 전체 이력을 모두 참조하는 대신, 다음의 요소들로 구성된 재구성된 작업 공간(Workspace)만을 참조한다.
- 질문 $q$
- 압축된 메모리 역할을 하는 진화하는 보고서 $S_t$
- 직전 상호작용의 맥락 $(a_t, o_t)$

업데이트 과정은 다음과 같다.
$$S_t, \tau_{t+1}, a_{t+1} \sim \pi(\cdot | S_{t-1}, a_t, o_t)$$

### 3. 훈련 파이프라인 (Training Recipe)
모델은 Qwen3-30B-A3B-Base를 기반으로 하며, 다음의 단계를 거친다.

#### A. Agentic Mid-training (Agentic CPT)
- **목적:** 에이전트적 행동에 대한 강한 귀납 편향을 부여한다.
- **절차:** 32K에서 128K로 컨텍스트 길이를 확장하며 두 단계의 Continual Pre-training을 수행한다. Next-Token Prediction 손실 함수를 사용하며, 일반 프리트레이닝 데이터를 일부 섞어 일반화 성능을 유지한다.
- **데이터 합성:** 질문 생성 $\rightarrow$ 계획 수립(Planning) $\rightarrow$ 추론(Reasoning) $\rightarrow$ 의사결정(Decision-making)의 전 과정을 포괄하는 대규모 합성 데이터를 생성하여 학습시킨다.

#### B. Agentic Post-training
- **SFT (Cold Start):** 고성능 오픈소스 모델이 생성한 궤적 중 리젝션 샘플링(Rejection Sampling)을 통해 엄선된 데이터를 사용하여 초기 정책을 학습한다. ReAct 모드와 Context Management 모드를 혼합하여 학습시킨다.
- **Agentic RL:** GRPO를 변형한 알고리즘을 적용한다. 정답 일치 여부에 따른 이진 보상(Binary Reward, 0 또는 1)만을 사용한다.
  - **손실 함수:** 
    $$J(\theta) = \mathbb{E}_{(q,y) \sim D, \{H_i\}_{i=1}^G \sim \pi_{\theta_{old}}} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|H_i|} \sum_{j=1}^{|H_i|} \min \left( r_{i,j}(\theta) \hat{A}_{i,j}, \text{clip}(r_{i,j}(\theta), 1-\epsilon_{low}, 1+\epsilon_{high}) \hat{A}_{i,j} \right) \right]$$
  - 여기서 $\hat{A}_{i,j} = R_i - \text{mean}(\{R_i\}_{i=1}^G)$ 로 이득(Advantage)을 추정한다.
- **Dynamic Data Curation:** 모델의 성능 향상에 따라 학습 데이터를 실시간으로 교체한다. 너무 쉽거나 너무 어려운 문제는 제외하고, 현재 모델 수준에서 적절히 도전적인 문제들로 데이터셋을 지속적으로 갱신하는 데이터 플라이휠(Data Flywheel) 구조를 가진다.

#### C. Model Merging
서로 다른 성능 선호도를 가진 여러 모델 변체들을 가중 평균하여 최종 모델을 생성한다.
$$\theta_{merged} = \sum_k \alpha_k \cdot \theta^{(k)}, \text{ s.t. } \sum_k \alpha_k = 1, \alpha_k \geq 0$$

## 📊 Results

### 1. 정량적 성능 평가
Tongyi DeepResearch (30B-A3B)는 총 30.5B 파라미터 중 토큰당 3.3B만 활성화되는 MoE 구조임에도 불구하고, 다수의 벤치마크에서 SOTA를 달성하였다.

- **주요 결과 (Avg@3):**
  - Humanity's Last Exam: 32.9 (OpenAI DeepResearch 26.6 대비 우세)
  - BrowseComp: 43.4
  - GAIA: 70.9
  - xbench-DeepSearch: 75.0
  - FRAMES: 90.6

특히 OpenAI o3, DeepSeek-V3.1과 같은 강력한 베이스라인을 능가하거나 대등한 성능을 보였다.

### 2. Heavy Mode (Test-time Scaling)
테스트 단계에서 연산량을 늘려 성능을 극대화하는 Heavy Mode를 제안하였다.
- **구조:** $n$개의 에이전트를 병렬로 배치하여 서로 다른 경로로 연구를 수행하게 한 뒤, 각 에이전트가 생성한 압축 보고서 $S_T^u$들을 합성 모델(Synthesis Model)이 통합하여 최종 답변을 도출한다.
- **결과:** Humanity's Last Exam에서 38.3%, BrowseComp-ZH에서 58.1%로 성능이 크게 향상되어, Test-time scaling의 효과를 입증하였다.

### 3. 분석 결과
- **상호작용 횟수와 성능:** 상호작용 횟수(Interaction turns)가 증가할수록, 즉 컨텍스트 길이가 늘어날수록 BrowseComp 등의 벤치마크에서 성능이 지속적으로 향상되는 Scaling Curve를 확인하였다.
- **시뮬레이션 환경의 유효성:** Wiki 기반 시뮬레이션 환경에서의 보상 곡선이 실제 환경의 곡선과 매우 유사하게 나타나, 효율적인 알고리즘 반복 실험이 가능함을 보였다.

## 🧠 Insights & Discussion

### 강점 및 성과
본 논문은 매우 효율적인 파라미터 규모(활성 3.3B)로 거대 모델들을 압도하는 성능을 낸 점이 인상적이다. 특히 **Mid-training $\rightarrow$ SFT $\rightarrow$ RL**로 이어지는 단계적 학습 체계와, 환경의 특성을 고려한 **Prior $\rightarrow$ Simulated $\rightarrow$ Real** 환경 전략이 유기적으로 결합되어 안정적인 학습을 가능케 했다. 또한, 단순한 알고리즘의 개선보다 **데이터의 품질과 훈련 환경의 안정성**이 에이전트 성능에 더 결정적인 영향을 미친다는 실무적인 통찰을 제공한다.

### 한계 및 미해결 과제
- **컨텍스트 길이의 한계:** 현재의 128K 윈도우로는 극도로 복잡한 장기 과제를 처리하기에 여전히 부족함이 있으며, 더 확장된 컨텍스트 관리 메커니즘이 필요하다.
- **모델 규모의 확장성:** 현재는 소형 모델의 효율성을 증명했으나, 더 큰 규모의 모델에서도 동일한 에이전트 학습 프레임워크가 동일한 효율을 낼지는 추가 검증이 필요하다.
- **보고서 생성 품질:** 정답률은 높지만, 최종 리포트의 충실도(Fidelity)와 사용자 선호도 정렬(Alignment) 부분은 여전히 개선의 여지가 있다.

## 📌 TL;DR

Tongyi DeepResearch는 **Agentic Mid-training과 Post-training을 통합한 엔드-투-엔드 학습 프레임워크**를 통해 구축된 오픈소스 연구 에이전트이다. 완전 자동화된 합성 데이터 파이프라인과 다단계 환경 설계를 통해, **단 3.3B의 활성 파라미터만으로 OpenAI o3 등 최첨단 폐쇄형 모델들을 능가하는 Deep Research 성능**을 달성하였다. 이 연구는 향후 자율적인 정보 탐색 및 지식 합성 에이전트의 민주화와 확장에 중요한 기준점이 될 가능성이 높다.