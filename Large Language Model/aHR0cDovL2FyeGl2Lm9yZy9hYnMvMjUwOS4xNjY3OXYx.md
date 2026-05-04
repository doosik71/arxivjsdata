# Reinforcement Learning Meets Large Language Models: A Survey of Advancements and Applications Across the LLM Lifecycle

Keliang Liu et al. (2025)

## 🧩 Problem to Solve

최근 Large Language Models(LLMs)는 다양한 작업에서 뛰어난 성능을 보이고 있으나, 여전히 인간의 세밀한 의도를 정확히 포착하지 못하거나 오해의 소지가 있는 안전하지 않은 출력을 생성하는 등의 한계가 존재한다. 특히 복잡한 추론(reasoning) 능력에서 상당한 결함이 발견되고 있으며, 이는 모델이 단순히 다음 토큰을 예측하는 수준을 넘어 논리적이고 정밀한 문제 해결 능력을 갖춰야 함을 시사한다.

기존의 서베이 논문들은 강화학습(Reinforcement Learning, RL)을 이용한 LLM의 정렬(Alignment) 기술이나 추론 시점(Inference-time)의 RL에 국한되어 다루는 경향이 있었다. 따라서 본 논문은 LLM의 전체 생애주기(Lifecycle)—즉, 사전 학습(Pre-training), 정렬 미세 조정(Alignment Fine-tuning), 그리고 강화된 추론(Reinforced Reasoning) 단계—전반에 걸쳐 RL이 어떻게 적용되고 모델의 성능을 끌어올리는지를 체계적으로 분석하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 LLM 개발의 전 과정을 포괄하는 RL 적용 프레임워크를 제시한 것이다. 특히 다음과 같은 직관과 설계 아이디어를 중심으로 논의를 전개한다.

1.  **생애주기 관점의 체계화**: RL을 단순한 후처리 단계가 아니라 사전 학습부터 추론 최적화까지 이어지는 일련의 흐름으로 정의하고, 각 단계에서의 목적과 방법론을 분류하였다.
2.  **RLVR(Reinforcement Learning with Verifiable Rewards)의 강조**: 최근 OpenAI-o1이나 DeepSeek-R1과 같은 모델에서 핵심이 된 RLVR 패러다임을 집중 분석한다. 이는 사람이 매긴 주관적 점수가 아닌, 프로그램 실행 결과나 수학적 정답 확인과 같이 '객관적으로 검증 가능한 보상'을 통해 모델의 추론 능력을 한계치까지 밀어붙이는 방식이다.
3.  **리소스 통합**: RL 기반 LLM 학습에 필수적인 데이터셋, 벤치마크, 오픈소스 프레임워크를 집대성하여 연구자들이 즉시 활용할 수 있는 로드맵을 제공한다.

## 📎 Related Works

기존 연구들은 주로 RLHF(Reinforcement Learning from Human Feedback)를 통한 인간 선호도 정렬에 집중해 왔다. 일부 최신 서베이들이 추론 시점의 RL을 다루기 시작했으나, 이는 단편적인 접근에 그쳤으며 사전 학습 단계에서의 RL 적용이나 다중 모달리티(Multimodal)로의 확장성을 충분히 다루지 못했다.

본 논문은 기존 접근 방식과 달리, LLM의 능력이 단순히 데이터 양에 의해 결정되는 것이 아니라, 적절한 RL 루프를 통해 '사고 과정(Chain-of-Thought)'을 최적화함으로써 비약적으로 상승할 수 있음을 강조하며, 이를 전 생애주기 관점에서 통합적으로 분석한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. RL의 기초 이론 및 LLM 적용
LLM의 RL은 기본적으로 Markov Decision Process(MDP)로 모델링된다. 모델(Agent)은 상태(State, 입력 프롬프트)에서 행동(Action, 토큰 생성)을 취하고, 환경으로부터 보상(Reward)을 받으며 누적 보상을 최대화하는 정책 $\pi$를 학습한다.

#### 정책 학습(Policy Learning)
- **PPO (Proximal Policy Optimization)**: 정책 업데이트가 너무 급격하게 일어나 모델이 붕괴되는 것을 막기 위해 Clipped Surrogate Objective를 사용한다.
- **GRPO (Group Relative Policy Optimization)**: DeepSeek-Math에서 제안된 방식으로, 별도의 가치 네트워크(Value Network) 없이 하나의 프롬프트에 대해 여러 개의 응답 그룹을 생성하고, 그룹 내의 상대적 보상을 통해 이점(Advantage)을 계산한다.
  - 그룹 내 보상의 평균을 기준점(Baseline)으로 삼아, 평균보다 높은 보상을 받은 응답의 확률은 높이고 낮은 응답은 낮춘다.
  - 이점 계산식: $\hat{A}_{i,t} = \frac{r_i - \max(\{r_j\}_{j=1}^G)}{\text{std}(\{r_j\}_{j=1}^G)}$ (단, $G$는 그룹 크기)

#### 가치 학습(Value Learning)
- Q-learning, DQN 등 가치 함수를 추정하는 방식이 있으나, LLM은 행동 공간(Action Space, 모든 가능한 토큰 조합)이 너무 방대하여 직접적인 Q-table 구축이 어렵다. 따라서 주로 정책 기반 방법이나 Reward Model을 통한 간접적 가치 추정이 사용된다.

### 2. 생애주기별 RL 적용 전략

#### 사전 학습 단계 (Pre-training)
다음 토큰 예측(Next-token prediction) 작업을 RL 기반의 추론 작업으로 재구성하여, 올바른 예측 시 검증 가능한 보상을 제공함으로써 초기 단계부터 추론 능력을 배양한다.

#### 정렬 단계 (Alignment)
- **RLHF**: 인간의 선호도를 학습한 Reward Model을 통해 모델의 출력 방향을 조정한다.
- **DPO (Direct Preference Optimization)**: 명시적인 Reward Model 없이 선호도 데이터를 통해 직접 정책을 최적화하여 복잡성과 불안정성을 줄인다.
- **RLAIF**: 인간 대신 AI가 생성한 피드백을 사용하여 확장성을 높인다.

#### 추론 강화 단계 (Reinforced Reasoning/RLVR)
검증 가능한 보상(Verifiable Rewards)을 사용하여 모델이 정답에 도달할 때까지 스스로 사고 과정을 반복하도록 유도한다.
- **Multimodal Reasoning**: 시각 정보와 텍스트 추론을 결합하여 Vision-R1과 같은 모델을 구축한다.
- **Adaptive Thinking**: 문제의 난이도에 따라 사고 과정의 길이를 스스로 조절하는 Adaptive Length Reasoning을 학습시킨다.
- **Agent RL**: 도구 사용(Tool-use) 및 장기적 의사결정(Long-horizon decision making)을 위해 지연된 보상(Delayed Reward) 문제를 해결하는 RL 전략을 사용한다.

## 📊 Results

### 정량적 성능 향상
논문에서 제시한 Table 1에 따르면, RL을 적용한 모델들은 베이스라인 대비 압도적인 성능 향상을 보인다.
- **OpenAI-o1**: AIME2024 벤치마크에서 9.3% $\rightarrow$ 79.2%로 급증.
- **DeepSeek-R1**: MATH-500에서 90.2% $\rightarrow$ 97.3%로 향상되었으며, AIME2024에서는 39.2%에서 79.8%로 크게 상승하였다.

### 주요 벤치마크 및 지표
- **수학/코딩**: GSM8K, MATH, AIME, LiveCodeBench 등을 통해 논리적 정확성을 측정한다.
- **일반 지식/추론**: MMLU, GPQA-Diamond, HLE(Humanity's Last Exam) 등을 통해 고도의 전문 지식 및 추론 능력을 평가한다.
- **정렬**: IFEval, Arena-Hard 등을 통해 인간의 지시사항 준수 여부를 측정한다.

## 🧠 Insights & Discussion

### 강점 및 가능성
RL, 특히 RLVR은 모델이 단순히 학습 데이터를 암기하는 것이 아니라, 보상 신호를 통해 최적의 해결 경로를 스스로 탐색하게 함으로써 '창발적 추론 능력'을 끌어낼 수 있음을 보여주었다. 특히 GRPO와 같은 알고리즘은 가치 네트워크의 메모리 오버헤드를 제거하여 대규모 모델의 RL 학습 효율성을 극대화하였다.

### 한계 및 논의 사항
1.  **능력의 확장 vs 샘플링 효율**: RLVR이 실제로 모델에 없던 새로운 추론 능력을 생성하는 것인지, 아니면 이미 사전 학습된 분포 내에서 정답일 확률이 높은 경로를 더 잘 샘플링하게 만드는 것인지에 대한 학술적 논쟁이 있다.
2.  **엔트로피 붕괴(Entropy Collapse)**: 대규모 RL 학습 시 정책의 엔트로피가 급격히 낮아지며 다양성이 사라지는 현상이 발생한다. 이를 해결하기 위한 엔트로피 관리 및 고엔트로피 토큰(논리 연결어 등)의 선택적 업데이트 전략이 필요하다.
3.  **보상 해킹(Reward Hacking)**: 모델이 실제 문제를 해결하지 않고 보상 함수의 허점을 이용해 높은 점수를 얻으려는 경향이 있으며, 이를 방지하기 위한 정교한 Reward Shaping이 필수적이다.

## 📌 TL;DR

본 논문은 LLM의 **사전 학습 $\rightarrow$ 정렬 $\rightarrow$ 추론 강화**로 이어지는 전체 생애주기에 걸친 강화학습(RL) 적용 방안을 체계적으로 분석한 서베이 보고서이다. 특히 **검증 가능한 보상(RLVR)**과 **GRPO** 알고리즘이 현대의 고성능 추론 모델(o1, R1 등)의 핵심 동력임을 밝히고, 이를 위한 데이터셋과 프레임워크를 통합적으로 제시하였다. 이 연구는 향후 LLM이 단순한 텍스트 생성기를 넘어, 스스로 생각하고 수정하는 **자율적 추론 에이전트**로 진화하는 데 필요한 기술적 이정표를 제공한다.