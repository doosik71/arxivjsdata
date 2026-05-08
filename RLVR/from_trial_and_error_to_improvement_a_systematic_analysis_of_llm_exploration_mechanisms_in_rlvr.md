# Technical Report on Slow Thinking with LLMs: Exploration Mechanism

Jia Deng, Jie Chen, Zhipeng Chen, Daixuan Cheng, Fei Bai, Beichen Zhang, Yinqian Min, Yanzipeng Gao, Wayne Xin Zhao, Ji-Rong Wen (2025)

## 🧩 Problem to Solve

본 논문은 Verifiable Rewards를 이용한 강화학습(Reinforcement Learning with Verifiable Rewards, RLVR) 환경에서 대규모 언어 모델(LLM)의 추론 능력을 향상시키는 핵심 기제인 탐색(Exploration) 메커니즘을 체계적으로 분석하는 것을 목표로 한다.

RLVR은 규칙 기반의 피드백을 통해 LLM이 복잡한 추론 체인을 생성하고 정제하도록 유도하며, 이 과정은 모델이 얼마나 효과적으로 정답에 이르는 경로를 탐색하느냐에 따라 성패가 갈린다. 하지만 기존 연구들은 RLVR의 경험적 성공에만 집중했을 뿐, LLM의 탐색 행동을 지배하는 근본적인 메커니즘, 즉 탐색 공간이 어떻게 구성되는지, 탐색 능력이 어떻게 실제 성능 향상으로 전환되는지에 대한 상세한 분석이 부족했다. 따라서 본 연구는 탐색 공간의 정량적 측정, 엔트로피와 성능 간의 상호작용, 그리고 탐색 능력을 효율적으로 성능으로 변환하는 최적화 방법을 규명하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 RLVR에서의 탐색 메커니즘을 세 가지 차원에서 체계적으로 분석하고 이를 바탕으로 성능 향상 방안을 제시한 점이다.

첫째, LLM의 탐색 능력 경계(Capability Boundary)를 정량화하기 위해 Pass@k 외에도 $k$-rollout Unsolvable Problems와 Rollout Branching Factor라는 새로운 지표를 도입하여 탐색 공간의 구조를 분석하였다.

둘째, 정책 분포의 엔트로피(Entropy)와 모델 성능 사이의 관계를 다각도로 분석하였다. 특히 학습 단계(Rising vs Plateau stage), 개별 인스턴스의 Perplexity(PPL), 그리고 토큰의 위치(Position)라는 세 가지 세밀한 관점에서 엔트로피가 어떻게 성능에 영향을 미치는지 규명하였다.

셋째, 분석된 통찰을 바탕으로 탐색 능력을 유지하고 최적화 효율을 높이는 구체적인 방법론을 제안하였다. 여기에는 RFT(Rejection-sampling Fine-Tuning)에서의 다양성 중심 데이터 선택 전략과, PPL 및 토큰 위치를 활용한 Advantage Shaping 기법이 포함된다.

## 📎 Related Works

기존의 RLVR 연구들은 주로 엔트로피 감소가 성능 향상을 이끈다는 점이나, Clip-higher와 같은 기법을 통해 탐색을 강화하는 방식에 집중해 왔다. 특히 엔트로피와 보상 사이의 관계를 지수 함수 형태로 모델링하거나, 고엔트로피 토큰이 추론의 핵심 분기점(Forking tokens) 역할을 한다는 연구들이 존재한다.

본 논문은 이러한 기존 연구들이 탐색 메커니즘을 다소 거시적으로 다루었거나 단편적인 측면만 분석했다는 한계를 지적한다. 본 연구는 단순히 엔트로피의 증감을 보는 것을 넘어, SFT와 RL이 각각 탐색 경계에 미치는 서로 다른 영향(SFT는 경계 확장, RL은 Pass@1 정밀화)을 대조 분석하고, 토큰 수준의 미시적 분석을 통해 최적화 효율을 높이는 구체적인 Advantage 조정 방식을 제안함으로써 기존 연구와 차별화된다.

## 🛠️ Methodology

### 1. 탐색 능력의 정량화 지표

모델의 탐색 능력을 측정하기 위해 다음과 같은 지표들을 사용한다.

- **Pass@k**: $k$번의 시도 중 최소 한 번이라도 정답을 맞힐 확률을 측정한다. 편향되지 않은 추정치(unbiased estimator)는 다음과 같이 계산된다.
$$ \text{Pass@k} = \mathbb{E}_{q \sim \mathcal{D}, \{o_i\}_{i=1}^n \sim \pi_\theta(\cdot|q)} \left[ 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} \right] $$
여기서 $n$은 샘플링 횟수, $c$는 정답의 개수이다.
- **$k$-rollout Unsolvable Problems**: $k$번의 시도 후에도 해결하지 못한 문제들의 집합을 정의하여 모델의 능력 한계를 시각화한다.
- **Token-level Entropy**: 토큰 생성 과정의 불확실성을 측정한다.
$$ H_i = -\sum_{t_i \in V} \pi_\theta(t_i | t_{<i}) \log \pi_\theta(t_i | t_{<i}) $$
- **Rollout Branching Factor**: 상위 95% 확률 질량(top-p=0.95) 내에 포함된 후보 토큰의 수를 측정하여 생성의 다양성을 평가한다.

### 2. 엔트로피-성능 상호작용 분석 (Fine-grained Analysis)

본 논문은 RLVR 학습 과정을 두 단계로 나누어 분석한다.

- **Rising Stage**: 성능이 급격히 오르고 엔트로피가 감소하는 단계이다. 분석 결과, 엔트로피 감소는 주로 오답 샘플(Negative samples)에서 빠르게 일어나며, 이는 모델이 잘못된 추론 경로를 제거함으로써 유효한 추론 패턴을 형성하는 과정임을 밝혀냈다.
- **Plateau Stage**: 성능 향상이 둔화되는 단계이다. 이때의 학습은 극소수의 고엔트로피, 고그라디언트 토큰, 특히 수식이나 기호 같은 Formal Reasoning 토큰의 확률을 조정하는 것에 집중된다.

### 3. 성능 최적화 방법론

탐색 능력을 성능으로 변환하기 위해 다음과 같은 기법을 제안한다.

- **Exploration-aware RFT**: RFT 과정에서 5%의 노이즈 데이터(오답)를 섞거나, 평균 엔트로피 및 Rollout Branching Factor가 높은 데이터를 우선 선택하여 모델이 지나치게 결정론적으로 변하는 것을 방지하고 탐색 공간을 유지한다.
- **PPL-based Advantage Shaping**: 낮은 PPL을 가진 샘플이 더 안정적인 추론 경로를 제공한다는 점에 착안하여, Advantage를 다음과 같이 조정한다.
$$ \tilde{A}^i_t = A^i_t \cdot (1 - \alpha \cdot w_{ppl}(o_i)) $$
여기서 $w_{ppl}$은 표준화된 $\log\text{PPL}$ 가중치이다.
- **Position-based Advantage Shaping**: 추론 체인의 후반부 토큰이 최종 결정에 더 결정적인 영향을 미친다는 점을 이용하여, 후반부 토큰에 보너스 $b^i_t$를 부여한다.
$$ \tilde{A}'^i_t = A^i_t + \text{sign}(A^i_t) \cdot b^i_t $$

## 📊 Results

### 실험 설정

- **데이터셋**: AIME24, AIME25, AMC23, MATH500 등 고난도 수학 벤치마크를 사용하였다.
- **모델**: Qwen2.5-7B 및 Qwen2.5-Math-7B를 베이스라인으로 사용하였으며, GRPO 알고리즘을 적용하였다.
- **비교 대상**: Base 모델, 일반 GRPO, 그리고 제안하는 PPL/Position 기반 Advantage Shaping 모델을 비교하였다.

### 주요 결과

- **SFT vs RL**: SFT는 외부 데이터를 통해 Pass@k(탐색 경계)를 확장시키지만, RL은 Pass@1(착취/활용)을 정교화하는 대신 탐색 공간을 좁히는 경향이 있음을 확인하였다.
- **TIR(Tool-Integrated Reasoning)**: 코드 인터프리터와 같은 외부 도구를 통합했을 때 Pass@k 성능이 유의미하게 향상되어, 도구 활용이 모델의 능력 경계를 확장하는 강력한 수단임을 보였다.
- **Advantage Shaping 효과**: PPL 기반 및 Position 기반의 Advantage 조정 기법을 적용했을 때, GRPO 베이스라인 대비 평균적으로 Qwen2.5-7B에서는 1.51%, Qwen2.5-Math-7B에서는 2.31%의 성능 향상을 거두었다.
- **추론 패턴 변화**: 제안 방법론을 적용한 모델은 GRPO 베이스라인보다 더 상세한 단계별 풀이(Step-by-step breakdown)를 생성하며, 특히 Position 기반 방법은 오류를 발견하고 되돌아가는(backtrack) 깊은 추론 양상을 보였다.

## 🧠 Insights & Discussion

본 논문은 RLVR에서 탐색과 활용(Exploration-Exploitation)의 트레이드오프를 심도 있게 다루었다.

**강점 및 통찰**:
가장 주목할 점은 RL이 단순히 정답 경로를 강화하는 것이 아니라, 오답 경로의 엔트로피를 빠르게 낮춤으로써 '정답으로 가는 길'을 찾아내는 과정이라는 분석이다. 또한, 모든 토큰이 동일하게 중요하지 않으며, 추론 체인의 후반부 토큰과 낮은 PPL을 가진 인스턴스가 학습 효율성을 결정짓는 핵심 요소임을 정량적으로 증명하였다.

**한계 및 논의**:
Pass@k가 높더라도 실제 추론 과정(Reasoning chain)이 논리적으로 타당하지 않은 '우연한 정답'의 가능성이 존재한다. 저자들은 수동 검수를 통해 대부분의 정답이 타당한 추론에 기반함을 확인했으나, 이를 자동으로 검증할 수 있는 프로세스 레벨의 평가(Process-level evaluation) 도입이 향후 과제로 남아 있다. 또한, 특정 도메인(수학)에서의 RL 학습이 타 도메인(일반 상식 등)의 능력을 약화시킬 수 있다는 점(Catastrophic forgetting 유사 현상)이 관찰되어 이에 대한 보완이 필요하다.

## 📌 TL;DR

본 연구는 RLVR에서 LLM의 탐색 메커니즘을 분석하여, SFT는 탐색 경계를 확장하고 RL은 이를 정밀화(Pass@1 향상)하지만 탐색 공간을 축소시킨다는 점을 밝혔다. 특히 낮은 PPL 샘플과 추론 체인 후반부의 토큰이 학습에 결정적이라는 통찰을 통해, 이를 활용한 Advantage Shaping 기법을 제안하여 수학적 추론 성능을 유의미하게 향상시켰다. 이 연구는 향후 더 효율적인 RLVR 시스템을 구축하기 위한 체계적인 분석 프레임워크를 제공한다.
