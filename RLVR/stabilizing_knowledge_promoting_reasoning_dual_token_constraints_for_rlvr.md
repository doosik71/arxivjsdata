# Stabilizing Knowledge, Promoting Reasoning: Dual-Token Constraints for RLVR

Jiakang Wang, Runze Liu, Fuzheng Zhang, Xiu Li, and Guorui Zhou (2025)

## 🧩 Problem to Solve

본 논문은 검증 가능한 보상을 이용한 강화학습(Reinforcement Learning with Verifiable Rewards, RLVR)을 통해 거대 언어 모델(LLM)의 추론 능력을 향상시킬 때 발생하는 **토큰별 역할 차이 무시** 문제를 해결하고자 한다.

일반적인 RLVR 알고리즘은 모든 토큰에 동일한 훈련 신호를 적용한다. 그러나 실제 추론 과정에서 토큰은 두 가지 서로 다른 역할을 수행한다. 첫째는 사실적 지식이나 도메인 지식을 담고 있는 저엔트로피(low-entropy) 토큰이며, 둘째는 논리적 연결 및 단계별 추론을 가이드하는 고엔트로피(high-entropy) 토큰이다.

기존의 일부 연구에서는 그래디언트 마스킹(gradient masking)이나 비동기 업데이트를 통해 이들을 분리하려 했으나, 이는 문장 내 토큰 간의 세밀한 세부 의미적 및 구문적 의존성(semantic and syntactic dependencies)을 깨뜨려 오히려 학습 효율을 저하시키는 결과를 초래했다. 따라서 본 논문의 목표는 지식 관련 토큰의 안정성은 유지하면서 추론 관련 토큰의 탐색은 촉진하는 동기화된(synchronized) 엔트로피 인식 학습 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문은 **Archer**라는 엔트로피 인식 RLVR 접근 방식을 제안하며, 핵심 아이디어는 **이중 토큰 제약(Dual-Token Constraints)**을 통해 토큰의 특성에 따라 서로 다른 최적화 강도를 적용하는 것이다.

중심적인 설계 직관은 다음과 같다.

1. **추론 토큰(Reasoning tokens):** 고엔트로피 특성을 가지며, 논리적 패턴 학습을 위해 더 넓은 탐색 범위(더 높은 Clipping 임계값과 더 약한 KL 정규화)가 필요하다.
2. **지식 토큰(Knowledge tokens):** 저엔트로피 특성을 가지며, 모델이 이미 보유한 사실적 지식을 보존하기 위해 엄격한 제약(더 낮은 Clipping 임계값과 더 강한 KL 정규화)이 필요하다.

이 모든 과정은 마스킹이나 분리 업데이트가 아닌 **동기적 업데이트(synchronous updates)** 방식으로 수행되어 토큰 간의 의존성을 유지한다.

## 📎 Related Works

본 논문은 다음과 같은 기존 연구들을 기반으로 하며, 그 한계를 지적한다.

1. **GRPO (Group Relative Policy Optimization):** 가치 모델(Value model) 없이 그룹 내 상대적 보상을 통해 이점을 추정하는 효율적인 RL 방법론이다.
2. **DAPO (Decouple Clip and Dynamic Sampling Policy Optimization):** GRPO를 개선하여 Clipping 범위의 분리 및 동적 샘플링 등을 도입한 알고리즘이다.
3. **토큰 수준 분석 연구:** 최근 연구들은 고엔트로피 토큰이 논리적 전이 지점(forking tokens)에서 주로 나타난다는 점을 밝혀냈다. 이에 따라 일부 연구자들은 저엔트로피 토큰의 업데이트를 완전히 차단하는 그래디언트 마스킹 기법을 사용했다.

**차별점:** Archer는 기존의 '완전 차단' 방식이 토큰 간의 유기적 연결을 해친다는 점에 주목하여, 차단이 아닌 **제약의 강도를 조절하는 방식**으로 접근함으로써 지식 보존과 추론 향상이라는 두 마리 토끼를 동시에 잡고자 한다.

## 🛠️ Methodology

### 1. 응답 수준 엔트로피를 통한 핵심 토큰 식별

기존의 배치(batch) 수준 통계는 프롬프트마다 엔트로피 편차가 커서 오분류 가능성이 높다. Archer는 이를 해결하기 위해 각 응답(response) 내에서 독립적으로 엔트로피 분위수(quantile)를 계산하는 **응답 수준 엔트로피 통계(Response-level Entropy Statistics)** 방식을 사용한다.

응답 $\text{resp}_i$ 내의 토큰 $t$에 대한 엔트로피를 $e_{it}$라고 할 때, $\gamma$-분위수 임계값 $\tau_i^\gamma$는 다음과 같이 정의된다.
$$\tau_i^\gamma = \text{Quantile}(\{e_{it} \mid t=1, \dots, |\text{resp}_i|\}, \gamma)$$
여기서 $\gamma$는 보통 0.8(80번째 백분위수)로 설정된다.

### 2. 이중 토큰 제약 (Dual-Token Constraints)

식별된 임계값을 기준으로 토큰을 지식 토큰과 추론 토큰으로 나누고, 각각에 대해 서로 다른 제약 조건을 적용한다.

**A. Clipping 제약:**
정책 업데이트의 크기를 조절하는 clipping 범위 $\epsilon(e_{it})$를 다음과 같이 차등 적용한다.
$$\epsilon(e_{it}) = \begin{cases} \epsilon_r & \text{if } e_{it} \ge \tau_i^\gamma \\ \epsilon_k & \text{otherwise} \end{cases}$$
여기서 $\epsilon_r$은 추론 토큰을 위한 넓은 범위, $\epsilon_k$는 지식 토큰을 위한 좁은 범위이다.

**B. KL 제약:**
참조 모델(reference model)과의 이격을 제한하는 KL 발산 페널티 계수 $\beta(e_{it})$를 차등 적용한다.
$$\beta(e_{it}) = \begin{cases} \beta_r & \text{if } e_{it} \ge \tau_i^\gamma \\ \beta_k & \text{otherwise} \end{cases}$$
지식 토큰에는 강한 제약($\beta_k$ 높음)을 주어 사실적 지식을 보존하고, 추론 토큰에는 약한 제약($\beta_r$ 낮음)을 주어 유연한 학습을 허용한다.

### 3. 최종 목적 함수 (TDPO)

위의 제약 조건들을 통합한 Archer의 목적 함수 $L_{\text{TDPO}}(\theta)$는 다음과 같다.
$$L_{\text{TDPO}}(\theta) = \mathbb{E} \left[ \frac{1}{\sum |\text{resp}_i|} \sum_{i=1}^G \sum_{t=1}^{|\text{resp}_i|} \left( \min(r_{it}(\theta)\hat{A}_{it}, \text{clip}(r_{it}(\theta), 1-\epsilon(e_{it}), 1+\epsilon(e_{it}))\hat{A}_{it}) - \beta(e_{it}) D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right) \right]$$
여기서 $r_{it}(\theta)$는 중요도 샘플링 비율(importance sampling ratio)이며, $\hat{A}_{it}$는 GRPO 방식의 그룹 상대적 이점(advantage) 값이다.

## 📊 Results

### 1. 실험 설정

- **베이스 모델:** DeepSeek-R1-Distill-Qwen-1.5B
- **데이터셋:** 수학(AIME24/25, AMC23, MATH-500 등) 및 코드(LiveCodeBench v5/v6)
- **비교 대상:** Base 모델, DAPO, DeepScaleR, DeepCoder, FastCuRL, Nemotron-1.5B 등

### 2. 주요 결과

- **성능 향상:** 수학 및 코드 벤치마크 모두에서 기존 RLVR 방법론(DAPO 등) 및 유사 크기의 SOTA 모델들을 능가했다. 특히 AIME24에서 DAPO 대비 Pass@1 성능이 6.6% 상승하는 성과를 보였다.
- **효율성:** 복잡한 다단계 학습을 거친 타 모델들과 달리, 단일 단계(single-stage) 학습만으로도 최상위 성능을 달성했으며, 사용된 GPU 시간 또한 훨씬 적었다.
- **추론 다양성:** Pass@1뿐만 아니라 Pass@K 지표에서도 우수한 성적을 거두어, 모델의 잠재적 추론 능력이 향상되었음을 입증했다.

## 🧠 Insights & Discussion

### 1. KL 가중치와 Clipping 범위의 트레이드-오프

분석 결과, 저엔트로피 토큰에 대한 제약 강도는 모델의 붕괴(collapse)와 학습 속도 사이의 균형을 결정한다.

- **KL 가중치가 너무 낮을 때 ($\beta=0$):** 학습 초기에는 성능이 빠르게 오르지만, 곧 엔트로피가 급격히 떨어지고 반복 문구가 증가하는 모델 붕괴 현상이 나타났다.
- **KL 가중치가 너무 높을 때:** 모델의 안정성은 유지되나 학습 속도가 매우 느려져 최종 성능이 저하된다.
- **Clipping 범위 ($\epsilon_k$):** 저엔트로피 토큰의 clipping 범위를 좁게 유지하는 것이 모델 붕괴를 막고 최종 성능을 높이는 데 필수적이다.

### 2. 수학 RL과 코드 RL의 상호 증진 효과

수학 데이터로 RL 학습을 한 모델이 코드 성능이 오르고, 반대의 경우도 성립함을 확인했다. 논문은 이를 '주제적 유사성'보다는 '문제의 내재적 난이도'와 '추론 프로세스의 개선' 관점에서 해석한다. RL은 새로운 지식을 주입하는 것이 아니라, 모델이 이미 가진 지식을 구조화(Structural Organization)하고 세부 사항에 주의(Attention to Details)하며 맥락적 일관성(Contextual Consistency)을 유지하게 만드는 능력을 배양한다는 점을 시사한다.

## 📌 TL;DR

본 논문은 LLM의 RLVR 학습 시 토큰의 역할(지식 vs 추론)에 따라 엔트로피 기반으로 서로 다른 제약(Clip range, KL weight)을 적용하는 **Archer** 프레임워크를 제안한다. 지식 토큰은 엄격하게 보존하고 추론 토큰은 유연하게 탐색하게 함으로써, 모델 붕괴 없이 추론 능력을 극대화했다. 결과적으로 1.5B 규모의 모델에서 수학 및 코드 추론 성능의 SOTA를 달성했으며, 이는 RL이 지식의 추가가 아닌 **기존 능력의 최적화 및 통합 과정**임을 보여준다.
