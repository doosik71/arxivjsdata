# Beyond Pass@1: Self-play with Variational Problem Synthesis Sustains RLVR

Xiao Liang, Zhongzhi Li, Yeyun Gong, Yelong Shen, Ying Nian Wu, Zhijiang Guo, Weizhu Chen (2025)

## 🧩 Problem to Solve

최근 Large Language Models (LLMs)의 복잡한 추론 능력을 향상시키기 위해 Verifiable Rewards를 이용한 강화학습인 RLVR (Reinforcement Learning with Verifiable Rewards)이 핵심 패러다임으로 부상하였다. 그러나 표준적인 RLVR 학습, 특히 GRPO와 같은 최적화 방식은 Pass@1 성능은 높이지만, 그 대가로 Policy Entropy(정책 엔트로피)를 감소시켜 생성 결과물의 다양성을 저해한다는 문제점이 있다.

Policy Entropy의 붕괴는 모델이 훈련 데이터셋에 존재하는 정답 궤적을 단순히 암기하여 보상을 얻는 '리워드 해킹' 현상을 초래한다. 이는 결과적으로 모델의 추론 잠재력의 상한선이라 할 수 있는 Pass@k 성능의 정체로 이어지며, 더 나아가 탐색(Exploration) 기회의 상실로 인해 Pass@1 성능마저 결국 정체되는 결과를 낳는다. 따라서 지속 가능한 RLVR 학습을 위해서는 훈련 과정에서 정책 엔트로피를 유지하고 생성 다양성을 확보하여 Pass@k 성능을 지속적으로 끌어올리는 것이 필수적이다.

## ✨ Key Contributions

본 논문은 모델 스스로 훈련 데이터를 생성하고 이를 통해 학습하는 온라인 Self-play 방식의 **SvS (Self-play with Variational problem Synthesis)** 전략을 제안한다.

SvS의 핵심 아이디어는 모델이 성능이 저조한(underperforming) 문제에 대해 생성한 '정답 솔루션'을 컨텍스트로 사용하여, 해당 문제의 변형 문제(Variational Problems)를 스스로 합성하는 것이다. 이때 합성된 변형 문제는 원본 문제와 동일한 정답(Reference Answer)을 공유하도록 설계되어, 추가적인 정답 라벨링 비용 없이도 데이터의 다양성을 확보할 수 있다. 이를 통해 모델은 동일한 정답에 도달하기 위한 더 다양하고 새로운 추론 경로를 탐색하게 되며, 결과적으로 정책 엔트로피 붕괴를 막고 추론 능력의 경계를 확장한다.

## 📎 Related Works

기존의 RLVR 연구들은 주로 KL 제약 조건을 제거하거나 Clip-Higher 전략을 도입하여 탐색을 강화하는 방향으로 진행되었다. 또한, 데이터 다양성을 높이기 위해 외부 LLM을 이용해 문제를 재구성(Rephrasing)하거나 합성 데이터를 생성하는 방식이 사용되었다.

하지만 외부 LLM을 이용한 데이터 증강은 원본 문제와의 의미론적 불일치(Semantic Inconsistency)를 유발하여 정답 라벨의 정확성을 해칠 수 있으며, 모델의 현재 능력 수준과 맞지 않는 데이터가 생성될 위험이 있다. 반면, SvS는 모델이 직접 생성한 정답 솔루션을 기반으로 변형 문제를 만들기 때문에, 모델의 현재 능력 수준에 정렬된(aligned) 데이터를 생성할 수 있으며, 정답의 정확성이 보장된다는 점에서 기존 방식과 차별화된다.

## 🛠️ Methodology

SvS는 모델이 문제를 풀고, 새로운 문제를 만들고, 다시 그 문제를 푸는 세 가지 단계의 루프를 통해 자기 개선(Self-improvement)을 달성한다. 전체 파이프라인은 다음과 같다.

### 1. 전체 시스템 구조 및 절차

매 학습 스텝마다 다음 세 가지 구성 요소가 결합된 데이터 버퍼 $\mathcal{B}$를 구성하여 정책 $\pi_\theta$를 업데이트한다.

1. **Original Problem Solving**: 원본 훈련 셋 $\mathcal{D}$에서 샘플링된 문제 $x$에 대해 솔루션 $y_i$들을 생성한다.
2. **Variational Problem Synthesis**: 모델의 정답률이 특정 범위 $[acc_l, acc_h]$ 내에 있는 '성능 저조 문제'를 식별하고, 이 문제들의 정답 솔루션 $y_i$를 입력으로 하여 변형 문제 $\hat{x}_j$들을 합성한다.
3. **Synthetic Problem Solving**: 생성된 변형 문제 $\hat{x}_j$를 모델이 다시 해결하며 솔루션 $\hat{y}_k$를 생성한다.

### 2. 주요 방정식 및 손실 함수

**정답 보상 (Correctness Reward):**
원본 문제와 변형 문제 모두 정답 $a$와의 일치 여부로 보상을 결정한다.
$$R_c(y, a) = \mathbb{I}(\text{Extract}(y) = a)$$

**변형 문제 합성 보상 (Reward Shaping for Synthesis):**
단순히 정답을 맞힐 수 있는 문제를 만드는 것이 아니라, 모델이 적절한 난이도를 가진 문제를 생성하도록 유도한다. 변형 문제 $\hat{x}$의 정답률 $\text{Acc}(\hat{x}, a)$가 특정 범위 $[\hat{acc}_l, \hat{acc}_h]$ 내에 있을 때만 양의 보상을 부여한다.
$$R_v(\hat{x}) = \mathbb{I}(\hat{acc}_l \le \text{Acc}(\hat{x}, a) \le \hat{acc}_h)$$
이는 모델이 너무 쉬운 문제(힌트가 포함된 문제)나 아예 풀 수 없는 문제를 만드는 것을 방지하기 위함이다.

**정책 업데이트 (GRPO):**
최종적으로 수집된 데이터는 GRPO (Group Relative Policy Optimization) 목적 함수를 통해 업데이트된다. 각 토큰의 Advantage $A_{i,t}$는 그룹 내 보상의 평균과 표준편차를 이용해 정규화된다.
$$A_{i,t} = \frac{r_i - \text{mean}(r_1, \dots, r_G)}{\text{std}(r_1, \dots, r_G)}$$
최종 목적 함수 $J(\theta)$는 다음과 같이 Clipping된 확률 비율과 KL 발산 항을 포함한다.
$$J(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \min \left( r_{i,t}(\theta) A_{i,t}, \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon) A_{i,t} \right) - \beta D_{KL}(\pi_\theta \| \pi_{\text{ref}}) \right]$$

## 📊 Results

### 실험 설정

- **모델**: Qwen2.5 (3B, 32B), LLaMA-3.1 (8B)
- **데이터셋**: MATH-12k, DAPO-17k
- **평가 지표**: Pass@1 (32회 평균), Pass@32, Pass@k (1 to 1024 scaling)
- **벤치마크**: AIME 24/25, Beyond-AIME, MATH-500, Olympiad-Bench 등 12개 추론 벤치마크

### 주요 결과

1. **추론 능력의 상한선 확장**: 가장 도전적인 벤치마크인 AIME 24와 AIME 25에서 Pass@32 성능이 표준 RLVR 대비 각각 **18.3%p**와 **22.8%p**라는 압도적인 절대 이득을 기록하였다.
2. **지속 가능한 학습**: Figure 5에서 보이듯, 표준 RLVR은 학습이 진행됨에 따라 정책 엔트로피가 급격히 감소하지만, SvS는 엔트로피를 안정적인 범위 내에서 유지하며 성능이 정체되지 않고 계속 상승하는 모습을 보였다.
3. **Pass@k 스케일링**: Pass@k를 $k=1$부터 $1024$까지 확장했을 때, SvS는 모든 $k$ 값에서 표준 RLVR을 상회하였으며, 특히 $k$가 커질수록 성능 격차가 벌어지는 것을 확인하였다. 이는 모델이 더 다양하고 고도화된 추론 전략을 학습했음을 의미한다.
4. **범용성 확인**: 수학뿐만 아니라 코드 생성 작업(PRIME-RL 데이터셋)에서도 동일한 전략을 적용한 결과, 표준 RLVR 대비 훨씬 적은 학습 스텝(약 1/5 수준)만으로도 더 높은 성능을 달성하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

SvS는 모델이 정답 솔루션을 기반으로 역으로 문제를 생성하게 함으로써, 모델이 문제의 의미론적 구조를 더 깊이 이해하게 만든다. 특히 '성능 저조 문제'에 집중하여 증강함으로써 모델의 취약점을 효율적으로 공략한다. 또한, 변형 문제의 정답률을 기반으로 보상을 설계한 Reward Shaping은 모델이 정답을 직접 노출하는 식의 리워드 해킹을 방지하고, 적절한 난이도의 문제를 생성하도록 강제하는 핵심 장치로 작용한다.

### 한계 및 비판적 해석

본 논문에서는 합성된 문제의 유효성을 SOTA LLM(Qwen3, O3)을 통해 검증하였으며 약 80% 이상의 유효성을 보였다고 주장한다. 하지만 일부 문제는 텍스트 설명이 부자연스럽거나 모호함에도 불구하고 정답이 존재한다는 이유로 '유효'하다고 판단되었을 가능성이 있다. 또한, 변형 문제 생성 시 원본 문제의 구조를 너무 많이 유지할 경우 실질적인 다양성 확보에 한계가 있을 수 있다.

그럼에도 불구하고, 외부 가이드나 증류(distillation) 없이 오직 모델 스스로의 Self-play만으로 추론의 경계를 확장했다는 점은 매우 고무적이며, 이는 향후 LLM의 자율적 학습 프레임워크 구축에 중요한 시사점을 제공한다.

## 📌 TL;DR

본 논문은 RLVR 학습 시 발생하는 정책 엔트로피 붕괴와 Pass@k 성능 정체 문제를 해결하기 위해, 모델이 스스로 변형 문제를 생성하고 해결하는 **SvS (Self-play with Variational problem Synthesis)** 전략을 제안한다. 이 방법은 추가 라벨링 없이 데이터 다양성을 확보하여 모델의 추론 잠재력을 극대화하며, 특히 AIME와 같은 고난도 벤치마크에서 Pass@32 성능을 비약적으로 향상시킨다. 이는 자가 개선(Self-improvement) 루프가 LLM의 추론 한계를 돌파하는 효율적인 방법이 될 수 있음을 입증한다.
