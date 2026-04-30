# Logical Reasoning in Large Language Models: A Survey

Hanmeng Liu, Zhizhang clang Fu, Mengru Ding, Ruoxi Ning, Chaoli Zhang, Xiaozhang Liu and Yue Zhang (2025)

## 🧩 Problem to Solve

최근 OpenAI o3나 DeepSeek-R1과 같은 고급 추론 모델의 등장으로 대규모 언어 모델(LLM)의 추론 능력이 비약적으로 발전하였으나, 엄격한 의미의 '논리적 추론(Logical Reasoning)' 수행 능력은 여전히 미해결 과제로 남아 있다. 특히 많은 기존 연구와 서베이들이 논리적 추론을 단순한 Chain-of-Thought(CoT)와 같은 일반적인 휴리스틱 전략과 혼동하여 다루는 경향이 있다.

논리적 추론은 법률 분석이나 과학적 발견과 같이 정확성과 검증 가능성이 필수적인 도메인에서 매우 중요하다. 따라서 본 논문의 목표는 일반적인 휴리스틱 접근 방식이 아닌, **형식적 및 상징적 논리(Formal and Symbolic Logic)** 기반의 추론에 초점을 맞추어 LLM의 논리적 추론 능력을 종합적으로 분석하고 체계화하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 LLM의 논리적 추론을 단순한 텍스트 생성 능력이 아닌, 형식 논리의 관점에서 재정의하고 분석했다는 점이다.

1.  **논리적 추론의 체계적 분류**: 추론 패러다임을 연역적(Deductive), 귀납적(Inductive), 가추적(Abductive), 유추적(Analogical) 추론의 네 가지 핵심 유형으로 구분하여 정의하였다.
2.  **평가 체계의 분석**: 규칙 기반, 전문가 설계, 시험 기반 데이터셋으로 분류하여 기존 벤치마크들의 특성과 한계를 분석하였다.
3.  **추론 능력 향상 전략의 정식화**: 데이터 중심, 모델 중심, 외부 지식 활용, 뉴로-심볼릭(Neuro-symbolic) 접근 방식을 수학적 최적화 관점에서 정식화하여 제시하였다.
4.  **미래 방향성 제시**: 강건성(Robustness)과 일반화(Generalization), 해석 가능성과 성능 사이의 트레이드오프 등 현재 LLM이 직면한 핵심 갈등 구조를 분석하였다.

## 📎 Related Works

기존의 추론 관련 서베이들은 주로 LLM의 일반적인 문제 해결 능력이나 CoT와 같은 프롬프팅 기법에 집중하였다. 그러나 이러한 접근 방식은 모델이 실제로 논리적 규칙을 따르는지, 아니면 단순히 훈련 데이터의 통계적 패턴을 모방하는지를 구분하지 못한다는 한계가 있다.

본 논문은 이러한 한계를 극복하기 위해 **상징적 논리(Symbolic Logic)**와 LLM의 결합에 집중하며, 특히 정형화된 논리 체계(예: First-Order Logic)를 통해 추론의 정확성을 검증하려는 시도들을 중점적으로 다룬다.

## 🛠️ Methodology

본 논문은 LLM의 논리적 추론 능력을 향상시키기 위한 방법론을 네 가지 주요 관점으로 분류하고 각각을 최적화 문제로 정의한다.

### 1. Data-Centric Approaches
데이터 중심 접근법은 정교하게 큐레이션된 데이터셋을 통해 모델의 성능을 높이는 방법이다. 이를 다음과 같이 정식화한다.
$$D^* = \arg \max_D R(M_D)$$
여기서 $D$는 훈련 데이터셋, $M_D$는 해당 데이터로 학습된 모델, $R$은 성능 평가 지표이다. 
- **Expert-Curated**: FOLIO와 같이 전문가가 작성한 정형 논리 데이터 활용.
- **Synthetic**: 규칙 기반으로 생성된 합성 데이터(예: RuleTaker) 활용.
- **LLM-Distilled**: GPT-4와 같은 고성능 모델이 생성한 추론 경로(Reasoning Chain)를 증류하여 학습.

### 2. Model-Centric Approaches
모델 파라미터와 디코딩 전략을 최적화하는 방법이며, 다음과 같이 정의된다.
$$(\theta^*, S^*) = \arg \max_{\theta, S} R(M_\theta, S)$$
여기서 $\theta$는 학습 가능한 파라미터, $S$는 디코딩 전략(CoT, Verification 등)을 의미한다.
- **Instruction Fine-Tuning (IFT)**: 특정 논리 태스크에 특화된 지시어 학습.
- **Reinforcement Learning (RL)**: DeepSeek-R1과 같이 RL을 통해 긴 추론 사슬(Long-CoT)을 생성하도록 유도.
- **Inference-Time Decoding**: Graph-of-Thought(GoT)나 MCTS(Monte Carlo Tree Search)를 통한 추론 경로 탐색 및 최적화.

### 3. External Knowledge Utilization
모델 내부 지식의 한계(Hallucination)를 극복하기 위해 외부 지식을 통합하는 방법이다.
$$(M^*, K^*) = \arg \max_{M, K} R(M, K)$$
여기서 $K$는 지식 통합 전략(Retrieval-augmented, Knowledge Graph 등)을 의미한다. Lean과 같은 수학적 증명 도구에서 데이터를 추출하거나, 복잡한 문제를 하위 질문으로 분해하여 지식 그래프와 결합하는 방식이 포함된다.

### 4. Neuro-Symbolic Approaches
딥러닝의 표현 능력과 심볼릭 논리의 정밀함을 결합하는 하이브리드 방식이다.
$$(M^*, P^*) = \arg \max_{M, P} R(P(M(x)))$$
- $M$: 입력 $x$를 형식 언어 $L$의 심볼릭 표현 $z$로 매핑하는 신경망 모델 ($z = M(x)$).
- $P$: 심볼릭 표현 $z$를 입력받아 최종 결과 $y$를 도출하는 심볼릭 솔버 ($y = P(z)$).
- **작동 방식**: LLM이 자연어를 1차 논리(FOL) 식 등으로 변환하면, 외부의 정형 논리 솔버(Theorem Prover)가 이를 계산하여 정확한 답을 도출한다.

## 📊 Results

### 평가 벤치마크 분석
논문은 논리적 추론 평가를 위해 다양한 데이터셋을 분석하였다.
- **NLI (Natural Language Inference)**: ConTRoL, FOLIO 등이 있으며, 전제가 결론을 함축하는지(Entailment)를 평가한다.
- **MRC (Machine Reading Comprehension)**: LogiQA, ReClor 등이 있으며, 지문을 읽고 논리적 추론을 통해 정답을 찾는 능력을 평가한다.

### 추론 패러다임별 성능 분석
- **연역적 추론(Deductive)**: 기본적인 구성적 증명은 잘 수행하나, 예시가 없는 가설적 하위 증명이나 구문적 변형에 매우 민감한 모습을 보인다.
- **귀납적 추론(Inductive)**: 일반적인 패턴 인식은 가능하나, 트랜스포머 구조 자체가 근본적인 논리 원칙을 학습하는 데 한계가 있음이 지적되었다.
- **가추적 추론(Abductive)**: 불완전한 정보에서 가장 그럴듯한 설명을 생성하는 작업에서 LLM이 여전히 어려움을 겪고 있다.
- **유추적 추론(Analogical)**: 유추의 복잡성이 증가할수록 성능이 급격히 하락하며, 일부 연구에서는 모델이 실제 유추를 하는 것이 아니라 표면적인 패턴에 의존한다는 의문을 제기한다.

### 지표의 변화
단순 정확도(Accuracy)나 F1-score에서 벗어나, 논리적으로 동일한 입력에 대해 일관된 답을 내놓는지 측정하는 **일관성(Consistency)**, 분포 외 데이터에 대한 **일반화(Generalization)**, 추론 단계의 명확성을 측정하는 **설명 가능성(Explainability)** 등의 정밀한 지표가 도입되고 있다.

## 🧠 Insights & Discussion

### 주요 갈등 구조 (Unresolved Tensions)
1.  **강건성 vs 일반화**: 특정 데이터셋(예: FOLIO)으로 튜닝된 모델은 통제된 환경에서는 뛰어나지만, 약간의 구문 변화(Adversarial Perturbations)에도 성능이 급락한다. 이는 모델이 인과 관계가 아닌 통계적 상관관계에 의존하기 때문이다.
2.  **해석 가능성 vs 성능**: 뉴로-심볼릭 방식은 단계별 증명을 제공하여 해석 가능성이 높지만, 지식 베이스가 커질수록 확장성(Scalability) 문제가 발생한다. 반면 블랙박스 형태의 데이터 기반 모델은 성능은 높으나 추론 과정을 신뢰할 수 없다.
3.  **평가 체계의 엄밀성**: 현재의 객관식(Multiple-choice) 벤치마크는 실제 추론 능력이 아니라 단순한 패턴 인식 능력을 측정하고 있을 가능성이 크다.

### 비판적 해석
LLM이 '논리(Logic)'와 '어휘(Lexicon)'를 완전히 분리하여 처리하지 못하는 한, 고위험 도메인(법률, 의료 등)에 그대로 적용하는 것은 위험하다. 단순한 프롬프팅 기법을 넘어, 미분 가능한 정리 증명기(Differentiable Theorem Provers)와 같은 하이브리드 구조로의 전환이 필수적이다.

## 📌 TL;DR

본 논문은 LLM의 논리적 추론을 일반적인 휴리스틱이 아닌 **형식적/상징적 논리** 관점에서 분석한 종합 서베이이다. 추론을 네 가지 패러다임(연역, 귀납, 가추, 유추)으로 분류하고, 이를 향상시키기 위한 전략을 데이터, 모델, 외부 지식, 뉴로-심볼릭의 네 가지 최적화 프레임워크로 정식화하였다. 특히 단순한 성능 향상보다는 **강건성, 일관성, 해석 가능성**의 확보가 향후 연구의 핵심이며, 이를 위해 신경망의 유연성과 심볼릭 논리의 정밀함을 결합한 하이브리드 아키텍처가 필수적임을 강조한다.