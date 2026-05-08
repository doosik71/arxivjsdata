# MEML-GRPO: Heterogeneous Multi-Expert Mutual Learning for RLVR

Weitao Jia, Jinghui Lu, Haiyang Yu, et al. (2026)

## 🧩 Problem to Solve

본 논문은 Verifiable Rewards를 이용한 강화학습(Reinforcement Learning with Verifiable Rewards, RLVR)이 대규모 언어 모델(LLM)의 추론 능력을 향상시키는 데 효과적이지만, **Reward Sparsity(보상 희소성)** 문제로 인해 학습이 정체되는 한계를 해결하고자 한다.

Reward Sparsity는 특히 난이도가 높은 작업에서 발생하며, 모델이 생성한 모든 후보 답변이 정답과 일치하지 않아 보상이 0이 될 때 발생한다. 이 경우 모델은 학습을 위한 긍정적인 신호를 전혀 받지 못하게 되어, 기존 모델이 이미 알고 있는 경로 내에서만 최적화될 뿐 새로운 추론 경로를 탐색하거나 능력을 확장하는 데 어려움을 겪는다. 즉, 현재의 on-policy RLVR 방식은 모델의 초기 능력이 부족할 경우 조기에 성능 정체(plateau) 현상이 나타나는 치명적인 문제가 있다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 서로 다른 특성을 가진 **Heterogeneous pre-trained models(이종 사전학습 모델)**들의 보완적인 강점을 활용하여 reward sparsity 문제를 극복하는 것이다. 이를 위해 다음과 같은 세 가지 핵심 설계 아이디어를 제안한다.

1. **Multi-Expert Fine-Tuning (MEF)**: 다양한 외부 전문가 모델의 추론 스타일을 모사하는 시스템 프롬프트를 도입하여, 모델이 하나의 정답 경로가 아닌 다양한 추론 경로를 생성할 수 있도록 한다.
2. **Reinforced Inter-Expert Learning (RIEL)**: 전문가들 간의 지식 공유 메커니즘을 통해, 성능이 낮은 전문가가 성능이 높은 전문가의 성공적인 추론 경로를 학습하도록 유도한다.
3. **Hard Example Accumulation via SFT Buffer**: 모든 전문가가 해결하지 못한 매우 어려운 문제들에 대해 Ground Truth를 이용한 Supervised Fine-Tuning(SFT)을 주기적으로 수행함으로써, 학습 신호가 완전히 끊기는 것을 방지한다.

## 📎 Related Works

최근 DeepSeek-r1, OpenAI-o1 등 RLVR을 통해 복잡한 추론 능력을 향상시킨 연구들이 다수 발표되었다. 특히 정답 일치 여부나 유닛 테스트 통과 여부와 같은 binary correctness signals를 활용한 방식은 인간의 주석 없이도 확장 가능한 학습 방법으로 주목받고 있다.

그러나 일부 연구에서는 RLVR이 모델에게 진정으로 새로운 추론 능력을 부여하는 것이 아니라, 단지 모델이 이미 알고 있는 경로 중 보상을 받을 확률이 높은 경로로 유도(steering)하는 것에 불과하다고 주장한다. 본 논문은 이러한 기존 방식이 on-policy 학습에만 의존하기 때문에 탐색 공간의 제약이 크다는 점을 지적하며, 이종 모델들의 다양한 경로를 학습 과정에 통합함으로써 인지적 제약을 초월하고 탐색 능력을 극대화하려는 차별점을 가진다.

## 🛠️ Methodology

MEML-GRPO 프레임워크는 크게 두 단계인 **Multi-Expert Fine-tuning (MEF)**과 **Reinforced Inter-Expert Learning (RIEL)**으로 구성된다.

### 1. Multi-Expert Fine-tuning (MEF)

기본 모델이 여러 전문가의 행동을 모사할 수 있도록 하는 단계이다.

- **데이터셋 구축**: $N$개의 이종 전문가 모델 $E = \{E_1, E_2, \dots, E_N\}$로부터 각 질문 $Q$에 대한 답변 $A^{(i)}$를 생성한다.
- **전문가 프롬프트**: 각 전문가의 고유한 추론 스타일을 제어하기 위해 "You are Expert $i$, please provide an answer..."와 같은 시스템 프롬프트 $P_i$를 추가하여 데이터셋 $D_{ME}$를 구성한다.
- **학습 목표**: 전문가 답변의 조건부 로그 가능도(conditional log-likelihood)를 최대화하는 SFT를 수행한다.
$$L_{MEF} = -\sum_{j=1}^{M} \sum_{i=1}^{N} \log p_{\theta}(A^{(i)}_j | Q_j, P_i)$$

### 2. Reinforced Inter-Expert Learning (RIEL)

강화학습과 상호 학습을 통해 추론 능력을 고도화하는 단계이다.

**가. Response Sampling 및 Intra-Expert Advantage 추정**
각 전문가 프롬프트 $P_i$에 대해 $G$개의 응답 $O^i_g$를 샘플링하고, 보상 함수 $r(O^i_g)$를 통해 품질을 평가한다. GRPO 방식을 따라 동일 전문가 그룹 내의 평균 보상을 이용해 Advantage $A^i_g$를 계산한다.
$$A^i_g = r(O^i_g) - \frac{1}{G} \sum_{g'=1}^{G} r(O^i_{g'})$$
개별 전문가의 GRPO 손실 함수는 다음과 같다.
$$L^{(i)}_{GRPO} = -\mathbb{E}_{O^i_g \sim \pi_{\theta}} [\log \pi_{\theta}(O^i_g | Q, P_i) \cdot \max(A^i_g, 0)]$$

**나. Inter-Expert Mutual Learning (상호 학습)**
전문가 간 지식 전이를 위해 보상 평균이 가장 높은 전문가 $E^+$와 가장 낮은 전문가 $E^-$를 식별한다. $E^-$가 $E^+$의 고품질 응답 $O^+$를 생성하도록 KL Divergence 정규화 항을 도입한다.
$$L_{KL} \approx \log p_{\theta}(O^+ | Q, \text{prompt}_{E^-}) - \log p_{\theta}(O^+ | Q, \text{prompt}_{E^+})$$
이는 성능이 낮은 전문가가 높은 전문가의 출력 분포를 따라가게 함으로써 전체적인 성능 상한선을 끌어올린다.

**다. Hard Example Accumulation (SFT 버퍼)**
모든 전문가가 실패한 문제는 보상 신호가 없어 학습이 불가능하다. 이를 해결하기 위해 용량이 $B$인 버퍼를 유지하며, 전문가들이 $G$개 중 $K$개 이상의 오답을 낸 경우 확률적으로 $\frac{K}{G}$에 따라 Ground Truth($O_{gt}$) 쌍을 버퍼에 저장한다. 버퍼가 가득 차면 다음의 SFT 손실 함수를 통해 업데이트를 수행한다.
$$L_{SFT} = -\sum_{(Q, P_i, O_{gt}) \in B} \log p_{\theta}(O_{gt} | Q, P_i)$$

### 3. Overall Training Objective

최종 학습 목적 함수는 다음과 같이 세 가지 손실 함수의 합으로 정의된다.
$$L_{total} = L_{GRPO} + \lambda_1 L_{KL} + \lambda_2 L_{SFT}$$

## 📊 Results

### 실험 설정

- **데이터셋**: GSM8K(수학), MathQA(수학), StrategyQA(상식 추론).
- **전문가 모델**: Ground Truth(Expert 0), DeepSeek-r1(Expert 1), Doubao-1.5-thinking(Expert 2).
- **평가 모델**: Qwen2.5-1.5B-Math, Llama3.2-1B-Instruct.
- **평가 지표**: Exact-match accuracy.

### 주요 결과

- **SOTA RLVR 대비 성능**: MEML-GRPO는 모든 데이터셋에서 기존 GRPO 및 Dr.GRPO보다 뛰어난 성능을 보였다. Qwen 모델에서는 평균 4.89%, Llama 모델에서는 11.33%의 성능 향상을 달성하였다.
- **Reward Sparsity 해결**: 단일 전문가 기반 RL보다 다중 전문가 기반의 MEML-GRPO가 월등히 높은 성능을 보였는데, 이는 다양한 시스템 프롬프트를 통해 정답 경로를 발견할 확률을 높였기 때문이다.
- **Majority Voting(MV)과의 비교**: 학습 후 단일 전문가만으로 추론했을 때의 성능이, 일반적인 MoE-SFT-GRPO 모델의 Majority Voting 결과보다 더 높게 나타났다. 이는 상호 학습(Mutual Learning)을 통해 지식이 모델 내부에 내재화되었음을 시사한다.

## 🧠 Insights & Discussion

본 논문은 RLVR의 고질적인 문제인 Reward Sparsity를 해결하기 위해 '다양성'과 '상호 학습'이라는 관점을 성공적으로 도입하였다.

**강점 및 통찰:**

- **추론 효율성**: 일반적인 앙상블이나 Majority Voting 방식은 추론 시 $N$번의 생성이 필요하여 비용이 높지만, MEML-GRPO는 학습 단계에서 상호 학습을 통해 지식을 통합했으므로 추론 시에는 단일 모델의 지연 시간(latency)만으로 높은 성능을 낼 수 있다.
- **학습 안정성**: SFT 버퍼를 통한 Hard Example 처리는 강화학습의 불안정성과 보상 부재 문제를 보완하는 안전장치 역할을 한다.
- **효율적인 구현**: vLLM의 Paged Attention과 배치 롤아웃을 사용하여, 전문가 수가 늘어남에도 불구하고 학습 시간 증가폭을 20-30% 수준으로 억제하였다.

**한계 및 논의:**

- 본 연구는 상대적으로 작은 파라미터 규모(1B-1.5B)의 모델에서 검증되었다. 모델 규모가 커졌을 때 Heterogeneous한 스타일을 수용하는 능력이 어떻게 변화할지에 대한 분석이 추가로 필요하다.
- 사용된 전문가 모델(DeepSeek-r1 등)의 성능에 의존적일 가능성이 있으며, 전문가 모델 간의 성능 격차가 매우 클 때 상호 학습의 효율성이 어떻게 달라지는지에 대한 정밀한 분석은 명시되지 않았다.

## 📌 TL;DR

MEML-GRPO는 RLVR의 **Reward Sparsity** 문제를 해결하기 위해 다양한 이종 모델의 추론 스타일을 시스템 프롬프트로 학습(MEF)시키고, 전문가 간의 KL Divergence 기반 상호 학습(RIEL)과 Hard Example 전용 SFT 버퍼를 결합한 프레임워크이다. 이를 통해 추론 시 추가 비용 없이도 다중 모델의 앙상블 효과에 가까운 성능 향상을 이루었으며, 특히 Llama와 Qwen 모델에서 유의미한 성능 도약을 증명하였다. 이 연구는 향후 LLM의 자가 개선(Self-improvement) 루프에서 외부 지식을 효율적으로 통합하는 새로운 방법론을 제시한다.
