# Open-Medical-R1: How to Choose Data for RLVR Training at Medicine Domain

Zhongxi Qiu, Zhang Zhang, Yan Hu, Heng Li, Jiang Liu (2025)

## 🧩 Problem to Solve

본 연구는 의료 도메인에서 Verified Rewards를 이용한 강화학습(Reinforcement Learning with Verified Rewards, RLVR)을 수행할 때, 최적의 학습 데이터를 어떻게 선택하고 구성할 것인가에 대한 문제를 해결하고자 한다.

최근 DeepSeek-R1-Zero와 같은 모델들이 RLVR을 통해 수학 및 논리 퍼즐 분야에서 뛰어난 추론 능력을 보여주었으나, 의료와 같은 전문 도메인에 대한 적용 연구는 상대적으로 부족한 실정이다. 특히, 의료 데이터의 품질, 복잡성 및 구성은 모델이 견고한 추론 패턴을 형성하는 데 결정적인 영향을 미친다. 기존에 MedXpertQA와 같은 벤치마크 데이터를 학습에 사용하려는 시도가 있었으나, 이는 지식 유출(knowledge leakage) 문제를 야기하여 평가의 신뢰성을 떨어뜨릴 위험이 있다. 따라서 본 논문의 목표는 MedQA-USMLE 데이터셋 내에서 어떤 샘플링 전략이 모델의 의료 추론 능력을 극대화하면서도 일반적인 추론 성능과 강건성(robustness)을 유지할 수 있는지 분석하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 의료 도메인 RLVR 학습을 위한 데이터 선택 전략의 중요성을 규명하고, 이를 위해 다음과 같은 실험적 분석을 수행한 점이다.

- **데이터 샘플링 전략의 비교 분석**: 무작위 샘플링(Random Sampling)을 베이스라인으로 하여, 세 가지 서로 다른 모델(Phi-4, Gemma-3-27b-it, Gemma-3-12b-it)을 이용한 필터링 전략을 제안하고 그 효과를 비교하였다.
- **도메인 특화 성능과 일반 성능의 트레이드오프 분석**: 다양한 벤치마크(MMLU, GSM8K, MMLU-Pro, CMMLU)를 통해, 데이터 선택 방식이 의료 도메인 성능뿐만 아니라 일반적인 추론 능력과 강건성에 미치는 영향을 정량적으로 평가하였다.
- **필터링 모델의 크기와 효과 관계 규명**: 학습 대상 모델과 동일한 모델로 필터링했을 때와 더 큰 규모의 모델로 필터링했을 때의 성능 차이를 분석하여, 최적의 데이터 구성 방안에 대한 통찰을 제공하였다.

## 📎 Related Works

본 논문은 DeepSeek-R1-Zero가 보여준 RLVR의 잠재력, 즉 방대한 추론 데이터 없이도 강화학습만으로 Chain-of-Thought(CoT) 능력을 확보할 수 있다는 점에 기반한다. 이를 복제하려는 시도로 Open-R1, TinyZero, Logic-RL 등이 있었으나, 이들은 주로 수학이나 논리 문제에 집중하였다.

의료 분야에 RLVR을 적용한 선행 연구로는 PPO 알고리즘을 사용해 SFT 대비 정확도를 8% 향상시킨 Med-RLVR과, 다중 모달 의료 진단을 위해 GRPO를 도입한 연구 등이 있다. 또한 Baichuan-M1-14B를 GRPO로 미세 조정하거나 소량의 샘플(500개)만으로 성능을 높인 사례가 보고되었다. 그러나 기존 연구들은 '어떤 데이터를 선택하여 학습시키는 것이 최적인가'라는 데이터 구성 전략에 대해서는 충분히 다루지 않았으며, 본 논문은 이 지점을 차별점으로 삼아 연구를 진행하였다.

## 🛠️ Methodology

### 전체 파이프라인 및 GRPO 알고리즘

본 연구는 base 모델로 Gemma-3-12b-it를 사용하며, 효율적인 정책 최적화를 위해 Group Relative Policy Optimization (GRPO) 알고리즘을 채택하였다. GRPO는 기존 PPO와 달리 별도의 Value Function이나 Critic 네트워크를 필요로 하지 않아 계산 효율성이 높다.

1. **Action Sampling**: 현재 정책 $\pi_\theta(a|s)$로부터 상태 $s$에 대해 $G$개의 액션 그룹 $\{a_1, a_2, ..., a_G\}$을 샘플링한다.
2. **Reward Evaluation**: 각 액션 $a_i$에 대해 환경에서 보상 $r_i$를 측정한다.
3. **Relative Advantage Calculation**: 그룹 내 평균 보상을 기준으로 상대적 이점(Relative Advantage) $\hat{A}_i$를 다음과 같이 계산한다.
   $$\hat{A}_i = r_i - \frac{1}{G} \sum_{j=1}^{G} r_j$$
4. **Policy Update**: 다음과 같은 Clipped Objective 함수 $J_{GRPO}(\theta)$를 통해 정책 파라미터 $\theta$를 업데이트한다.
   $$J_{GRPO}(\theta) = E_{s,a \sim \pi_\theta} \left[ \min \left( r(s,a)\hat{A}, \text{clip}(r(s,a), 1-\epsilon, 1+\epsilon)\hat{A} \right) \right]$$
   여기서 $r(s,a) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}$는 현재 정책과 이전 정책 간의 확률 비율이며, $\epsilon$은 업데이트 크기를 제한하는 하이퍼파라미터이다.

### 데이터 샘플링 및 필터링 전략

단순 무작위 추출은 모델의 현재 능력치 대비 문제의 난이도를 고려하지 못한다. 이를 해결하기 위해 다음과 같은 필터링 전략을 사용하였다.

- **필터링 프로세스**: 필터링 모델(Phi-4, Gemma-3-27b-it, Gemma-3-12b-it)이 후보 샘플에 대해 응답을 생성하게 한다. 이때 모델이 정해진 형식을 준수하며 정답을 맞히면 'Easy', 그렇지 않으면 'Hard'로 분류한다.
- **최종 데이터셋 구성**: 'Hard' 샘플 400개와 'Easy' 샘플 100개를 선택하여 총 500개의 균형 잡힌 챌린징한 코퍼스를 구축하였다.

### 학습 설정

- **프레임워크**: Unsloth를 사용하였으며, NVIDIA GeForce RTX 4090 (24GB VRAM) 1대에서 학습하였다.
- **보상 함수**: Format reward, Accuracy reward, XML count reward의 세 가지 구성 요소로 보상을 계산하였다.
- **하이퍼파라미터**: Learning rate $2 \times 10^{-5}$, Cosine annealing scheduler, Batch size 3, 3 rollouts per sample, Gradient accumulation step 4를 적용하였다.

## 📊 Results

### 정량적 평가 결과

MMLU, CMMLU, MMLU-Pro, GSM8K 벤치마크를 통해 평가한 결과, 필터링된 데이터를 사용한 모델들이 무작위 샘플링 모델보다 전반적으로 우수한 성능을 보였다.

- **Self-Filtering (Gemma-3-12b-it) 결과**: MMLU에서 가장 높은 점수($0.6745$)를 기록하며 의료 관련 도메인에서 뚜렷한 향상을 보였으나, 타 벤치마크에서는 오히려 베이스라인보다 성능이 소폭 하락하는 경향을 보였다.
- **Larger Model Filtering (Gemma-3-27b-it, Phi-4) 결과**: 성능 향상 폭은 self-filtering보다 작을 수 있으나, 여러 벤치마크에 걸쳐 더 안정적인 성능을 유지하며 높은 강건성(robustness)을 나타냈다.

### 의료 도메인 상세 분석

MMLU의 세부 항목 중 고등학교 생물학(High school biology), 의료 유전학(Medical genetics), 전문 의학(Professional medicine) 분야에서 self-filtering 모델이 가장 큰 성능 향상을 보였다. 하지만 CMMLU 및 MMLU-Pro 결과에서는 self-filtering 모델의 강건성이 부족함이 드러났으며, 동일 시리즈의 더 큰 모델(Gemma-3-27b)로 필터링한 데이터로 학습했을 때 더 나은 일반화 성능을 보였다.

## 🧠 Insights & Discussion

본 연구는 의료 도메인에서 RLVR 학습 시 데이터 선택 전략이 모델의 최종 성능과 일반화 능력에 결정적인 영향을 미친다는 것을 입증하였다.

**강점 및 발견**:

- 단순히 데이터를 많이 사용하는 것보다, 모델의 수준에 맞는 'Hard' 샘플을 전략적으로 배치하는 것이 추론 능력 향상에 효과적이다.
- 학습 대상 모델이 스스로 필터링한 데이터(Self-filtered data)로 학습할 경우, 해당 도메인 내에서의 성능은 극대화되지만, 타 도메인으로의 전이 능력이나 일반적인 강건성은 저하되는 현상이 관찰되었다.

**한계 및 논의**:

- **데이터 규모의 제한**: 실험에 사용된 데이터셋이 500개로 매우 소량이다. 이는 성능 향상을 이끌어내기에는 부족할 수 있으며, 결과의 변동성을 키웠을 가능성이 있다.
- **필터링 전략의 단순함**: 단순히 정답 여부와 형식 준수 여부로만 'Easy/Hard'를 나눈 전략은 매우 단순하므로, 더 정교한 난이도 측정 지표가 필요하다.
- **언어적 영향**: 단일 언어(영어) 데이터로 GRPO 학습을 진행했을 때 다국어 성능(CMMLU 등)이 일부 하락하는 현상이 발생하였다.

## 📌 TL;DR

본 논문은 의료 도메인의 RLVR 학습을 위해 MedQA-USMLE 데이터셋에서 최적의 데이터를 선택하는 전략을 연구하였다. 실험 결과, 무작위 추출보다 모델 기반 필터링이 효과적이며, 특히 **학습 모델 본인이 필터링한 데이터는 도메인 특화 성능을 극대화하지만, 더 큰 모델이 필터링한 데이터는 전반적인 강건성을 높인다**는 것을 확인하였다. 이 결과는 전문 도메인 LLM 구축 시 데이터의 '양'보다 '질적 구성'과 '필터링 주체'의 선택이 매우 중요함을 시사하며, 향후 검색 도구 통합이나 전체 파라미터 학습으로 확장될 가능성이 높다.
